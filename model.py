import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import DataManagerLazy as DM
from os.path import splitext
import multiprocessing as mp

from multiprocessing import Process, Queue
import cv2
import math
import datetime
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import shutil
import vnet
from functools import reduce
import operator
import time
from tqdm import tqdm
import torchbiomed.loss as bioloss
import vnet_batchNorm
import resnet3D
import DataManager as DMoriginal
import lungDataset
import ImageTransform3D
import torchvision.transforms as transforms


class Model(object):

    ''' the network model for training, validation and testing '''
    params = None
    dataManagerTrain = None
    dataManagerValidation = None
    min_loss = 9999999999
    min_loss_accuracy = 0
    max_accuracy = 0
    max_accuracy_loss = 0
    best_iteration_acc = 0
    best_iteration_loss = 0

    def __init__(self, params):
        self.params = params

    def getValidationLossAndAccuracy(self, model):
        '''get the segmentation loss and accuracy of the validation data '''
        files_num = len(self.dataManagerValidation.fileList)
        loss = 0.0
        accuracy = 0.0
        ResultImages = dict()
        for i in xrange(files_num):
            img = self.dataManagerValidation.loadImage(self.dataManagerValidation.fileList[i])
            label = self.dataManagerValidation.loadGT(self.dataManagerValidation.fileList[i])
            (numpyResult, temploss) = self.produceSegmentationResult(
                model, img, label, calLoss=True)
            loss += temploss
            LabelResult = numpyResult

            '''cv2.imshow('0',LabelResult[:,:,32])
            cv2.waitKey(0)
            cv2.imshow('1',numpyGTs[keysIMG[i]][:,:,32])
            cv2.waitKey(0)'''
            right = float(np.sum(LabelResult == label[:, :, :]))
            tot = float(LabelResult.shape[0] * LabelResult.shape[1] * LabelResult.shape[2])
            accuracy += right / tot
        return (loss / files_num, accuracy /files_num)

    def getTestResultImages(self, model, returnProbability=False):
        ''' return the segmentation results of the testing data'''
        numpyImages = self.dataManagerTesting.numpyImages
        numpyGTs = self.dataManagerTesting.numpyGTs
        loss = 0.0
        accuracy = 0.0
        ResultImages = dict()
        keysIMG = numpyImages.keys()
        for i in xrange(len(keysIMG)):
            (numpyResult, temploss) = self.produceSegmentationResult(
                model, numpyImages[keysIMG[i]], numpyGTs[keysIMG[i]], calLoss=True, returnProbability=returnProbability)
            loss += temploss
            if returnProbability:
                LabelResult = numpyResult
            else:
                LabelResult = numpyResult
            '''cv2.imshow('0',LabelResult[:,:,32])
            cv2.waitKey(0)
            cv2.imshow('1',numpyGTs[keysIMG[i]][:,:,32])
            cv2.waitKey(0)'''
            right = float(np.sum(LabelResult == numpyGTs[keysIMG[i]][:, :, :]))
            tot = float(LabelResult.shape[0] * LabelResult.shape[1] * LabelResult.shape[2])
            accuracy += right / tot
            ResultImages[keysIMG[i]] = LabelResult
        print "loss: ", loss / len(keysIMG), " acc: ", accuracy / len(keysIMG)
        return ResultImages

    def produceSegmentationResult(self, model, numpyImage, numpyGT=0, calLoss=False, returnProbability=False):
        ''' produce the segmentation result, one time one image'''
        model.eval()
        numpyImage = np.copy(numpyImage)
        tempresult = np.zeros(
            (numpyImage.shape[0], numpyImage.shape[1], numpyImage.shape[2]), dtype=np.float32)
        tempWeight = np.zeros(
            (numpyImage.shape[0], numpyImage.shape[1], numpyImage.shape[2]), dtype=np.float32)
        height = int(self.params['DataManagerParams']['VolSize'][0])
        width = int(self.params['DataManagerParams']['VolSize'][1])
        depth = int(self.params['DataManagerParams']['VolSize'][2])

        batchData = np.zeros([1, 1, height, width, depth])
        batchLabel = np.zeros([1, 1, height, width, depth])

        stride_height = int(self.params['DataManagerParams']['TestStride'][0])
        stride_width = int(self.params['DataManagerParams']['TestStride'][1])
        stride_depth = int(self.params['DataManagerParams']['TestStride'][2])
        whole_height = int(numpyImage.shape[0])
        whole_width = int(numpyImage.shape[1])
        whole_depth = int(numpyImage.shape[2])
        ynum = int(math.ceil((whole_height - height) / float(stride_height))) + 1
        xnum = int(math.ceil((whole_width - width) / float(stride_width))) + 1
        znum = int(math.ceil((whole_depth - depth) / float(stride_depth))) + 1
        loss = 0
        acc = 0
        tot = 0
        # crop the image
        for y in xrange(ynum):
            for x in xrange(xnum):
                for z in xrange(znum):
                    if(y * stride_height + height < whole_height):
                        ystart = y * stride_height
                        yend = ystart + height
                    else:
                        ystart = whole_height - height
                        yend = whole_height
                    if(x * stride_width + width < whole_width):
                        xstart = x * stride_width
                        xend = xstart + width
                    else:
                        xstart = whole_width - width
                        xend = whole_width
                    if(z * stride_depth + depth < whole_depth):
                        zstart = z * stride_depth
                        zend = zstart + depth
                    else:
                        zstart = whole_depth - depth
                        zend = whole_depth
                    tot += 1
                    batchData[0, 0, :, :, :] = numpyImage[ystart:yend, xstart:xend, zstart:zend]
                    if(calLoss):
                        batchLabel[0, 0, :, :, :] = numpyGT[ystart:yend, xstart:xend, zstart:zend]
                    else:
                        batchLabel[0, 0, :, :, :] = np.zeros(numpyImage[ystart:yend, xstart:xend, zstart:zend].shape)
                    data = torch.from_numpy(batchData).float()
                    # volatile is used in the input variable for the inference,
                    # which indicates the network doesn't need the gradients, and this flag will transfer to other variable
                    # as the network computating
                    data = Variable(data, volatile=True).cuda()
                    #data = Variable(data).cuda()
                    target = torch.from_numpy(batchLabel).long()
                    target = Variable(target).cuda()

                    original_shape = data[0, 0].size()
                    output = model(data)
                    target = target.view(target.numel())
                    temploss = F.nll_loss(output, target)
                    #temploss = bioloss.dice_loss(output, target)
                    # be carefull output is the log-probability, not the raw probability
                    # max(1) return a tumple,the second item is the index of the max
                    output = output.data.max(1)[1]
                    output = output.view(original_shape)
                    output = output.cpu()

                    temploss = temploss.cpu().data[0]
                    loss = loss + temploss
                    # print temptrain_loss
                    tempresult[ystart:yend, xstart:xend, zstart:zend] = tempresult[
                        ystart:yend, xstart:xend, zstart:zend] + output.numpy()
                    tempWeight[ystart:yend, xstart:xend, zstart:zend] = tempWeight[
                        ystart:yend, xstart:xend, zstart:zend] + 1
        tempresult = tempresult / tempWeight
        # important! change the model back to the training phase!
        model.train()
        return (tempresult, loss)

    def create_temp_images(self, img, img_name):
        # create the 2D image based on the sitk image

        shape = img.shape
        n = math.sqrt(shape[0])
        n = int(n + 1)
        out_img = np.zeros([n * shape[1], n * shape[2]])
        img = (img * 255).astype(np.uint8)
        for i in xrange(n):
            for j in xrange(n):
                if i * n + j < shape[0]:
                    out_img[i * shape[1]:i * shape[1] + shape[1], j * shape[2]:j *
                            shape[2] + shape[2]] = img[i * n + j, :, :]

        cv2.imwrite(os.path.join('tempImg', img_name + '.png'), out_img)

    def prepareDataThread(self, dataQueue, numpyImages, numpyGT):
        ''' the thread worker to prepare the training data'''
        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']

        nr_iter_dataAug = nr_iter*batchsize
        np.random.seed()
        last_blood=False
        files_num = len(self.dataManagerTrain.fileList)
        batchsize = self.params['ModelParams']['batchsize']
        for i in xrange(nr_iter_dataAug/self.params['ModelParams']['nProc']):
            batchData = torch.zeros(batchsize, 1, self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2])
            batchLabel = torch.zeros(batchData.size())
            batchData = Variable(batchData).float()
            batchLabel = Variable(batchLabel).long()
            for j in range(batchsize):
            # create the postive image which contains the true lesion
                getRightCrop = False
                random_file = np.random.randint(files_num)
                tempimage = self.dataManagerTrain.loadImage(self.dataManagerTrain.fileList[random_file])
                tempGT = self.dataManagerTrain.loadGT(self.dataManagerTrain.fileList[random_file])
                while not getRightCrop:
                    image_height = tempimage.shape[0]
                    image_width = tempimage.shape[1]
                    image_depth = tempimage.shape[2]
                    starty = np.random.randint(
                        image_height - self.params['DataManagerParams']['VolSize'][0])
                    startx = np.random.randint(
                        image_width - self.params['DataManagerParams']['VolSize'][1])
                    startz = np.random.randint(image_depth - self.params['DataManagerParams']['VolSize'][2])
                    cropimage = tempimage[starty: starty + self.params['DataManagerParams']['VolSize'][
                        0], startx: startx + self.params['DataManagerParams']['VolSize'][1], startz: startz + self.params['DataManagerParams']['VolSize'][2]]
                    cropGT = tempGT[starty: starty + self.params['DataManagerParams']['VolSize'][
                        0], startx: startx + self.params['DataManagerParams']['VolSize'][1], startz: startz + self.params['DataManagerParams']['VolSize'][2]]

                    cropimage = cropimage.astype(dtype=np.float32)
                    cropGT = cropGT.astype(dtype=np.float32)
                    # skip the image not containing the mircrobleed
                    # if cropGT.sum()<1:
                    #     continue
                    if tempGT.sum() < 1 and not last_blood:
                        continue
                    if tempGT.sum() < 1:
                        last_blood = False
                    else:
                        last_blood = True
                    getRightCrop = True
                    randomi = np.random.randint(4)
                    cropimage = np.rot90(cropimage, randomi)
                    cropGT = np.rot90(cropGT, randomi)
                    cropimage = torch.from_numpy(np.ascontiguousarray(cropimage))
                    cropGT = torch.from_numpy(np.ascontiguousarray(cropGT))
                    batchData[j, 0, :, :, :] = cropimage
                    batchLabel[j, 0, :, :, :] = cropGT
            #tempimage = tempimage * (np.random.rand(1) + 0.5)

            dataQueue.put(tuple((batchData,batchLabel)))

    def save_checkpoint(self, state, path, prefix, filename='checkpoint.pth.tar'):
        ''' save the snapshot'''
        prefix_save = os.path.join(path, prefix)
        name = prefix_save + str(state['iteration']) + '_' + filename
        torch.save(state, name)

    def trainThread(self, model, trainData_loader):
        '''train the network and plot the training curve'''
        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']

        test_interval = self.params['ModelParams']['testInterval']
        train_interval = 50
        train_loss = np.zeros(nr_iter)
        train_accuracy = np.zeros(nr_iter / train_interval)
        testloss = np.zeros(nr_iter / test_interval)
        testaccuracy = np.zeros(nr_iter / test_interval)
        tempaccuracy = 0
        temptrain_loss = 0
 
        print("build vnet")

        model.train()
        #model.eval()
        model.cuda()
        optimizer = optim.Adam(model.parameters(), weight_decay=1e-8, lr=self.params['ModelParams']['baseLR'])
        
        it = 0
        while True:
            for origin_it, (data, target) in enumerate(trainData_loader):
                it += 1
                

                optimizer.zero_grad()

                data = data.cuda()
                target = target.cuda(async=True)
                data = torch.autograd.Variable(data)
                target = torch.autograd.Variable(target)

                output = model(data)
                target = target.view(target.numel())
                loss = F.nll_loss(output, target)
                #loss = bioloss.dice_loss(output, target)
                loss.backward()
                optimizer.step()

                temploss = loss.cpu().data[0]
                temptrain_loss = temptrain_loss + temploss
                # print temptrain_loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                incorrect = pred.ne(target.data).cpu().sum()
                tempaccuracy = tempaccuracy + 1.0 - float(incorrect) / target.numel()

                if np.mod(it, train_interval) == 0:

                    train_accuracy[it / train_interval] = tempaccuracy / (train_interval)
                    train_loss[it / train_interval] = temptrain_loss / (train_interval)
                    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), " training: iter: ", self.params['ModelParams']['snapshot'] + it, " loss: ", train_loss[it / train_interval], ' acc: ', train_accuracy[it / train_interval]
                    plt.clf()
                    plt.subplot(2, 2, 1)
                    plt.plot(range(1, it / train_interval), train_loss[1:it / train_interval])
                    plt.subplot(2, 2, 2)
                    plt.plot(range(1, it / train_interval), train_accuracy[1:it / train_interval])
                    plt.subplot(2, 2, 3)
                    plt.plot(range(1, it / test_interval),
                            testloss[1:it / test_interval])
                    plt.subplot(2, 2, 4)
                    plt.plot(range(1, it / test_interval),
                            testaccuracy[1:it / test_interval])

                    tempaccuracy = 0.0
                    temptrain_loss = 0.0
                    plt.pause(0.00000001)

                if np.mod(it, test_interval) == 0:
                    (testloss[it / test_interval], testaccuracy[it / test_interval]
                    ) = self.getValidationLossAndAccuracy(model)

                    if testaccuracy[it / test_interval] >= self.max_accuracy:
                        self.max_accuracy = testaccuracy[it / test_interval]
                        self.min_accuracy_loss = testloss[it / test_interval]
                        self.best_iteration_acc = self.params['ModelParams']['snapshot'] + it
                        self.save_checkpoint({'iteration': self.params['ModelParams']['snapshot'] + it,
                                            'state_dict': model.state_dict(),
                                            'best_acc': True},
                                            self.params['ModelParams']['dirSnapshots'], self.params['ModelParams']['tailSnapshots'])

                    if testloss[it / test_interval] <= self.min_loss:
                        self.min_loss = testloss[it / test_interval]
                        self.min_loss_accuracy = testaccuracy[it / test_interval]
                        self.best_iteration_loss = self.params['ModelParams']['snapshot'] + it
                        self.save_checkpoint({'iteration': self.params['ModelParams']['snapshot'] + it,
                                            'state_dict': model.state_dict(),
                                            'best_acc': False},
                                            self.params['ModelParams']['dirSnapshots'], self.params['ModelParams']['tailSnapshots'])

                    print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print "\ntesting: best_acc: " + str(self.best_iteration_acc) + " loss: " + str(self.min_accuracy_loss) + " accuracy: " + str(self.max_accuracy)
                    print "testing: best_loss: " + str(self.best_iteration_loss) + " loss: " + str(self.min_loss) + " accuracy: " + str(self.min_loss_accuracy)
                    print "testing: iteration: " + str(self.params['ModelParams']['snapshot'] + it) + " loss: " + str(testloss[it / test_interval]) + " accuracy: " + str(testaccuracy[it / test_interval]) + '\n'
                    plt.clf()
                    plt.subplot(2, 2, 1)
                    plt.plot(range(1, it / 100), train_loss[1:it / 100])
                    plt.subplot(2, 2, 2)
                    plt.plot(range(1, it / 100), train_accuracy[1:it / 100])
                    plt.subplot(2, 2, 3)
                    plt.plot(range(1, it / test_interval),
                            testloss[1:it / test_interval])
                    plt.subplot(2, 2, 4)
                    plt.plot(range(1, it / test_interval),
                            testaccuracy[1:it / test_interval])
                    plt.pause(0.00000001)

                matplotlib.pyplot.show()

    def weights_init(self, m):
        ''' initialize the model'''
        classname = m.__class__.__name__
        if classname.find('Conv3d') != -1:
            nn.init.kaiming_normal(m.weight)
            #m.bias.data.zero_()

    def train(self):
        ''' train model'''
        torch.cuda.set_device(self.params['ModelParams']['device'])
        # we define here a data manager object
        self.datasetTrain = lungDataset.lungDataset(self.params['ModelParams']['dirTrain'], self.params['ModelParams']['dirResult'], 
                                                    transform = [ImageTransform3D.RandomCropSegmentation3D(self.params['DataManagerParams']['VolSize'], containLeision=2), ImageTransform3D.RandomRotateSegmentation3D(), ImageTransform3D.ToTensorSegmentation()])
        trainData_loader = torch.utils.data.DataLoader(self.datasetTrain, batch_size= self.params['ModelParams']['batchsize'], shuffle=True, num_workers= self.params['ModelParams']['nProc'], pin_memory=True)
        
        self.dataManagerValidation = DM.DataManagerLazy(self.params['ModelParams']['dirValidation'], 
                                                    self.params['ModelParams']['dirResult'], 
                                                    self.params['DataManagerParams'])
        self.dataManagerValidation.loadTestData()

        # create the network
        #model = resnet3D.resnet34(nll = False)
        model = vnet.VNet(nll=True)
        model = torch.nn.DataParallel(model, device_ids = self.params['ModelParams']['device_ids'])

        # train from scratch or continue from the snapshot
        if (self.params['ModelParams']['snapshot'] > 0):
            print "=> loading checkpoint ", str(self.params['ModelParams']['snapshot'])
            prefix_save = os.path.join(self.params['ModelParams']['dirSnapshots'],
                                       self.params['ModelParams']['tailSnapshots'])
            name = prefix_save + str(self.params['ModelParams']['snapshot']) + '_' + "checkpoint.pth.tar"
            checkpoint = torch.load(name)
            model.load_state_dict(checkpoint['state_dict'])
            print "=> loaded checkpoint ", str(self.params['ModelParams']['snapshot'])
        else:
            model.apply(self.weights_init)

        plt.ion()

        self.trainThread(model, trainData_loader)

    def test(self, snapnumber):
        # produce the results of the testing data
        torch.cuda.set_device(self.params['ModelParams']['device'])
        self.dataManagerTesting = DMoriginal.DataManager(self.params['ModelParams']['dirTest'], 
                                                 self.params['ModelParams']['dirResult'], 
                                                 self.params['DataManagerParams'], 
                                                 self.params['TestParams']['ProbabilityMap'])
        self.dataManagerTesting.loadTestData()
        
        #model = resnet3D.resnet34(nll = False)
        model = vnet.VNet(nll=False)
        
        prefix_save = os.path.join(self.params['ModelParams']['dirSnapshots'],
                                   self.params['ModelParams']['tailSnapshots'])
        name = prefix_save + str(snapnumber) + '_' + "checkpoint.pth.tar"
        checkpoint = torch.load(name)
        # load the snapshot into the model
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()
        #produce the segementation results
        results = self.getTestResultImages(model, self.params['TestParams']['ProbabilityMap'])

        for key in results:
            self.dataManagerTesting.writeResultsFromNumpyLabel(results[key], key)
