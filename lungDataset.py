from __future__ import print_function
import torch.utils.data as dataloader
import copy
import math
import os
from os import listdir
from os.path import isfile, join, splitext
import numpy as numpy
import SimpleITK as sitk 
import skimage.transform
import torch.utils.data as data
import numpy as np
import csv


class lungDataset(data.Dataset):
    def createImageFileList(self):
        self.fileList = list()
        for f in listdir(self.srcFolder):
            for file in listdir(join(self.srcFolder, f)):
                if 'img' in file:
                    self.fileList.append(f)
                    break
        print ('FILE LIST: ' + str(self.fileList))
       
    
    def getNumpyData_forImg(self, img, key):
        ''' load numpy data from sitk data'''

        img = sitk.GetArrayFromImage(img).astype(dtype=np.float32)
        self.originalSizes[key]=img.shape
        self.dimensionMin[key] = np.argmin(img.shape)
        if self.dimensionMin[key] == 0:
            img = np.transpose(img, [1,2,0])
        elif self.dimensionMin[key] == 1:
            img = np.transpose(img, [0,2,1])

        # order =3 Bi-cubic, resize the whole image to the same size
        #skimage can only operate on the float(range: -1 to 1) and int image
        #reserve_range : bool, Whether to keep the original range of values. Otherwise, the input image is converted according to the conventions of img_as_float.
        #ret[key] = skimage.transform.resize(ret[key],self.params['WholeSize'],order=3, mode='reflect', preserve_range=True)
        # clip the top 3% intensity, and normalize the img into the range of 0~1
        # max_value = np.sort(img, axis=None)[int(img.size*0.97)]
        # #ret[key][ret[key]>max_value] = max_value
        # img = (img)/max_value
        '''cv2.imshow("",ret[key][:,:,60])
        cv2.waitKey(0)'''
        #img = (img-20)/200.0
        return img

    def getNumpyData_forLabel(self, img, key):
           
        img = sitk.GetArrayFromImage(img).astype(dtype=np.float32)
        self.originalSizes[key] = img.shape
        self.dimensionMin[key] = np.argmin(img.shape)
        if self.dimensionMin[key] == 0:
            img = np.transpose(img, [1,2,0])
        elif self.dimensionMin[key] == 1:
            img = np.transpose(img, [0,2,1])

        return img

    def loadLesionPositions(self, f):
        if f in self.lesionPositions:
            return
        with open(join(join(self.srcFolder, f), 'nodule.csv')) as csvFile:
            reader = csv.reader(csvFile)
            reader = list(reader)
            transformedPostions = []
            for row in reader[1:]:
                if len(row) != 4:
                    continue
                (z, y, x, p) = (int(math.ceil(float(num))) for num in row)
                if self.dimensionMin[f] == 0:
                    (x, y, z) = (y, z, x)
                elif self.dimensionMin[f] ==1:
                    (x, y, z) = (x, z, y)
                transformedPostions.append((x, y, z))
            self.lesionPositions[f] = transformedPostions


    def __init__ (self, srcFolder, resultsDir, training = True, transform = None, probabilityMap=False):
        super(lungDataset, self).__init__()
        self.transform = transform
        self.srcFolder = srcFolder
        self.resultsDir = resultsDir
        self.probabilityMap = probabilityMap
        self.createImageFileList()
        self.originalSizes = {}
        self.dimensionMin = {}
        self.training = training
        self.lesionPositions = {}
    
    def __len__(self):
        print ('totally ', len(self.fileList), ' samples')
        return len(self.fileList)

    def __getitem__(self, index):
        img = sitk.Cast(sitk.ReadImage(
            join(join(self.srcFolder, self.fileList[index]), 'img.nii')), sitk.sitkFloat32)
        img = self.getNumpyData_forImg(img, self.fileList[index])

        label = sitk.Cast(sitk.ReadImage(
            join(join(self.srcFolder, self.fileList[index]), 'label.nii')), sitk.sitkFloat32)
        label = self.getNumpyData_forLabel(label, self.fileList[index])
        if self.training:
            pass
            #self.loadLesionPositions(self.fileList[index])
        # img = np.zeros([100,100,100])
        # label = np.ones([100,100,100])
        if self.transform:
            for f in self.transform:
                fileName = self.fileList[index]
                img, label = f(img, label, fileName, None)
        return img, label, fileName
    
