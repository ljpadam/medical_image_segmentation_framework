import copy
import math
import os
from os import listdir
from os.path import isfile, join, splitext

import cv2
import numpy as np
import SimpleITK as sitk
import skimage.transform
import multiprocessing
from multiprocessing import Pool


class DataManagerLazy(object):
    params = None
    srcFolder = None
    resultsDir = None

    fileList = None
    gtList = None

    sitkImages = None
    sitkGT = None
    numpyImages = None
    numpyGTs = None
    meanIntensityTrain = None
    probabilityMap = False
    originalSizes = False

    def __init__(self, srcFolder, resultsDir, parameters, probabilityMap=False):
        self.params = parameters
        self.srcFolder = srcFolder
        self.resultsDir = resultsDir
        self.probabilityMap = probabilityMap

    def createImageFileList(self):
        self.fileList = list()
        for f in listdir(self.srcFolder):
            for file in listdir(join(self.srcFolder, f)):
                if 'img' in file:
                    self.fileList.append(f)
                    break

        print 'FILE LIST: ' + str(self.fileList)


    def image_histogram_equalization(self, image, number_bins=256):
        # from
        # http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

        # get image histogram
        image_histogram, bins = np.histogram(
            image.flatten(), number_bins, normed=True)
        cdf = image_histogram.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape), cdf

    def create_temp_images(self, img_name):
        # create the 2D image based on the sitk image
        img = sitk.GetArrayFromImage(
            self.sitkImages[img_name]).astype(dtype=float)
        shape = img.shape
        n = math.sqrt(shape[0])
        n = int(n)
        out_img = np.zeros([n * 512, n * 512])
        img = (img * 255).astype(np.uint8)
        for i in xrange(n):
            for j in xrange(n):
                out_img[i * 512:i * 512 + 512, j * 512:j *
                        512 + 512] = img[i * n + j, :, :]

        cv2.imwrite(os.path.join('tempImg', img_name + '.png'), out_img)

    def loadImage(self, f):
        img = sitk.Cast(sitk.ReadImage(
            join(join(self.srcFolder, f), 'img.nii')), sitk.sitkFloat32)
        img = self.getNumpyData_Normalization(img,f)
        return img
        

    def loadGT(self, f):
        label = sitk.Cast(sitk.ReadImage(
            join(join(self.srcFolder, f), 'label.nii')), sitk.sitkFloat32)
        label = self.getNumpyData(label, f)
        return label


    def loadTrainingData(self):
        self.createImageFileList()
        self.dimensionMin = dict()
        self.originalSizes = dict()

    def loadTestData(self):
        self.createImageFileList()
        self.dimensionMin = dict()
        self.originalSizes = dict()

    def getNumpyData_Normalization(self, img, key):
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
        #print ret[key].shape
        # clip the top 3% intensity, and normalize the img into the range of 0~1
        # max_value = np.sort(img, axis=None)[int(img.size*0.97)]
        # #ret[key][ret[key]>max_value] = max_value
        # img = (img)/max_value
        '''cv2.imshow("",ret[key][:,:,60])
        cv2.waitKey(0)'''
        return img
    
    def getNumpyData(self, img, key):
           
        img = sitk.GetArrayFromImage(img).astype(dtype=np.float32)
        self.originalSizes[key] = img.shape
        self.dimensionMin[key] = np.argmin(img.shape)
        if self.dimensionMin[key] == 0:
            img = np.transpose(img, [1,2,0])
        elif self.dimensionMin[key] == 1:
            img = np.transpose(img, [0,2,1])

        return img

    def writeResultsFromNumpyLabel(self, result, key):
        ''' save the segmentation results to the result directory'''
        if self.dimensionMin[key] == 0:
            result = np.transpose(result, [2, 0, 1])
        elif self.dimensionMin[key] == 1:
            result = np.transpose(result, [0, 2, 1])
        
        if result.shape != self.originalSizes[key]:
            print "result shape is wrong!!!"

        if self.probabilityMap:
            result = result * 255
        else:
            result = result>0.5
            result = result.astype(np.uint8)
        toWrite = sitk.GetImageFromArray(result)

        toWrite = sitk.Cast(toWrite, sitk.sitkUInt8)

        writer = sitk.ImageFileWriter()
        filename, ext = splitext(key)
        # print join(self.resultsDir, filename + '_result' + ext)
        writer.SetFileName(join(self.resultsDir, filename + '_result.nii'))
        writer.Execute(toWrite)
