import sys
import os
import numpy as np
import SimpleITK as sitk
from os import listdir
from os.path import isfile, join, splitext
from scipy import ndimage
from scipy.ndimage import measurements
from pathos.multiprocessing import ProcessingPool as Pool
# compute the metrix for the small objects detection task, only comput the sensitivity, precision and the number of false positive

ground_truth_path = '/home/ljp/data/lung/manual/testing/'
results_path = './result/'
ground_truth_labels = dict()
results_labels = dict()
names = []


def load_labels():
    for folder in listdir(ground_truth_path):
        label = sitk.Cast(sitk.ReadImage(
            join(join(ground_truth_path, folder), 'label.nii.gz')), sitk.sitkFloat32)
        #temp=sitk.GetArrayFromImage(label)
        #print temp.shape
        ground_truth_labels[folder] = np.transpose(sitk.GetArrayFromImage(
            label).astype(dtype=float), [2, 0, 1])
        #print ground_truth_labels[folder].shape
        label = sitk.Cast(sitk.ReadImage(
            join(results_path, folder + "_result.nii")), sitk.sitkFloat32)
        results_labels[folder] = np.transpose(sitk.GetArrayFromImage(
            label).astype(dtype=float), [2, 0, 1])

def load_names():
    
    for folder in listdir(ground_truth_path):
        names.append(folder)


def dilation(img):
    s=ndimage.generate_binary_structure(3,3)
    img=ndimage.binary_dilation(img,structure=s).astype(img.dtype)
    return img

def save_img(img,path):
    #print img.shape
    result = np.transpose(img, [1, 2, 0])
    #print result.shape
    toWrite = sitk.GetImageFromArray(result)
    toWrite = sitk.Cast(toWrite, sitk.sitkUInt8)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(toWrite)


def get_TP_FP_FN(name):
    #GT_label=dilation(GT_label)
    #result_label=dilation(result_label)
    
    s=np.ones([3,3,3])
    GT_label = sitk.Cast(sitk.ReadImage(
            join(join(ground_truth_path, name), 'label.nii.gz')), sitk.sitkFloat32)
        #temp=sitk.GetArrayFromImage(label)
        #print temp.shape
    GT_label = np.transpose(sitk.GetArrayFromImage(
            GT_label).astype(dtype=float), [2, 0, 1])

    label = sitk.Cast(sitk.ReadImage(
            join(results_path, name + "_result.nii")), sitk.sitkFloat32)
    result_label = np.transpose(sitk.GetArrayFromImage(
            label).astype(dtype=float), [2, 0, 1])

    GT_labeled, GT_num = measurements.label(GT_label,structure=s)
    result_labeled, result_num = measurements.label(result_label, structure=s)
    #save_img(GT_labeled,'result.nii')
    TP = 0.0
    TP_check = 0.0
    FN = 0.0
    FP = 0.0
    print "GT_num: "+str(GT_num)
    for ii in xrange(GT_num):
        i=ii+1
        GT_mask = np.zeros(GT_labeled.shape)
        GT_mask[GT_labeled == i] = 1
        GT_mask = GT_mask + result_label
        if(np.sum(GT_mask > 1.5) > 0):
            TP = TP + 1
        else:
            FN = FN + 1
    print "result_num: ", result_num
    for ii in xrange(result_num):
        i=ii+1
        result_mask = np.zeros(result_label.shape)
        result_mask[result_labeled == i] = 1
        result_mask = result_mask + GT_label
        if(np.sum(result_mask > 1.5) > 0):
            TP_check = TP_check + 1
        else:
            FP = FP + 1
    print "FN: "+str(FN)
    if(TP != TP_check):
        print "TP wrong!"
    return (TP, FP, FN)


def compute_metrix():
    sensitivity = 0.0
    precision = 0.0
    FP_total=0
    GT_label_list = []
    
    interval = 30
    
    results = []
    
    p = Pool(interval)
    results = p.map(get_TP_FP_FN, names) #load the image in the sub-process, and don't load them in the  main process and pass them as the parameters, otherwise
    #the memory are not sufficient
      
    
    i = 0
    for TP,FP, FN in results:
        FP_total = FP_total + FP
        sensitivity = sensitivity + float(TP / (TP + FN))
        print names[i], 'sensitivity: ', float(TP / (TP + FN)), 'FP:', FP
        i = i+1
        if TP + FP > 0:
            precision = precision +(TP/(TP+FP))
    imgs_num = len(names)
    sensitivity = sensitivity/ imgs_num
    precision = precision/ imgs_num
    FP_avg = FP_total/ imgs_num
    return sensitivity, precision, FP_avg

    # TP_sum = 0
    # FP_sum = 0
    # FN_sum = 0
    # for TP, FP, FN in results:
    #     TP_sum += TP
    #     FP_sum+=FP
    #     FN_sum +=FN
    # sensitivity = TP_sum/(TP_sum + FN_sum)
    # precision = TP_sum/(TP_sum + FP_sum)
    # FP_avg = FP_sum/len(names)
    # print "TP: ", TP_sum
    # print "FN: ", FN_sum
    # print "FP: ", FP_sum
    # return sensitivity, precision, FP_avg



#load_labels()
load_names()
sensitivity, precision, FP_avg = compute_metrix()
print "sensitivity: ", sensitivity
print "precision: ", precision
print "FP_avg: ", FP_avg
