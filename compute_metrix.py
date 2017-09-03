import sys
import os
import numpy as np
import SimpleITK as sitk
from os import listdir
from os.path import isfile, join, splitext
from scipy import ndimage
from scipy.ndimage import measurements
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

ground_truth_path = '/home/ljp/from_Dfwang/WML_corrected/testing'
results_path = './result/'
ground_truth_labels = dict()
results_labels = dict()


def load_labels():
    for folder in listdir(ground_truth_path):
        label = sitk.Cast(sitk.ReadImage(
            join(join(ground_truth_path, folder), 'label.nii')), sitk.sitkFloat32)
        # temp=sitk.GetArrayFromImage(label)
        # print temp.shape
        ground_truth_labels[folder] = np.transpose(sitk.GetArrayFromImage(
            label).astype(dtype=float), [2, 0, 1])
        # print ground_truth_labels[folder].shape
        label = sitk.Cast(sitk.ReadImage(
            join(results_path, folder + "_result.nii")), sitk.sitkFloat32)

        results_labels[folder] = np.transpose(sitk.GetArrayFromImage(
            label).astype(dtype=float), [2, 0, 1])
        results_labels[folder] = (
            results_labels[folder] ).astype(dtype=np.uint8)
        print results_labels[folder].shape










def dilation(img):
    s = ndimage.generate_binary_structure(3, 3)
    img = ndimage.binary_dilation(img, structure=s).astype(img.dtype)
    return img


def save_img(img, path):
    # print img.shape
    result = np.transpose(img, [1, 2, 0])
    # print result.shape
    toWrite = sitk.GetImageFromArray(result)
    toWrite = sitk.Cast(toWrite, sitk.sitkUInt8)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(toWrite)


def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError(
            "Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def compute_metrix():
    dice_tot = 0.0
    for label_name in ground_truth_labels:

        GT_label = ground_truth_labels[label_name]
        result_label = results_labels[label_name]
        temp_dice = dice(GT_label, result_label)
        print "label_name: " + str(label_name), " ", temp_dice
        dice_tot += temp_dice
    dice_avg = dice_tot / len(ground_truth_labels)
    return dice_avg

#return only the pixels contained in the FOV, for both images and masks

def computeAUC(results, labels):
    accuracy_tot = 0
    specificity_tot = 0
    sensitivity_tot = 0
    precision_tot = 0
    for name in results:
        result = np.copy(results[name])
        result = result.reshape([result.size])
        label = np.copy(labels[name])
        label = label.reshape((result.size))
        fpr, tpr, thresholds = roc_curve(label, result)
        AUC_ROC = roc_auc_score(label, result)
        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(label, result)
        # so the array is increasing (you won't get negative AUC)
        precision = np.fliplr([precision])[0]
        # so the array is increasing (you won't get negative AUC)
        recall = np.fliplr([recall])[0]
        AUC_prec_rec = np.trapz(precision, recall)

    # Confusion matrix
        threshold_confusion = 0.5
        y_pred = np.zeros((result.shape[0]))
        y_pred[result>=threshold_confusion] = 1
        confusion = confusion_matrix(label, y_pred)
        accuracy = 0
        if float(np.sum(confusion)) != 0:
            accuracy = float(
                confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        accuracy_tot += accuracy
        
        specificity = 0
        if float(confusion[0, 0] + confusion[0, 1]) != 0:
            specificity = float(confusion[0, 0]) / \
                float(confusion[0, 0] + confusion[0, 1])
        specificity_tot += specificity
        
        sensitivity = 0
        if float(confusion[1, 1] + confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1]) / \
                float(confusion[1, 1] + confusion[1, 0])
        sensitivity_tot += sensitivity
        precision = 0
        if float(confusion[1, 1] + confusion[0, 1]) != 0:
            precision = float(confusion[1, 1]) / \
                float(confusion[1, 1] + confusion[0, 1])
        precision_tot += precision

    print "Acc: ", accuracy_tot/len(results)
    print "spec: ", specificity_tot/len(results)
    print 'Sens: ', sensitivity_tot/len(results)
    print 'Prec: ', precision_tot/len(results)
        


load_labels()
dice_avg = compute_metrix()
print "dice: ", dice_avg
computeAUC(results_labels, ground_truth_labels)

