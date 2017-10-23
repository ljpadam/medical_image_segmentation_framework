import sys
import os
import numpy as np
import model as model

if __name__ == '__main__':

    basePath = os.getcwd() # get current path

    params = dict() # all parameters
    params['DataManagerParams'] = dict() # parameters for data manager class
    params['ModelParams'] = dict() # parameters for model
    params['TestParams'] = dict() # parameters for testing

    # params of the algorithm

    params['ModelParams']['device'] = 0 # the id of the GPU
    params['ModelParams']['snapshot'] = 46000 #85000
    params['ModelParams']['dirTrain'] = '/home/ljp/data/lung/manual/randomcrops/training' # the directory of training data
    #params['ModelParams']['dirTest'] = '/home/ljp/from_Dfwang/WML/testing'
    # where we need to save the results (relative to the base path)
    params['ModelParams']['dirResult'] = "./result" # the directory of the results of testing data
    params['ModelParams']['dirValidation']='/home/ljp/data/lung/manual/validation/' #the directory of the validation data
    params['ModelParams']['dirTest']='/home/ljp/data/lung/manual/testing/' #the directory of the testing data
    # params['ModelParams']['dirResult']="/home/ftp/data/output/" #where we need to save the results (relative to the base path)
    # where to save the models while training
    params['ModelParams']['dirSnapshots'] = "/media/ljp/2775494b-5f8e-4d4b-976a-1c1b403a1bb9/pytorch/" # the directory of the model snapshots for training
    params['ModelParams']['tailSnapshots'] = 'lung/vnet/' # the full path of the model snapshots is the join of dirsnapshots and presnapshots
    params['ModelParams']['batchsize'] = 10  # the batch size
    params['ModelParams']['numIterations'] = 80000000  # the number of total training iterations
    params['ModelParams']['baseLR'] = 0.0003  # the learning rate, initial one
    params['ModelParams']['nProc'] = 20  # the number of threads to do data augmentation
    params['ModelParams']['testInterval'] = 2000  # the number of training interations between testing
    params['ModelParams']['device_ids'] = [0, 1] # the id of the GPUs for the multi-GPU


    # params of the DataManager
    params['DataManagerParams']['VolSize'] = np.asarray([64, 64, 24], dtype=int) # the size of the crop image
    params['DataManagerParams']['TestStride'] = np.asarray([64, 64, 24], dtype=int) # the stride of the adjacent crop image in testing phase and validation phase


    # Ture: produce the probaility map in the testing phase, False: produce the  label image
    params['TestParams']['ProbabilityMap'] = False

    model = model.Model(params)
    train = [i for i, j in enumerate(sys.argv) if j == '-train']
    if len(train) > 0:
        model.train() #train model

    test = [i for i, j in enumerate(sys.argv) if j == '-test']
    for i in sys.argv:
        if(i.isdigit()):
            snapnumber = i
            break
    if len(test) > 0:
        model.test(snapnumber) # test model, the snapnumber is the number of the model snapshot
