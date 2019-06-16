#!/usr/bin/env python3

'''
This script is to unify all the "SW_resnet" scripts. 
We also output training settings in a more consistent way.
implementation of architecture AIA --> EVE mapping.
We take a resnet from pytorch model zoo,, replace the final fc layer to a 1x14 output
corresponding to EVE, and first layer by a depth 9 convolution corresponding to AIA channels.

'''

import sys
sys.path.append('Utilities/') #add to pythonpath to get Dataset, hardcoded at the moment
sys.path.append('../drn/') #add to pythonpath to get drn models, hardcoded atm.

import matplotlib
matplotlib.use('agg',warn=False, force=True)
from SW_Dataset_bakeoff import *
from SW_visualization import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models,transforms
import numpy as np
import time
import sys
import copy
import argparse
import json
import pdb
from scipy.special import logit

import os
### just to helper to create net directories in results/
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)



def test_model(model, dataloader):
    outputs = []
    for batchI,(inputs, _) in enumerate(dataloader):
        if batchI % 20 == 0:
            print("%06d/%06d" % (batchI,len(dataloader)))
            print(inputs.shape)
        inputs = inputs.cuda(async = True)
        output = model(inputs)
        output = output.cpu().detach().numpy()
        outputs.append(output)
    outputs = np.concatenate(outputs,axis=0)
    return outputs

def isLocked(path):
    if os.path.exists(path):
        return True
    try:
        os.mkdir(path)
        return False
    except:
        return True

def unlock(path):
    os.rmdir(path)


def eve_unscale(y,mean,std,nonlinearity,sigmoid, zscore):
    
    if zscore :
        y = y*std+mean
        if sigmoid:
            y = logit(y) 
        if nonlinearity == "sqrt":
            y = np.power(y,2)
        elif nonlinearity == "log":
            y = np.expm1(y)
    else :
        y *= mean
      
    return y 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',dest='src',required=True)
    parser.add_argument('--models',dest='models',required=True)
    parser.add_argument('--target',dest='target',required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args() 

    #handle setup
    if not os.path.exists(args.target):
        os.mkdir(args.target)

    cfgs = [fn for fn in os.listdir(args.src) if fn.endswith(".json")]
    cfgs.sort()
    
    for cfgi,cfgPath in enumerate(cfgs):

        cfg = json.load(open("%s/%s" % (args.src,cfgPath)))
        print(cfg)
        cfgName = cfgPath.replace(".json","")

        targetBase =  "%s/%s/" % (args.target,cfgName)
        if not os.path.exists(targetBase):
            os.mkdir(targetBase)

        modelBase = "%s/%s" % (args.models,cfgName)
        modelFile = "%s/%s_model.pt" % (modelBase,cfgName)

        print(modelFile)
        if not os.path.exists(modelFile):
            continue

        target = "%s/%s.npy" %(targetBase,cfgName)
        if os.path.exists(target) or isLocked(target+".lock"):
            continue


        sw_net = None
     #   if cfg['arch'] == "resnet_18":
            ### resnet : change input to 9 channels and output to 14
            #sw_net = models.resnet18(pretrained = False)
            #num_ftrs = sw_net.fc.in_features
            #sw_net.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            #sw_net.fc = nn.Linear(num_ftrs, 14)
      #      sw_net = torch.load(modelFile)
      #      sw_net.cuda()
      #  elif cfg['arch'] == "avg_mlp":
       #     sw_net = torch.load(modelFile) 
       #     sw_net.cuda()
       # else:
        sw_net = torch.load(modelFile)
        sw_net.cuda()
            #print("Model not defined")


        ### Some inputs
        #
        data_root = "/fastdata2/data_alex/2011p4_new/"

        EVE_path = "%s/irradiance_6m_residual.npy" % data_root
        EVE_path_orig = "%s/irradiance_6m.npy" % data_root


        resid = np.load("residuals_2011p4_new.npz")

        #now this should saturate the gpus
        #data_root = "/run/shm/AIA_256_20112014/"
        
        
        crop = False
        flip = False
        batch_size = 64
        resolution = 256
        crop_res = 240
        zscore = cfg['zscore']
        
        
        if (zscore): # we apply whatever scaling if zscore is on
            aia_mean = np.load("%s/aia_sqrt_mean.npy" % data_root)
            aia_std = np.load("%s/aia_sqrt_std.npy" % data_root)
            aia_transform = transforms.Compose([transforms.Normalize(tuple(aia_mean),tuple(aia_std))])
        else : # we don't sqrt and just divide by the means. just need to trick the transform
            aia_mean = np.zeros(14)
            aia_std = np.load('%s/aia_mean.npy' % data_root)
            aia_transform = transforms.Compose([transforms.Normalize(tuple(aia_mean),tuple(aia_std))])
            
        ### Dataset & Dataloader for test

        test_real = SW_Dataset(EVE_path_orig, data_root, data_root, resolution, cfg['eve_transform'], cfg['eve_sigmoid'], split = 'test', AIA_transform = aia_transform, crop = crop, flip = flip, crop_res = crop_res, zscore = zscore,self_mean_normalize=True)

        sw_datasets = {x: SW_Dataset(EVE_path, data_root, data_root, resolution, cfg['eve_transform'], cfg['eve_sigmoid'], split = x, AIA_transform = aia_transform, crop = crop, flip = flip, crop_res = crop_res, zscore = zscore,self_mean_normalize=True) for x in ['test']}
        sw_dataloaders = {x: torch.utils.data.DataLoader(sw_datasets[x], batch_size = batch_size, shuffle = False, num_workers=8) for x in ['test']}
        dataset_sizes = {x: len(sw_datasets[x]) for x in ['test']}

        DS = sw_datasets['test']

        prediction = test_model(sw_net, sw_dataloaders['test'])
        prediction = prediction/100
        prediction_us = eve_unscale(prediction, DS.EVE_means, DS.EVE_stds, cfg['eve_transform'], cfg['eve_sigmoid'], zscore)

        absErrorPointwise = np.abs(prediction_us-DS.EVE) / DS.EVE
        absErrorPointwise[DS.EVE <= 0] = np.nan

        resid = np.load("residuals_2011p4_new.npz")
        NP = resid['initTe']+prediction_us

        PR = np.abs(resid['initTe']-test_real.EVE) / test_real.EVE
        PR[test_real.EVE<0] = np.nan
        absErrorLin = np.nanmean(PR,axis=0)

        NPR =np.abs(NP-test_real.EVE) / test_real.EVE
        NPR[test_real.EVE<0] = np.nan
        absError = np.nanmean(NPR,axis=0)

        print("Initial")
        summaryStats = "Min: %.4f Mean: %.4f Median: %.4f Max: %.4f" % (np.min(absErrorLin),np.mean(absErrorLin),np.median(absErrorLin),np.max(absErrorLin))
        print(summaryStats)
        print(absErrorLin)


        summaryStats = "Min: %.4f Mean: %.4f Median: %.4f Max: %.4f" % (np.min(absError),np.mean(absError),np.median(absError),np.max(absError))
        print(summaryStats)
        print(absError)

        resFile = open("%s/%s.txt" % (targetBase,cfgName),"w")
        resFile.write(summaryStats+"\n")
        for i in range(14):
            resFile.write("%.4f " % absError[i])
        resFile.close()

        np.save(target,NP)

        unlock(target+".lock")

