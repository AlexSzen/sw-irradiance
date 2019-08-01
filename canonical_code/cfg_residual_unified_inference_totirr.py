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
from SW_Dataset_inference import *
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
torch.backends.cudnn.benchmark = False
import os
### just to helper to create net directories in results/
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def printErrors(R):
    names = np.load("/fastdata2/FDL/EVE/np/name.npy")
    inds = [0,1,2,3,4,5,6,7,8,9,10,11,12,14]
    names = names[inds]
    names = np.append(names,'tot_irr_megsa')
    for i in range(R.shape[0]):
        if i != 0:
            print("; ",end=' '),
        print("%s: %.2f%%" % (names[i].strip(),R[i]),end=' ')
    print('')

def getResid(y,yp,mask,flare=None,flarePct=0.975):
    resid = np.abs(y-yp)
    resid = resid / np.abs(y) * 100
    resid[mask] = np.nan
    if flare is None:
        return np.nanmean(resid,axis=0)
    else:
        N = y.shape[0]
        FeXX = y[:,2]
        order = np.argsort(FeXX)
        cutoff = int(y.shape[0]*flarePct)
        if flare:
            keep = order[cutoff:]
        else:
            keep = order[:cutoff]
        return np.nanmean(resid[keep,:],axis=0)

def print_analysis(y,yp,mask):
    print("Overall")
    printErrors(getResid(y,yp,mask))
    print("Flare")
    printErrors(getResid(y,yp,mask,flare=True))
    print("Non-Flare")
    printErrors(getResid(y,yp,mask,flare=False))

def test_model(model, dataloader):
    outputs = []
    for batchI,inputs in enumerate(dataloader):
        if batchI % 20 == 0:
            print("%06d/%06d" % (batchI,len(dataloader)))
        torch.backends.cudnn.benchmark = False
        inputs = inputs.cuda(async = True)
        torch.backends.cudnn.benchmark = False
        output = model(inputs)
        torch.backends.cudnn.benchmark = False
        output = output.cpu().detach().numpy()
        torch.backends.cudnn.benchmark = False
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

def addOne(X):
    return np.concatenate([X,np.ones((X.shape[0],1))],axis=1)

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
    parser.add_argument('--data_root',dest='data_root',required=True)
    parser.add_argument('--phase',dest='phase',default='test')
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


        torch.backends.cudnn.benchmark = False
        sw_net = None
        sw_net = torch.load(modelFile)
        sw_net.cuda()


        torch.backends.cudnn.benchmark = False
        ### Some inputs
        #
        #data_root = "/fastdata/FDL/trainSetups/2011_new/"

        data_root = args.data_root

        phase = args.phase
        phaseAbbrev = {"train":"Tr","val":"Va","test":"Te"}[phase]

#        EVE_path = "%s/irradiance_30mn_residual_14ptot.npy" % data_root
#        EVE_path_orig = "%s/irradiance_30mn_14ptot.npy" % data_root

        model = np.load("%s/residual_initial_model.npz" % data_root)
        feats = np.load("%s/mean_std_feats.npz" % data_root)

        XTe = addOne((feats['X']-model['mu']) / model['sig'])
        initialPredict = np.dot(XTe,model['model'].T)


        #resid = np.load("residuals_2011_new.npz")

        
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


        sw_datasets = {x: SW_Dataset(data_root, data_root, data_root, resolution, cfg['eve_transform'], cfg['eve_sigmoid'], split = x, AIA_transform = aia_transform, crop = crop, flip = flip, crop_res = crop_res, zscore = zscore,self_mean_normalize=True) for x in [phase]}
        sw_dataloaders = {x: torch.utils.data.DataLoader(sw_datasets[x], batch_size = batch_size, shuffle = False, num_workers=8) for x in [phase]}
        dataset_sizes = {x: len(sw_datasets[x]) for x in [phase]}

        DS = sw_datasets[phase]

        prediction = test_model(sw_net, sw_dataloaders[phase])
        prediction = prediction/100
        prediction_us = eve_unscale(prediction, DS.EVE_means, DS.EVE_stds, cfg['eve_transform'], cfg['eve_sigmoid'], zscore)


        #NP = resid['init'+phaseAbbrev]+prediction_us
        NP = initialPredict + prediction_us

 #       PR = np.abs(initialPredict-test_real.EVE) / test_real.EVE
 #       PR[test_real.EVE<0] = np.nan
 #       absErrorLin = np.nanmean(PR,axis=0)

  #      NPR =np.abs(NP-test_real.EVE) / test_real.EVE
   #     NPR[test_real.EVE<0] = np.nan
    #    absError = np.nanmean(NPR,axis=0)

        np.save(target,NP)

        unlock(target+".lock")

