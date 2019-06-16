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

#from SW_Dataset import *
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


def hijack_resnet_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)

    return x


def hijack_canet_forward(self, x):
    x = self.features(x)
    x = self.pool(x)
    return x



import os
### just to helper to create net directories in results/
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)



def test_model(model, dataloader):
    allInputs, allOutputs = [], []
    for batchI,(inputs, _) in enumerate(dataloader):
        allInputs.append(inputs.cpu().detach().numpy())
        inputs = inputs.cuda(async = True)
        f = model.forward
        output = f(model,inputs)
        output = output.cpu().detach().numpy()
        allOutputs.append(output)
    allOutputs = np.concatenate(allOutputs,axis=0)
    allInputs = np.concatenate(allInputs,axis=0)
    return allOutputs, allInputs

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


def eve_unscale(y,mean,std,nonlinearity,sigmoid):
    y = y*std+mean
    if sigmoid:
        y = logit(y) 
    if nonlinearity == "sqrt":
        y = np.power(y,2)
    elif nonlinearity == "log":
        y = np.expm1(y)
    return y 


def eve_unscale_tensor(y,mean,std,nonlinearity,sigmoid):
    y = y*np.reshape(std,(1,14,1,1))+np.reshape(mean,(1,14,1,1))
    if sigmoid:
        y = logit(y) 
    if nonlinearity == "sqrt":
        y = np.power(y,2)
    elif nonlinearity == "log":
        y = np.expm1(y)
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


        sw_net = torch.load(modelFile)
        if cfg['arch'] == "resnet_18":
            ### resnet : change input to 9 channels and output to 14
            #sw_net = models.resnet18(pretrained = False)
            #num_ftrs = sw_net.fc.in_features
            #sw_net.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            #sw_net.fc = nn.Linear(num_ftrs, 14)
            sw_net.cuda()

            current_fc = sw_net.fc

            newConv = nn.Conv2d(current_fc.in_features,current_fc.out_features,kernel_size=1)
            newConv.bias.data = current_fc.bias.data.clone()
            S = current_fc.weight.shape
            newConv.weight.data = current_fc.weight.view(S[0],S[1],1,1).data.clone()
            sw_net.avgpool = newConv
            sw_net.fc = nn.Sequential()
            sw_net.forward = hijack_resnet_forward

        elif cfg['arch'] == "avg_mlp":
            sw_net = torch.load(modelFile) 
            sw_net.cuda()

        elif cfg['arch'].startswith("anet"):

            current_fc = sw_net.model
            newConv = nn.Conv2d(current_fc.in_features,current_fc.out_features,kernel_size=1)
            newConv.bias.data = current_fc.bias.data.clone()
            S = current_fc.weight.shape
            newConv.weight.data = current_fc.weight.view(S[0],S[1],1,1).data.clone()
            sw_net.pool = newConv
            sw_net.model = nn.Sequential()
            sw_net.forward = hijack_canet_forward

        else:
            print("Model not defined")


        ### Some inputs
        EVE_path = '/data/NASAFDL2018/SpaceWeather/Team1-Meng/EVE/np/irradiance.npy'
        data_root = "%s/" % cfg['data_csv_dir'] #'/scratch/AIA_2011_224/'

        #now this should saturate the gpus
        data_root = "/run/shm/AIA_2011_256/"

        batch_size = 64
        resolution = 224
        
        aia_mean = np.load("%s/aia_sqrt_mean.npy" % data_root)
        aia_std = np.load("%s/aia_sqrt_std.npy" % data_root)
        aia_transform = transforms.Compose([transforms.Normalize(tuple(aia_mean),tuple(aia_std))])

        ### Dataset & Dataloader for test
        sw_datasets = {x: SW_Dataset(EVE_path, data_root, data_root, resolution, cfg['eve_transform'], cfg['eve_sigmoid'], split = x, AIA_transform = aia_transform) for x in ['test_mini']}
        sw_dataloaders = {x: torch.utils.data.DataLoader(sw_datasets[x], batch_size = batch_size, shuffle = False, num_workers=8) for x in ['test_mini']}
        dataset_sizes = {x: len(sw_datasets[x]) for x in ['test_mini']}

        DS = sw_datasets['test_mini']
        prediction, input_tensor = test_model(sw_net, sw_dataloaders['test_mini'])

        prediction_us = eve_unscale_tensor(prediction, DS.EVE_means, DS.EVE_stds, cfg['eve_transform'], cfg['eve_sigmoid'])

        np.save(target,prediction_us)
        np.save(target+".input.npy",input_tensor)

        unlock(target+".lock")

