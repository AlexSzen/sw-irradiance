#!/usr/bin/env python3
#
#

'''
Given a directory of config files, trains nets to map
9 AIA channels to 15 EVE MEGS-A channels using a previously
fitted linear model (see setup_residual_totirr.py)
'''

import sys
sys.path.append('Utilities/') #add to pythonpath to get Dataset, hardcoded at the moment

from SW_Dataset_bakeoff_totirr import *
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
import alt_models
import random

import os
### just to helper to create net directories in results/
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
        
### helper training function. 
def train_model(model, dataloaders, device, criterion, optimizer, scheduler, num_epochs):

    best_weights = copy.deepcopy(model.state_dict())
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):

        print('Epoch %f / %f' %(epoch, num_epochs))

        for phase in ['train', 'val']:

            ### set the model in training or evaluation state
            if phase =='train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0
            running_count = 0


            phase_losses = []

            for batchI,(inputs, labels) in enumerate(dataloaders[phase]):
                ### needs to be switched to device 0 for DataParallel to work, not sure why they did it this way.
                with torch.cuda.device(0):
                    inputs = inputs.cuda(async = True)
                    labels = labels.cuda(async = True)


                    ### zero the param gradients
                optimizer.zero_grad()

                ### forward, track only for train phase
                with torch.set_grad_enabled(phase=='train'):
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                ### backward + optimize if in training phase
                if phase=='train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
                    optimizer.step()

                pastLoss = running_loss*1.0 / (running_count+1e-8)

                running_loss += loss.item() * inputs.size(0)
                running_count += inputs.size(0)


                phase_losses.append(loss.item())

                newLoss = (running_loss*1.0) / (running_count+1e-8)

                if batchI % 20 == 0:
                    print("%06d/%06d" % (batchI,len(dataloaders[phase])),end='')
                    print("   Mean: %8.5f P25: %8.5f Median: %8.5f P75: %8.5f Max: %8.5f" % (sum(phase_losses)/len(phase_losses),np.percentile(phase_losses,25),np.median(phase_losses),np.percentile(phase_losses,75),max(phase_losses)))
                
             
            epoch_loss = running_loss/dataset_sizes[phase]
            print('%s loss = %f' %(phase, epoch_loss))
            if phase == 'train':
                train_loss_list.append(epoch_loss)
            else : 
                val_loss_list.append(epoch_loss)

            ### take first loss as reference at first epoch
            if phase == 'val' and epoch == 0:
                worst_loss = epoch_loss

            ### keep track of best weights if model improves on validation set
            if phase == 'val' and epoch_loss < worst_loss:
                best_weights = copy.deepcopy(model.state_dict())
                worst_loss = epoch_loss



    ### load best weights into model
    model.load_state_dict(best_weights)
    return model, np.asarray(train_loss_list), np.asarray(val_loss_list)

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',dest='src',required=True)
    parser.add_argument('--target',dest='target',required=True)
    parser.add_argument('--data_root',dest='data_root',required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args() 

    #handle setup
    if not os.path.exists(args.target):
        os.mkdir(args.target)

    cfgs = [fn for fn in os.listdir(args.src) if fn.endswith(".json")]
    cfgs.sort()
    
    random.shuffle(cfgs)

    
    for cfgi,cfgPath in enumerate(cfgs):

        cfg = json.load(open("%s/%s" % (args.src,cfgPath)))

        print(cfg)

        cfgName = cfgPath.replace(".json","")

        targetBase =  "%s/%s/" % (args.target,cfgName)
        if not os.path.exists(targetBase):
            os.mkdir(targetBase)

        modelFile = "%s/%s_model.pt" % (targetBase,cfgName)
        logFile = "%s/%s_log.txt" % (targetBase, cfgName)

        print(modelFile)
        if os.path.exists(modelFile) or isLocked(modelFile+".lock"):
            continue

        sw_net = None
        
        if cfg['arch'] == "resnet_18":
            ### resnet : change input to 9 channels and output to 14
            sw_net = models.resnet18(pretrained = False)
            num_ftrs = sw_net.fc.in_features
            sw_net.avgpool = nn.AdaptiveAvgPool2d((1,1))
            sw_net.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            if cfg['dropout']:
                sw_net.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_ftrs, 15)
                )
            else:
                sw_net.fc = nn.Linear(num_ftrs, 15)
        
        elif cfg['arch'] == "drn_d_22":
            sw_net = drn_d_22(pretrained = False)
            num_ftrs = sw_net.fc.in_channels
            sw_net.layer0[0] = nn.Conv2d(9, 16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
            if cfg['dropout']:
                sw_net.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Conv2d(num_ftrs, 15, kernel_size=(1, 1), stride=(1, 1))
                )
            else:
                sw_net.fc = nn.Conv2d(num_ftrs, 15, kernel_size=(1, 1), stride=(1, 1))  
                
        elif cfg['arch'] == "avg_mlp":
            sw_net = alt_models.AvgMLP(9,1024,15, cfg['dropout'])
        elif cfg['arch'] == 'augresnet':
            sw_net = alt_models.AugResnet2(models.resnet18(pretrained=False), 15, cfg['dropout'])
        elif cfg['arch'] == 'augCNN':
            sw_net = alt_models.AugmentedCNN()
        elif cfg['arch'] == 'avgLinear':
            sw_net = alt_models.AvgLinearMap(9,15)

        elif cfg['arch'].startswith("anet64bn"):
            layerCount = int(cfg['arch'].split("_")[1])
            sw_net = alt_models.ChoppedAlexnet64BN(layerCount,15,cfg['dropout'])

        elif cfg['arch'].startswith("anet64"):
            layerCount = int(cfg['arch'].split("_")[1])
            sw_net = alt_models.ChoppedAlexnet64(layerCount,15,cfg['dropout'])

        elif cfg['arch'].startswith("anet"):
            layerCount = int(cfg['arch'].split("_")[1])
            if (len(cfg['arch'].split("_")) == 2): ###name is anet_numlayers
                sw_net = alt_models.ChoppedAlexnet(layerCount,15,cfg['dropout'])
            elif (len(cfg['arch'].split("_")) == 3): ### name is anet_numlayers_bn
                sw_net = alt_models.ChoppedAlexnetBN(layerCount,15,cfg['dropout'])
            else:
                raise ValueError('Arch not defined')
        elif cfg['arch'].startswith("vgg"):
            layerCount = int(cfg['arch'].split("_")[1])
            sw_net = alt_models.ChoppedVGG(layerCount,15,cfg['dropout'])
        else:
            print("Model not defined")
        
        sw_net = sw_net.cuda()

        criterion = None
        if cfg['loss'] == "L2":
            criterion = nn.MSELoss()
        elif cfg['loss'] == "L1":
            criterion = nn.SmoothL1Loss()


        ### Some inputs

        #now this should saturate the gpus
        #data_root = "/fastdata/FDL/trainSetups/2011p2_new/"
        data_root = args.data_root

        EVE_path = "%s/irradiance_30mn_residual_14ptot.npy" % data_root
        #'/scratch/EVE/irradiance.npy'

        csv_dir = data_root
        #csv_dir = '/run/shm/AIA_2011_256/'
        crop = cfg['crop']
        batch_size = 64
        resolution = 256
        crop_res = 240
        flip = cfg['flip']
        zscore = cfg['zscore']
        
        
        if (zscore): # we apply whatever scaling if zscore is on
            aia_mean = np.load("%s/aia_sqrt_mean.npy" % data_root)
            aia_std = np.load("%s/aia_sqrt_std.npy" % data_root)
            aia_transform = transforms.Compose([transforms.Normalize(tuple(aia_mean),tuple(aia_std))])
        else : # we don't sqrt and just divide by the means. just need to trick the transform
            aia_mean = np.zeros(14)
            aia_std = np.load('%s/aia_mean.npy' % data_root)
            aia_transform = transforms.Compose([transforms.Normalize(tuple(aia_mean),tuple(aia_std))])
            
        ### Dataset & Dataloader for train and validation
        sw_datasets = {x: SW_Dataset(EVE_path, data_root, csv_dir, resolution, cfg['eve_transform'], cfg['eve_sigmoid'], split = x, AIA_transform = aia_transform, flip=flip, crop = crop, crop_res = crop_res, zscore = zscore,self_mean_normalize=True) for x in ['train', 'val']}

        sw_dataloaders = {x: torch.utils.data.DataLoader(sw_datasets[x], batch_size = batch_size, shuffle = True, num_workers=8) for x in ['train', 'val']}

        dataset_sizes = {x: len(sw_datasets[x]) for x in ['train', 'val']}

        sw_datasets['train'][0]

        initLR, stepSize = cfg['lr_start'], cfg['lr_step_size']
        weightDecay = cfg['weight_decay'] 
        n_epochs = stepSize*3

        optimizer = optim.SGD(sw_net.parameters(), lr=initLR, momentum=0.9, nesterov = True,weight_decay=weightDecay)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=stepSize, gamma=0.1)


        print(sw_net)
        print(optimizer)

        training_time = time.time()
        sw_net, train_loss, val_loss = train_model(sw_net, sw_dataloaders, "cuda", criterion, optimizer, exp_lr_scheduler, n_epochs)
        training_time = time.time() - training_time

        ### save net, losses and info in corresponding net directory
        torch.save(sw_net, modelFile)

        np.save("%s/train_loss.npy" % targetBase, train_loss)
        np.save("%s/val_loss.npy" % targetBase, val_loss)

        net_name = cfgName

        F = open(logFile, 'w')

        F.write('Info and hyperparameters for ' + net_name + '\n')
        F.write('Loss : ' + str(criterion) + '\n')
        F.write('Optimizer : ' + str(optimizer) + '\n')
        F.write('Scheduler : ' + str(exp_lr_scheduler) + ', step_size : '+ str(exp_lr_scheduler.step_size)+', gamma : ' + str(exp_lr_scheduler.gamma)+'\n')
        F.write('Number of epochs : ' + str(n_epochs) + '\n')
        F.write('Batch size : ' +str(batch_size) + '\n')
        F.write('Image resolution :' + str(resolution) + '\n')
        F.write('Training time : '+ str(training_time)+'s')
        F.write('Net architecture : \n' + str(sw_net))
        F.close()

        unlock(modelFile+".lock")

