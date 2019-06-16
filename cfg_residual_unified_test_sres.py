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
import skimage.transform

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
    for batchI,(inputs, _) in enumerate(dataloader):
        if batchI % 20 == 0:
            print("%06d/%06d" % (batchI,len(dataloader)))
        inputs = inputs.cuda(async = True)
        f = model.forward
        output = f(model,inputs)
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



#############Helper functions #######################

def hijack_canet_forward(self, x):
    x = self.features(x)
    x = self.pool(x)
    return x


class AIALoader:
    #super duper easy thing to avoid double csv parsing
    def __init__(self,data_root,split):
        df_indices = pd.read_csv(data_root+split+'.csv')
        self.index_aia = data_root + np.asarray(df_indices[[channel for channel in df_indices.columns[2:-1]]])

    def __getitem__(self,ind):
        index_aia_i = self.index_aia[ind,:]
        AIA_sample = np.asarray( [np.expand_dims(np.load(channel.replace('fits.',''))['x'],axis=0) for channel in index_aia_i], dtype = np.float64 )
        AIA_sample = np.concatenate(AIA_sample,axis=0) 
        return AIA_sample

    def __len__(self):
        return self.index_aia.shape[0]

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
        #if os.path.exists(target) or isLocked(target+".lock"):
        #    continue


        sw_net = None
        sw_net = torch.load(modelFile)


        #rewrite fc (removing dropout) as conv layer
        current_fc = list(sw_net.model.children())[-1]

        newConv = nn.Conv2d(current_fc.in_features,current_fc.out_features,kernel_size=1)
        newConv.bias.data = current_fc.bias.data.clone()
        S = current_fc.weight.shape
        newConv.weight.data = current_fc.weight.view(S[0],S[1],1,1).data.clone()
        #pretend it's the pooling layer, and use a different forward pass that doesn't
        #reshape to make things the right shape for the linear module
        sw_net.pool = newConv
        sw_net.forward = hijack_canet_forward
        sw_net.cuda()


        ### Some inputs
        #
        #data_root = "/fastdata/FDL/trainSetups/2011_new/"

        data_root = args.data_root

        phase = args.phase
        phaseAbbrev = {"train":"Tr","val":"Va","test":"Te"}[phase]

        EVE_path = "%s/irradiance_6m_residual_alt.npy" % data_root
        EVE_path_orig = "%s/irradiance_6m.npy" % data_root

        model = np.load("%s/residual_initial_model.npz" % data_root)

        #decompose into spatially resolvable (mean features), unresolvable
        #(std features, bias)
        w, muPreprocess, sigPreprocess = model['model'], model['mu'], model['sig']
        
        muPreprocess_mean, muPreprocess_sig = muPreprocess[:9], muPreprocess[9:]
        sigPreprocess_mean, sigPreprocess_sig = sigPreprocess[:9], sigPreprocess[9:]
        model_mean, model_sig, model_one = w[:,:9], w[:,9:18], w[:,18]

        #reshape 
        muPreprocess_mean = np.reshape(muPreprocess_mean,[9,1,1])
        sigPreprocess_mean = np.reshape(sigPreprocess_mean,[9,1,1])


        #feats = np.load("%s/mean_std_feats.npz" % data_root)
        #XTe = addOne((feats['X'+phaseAbbrev]-model['mu']) / model['sig'])
        #initialPredict = np.dot(XTe,model['model'].T)

        AIADataset = AIALoader(data_root, phase)


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

        sw_datasets = {x: SW_Dataset(EVE_path, data_root, data_root, resolution, cfg['eve_transform'], cfg['eve_sigmoid'], split = x, AIA_transform = aia_transform, crop = crop, flip = flip, crop_res = crop_res, zscore = zscore,self_mean_normalize=True) for x in [phase]}
        sw_dataloaders = {x: torch.utils.data.DataLoader(sw_datasets[x], batch_size = batch_size, shuffle = False, num_workers=8) for x in [phase]}
        dataset_sizes = {x: len(sw_datasets[x]) for x in [phase]}

        DS = sw_datasets[phase]

        prediction = test_model(sw_net, sw_dataloaders[phase])
        prediction = prediction/100

        DS.EVE_means = np.reshape(DS.EVE_means,(1,DS.EVE_means.size,1,1))
        DS.EVE_stds = np.reshape(DS.EVE_stds,(1,DS.EVE_stds.size,1,1))

        prediction_us = eve_unscale(prediction, DS.EVE_means, DS.EVE_stds, cfg['eve_transform'], cfg['eve_sigmoid'], zscore)

        #ok we have convnet predictions, now make the spatially resolved stuff
        SRESes = []
        for i in range(len(sw_datasets[phase].index_eve)):
            X = AIADataset[i]
            #compute features
            stdFeat = np.std(X,axis=(1,2))
            stdP = np.dot((stdFeat - muPreprocess_sig) / sigPreprocess_sig, model_sig.T)
            oneP = model_one

            XNorm = (X-muPreprocess_mean) / sigPreprocess_mean

            PBlock = np.zeros((w.shape[0],X.shape[1],X.shape[2]))
            for k in range(w.shape[0]):
                model_c = np.reshape(model_mean[k,:],[9,1,1])
                PBlock[k,:,:] = np.sum(XNorm*model_c,axis=0)
            PBlock += np.reshape(model_one+stdP,(14,1,1))
            #upsample prediction_us to 256x256
            
            prediction_us_up = np.zeros((w.shape[0],X.shape[1],X.shape[2]))
            for k in range(w.shape[0]):
                B = prediction_us[i,k,:,:]
                sz = (X.shape[1],X.shape[2])
                xf = skimage.transform.resize
                #defaults do weird stuff that doesn't preserve the average
                prediction_us_up[k,:] = xf(B,sz,anti_aliasing=False,preserve_range=False,clip=False,mode='edge')
           
            #Averaging this over HxW produces the previous predictions 
            SRES = PBlock+prediction_us_up
            SRES = prediction_us_up

            CNNP = prediction_us[i,:,:,:]+np.mean(PBlock,axis=(1,2),keepdims=True)

            #np.save("%s/%06d_eve.npy" % (targetBase,i),PBlock+prediction_us_up)
            #np.save("%s/%06d_eve_cnn.npy" % (targetBase,i),prediction_us_up)
            np.save("%s/%06d_eve_cnnres.npy" % (targetBase,i),CNNP)
            #np.save("%s/%06d_eve_base.npy" % (targetBase,i),PBlock)
            SRESes.append(np.expand_dims(np.vstack([np.min(SRES,axis=(1,2)),np.max(SRES,axis=(1,2))]),axis=0))

        SRESes = np.concatenate(SRESes,axis=0)
        np.save(target,SRESes)

        pdb.set_trace()
        unlock(target+".lock")

