#!/usr/bin/env python3

'''
Pytorch Dataset class to load AIA and EVE dataset.
Currently takes all 9 AIA channels and outputs 14 EVE channels. For now in EVE we discard MEGS-B data because
undersampled, as well as channel 14 because heavily undersampled.
'''


import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import sys
import math
import pdb
import skimage.transform

def aia_scale(aia_sample, zscore = True, self_mean_normalize=False):
    if not self_mean_normalize:
        bad = np.where(aia_sample <= 0.0)
        aia_sample[bad] = 0.0
    
    if (zscore): ### if zscore return sqrt 
        return np.sign(aia_sample) * np.sqrt(np.abs(aia_sample))
    else: ### otherwise we just wanna divide the unsrt image by the means
        return aia_sample
      
    

### means and stds for EVE preprocessing, computed on sqrt of EVE


### def sigmoid for normalization below
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


### scaling for mean and std processing
def eve_scale(eve_sample, eve_means, eve_stds, scale = 'sqrt', eve_sigmoid = False, zscore = True):   
    
    #### remove -1 values
    bad_eve = np.where(eve_sample ==-1)
    eve_sample[bad_eve] = 0.0    
    if (zscore):
        if (scale == 'sqrt'): 

            eve_sample = np.sqrt(eve_sample)       
            if (eve_sigmoid):
                eve_sample = sigmoid(eve_sample)            
            eve_sample -= eve_means
            eve_sample /= eve_stds

        elif (scale == 'log'):

            eve_sample = np.log1p(eve_sample)       
            if (eve_sigmoid):
                eve_sample = sigmoid(eve_sample)            
            eve_sample -= eve_means
            eve_sample /= eve_stds 
            
        else :
            raise ValueError('Unknown scaling argument')
        
    else: ### don't do any scaling just divide by the means 
        eve_sample /= eve_means
        
    return eve_sample



class SW_Dataset(Dataset):

    ''' Dataset class to get inputs and labels for AIA-->EVE mapping'''

    def __init__(self, EVE_path, AIA_root, index_file, resolution, EVE_scale, EVE_sigmoid, split = 'train', AIA_transform = None, flip = False, crop = False, crop_res = None, zscore = True, self_mean_normalize=False):

        ''' Input path for aia and eve index files, as well as data path for EVE.
            We load EVE during init, but load AIA images on the fly'''

        ### data split (train, val, test)
        self.split = split  
        
        ### do we perform random flips?
        self.flip = flip
        
        df_indices = pd.read_csv(index_file+'index.csv')
        
        ### resolution. normal is 224, if 256 we perform crops
        self.resolution = resolution
        self.crop = crop
        self.crop_res = crop_res
        
        self.self_mean_normalize = self_mean_normalize
        

        self.index_aia = AIA_root + np.asarray(df_indices[[channel for channel in df_indices.columns[1:-1]]])

        ### AIA transform : means and stds of sqrt(AIA)
        self.AIA_transform = AIA_transform

            
        #crop arguments
        if (self.crop and self.crop_res == None):
            raise ValueError('If crops are on, please specify a crop resolution')
        if (self.crop and self.crop_res > self.resolution):
            raise ValueError('Cropping resolution must be smaller than initial resolution')
        
        print('Loaded inference dataset with ' + str(len(self.index_aia))+' examples' )

    def __len__(self):

        ### return the number of time steps
        return len(self.index_aia)

    def __getitem__(self, index):

        ### Training in paper is done on 256 images but new data is 512 so we downsample here.
        AIA_sample = np.asarray( [np.expand_dims(np.load(channel.replace('fits.',''))['x'],axis=0) for channel in self.index_aia[index, :]], dtype = np.float32 )
        AIA_sample = np.concatenate(AIA_sample,axis=0)
        divide=2
        AIA_down = np.asarray( ( [np.expand_dims(divide*divide*skimage.transform.downscale_local_mean(AIA_sample[i,:,:], (divide, divide)), axis=0) for i in range(AIA_sample.shape[0])]), dtype=np.float32 )
        AIA_sample = np.concatenate(AIA_down, axis = 0)
        if self.self_mean_normalize:
            AIA_sample = AIA_sample - np.mean(AIA_sample,axis=(1,2),keepdims=True)
        
        ### if resolution is above 224, we perform random crops
        if (self.crop):
            hcrop_start = np.random.randint(0, high = (self.resolution - self.crop_res) + 1)
            hcrop_end = hcrop_start + self.crop_res
            
            vcrop_start = np.random.randint(0, high = (self.resolution - self.crop_res) + 1)
            vcrop_end = vcrop_start + self.crop_res
            
            AIA_sample = AIA_sample[:,vcrop_start : vcrop_end, hcrop_start : hcrop_end]
            

        ### random flips
        if (self.flip):
            AIA_sample_temp = AIA_sample
            p = np.random.rand()
            if (p>0.5):
                AIA_sample_temp = np.flip(AIA_sample_temp, axis = 2)
            d = np.random.rand()
            if (d>0.5):
                AIA_sample_temp = np.flip(AIA_sample_temp, axis = 1)
                
            ### need to make a copy because pytorch doesnt support negative strides yet    
            AIA_sample = torch.from_numpy(aia_scale(AIA_sample_temp.copy(), self.zscore,self.self_mean_normalize))
        else:
            AIA_sample = torch.from_numpy(aia_scale(AIA_sample, self.zscore, self.self_mean_normalize))  
            
       
    
        if (self.AIA_transform):
            AIA_sample = self.AIA_transform(AIA_sample)

        return AIA_sample
