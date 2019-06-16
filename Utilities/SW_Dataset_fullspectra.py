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

def aia_scale(aia_sample):
    bad = np.where(aia_sample <= 0.0)
    aia_sample[bad] = 0.0
    return np.sqrt(aia_sample)

### means and stds for EVE preprocessing, computed on sqrt of EVE

#eve_means =  [0.00193715, 0.0014411,  0.00083583, 0.00705061, 0.00564737, 0.00654392,
# 0.00545495, 0.00486332, 0.00354462, 0.00511692, 0.00451823, 0.02023068,
# 0.00331559, 0.00571566]

#eve_stds = [0.00047821, 0.00034981, 0.00020439, 0.00170968, 0.00137376, 0.00159757,
# 0.00134036, 0.00120125, 0.000907,   0.00123816, 0.00126414, 0.00485915,
# 0.00098817, 0.00138324]

### def sigmoid for normalization below
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


### scaling for mean and std processing
def eve_scale(eve_sample, eve_means, eve_stds, scale = 'sqrt', eve_sigmoid = False):   
    
    #### remove -1 values
    bad_eve = np.where(eve_sample == -1)
    eve_sample[bad_eve] = 0.0    
    
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
        
    return eve_sample



class SW_Dataset(Dataset):

    ''' Dataset class to get inputs and labels for AIA-->EVE mapping'''

    def __init__(self, EVE_path, AIA_root, index_file, resolution, EVE_scale, EVE_sigmoid, split = 'train', AIA_transform = None, flip = False, crop = False, crop_res = None):

        ''' Input path for aia and eve index files, as well as data path for EVE.
            We load EVE during init, but load AIA images on the fly'''

        ### data split (train, val, test)
        self.split = split  
        
        ### do we perform random flips?
        self.flip = flip
        
        ### load indices from csv file for the given split
        df_indices = pd.read_csv(index_file+split+'.csv')
        
        ### resolution. normal is 224, if 256 we perform crops
        self.resolution = resolution
        self.crop = crop
        self.crop_res = crop_res
        
        ### all AIA channels. first two columns are junk
        self.index_aia = AIA_root + np.asarray(df_indices[[channel for channel in df_indices.columns[2:-1]]])
        #self.index_aia = AIA_root + np.asarray(df_indices[[channel for channel in df_indices.columns[1:10]]])

        ### last column is EVE index
        self.index_eve = np.asarray(df_indices[df_indices.columns[-1]])
       
        ### EVE processing. What scaling do we use? Do we apply sigmoid? Pass means and stds (computed on scaled EVE)
        ### They are located in the index file (csv) path)
        ### Name is hardcoded based on David's normalization files
        
        self.EVE_scale = EVE_scale
        self.EVE_sigmoid = EVE_sigmoid
        self.EVE_means = np.load(index_file + 'eve_'+EVE_scale+'_mean.npy')
        self.EVE_stds = np.load(index_file + 'eve_'+EVE_scale+'_std.npy')
        if (EVE_sigmoid):
            self.EVE_means = np.load(index_file+ 'eve_'+EVE_scale+'sigmoid'+'_mean.npy')
            self.EVE_stds = np.load(index_file + 'eve_'+EVE_scale+'sigmoid'+'_std.npy')
        
        
        full_EVE = np.load(EVE_path)   
        #self.EVE = np.zeros((len(self.index_eve), full_EVE.shape[1]))
        self.EVE = full_EVE[self.index_eve,:]
                        
        ### AIA transform : means and stds of sqrt(AIA)
        self.AIA_transform = AIA_transform

        ### Check for inconsistencies
        #data length
        if (len(self.index_eve) != len(self.index_aia)):
            raise ValueError('Time length of EVE and AIA are different')
            
        #crop arguments
        if (self.crop and self.crop_res == None):
            raise ValueError('If crops are on, please specify a crop resolution')
        if (self.crop and self.crop_res > self.resolution):
            raise ValueError('Cropping resolution must be smaller than initial resolution')
        
        print('Loaded ' + split + ' Dataset')

    def __len__(self):

        ### return the number of time steps
        return len(self.index_eve)

    def __getitem__(self, index):

        ### AIA is [time_steps, 9, resolution, resolution]
        ### EVE is [time_steps, 14]
        
        ### load AIA on the fly. load to float64 otherwise pytorch complains (could do float32 too)
        ### replace the 'fits.' from the file to adapt for the new names
        AIA_sample = np.asarray( [np.load(channel.replace('fits.','')) for channel in self.index_aia[index, :]], dtype = np.float32 ) 
        
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
            AIA_sample = torch.from_numpy(aia_scale(AIA_sample_temp.copy()))
        else:
            AIA_sample = torch.from_numpy(aia_scale(AIA_sample))  
            
        
        EVE_sample = torch.from_numpy(eve_scale(self.EVE[index, :], self.EVE_means, 
                                      self.EVE_stds, self.EVE_scale, self.EVE_sigmoid))
    
        if (self.AIA_transform):
            AIA_sample = self.AIA_transform(AIA_sample)

        return AIA_sample, EVE_sample
