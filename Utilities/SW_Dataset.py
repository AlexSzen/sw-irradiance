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

#def eve_scale(eve_sample):    
#    bad_eve = np.where(eve_sample == -1)
#    eve_sample[bad_eve] = 0.0
    #just a small test to scale the inputs differently based on looking at the data
#    test = np.asarray([1000, 1000, 2000, 500, 500, 500, 500, 500, 500, 500, 500, 100, 500, 500])
#    return test*np.sqrt(eve_sample)

### means and stds for EVE preprocessing, computed on sqrt of EVE

eve_means =  [0.00193715, 0.0014411,  0.00083583, 0.00705061, 0.00564737, 0.00654392,
 0.00545495, 0.00486332, 0.00354462, 0.00511692, 0.00451823, 0.02023068,
 0.00331559, 0.00571566]

eve_stds = [0.00047821, 0.00034981, 0.00020439, 0.00170968, 0.00137376, 0.00159757,
 0.00134036, 0.00120125, 0.000907,   0.00123816, 0.00126414, 0.00485915,
 0.00098817, 0.00138324]

### different scaling for mean and std processing
def eve_scale(eve_sample):    
    bad_eve = np.where(eve_sample == -1)
    eve_sample[bad_eve] = 0.0
    eve_sample = np.sqrt(eve_sample)
    eve_sample -= eve_means
    eve_sample /= eve_stds
    
    return eve_sample



class SW_Dataset(Dataset):

    ''' Dataset class to get inputs and labels for AIA-->EVE mapping'''

    def __init__(self, EVE_path, AIA_root, index_file, resolution, split = 'train', AIA_transform = None, flips = False):

        ''' Input path for aia and eve index files, as well as data path for EVE.
            set can be train, val, or test. train/val/test splits are 60/20/20 %, hardcoded at the moment'''
        
        self.flips = flips

        df_indices = pd.read_csv(index_file)
        ### all AIA channels. first two columns are junk
        index_aia = AIA_root + np.asarray(df_indices[[channel for channel in df_indices.columns[2:-1]]])
        ### last column is EVE index
        index_eve = np.asarray(df_indices[df_indices.columns[-1]])
        ### train/val/test splits
        start_percent = 0.
        end_percent = 0.
        if split == 'train':
            end_percent = 0.7
        elif split == 'val':
            start_percent = 0.7
            end_percent = 0.85
        elif split == 'test':
            start_percent = 0.85
            end_percent = 1.
        else:
            raise ValueError('Unknown set type')
        
        
        len_set = len(np.asarray(range(int(start_percent * index_aia.shape[0]), int(end_percent*index_aia.shape[0]))))
       
        ### need to get rid of the line 13 because no measurements !
        full_EVE = np.load(EVE_path)[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14]]    
        self.EVE = np.zeros((len_set, full_EVE.shape[1]))
        self.AIA = np.zeros((len_set, index_aia.shape[1], resolution, resolution))

        
        count = 0 #need a different counter because it is the file index counter
        for it in range(int(start_percent * index_aia.shape[0]), int(end_percent*index_aia.shape[0])):
                #if (count%10==0):print(count)
                self.AIA[count, :] = np.asarray( [np.load(channel) for channel in index_aia[it, :]] ) 
                self.EVE[count, :] = full_EVE[index_eve[it], : ]
                count += 1
	    ### only keep relevant indices in EVE
        
        print(self.AIA.shape)
        print(self.EVE.shape)

        self.AIA_transform = AIA_transform

        # Check that data length is consistent
        if (len(self.EVE) != len(self.AIA)):
            raise ValueError('Time length of EVE and AIA are different')
        print('Loaded ' + split + ' Dataset')

    def __len__(self):

        ### return the number of time steps
        return len(self.EVE)

    def __getitem__(self, index):

        ### AIA is [time_steps, 9, resolution, resolution]
        ### EVE is [time_steps, 14]

        AIA_sample = aia_scale(self.AIA[index,:,:,:])
        EVE_sample = torch.from_numpy(eve_scale(self.EVE[index, :]))
        
        ### random flips
        
        if self.flips:
            p = np.random.rand()
            if (p>0.5):
                AIA_sample_temp = np.flip(AIA_sample, axis = 2)
            d = np.random.rand()
            if (d>0.5):
                AIA_sample_temp = np.flip(AIA_sample, axis = 1)
            
            ### need a copy because pytorch doesn't support negative strides     
            AIA_sample = torch.from_numpy(AIA_sample_temp.copy())
        else:
            AIA_sample = torch.from_numpy(AIA_sample)
        
        if (self.AIA_transform):
            AIA_sample = self.AIA_transform(AIA_sample)
        

       
        return AIA_sample, EVE_sample
