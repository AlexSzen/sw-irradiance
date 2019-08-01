#!/usr/bin/env python3

'''
Take a folder and setup the AIA features for the linear model to do inference.
'''

import sys
import numpy as np
import skimage.transform
import pandas as pd
import multiprocessing
from sklearn.linear_model import SGDRegressor
import pdb
import argparse


def handleStd(index_aia_i):
    # Factor 4 because different image resolution (512) than when training (256)
    AIA_sample = np.asarray( [np.expand_dims(np.load(channel.replace('fits.',''))['x'],axis=0) for channel in index_aia_i], dtype = np.float64 )
    AIA_sample = np.concatenate(AIA_sample,axis=0)
    divide=2
    AIA_down = np.asarray( ( [np.expand_dims(divide*divide*skimage.transform.downscale_local_mean(AIA_sample[i,:,:], (divide, divide)), axis=0) for i in range(AIA_sample.shape[0])]), dtype=np.float64 )
    AIA_sample = np.concatenate(AIA_down, axis = 0)
    X = np.mean(AIA_sample,axis=(1,2))
    X = np.concatenate([X,np.std(AIA_sample,axis=(1,2))],axis=0)
    return np.expand_dims(X,axis=0)

def getX(data_root):


    df_indices = pd.read_csv(data_root+'index.csv')
    index_aia = data_root + np.asarray(df_indices[[channel for channel in df_indices.columns[1:-1]]])

    Xs = []

    fnList = []
    for i in range(0,len(index_aia)):

        fns = index_aia[i,:]
        fnList.append(fns)

        if i % 100 == 0:
            print("%d/%d" % (i,len(index_aia)))


    P = multiprocessing.Pool(16)
    Xs = P.map(handleStd,fnList)
    P.close()


    X = np.concatenate(Xs,axis=0)


    return X



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',dest='base',required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    data_root = args.base

    X = getX(data_root)
    print("after get X")
    np.savez_compressed("%s/mean_std_feats.npz" % data_root,X = X)

