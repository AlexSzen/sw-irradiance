#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import multiprocessing
import argparse

#helper function only handles global functions


def handleMean(index_aia_i):
    AIA_sample = np.asarray( [np.expand_dims(np.load(channel.replace('fits.',''))['x'],axis=0) for channel in index_aia_i], dtype = np.float64 )
    AIA_sample = np.concatenate(AIA_sample,axis=0) 
    X = np.mean(AIA_sample,axis=(1,2))
    return np.expand_dims(X,axis=0)


def handleStd(index_aia_i):
    AIA_sample = np.asarray( [np.expand_dims(np.load(channel.replace('fits.',''))['x'],axis=0) for channel in index_aia_i], dtype = np.float64 )
    AIA_sample = np.concatenate(AIA_sample,axis=0) 
    X = np.mean(AIA_sample,axis=(1,2))
    X = np.concatenate([X,np.std(AIA_sample,axis=(1,2))],axis=0)
    return np.expand_dims(X,axis=0)


__featToFunction = {'mean':handleMean,'meanstd':handleStd}

def getXy(EVE_path,data_root,split,function,numworkers):
    EVE = np.load(EVE_path)

    EVE = EVE[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14]]

    df_indices = pd.read_csv(data_root+split+'.csv')
    index_aia = data_root + np.asarray(df_indices[[channel for channel in df_indices.columns[2:-1]]])
    index_eve = np.asarray(df_indices[df_indices.columns[-1]]).astype(int)

    Xs, ys = [], []

    fnList = []
    for i in range(0,len(index_eve),1):
        
        fns = index_aia[i,:] 
        fnList.append(fns)

        y = EVE[index_eve[i],:]

        ys.append(np.expand_dims(y,axis=0))

    print("Handling %d files for split %s" % (len(fnList),split))
    P = multiprocessing.Pool(numworkers)
    Xs = P.map(function,fnList)
    P.close()

    X, y = np.concatenate(Xs,axis=0), np.concatenate(ys,axis=0)
   
    mask = y < 0
    y[mask] = 0
 
    return X, y, mask



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',dest='base',required=True,help='Root of the data')
    parser.add_argument('--target',dest='target',required=True,help='Target .npz file')
    parser.add_argument('--feat',dest='feat',default='mean',help='What to put in the file')
    parser.add_argument('--numworkers',dest='numworkers',default=4,type=int,help='How many workers to use')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    data_root = args.base

    if args.feat not in __featToFunction:
        print("Invalid feature '%s', must be one of %s" % (args.feat,",".join(__featToFunction.keys())))
    function = __featToFunction[args.feat]

    ### Some inputs
    EVE_path = "%s/irradiance_6m.npy" % (data_root)

    XTr, yTr, maskTr = getXy(EVE_path, data_root, "train",function,args.numworkers)
    XVa, yVa, maskVa = getXy(EVE_path, data_root, "val",function,args.numworkers)
    XTe, ___, ______ = getXy(EVE_path, data_root, "test",function,args.numworkers)

    np.savez(args.target,**{'XTr':XTr,'XTe':XTe,'XVa':XVa,'yTr':yTr,'yVa':yVa,'maskTr':maskTr,'maskVa':maskVa})


