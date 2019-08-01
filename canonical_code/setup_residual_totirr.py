#!/usr/bin/env python3
#
# Take a folder and setup the targets for predicting y-y_{linear}
#

import sys
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.linear_model import SGDRegressor
import pdb
import argparse

def getEVEInd(data_root,split):
    xind = [0,1,2,3,4,5,6,7,8,9,10,11,12,14,-1]
    
    df_indices = pd.read_csv(data_root+split+'.csv')
    yind = np.asarray(df_indices[df_indices.columns[-1]]).astype(int)

    return yind, xind


def handleStd(index_aia_i):
    # Factor 4 because new different image resolution (512) than when training (256)
    AIA_sample = np.asarray( [np.expand_dims(np.load(channel.replace('fits.',''))['x'],axis=0) for channel in index_aia_i], dtype = np.float64 )
    AIA_sample = np.concatenate(AIA_sample,axis=0)
    divide=2
    AIA_down = np.asarray( ( [np.expand_dims(divide*divide*skimage.transform.downscale_local_mean(AIA_sample[i,:,:], (divide, divide)), axis=0) for i in range(AIA_sample.shape[0])]), dtype=np.float64 )
    AIA_sample = np.concatenate(AIA_down, axis = 0)
    X = np.mean(AIA_sample,axis=(1,2))
    X = np.concatenate([X,np.std(AIA_sample,axis=(1,2))],axis=0)
    return np.expand_dims(X,axis=0)


def getXy(EVE_path,data_root,split):
    EVE = np.load(EVE_path)

    EVE = EVE[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,14,-1]]

    df_indices = pd.read_csv(data_root+split+'.csv')
    index_aia = data_root + np.asarray(df_indices[[channel for channel in df_indices.columns[2:-1]]])
    index_eve = np.asarray(df_indices[df_indices.columns[-1]]).astype(int)

    Xs, ys = [], []

    fnList = []
    for i in range(0,len(index_eve)):
        
        fns = index_aia[i,:] 
        fnList.append(fns)

        if i % 100 == 0:
            print("%s %d/%d" % (split,i,len(index_eve)))

        y = EVE[index_eve[i],:]

        ys.append(np.expand_dims(y,axis=0))
    
    P = multiprocessing.Pool(16)
    Xs = P.map(handleStd,fnList)
    P.close()

    X, y = np.concatenate(Xs,axis=0), np.concatenate(ys,axis=0)
   
    mask = y < 0
    y[mask] = 0
 
    return X, y, mask

def addOne(X):
    return np.concatenate([X,np.ones((X.shape[0],1))],axis=1)

def getResid(y,yp,mask,flare=None,flarePct=0.975):
    resid = np.abs(y-yp)
    resid = resid / np.abs(y)
    resid[mask] = np.nan
    return np.nanmean(resid,axis=0)

def fitSGDR_Huber(X,Y,maxIter=10,epsFrac=1.0,logalpha=-4):
    alpha = 10**logalpha
    K = Y.shape[1]
    models = []
    for j in range(K): 
        model = SGDRegressor(loss='huber',alpha=alpha,max_iter=maxIter,epsilon=np.std(Y[:,j])*epsFrac,learning_rate='invscaling',random_state=1,fit_intercept=False)
        model.fit(X,Y[:,j])
        models.append(model)
    return models

def applySGDmodel(X,models):
    yp = []
    for j in range(len(models)): 
        yp.append(np.expand_dims(models[j].predict(X),axis=1))
    return np.concatenate(yp,axis=1)

def cvSGDH(XTr,yTr,XVa,yVa,maskVa):
    bestPerformance, bestP, bestA = np.inf, 1, 0
    print("CV'ing huber epsilon, regularization")
    for p in range(1,10,1):
        for a in range(-5,1):
            model = fitSGDR_Huber(XTr,yTr,maxIter=10,epsFrac=1.0/p,logalpha=a)
            
            yTrp = applySGDmodel(XTr,model)
            yVap = applySGDmodel(XVa,model)
            residVa = getResid(yVa,yVap,maskVa)
            perf = np.mean(residVa)
            print("a = 10e%d, eps = %f => %f" % (a, 1.0 / p, perf))
            if perf < bestPerformance:
                bestPerformance, bestP, bestA = perf, p, a

    print("Best a = 10e%d, eps = %f" % (bestA,1.0/bestP))
    model = fitSGDR_Huber(XTr,yTr,maxIter=100,epsFrac=1.0/bestP,logalpha=bestA)
    yTrp = applySGDmodel(XTr,model)
    yVap = applySGDmodel(XVa,model)
    W = np.concatenate([np.expand_dims(m.coef_,axis=0) for m in model],axis=0)
    return W

def getNormalize(XTr):
    mu = np.mean(XTr,axis=0)
    sig = np.std(XTr,axis=0)
    sig[sig==0] = 1e-8

    return mu, sig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',dest='base',required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    data_root = args.base
    EVE_path = "%s/irradiance_30mn_14ptot.npy" % data_root

    #get the data
    XTe, ___, ______ = getXy(EVE_path, data_root, "test")
    XTr, yTr, maskTr = getXy(EVE_path, data_root, "train")
    XVa, yVa, maskVa = getXy(EVE_path, data_root, "val")

    np.savez_compressed("%s/mean_std_feats.npz" % data_root,XTr=XTr,XVa=XVa,XTe=XTe)

    mu, sig = getNormalize(XTr)

    XTr = addOne((XTr-mu) / sig)
    XVa = addOne((XVa-mu) / sig)
    XTe = addOne((XTe-mu) / sig)

    model = cvSGDH(XTr,yTr,XVa,yVa,maskVa)

    #Predictions = X*W'
    yTrp = np.dot(XTr,model.T)
    yVap = np.dot(XVa,model.T) 
    yTep = np.dot(XTe,model.T)

    #these are the new targets
    diffTr = yTr - yTrp; diffTr[maskTr] = 0
    diffVa = yVa - yVap; diffVa[maskVa] = 0
    
    #update EVE
    EVE = np.load(EVE_path)
    updates = [("train",diffTr),("val",diffVa)]
    for phaseName,newVals in updates:
        yind, xind = getEVEInd(data_root, phaseName)
        for yii,yi in enumerate(yind):
            EVE[yi,xind] = newVals[yii,:]


    #new statistics
    residualMean = np.mean(diffTr,axis=0)   
    residualStd = np.std(diffTr,axis=0)   
 
    np.save("%s/eve_residual_mean_14ptot.npy" % data_root, residualMean)
    np.save("%s/eve_residual_std_14ptot.npy" % data_root, residualStd)

    #rescale targets
    EVE *= 100

    #Save the new target and the model
    np.save("%s/irradiance_30mn_residual_14ptot.npy" % data_root, EVE)
    np.savez_compressed("%s/residual_initial_model.npz" % data_root,model=model,mu=mu,sig=sig)

    
