#!/usr/bin/env python3

import numpy as np
import argparse
import pdb
import pickle
from sklearn.ensemble import GradientBoostingRegressor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats',dest='feats',required=True,help='Feature file')
    parser.add_argument('--target',dest='target',required=True,help='Prediction file')
    parser.add_argument('--phase',dest='phase',default='test',help='Split to predict {train/val/test}')
    args = parser.parse_args()
    return args


def addone(X):
    return np.concatenate([X,np.ones((X.shape[0],1))],axis=1)

def fitGBR(X,y,**kwargs):
    models = []
    for j in range(y.shape[1]):
        regr = GradientBoostingRegressor(random_state=0,loss="huber",**kwargs)
        regr.fit(X,y[:,j])
        models.append(regr)
    return models

def applyGBR(X,models):
    yp = []
    for j in range(len(models)): 
        yp.append(np.expand_dims(models[j].predict(X),axis=1))
    return np.concatenate(yp,axis=1)

def getResid(y,yp,mask):
    resid = np.abs(y-yp)
    resid = resid / np.abs(y)
    resid[mask] = np.nan
    return np.nanmean(resid,axis=0)



if __name__ == "__main__":
    Y_SCALE = 100000

    args = parse_args()

    data = np.load(args.feats)

    XTr, XTe, XVa = data['XTr'], data['XTe'], data['XVa']
    yTr, yVa = data['yTr'], data['yVa']

    mu, sig = np.mean(XTr,axis=0), np.std(XTr,axis=0)

    XTr = addone((XTr-mu)/sig)
    XVa = addone((XVa-mu)/sig)
    XTe = addone((XTe-mu)/sig)

    #scale the y so that we're not at ridiculous values that are close to zero
    yTr, yVa = yTr*Y_SCALE, yVa*Y_SCALE

    bestParams, bestError = None, np.inf
    for alpha in [0.8,0.9,0.95]:
        for max_depth in [2,3,4]:
            for min_samples_leaf in [50,100,150,200]:
                model = fitGBR(XTr,yTr,alpha=alpha,max_depth=max_depth,min_samples_leaf=min_samples_leaf,verbose=0,n_estimators=10)
                yVap = applyGBR(XVa,model)

                residVa = getResid(yVa,yVap,data['maskVa'])
                perf = np.mean(residVa)
                print("%f, %d, %d -> %.2f" % (alpha,max_depth,min_samples_leaf,perf*100))

                if perf < bestError:
                    bestParams, bestError = (alpha, max_depth, min_samples_leaf), perf

    model = fitGBR(XTr,yTr,alpha=bestParams[0],max_depth=bestParams[1],min_samples_leaf=min_samples_leaf,n_estimators=100,verbose=1)

    if args.phase == 'train':
        yTrp = applyGBR(XTr,model)/Y_SCALE
        np.save(args.target,yTrp)
    elif args.phase == 'val':
        yVap = applyGBR(XVa,model)/Y_SCALE
        np.save(args.target,yVap)
    else:
        yTep = applyGBR(XTe,model)/Y_SCALE
        np.save(args.target,yTep)

