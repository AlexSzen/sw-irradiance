#!/usr/bin/env python3

import numpy as np
np.warnings.filterwarnings('ignore')
import argparse
from sklearn.linear_model import SGDRegressor

def getResid(y,yp,mask,flare=None,flarePct=0.975):
    resid = np.abs(y-yp)
    resid = resid / np.abs(y)
    resid[mask] = np.nan
    return np.nanmean(resid,axis=0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats',dest='feats',required=True,help='Feature file')
    parser.add_argument('--target',dest='target',required=True,help='Prediction file')
    parser.add_argument('--phase',dest='phase',default='test',help='Split to predict {train/val/test}')
    args = parser.parse_args()
    return args

def addone(X):
    return np.concatenate([X,np.ones((X.shape[0],1))],axis=1)

def fitSGDR_Huber(X,Y,maxIter=10,epsFrac=1.0,logalpha=-4):
    alpha = 10**logalpha
    K = Y.shape[1]
    models = []
    for j in range(K): 
        model = SGDRegressor(loss='huber',alpha=alpha,max_iter=maxIter,epsilon=np.std(Y[:,j])*epsFrac,learning_rate='invscaling',random_state=1)
        model.fit(X,Y[:,j])
        models.append(model)
    return models

def applySGDmodel(X,models):
    yp = []
    for j in range(len(models)): 
        yp.append(np.expand_dims(models[j].predict(X),axis=1))
    return np.concatenate(yp,axis=1)


if __name__ == "__main__":

    args = parse_args()

    data = np.load(args.feats)

    XTr, XVa = data['XTr'], data['XVa']
    mu, std = np.mean(XTr,axis=0), np.std(XTr,axis=0)

    yTr, yVa = data['yTr'], data['yVa']
    XTr, XVa = addone((XTr-mu)/std), addone((XVa-mu)/std)

    bestPerformance, bestP, bestA = np.inf, 1, 0
    print("CV'ing huber epsilon, regularization")
    for p in range(1,10,1):
        for a in range(-5,1):
            model = fitSGDR_Huber(XTr,yTr,maxIter=10,epsFrac=1.0/p,logalpha=a)
            
            yTrp = applySGDmodel(XTr,model)
            yVap = applySGDmodel(XVa,model)
            residVa = getResid(yVa,yVap,data['maskVa'])
            perf = np.mean(residVa)
            print("a = 10e%d, eps = %f => %f" % (a, 1.0 / p, perf))
            if perf < bestPerformance:
                bestPerformance, bestP, bestA = perf, p, a

    model = fitSGDR_Huber(XTr,yTr,maxIter=100,epsFrac=1.0/bestP,logalpha=bestA)

    if args.phase == 'train':
        yTrp = applySGDmodel(XTr,model)
        np.save(args.target,yTrp)
    elif args.phase == 'val':
        yVap = applySGDmodel(XVa,model)
        np.save(args.target,yVap)
    else:
        XTe = addone((data['XTe']-mu)/std)
        yTep = applySGDmodel(XTe,model)
        np.save(args.target,yTep)

