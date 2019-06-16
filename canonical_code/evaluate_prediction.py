#!/usr/bin/env python3

import numpy as np
np.warnings.filterwarnings('ignore')
import pandas as pd
import argparse
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',dest='base',required=True,help='Root of the data')
    parser.add_argument('--predictions',dest='predictions',required=True,help='Prediction file')
    parser.add_argument('--phase',dest='phase',default='test',help='Split that this is a prediction of {train/val/test}')
    parser.add_argument('--doflare',dest='doflare',default=0,help='Whether to show flare results')
    args = parser.parse_args()
    return args

def getResid(y,yhat,mask):
    resid = np.abs(y-yhat)
    resid[mask] = np.nan
    resid = resid / y
    return np.nanmean(resid,axis=0)

def printAnalysis(resid,names):
    print("[",end='')
    for i in range(len(resid)):
        if i != 0:
            print("; ",end='')
        print("%s: %.1f" % (names[i],resid[i]),end='')
    print("]")

    minv, meanv, medv, maxv = np.min(resid), np.mean(resid), np.median(resid), np.max(resid)
    print("Min\tMean\tMedian\tMax")
    print("%.2f\t%.2f\t%.2f\t%.2f" % (minv, meanv, medv, maxv))

if __name__ == "__main__":

    args = parse_args()
    yhat = np.load(args.predictions)
    
    EVE = np.load("%s/irradiance_6m.npy" % (args.base))
    flareIndictor = np.load("%s/flare_indicator.npy" % args.base)
    names = np.load("%s/name.npy" % args.base)
    chanUse = [0,1,2,3,4,5,6,7,8,9,10,11,12,14]

    EVE = EVE[:,chanUse]
    names = [n.strip() for n in names[chanUse]]

    df_indices = pd.read_csv(args.base+"/"+args.phase+'.csv')
    index_eve = np.asarray(df_indices[df_indices.columns[-1]]).astype(int)

    y = EVE[index_eve,:]
    flareIndictor = flareIndictor[index_eve]

    #make the mask
    mask = y<0
    y[mask] = 0

    resid = getResid(y,yhat,mask)*100
    printAnalysis(resid,names)

    if args.doflare:
        flareYes = flareIndictor>0
        residFlare = getResid(y[flareYes,:],yhat[flareYes,:],mask[flareYes,:])

        flareNo = flareIndictor==0
        residNoFlare = getResid(y[flareNo,:],yhat[flareNo,:],mask[flareNo,:])

        print("Flare")  
        printAnalysis(residFlare*100,names)

        print("No Flare")
        printAnalysis(residNoFlare*100,names)


