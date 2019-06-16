#!/usr/bin/env python3

import numpy as np
np.warnings.filterwarnings('ignore')
import pandas as pd
import argparse
import pdb
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',dest='base',required=True,help='Root of the data')
    parser.add_argument('--predictions',dest='predictions',required=True,help='Prediction folder')
    parser.add_argument('--phase',dest='phase',default='test',help='Split that this is a prediction of {train/val/test}')
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

    npys = []
    for dirname, folds, fns in os.walk(args.predictions):
        for fn in fns:
            if fn.endswith(".npy"):
                npys.append("%s/%s" % (dirname,fn))
    npys.sort()

    for fn in npys:
        print(fn)
        os.system("./evaluate_prediction.py --base %s --phase %s --prediction %s" % (args.base,args.phase,fn)) 


