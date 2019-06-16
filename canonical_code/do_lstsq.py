#!/usr/bin/env python3

import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats',dest='feats',required=True,help='Feature file')
    parser.add_argument('--target',dest='target',required=True,help='Prediction file')
    parser.add_argument('--phase',dest='phase',default='test',help='Split to predict {train/val/test}')
    args = parser.parse_args()
    return args


def addone(X):
    return np.concatenate([X,np.ones((X.shape[0],1))],axis=1)

if __name__ == "__main__":

    args = parse_args()

    data = np.load(args.feats)
        
    XTr = addone(data['XTr'])

    W,_,_,_ = np.linalg.lstsq(XTr,data['yTr'],rcond=-1)

    if args.phase == 'train':
        yTrp = np.dot(XTr,W)
        np.save(args.target,yTrp)
    elif args.phase == 'val':
        XVa = addone(data['XVa'])
        yVap = np.dot(XVa,W)
        np.save(args.target,yVap)
    else:
        XTe = addone(data['XTe'])
        yTep = np.dot(XTe,W)
        np.save(args.target,yTep)

