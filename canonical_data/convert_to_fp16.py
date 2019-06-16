#!/usr/bin/python
#Given:
#   a folder --src containing npz files
#   a folder --target to contain npz files
#Converts the source npz to target npz:
#   -converting to float16 so long as it doesn't result in infs or nans

import os, pdb
import numpy as np
import argparse
import warnings
import multiprocessing

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',dest='src',required=True)
    parser.add_argument('--target',dest='target',required=True)
    args = parser.parse_args()
    return args

def handle(t):
    fn, remote, local = t

    X = np.load(remote)['x']
    X16 = X.astype(np.float16)

    if np.any(np.isinf(X16).ravel()) or np.any(np.isnan(X16).ravel()):
        #print("Truncating would overflow fp")
        X16 = X

    np.savez_compressed(local,x=X16)

if __name__ == "__main__":
    args = parse_args()

    print(args)
    src, target = args.src, args.target

    if not os.path.exists(target):
        os.mkdir(target)

    print("Making directories")
    ts = []
    remoteFns, localFns = [], []
    for p,dirs,files in os.walk(src):
        p = p.replace(src,'')
        tp = "%s/%s" % (target,p)
        if not os.path.exists(tp):
            os.mkdir(tp)
        
        for fni,fn in enumerate(files):
            if not fn.endswith(".npz"):
                continue

            ts.append((fn, "%s/%s/%s" % (src,p,fn),"%s/%s/%s" % (target,p,fn)))

    ts.sort()
    import random
    random.shuffle(ts)

    print("Converting")
    P = multiprocessing.Pool(4)
    P.map(handle,ts)
