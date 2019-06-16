#!/usr/bin/python
#Given:
#   a folder --src containing fits files
#   a folder --target to contain npz files
#   an integer --scale (a proper divisor of 1024) containing the target output size
#Converts the source fits to target fits:
#   -Rescaling the sun to a constant pixel size and correcting for invalid interpolation

import os, pdb
import numpy as np
import skimage.transform
import argparse
import matplotlib
matplotlib.use('agg',warn=False, force=True)
import sunpy.sun as sun
from sunpy.map import Map
import sunpy.io
import pdb
import warnings
import multiprocessing

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',dest='src',required=True)
    parser.add_argument('--target',dest='target',required=True)
    parser.add_argument('--scale',dest='scale',required=True,type=int)
    args = parser.parse_args()
    return args



def handle(t):
    fn, remote, local, scale = t
    try:
        print(local)
        Xd = Map(remote)
    except:
        print("FILE CORRUPTED: %s" % remote)
        return
    X = Xd.data
    #X = np.load("%s/%s/%s" % (src,p,fn))

    #make a valid mask; we'll use this to correct for downpush when interpolating AIA
    validMask = 1.0 * (X > 0)
    X[np.where(X<=0)] = 0.0

    # Target angular size
    trgtAS = 976.0

    # Scale factor
    rad = Xd.meta['RSUN_OBS']
    scale_factor = trgtAS/rad

    #fix the translation
    t = (X.shape[0]/2.0)-scale_factor*(X.shape[0]/2.0)
    #rescale and keep center
    XForm = skimage.transform.SimilarityTransform(scale=scale_factor,translation=(t,t))

    Xr = skimage.transform.warp(X,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))
    Xd = skimage.transform.warp(validMask,XForm.inverse,preserve_range=True,mode='edge',output_shape=(X.shape[0],X.shape[0]))

    #correct for interpolating over valid pixels
    Xr = np.divide(Xr,(Xd+1e-8))


    #figure out the integer factor to downsample by mean
    divideFactor = np.int(X.shape[0] / scale)

    Xr = skimage.transform.downscale_local_mean(Xr,(divideFactor,divideFactor))

    #cast to fp32
    Xr = Xr.astype('float32')
#    np.save(local,Xr)
    np.savez_compressed(local,x=Xr)


if __name__ == "__main__":
    args = parse_args()

    print(args)
    src, target, scale = args.src, args.target, args.scale

    if not os.path.exists(target):
        os.mkdir(target)

    ts = []
    remoteFns, localFns = [], []
    for p,dirs,files in os.walk(src):
        print(p)
        p = p.replace(src,'')
        tp = "%s/%s" % (target,p)
        if not os.path.exists(tp):
            os.mkdir(tp)

        for fni,fn in enumerate(files):
            if fni % 10 == 0:
                print("\t%04d/%04d" % (fni,len(files)))
            if not fn.endswith(".fits"):
                continue


            ts.append((fn, "%s/%s/%s" % (src,p,fn),"%s/%s/%s" % (target,p,fn.replace(".fits",".npz")), args.scale))

    ts.sort()
    import random
    random.shuffle(ts)

    P = multiprocessing.Pool(4)
    P.map(handle,ts)
