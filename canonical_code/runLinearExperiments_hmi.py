#!/usr/bin/python

import os

target = "linear_exp_hmi/"

if not os.path.exists(target):
    os.mkdir(target)

srcs = {"weeklyhmi":"/fastdata/FDL/trainSetups/weeklySplit_HMI/",
        "2011":"/fastdata/FDL/trainSetups/HMI_2011/",
        "year":"/fastdata/FDL/trainSetups/HMI_2011p4/"}

feats = ["mean","meanstd"]

if 0:
    for src in srcs.keys():
        srcPath = srcs[src]
        for feat in feats:
            if os.path.exists("%s/%s_%s.npz" % (target,src,feat)):
                continue

            com = "./dump_baseline_feat.py --base %s/ --target %s/%s_%s.npz --feat %s --numworkers 8" % (srcPath,target,src,feat,feat)
            print(com)
            os.system(com)
if 0:
    for src in srcs.keys():
        srcPath = srcs[src]
        for feat in feats:
            for phase in ["train","val","test"]:
                com = "./do_lstsq.py --feats %s/%s_%s.npz" % (target,src,feat)
                com += " --target %s/%s_%s_%s.npy" % (target,src,feat,phase)
                com += " --phase %s" % phase
                print com
                os.system(com)


if 0:
    for src in srcs.keys():
        srcPath = srcs[src]
        for feat in feats:
            print("%s -- %s" % (src,feat))
            for phase in ["train","val","test"]:
                com = "./evaluate_prediction.py --base %s" % (srcPath)
                com += " --prediction %s/%s_%s_%s.npy" % (target,src,feat,phase)
                com += " --phase %s" % phase
                os.system(com)

if 0:
    for src in srcs.keys():
        srcPath = srcs[src]
        for feat in feats:
            for phase in ["train","val","test"]:
                com = "./do_huber.py --feats %s/%s_%s.npz" % (target,src,feat)
                com += " --target %s/%s_%s_%s_huber.npy" % (target,src,feat,phase)
                com += " --phase %s" % phase
                print com
                os.system(com)

if 1:
    for src in srcs.keys():
        srcPath = srcs[src]
        for feat in feats:
            print("%s -- %s" % (src,feat))
            for phase in ["train","val","test"]:
                com = "./evaluate_prediction.py --base %s" % (srcPath)
                com += " --prediction %s/%s_%s_%s_huber.npy" % (target,src,feat,phase)
                com += " --phase %s" % phase
                os.system(com)

