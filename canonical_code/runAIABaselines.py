#!/usr/bin/python

import os, multiprocessing

target = "linear_exp/"

if not os.path.exists(target):
    os.mkdir(target)

feats = ["mean","meanstd"]

srcs = {"weekly":"/fastdata2/FDL/trainSetups/weeklySplit_new/", "year":"/fastdata2/FDL/trainSetups/2011p4_new_fp16/", "2011":"/fastdata2/FDL/trainSetups/2011_new/"}

#dump data
for src in srcs.keys():
    srcPath = srcs[src]
    for feat in feats:
        com = "./dump_baseline_feat.py --base %s/ --target %s/%s_%s.npz --feat %s --numworkers 32" % (srcPath,target,src,feat,feat)
        print(com)
        #os.system(com)

#learning
allComs = []
for src in srcs.keys():
    srcPath = srcs[src]
    for feat in feats:
        for phase in ["train","val","test"]:

            #OLS
            com = "./do_lstsq.py --feats %s/%s_%s.npz" % (target,src,feat)
            com += " --target %s/%s_%s_%s.npy" % (target,src,feat,phase)
            com += " --phase %s" % phase
            allComs.append(com)

            #Huber
            com = "./do_huber.py --feats %s/%s_%s.npz" % (target,src,feat)
            com += " --target %s/%s_%s_%s_huber.npy" % (target,src,feat,phase)
            com += " --phase %s" % phase
            allComs.append(com)

            #Gradient Boosting
            com = "./do_gboost.py --feats %s/%s_%s.npz" % (target,src,feat)
            com += " --target %s/%s_%s_%s_gboost.npy" % (target,src,feat,phase)
            com += " --phase %s" % phase
            allComs.append(com)


P = multiprocessing.Pool(16)
P.map(os.system,allComs)


print ("="*40)+"OLS"+("="*40)
for src in srcs.keys():
    srcPath = srcs[src]
    for feat in feats:
        print("%s -- %s" % (src,feat))
        for phase in ["train","val","test"]:
            com = "./evaluate_prediction.py --base %s" % (srcPath)
            com += " --prediction %s/%s_%s_%s.npy" % (target,src,feat,phase)
            com += " --phase %s" % phase
            os.system(com)


print ("="*40)+"Huber"+("="*40)
for src in srcs.keys():
    srcPath = srcs[src]
    for feat in feats:
        print("%s -- %s" % (src,feat))
        for phase in ["train","val","test"]:
            com = "./evaluate_prediction.py --base %s" % (srcPath)
            com += " --prediction %s/%s_%s_%s_huber.npy" % (target,src,feat,phase)
            com += " --phase %s" % phase
            os.system(com)

print ("="*40)+"GBoost"+("="*40)
for src in srcs.keys():
    srcPath = srcs[src]
    for feat in feats:
        print("%s -- %s" % (src,feat))
        for phase in ["train","val","test"]:
            com = "./evaluate_prediction.py --base %s" % (srcPath)
            com += " --prediction %s/%s_%s_%s_gboost.npy" % (target,src,feat,phase)
            com += " --phase %s" % phase
            os.system(com)

