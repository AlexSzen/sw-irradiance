#!/usr/bin/python

import os, pdb, multiprocessing

fns, paths = [], []
for y in range(14,15):
    for i in range(1,13):
        fold = "hm%d_%02d" % (y,i)
        for fn in os.listdir(fold):
            if fn.endswith(".npz"):
                fns.append(fn)
                paths.append("%s/%s" % (fold,fn))

def mkdirp(path):
    s = ""
    for i in range(len(path)):
        s += path[i]+"/"
        if not os.path.exists(s):
            try:
                os.mkdir(s)
            except:
                pass

print len(fns)

target = "HMI_14/"

if not os.path.exists(target):
    os.mkdir(target)

def process(fni):
    fn = fns[fni]
    if fni % 1000 == 0:
        print("%8d/%8d = %.2f" % (fni,len(fns),100*float(fni)/len(fns)))
    year, month, day = fn[11:15], fn[15:17], fn[17:19]
    channel = fn.split("_")[-1].replace(".npz","")

    pathfold = (target,channel,year,month,day)
    mkdirp(pathfold)
    fnpath = "%s/%s" % ("/".join(pathfold),fn.replace("hmi.M_720s.","HMI").replace("00_","_"))

    os.system("cp %s %s" % (paths[fni],fnpath))
    
P = multiprocessing.Pool(16)
P.map(process,range(len(fns)))

