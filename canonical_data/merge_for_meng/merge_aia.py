#!/usr/bin/python

import os, pdb, multiprocessing

fns, paths = [], []

src = "2016/"
for pWalk, _, fnsWalk in os.walk(src):
    for fn in fnsWalk:
        if fn.endswith(".npz"):
            fns.append(fn)
            paths.append("%s/%s" % (pWalk,fn))
    
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

target = "/fastdata2/FDL/AIA_Future/"

if not os.path.exists(target):
    os.mkdir(target)

def process(fni):
    fn = fns[fni]
    if fni % 1000 == 0:
        print("%8d/%8d = %.2f" % (fni,len(fns),100*float(fni)/len(fns)))

    year, month, day = fn[3:7], fn[7:9], fn[9:11]
    channel = str(int(fn.split("_")[-1].replace(".npz","")))

    pathfold = (target,channel,year,month,day)
    mkdirp(pathfold)
    fnpath = "%s/%s" % ("/".join(pathfold),fn)
    os.system("cp %s %s" % (paths[fni],fnpath))

#process(0)    
P = multiprocessing.Pool(16)
P.map(process,range(len(fns)))

