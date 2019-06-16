#!/usr/bin/python

import pdb, datetime

base = "/fastdata/FDL/trainSetups/weeklySplit_new/"
src = "%s/all.csv" % base
lines = open(src).read().strip().split("\n")

outputs = [t % base for t in ["%s/train.csv","%s/val.csv","%s/test.csv"]]
outputs = [open(fn,"w") for fn in outputs]
inds = [(0,2,4),(1,),(3,)]

for j in range(len(outputs)):
    outputs[j].write(lines[0]+"\n")

for i in range(1,len(lines)):
    l = map(int,lines[i].split(",")[0].split()[0].split("-"))
    w = datetime.date(l[0],l[1],l[2]).isocalendar()[1]
    wm5 = w % 5
    for j in range(len(inds)):
        if wm5 in inds[j]:
            outputs[j].write(lines[i]+"\n")
    print(w)

map(lambda f: f.close(), outputs)
