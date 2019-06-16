#!/usr/bin/env python
#find 404s (empty files) and quality != 0 fits files and delete them

import os, multiprocessing, pdb
import sunpy.io
import matplotlib
matplotlib.use('agg',warn=False, force=True)

from sunpy.map import Map


def handle(fn):
    if os.path.getsize(fn) == 0:
        return False
    Xh = sunpy.io.read_file_header(fn)
    if Xh[1]['QUALITY'] > 0:
        return False
    return True


if __name__ == "__main__":
    src = "/data/data4/dfouhey/AIA_2014/"

    condition = lambda fn: True

    toHandle = []
    for p,dirs,files in os.walk(src):
        for fni,fn in enumerate(files):
            if fn.endswith(".fits") and condition(fn):
                toHandle.append("%s/%s" % (p,fn))

    print(len(toHandle))
    P = multiprocessing.Pool(32)
    valid = P.map(handle,toHandle)
    toDelete = [toHandle[i] for i in range(len(toHandle)) if not valid[i]]
    for fn in toDelete:
        print(fn)
        os.unlink(fn)
