import numpy as np
import os
import pdb
import multiprocessing

def checkForNan(fn):
    Xs = np.load(fn)
    X = Xs['x']

    if np.any(np.isinf(X)[:]) or np.any(np.isnan(X)[:]):
        print("Inf/NaN in %s" % fn)
        return fn

    return None

if __name__ == "__main__":
    srcs = ["/data/data%d/dfouhey/NP_AIA_%d_256/" % (i,2010+i) for i in range(1,4)]
    srcs += ["/data/data%d/dfouhey/AIA_%d_256/" % (i,2010+i) for i in range(1,4)]

    srcs = ["/data/data%d/dfouhey/AIA_512" % i for i in range(1,4)]
    #srcs = ["/opt/bigdata/FDL/AIA/"]

    for src in srcs:
        print(src)
        count = 0
        files = []
        for base, _, fns in os.walk(src):
            for fn in fns:
                if not fn.endswith(".npz"):
                    continue
                files.append(base+"/"+fn)
        #print(len(files))
        P = multiprocessing.Pool(64)
        files = P.map(checkForNan,files)
        files = [fn for fn in files if fn is not None]
        print(files)

