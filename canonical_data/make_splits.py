#!/usr/bin/env python
#Take a csv file and turn it into {train,val,test}.csv

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',dest='src',required=True)
    parser.add_argument('--splits',dest='splits',required=False,default='rve',help='rve=train/val/test;rv=train/val;re=train/test')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    TRAIN_END = 0.7
    VAL_END = 0.9

    args = parse_args()

    lines = open(args.src).read().strip().split("\n")
    #lines.replace('fits.','')
    header, rest = lines[0], lines[1:]
    
    print(args.src)
    
    N = len(rest)
    splTr = int(TRAIN_END*N)
    splVal = int(VAL_END*N)

    train = rest[:splTr]
    if args.splits == 'rv':
        val = rest[splTr:]
        test = []
    elif args.splits == 're':
        test = rest[splTr:]
        val = []
    else: #(args.splits == 'rve' or anything else )
        val = rest[splTr:splVal]
        test = rest[splVal:]
        
    base = args.src.rsplit("/",1)[0]
  
    open(base+"/train.csv","w").write(header+"\n"+("\n".join(train)))
    if len(val):
        open(base+"/val.csv","w").write(header+"\n"+("\n".join(val)))
    if len(test):
        open(base+"/test.csv","w").write(header+"\n"+("\n".join(test)))





