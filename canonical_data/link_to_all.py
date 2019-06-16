#!/usr/bin/python
#Given folder --data, make links for everything in the folder --base
#Useful for: given the AIA data, make a new folder for experiments

import os, argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',dest='base',required=True)
    parser.add_argument('--data',dest='data',required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    folders = os.listdir(args.data)

    for fold in folders:
        os.system("ln -s %s/%s %s/%s" % (args.data,fold,args.base,fold))
