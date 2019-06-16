#!/usr/bin/python
#Download the JSOC data

import os
import datetime
import multiprocessing
import pdb

#http://jsoc.stanford.edu/data/aia/synoptic/2011/10/09/H0700/AIA20111009_0706_0094.fits
#http://jsoc.stanford.edu/data/aia/synoptic/2011/10/09/H0700/AIA20111009_0706_0094.fits
srcUrl = "http://jsoc.stanford.edu/data/aia/synoptic/"
targetBase = "/data/data1/dfouhey/AIA_2011/"

def fitsfn(year,month,day,hour,minute,wavelength):
    fn = "AIA%04d%02d%02d_" % (year,month,day)
    fn += "%02d%02d_" % (hour,minute)
    fn += "%04d.fits" % (wavelength)
    return fn

def remoteFn(year,month,day,hour,minute,wavelength):
    dayprefix = "%s/%04d/%02d/%02d/" % (srcUrl,year,month,day)
    timebit = "H%02d00/" % (hour)
    fn = fitsfn(year,month,day,hour,minute,wavelength)
    return "%s%s%s" % (dayprefix,timebit,fn)

def localFn(year,month,day,hour,minute,wavelength):
    prefix = "%s/%d/%04d/%02d/%02d/"% (targetBase,wavelength,year,month,day)
    fn = fitsfn(year,month,day,hour,minute,wavelength)
    return "%s%s" % (prefix,fn)

def localFolder(year,month,day,hour,minute,wavelength):
    prefix = "%s/%d/%04d/%02d/%02d/"% (targetBase,wavelength,year,month,day)
    return prefix



DS = datetime.date(2011,1,1)
DE = datetime.date(2012,1,1)
dayDelta = datetime.timedelta(days=1)

hours = range(24)
minutes = range(0,60,6)
wavelengths = [131,1600,1700,171,193,211,304,335,94]
#wavelengths = [94]
#wavelengths = [131,1600,1700,171,193,211,304,335]

current = DS

remotePaths, localPaths, localFolders = [], [], []

while current < DE:
    year, month, day = current.year, current.month, current.day
    for hour in hours:
        for minute in minutes:
            for wl in wavelengths:
                lfn = localFn(year,month,day,hour,minute,wl)
                if not os.path.exists(lfn):
                    remotePaths.append(remoteFn(year,month,day,hour,minute,wl))
                    localPaths.append(lfn)
                    localFolders.append(localFolder(year,month,day,hour,minute,wl))

    current += dayDelta

print len(remotePaths), "to download"
localFolder = list(set(localFolders))

print len(localFolders), "folders to make"
for lfi,lf in enumerate(localFolders):
    print("%06d/%06d" % (lfi,len(localFolders)))
    if not os.path.exists(lf):
        os.system("mkdir -p %s" % lf)

print "About to download",len(remotePaths)


def handle(i):
    if i % 10 == 0:
        print "%06d/%06d" % (i,len(remotePaths))
    if not os.path.exists(localPaths[i]):
        os.system("wget -q %s -O %s" % (remotePaths[i],localPaths[i]))

toHandle = range(len(localFolders))

P = multiprocessing.Pool(4)
P.map(handle,toHandle)
    


