#!/usr/bin/python
#
#Join AIA and eve

import os
import pdb
import numpy as np
import datetime
import pandas as pd
import argparse

def fitsfn(year,month,day,hour,minute,wavelength):
    fn = "AIA%04d%02d%02d_" % (year,month,day)
    fn += "%02d%02d_" % (hour,minute)
    fn += "%04d.npz" % (wavelength)
    return fn

def localFn(year,month,day,hour,minute,wavelength):
    prefix = "%d/%04d/%02d/%02d/"% (wavelength,year,month,day)
    fn = fitsfn(year,month,day,hour,minute,wavelength)
    return "%s%s" % (prefix,fn)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eve_root", dest="eve_root", required=True)
    parser.add_argument("--aia_root", dest="aia_root", required=True)
    parser.add_argument("--target", dest="target", required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    #Where the EVE data is
    src = args.eve_root
    #Where the 6m data should be
    target = args.target
    #Where the AIA data is
    AIA_Base = args.aia_root

    if not os.path.exists(target):
        os.mkdir(target)

    ### Edit to get the necessary cadence.
    minutes = range(0,60,10)

    #Nx1, Nx#EVE, Nx1
    dates = np.load(src+"/iso.npy")
    Y = np.load(src+"/irradiance.npy")
    reind = -np.ones((Y.shape[0],),dtype=np.uint64)

    #date time to eve index; we'll then reindex
    eveDateToInd = {}
    #round off error; use this to figure out whether to accept or reject
    #eve measurements that fall into the same bin
    eveDateToIndResolution = {}

    for i in range(len(dates)):
        if i % 10000 == 0:
            print("%d/%d = %.2f") % (i,len(dates),float(i)/len(dates)*100)

        d = dates[i][:-1].replace("T"," ")
        dtOrig = datetime.datetime.strptime(d,"%Y-%m-%d %H:%M:%S")

        ts = pd.Timestamp(d).round('min')
        dt = datetime.datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")

        #get the round-off error
        roundError = max(dt-dtOrig,dtOrig-dt)

        #don't bother if it's not the right minute
        if dt.minute not in minutes:
            continue

        if dt not in eveDateToInd or roundError < eveDateToIndResolution[dt]:
            eveDateToInd[dt] = i
            eveDateToIndResolution[dt] = roundError

    #these are the possible eve measurements we want to keep (1...K)
    inds = list(eveDateToInd.values())
    inds.sort()

    #these are Kx1 and Kx#EVE respectively
    dateReind = dates[inds]
    YReind = Y[inds]

    #this just maps any old eve index to the new ones
    reind[inds] = range(len(inds))

    #save them
    np.save(target+"/iso_10m.npy",dateReind)
    np.save(target+"/irradiance_10m.npy",YReind)

    #make a csv for every year

    for y in range(2011,2015):
        startDate = datetime.date(y,1,1)
        endDate = datetime.date(y+1,1,1)
        dayDelta = datetime.timedelta(days=1)

        hours = range(24)

        #get the dates
        wavelengths = [131,1600,1700,171,193,211,304,335,94]
        current = startDate


        fh = open("%s/%04d.csv" % (target,y),"w")
        fh.write("AIA_TIME,EVE_TIME,")
        for wl in wavelengths:
            fh.write("AIA_%d," % wl)
        fh.write("EVE_ind\n")

        aiaMissing, eveMissing, present = 0, 0, 0
        while current < endDate:
            year, month, day = current.year, current.month, current.day
            if day == 1:
                print (year,month,day)

            for hour in hours:
                for minute in minutes:

                    #assume they're all present
                    allPresent = True
                    fns = []
                    for wl in wavelengths:
                        localFilename = localFn(year,month,day,hour,minute,wl)

                        fullFilename = "%s/%s" % (AIA_Base, localFilename)

                        fns.append(localFilename)
                        if not os.path.exists(fullFilename):
                            allPresent = False
                            break
                    
                    if not allPresent:
                        aiaMissing += 1
                        continue

                    #look up the time
                    dt = datetime.datetime(year, month, day, hour, minute, 0)

                    if dt not in eveDateToInd:
                        eveMissing += 1
                        continue

                    #write the csv out
                    newEveIndex = reind[eveDateToInd[dt]]
                    eveDate = dates[eveDateToInd[dt]]

                    fh.write("%s,%s," % (str(dt),str(eveDate)))
                    for fn in fns:
                        fh.write("%s," % fn)
                    fh.write("%d\n" % newEveIndex)
                    present += 1


            current += dayDelta

        fh.close()

        print("Missing: %d AIA %d EVE\nPresent: %d")% (aiaMissing,eveMissing,present)



