#!/usr/bin/python
#
#Check if any AIA are missing and make a csv with file names, used for inference of unexistent EVE.

import os
import pdb
import numpy as np
import datetime
import pandas as pd

def fitsfn(year,month,day,hour,minute,wavelength):
    fn = "AIA%04d%02d%02d_" % (year,month,day)
    fn += "%02d%02d_" % (hour,minute)
    fn += "%04d.npz" % (wavelength)
    return fn

def localFn(year,month,day,hour,minute,wavelength):
    prefix = "%04d/%02d/%02d/"% (wavelength,month,day)
    fn = fitsfn(year,month,day,hour,minute,wavelength)
    return "%s%s" % (prefix,fn)

if __name__ == "__main__":

    #Where the 6m data should be
#    target = "/home/szenicer/FDL/sw2018-irradiance/test"
    target = '/home/alexszn/data/2015/'
    #Where the AIA data is
    AIA_Base = "/w/SDO/2015/AIA/"

    if not os.path.exists(target):
        os.mkdir(target)

    minutes = range(0,60,6)



    for y in range(2015,2016):
        startDate = datetime.date(y,1,1)
        endDate = datetime.date(y+1,1,1)
        dayDelta = datetime.timedelta(days=1)

        hours = range(24)

        #get the dates
        wavelengths = [131,1600,1700,171,193,211,304,335,94]
        current = startDate


        fh = open("%s/%04d.csv" % (target,y),"w")
        fh.write("AIA_TIME,")
        for wl in wavelengths:
            fh.write("AIA_%d," % wl)
        fh.write("\n") 
        aiaMissing, present = 0, 0
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

                    dt = datetime.datetime(year, month, day, hour, minute, 0)
                    fh.write("%s," % (str(dt)))
                    for fn in fns:
                        fh.write("%s," % fn)
                
                    fh.write("\n") 
                    present += 1


            current += dayDelta

        fh.close()

        print("Missing: %d AIA\nPresent: %d" % (aiaMissing,present))



