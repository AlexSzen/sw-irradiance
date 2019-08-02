'''
concatenate different years into one
'''

import numpy as np
import pandas as pd
import argparse
import os

def parse_args():

    parser=argparse.ArgumentParser()
    parser.add_argument('--totirr_root', dest='totirr_root', required=True)
    parser.add_argument('--target', dest='target', required=True)
    args = parser.parse_args()
    return args


args = parse_args()
data_root = args.totirr_root
target = args.target

c1 = pd.read_csv(os.path.join(data_root,'2011_eve_megsA_mean_irradiance.csv')) 
c2 = pd.read_csv(os.path.join(data_root,'2012_eve_megsA_mean_irradiance.csv'))
c3 = pd.read_csv(os.path.join(data_root,'2013_eve_megsA_mean_irradiance.csv')) 
c4 = pd.read_csv(os.path.join(data_root,'2014_eve_megsA_mean_irradiance.csv'))

a1 = np.load(os.path.join(data_root,'2011_eve_megsA_mean_irradiance.npy'))
a2 = np.load(os.path.join(data_root,'2012_eve_megsA_mean_irradiance.npy'))
a3 = np.load(os.path.join(data_root,'2013_eve_megsA_mean_irradiance.npy'))
a4 = np.load(os.path.join(data_root,'2014_eve_megsA_mean_irradiance.npy'))

# concatenate numpy array
tot_a = np.concatenate((a1, a2, a3, a4), axis = 0) 

# offset indices of csvs before concatenating
c2['row_name'] = c2['row_name'] + c1.shape[0]
c3['row_name'] = c3['row_name'] + c1.shape[0] + c2.shape[0]
c4['row_name'] = c4['row_name'] + c1.shape[0] + c2.shape[0] + c3.shape[0]

# concatenate csv files
tot_c = pd.concat((c1, c2, c3, c4), axis = 0)

# dump arrays
np.save(os.path.join(target,'2011p4_totirr.npy'), tot_a)
tot_c.to_csv(os.path.join(target,'2011p4_totirr.csv'))
