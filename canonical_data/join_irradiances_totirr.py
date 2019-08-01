'''
We join 14 channel irradiance with total megsa irradiance.
We ignore the seconds offset and just match to the minute.
In v2 we add a column to numpy array instead of csv
'''

import numpy as np
import pandas as pd
import sys

out_dir = '/fastdata2/data_alex/2011p4_new_fp16_30mn_totirr/'
c1 = pd.read_csv('/fastdata2/data_alex/2011p4_new_fp16_10mn/2011_2014.csv')
c2 = pd.read_csv('2011p4_totirr.csv')
a_irr = np.load('2011p4_totirr.npy')
a_init = np.load('/fastdata2/data_alex/2011p4_new_fp16_10mn/irradiance_10m.npy')

# Extend irradiance array
ext = np.full((a_init.shape[0], 1), -100.)
a_ext = np.hstack((a_init, ext))


# Indices to drop
drop_inds = []

for i in range(len(c1)):

    date = c1['EVE_TIME'].iloc[i].strip().split('T')[0]
    time = c1['EVE_TIME'].iloc[i].strip().split('T')[1]
    hour = time.split(':')[0]
    mn = time.split(':')[1]

    # We neglect the 10s difference between the two irradiance data sources.
    target = date+' '+hour+':'+mn+':'+'00'
    # Try to find a matching time for total irradiance
    ind = c2['row_name'].iloc[c2.index[c2['date_obs'] == target]]
    # Index in initial irradiance matrix
    ind2 = c1['EVE_ind'].iloc[i]

    # If index not found we drop the row
    if ind.shape[0] == 0:
        drop_inds.append(i)
    else:
        tot_irr_target = a_irr[ind]
        # Check for nan and negative values and drop that row
        if tot_irr_target != tot_irr_target or tot_irr_target <= 0.:
            drop_inds.append(i)
        # Add total irradiance to irradiance matrix
        else:
            a_ext[ind2,-1] = tot_irr_target

# We drop the rows where the total irradiance value is bad or where there is no time match
c1 = c1.drop(drop_inds)        

np.save(out_dir+'irradiance_30mn_14ptot.npy', a_ext)
c1.to_csv(out_dir+'irradiance_30mn_14ptot.csv',index=False)
