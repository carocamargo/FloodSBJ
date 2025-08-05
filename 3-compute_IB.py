#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 14:50:18 2025

Apply IB
@author: carocamargo
"""

from utils_work import open_slp, reg_composite
from utils_work import complete_ds, cor_distance, add_new_ib

import xarray as xr
import numpy as np


#%% inputs
path_tg = '/Users/carocamargo/Documents/data/floodSBJ/'
file = 'tgs_complete_66const' # from compute_tides.py


#%% run

ds = xr.open_dataset(path_tg+file+'.nc')
ds = ds.where(ds.name!='Windmill Point',drop=True)
# names = np.array(ds.name)
# duration_wl, duration_atm, total_len, wl_len, atm_len = get_duration(ds)
dslp = open_slp(ds_tg=ds,time_slice=True)
if len(ds.time)> len(dslp.time):
    ds = ds.sel(time=slice(dslp.time[0],dslp.time[-1]))
    if len(ds.time)== len(dslp.time):
        print('Files matched.')
mpa = np.array(dslp.mean_pa*100)

# nib = np.zeros((len(ds.time),len(ds.station)))
# lib = np.zeros((len(ds.time),len(ds.station)))
# dic = {}

# for i in range(len(ds.station)):
#     nib[:,i] = compute_ib(np.array(ds.atm_pressure[:,i])*100,mpa) # pressures is in mbar = 100 kg/ms2
#     lib[:,i] = compute_ib_local(np.array(ds.atm_pressure[:,i])*100) # pressures is in mbar = 100 kg/ms2

# cor_st = stations_cor(ds)
    

cor, p, distances = cor_distance(ds, # takes a while...
                                  method='pearson',plot=False, 
                                  save=False,
                                
                                  )
# takes a while...
atm_completed, regional_average = reg_composite(ds,distances,
                                                plot1=False,plot2=True,
                                                info=False)

ds = complete_ds(ds, atm_completed,mpa)

ds = add_new_ib(ds)
#% % save updated ds
ds.attrs['Scripts']='compute_ib (uses data from compute_tide and download_TG)'
ds.to_netcdf(path_tg+file+'_IB.nc')