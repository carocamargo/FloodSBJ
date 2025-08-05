#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 17:08:57 2025

Download and wrap tide gauge
@author: carocamargo
"""

from utils_work import noaa_Download_loop , wrap_csv

import xarray as xr
import numpy as np
    
    
#%%  requests info:
datum = 'MSL'
datum='MHHW'
fmt = 'csv'
interval='h'
path = '/Users/carocamargo/Documents/data/floodSBJ/'

start_y = 2014
end_y = 2023
variables=[ 'air_pressure',
           'hourly_height','predictions']

#% % loop over stations in New England:
stations = [
        8443970, # Boston, MA - Nov 16 1988
        8447386, # Fall River, MA - Jun 09, 1999
        
        8447930, # Woods Hole, MA - Nov 17, 1988
        8449130, # Nantucket Island, MA - 	Sep 18, 1990
        8452660, # Newport, RI - 	Sep 23, 1991
        8461490, # New London, CT - Oct 19, 2020
        8510560, # Montauk, NY - Sep 21, 1989
        8465705, # New Haven, CT
        8467150, # Bridgeport, CT
        8516945, # Kings Point, NY
        8531680, # Sandy Hook, NJ
        8534720, # Atlantic City, NJ
]

    
#% % loop oer variables and stations - download data, wrap to create netcdf
for var in variables:
    print(var)
    for station_id in stations:

        print('Downloading for {}\n'.format(station_id))
        # 1. Download
        noaa_Download_loop(station_id,var,start_y,end_y,
                            datum=datum,fmt=fmt,interval=interval,
                            path = path)
        print('')
    # 2. Wrap around all stations and create one netcdf for that variable and time period
    subpath = path+var+'/'
    wrap_csv(var = var,path = subpath,
             save = True,
             interval=interval,y0=start_y,y1=end_y)

    print('{} netcdf created'.format(var))

    
#% % 3. merge all netcdfs to create one file for all variables for this time period
# open tides
file = 'tgs_tides_combined_{}_{}.nc'.format(start_y,end_y)
dt = xr.open_dataset(path+file)

# open sea-level
file = 'tgs_sl_combined_{}_{}.nc'.format(start_y,end_y)
ds = xr.open_dataset(path+file)

# open atm pressure
file = 'tgs_atmp_combined_{}_{}.nc'.format(start_y,end_y)
da = xr.open_dataset(path+file)


# match times:
dt = dt.sel(time=slice(ds.time[0],ds.time[-1]))
da = da.sel(time=slice(ds.time[0],ds.time[-1]))

dt

ds['tide'] = (('time','station'),np.array(dt['tides']))
ds['atm_pressure'] = (('time','station'),np.array(da['atmp']))


file = 'tgs_combined.nc'
ds.to_netcdf(path+file)

