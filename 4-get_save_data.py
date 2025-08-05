#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 15:04:27 2025

1. open SBJ data from Camargo et al (2024)
https://zenodo.org/records/10814048

2. open ERA5 wind stress 

3. Open tg

4. Merge all 

5. band pass data also 

@author: carocamargo
"""

import pandas as pd
import xarray as xr
import numpy as np
from utils_work import cut_ds, find_wind_lm
from utils_work import noaaMetadata, butter_pass, save_dict
#%% save info
path_save = '/Users/carocamargo/Documents/data/floodSBJ/'
outname = 'df_sl_dic_bp_mhhw'


#%% get data
path_to_data = '/Users/carocamargo/Documents/data/SBJ/'
file = 'jet.csv'
df = pd.read_csv(path_to_data+file)
df['date']=pd.to_datetime(df['date']) # ensure its datetime
df.rename({"date":'time'},axis=1,inplace=True)
df_xr = df.set_index('time').to_xarray()


# wind stress
era5_file='/Users/carocamargo/Documents/data/ERA5/wind_stress_mean_2014-2023.nc'
dm = xr.open_dataset(era5_file)
era5_filemask='/Users/carocamargo/Documents/data/ERA5/mask.nc'
dmask = xr.open_dataset(era5_filemask)
mask = np.array(dmask['lsm'].isel(valid_time=0))
mask[mask>0.5]=np.nan
mask[np.isfinite(mask)]=1
dm['mask']=(('latitude','longitude'),mask)

# reduce file
bbox = [-76,-67,38,45]
dm = cut_ds(dm,bbox,lon_name='longitude',lat_name='latitude')

path_tg = '/Users/carocamargo/Documents/data/floodSBJ/'
file = 'tgs_complete_66const_IB.nc' # from compute_tides.py
ds = xr.open_dataset(path_tg+file)
# merge all
da = xr.merge([dm, ds, df_xr], join='inner')


#%% find wind at the TG
dfs = {}
stations = [station for station in np.array(da.station)]
# # check, if we have too many stations, then:
# stations = ['8447930', 
#         '8447386', '8452660', '8461490', '8465705', '8510560',
#         # # extend the region:
#         '8443970', # boston
#         '8447435',# chatam
#         '8449130', # nantucket
#         '8467150', # brigeport
#         '8516945',# kings point
#         '8531680' ,# sandy hook
        
#         # add more for ATW:
#         '8534720', # 'atlantic_city'
#         ]
# da = da.sel(station=stations)
for i, station_id in enumerate(stations):  
    print(station_id)
    #% %
    dx = da.sel(station=station_id)
    sl = np.array(dx['level'])
    tide = np.array(dx['tide_utide'])
    ib = np.array(dx['nib'])
    sl_cor = np.array(sl-tide-ib)

    Q = np.array(dx['Qy'])
    
    lat = np.array(dx['lat'])
    lon = np.array(dx['lon'])
    u_raw,v_raw = find_wind_lm(lat,lon,dx)
    
    df = pd.DataFrame({'time':dx.time,
                       'SL_cor':sl_cor,
                       'SL_tg':sl,
                       'tide':tide,
                       'ib':ib,
                       'Qy':Q,
                       'u_raw':u_raw,
                       'v_raw':v_raw,
                       })
    # df = df.fillna(method='ffill')  # or
    df = df.ffill()
    # band pass
    lowcut = 1.0 / (15.0 * 24)# , # 1/15 days in hours
    highcut = 1.0 / 24  # 1 day in hours
    df['SL_bp'] = butter_pass(df['SL_cor'],lowcut,highcut)
    df['Qy_bp' ]= butter_pass(df['Qy'], lowcut, highcut)
    df['u_bp'] = butter_pass(df['u_raw'], lowcut, highcut)
    df['v_bp'] = butter_pass(df['v_raw'], lowcut, highcut)
    

    dic = noaaMetadata(station_id)
    dfs[station_id] = {'df':df,
                       'dic':dic}
    
# save dicts of dfs
save_dict(dfs, path_save,outname)

    