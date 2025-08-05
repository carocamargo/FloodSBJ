#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:29:38 2025

@author: carocamargo
"""

import numpy as np
import pandas as pd
import pickle
import xarray as xr
from scipy.stats import pearsonr

from functions_work import find_closest_cell, open_ERA5, insitu_wind, noaaFlood, noaaMetadata, myOLS,  spectral_rel, butter_pass

from functions_work import get_data_tg, find_wind, regress_wind, organize_df_SL, open_era_SBJ, bp_winds, save_dict

def run(path_to_data = '/Users/carocamargo/Documents/data/timeseries_66_ft/',
        path_save = '/Users/carocamargo/Documents/data/sealevel/',
        era5_file = '/Users/carocamargo/Documents/data/ERA5/wind_single_2013-2024.nc',
        outname = 'df_sl_dic_bp_mhhw'
        ):
    
    # outname = 'df_sl_dic' # file name without band passing winds
    # outname = 'df_sl_dic_bp' # file name with band passing winds
    # outname = 'df_sl_dic_bp_mhhw' # file name with band passing winds, SL in MHHW
    
    dfs = {}
    stations = ['8447930', 
            '8447386', '8452660', '8461490', '8465705', '8510560',
            # # extend the region:
            '8443970', # boston
            '8447435',# chatam
            '8449130', # nantucket
            '8467150', # brigeport
            '8516945',# kings point
            '8531680' ,# sandy hook
            
            # add more for ATW:
            '8534720', # 'atlantic_city'
            '8536110',# 'cape_may'
            '8557380',# 'lewes'
            '8631044',#'wachapreague'
            '8632200',#'kipopeke'
            '8638901',#'chesapeake'
            '8651370',#'duck'

            ]
        
    dx = open_era_SBJ(era5_file)
    for station_id in stations:
        print(station_id)
        df, dic = get_data_tg(path_to_data=path_to_data,
                              station_id=station_id)
        if len(df)!= len(dx.time):
            print('Time doesnt match')
            dx = dx.sel(time=slice(df.time[0], df.time[len(df)-1]))
        # thresholds = dic['flood_thresholds']
        
        lat_tg = float(dic['lat']); lon_tg = float(dic['lon'])
        # without band passing winds:
        # df['u'], df['v'] = find_wind(lat_tg,lon_tg,dx)
        # band_passing winds:
        df['u_raw'], df['v_raw'] = find_wind(lat_tg,lon_tg,dx)
        # u_raw, v_raw = find_wind(lat_tg,lon_tg,dx)
               
        df = bp_winds(df) # df['u'] and df['v'] have been band-passed
        
        df['SL_wind'], df['SL_nw'], stats  = regress_wind(df,var_name='SL')
        
        df_sl=organize_df_SL(df)
        
        dfs[station_id] = {'df':df,
                           'dic':dic,
                           'wind_stats':stats,
                           'df_sl':df_sl,
                           }
    
    save_dict(dfs, outname,path_save)
# #%% test
# path_to_data = '/Users/carocamargo/Documents/data/timeseries_MHHW/'
# path_save = '/Users/carocamargo/Documents/data/sealevel/'
# era5_file='/Users/carocamargo/Documents/data/ERA5/wind_stress_accum_2014-2023.nc'
# outname = 'df_sl_dic_bp_mhhw_stress'
# station_id = '8447930'

#%% run
# uses data from GRL 2024 paper
run(path_to_data = '/Users/carocamargo/Documents/data/timeseries_MHHW/',
        path_save = '/Users/carocamargo/Documents/data/sealevel/',
        era5_file='/Users/carocamargo/Documents/data/ERA5/wind_stress_mean_2014-2023.nc',
        outname = 'df_sl_dic_bp_mhhw_stress_mean'
        )