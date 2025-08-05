#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:32:54 2024


Script that opens data from NDBC stations 
(note the data has to be manually downloaded)
(https://www.ndbc.noaa.gov/)

plot available data 

and saves it in a single dictionary 

@author: carocamargo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#%%
def meta():
    metadata={'WDIR':{'long_name':'Wind direction',
                       'units':'degrees clockwise from N'},
              'WSPD':{'long_name':'Wind speed',
                      'units':'m/s'},
              'GST':{'long_name':'Gust Speed',
                     'units':'m/s'},
              'WVHT':{'long_name':'Significant wave height',
                     'units':'m'},
              'DPD':{'long_name':'Dominant wave period',
                     'units':'seconds'},
              'APD':{'long_name':'Average wave period',
                     'units':'seconds'},
              'MWD':{'long_name':'Dominant period direction',
                     'units':'degrees clockwise from N'},
              'PRES':{'long_name':'Sea level pressure',
                     'units':'hPa'},
              'ATMP':{'long_name':'Air Temperature',
                     'units':'C'},
              'WTMP':{'long_name':'Sea surface Temperature',
                     'units':'C'},
              'DEWP':{'long_name':'Dew Point',
                     'units':''},
              'VIS':{'long_name':'visibility',
                     'units':'nautical miles'},
              'PTDY':{'long_name':'Pressure tendency',
                     'units':'hPa'},
              'TIDE':{'long_name':'Water level above or below MLLW',
                     'units':'feet'}
              }
    stations_meta={'44008':{'long_name':'Nantucket Shoals'},
                  '44097':{'long_name':'Block Island'},
                  # '44085':{'long_name':'Buzzards Bay'}
                  }
    
    return stations_meta,metadata

def read_buoy(path='/Users/carocamargo/Documents/data/wave/',
              station_id ='44008',
              plot=False,
              ):
    # https://www.ndbc.noaa.gov/
    stations_meta,metadata = meta()
    files = [file for file in os.listdir(path) if not file.startswith('.') 
             and not file.startswith('readme')
             and not file.startswith('_')
             and file.startswith(station_id)
             ]
    flist = sorted([file for file in files if file.startswith(station_id)])
    dfs = []
    for file in flist:
        df = pd.read_csv(path+file,delim_whitespace=True, skiprows=[1])
        # Remove missing datas
        df.replace(999, np.nan, inplace=True)
        df.replace(99, np.nan, inplace=True)
        df.replace(9999, np.nan, inplace=True)
        df.replace('MM',np.nan, inplace=True)
        dfs.append(df)
        
    df = pd.concat(dfs, ignore_index=True)
    # y0 = df['#YY'].min()
    # yf = df['#YY'].max()
    df['Datetime'] = pd.to_datetime(df[['#YY', 'MM', 'DD', 'hh', 'mm']].astype(str).agg(' '.join, axis=1), format='%Y %m %d %H %M')
    
    if plot:
        varis = ['WDIR', 'WSPD', 
                 # 'GST', 
                 'WVHT', 'DPD',
               'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']
        plt.figure(figsize=(15,10))
        x = df['Datetime']
        for i,v in enumerate(varis):
            plt.subplot(4,3,int(i+1))
            # y = np.array(df[v])
            y = pd.to_numeric(df[v], errors='coerce').to_numpy()
            y[y==999]=np.nan
            # df[v].plot(ax=ax)
            # plt.plot(x,y,'-',linewidth=1)
            plt.scatter(x,y,s=5)
            if v=='WSPD':
                y = pd.to_numeric(df[v], errors='coerce').to_numpy()
                y[y==999]=np.nan
                plt.plot(x,y,label='GST',alpha=0.5)
                plt.scatter(x,y,s=5)
    
                plt.legend()
            plt.title(metadata[v]['long_name'])
            plt.ylabel(metadata[v]['units'])    
        plt.suptitle('Buoy station: {} (# {})'.format(stations_meta[station]['long_name'],station))
        plt.tight_layout()
        plt.show()
    
    return df.set_index('Datetime',inplace=True)

# def wrap_buoys()
#%%

# https://www.ndbc.noaa.gov/faq/measdes.shtml
#  Both Realtime and Historical files show times in UTC only.
#%%
dic = {}
cat = pd.read_excel('/Users/carocamargo/Documents/data/wave/stations_info.xlsx')
cat["lon_W_360"] = (360 - cat["lon_W"]) % 360
path = '/Users/carocamargo/Documents/data/wave/met/'
files = [file for file in os.listdir(path) if not file.startswith('.') 
         and not file.startswith('readme')
         and not file.startswith('_')
         # and file.startswith(station_id)
         ]
_,metadata= meta()
stations = np.unique([file.split('h')[0] for file in files])
stations = [str(row['id']) for i,row in cat.iterrows()]
stations.remove('BUZM3')
for station in stations:
    print(station)

    row = cat.loc[cat['id'].astype(str) == station.upper()]
    # print(row)
    #% %
    # station = stations[1]
    flist = sorted([file for file in files if file.startswith(station)])
    #% % 
    dfs = []
    for file in flist:
        df = pd.read_csv(path+file,delim_whitespace=True, skiprows=[1])
        # Remove missing datas
        df.replace(999, np.nan, inplace=True)
        df.replace(99, np.nan, inplace=True)
        df.replace(9999, np.nan, inplace=True)
        df.replace('MM',np.nan, inplace=True)
        dfs.append(df)
        
    df = pd.concat(dfs, ignore_index=True)
    y0 = df['#YY'].min()
    yf = df['#YY'].max()
    df['time'] = pd.to_datetime(df[['#YY', 'MM', 'DD', 'hh', 'mm']].astype(str).agg(' '.join, axis=1), format='%Y %m %d %H %M')
    varis = ['WDIR', 'WSPD', 
             # 'GST', 
             'WVHT', 'DPD',
           'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']
    x = df['time']
    dwave = df# [['WVHT','DPD','APD','MWD','WDIR']]
    dwave.set_index(df['time'],inplace=True)
    dic[station] = {'df':dwave,
                    'meta':{'long_name':row['long_name'][row.index[0]],
                            'lat':row['lat_N'][row.index[0]],
                            'lon':row['lon_W_360'][row.index[0]],
                            'depth':row['depth'][row.index[0]],
                            'id':row['id'][row.index[0]],
                            }
                    }
    #%%
    fig = plt.figure(figsize=(15,10))
    
    for i,v in enumerate(varis):
        ax = plt.subplot(4,3,int(i+1))
        # y = np.array(df[v])
        y = pd.to_numeric(df[v], errors='coerce').to_numpy()
        y[y==999]=np.nan
        # df[v].plot(ax=ax)
        # plt.plot(x,y,'-',linewidth=1)
        plt.scatter(x,y,s=5)
        if v=='WSPD':
            y = pd.to_numeric(df[v], errors='coerce').to_numpy()
            y[y==999]=np.nan
            plt.plot(x,y,label='GST',alpha=0.5)
            plt.scatter(x,y,s=5)

            plt.legend()
        plt.title(metadata[v]['long_name'])
        # plt.ylabel(metadata[v]['units'])    
    plt.suptitle('Buoy station: {} (# {})'.format(row['long_name'][row.index[0]],station))
    plt.tight_layout()
    plt.show()
    
    # df.to_csv(path+'_{}_{}-{}.txt'.format(station,y0,yf), sep='\t', index=False)
    # df[varis].plot()
#%% 
import pickle
def save_dict(data,path,filename):
    
    outname = "{}{}.pkl".format(path,filename)
    with open(outname, "wb") as f:
        pickle.dump(data, f)
#%% save dict
path_save = '/Users/carocamargo/Documents/data/floodSBJ/'
filename = 'waves_complete'
save_dict(dic,path_save,filename)