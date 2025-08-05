#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 14:39:13 2025

Apply tides

@author: carocamargo
"""
import xarray as xr
import numpy as np
from utils_work import compute_tide


#%% inputs


path_tg = '/Users/carocamargo/Documents/data/floodSBJ/'
file = 'tgs_combined.nc'
ds = xr.open_dataset(path_tg+file)

# all but SA and SSA:
const66 = ['M2', 'K1', 'O1',  'N2', 'MSM', 'MM', 'P1', 'H1', 'S2',
       'H2', 'MF', 'MSF', 'MU2', 'S1', 'UPS1', 'L2', '2N2', 'M4', 'MSN2',
       'NU2', 'BET1', 'Q1', 'R2', 'ETA2', 'GAM2', 'PHI1', 'THE1', 'MKS2',
       'MO3', 'SO1', 'LDA2', 'SIG1', 'PSI1', 'ALP1', 'PI1', 'TAU1', '2Q1',
       'T2', 'NO1', 'J1', 'RHO1', 'OO1', 'SK3', 'M3', 'K2', '2MN6', 'OQ2',
       'EPS2', 'MS4', 'MN4', 'MK4', 'M6', 'SN4', 'S4', 'SO3', 'MK3',
       'SK4', '2MS6', 'M8', '2MK5', 'MSK6', '3MK7', '2SM6', 'CHI1',
       '2MK6', '2SK5']

#% % compute tide, asking for 66 constituents

tides = np.zeros((len(ds.time),len(ds.station)))
nib = np.zeros((len(ds.time),len(ds.station)))
dic = {}
verbose = True
names = [name for name in np.array(ds.name)]
for i in range(len(ds.station)):
    if verbose:
        print(names[i])
        print('Computing Tides')
    coef,tide = compute_tide(np.array(ds.time), np.array(ds.level[:,i]),
                             lat=np.array(ds.lat[i]),
                             constit=const66,
                             nodal=False,
                             verbose=verbose
                             
                             )
    tides[:,i] = np.array(tide.h)
    dic[names[i]]={'tide':tide,
                   'coef':coef}
    
#% % add to dataset
ds['tide_noaa']=ds['tide']
ds['tide_utide']=(('time','station'),tides)    
ds = ds.drop_vars(['tide'])


# ds['tide_dict']=(('station'),[dic[k]['tide'] for k in dic.keys()])
# ds['coef_dict']=(('station'),[dic[k]['coef'] for k in dic.keys()])
# ds['tide_dict'].attrs['description']='output of utide.reconstruct()'
# ds['coef_dict'].attrs['description']='output of utide.solve()'


ds['level'].attrs={'units':'m',
                    'long_name':'water level in relation to MSL',
                    'period':'MSL is in relation to 1983-2001'
                    }
ds['tide_noaa'].attrs={'units':'m',
                        'long_name':'NOAA Tide Prediction',
                        }

ds['tide_utide'].attrs={'units':'m',
                        'long_name':'Tide Prediction computed with Utide package',
                        'comment':'66 tidal constituents',
                        'const':const66
                        }

ds.attrs['Description']='Water levels (m) from tide gauges along New England Coast and Mid-Atlantic Bight.'
ds.attrs['Scripts']='compute_tides.py (uses data from download_TG.py'
#% % save
ds.to_netcdf(path_tg+'tgs_complete_66const.nc')


