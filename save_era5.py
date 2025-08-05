#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:31:47 2025

Get downloaded hourly data from eRA5 from 2013-2023, and save it in a single netcdf. 

To download ERA5 data, go to: 
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download
and choose 'Reanalysis', and choose wind stress variable. 

@author: carocamargo
"""
import xarray as xr
import numpy as np
#%% stress mean
'''
Mean [north]eastward turbulent surface stress (N m-2)
Air flowing over a surface exerts a stress (drag) that transfers momentum to the surface and slows the wind. 
This parameter is the component of the mean surface stress in an [northward]eastward direction, 
associated with turbulent eddies near the surface and turbulent orographic form drag. 
It is calculated by the ECMWF Integrated Forecasting System's turbulent diffusion and turbulent orographic form drag schemes. 
The turbulent eddies near the surface are related to the roughness of the surface. 
The turbulent orographic form drag is the stress due to the valleys, hills and mountains on horizontal scales below 5km, 
which are specified from land surface data at about 1 km resolution. 
(The stress associated with orographic features with horizontal scales between 5 km and 
 the model grid-scale is accounted for by the sub-grid orographic scheme.) 
Positive (negative) values indicate stress on the surface of the Earth in an [northward (southward)]eastward (westward) direction. 
This parameter is a mean over a particular time period (the processing period) which depends on the data extracted. 
For the reanalysis, the processing period is over the 1 hour ending at the validity date and time.
For the ensemble members, ensemble mean and ensemble spread, the processing period is over the 3 hours ending at the validity date and time.
'''
path = '/Users/carocamargo/Documents/data/ERA5/stress/avg/'
outname = 'wind_stress_mean_2014-2023'
dm = xr.open_mfdataset(path+'*.nc')
u_name = 'metss'
v_name  = 'mntss'
dm = dm.rename({'valid_time':'time',
                # 'latitude':'lat',
                # 'longitude':'lon'
                })
# print(dm)
ds = xr.Dataset(data_vars={'u':(('time','latitude','longitude'),np.array(dm[u_name])),
                           'v':(('time','latitude','longitude'),np.array(dm[v_name])),
                           },
                coords={'time':np.array(dm.time),
                        'latitude':np.array(dm.latitude),
                        'longitude':np.array(dm.longitude)})
ds.u.attrs = dm[u_name].attrs
ds.v.attrs = dm[v_name].attrs
ds.attrs['script']='save_era5.py'
ds.to_netcdf('/Users/carocamargo/Documents/data/ERA5/'+outname+'.nc')