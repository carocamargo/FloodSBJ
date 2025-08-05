#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:50:49 2025

A. Get data
B. Prep it
C. Run analysis

@author: carocamargo
"""

### GET DATA
# 1. Download TG from NOAA
download_tidegauge() # needs internet
# will save tg_combined.nc

# 2. Remove tides (uses Utide)
compute_tides()

# 3. Apply IB correction (in-situ barom pres)
apply_IB()

# 4. Get SBJ (link from GRL), wind stress, band-pass data and save dfs
save_era5()
get_save_data()


### ANALYSIS 
# 8. Analysis: bootstrap, regress, stats, and save it for plotting
analysis()

# 9. Save extra data for plotting
save_waves()
save_noaa_thresholds()


#%% Libraries &  user defined functions


def download_tidegauge():
    #
    from utils_work import noaa_Download_loop , wrap_csv

    import xarray as xr
    import numpy as np
    ##
    datum = 'MSL'
    datum='MHHW'
    fmt = 'csv'
    interval='h'
    path = '/Users/carocamargo/Documents/data/sealevel/NOAA/MHHW/'

    start_y = 2010
    end_y = 2023
    variables=['air_pressure','hourly_height','predictions']

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
            516945, # Kings Point, NY
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
    
    return

def compute_tides():
    
    import xarray as xr
    import numpy as np
    from utils_work import compute_tide


    #% % inputs
    path_tg = '/Users/carocamargo/Documents/data/sealevel/NOAA/MHHW/'
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
    
    return

#%% IB
def apply_IB():
    
    from utils_work import open_slp, reg_composite
    from utils_work import complete_ds, cor_distance, add_new_ib
    
    import xarray as xr
    import numpy as np
    
    
    #% % inputs
    path_tg = '/Users/carocamargo/Documents/data/sealevel/NOAA/MHHW/'
    file = 'tgs_complete_66const' # from compute_tides.py
    
    
    #% % run
    
    ds = xr.open_dataset(path_tg=path_tg,
                         file=file+'.nc')
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
    
    return

def get_save_data():
    import pandas as pd
    import xarray as xr
    import numpy as np
    from utils_work import cut_ds, find_wind_lm
    from utils_work import noaaMetadata, butter_pass, save_dict
    #% % save info
    path_save = '/Users/carocamargo/Documents/data/sealevel/'
    outname = 'df_sl_dic_bp_mhhw'


    #% % get data
    path_to_data = '/Users/carocamargo/Documents/data/SBJ/'
    file = 'jet.csv'
    df = pd.read_csv(path_to_data+file)
    df['date']=pd.to_datetime(df['date']) # ensure its datetime
    df.rename({"date":'time'},axis=1,inplace=True)
    df_xr = df.set_index('time').to_xarray()

    era5_file='/Users/carocamargo/Documents/data/ERA5/wind_single_2013-2024.nc'
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

    path_tg = '/Users/carocamargo/Documents/data/sealevel/NOAA/MHHW/'
    file = 'tgs_complete_66const_IB.nc' # from compute_tides.py
    file='tgs_complete_66const_cor.nc' # new files are now IB # Change this after rerunning it!!!!!!!!!!
    ds = xr.open_dataset(path_tg+file)
    # merge all
    da = xr.merge([dm, ds, df_xr], join='inner')


    # %% find wind at the TG
    dfs = {}
    stations = [station for station in np.array(da.station)]
    # check, if we have too many stations, then:
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
            ]
    da = da.sel(station=stations)
    for i, station_id in enumerate(stations):  
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
        
        # band pass
        lowcut = 1.0 / (15.0 * 24), # 1/15 days in hours
        highcut = 1.0 / 24  # 1 day in hours
        SS = butter_pass(sl_cor,lowcut,highcut)
        Qbp = butter_pass(Q, lowcut, highcut)
        u_bp = butter_pass(u_raw, lowcut, highcut)
        v_bp = butter_pass(v_raw, lowcut, highcut)
        
        # make dataframe
        df = pd.DataFrame({'time':dx.time,
                           'SL_cor':sl_cor,
                           'SL_tg':sl,
                           'tide':tide,
                           'ib':ib,
                           'SL_bp':SS, # 
                           'Q_bp':Qbp,
                           'Qy':Q,
                           'u_raw':u_raw,
                           'v_raw':v_raw,
                           'u_bp':u_bp,
                           'v_bp':v_bp
                           })
        dic = noaaMetadata(station_id)
        dfs[station_id] = {'df':df,
                           'dic':dic}
        
    # save dicts of dfs
    save_dict(dfs, outname,path_save)
    
