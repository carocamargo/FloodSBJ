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

# water level, atm pressure and tidal predictions at tide gauges
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

# tides
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

    return

    



def run_analysis(dfs,stations,
        fit='ridge',
        testrandom=True,hilb=False,onlyminor=False,
        score='var',
        alpha=0.3, boot_method='rand',norm=True,
        block_size=30,NB=1000,
        ):
    # score = 'var'
    # alpha = 0.3
    # boot_method = 'rand'
    # block_size=30
    # norm=True
    # NB  = 1000 # number of bootstraps replicas
    NID = len(stations) # number of stations
    NT = len(dfs[stations[0]]['df']) # length of time series (full - hourly)
    level=95
    out = {}
    table = np.zeros((len(stations),4*4))
    #% % get data of one station
    # istation =2
    # station = stations[istation]
    #% %
    print('\n Start Loop \n')
    for istation, station in enumerate(stations):
        #% %
        print(station)
        df = dfs[station]['df']
        # df['SL_hf']=np.array(df['SL']-df['SL_bp'])
        ref = thresholds[station]['nos_minor']
        ref_mod = thresholds[station]['nos_moderate']
        
        name = dfs[station]['dic']['name']
        if 'time' in df.columns:
            df['time']=pd.to_datetime(df['time'])
        else:
            df['time'] = pd.to_datetime(df.index)
        # set index
        df.set_index('time',inplace=True)
        # resample to daily max
        df = df.resample('1D').apply(lambda x: x.loc[x['SL_tg'].idxmax()])
        ND = len(df) # length of time series (daily)
        # flood_days = dfs_sl[station]['flood_days'].index
        
        if onlyminor:
            flood_days = df[(df['SL_tg'] > ref) & (df['SL_tg'] < ref_mod)].index
        else:
            flood_days = df[df['SL_tg']>ref].index
        
        #% % get data we want
        x_col = ['Qy_bp','u_bp','v_bp']
        # compute hilbert transform of each parameter
        if hilb:
            for var in ['Qy_bp','u_bp','v_bp']:
                df['{}_h'.format(var)] = np.imag(hilbert(np.array(df[var])))
                # df['{}_h'.format(var)] = np.abs(hilbert(np.array(df[var])))
                
                x_col.append('{}_h'.format(var))
            
        # 
        X = [np.array(df[col]) for col in x_col] # list
        x = np.array(X).T # array
        Y = np.array(df['SL_bp'])
        NP = len(x_col)
        if norm:
            ybar = np.nanmean(Y)
            Y = np.array(Y - ybar) # centralize Y
            xbar = np.nanmean(x,axis=0) # one mean per parameters
            x = np.array(x - xbar) # centralize X - remove mean from each column
            xstd = np.sqrt(ND-1)*np.std(x,axis=0);
            x = np.array(x/xstd) # normalize X
        
        acc_idx = np.isfinite(Y)
        if fit=='linear':
            it, s = lin_fit(x[acc_idx],Y[acc_idx])
        elif fit=='ridge':
            it, s = ridge_fit(x[acc_idx],Y[acc_idx],alpha)
        else:
            print('Fit not recognized.')
        
        # scale back units
        if norm:
            # coefs
            s = np.array(s/xstd)
            # it = it - np.sum(s * xbar)  # Adjust intercept to align with centralized predictors
            # variables 
            Y = np.array(df['SL_bp'])
            x = np.array(X).T # array

        preds = {}
        predsh = {}
        r2s = {}
        
        ## calculate preds for each param
        # Overall
        y_pred = np.array(x @ s)
        preds['overall'] = y_pred   
        
        if score=='var':
            r2s['overall'] = r2_var(Y, y_pred)
        elif score=='r2':
            r2s['overall'] = r2_score(Y, y_pred)
        else:
            print(f"Score method '{score}' is not recognized. Use 'var' or 'r2'.")
        # print(r2s)
        #% %
        for i, col in enumerate(x_col):
            y_pred = np.array(x[:,i] * s[i])
            preds[col] = y_pred
            if score=='var':
                r2s[col] = r2_var(Y, preds[col])
            else:
                r2s[col] = r2_score(Y, preds[col])
            # print(r2s[col])
        
       
        # re do scores for flood days
        r2s_flood = {}
        params = ['overall'] + x_col
        da = pd.DataFrame(preds,columns=params)
        da['time'] = df.index
        da['y'] = np.array(Y)
        da.set_index('time',inplace=True)
        daf = da.loc[flood_days]
        n_flood = len(flood_days)
        for ip,p in enumerate(params):
            y_pred = np.array(daf[p])
            r2s_flood[p] = r2_var(np.array(daf['y']),y_pred)
        #% %
        # ranomly select the same aom=mou nt of flood days and compute R2
        # Number of times you want to sample
        if testrandom:
            num_samples = 10000 
            r2_dist = np.zeros((len(params),num_samples))
            ii = np.arange(0,ND)
            # Randomly select 30 numbers multiple times
            samples = [np.random.choice(ii, n_flood, replace=False) for _ in range(num_samples)]
            for k, sample in enumerate(samples):
                daf = da.iloc[sample]
                for ip,p in enumerate(params):
                    y_pred = np.array(daf[p])
                    r2_dist[ip,k]  = r2_var(np.array(daf['y']),y_pred)
            # #% %
            # for ip,p in enumerate(params):
            #     plt.subplot(2,2,ip+1)
            #     plt.hist(r2_dist[ip,:],color='lightblue');
            #     plt.axvline(np.mean(r2_dist[ip,:]),c='blue',linestyle='-')
            #     plt.axvline(r2s_flood[p],c='black',linestyle='--')
            #     plt.axvline(r2s[p],c='red',linestyle='--')
                
                
            #     plt.title(p)
            # plt.tight_layout()
            # plt.suptitle(name)
            # plt.show()
            #% %
            table[istation,0] = r2s[params[0]]
            table[istation,1] = r2s_flood[params[0]]
            table[istation,2] = np.percentile(r2_dist[0,:] , [(100-level)/2, (100+level)/2])[0]
            table[istation,3] = np.percentile(r2_dist[0,:] , [(100-level)/2, (100+level)/2])[1]

            table[istation,4] = r2s[params[1]]
            table[istation,5] = r2s_flood[params[1]]
            table[istation,6] = np.percentile(r2_dist[1,:] , [(100-level)/2, (100+level)/2])[0]
            table[istation,7] = np.percentile(r2_dist[1,:] , [(100-level)/2, (100+level)/2])[1]
            
            table[istation,8] = r2s[params[2]]
            table[istation,9] = r2s_flood[params[2]]
            table[istation,10] = np.percentile(r2_dist[2,:] , [(100-level)/2, (100+level)/2])[0]
            table[istation,11] = np.percentile(r2_dist[2,:] , [(100-level)/2, (100+level)/2])[1]
            
            table[istation,12] = r2s[params[3]]
            table[istation,13] = r2s_flood[params[3]]
            table[istation,14] = np.percentile(r2_dist[3,:] , [(100-level)/2, (100+level)/2])[0]
            table[istation,15] = np.percentile(r2_dist[3,:] , [(100-level)/2, (100+level)/2])[1]
            
            
            np.percentile(r2_dist[ip,k] , [(100-level)/2, (100+level)/2])  # CI
            
            # dt = pd.DataFrame(np.round(table,2),index=names)
            # dt.to_clipboard()
        
        #% %
        # now we bootstrap the samples
        # we also want to bootstrap sl_raw
        S = np.array(df['SL_tg'])
        XS = X + [S]
        X_boot, Y_boot = bootstrap(XS, Y, 
                                    NB=NB,
                                    method=boot_method,
                                    block_size=block_size
                                    )
 
        
        slopes = np.zeros((NB,NP))
        r2s_bt = np.zeros((NB,NP+1))
        r2s_flood_bt = np.zeros((NB,NP+1))
        
        ii = np.arange(0,ND)
        ib=0
        for ib in range(NB):
            # bootstrap here
            np.random.shuffle(ii)
            # xb = np.array (x[ii,:])
            # yb = np.array(Y[ii])
            # get bootstrap sample
            xb = np.array(X_boot[0:len(X),ib,:].T)
            yb = np.array(Y_boot[ib])
            # normalize
            if norm:
                ybar = np.nanmean(yb)
                yb = np.array(yb - ybar) # centralize Y
                xbar = np.nanmean(xb,axis=0) # one mean per parameters
                xb = np.array(xb - xbar) # centralize X - remove mean from each column
                xstd = np.sqrt(ND-1)*np.std(xb,axis=0);
                xb = np.array(xb/xstd) # normalize X
            # fit
            if fit=='linear':
                _,slopes[ib,:] = lin_fit(xb, yb) 
            else:
                _,slopes[ib,:] = ridge_fit(xb,yb,alpha=alpha)                                
            # scale back units
            if norm:
                # coefs
                slopes[ib,:] = np.array(slopes[ib,:]/xstd)
                # xb = np.array (x[ii,:])
                # yb = np.array(Y[ii])
                xb = np.array(X_boot[0:len(X),ib,:].T)

                yb = np.array(Y_boot[ib])
                
            # compute prediction and r2 scores
            # overall
            y_pred =  np.array(xb @ slopes[ib,:])
            da = pd.DataFrame({'overall':y_pred})
            if score=='var':
                r2s_bt[ib,0] = r2_var(yb, y_pred)
            else:
                r2s_bt[ib,0] = r2_score(yb, y_pred)
            # each param
            for i, col in enumerate(x_col):
                y_pred = np.array(xb[:,i] * slopes[ib,i])
                da[col] = np.array(y_pred)
                if score=='var':
                    r2s_bt[ib,i+1] = r2_var(yb, y_pred)
                else:
                    r2s_bt[ib,i+1] = r2_score(yb, y_pred)
            
            params = [col for col in da.columns]
            # Add time and y to the dataframe
            da['time'] = df.index
            da['y'] = np.array(yb)
            da['SL_tg'] = np.array(X_boot[-1,ib,:])
            da.set_index('time',inplace=True)
            # select days of flood
            # da = da.loc[flood_days]
            if onlyminor:
                da = da[(da['SL_tg']>ref) & (da['SL_tg']<ref_mod)]
            else:
                da = da[da['SL_tg']>ref]
            # 
            # compute stats for days of flood
            for ip,p in enumerate(params):
                y_pred = np.array(da[p])
                if score=='var':
                    r2s_flood_bt[ib,ip] = r2_var(np.array(da['y']),y_pred)
                else:
                    r2s_flood_bt[ib,ip] = r2_score(np.array(da['y']),y_pred)
                    
        # get statistics of slope/regression coefficient
        stats_slope = get_stats(slopes,level)
        stats_r2 = get_stats(r2s_bt,level)
        stats_r2_flood = get_stats(r2s_flood_bt,level)
        
        # get all the results for this station
        #% %
        
            
        orig = {'X':x,
                'Y':Y,
                'slope':s,
                'preds':preds,
                'r2s':r2s,
                'r2s_floods':r2s_flood,
                'n_flood':len(flood_days),
                }
        if NB>=100:
            boot = {
                    # 'X':X_boot,
                    # 'Y':Y_boot,
                    'slope':slopes,
                    'r2s':r2s_bt,
                    'stats_slope':stats_slope,
                    'stats_r2':stats_r2,
                    'r2s_flood':r2s_flood_bt,
                    'stats_r2_flood':stats_r2_flood,
                         }
        else:
            boot = {
                    'stats_slope':stats_slope,
                    'stats_r2':stats_r2,
                    'r2s_flood':r2s_flood_bt,
                    'stats_r2_flood':stats_r2_flood,
                         }
        out[station]={'original':orig,
                      'bootstrap':boot
                      }
        
        #% %
        print('coef:')
        for i,p in enumerate(['Qy_bp', 'u_bp', 'v_bp']):
            print('{}: {:.2f}, {:.2f} [{:.2f} {:.2f}] '.format(p, s[i],
                                                               stats_slope['med'][i],
                                                   stats_slope['cis'][i][0],
                                                   stats_slope['cis'][i][1],
                                                   ))
        print('R2:')
        for i,p in enumerate(['overall','Qy_bp', 'u_bp', 'v_bp']):
            print('{}: {:.2f}, {:.2f} [{:.2f} {:.2f}] '.format(p, r2s[p],
                                                               stats_r2['med'][i],
                                                               stats_r2['cis'][i][0],
                                                               stats_r2['cis'][i][1],
                                                   ))
            #% %
        print('R2 flood:')
        for i,p in enumerate(['overall','Qy_bp', 'u_bp', 'v_bp']):
            print('{}: {:.2f}, {:.2f} [{:.2f} {:.2f}] '.format(p, r2s_flood[p],
                                                   stats_r2_flood['med'][i],
                                                    stats_r2_flood['cis'][i][0],
                                                   stats_r2_flood['cis'][i][1],
                                                   ))
    
    # % % save 
    path_save = path_data
    filename = 'SBJ_SS_regressions_{}_{}'.format(fit,NB)
    if onlyminor:
        filename = filename+'_minorflood'
    if norm:
        filename = filename+'_norm'
    if hilb:
        filename = filename+'_hilb'
    
    
    save_dict(out,path_save,filename)

    return

def analysis()
        
    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt
    from scipy.signal import hilbert
    from utils_work import open_dict, get_sbj_df, join_sl_sbj, lin_fit, ridge_fit
    from utils_work import r2_score,r2_var, bootstrap, get_stats, save_dict

    stations = [
            '8443970', # boston
    #       '8447435',# chatam
            '8449130', # nantucket
            '8447930', # woods hole
            '8447386', # fall river
            '8452660', # Newport
            '8510560', # Montauk
            '8461490', # New London
            '8465705', # New Haven
            '8467150', # brigeport
            '8516945',# kings point
            '8531680', # sandy hook
            '8534720' # ; % Atlantic City, NJ
            ]
    #%%
    path_data = '/Users/carocamargo/Documents/data/floodSBJ/'
    outname = 'df_sl_dic_bp_mhhw' # use winds bp -- for details see get_save_data
    dfs = open_dict(path_data,outname)
    names = [dfs[stations[i]]['dic']['name'] for i,_ in enumerate(stations)]
    #%% threshold
    datum = 'MHHW'
    thresholds = open_dict(path_data,'noaaFlood_MHHW',)

    #%% run with changing parameters
    for fit in ['linear', 'ridge']: 
        for hilb in [True, False]:
            run_analysis(dfs,stations,fit=fit,hilb=hilb)
    #%% run only minor, ridge, no hilb
    run_analysis(dfs,stations,fit='ridge',hilb=False,onlyminor=True)
    return


def save_era5():

    """
   
    Get downloaded hourly data from eRA5 from 2013-2023, and save it in a single netcdf. 

    To download ERA5 data, go to: 
    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download
    and choose 'Reanalysis', and choose wind stress variable. 

    """

    import xarray as xr 
    import numpy as np 

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


    return



def save_noaa_thresholds():

    import pickle
    from utils_work import noaaFlood
    def save_dict(data,path,filename):
        
        outname = "{}{}.pkl".format(path,filename)
        with open(outname, "wb") as f:
            pickle.dump(data, f)
            
    stations = [
                '8443970', # boston
    #             '8447435',# chatam
                '8449130', # nantucket
                '8447930', # woods hole
                '8447386', # fall river
                '8452660', # Newport
                '8510560', # Montauk
                '8461490', # New London
                '8465705', # New Haven
                '8467150', # brigeport
                '8516945',# kings point
                '8531680', # sandy hook
                '8534720' # ; % Atlantic City, NJ
                ]
    path = '/Users/carocamargo/Documents/data/floodSBJ/'

    datum='MHHW'
    for datum in ['MHHW','MSL']:
        thresholds = {station:noaaFlood(station,datum=datum) for station in stations}
        save_dict(thresholds,path,'noaaFlood_{}'.format(datum))



    return


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

def save_dict(data,path,filename):

    outname = "{}{}.pkl".format(path,filename)
    with open(outname, "wb") as f:
        pickle.dump(data, f)
    return 

def save_waves():

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
    
    import pickle
    #%% save dict
    path_save = '/Users/carocamargo/Documents/data/floodSBJ/'
    filename = 'waves'
    save_dict(dic,path_save,filename)

    return
