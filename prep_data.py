#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:15:40 2025

Analysis for Camargo et al (2025), Earth's Futurr
@author: carocamargo
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.signal import hilbert
# from utils_work import r2_score,r2_var,lin_fit,ridge_fit,open_dict,get_sbj_df,join_sl_sbj
from functions_work import open_dict, get_sbj_df, join_sl_sbj, lin_fit, ridge_fit, r2_score,r2_var, bootstrap, get_stats

#%% reorderd by position in coastline from North to South
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


#%% def open_data(stations):
# tide gauges
path_data = '/Users/carocamargo/Documents/data/sealevel/'
outname = 'df_sl_dic_bp_mhhw_stress_mean' # use winds bp -- for details see save_dict_tgs.py 
    
dfs_sl = open_dict(outname,path_data)
names = [dfs_sl[stations[i]]['dic']['name'] for i,_ in enumerate(stations)]
df_sbj = get_sbj_df()
dfs_sl_sbj = join_sl_sbj(dfs_sl,df_sbj,convert=True)
# return dfs_sl_sbj
#%% threshold
path = '/Users/carocamargo/Documents/data/floodSBJ/'
datum = 'MHHW'
thresholds = open_dict('noaaFlood_MHHW',path)

#%% parameters
fit = 'ridge'
station = stations[0]
testrandom=True
hilb=False
onlyminor = False
#%%
for fit in ['linear','ridge']:
    #% %
    for hilb in [True, False]:
        #%%
        score = 'var'
        alpha = 0.3
        boot_method = 'rand'
        block_size=30
        norm=True
        NB  = 1000 # number of bootstraps replicas
        NID = len(stations) # number of stations
        NT = len(df_sbj) # length of time series (full - hourly)
        level=95
        out = {}
        table = np.zeros((len(stations),4*4))
        #% % get data of one station
        
        station = stations[2]
        #% %
        print('\n Start Loop \n')
        for istation, station in enumerate(stations):
            #% %
            print(station)
            df = dfs_sl_sbj[station]['df']
            # df['SL_hf']=np.array(df['SL']-df['SL_bp'])
            ref = thresholds[station]['nos_minor']
            ref_mod = thresholds[station]['nos_moderate']
            
            name = dfs_sl[station]['dic']['name']
            if 'time' in df.columns:
                df['time']=pd.to_datetime(df['time'])
            else:
                df['time'] = pd.to_datetime(df.index)
            # set index
            df.set_index('time',inplace=True)
            # resample to daily max
            df = df.resample('1D').apply(lambda x: x.loc[x['sl_raw'].idxmax()])
            ND = len(df) # length of time series (daily)
            # flood_days = dfs_sl[station]['flood_days'].index
            
            if onlyminor:
                flood_days = df[(df['sl_raw'] > ref) & (df['sl_raw'] < ref_mod)].index
            else:
                flood_days = df[df['sl_raw']>ref].index
            
            #% % get data we want
            x_col = ['Qy_bp','u','v']
            # compute hilbert transform of each parameter
            if hilb:
                for var in ['Qy_bp','u','v']:
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
                
            if fit=='linear':
                it, s = lin_fit(x,Y)
            elif fit=='ridge':
                it, s = ridge_fit(x,Y,alpha)
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
                #% %
                for ip,p in enumerate(params):
                    plt.subplot(2,2,ip+1)
                    plt.hist(r2_dist[ip,:],color='lightblue');
                    plt.axvline(np.mean(r2_dist[ip,:]),c='blue',linestyle='-')
                    plt.axvline(r2s_flood[p],c='black',linestyle='--')
                    plt.axvline(r2s[p],c='red',linestyle='--')
                    
                    
                    plt.title(p)
                plt.tight_layout()
                plt.suptitle(name)
                plt.show()
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
            
            #%%
            # now we bootstrap the samples
            # we also want to bootstrap sl_raw
            S = np.array(df['sl_raw'])
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
                da['sl_raw'] = np.array(X_boot[-1,ib,:])
                da.set_index('time',inplace=True)
                # select days of flood
                # da = da.loc[flood_days]
                if onlyminor:
                    da = da[(da['sl_raw']>ref) & (da['sl_raw']<ref_mod)]
                else:
                    da = da[da['sl_raw']>ref]
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
            for i,p in enumerate(['Qy_bp', 'u', 'v']):
                print('{}: {:.2f}, {:.2f} [{:.2f} {:.2f}] '.format(p, s[i],
                                                                   stats_slope['med'][i],
                                                       stats_slope['cis'][i][0],
                                                       stats_slope['cis'][i][1],
                                                       ))
            print('R2:')
            for i,p in enumerate(['overall','Qy_bp', 'u', 'v']):
                print('{}: {:.2f}, {:.2f} [{:.2f} {:.2f}] '.format(p, r2s[p],
                                                                   stats_r2['med'][i],
                                                                   stats_r2['cis'][i][0],
                                                                   stats_r2['cis'][i][1],
                                                       ))
                #% %
            print('R2 flood:')
            for i,p in enumerate(['overall','Qy_bp', 'u', 'v']):
                print('{}: {:.2f}, {:.2f} [{:.2f} {:.2f}] '.format(p, r2s_flood[p],
                                                       stats_r2_flood['med'][i],
                                                        stats_r2_flood['cis'][i][0],
                                                       stats_r2_flood['cis'][i][1],
                                                       ))
        
        # % % save 
        path_save = '/Users/carocamargo/Documents/data/floodSBJ/'
        filename = 'SBJ_SS_regressions_{}_{}_minorflood'.format(fit,NB)
        if norm:
            filename = filename+'_norm'
        if hilb:
            filename = filename+'_hilb'
        
        
        save_dict(out,path_save,filename)
    # # Load it back


