"""
Created on Thu Jul 24 17:08:57 2025

utils_work.py 

Aux functions


@author: carocamargo
"""


import requests
import json
import os
import pandas as pd
import numpy as np
import xarray as xr


def noaaTidesCurrents(station_id,product, start_date, end_date, 
                 datum = 'MSL',
                 interval='1H',
                 fmt='csv',
                 info=True,
                 save = True,
                 path='/Users/carocamargo/Documents/data/sealevel/NOAA/'):
    """
    
    Function to download data from NOAA tides & Currents
    
    Generic download function (tides, sea level, ...)
    
    Download data for an entire year! 
    
    Adapted from matlab script of Dr. Chris Piecuch
    https://github.com/christopherpiecuch/floodAR/blob/main/noaaTide.m

    Dependencies: 'Requests', 'json', library
    
    
    Parameters
    ----------
    station_id : int
        Station id (e.g., WHOI is 8447930).
    product: str
        Product values are air_gap, air_pressure, air_temperature, conductivity, 
        currents, currents_survey, currents_predictions, daily_mean, datums, 
        high_low, hourly_height, humidity, monthly_mean, one_minute_water_level, 
        predictions, salinity, visibility, water_level, water_temperature, ofs_water_level and wind 
        
    start_yr : int
        start year.
    end_yr : int
        end year of query.
    datum: str
        Reference variable of the water level. 
        Options: CRD ( Columbia River Datum);
                IGLD (International Great Lakes Datum)
                LWD (Great Lakes Low Water Datum)
                MHHW (Mean Higher High Water)
                MHW (Mean High Water)
                MTL (Mean Tide Level)
                MSL (Mean Sea Level) - Default
                MLW (Mean Low Water)
                MLLW (Mean Lower Low Water)
                NAVD (North American Verical Datum)
                STND (Station Datum)
    fmt: str
        format of data to be downloaded. 
        Options: json, csv (Default) or xml(no done yet)
    path: str
        path to folder to save files
    
    Example usage
    ----------
    station_id = "8447930"  # Station ID for WHOI, MA
    start_date = "20230101"
    end_date = "20230107"
    datum = 'MSL'
    fmt = 'json'
    r = noaaSeaLevel(station_id, start_date, end_date)   
    """
    
    # from datetime import datetime, timedelta

    
    # Hourly - The size limit for data retrieval for this product is 365 days 
    
    # formats: xml, json, csv
    
    #% %
    base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    # https://api.tidesandcurrents.noaa.gov/api/prod/
    
    payload = {
        "product": product,
        "application": "NOS.COOPS.TAC.WL",
        "begin_date": start_date,
        "end_date": end_date,
        "station": station_id,
        "datum": datum,# "MTL",  # Mean T|ide Level "MHHW" - Mean High Water
        "units": "metric",
        "time_zone": "GMT",  # Local Standard Time/Local Daylight Time
        "format": fmt,
        "interval":interval,
    }
    
    response = requests.get(base_url, params=payload)
    if response.status_code == 400:
        print('Error:', response.text)
        
    elif response.status_code == 200:
        if fmt =='json':
            # data = json.loads(response.text)
            
            # Extract sea level data
            sea_level_data = response.json()
            
            
            if save:
                filename = path+"{}_{}_{}_{}_{}.json".format(
                    product,station_id,start_date,end_date,datum)
                with open(filename, "w") as f:
                    json.dump(sea_level_data['data'], f)
            
            
   
        elif fmt =='csv':
            sea_level_data = response.text
            # Save the CSV file
            if save:
                filename = path+"{}_{}_{}_{}_{}.csv".format(
                    product,station_id,start_date,end_date,datum)
                with open(filename, "w") as f:
                    f.write(response.text)
        if info:
            print("Data downloaded successfully!")
        return sea_level_data
    
    else:
        print("Error occurred while downloading the data.")
        

#%% NOAA station metadata
def noaaMetadata(station_id,info=False):
    ''' 
    Inputs station id, returns dictionary with metadata (ID, name, lat, lon)
    
    # Example usage
    station_id = "9414290"  # Station ID for San Francisco, CA
    noaaMetadata(station_id)

    '''
    
    import requests
    base_url = "https://tidesandcurrents.noaa.gov/api/datagetter"
    payload = {
        "product": "hourly_height",
        "application": "NOS.COOPS.TAC.WL",
        "begin_date": "20220101",
        "end_date": "20220102",
        "station": station_id,
        "datum": "MSL",
        "units": "metric",
        "time_zone": "GMT",  # Local Standard Time/Local Daylight Time
        "format": "json",
    }
    
    response = requests.get(base_url, params=payload)
    data = response.json()

    if "error" in data:
        print("Error:", data["error"]["message"])
    else:
        # Extract metadata
        metadata = data["metadata"]
        
        if info:
            # Print station metadata
            print("Station ID:", metadata["id"])
            print("Name:", metadata["name"])
            print("Latitude:", metadata["lat"])
            print("Longitude:", metadata["lon"])
        
        return metadata


def noaa_Download_loop(station_id, product,start_year, end_year, 
                       interval='1H',
                 datum = 'MSL',
                 fmt='json',
                 path='/Users/carocamargo/Documents/data/sealevel/NOAA/'):

    """
    
    Function to download data from NOAA tides & Currents
    
    Adapted from matlab script of Dr. Chris Piecuch
    https://github.com/christopherpiecuch/floodAR/blob/main/noaaSealevel.m
   
    Dependencies: 'Requests', 'json', library
    
    
    Parameters
    ----------
    station_id : int
        Station id (e.g., WHOI is 8447930).
    product: str
        Product values are air_gap, air_pressure, air_temperature, conductivity, 
        currents, currents_survey, currents_predictions, daily_mean, datums, 
        high_low, hourly_height, humidity, monthly_mean, one_minute_water_level, 
        predictions, salinity, visibility, water_level, water_temperature, ofs_water_level and wind 
        
    start_yr : int
        start year.
    end_yr : int
        end year of query.
    datum: str
        Reference variable of the water level. 
        Options: CRD ( Columbia River Datum);
                IGLD (International Great Lakes Datum)
                LWD (Great Lakes Low Water Datum)
                MHHW (Mean Higher High Water)
                MHW (Mean High Water)
                MTL (Mean Tide Level)
                MSL (Mean Sea Level) - Default
                MLW (Mean Low Water)
                MLLW (Mean Lower Low Water)
                NAVD (North American Verical Datum)
                STND (Station Datum)
    fmt: str
        format of data to be downloaded. 
        Options: json, csv (Default) or xml(no done yet)
    path: str
        path to folder to save files
    
    Example usage
    ----------
    station_id = "8447930"  # Station ID for WHOI, MA
    start_date = "20230101"
    end_date = "20230107"
    datum = 'MSL'
    fmt = 'json'
    r = noaaSeaLevel(station_id, start_date, end_date)   
    """
    import os
    import calendar
    path = path+product+'/'
    if not os.path.exists(path):
        os.makedirs(path)
     
    filename = path+"{}_{}_{}_{}_{}.csv".format(product,
        station_id,start_year,end_year,datum)
    
    # extract metadata
    metadata = noaaMetadata(station_id)
    
    # Start Loop over time:

    # time limit is 31 days
    if product=='wind':
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Determine the number of days in the current month
                num_days = calendar.monthrange(year, month)[1]
                
                # Construct start and end dates using the determined number of days
                start_date = '{}{}01'.format(year, str(month).zfill(2))
                end_date = '{}{}{}'.format(year, str(month).zfill(2), str(num_days).zfill(2))
                # print(end_date)
                # Extract data:
                data = noaaTidesCurrents(station_id, product,
                                         start_date, end_date,
                                         interval=interval,
                                         fmt='csv', datum=datum,
                                         info=False,
                                         save=False
                                     )
                # Save metadata and sea level data to a CSV file
                if year==start_year and month==1: # make a new csv file
                
                    with open(filename, "w", newline="") as f:
                        
                        ## Write metadata
                        for key in metadata.keys():
                            f.write('#{} \n'.format(key))
                            f.write(metadata[key]+'\n')
                            
                        # Write sea level data
                        f.write(data)
                    
                else: # append to existing file
                    with open(filename, "a", newline="") as f:
                        f.write(data[36:]) # skip headers
    
        print("Data downloaded successfully!")
    else:
        # time limit is 1 year
        for year in range(start_year,end_year+1): # we can only download 1 year per time
            start_date = '{}0101'.format(year)
            end_date = '{}1231'.format(year)
            
            data = noaaTidesCurrents(station_id, product,
                                             start_date, end_date,
                                             interval=interval,
                                             fmt='csv', datum=datum,
                                             info=False,
                                             save=False
                                             )
            
            # Save metadata and sea level data to a CSV file
            if year==start_year: # make a new csv file
                
                with open(filename, "w", newline="") as f:
                    
                    ## Write metadata
                    for key in metadata.keys():
                        f.write('#{} \n'.format(key))
                        f.write(metadata[key]+'\n')
                        
                    # Write sea level data
                    f.write(data)
                    
            else: # append to existing file
                with open(filename, "a", newline="") as f:
                    f.write(data[36:]) # skip headers
    
        print("Data downloaded successfully!")
    
    return


#%% retrieve metadata of tide gauge (already downloaded)
def tg_ind_metadata(file):
    with open(file, "r") as f:
        lines = f.readlines()[:8]
    content = "".join(lines)
    keys = [c for c in content.split('\n') if c.startswith('#')]
    names = {'#id ':'ID', '#name ':'name', '#lat ':'lat', '#lon ':'lon'}
    values = [c for c in content.split('\n') if not c.startswith('#')]
    dic = {}
    
    for i,key in enumerate(keys):
        
        dic[names[key]]=values[i]  
    return dic

def tg_metadata(path):
    # path is the folder where the tide gauge data is 

    # get list of csvs
    flist = sorted([file for file in os.listdir(path) 
                    if not file.startswith('.') 
                    and file.endswith('.csv')])
    
    dic = {}
    for i,file in enumerate(flist):
        meta = tg_ind_metadata(path+file)
        dic[meta['ID']]=meta
    
    return dic

#%% open csv tide gauge
def open_tg_csv(file):
    df = pd.read_csv(file,header= 8,
                     names = ['date','level','Sigma','I','L'])
    df = df.drop(df[df['date']=='Error: No data was found. This product may not be offered at this station at the requested time.'].index)
    df = df.reset_index(drop=True)
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    return df
#%% wrap tide gauges 
def wrap_tg_csv(path_tg,save=False,freq='h'):
    #% % get metadata
    dic = tg_metadata(path_tg)
    
    # get list of csvs
    flist = sorted([file for file in os.listdir(path_tg) 
                    if not file.startswith('.') 
                    and file.endswith('.csv')])
    dfs= []
    # lens=[]
    IDs = []
    for i,file in enumerate(flist):
        # meta = metadata(path_tg+file)
        # dic[meta['ID']]=metafil.=e
        df = open_tg_csv(path_tg+file)
        # if i==0:
        #     df2=df
        # else:
            
        # print(len(df))
        # lens.append(len(df))
        dfs.append(df)
        IDs.append(file.split('_')[2])
    df_all = pd.concat(dfs#,axis=1
                       )
    
    #% % unite all in a dataframe of uniform time
    t0 = pd.to_datetime(df_all.date.min())
    t1 = pd.to_datetime(df_all.date.max())
    date_array = pd.date_range(start=t0, end=t1, freq=freq)
    df_tg = pd.DataFrame({'date':date_array})
    for i,df in zip(IDs,dfs):
        df['date']=pd.to_datetime(df['date'])
        new_cols = {'level':'level_{}'.format(i),
                    'Sigma':'Sigma_{}'.format(i),
                    'I':'I_{}'.format(i),
                    'L':'L_{}'.format(i),
                    }
        df = df.rename(columns=new_cols)
        df_tg = pd.merge(df_tg,df,on='date',how='outer')
    
    #% % make new arrays
    levels = np.zeros((len(df_tg),len(IDs)))
    sigmas = np.zeros((len(df_tg),len(IDs)))
    Is  = np.zeros((len(df_tg),len(IDs)))
    Ls  = np.zeros((len(df_tg),len(IDs)))
    for j,i in enumerate(IDs):
        levels[:,j] = np.array(df_tg['level_{}'.format(i)])
        
        sigmas[:,j] = np.array(df_tg['Sigma_{}'.format(i)])
        Is[:,j] = np.array(df_tg['I_{}'.format(i)])
        Ls[:,j] = np.array(df_tg['L_{}'.format(i)])
    
    #% % put it in a dataset
    ds = xr.Dataset(data_vars={'level':(('time','station'),levels),
                               'sigma':(('time','station'),sigmas),
                               'L':(('time','station'),Ls),
                               'I':(('time','station'),Is),
                               'name':(('station'),[dic[i]['name'] for i in IDs]),
                               'lat':(('station'),[float(dic[i]['lat']) for i in IDs]),
                               'lon':(('station'),[float(dic[i]['lon']) for i in IDs])
                               
                                 },
                    coords={'time':date_array,
                            'station':IDs}
                    )
    if save:
        # ds.to_netcdf(path_tg.split('hourly_height/')[0]+'tgs_sl_combined.nc')
        start_y = np.array(ds.time.min().dt.year)
        end_y = np.array(ds.time.max().dt.year)
        name = 'tgs_sl_combined'
        ds.to_netcdf(path_tg.split('hourly_height/')[0]+name+'_{}_{}.nc'.format(start_y,end_y))
        
    return ds
#%% open csv tide gauge - tides

def correct_time(df,freq='1H'):
    dates = np.array(df['date'])
    if np.any(dates=='00'):
        ind = np.where(dates=='00')[0]
        for i in ind:
            dates[i] = dates[i + 1].split(' ')[0]+" 00:00"
    df['date'] =dates
    t0 = pd.to_datetime(dates.min())
    t1 = pd.to_datetime(dates.max())
    date_array = pd.date_range(start=t0, end=t1, freq=freq)
    return date_array

def open_tide_csv(file,freq='1H'):
    df = pd.read_csv(file,header= 8,
                     names = ['date','tide'])
    df = df.drop(df[df['date']=='Error: No data was found. This product may not be offered at this station at the requested time.'].index)
    df['date']=correct_time(df,freq=freq)
    df = df.reset_index(drop=True)
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    return df
#%% wrap tide gauges - tides
def wrap_tide_csv(path,save=False,freq='1H'):
    #% % get metadata
    dic = tg_metadata(path)
    
    # get list of csvs
    flist = sorted([file for file in os.listdir(path) 
                    if not file.startswith('.') 
                    and file.endswith('.csv')])
    dfs= []
    # lens=[]
    IDs = []
    for i,file in enumerate(flist):
        # meta = metadata(path_tg+file)
        # dic[meta['ID']]=metafil.=e
        df = open_tide_csv(path+file,freq=freq)
        
        # if i==0:
        #     df2=df
        # else:
        IDs.append(file.split('_')[1])
        # print(len(df))
        # lens.append(len(df))
        df = df.rename(columns = {'tide':file.split('_')[1]})
        if i>0: # dates are all the same, so remove fromt he next files
            df = df.drop(['date'],axis=1)
        else:
            df['date']=pd.to_datetime(df['date'])
        dfs.append(df)
    # df_tg = pd.merge(dfs,on='date',how='outer')   
    df_all = pd.concat(dfs,axis=1
                       )
    df_all = df_all.rename(columns={'date':'time'})
    # date =np.array(df_all.date)
    df_all  = df_all.set_index('time')
    if freq =='H':
        df_all = df_all.resample('1H').mean()
    ds  = df_all.to_xarray()
    tides = np.zeros((len(df_all),len(IDs)))
    for j,i in enumerate(IDs):
        tides[:,j] = np.array(df_all['{}'.format(i)])
    ds = ds.assign_coords(station=('station',IDs))
    # ds = ds.assign_coords(station=('time',date))
    
    ds['tides'] = (('time','station'),tides)
    ds['name']=(('station'),[dic[i]['name'] for i in IDs])
    ds['lat']=(('station'),[float(dic[i]['lat']) for i in IDs])
    ds['lon']=(('station'),[float(dic[i]['lon']) for i in IDs])
    
    if save:
        name = 'tgs_tides_combined'
        # if freq=='H':
        #     name = name+'_h'
        

        start_y = np.array(ds.time.min().dt.year)
        end_y = np.array(ds.time.max().dt.year)
        ds.to_netcdf(path.split('predictions/')[0]+name+'_{}_{}.nc'.format(start_y,end_y))
    return ds
#%% atm pressure

def correct_time_atm(df):
    dates = np.array(df['date'])
    ind = np.where(dates=='1-01 00:00')[0]
    for i in ind:
        # print(df.date.iloc[(i-1):(i+2)])
        dates[i] = dates[i + 1].split(' ')[0]+" 00:00"
        
        
    ind = np.where(dates=='1-01 01:00')[0]
    for i in ind:
        dates[i] = dates[i + 1].split(' ')[0]+" 01:00"
    df['date'] =dates
    
    return df['date'] 

def open_atm_csv(file):
    df = pd.read_csv(file,header= 8,
                     names = ['date','pressure',
                               'X','N','R'
                              ]
                     )
    df = df.drop(['X','N','R'],axis=1)
    # df = df.drop(df[df['date']==].index)
    error_msg = ['Error: No data was found. This product may not be offered at this station at the requested time.',
        'Error: No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time.1-01 01:00',
        ' No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time.1-01 00:00',
        ]
    for error in error_msg:
        df = df.drop(df[df['date']==error].index)

    df['date']=correct_time_atm(df)
    # df = df.reset_index(drop=True)
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    return df

#% % wrap tide gauges - atm
def wrap_atm_csv(path,save=False,freq_in='6min',freq_out='H'):
    #% % get metadata
    dic = tg_metadata(path)
    
    # get list of csvs
    flist = sorted([file for file in os.listdir(path) 
                    if not file.startswith('.') 
                    and file.endswith('.csv')])
    dfs= []
    # lens=[]
    IDs = []
    i=0;file=flist[i]
    for i,file in enumerate(flist):
        # open file:
        df = open_atm_csv(path+file)

        IDs.append(file.split('_')[2])
        
        # rename column:
        df = df.rename(columns = {'pressure':'pressure_'+file.split('_')[2]}
                       )

        df['date']=pd.to_datetime(df['date'])

        dfs.append(df)

    df_all = pd.concat(dfs#,axis=1
                       )
    
    #% % unite all in a dataframe of uniform time
    t0 = pd.to_datetime(df_all.date.min())
    t1 = pd.to_datetime(df_all.date.max())
    date_array = pd.date_range(start=t0, end=t1, freq=freq_in)
    df_tg = pd.DataFrame({'date':date_array})
    for i,df in zip(IDs,dfs):
        # print(i) 
        df['date']=pd.to_datetime(df['date'])
        df_tg = pd.merge(df_tg,df,on='date',how='left')
        # print('merged {}'.format(i))
    
    # in case downloaded in higher
    if freq_out =='H':
        df_tg = df_tg.set_index('date')
        df_tg = df_tg.resample(freq_out).mean()
        
    #% % make new arrays
    levels = np.zeros((len(df_tg),len(IDs)))

    for j,i in enumerate(IDs):
        levels[:,j] = np.array(df_tg['pressure_{}'.format(i)])
        
    
    #% % put it in a dataset
    ds = xr.Dataset(data_vars={'atmp':(('time','station'),levels),
                               
                               'name':(('station'),[dic[i]['name'] for i in IDs]),
                               'lat':(('station'),[float(dic[i]['lat']) for i in IDs]),
                               'lon':(('station'),[float(dic[i]['lon']) for i in IDs])
                               
                                 },
                    coords={'time':date_array,
                            'station':IDs}
                    )

    
    if save:
        name = 'tgs_atmp_combined'
        start_y = np.array(ds.time.min().dt.year)
        end_y = np.array(ds.time.max().dt.year)
        ds.to_netcdf(path.split('air_pressure/')[0]+name+'_{}_{}.nc'.format(start_y,end_y))
    return ds

#%% 
def open_wind_csv(file):
    df = pd.read_csv(file,header= 8,
                     
                     )
    df = df.drop([' X',' R '],axis=1)
    df = df.rename(columns={'Date Time':'date', 
                            ' Speed':'Speed', 
                            ' Direction':'Angle', 
                            ' Direction.1':'Direction', 
                            ' Gust':'Gust',}) 
    # df = df.drop(df[df['date']==].index)
    error_msg = ['Error: No data was found. This product may not be offered at this station at the requested time.',
        'Error: No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time.1-01 01:00',
        ' No data was found. This product may not be offered at this station at the requested time. No data was found. This product may not be offered at this station at the requested time.1-01 00:00',
        ]
    for error in error_msg:
        df = df.drop(df[df['date']==error].index)

    # df['date']=correct_time_atm(df)
    # df = df.reset_index(drop=True)
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    return df
#%% wrap tide gauges - atm
def wrap_wind_csv(path,save=False):
    #% % get metadata
    dic = tg_metadata(path)
    
    # get list of csvs
    flist = sorted([file for file in os.listdir(path) 
                    if not file.startswith('.') 
                    and file.endswith('.csv')])
    #% %
    dfs= []
    # lens=[]
    IDs = []
    i=0;file=flist[i]
    for i,file in enumerate(flist):
        # open file:
        df = open_wind_csv(path+file)
        # print(len(df))
        IDs.append(file.split('_')[1])
        
        # rename column:
        cols =[c for c in df.columns if c!='date'
               ]
        for c in cols:
            df = df.rename(columns = {c:'{}_'.format(c)+IDs[i]}
                       )
        
        df.set_index('date',inplace=True)
        # df['date']=pd.to_datetime(df['date'])

        dfs.append(df)

    df_all = pd.concat(dfs,axis=1
                       )
    
    #% % unite all in a dataframe of uniform time
    df_all.reset_index(inplace=True)
    t0 = pd.to_datetime(min(df_all.date))
    t1 = pd.to_datetime(max(df_all.date))
    date_array = pd.date_range(start=t0, end=t1, freq='1H')
    if len(date_array) != len(df_all):
        df_tg = pd.DataFrame({'date':date_array})
        for i,df in zip(IDs,dfs):
            # print(i) 
            df.reset_index(inplace=True)
            df['date']=pd.to_datetime(df['date'])
            df_tg = pd.merge(df_tg,df,on='date',how='left')
        # print('merged {}'.format(i))
    else:
        df_tg = df_all
    
   
        
    #% % make new arrays
    speed = np.zeros((len(df_tg),len(IDs)))
    angle = np.zeros((len(df_tg),len(IDs)))
    direction = np.empty((len(df_tg),len(IDs)), dtype=object)
    gust = np.zeros((len(df_tg),len(IDs)))

    for j,i in enumerate(IDs):
        speed[:,j] = np.array(df_tg['Speed_{}'.format(i)])
        angle[:,j] = np.array(df_tg['Angle_{}'.format(i)])
        direction[:,j] = np.array(df_tg['Direction_{}'.format(i)])
        gust[:,j] = np.array(df_tg['Gust_{}'.format(i)])
        
    
    #% % put it in a dataset
    ds = xr.Dataset(data_vars={'speed':(('time','station'),speed),
                               'angle':(('time','station'),angle),
                               'direction':(('time','station'),direction),
                               'gust':(('time','station'),gust),

                               'name':(('station'),[dic[i]['name'] for i in IDs]),
                               'lat':(('station'),[float(dic[i]['lat']) for i in IDs]),
                               'lon':(('station'),[float(dic[i]['lon']) for i in IDs])
                               
                                 },
                    coords={'time':date_array,
                            'station':IDs}
                    )
#% %
    
    if save:
        name = 'tgs_wind_combined'
        start_y = np.array(ds.time.min().dt.year)
        end_y = np.array(ds.time.max().dt.year)
        ds.to_netcdf(path.split('wind/')[0]+name+'_{}_{}.nc'.format(start_y,end_y))
    return ds
#%%
def wrap_csv(var='hourly_height',path='./',save=True,
             interval='1H',y0=2009,y1=2013):
    
    if var=='air_pressure':
        ds = wrap_atm_csv(path=path,save=save,freq_in=interval,freq_out=interval)
    if var=='hourly_height':
        ds = wrap_tg_csv(path_tg=path,save=save,freq='h')
    if var =='monthly_mean':
        ds = wrap_tg_csv(path_tg=path,save=save,freq='ME')
    if var=='predictions':
        ds = wrap_tide_csv(path=path, save=save, freq=interval)
    if var =='wind':
        ds = wrap_wind_csv(path=path,save=save)
        
    return ds

#%% 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:18:50 2025

Utils 

@author: carocamargo
"""
import pickle 
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr
from scipy.signal import  butter, filtfilt
from scipy import signal
from sklearn import linear_model
import os

#%% working ones 
def save_dict(data,path,filename):
    
    outname = "{}{}.pkl".format(path,filename)
    with open(outname, "wb") as f:
        pickle.dump(data, f)
        
def open_dict(path,filename):
    outname = "{}{}.pkl".format(path,filename)
    with open(outname, "rb") as f:
        loaded_data = pickle.load(f)   
    return loaded_data
#% %

def block_bootstrap(X, Y, block_size, N):
    """Performs block bootstrapping on multiple time series in X_list and a single time series Y.

    Args:
        X_list (list of arrays): List of original time series.
        Y (array): Original time series.
        block_size (int): Length of each block.
        N (int): Number of bootstrap samples.

    Returns:
        samples_X (list of arrays): Bootstrap resampled time series for each series in X_list (N, len(X)).
        samples_Y (array): Bootstrap resampled time series for Y (N, len(Y)).
    """
    t = len(Y)
    if type(X)==list:
        num_series = len(X)
        samples_X = [np.zeros((N, t)) for _ in range(num_series)]
    else:
        samples_X = np.zeros((N, t))
    samples_Y = np.zeros((N, t))
    
    # Compute number of blocks needed
    num_blocks = int(np.ceil(t / block_size))
    
    for i in range(N):
        if type(X)==list:
            resampled_X = [[] for _ in range(num_series)]
        else:
            resampled_X = []
        resampled_Y = []
        
        for _ in range(num_blocks):
            start_idx = np.random.randint(0, t - block_size + 1)  # Random block start
            if type(X)==list:
                for j in range(num_series):
                    resampled_X[j].extend(X[j][start_idx : start_idx + block_size])  # Collect block
            else:
                resampled_X.extend(X[start_idx : start_idx + block_size])  # Collect block

            resampled_Y.extend(Y[start_idx : start_idx + block_size])  # Collect block
        if type(X)==list:
            for j in range(num_series):
                samples_X[j][i, :] = np.array(resampled_X[j][:t])  # Trim to original length if oversampled
            samples_X = np.array(samples_X)
        else:
            samples_X[i, :] = np.array(resampled_X[:t])  # Trim to original length if oversampled

        samples_Y[i, :] = np.array(resampled_Y[:t])  # Trim to original length if oversampled
    
    return samples_X, samples_Y

        
def rand_bootstrap(X,Y,N):
    NN = len(Y)  # time series size
    
    if type(X)==list:
        num_series = len(X)
        samples_X = [np.zeros((N, NN)) for _ in range(num_series)]
    else:
        samples_X = np.zeros((N, NN))
    samples_Y = np.zeros((N, NN))
    
    for i in range(N):
        ii = np.ceil(NN * np.random.rand(NN, 1)).astype(int)  # Bootstrap indices
        if type(X)==list:
            for j in range(num_series):
                samples_X[j][i, :] = np.array(X[j][ii-1][:,0]) 
            samples_X = np.array(samples_X)
        else:
            samples_X[i,:] = np.array(X[ii-1][:,0]) # 0 until length of NN-1
        samples_Y[i,:] = np.array(Y[ii-1][:,0])
    
    return samples_X, samples_Y

def bootstrap(X,Y,
              method='block',
              NB = 1000,
              block_size = 30,
              ):
    if method=='block':
        samples_X, samples_Y = block_bootstrap(X,Y,block_size,NB)
    elif method=='rand':
        samples_X, samples_Y = rand_bootstrap(X, Y, NB)
    else:
        raise ValueError(f"Bootstrap method '{method}' is not recognized.")
    return samples_X, samples_Y


def get_stats(data,level):
    NB, NP = data.shape
    mean = np.mean(data,axis=0)
    med = np.median(data,axis=0)
    cis = np.zeros((NP,2))
    for ip in range(NP):
        cis[ip] = np.percentile(data[:,ip], [(100-level)/2, (100+level)/2])  # CI
    ci_width = np.abs(cis[:,0] - cis[:,1])
    # perturbed_error = ci_width/2
    return {'mean':mean,'med':med,'cis':cis,'ci_widths':ci_width}
#%%
def find_closest_cell(ds, lat1, lon1,remove_nan=False):
    """
    Find the index of the closest latitude and longitude to the given coordinates.
    
    Parameters:
    ds (xarray.Dataset): The dataset containing 'latitude' and 'longitude' coordinates.
    lat1 (float): The latitude of the target point.
    lon1 (float): The longitude of the target point.
    
    Returns:
    tuple: The index of the closest latitude and longitude.
    """
    
    if remove_nan:
        # Drop latitudes and longitudes where u and v are NaN across all time steps
        ds = ds.dropna(dim='latitude', how='all', subset=['u', 'v'])
        ds = ds.dropna(dim='longitude', how='all', subset=['u', 'v'])


    # Extract latitude and longitude from the dataset
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    # Convert latitude and longitude to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lats_rad = np.radians(lats[:, np.newaxis])
    lons_rad = np.radians(lons)

    # Haversine formula to calculate distance between two lat/lon points
    dlat = lats_rad - lat1_rad
    dlon = lons_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lats_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    R = 6371  # Earth radius in kilometers
    distances = R * c

    # Find the index of the minimum distance
    min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
    
    return min_dist_idx


def myOLS(x,y,info=True):
    A = np.array(x)
    A = np.column_stack((np.ones(A.shape[0]), A))  # Add a column of ones

    # solve it
    # opt a
    # Calculate the coefficients using the normal equation
    # and using the pseudo-inverse
    # beta = (X^T * X)^(-1) * X^T * y  =>  beta = X_pseudo_inv * y
    # A_pseudo_inv = np.linalg.pinv(A)
    # beta = A_pseudo_inv.dot(sl)
    # print("Coefficients:", beta[1:])
    # print("Intercept:", beta[0])
    # # Make predictions
    # pred = A.dot(beta)
    
    # opt b
    solution=np.linalg.lstsq(A,y,rcond=None)[0] # coefs
    if info:
        print('intercept, Coefs:',solution)
    y_pred = np.matmul(A,solution)
    # intercept = solution[0]  # Intercept term
    # coefs = solution[1:]     # Coefficients
    # Calculate R²
    SS_total = np.sum((y - np.mean(y))**2)
    SS_residual = np.sum((y - y_pred)**2)
    r2 = 1 - (SS_residual / SS_total)
    if info:
        print('R squared: {:.2f} %'.format(r2*100))
    return y_pred, solution, r2


def r2_score(y,y_pred):
    SS_total = np.sum((y - np.mean(y)) ** 2)
    SS_residual = np.sum((y - y_pred) ** 2)        
    r2 = 1 - (SS_residual / SS_total)
    return r2 

def r2_var(y,y_pred):
    return 1 - np.var(y - y_pred) / np.var(y)

def lin_fit(x,y):
    if len(x.shape)==1:
        X = x.reshape(-1, 1) 
    else:
        X = x
    model = linear_model.LinearRegression()
    model.fit(X, y)

    # return solution
    intercept = model.intercept_
    coefs = model.coef_
    return intercept,coefs


def ridge_fit(x, y, alpha=1):
    if len(x.shape) == 1:
        X = x.reshape(-1, 1)
    else:
        X = x
    
    # Fit the full model
    model = linear_model.Ridge(alpha=alpha)
    model.fit(X, y)

    
    # Store full model coefficients
    intercept = model.intercept_
    coefs = model.coef_
    
    # solution = [model.intercept_]
    # solution.extend(model.coef_)
    
    return intercept,coefs


def get_sbj_df():
    '''
    Function that opens ERA5 wind data and SBJ time series.
    Identify wind grid ceels at the pioneer location and 
    at a remote north location
    Make linear multiple regression of winds on the sbj. 
    Make a dataframe with Q (total), Q_nw and Q_w (no wind and wind, based on the regression)
    and the wind vectors. 
    
    
    '''
    
    dm = xr.open_dataset('/Users/carocamargo/Documents/data/ERA5/wind_single_2013-2024.nc')

    # land mask
    dmask = xr.open_dataset('/Users/carocamargo/Documents/data/ERA5/gebco_mask.nc')
    dmask
    mask = np.array(dmask.elevation)
    mask[mask>0]=np.nan
    mask[np.isfinite(mask)]=1
    
    #% % open SBJ
    path_to_data = '/Users/carocamargo/Documents/data/timeseries_66_ft/'
    da = xr.open_dataset(path_to_data+'jet_SNE.nc')
    da = da[['date','Qy',
              # 'SL' # SNE SL
              ]]
    da = da.rename({'date': 'time'})
    
    # merge winds and SBJ
    ds = xr.merge([dm, da], join='inner')
    
    dx = ds
    Qy = np.array(dx.Qy)
    
    # find cells closest to Pioneer
    #     # Central Mooring Location: 40.1000, -70.8800
    ooip_lon =-70.8
    ooip_lat = 40.1
    ilat,ilon = find_closest_cell(dx, ooip_lat, ooip_lon)
    u_era5_sbj = np.array(dx['u'][:,ilat,ilon])
    v_era5_sbj = np.array(dx['v'][:,ilat,ilon])
    
    #% % remote wind
    lat_rem = 43.5; lon_rem = -62.5
    
    ilat,ilon = find_closest_cell(dx, lat_rem,lon_rem)
    u_era5_rem = np.array(dx['u'][:,ilat,ilon])
    v_era5_rem = np.array(dx['v'][:,ilat,ilon])
    
    # regress Qy with winds
    ival = np.isfinite(Qy)
    pred3,rc,r_v=myOLS(x=pd.DataFrame([v_era5_sbj[ival],
                                       u_era5_sbj[ival],
                                       u_era5_rem[ival],
                                       v_era5_rem[ival]
                                      ]
                                     ).T,y=Qy[ival],
                       info=False)
    
    pred = np.full_like(Qy,np.nan)
    pred[ival]=pred3
    r,p = pearsonr(Qy[ival],pred[ival])
    
    Q_wind = np.array(pred)
    Q_nw = np.array(Qy-pred)
    
    # # plot time series
    # plt.figure(figsize=(12,6))
    # i=360
    # j = i+(60*24) # (60 days)
    # # plt.plot(dx['time'][i:j],u_era5_tg[i:j],c='blue',label='u - ERA5 @TG (m/s)',alpha=0.5,linewidth=3)
    
    
    # plt.plot(dx['time'][i:j],Qy[i:j],c='gray',alpha=0.8,label='SBJ (Sv)',linestyle='-',linewidth=3)
    # plt.plot(dx['time'][i:j],Q_wind[i:j],c='red',alpha=0.8,label='SBJ (wind)',linestyle='--',linewidth=3)
    # plt.plot(dx['time'][i:j],Q_nw[i:j] - pred[i:j],c='blue',alpha=0.8,label='SBJ (no-wind)',linestyle=':',linewidth=3)
    
    
    # plt.legend(loc='upper left',ncol=3)
    # plt.title('r2 = {:.2f}; c= {:.2f}({:.3f})'.format(r_v,r,p))
    # plt.show()
    
    # make dataframe
    df_sbj = pd.DataFrame({'Datetime':dx.time,
                          'Qy':Qy,
                          'Qw':Q_wind,
                          'Qnw':Q_nw,
                          'u_pioneer':u_era5_sbj, # u component
                          'v_pioneer':v_era5_sbj, # v component
                          'ws_pioneer':np.array(np.sqrt(u_era5_sbj**2 + v_era5_sbj**2)), # wind speed,
                          'u_remote':u_era5_rem, # u component
                          'v_remote':v_era5_rem, # v component
                          'ws_remote':np.array(np.sqrt(u_era5_rem**2 + v_era5_rem**2)), # wind speed,
                           
                          }
                         )
    
    df_sbj.set_index('Datetime', inplace=True)
    return df_sbj

def butter_pass(y, lowcut, highcut, sr = 1, order=1, 
                nyquist = True,
                method='pad',btype='bandpass'
               ):
    '''
    Using signal.butter we create a butterworth filter
    We then apply the filter both forward and backword using signal.filtfilt
    
    '''
    if nyquist:
        f_nyq = 0.5 * sr # half of the sampling rate of a discrete-time signal
        lowcut = lowcut/f_nyq
        highcut = highcut/f_nyq
    b, a = butter(order,[lowcut,highcut], btype=btype)
    yf = filtfilt (b, a, y, method=method)
    
    return yf

def find_wind(lat,lon,dx):
    
    ilat,ilon = find_closest_cell(dx, lat,lon)
    u= np.array(dx['u'][:,ilat,ilon])
    v = np.array(dx['v'][:,ilat,ilon])
    return u, v
    
def regress_wind(df,var_name,u_name='u',v_name='v'):
    # regress SL with winds
    # SL = np.array(df['sl_raw']-df['tide']-df['ib'])
    SL = np.array(df[var_name])
    v=np.array(df[v_name])
    u=np.array(df[u_name])
    ival = np.isfinite(SL)
    pred3,rc,r_v=myOLS(x=pd.DataFrame([v[ival],
                                       u[ival],
                                       
                                      ]
                                     ).T,
                       y=np.array(SL[ival]),
                       info=False)
    
    pred = np.full_like(SL,np.nan)
    pred[ival]=pred3
    r,p = pearsonr(SL,pred[ival])
    
    SL_wind = np.array(pred)
    SL_nw = np.array(SL-pred)
    stats = [r_v, r, p]
    return SL_wind, SL_nw, stats

def get_data_tg(station_id,meta=True,
                path_to_data = '/Users/carocamargo/Documents/data/timeseries_66_ft/'):
    path = '{}rec/{}.txt'.format(path_to_data,station_id)
    df = pd.read_csv(path,
                     sep='\t', )
    df.drop('Unnamed: 0', axis=1,inplace=True)
    df['res'] = np.array(df['SL'] - df['SL_bp'])
    df['SL_bp_prime'] = np.array(df['SL_bp'] - df['SL_recons'])
    if meta:
        dic = noaaMetadata(station_id)
        datum='MSL'
        dic['flood_thresholds'] = noaaFlood(station_id,datum)
        
        return df, dic
    else:
        return df    
    
def organize_df_SL(df_sl):
    df = pd.DataFrame({'Datetime':pd.to_datetime(df_sl['time']),
                   'u_tg':df_sl['u'],
                   'v_tg':df_sl['v'],
                   'ws':np.array(np.sqrt(df_sl['v']**2 + df_sl['u']**2)),
                   'SL_cor':df_sl['SL'],
                   'SL_tg':df_sl['sl_raw'],
                   'SL_ib':df_sl['ib'],
                   'SL_tide':df_sl['tide'],
                    
                  })

    df.set_index('Datetime', inplace=True)
    return df

def open_era_SBJ(era5_file='/Users/carocamargo/Documents/data/ERA5/wind_single_2013-2024.nc'):
    #% % open ERA-5 winds
    dm = xr.open_dataset(era5_file)
    
    # land mask
    dmask = xr.open_dataset('/Users/carocamargo/Documents/data/ERA5/gebco_mask.nc')
    dmask
    mask = np.array(dmask.elevation)
    mask[mask>0]=np.nan
    mask[np.isfinite(mask)]=1
    
    #% % open SBJ
    path_to_data = '/Users/carocamargo/Documents/data/timeseries_66_ft/'
    da = xr.open_dataset(path_to_data+'jet_SNE.nc')
    da = da[['date','Qy',
              # 'SL' # SNE SL
              ]]
    da = da.rename({'date': 'time'})
    
    # merge winds and SBJ
    ds = xr.merge([dm, da], join='inner')
    
    return ds


def regress_sl_sbj(df_sbj, df_sl,
                   lowcut = 1.0 / (15.0 * 24), # 1/15 days in hours
                   highcut = 1.0 / 24  # 1 day in hours
):
    df = df_sl.join(df_sbj, how='left')

    # Band pass Q and SL_cor
    df['Q_bp']=butter_pass(np.array(df['Qy']),lowcut,highcut) # Q band-passed to 1-15 days
    df['SL_bp']=butter_pass(np.array(df['SL_cor']),lowcut,highcut) # SL_cor band-passed to 1-15 days
    df['SL_hf'] = np.array(df['SL_cor'] - df['SL_bp']) # SL_cor freq>15days or <1day

    df['alpha'] = np.array(df['SL_tide']+df['SL_ib']+df['SL_hf'])


    # Regress SL_bp with Q_bp
    df['SL_bp_Q'], _, _ = myOLS(x=df['Q_bp'],y=df['SL_bp'], info=False)# 1-15 days jet-related
    df['SL_bp_star'] = np.array(df['SL_bp'] - df['SL_bp_Q'])
    df['SL_nojet'] = np.array(df['alpha'] + df['SL_bp_star'])
    df['SL_jet'] = np.array(df['alpha'] + df['SL_bp_Q'])
    
    # band pass Q-wind and Q_nowind
    df['Q_nw_bp']=butter_pass(np.array(df['Qnw']),lowcut,highcut) # Q_windband-passed to 1-15 days
    df['Q_w_bp']=butter_pass(np.array(df['Qw']),lowcut,highcut) # Q_nowind band-passed to 1-15 days
    
    # Regress SL_bp with Q_nw_bp 
    df['SL_bp_Q_nw'], _, _ = myOLS(x=df['Q_nw_bp'],y=df['SL_bp'], info=False)# 1-15 days jet-related
    df['SL_bp_star_nw'] = np.array(df['SL_bp'] - df['SL_bp_Q_nw'])
    df['SL_nojet_nw'] = np.array(df['alpha'] + df['SL_bp_star_nw'])
    df['SL_jet_nw'] = np.array(df['alpha'] + df['SL_bp_Q_nw'])
    
    
    # Regress SL_bp with Q_w_bp
    df['SL_bp_Q_w'], _, _ = myOLS(x=df['Q_w_bp'],y=df['SL_bp'], info=False)# 1-15 days jet-related
    df['SL_bp_star_w'] = np.array(df['SL_bp'] - df['SL_bp_Q_w'])
    df['SL_nojet_w'] = np.array(df['alpha'] + df['SL_bp_star_w'])
    df['SL_jet_w'] = np.array(df['alpha'] + df['SL_bp_Q_w'])
    
    return df

def noaaMetadata(station_id,info=False):
    ''' 
    Inputs station id, returns dictionary with metadata (ID, name, lat, lon)
    
    # Example usage
    station_id = "9414290"  # Station ID for San Francisco, CA
    noaaMetadata(station_id)

    '''
    
    import requests
    base_url = "https://tidesandcurrents.noaa.gov/api/datagetter"
    payload = {
        "product": "hourly_height",
        "application": "NOS.COOPS.TAC.WL",
        "begin_date": "20220101",
        "end_date": "20220102",
        "station": station_id,
        "datum": "MSL",
        "units": "metric",
        "time_zone": "GMT",  # Local Standard Time/Local Daylight Time
        "format": "json",
    }
    
    response = requests.get(base_url, params=payload)
    data = response.json()

    if "error" in data:
        print("Error:", data["error"]["message"])
    else:
        # Extract metadata
        metadata = data["metadata"]
        
        if info:
            # Print station metadata
            print("Station ID:", metadata["id"])
            print("Name:", metadata["name"])
            print("Latitude:", metadata["lat"])
            print("Longitude:", metadata["lon"])
        
        return metadata
    
def noaaDatum(station_id,metric=True):
    import requests
    if metric:
        URL = f'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station_id}/datums.json?units=metric'
    else:
        URL = f'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station_id}/datums.json?'
    response = requests.get(URL)
    if response.status_code == 200:
        data = response.json()
        dat = data['datums']
        dat_dict = {entry['name']: {'description': entry['description'], 'value': entry['value']} for entry in dat}
        # ref = dat_dict[datum]['value']
    
    else:
        print('Issue in getting datum. please check URL and station id')
        dat_dict = {};
        # ref = np.nan
    
    return dat_dict#, ref

def noaaFlood(station_id,datum=False,metric=True):
    # formulas as
    # Safad et al., 2024: doi: 10.1038/s41467-024-48545-1
    import requests

    if metric:
        URL = f'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station_id}/floodlevels.json?units=metric'
    else:
        URL = f'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station_id}/floodlevels.json?'
    # Download data from the NOAA API
    response = requests.get(URL)
    
    flood={}
    # Check if the request was successful
    if response.status_code == 200:
        if datum:
            dat_dict = noaaDatum(station_id,metric=metric)
            ref = dat_dict[datum]['value']
            gt = dat_dict['GT']
        else:
            ref = 0
            datum='STND (Station Datum)'
            gt = 0
        data = response.json()
        for level in ['nos_minor', 'nos_moderate', 'nos_major', 'nws_minor', 'nws_moderate', 'nws_major']:
            if not data[level]:
                if level =='nos_minor':
                    if datum=='MSL':
                        msl  = dat_dict['MSL']['value']
                        mhhw = dat_dict['MHHW']['value']
                        data[level] = 0.5 + (0.04 * gt['value']) + mhhw # - msl # we remove ref below
                    elif datum =='MHHW':
                        data[level] = 0.5 + (0.04 * gt['value'])
                    elif datum =='MLLW':
                        data[level] = 0.5 + (1.04 * gt['value'])
                    else:
                        data[level]=np.nan
                else:
                    data[level]=np.nan
                        
            flood[level]=data[level] - ref
        flood['datum']=datum
        flood['datum_value']=ref
        flood['GT'] = gt
    else:
        print('Issue getting flood thresholds. Please check URL and station id')
    
    return flood


def join_sl_sbj(dfs_sl,df_sbj,
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
            ],
                info=False,
                convert=False
                ):
    out={}
    # Create copies to avoid modifying the original data
    # dfs_sl_c = dfs_sl.copy()
    # df_sbj_c = df_sbj.copy()
    for station_id in stations:
        # open dic
        df_dic_c = dfs_sl[station_id]# .copy()
        # get data
        df_sl = df_dic_c['df_sl']; 
        dic = df_dic_c['dic']
        # if convert:
            # datum_dif = msl_to_mhhw(station_id)
            # cols = [col for col in df_sl.columns if col.startswith('SL')]
            # for col in cols:
                # df_sl[col] = np.array(df_sl[col] - datum_dif)
            # df_dic_c['df_sl'] = df_sl
        # band pass winds
        # df_sl = bp_winds(df_sl)
        # print('band-passed winds')
        # join with SBj
        df = regress_sl_sbj(df_sbj,df_sl)
        # df_dic_c['df_reg'] = df
        thresholds = dic['flood_thresholds']; 
        if convert:
            thresholds =  noaaFlood(station_id,datum='MHHW')
            dic['flood_thresholds'] = thresholds
        ref_min = np.array(thresholds['nos_minor'])
        threshold = ref_min

        # compute daily maximas
        # daily_maxima = df.resample('D').max()
        daily_maxima = df.resample('1D').apply(lambda x: x.loc[x['SL_tg'].idxmax()])
        if info:
            print('{}: {}'.format(dfs_sl[station_id]['dic']['name'],
                len(daily_maxima[(daily_maxima['SL_tg'] > threshold)])))
        
        # compute days of flood, based on threshold
        
        n_flood_days = len(daily_maxima[(daily_maxima['SL_tg'] > threshold)])
        
        flood_days =  daily_maxima[ (daily_maxima['SL_tg'] > threshold) #| 
                                # (daily_maxima['height_nojet'] > ref_min) 
                                ]
        
        df_dic_c['n_flood_days'] = n_flood_days
        df_dic_c['flood_days'] = flood_days
        
        dic_flood = {}
        dic_flood_perc = {}
        
        n_flood_total = len(flood_days[flood_days['SL_tg']>ref_min])
        dic_flood['n_flood_total'] = n_flood_total
        # print(f'Out of the {n_flood_total} days of flood (SL_tg) that happened,\n how many of those would not be a flood if we:')
    
        # NOTE: we are saving the days of flood that WOULD NOT be flood anymore!
        for lb in [
    #         'SL_tg',
                   'SL_nojet','SL_nojet_nw','SL_nojet_w']:
            n_flood = n_flood_total- len(flood_days[flood_days[lb]>ref_min])
            perc = (n_flood*100)/n_flood_total
            dic_flood[lb] = n_flood
            dic_flood_perc[lb] = perc
        df_dic_c['flood_dic'] = dic_flood
        df_dic_c['flood_dic_perc'] = dic_flood_perc
        
        df_dic_c['daily_maxima'] = daily_maxima
        
        total_observations = len(daily_maxima)
        
        # compute probability of exceedance
        exceedances = daily_maxima[daily_maxima > threshold].count()
        df_dic_c['exceedance'] = exceedances
        
        # probability of raw data exceeding threshold:
        probability = exceedances['SL_tg'] / total_observations
        df_dic_c['prob_raw'] = probability*100
        prob_total = probability
        if info:
            print(f"The probability that Raw water height exceeds {threshold:.2f}m is {probability:.2%}")
        
        # probabilyt of SL no jet exceeding threshold
        probability = exceedances['SL_nojet'] / total_observations
        df_dic_c['prob_nojet'] = probability*100
        if info:
            print(f"The probability that Raw height - SBJ exceeds {threshold:.2f}m is {probability:.2%}")
        prob_sbj = probability
        
        prob_change = prob_total-prob_sbj
        prob_change = (prob_change * 100)/prob_total
        df_dic_c['prob_change'] = prob_change
        if info:
            print(f'Probability of change by removing the SBJ {prob_change:.2f}%')
            
            print('Days of flood that WOULD not be flood:')
            for lb in [
        #         'SL_tg',
                       'SL_nojet','SL_nojet_nw','SL_nojet_w']:
                print('Remove {} : {} ({:.2f}%)'.format(lb,dic_flood[lb],dic_flood_perc[lb]))
        # add again now with all the new computation to the dict
            print('')
        out[station_id] = df_dic_c
        print(station_id)
    return out

def bp_winds(df,u_name='u_raw',v_name='v_raw',
                   lowcut = 1.0 / (15.0 * 24), # 1/15 days in hours
                   highcut = 1.0 / 24  # 1 day in hours
             ):
    df['u'] = butter_pass(np.array(df[u_name]),lowcut,highcut)
    df['v'] = butter_pass(np.array(df[v_name]),lowcut,highcut)
    
    return df

def get_bathy():
    # Bathymetry
    path = '/Users/carocamargo/Documents/data/gebco_2023/'
    file ='GEBCO_USeast.nc'
    bathy = xr.open_dataset(path+file)
    # dm = open_ERA5()
    # bathy.interp(lat=dm.latitude, lon=dm.longitude, method='nearest')
    # bathy = bathy.coarsen(lat=7,lon=7).mean() # coarsen
    return bathy

def open_ERA5(level='single',remove_land=True):
    
    path = '/Users/carocamargo/Documents/data/ERA5/'
    if level=='pressure':
        path = '/Users/carocamargo/Documents/data/ERA5/pressure_levels/'
        dm = xr.open_mfdataset(path+'*.nc')
        dm = dm.squeeze('pressure_level')
        u_name = 'u'
        v_name  = 'v'
    else:
        path = '/Users/carocamargo/Documents/data/ERA5/single_levels/'
        u_name = 'u10'
        v_name  = 'v10'

        dm = xr.open_mfdataset(path+'*.nc')
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
    
    if remove_land: # this takes a while
        bathy = get_bathy()
        bathy = bathy.interp(lat=ds.latitude, lon=ds.longitude, 
                             method='nearest')
        # mask land values based on bathymetry
        ds = ds.where(bathy.elevation<0,np.nan)
        ds = ds.drop_vars(['lat', 'lon'])
    return ds
#% %
def buoy_meta():
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
                   '44085':{'long_name':'Buzzards Bay'},
                   '44011':{'long_name':'Georges Bank'},
                   'BUZM3':{'long_name':'Buzzards Bay (B)'},
                   '44020':{'long_name':'Nantucket Sound'},
                   '44025':{'long_name':'Long Island'},
                   '44017':{'long_name':'Mountauk'},
                  }
    st = pd.read_excel('/Users/carocamargo/Documents/data/wave/stations_info.xlsx',header=0)
    
    st.set_index('id',inplace=True)
    
    stations_meta = st.T.to_dict()
    stations_meta = {str(key): value for key, value in stations_meta.items()}

    return stations_meta,metadata

import matplotlib.pyplot as plt
def read_buoy(path='/Users/carocamargo/Documents/data/wave/',
              station_id ='44008',
              plot=False,
              ):
    # https://www.ndbc.noaa.gov/
    stations_meta,metadata = buoy_meta()
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
    y0 = df['#YY'].min()
    yf = df['#YY'].max()
    df['Datetime'] = pd.to_datetime(df[['#YY', 'MM', 'DD', 'hh', 'mm']].astype(str).agg(' '.join, axis=1), format='%Y %m %d %H %M')
    
    if plot:
        varis = ['WDIR', 'WSPD', 
                 # 'GST', 
                 'WVHT', 'DPD',
               'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']
        fig = plt.figure(figsize=(15,10))
        x = df['Datetime']
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
            plt.ylabel(metadata[v]['units'])    
        plt.suptitle('Buoy station: {} (# {})'.format(stations_meta[station_id]['long_name'],station_id))
        plt.tight_layout()
        plt.show()
    df.set_index('Datetime',inplace=True)
    return df


def buoy_wind(station_id):
    df =  read_buoy(station_id = station_id)
    # select only varibles we want
    df = df[['WDIR','WSPD','GST']]
    # correct WDIR
    df['WDIR'].fillna(0, inplace=True)
    df['ANG'] = np.deg2rad(np.array(df['WDIR']).astype(int)) # - 180) % 360 
    df['WDIR'] = np.array(df['WDIR']).astype(int)
    df['WDIR'].replace(0,np.nan,inplace=True)
    df['WDIR'].replace(999,np.nan,inplace=True)
    
    
    df['U'] = -np.sin(df['ANG'])
    df['V'] = -np.cos(df['ANG'])
    
    # f = df.index
    # plt.plot(f[1::],np.diff(f)/60000000000);plt.ylabel('freq (minutes)');plt.ylim(0,360)
    # we have some times a frequency of 10 min, others of 1h or more
    # take hourly means
    df = df.resample('1H').mean()
    
    # Round down the minutes to the nearest hour
    # df.index = df.index.floor('H')
    df.reset_index(inplace=True)
    
    return df
def insitu_wind():
    #% % open wind from tgs
    path = '/Users/carocamargo/Documents/data/sealevel/NOAA/'
    file = 'tgs_wind_combined_2014_2023.nc'
    ds = xr.open_dataset(path+file)

    ds['angle_rad'] = np.deg2rad(ds['angle'])# - 180) % 360 

    # Calculate components of wind velocity (u, v) from wind speed and direction
    ds['u'] = - np.sin(ds['angle_rad'])
    ds['v'] = - np.cos(ds['angle_rad'])
    #% %
    dtg = pd.DataFrame({'Datetime':ds.time,
                         'ws':np.array(ds.speed[:,0]*0.54444), # knots to m/s
                         'u':np.array(ds.u[:,0]),
                         'v':np.array(ds.v[:,0]),
                         'g':np.array(ds.gust[:,0]),
                         'ang' : np.deg2rad(ds['angle'][:,0]),
                          'dir':np.array(ds.angle[:,0])
                         })
    lats = np.array(ds.lat)
    lons = np.array(ds.lon)
    
    #% % merge tg with buoy
    stations_meta,metadata = buoy_meta()
    # tide gauges
    df_speed = pd.DataFrame({'Datetime':ds.time})
    df_u = pd.DataFrame({'Datetime':ds.time})
    df_v = pd.DataFrame({'Datetime':ds.time})
    
    for i,station in enumerate(np.array(ds.name)):
        df_speed[station] = np.array(ds.speed[:,i]*0.54444) # knots to m/s
        df_u[station] = np.array(ds.u[:,i])
        df_v[station] = np.array(ds.v[:,i])
        
        
    for i,station_id in enumerate(['44008',
                        # '44097', # wave only
                         # '44085', # wave only
                        '44011',
                          'BUZM3',
                         # 'buzm3'
                         '44020',
                         '44025',
                         '44017'
                       ]):
        df =  buoy_wind(station_id.lower())
        
        # read_buoy(station_id =station_id.lower())
        # print(df_speed.columns)
        # lats.extend(stations_meta[station_id]['lat_N'])
        lats = np.append(lats, stations_meta[station_id]['lat_N'])
        lons = np.append(lons, -stations_meta[station_id]['lon_W'])
        
        df_speed = pd.merge(df_speed,df[['Datetime','WSPD']],on='Datetime',how='left')
        df_speed.rename(columns={'WSPD':stations_meta[station_id]['long_name']},inplace=True)
        
        df_u = pd.merge(df_u,df[['Datetime','U']],on='Datetime',how='left')
        df_u.rename(columns={'U':stations_meta[station_id]['long_name']},inplace=True)
        
        df_v = pd.merge(df_v,df[['Datetime','V']],on='Datetime',how='left')
        df_v.rename(columns={'V':stations_meta[station_id]['long_name']},inplace=True)
        
        
    df_speed.set_index('Datetime',inplace=True)
    df_u.set_index('Datetime',inplace=True)
    df_v.set_index('Datetime',inplace=True)
    
    
    print(df_speed.columns)
    
    stations = [s for s in df_speed.columns]
    #% %
    dw = xr.Dataset(data_vars={'speed':(('time','station'),np.array(df_speed)), # speed
                               'u_angle':(('time','station'),np.array(df_u)), # angle
                               'v_angle':(('time','station'),np.array(df_v)), # angle
                               'U':(('time','station'),np.array(df_speed)*np.array(df_u)), # vector (angle * speed)
                               'V':(('time','station'),np.array(df_speed)*np.array(df_v)), # vector (angle * speed)
                               
                               'lat':(('station'),lats),
                               'lon':(('station'),lons)
                                                          },
                    coords={'time':ds.time,
                            'station':stations}
                    )
    
    return dw

def find_EDof(window,N=1,M=1,overlap=True):
    '''
    Find the Equivalent Degrees of Freedom for Spectra, depending on the Window function
    Based on Table 5.5, page 479 - Thomson & Emiry, 2014
    Based on Lecutre notes (UCSD_SIOC 221A (Gille)): Tables of “equivalent degrees of freedom” for the frequency domain don’t actually work
for windowed time domain data. One should follow Perival and Walden to get a full derivation of the actual degrees of freedom for overlapping segments
http://pordlabs.ucsd.edu/sgille/sioc221a_f17/lecture16_notes.pdf

    inputs:
    window: str - window function
    N: int - number of data point in the time series
    M: half-width of the window in the time (or spatial) domain. 
    N!= M for the truncated periodogram. 
    
    '''
    if overlap:
        
        if window =='bartlett': # same as triangle
            edof = 16/9
        elif window=='boxcar':  # truncated
            edof = 4/3
        elif window =='sinc': # Daniell
            edof = 2*(N/M)
        elif window=='hann': # 'Hanning'
            edof = 36/19
        elif window =='hamming':
            edof = 1.8
        elif window=='flat':
            edof=1
        else:
             raise TypeError("Window not recognized")
    else:
        if window =='bartlett': # same as triangle
            edof = 3 #*(N/M)
        elif window=='boxcar':  # truncated
            edof = N/M
        elif window =='sinc': # Daniell
            edof = 2 #*(N/M)
        elif window=='parzen':
            edof = 3.708614 #*(N/M)
        elif window=='hann': # 'Hanning'
            edof = (8/3)# *(N/M)
        elif window =='hamming':
            edof = 2.5164 #*(N/M)
        elif window=='flat':
            edof=1
        else:
             raise TypeError("Window not recognized")
    return edof

def coherence_CI(n,nperseg,fs,level,window='hann',effective_size=False,
                 noverlap=128):
    alpha = np.round(1 - (level/100) ,2)
    if effective_size:
        n = n//fs
    EDoF = (n) // (nperseg) 
    if noverlap>0:
        EDoF = EDoF * find_EDof(window)
    # if EDoF<=1:
    #     EDoF = find_EDof(window,N=n,M=nperseg/2)
    # EDoF =  1.8 * EDoF
    DoF = 2 * EDoF
    
    ci = 1 - alpha**(2/(DoF-2))
    ci = 1 - alpha**(1/(EDoF-1))
    return ci

def spectral_rel(x, y, 
                 x_name='x',x_units='units',
                 y_name='x',y_units='units',
                 effective_size=False,
                 plot=True,
                 noverlap=128,
                 fs=24, nperseg=24*15, window='hann', detrend='linear',
                 level=95
                 ):
    """Computes the power spectrum, coherence and admittance of two time series using the cross-spectral density method.

    Args:
    x: The input time series.
    y: The output time series.
    fs: The sampling frequency.
    nperseg: The segment length.
    window: The window function.
    detrend: The detrending method.

    Returns:
    A tuple containing the admittance magnitude and phase, respectively.
    """

    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        val = np.isfinite(x+y)
        x = x[val]
        y = y[val]
    
    
    # Compute the cross-spectral density of x and y.
    f,Pxy = signal.csd(x, y, fs=fs, nperseg=nperseg, 
                        noverlap=noverlap,
                       window=window, detrend=detrend)

    # Compute the power spectral density of x.
    _,Pxx = signal.welch(x, fs=fs, nperseg=nperseg, 
                          noverlap=noverlap, 
                         window=window, detrend=detrend)

    # Compute the power spectral density of y.
    _,Pyy = signal.welch(y, fs=fs, nperseg=nperseg, 
                          noverlap=noverlap,  
                         window=window, detrend=detrend)

    # Compute coherence
    Cxy = abs(Pxy)**2/(Pxx*Pyy)

    # Calculate the admittance
    Axy = np.array(Pxy/Pxx)
    # Calculate the admittance magnitude.
    Axy_mag = np.array(np.abs(Pxy) / np.abs(Pxx))    
    # Calculate the admittance phase.
    # Axy_phase = np.arctan2(np.imag(Axy), np.real(Axy))
    # Compute pahse in degrees, trasnform it from 0 to 360
    Axy_phase = np.array(np.angle(Axy,deg=True))
    Axy_phase =  (Axy_phase + 360) % 360

    # remove freq 0
    idx = np.where(f>0)
    Pxx = Pxx[idx]
    Pxy = Pxy[idx]
    Cxy = Cxy[idx]
    Axy = Axy[idx]
    Axy_phase = Axy_phase[idx]
    Axy_mag = Axy_mag[idx]
    f = f[idx]
    fr = np.array(1/f)
    # Output
    spec ={'Pxx':Pxx,
         'Pxy':Pxy,
         'Cxy':Cxy,
         'Axy':Axy,
         'Axy_phase':Axy_phase,
         'Axy_mag':Axy_mag,
         'freq':fr,
         'f':f}
    # Normalize power spectrums:
    # Nxy = Pxy/np.sum(Pxy**2)
    # Nxx = Pxx/np.sum(Pxx**2)
    # Nyy = Pyy/np.sum(Pyy**2)

    # Confidence Level:
    spec['ci'] = coherence_CI(len(x),nperseg,fs,level,window=window,effective_size=effective_size,noverlap=noverlap, )
    if plot:
        nrow=3;ncol=1

        # plt.figure(figsize=(15,5))
        plt.figure(figsize=(15,15))

        ax = plt.subplot(nrow,ncol,1)
        # plt.loglog(1/f,Pxy,label='Pxy ({},{})'.format(x_name,y_name),alpha=0.5)
        plt.loglog(1/f,Pxx,label='Pxx ({})'.format(x_name),alpha=0.5)
        plt.loglog(1/f,Pyy,label='Pyy ({})'.format(y_name),alpha=0.5)

        plt.legend()
        plt.xlabel('frequency')
        plt.ylabel('PSD [units/freq]')
        plt.title('Power Spectrum')

        ax = plt.subplot(nrow,ncol,2)
        plt.semilogx(1/f,np.array(Cxy))
        ax.scatter(1/f,Cxy,alpha=0.8);#plt.plot(Cxy)
        plt.ylabel('Coherence [-]')
        plt.xlabel('Frequency')
        plt.title('Coherence')

        ax = plt.subplot(nrow,ncol,3)
        plt.semilogx(1/f,np.array(Axy),label='Input x = {}, output y = {}'.format(x_name,y_name))
        ax.scatter(1/f,Axy,alpha=0.8);#plt.plot(Cxy)
        plt.legend()
        plt.ylabel('Admittance [{} / {}]'.format(y_units,x_units))
        plt.xlabel('Frequency')
        plt.title('Admittance')

        plt.tight_layout()
        plt.show()

    return spec



def noaa_Download_loop(station_id, product,start_year, end_year, 
                       interval='1H',
                 datum = 'MSL',
                 fmt='json',
                 path='/Users/carocamargo/Documents/data/sealevel/NOAA/'):

    """
    
    Function to download data from NOAA tides & Currents
    
    Adapted from matlab script of Dr. Chris Piecuch
    https://github.com/christopherpiecuch/floodAR/blob/main/noaaSealevel.m
   
    Dependencies: 'Requests', 'json', library
    
    
    Parameters
    ----------
    station_id : int
        Station id (e.g., WHOI is 8447930).
    product: str
        Product values are air_gap, air_pressure, air_temperature, conductivity, 
        currents, currents_survey, currents_predictions, daily_mean, datums, 
        high_low, hourly_height, humidity, monthly_mean, one_minute_water_level, 
        predictions, salinity, visibility, water_level, water_temperature, ofs_water_level and wind 
        
    start_yr : int
        start year.
    end_yr : int
        end year of query.
    datum: str
        Reference variable of the water level. 
        Options: CRD ( Columbia River Datum);
                IGLD (International Great Lakes Datum)
                LWD (Great Lakes Low Water Datum)
                MHHW (Mean Higher High Water)
                MHW (Mean High Water)
                MTL (Mean Tide Level)
                MSL (Mean Sea Level) - Default
                MLW (Mean Low Water)
                MLLW (Mean Lower Low Water)
                NAVD (North American Verical Datum)
                STND (Station Datum)
    fmt: str
        format of data to be downloaded. 
        Options: json, csv (Default) or xml(no done yet)
    path: str
        path to folder to save files
    
    Example usage
    ----------
    station_id = "8447930"  # Station ID for WHOI, MA
    start_date = "20230101"
    end_date = "20230107"
    datum = 'MSL'
    fmt = 'json'
    r = noaaSeaLevel(station_id, start_date, end_date)   
    """
    import os
    import calendar
    path = path+product+'/'
    if not os.path.exists(path):
        os.makedirs(path)
     
    filename = path+"{}_{}_{}_{}_{}.csv".format(product,
        station_id,start_year,end_year,datum)
    
    # extract metadata
    metadata = noaaMetadata(station_id)
    
    # Start Loop over time:

    # time limit is 31 days
    if product=='wind':
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Determine the number of days in the current month
                num_days = calendar.monthrange(year, month)[1]
                
                # Construct start and end dates using the determined number of days
                start_date = '{}{}01'.format(year, str(month).zfill(2))
                end_date = '{}{}{}'.format(year, str(month).zfill(2), str(num_days).zfill(2))
                # print(end_date)
                # Extract data:
                data = noaaTidesCurrents(station_id, product,
                                         start_date, end_date,
                                         interval=interval,
                                         fmt='csv', datum=datum,
                                         info=False,
                                         save=False
                                     )
                # Save metadata and sea level data to a CSV file
                if year==start_year and month==1: # make a new csv file
                
                    with open(filename, "w", newline="") as f:
                        
                        ## Write metadata
                        for key in metadata.keys():
                            f.write('#{} \n'.format(key))
                            f.write(metadata[key]+'\n')
                            
                        # Write sea level data
                        f.write(data)
                    
                else: # append to existing file
                    with open(filename, "a", newline="") as f:
                        f.write(data[36:]) # skip headers
    
        print("Data downloaded successfully!")
    else:
        # time limit is 1 year
        for year in range(start_year,end_year+1): # we can only download 1 year per time
            start_date = '{}0101'.format(year)
            end_date = '{}1231'.format(year)
            
            data = noaaTidesCurrents(station_id, product,
                                             start_date, end_date,
                                             interval=interval,
                                             fmt='csv', datum=datum,
                                             info=False,
                                             save=False
                                             )
            
            # Save metadata and sea level data to a CSV file
            if year==start_year: # make a new csv file
                
                with open(filename, "w", newline="") as f:
                    
                    ## Write metadata
                    for key in metadata.keys():
                        f.write('#{} \n'.format(key))
                        f.write(metadata[key]+'\n')
                        
                    # Write sea level data
                    f.write(data)
                    
            else: # append to existing file
                with open(filename, "a", newline="") as f:
                    f.write(data[36:]) # skip headers
    
        print("Data downloaded successfully!")
    
    return



def compute_tide(time,u,v=None,lat=None,
                 nodal=False,
                 constit=None,
                 detrend=True,
                 method='ols',
                 conf_int='linear',
                 Rayleigh = 1,
                 verbose=False
                 ):
    from utide import solve, reconstruct
    
    if v:
        if constit:
            constit = ['M2', 'N2', 'K1', 'O1', 'S2', 'SA', 'P1', 'M4', 'NU2', '2N2', 'Q1',
                    'L2', 'K2', 'MN4', 'M6', 'MSM', 'MF', 'SSA', 'MU2', 'MSF', '2MN6',
                    'MM', 'LDA2', 'MS4', 'NO1', 'J1', 'T2', '2MK5', 'OO1', 'RHO1',
                    '2MS6', 'H1', 'S1', 'SIG1', 'GAM2', 'M3', 'MK3', 'H2', 'PI1',
                    'OQ2', 'EPS2', 'CHI1', 'MK4', 'PHI1', 'S4', '2Q1', 'PSI1', 'M8',
                    'SO3', 'SO1', 'MSN2', 'THE1', '2MK6', 'MO3', 'BET1', 'TAU1',
                    'UPS1', 'SK4', 'SN4', 'SK3', '2SK5', 'ALP1', '3MK7', 'ETA2',
                    '2SM6', 'R2', 'MKS2', 'MSK6']
            coef = solve(time,u,v,
                        lat = lat, 
                        nodal=nodal,
                        constit=constit,
                        trend=detrend,
                        method=method,
                        conf_int=conf_int,
                        Rayleigh_min=Rayleigh,
                        verbose=verbose,
                       )
        else:
            coef = solve(time,u,v,
                        lat = lat, 
                        nodal=nodal,
                        # constit=constit,
                        trend=detrend,
                        method=method,
                        conf_int=conf_int,
                        Rayleigh_min=Rayleigh,
                        verbose=verbose,
                       )

    else:
        if constit:
            constit = ['M2', 'N2', 'K1', 'O1', 'S2', 'SA', 'P1', 'M4', 'NU2', '2N2', 'Q1',
                    'L2', 'K2', 'MN4', 'M6', 'MSM', 'MF', 'SSA', 'MU2', 'MSF', '2MN6',
                    'MM', 'LDA2', 'MS4', 'NO1', 'J1', 'T2', '2MK5', 'OO1', 'RHO1',
                    '2MS6', 'H1', 'S1', 'SIG1', 'GAM2', 'M3', 'MK3', 'H2', 'PI1',
                    'OQ2', 'EPS2', 'CHI1', 'MK4', 'PHI1', 'S4', '2Q1', 'PSI1', 'M8',
                    'SO3', 'SO1', 'MSN2', 'THE1', '2MK6', 'MO3', 'BET1', 'TAU1',
                    'UPS1', 'SK4', 'SN4', 'SK3', '2SK5', 'ALP1', '3MK7', 'ETA2',
                    '2SM6', 'R2', 'MKS2', 'MSK6']
            coef = solve(time,u,
                        lat = lat, 
                        nodal=nodal,
                        constit=constit,
                        trend=detrend,
                        method=method,
                        conf_int=conf_int,
                        Rayleigh_min=Rayleigh,
                        verbose=verbose,
                       )
        else:
            coef = solve(time,u,
                        lat = lat, 
                        nodal=nodal,
                        # constit=constit,
                        trend=detrend,
                        method=method,
                        conf_int=conf_int,
                        Rayleigh_min=Rayleigh,
                        verbose=verbose,
                       )        
    tide = reconstruct(time, coef, 
                         verbose=verbose
                         )
    return coef, tide
#%%%%%% IB functions
def open_slp(path = '/Users/carocamargo/Documents/data/NCEP/slp_ocean_mean_4xday2hourly_1990-2023.nc',
             ds_tg=None,time_slice=False):
    
    dslp = xr.open_dataset(path)
    if time_slice:
        dslp = dslp.sel(time=slice(ds_tg.time[0],ds_tg.time[-1]))
    return dslp

#% % IB functions
def compute_ib_local(pa,g = 9.8,rho = 1025):
    # Local IB effect  - Wunsch & Stammer, 1997 https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/96RG03037
    nib = -(pa/(rho*g))
    return nib

def compute_ib(pa,pamean,g = 9.8,rho = 1025):
    # IB response - Ponte, 2006 https://journals.ametsoc.org/view/journals/atot/23/4/jtech1864_1.xml
    
    nib = -(np.array(pa-pamean)/(rho*g))
    return nib
#%% recalculated nib and lib wih the new atmospheric pressure:
def add_new_ib(ds):
    nib = np.zeros((len(ds.time),len(ds.station)))
    lib = np.zeros((len(ds.time),len(ds.station)))

    
    for i in range(len(ds.station)):
        nib[:,i] = compute_ib(np.array(ds.atm_pressure_filled[:,i])*100,ds.mean_pa) # pressures is in mbar = 100 kg/ms2
        lib[:,i] = compute_ib_local(np.array(ds.atm_pressure_filled[:,i])*100) # pressures is in mbar = 100 kg/ms2

    # ds['nib_old']=ds['nib']
    if 'nib' in ds.data_vars:
        ds = ds.drop_vars(['nib'])
    ds['nib']=(('time','station'),nib)
    ds['nib'].attrs = {'units':'m',
                       'long_name':'Inverted barometer contribution to MSL',
                       'equation':'nib = - pa-pamean / rho*g',
                       'comment':'computed with atm_pressure_filled variable',
                       'source':'pamean is computed from NCEP renalysis'
                       }
    ds['lib']=(('time','station'),lib)
    ds['lib'].attrs = {'units':'m',
                       'long_name':'Local inverted barometer contribution to MSL',
                       'equation':'nib = - pa / rho*g',
                       'comment':'computed with atm_pressure_filled variable'
    
                       }
    return ds
#%% distance between stations
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from scipy import stats
def get_distance(c1 = [40,-70],
             c2 = [41,-71],
             unit='km'):
    # c1 = [np.array(ds.lat[i]),np.array(ds.lon[i])]
    
    c1_in_radians = [radians(_) for _ in c1]
    c2_in_radians = [radians(_) for _ in c2]
    result = haversine_distances([c1_in_radians, c2_in_radians])
    if unit=='km':
        dist = result[1][0] * 6371000/1000
    else:
        dist = result[1][0] * 6371000
    return dist

def cor_distance(ds, method='pearson',plot=True,save=False, 
                 path_to_figures='./',fname = 'IB_cor_dist_map.png'):
    cor = np.zeros((len(ds.station),len(ds.station)))
    p = np.zeros((len(ds.station),len(ds.station)))
    distances = np.zeros((len(ds.station),len(ds.station)))
    for j in range(len(ds.station)):
        # distances = np.zeros((len(ds.station)))
        x = np.array(ds.atm_pressure[:,j])
        for i in range(len(ds.station)):
            distances[j,i] = get_distance(c1 = [ds.lat[j],ds.lon[j]],
                                        c2 = [ds.lat[i],ds.lon[i]])
            y = np.array(ds.atm_pressure[:,i])
            accx = np.isfinite(y+x)
            if np.all(accx==False):
                cor[j,i]=np.nan;p[j,i]=np.nan
            else:
                # cors[j,i],p[j,i] = stats.pearsonr(x[accx], y[accx])
                if method =='pearson':
                    cor[j,i],p[j,i] = stats.pearsonr(x[accx], y[accx])
                elif method =='spearman':
                    cor[j,i],p[j,i] = stats.spearmanr(x[accx], y[accx])
                elif method =='kendall':
                    cor[j,i],p[j,i] = stats.kendalltau(x[accx], y[accx])
                else:
                    print('correlation method not identified')
    if plot:
        c=np.array(cor)
        p=np.array(p)
        corsquare = np.ones((cor.shape))
        for i in range(len(ds.station)):
            x=np.array(c[i,:])
            # print(x)
            # x = np.delete(x,i)
            df = pd.DataFrame({'value':x,'dist':distances[i,:]})
            df = df.sort_values('dist')
            corsquare[i,:] = np.array(df.value)
        #% %
        pmask = np.ones((p.shape))
        pmask[p>0.05]=np.nan
        #% %
        fig = plt.figure(
            figsize=(15,15))
        ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3,colspan=1)
        plt.pcolor(distances[0,:],ds.name,corsquare*pmask,
                    vmin=0.5,vmax=1,
                   # cmap=cmocean.cm.dense
                   )
        # plt.colorbar(label='correlation')
        plt.xlabel('Distance to TG')
        plt.title('{} pairwise correlation'.format(method))
        plt.yticks(fontsize=10)
        # plt.show()
        #% %
        ax = plt.subplot2grid((4, 1), (3, 0), colspan=1)
        
        for j in range(len(ds.station)):
            
            plt.scatter(distances[j,:],np.array(cor)[j,:],
                        label=np.array(ds.name[j]),
                        marker='o',
                        alpha=0.5,
                        c='gray'
                        )
            
            # plt.title(np.array(ds.name[j]))
        plt.xlabel('distance to TG (km)')
        # plt.axvline(200,linestyle='--',c='black')
        plt.ylabel('Correlation')
        plt.title('Pair-wise correlation - ATM')
        plt.tight_layout()
        plt.show()
        #% %
        # if save:
        #     kurs=path_to_figures+'IB_cor_dist.png'
            
        #     fig.savefig(kurs,format='png',dpi=300,bbox_inches='tight')
        
        
        # pmask[np.isfinite]
        fig = plt.figure(figsize=(10,10))
        
        #% %
        if save:
            kurs=path_to_figures+fname
            
            fig.savefig(kurs,format='png',dpi=300,bbox_inches='tight')
        
    return cor, p, distances

#%% include stations that are within 200km to make regional average
def reg_composite(ds,distances,plot1=False,info=True,plot2=False):
    x = np.array(ds.atm_pressure)
    regional_average = np.zeros((x.shape))
    atm_completed = np.array(ds.atm_pressure)
    i = 0
    for i in range(len(ds.station)):
        dist_to_tg = np.array(distances[i,:])
        inds = np.where(dist_to_tg<200)[0]
        local_mean = np.nanmean(x[:,inds],axis=1)
        regional_average[:,i] = local_mean
        
        y = np.array(ds.atm_pressure[:,i])
        len_missing = len(y[np.isnan(y)])
        y[np.isnan(y)] = regional_average[np.isnan(y),i]
        atm_completed[:,i] = np.array(y)
        if info:
            print('\nFor {}, regional averages of: '.format(np.array(ds.name[i])))
            print(np.array(ds.name[inds]))
            print('Filled {} out of {} ({} %) \n'.format(len_missing, len(y), 
                                                      len_missing/len(y) * 100
                                                      ))
            
        if plot1:
            plt.figure()
            ax = plt.subplot()
            plt.plot(local_mean,label = 'regional average',
                     # alpha=0.5,
                     linestyle='--',c='black')
            plt.title(np.array(ds.name[i]))
            for ii in inds:
                plt.plot(x[:,ii],alpha=0.5,label=np.array(ds.name[ii]))
        
            # plt.legend(ncol=3,loc ='lower right')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
                      ncol=3)
            plt.show()
    if plot2:
        xaxis = []
        plt.figure(figsize=(25,25))
        for i in range(len(ds.station)):
            x = np.array(atm_completed[:,i] - np.nanmean(atm_completed[:,i])
                         )
            xaxis.append(i*50)
            # x[np.isfinite(x)]=1*i
            # duration[:,i] = x
            plt.plot(ds.time,x+(i*50),label=np.array(ds.station[i]),alpha=0.5,c='C{}'.format(int(i)))  
            
            x = np.array(ds.atm_pressure[:,i] - ds.atm_pressure[:,i].mean())

            plt.plot(ds.time,x+(i*50),label=np.array(ds.station[i]),alpha=1,c='C{}'.format(int(i)))  
            
        plt.ylim(np.min(xaxis)-50,np.max(xaxis)+50)
        # plt.legend()

        plt.title('Atmospheric Pressure - completed',fontsize=20)
        plt.yticks(xaxis,np.array(ds.name),rotation=0,fontsize=15)
        plt.tight_layout()
        plt.show()
    return atm_completed, regional_average

#%% add completed record to ds
def complete_ds(ds, atm_completed,mpa):
    ds['atm_pressure_filled'] = (('time','station'),atm_completed)
    ds['atm_pressure_filled'].attrs = ds.atm_pressure.attrs
    ds['atm_pressure_filled'].attrs['comment']='Missing values replaced by regional average of measurements within 200km from station.'

    #% % add mean pa to ds
    ds['mean_pa'] = (('time'),mpa)
    ds['mean_pa'].attrs ={'long_name':'Mean Atmospheric Pressure over global oceans',
                          'units':'mbar',
                          'source':'NCEP Reanalysis I',
                          'time_frequency': 'original reanalysis was 4x day, downsampled to hourly'
                          }
    return ds

#%%
def cut_ds(ds,bbox,lon_name='lon',lat_name='lat'):
    lon_min,lon_max,lat_min,lat_max = bbox
    mask = (
        (ds[lon_name] <= lon_max)
        & (ds[lon_name] >= lon_min)
        & (ds[lat_name] <= lat_max)
        & (ds[lat_name] >= lat_min)
    ).compute()
    return ds.where(mask, drop=True)

def find_closest_cell_lm(ds, lat1, lon1,remove_nan=True,mask=True):
    """
    Find the index of the closest latitude and longitude to the given coordinates.
    
    Parameters:
    ds (xarray.Dataset): The dataset containing 'latitude' and 'longitude' coordinates.
    lat1 (float): The latitude of the target point.
    lon1 (float): The longitude of the target point.
    
    Returns:
    tuple: The index of the closest latitude and longitude.
    """
    
    if remove_nan:
        # Drop latitudes and longitudes where u and v are NaN across all time steps
        # ds = ds.dropna(dim='latitude', how='all', subset=['u', 'v'])
        # ds = ds.dropna(dim='longitude', how='all', subset=['u', 'v'])
        # this is not working, so we can create a mask instead
        if mask:
            mask = np.array(ds.mask)
            # mask[np.isfinite(mask)]=np.nan
        elif 'time' in ds.coords:
            mask = np.array(ds['u'].mean(dim='time')*ds['v'].mean(dim='time'))
            
        else:
            mask = np.array(ds['u']*ds['v'])
        mask[np.isfinite(mask)]=1
        

    # Extract latitude and longitude from the dataset
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    # Convert latitude and longitude to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lats_rad = np.radians(lats[:, np.newaxis])
    lons_rad = np.radians(lons)

    # Haversine formula to calculate distance between two lat/lon points
    dlat = lats_rad - lat1_rad
    dlon = lons_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lats_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    R = 6371  # Earth radius in kilometers
    distances = R * c
    
    if remove_nan:
        distances[np.isnan(mask)] = 999999
    # Find the index of the minimum distance
    min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
    
    return min_dist_idx

def find_wind_lm(lat,lon,dx):
    
    ilat,ilon = find_closest_cell_lm(dx, lat,lon)
    u= np.array(dx['u'][:,ilat,ilon])
    v = np.array(dx['v'][:,ilat,ilon])
    return u, v

