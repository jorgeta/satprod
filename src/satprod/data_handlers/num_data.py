import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

from satprod.data_handlers.data_utils import sin_transform, cos_transform

from tasklog.tasklogger import logging

def read_wind_data(path):
    '''
    Read and concatenate wind speed and wind direction NWP prediction data.
    The data is based on NWP (Numerical Weather Predictions), where the column
    'date_calc' is the time of prediction, and 'date_forecast' is the time of
    prediction. The wind direction given in degrees (0-360), is transformed into 
    cos and sin values (0 deg -> cos is 1, sin is 0, 180 deg -> cos is -1, sin is 0)

    Returns a dataframe containing wind speed and wind direction for each park.
    '''
    logging.info('Reading wind data.')

    # treat inf values as NaN
    pd.options.mode.use_inf_as_na = True

    # read weather data
    bess_weather = pd.read_csv(f'{path}/bess_weather.csv')
    skom_weather = pd.read_csv(f'{path}/skom_weather.csv')
    vals_weather = pd.read_csv(f'{path}/vals_weather.csv')
    yvik_weather = pd.read_csv(f'{path}/yvik_weather.csv')
    
    # time columns to datetime objects
    bess_weather['date_calc'] = pd.to_datetime(bess_weather['date_calc'])
    bess_weather['date_forecast'] = pd.to_datetime(bess_weather['date_forecast'])

    # rename to distinguish parks
    bess_weather.columns = ['date_calc', 'date_forecast', 'wind_speed_bess', 'wind_direction_bess']
    skom_weather.columns = ['date_calc_del', 'date_forecast_del', 'wind_speed_skom', 'wind_direction_skom']
    vals_weather.columns = ['date_calc_del', 'date_forecast_del', 'wind_speed_vals', 'wind_direction_vals']
    yvik_weather.columns = ['date_calc_del', 'date_forecast_del', 'wind_speed_yvik', 'wind_direction_yvik']

    # concatenate data
    df = pd.concat([bess_weather, skom_weather, vals_weather, yvik_weather], axis=1, join="inner")

    # delete excess time columns
    df = df.drop(columns=['date_calc_del', 'date_forecast_del'])

    # treat values above limit to be unrealistic, and tranform them to NaN
    limit = 400
    assert limit > 360, 'In order to not set degree (direction) values to NaN'
    df_num = df[
        ['wind_speed_bess', 'wind_direction_bess',
       'wind_speed_skom', 'wind_direction_skom', 
       'wind_speed_vals', 'wind_direction_vals', 
       'wind_speed_yvik', 'wind_direction_yvik']
       ].apply(lambda x: [y if y <= limit else np.nan for y in x])

    # interpolate to remove a few inf and nan values
    df_num = df_num.interpolate(method ='linear', limit_direction ='forward')
    
    df = pd.concat([df[['date_calc', 'date_forecast']], df_num], axis=1)

    def direction_transforms(wind_df, park: str):
        wind_df[f'wind_direction_cos_{park}'] = cos_transform(wind_df[f'wind_direction_{park}'].values)
        wind_df[f'wind_direction_sin_{park}'] = sin_transform(wind_df[f'wind_direction_{park}'].values)
        wind_df = wind_df.drop(columns=[f'wind_direction_{park}'])
        return wind_df

    parks = ['bess', 'skom', 'vals', 'yvik']
    for park in parks:
        df = direction_transforms(df, park)

    logging.info('Done reading wind data.')
    return df

def read_production_data(path: str):
    '''
    Read and concatenate production data.

    Returns a dataframe containing hourly production at each park.
    '''
    logging.info('Reading production data.')

    # read raw production data
    bess_prod = pd.read_csv(f'{path}/bess_prod.csv')
    skom_prod = pd.read_csv(f'{path}/skom_prod.csv')
    vals_prod = pd.read_csv(f'{path}/vals_prod.csv')
    yvik_prod = pd.read_csv(f'{path}/yvik_prod.csv')
    
    def format_production_data(df, tag: str):
        '''
        Sets index to datetime, and renames the columns so it is identifyable
        where the data comes from.

        Returns dataframes with said changes.
        '''

        df.time = pd.to_datetime(df.time)
        df = df.set_index(['time'])
        df.columns = [df.columns.values[0] + f'_{tag}']
        return df
    
    # concatenate formatted production data
    df = pd.concat([
        format_production_data(bess_prod, 'bess'), 
        format_production_data(skom_prod, 'skom'),
        format_production_data(vals_prod, 'vals'),
        format_production_data(yvik_prod, 'yvik')
    ], axis=1)

    # due to clearly untrue values, certain production values at yvik are removed
    df.production_yvik.loc['2019-12-16 10:00:00':] = np.nan
    
    logging.info('Done reading production data.')
    return df

def format_numeric_data(path: str, pred_limit: int = 6):
    '''
    Reads wind and production data.
    Converts the wind data to hourly data, ignoring the variation in how recent the predictions are.
    The data is split into two dataframes:
     - enc_df
        - for input to encoder
        - given a time (row), no future predictions are included
     - dec_df
        - for input to decoder
        - given a time (row), only future wind speed and wind direction data is included,
          until the pred_limit (default 6 hours) number of hours ahead.
    The production data is not forecasted values, and in therefore concatenated with the enc_df.

    Returns enc_df and dec_df as they are described above.
    '''

    # read wind and prod data
    wind_df = read_wind_data(path)
    prod_df = read_production_data(path)
    
    # stop writing warning to console concerning potential copying error not relevant in this case
    pd.options.mode.chained_assignment = None

    logging.info('Formatting data.')
    
    # convert the date_forecast columns to contain an integer represeting number of hours ahead of calculation
    wind_df['date_forecast'] = wind_df['date_forecast'] - wind_df['date_calc']
    timelist = wind_df['date_forecast'].dt.components
    wind_df['date_forecast'] = timelist['days'].values*24+timelist['hours'].values

    # rename date_calt to time
    wind_df = wind_df.rename(columns={'date_calc': 'time'})

    # add date_forecast column as a part of the columns indexing
    wind_df = wind_df.pivot(index='time', columns='date_forecast')
    
    # insert empty rows for the hours without data
    wind_df = wind_df.asfreq(freq='H')

    # in case of more than pred_limit rows with nan values, we only want to fill up to pred_limit
    #nan_row_counter = 0

    logging.info('Filling wind prediction dataframe using forecasts.')

    for j in range(len(wind_df.columns.levels[0].values)):
        arr = wind_df[str(wind_df.columns.levels[0].values[j])].values

        for g in range(1, len(arr)):
            for h in range(len(arr[g])-1):
                if np.isnan(arr[g][h]):
                    arr[g][h] = arr[g-1][h+1]

        wind_df[str(wind_df.columns.levels[0].values[j])] = arr

        logging.info('Done with', wind_df.columns.levels[0].values[j])

    logging.info('Done filling wind prediction dataframe.')
    logging.info('Creating new dataframes.')
    # data input to decoder
    dec_df = wind_df.copy()
    for col in dec_df.columns:
        if col[1] > pred_limit or col[1] < 1:
            dec_df = dec_df.drop(columns=col)
    dec_df.columns = [''.join(str(col[0])+'+'+str(int(col[1]))+'h').strip() for col in dec_df.columns.values]

    # data input to encoder
    enc_df = wind_df.copy()
    for col in enc_df.columns:
        if col[1] > 0:
            enc_df = enc_df.drop(columns=col)
    enc_df.columns = [col[0] for col in enc_df.columns.values]

    # concat wind data and production data and store in encoder dataframe
    enc_df = pd.concat([enc_df, prod_df], axis=1)

    logging.info('Done formatting data.')

    return enc_df, dec_df

def write_formatted_data_inc_nan(enc_df_inc_nan, dec_df_inc_nan, path: str):
    '''
    Write enc_df and dec_df to file, the versions where missing data is not filled.
    '''
    logging.info('Writing formatted data to file.')

    enc_df_inc_nan.to_csv(f'{path}/enc_df_inc_nan.csv')
    dec_df_inc_nan.to_csv(f'{path}/dec_df_inc_nan.csv')

def write_formatted_data_no_nan(enc_df_no_nan, dec_df_no_nan, path: str):
    '''
    Write enc_df and dec_df to file, the versions where missing data is filled.
    '''
    logging.info('Writing formatted data to file.')

    enc_df_no_nan.to_csv(f'{path}/enc_df_no_nan.csv')
    dec_df_no_nan.to_csv(f'{path}/dec_df_no_nan.csv')

def read_formatted_data_inc_nan(path: str):
    '''
    Read enc_df_inc_nan and dec_df_inc_nan from file.

    Returns enc_df_inc_nan and dec_df_inc_nan.
    '''

    enc_df_inc_nan = pd.read_csv(f'{path}/enc_df_inc_nan.csv')
    enc_df_inc_nan['time'] = pd.to_datetime(enc_df_inc_nan['time'])
    enc_df_inc_nan = enc_df_inc_nan.set_index(['time'])
    
    dec_df_inc_nan = pd.read_csv(f'{path}/dec_df_inc_nan.csv')
    dec_df_inc_nan['time'] = pd.to_datetime(dec_df_inc_nan['time'])
    dec_df_inc_nan = dec_df_inc_nan.set_index(['time'])

    return enc_df_inc_nan, dec_df_inc_nan

def read_formatted_data_no_nan(path: str):
    '''
    Read enc_df_no_nan and dec_df_no_nan from file.

    Returns enc_df_no_nan and dec_df_no_nan.
    '''

    enc_df_no_nan = pd.read_csv(f'{path}/enc_df_no_nan.csv')
    enc_df_no_nan['time'] = pd.to_datetime(enc_df_no_nan['time'])
    enc_df_no_nan = enc_df_no_nan.set_index(['time'])
    
    dec_df_no_nan = pd.read_csv(f'{path}/dec_df_no_nan.csv')
    dec_df_no_nan['time'] = pd.to_datetime(dec_df_no_nan['time'])
    dec_df_no_nan = dec_df_no_nan.set_index(['time'])

    return enc_df_no_nan, dec_df_no_nan

def format_data():
    '''
    Handles formatting of data and storing it.
    '''

    cd = str(os.path.dirname(os.path.abspath(__file__)))
    root = f'{cd}/../../..'
    enc_df_inc_nan, dec_df_inc_nan = format_numeric_data(f'{root}/data/num')
    path = f'{root}/data/formatted'
    write_formatted_data_inc_nan(enc_df_inc_nan, dec_df_inc_nan, path)

if __name__=='__main__':
    format_data()
    