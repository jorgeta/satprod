import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import pickle
from sklearn.linear_model import LinearRegression
from satprod.configs.config_utils import read_yaml

from satprod.data_handlers.data_utils import (
    sin_transform, 
    cos_transform, 
    get_columns,
    cos_temporal_transform,
    sin_temporal_transform
)

from tasklog.tasklogger import logging

class NumericalDataHandler():

    def __init__(self):
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'
        self.config = read_yaml(f'{self.root}/config.yaml')
        self.parks = ['bess', 'skom', 'vals', 'yvik']
        self.raw_prod_data_path = f'{self.root}/data/measured_prod'
        self.raw_wind_data_path = f'{self.root}/data/weather_forecasts'
        self.old_raw_data_path = f'{self.root}/data/old_data/num'
        self.old_formatted_data_path = f'{self.root}/data/old_data/formatted'
        self.formatted_data_path = f'{self.root}/data/formatted'
        self.benchmark_data_path = f'{self.root}/data/prod_forecasts'
        
        os.makedirs(f'{self.raw_prod_data_path}', exist_ok=True)
        os.makedirs(f'{self.raw_wind_data_path}', exist_ok=True)
        os.makedirs(f'{self.old_raw_data_path}', exist_ok=True)
        os.makedirs(f'{self.formatted_data_path}', exist_ok=True)
        
        self.pred_horizon = self.config.data_config.pred_sequence_length
        self.valid_start_date = datetime(**self.config.data_config.valid_start)
        self.test_start_date = datetime(**self.config.data_config.test_start)
        self.test_end_date = datetime(**self.config.data_config.test_end)
        
        # treat inf values as NaN
        pd.options.mode.use_inf_as_na = True
        
        # stop writing warning to console concerning potential copying error not relevant in this case
        pd.options.mode.chained_assignment = None
    
    def update_formatted_files(self):
        wind_df = self.get_wind_data()
        
        prod_df = self.get_prod_data()
        logging.info('Removing first half year of production data at the different parks.')
        
        start_dates = {
            'bess': '2009-01-23 00:00:00', 
            'skom': '2016-09-08 00:00:00', 
            'vals': '2007-04-28 00:00:00',
            'yvik': '2015-10-16 10:00:00'
        }

        for park in self.parks:
            date = datetime.strptime(start_dates[park], '%Y-%m-%d %H:%M:%S')-timedelta(hours=1)
            prod_df[f'production_{park}'].loc[:date] = np.nan
        
        logging.info('Replacing outliers with NaN.')
        prod_df = self.__clean_prod_data(prod_df)
        
        df = pd.concat([wind_df, prod_df], axis=1).asfreq('H')
        
        # add temporal features
        df['temporal_cos'] = cos_temporal_transform(df.index.dayofyear.values)
        df['temporal_sin'] = sin_temporal_transform(df.index.dayofyear.values)
        for col in get_columns(df, 'temporal').columns:
            for i in range(1, self.pred_horizon+1):
                df[f'{col}+{i}h'] = df[col].shift(periods=-i)
        
        self.write_formatted_data(df, nan=True)
        
        # fill isolated missing values
        prod_df = self.__fill_missing_prod_values(prod_df, wind_df)
        df = pd.concat([wind_df, prod_df], axis=1).asfreq('H')
        
        # add temporal features
        df['temporal_cos'] = cos_temporal_transform(df.index.dayofyear.values)
        df['temporal_sin'] = sin_temporal_transform(df.index.dayofyear.values)
        for col in get_columns(df, 'temporal').columns:
            for i in range(1, self.pred_horizon+1):
                df[f'{col}+{i}h'] = df[col].shift(periods=-i)
        
        self.write_formatted_data(df, nan=False)
    
    def write_formatted_data(self, df: pd.DataFrame, nan: bool):
        filename = 'df_nan' if nan else 'df_filled'
        df.to_csv(f'{self.formatted_data_path}/{filename}.csv')
        
    def read_formatted_data(self, nan: bool=False) -> pd.DataFrame:
        filename = 'df_nan' if nan else 'df_filled'
        df = pd.read_csv(f'{self.formatted_data_path}/{filename}.csv')
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index(['time'])
        return df
    
    def format_benchmark_data(self):
        park_dfs = {}
        for park in self.parks:
            df = pd.read_csv(f'{self.benchmark_data_path}/{park}_prod_forecasts.csv')
            
            df['time'] = pd.to_datetime(df['pred_time'])
            df = df.drop(columns=['pred_time', 'Unnamed: 0'])
            df = pd.pivot_table(df, values='predicted',index='time',columns='horizon')
            
            for col in df.columns:
                if col > 5:
                    df = df.drop(columns=col)
            
            df = df.loc[self.test_start_date:self.test_end_date]
            
            df.columns = [str(i) for i in range(self.pred_horizon+1)]
        
            path = f'{self.root}/storage/{park}/TE'
            os.makedirs(path, exist_ok=True)
            with open(f'{path}/TE_predictions.pickle', 'wb') as pred_file:
                best_model = pickle.dump(df, pred_file)
    
    def get_wind_data(self) -> pd.DataFrame:
        weather = {}
        for park in self.parks:
            if park=='skom': continue
            weather[park] = pd.read_csv(f'{self.raw_wind_data_path}/{park}_weather_forecasts.csv')
        
        # time columns to datetime objects
        weather[self.parks[0]]['validdate'] = pd.to_datetime(
            weather[self.parks[0]]['validdate']
        )
        
        # rename to distinguish parks
        weather[self.parks[0]].columns = [
            'time', f'wind_speed_{self.parks[0]}', f'wind_direction_{self.parks[0]}']
        for park in self.parks[2:]:
            weather[park].columns = ['time_del', f'wind_speed_{park}', f'wind_direction_{park}']
        
        # concatenate data
        df = pd.concat(list(weather.values()), axis=1, join="inner")
        
        # delete excess time columns
        df = df.drop(columns=['time_del'])
        
        # set time columns as datetimeindex
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index(['time'])
        
        # fill in NaN values where there are missing values
        df = df.asfreq(freq='H')
        
        # remove time zone information since all data is in UTC
        df = df.tz_localize(None)
        
        # convert wind direction in degrees into two features representing cosine and sine of the degrees
        for park in self.parks:
            if park=='skom': continue
            df = self.__direction_transforms(df, f'wind_direction_{park}')
        
        # add forecast features ('wind_speed_bess+1h', 'wind_direction_bess_cos+1h', ...)
        df = self.__add_forecasts_to_wind_data(df)
        
        return df
    
    def __add_forecasts_to_wind_data(self, df: pd.DataFrame) -> pd.DataFrame():
        for col in df.columns:
            for i in range(1, self.pred_horizon+1):
                df[f'{col}+{i}h'] = df[col].shift(periods=-i)
        return df
        
    def __direction_transforms(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        assert column in df.columns, f'{column} needs to be a column in the input dataframe.'
        
        df[f'{column}_cos'] = cos_transform(df[column].values)
        df[f'{column}_sin'] = sin_transform(df[column].values)
        df = df.drop(columns=[column])
        return df
    
    def get_prod_data(self) -> pd.DataFrame():
        # read raw production data
        prod = {}
        for park in self.parks:
            prod[park] = pd.read_csv(f'{self.raw_prod_data_path}/{park}_prod_total.csv')
            prod[park].time = pd.to_datetime(prod[park].time)
            prod[park] = prod[park].set_index(['time'])
            prod[park].columns = [f'production_{park}']
        
        # concatenate formatted production data
        df = pd.concat(list(prod.values()), axis=1)
        
        return df
    
    def __clean_prod_data(self, prod_df) -> pd.DataFrame():
        # lower limit values found by plotting data
        limits = {'bess': 0.0, 'vals': -0.1, 'skom': -0.23, 'yvik': -0.01}
        
        for col in prod_df.columns:
            park = col.split('_')[-1]
            prod_df[col][prod_df[col] < limits[park]] = np.nan
        
        # remove a bit more than a year of data with more production than the other years at Valsneset
        max_production_vals = prod_df.production_vals.iloc[-24*365*5:].max() # find max production over last five years
        
        max_values_vals = prod_df.production_vals[prod_df.production_vals > max_production_vals]
        removal_interval_vals = (max_values_vals.index[0], max_values_vals.index[-1])
        
        prod_df.production_vals = pd.concat([
            prod_df.production_vals[:removal_interval_vals[0]-timedelta(hours=24)],
            prod_df.production_vals[removal_interval_vals[1]+timedelta(hours=24):]
        ], axis=0).asfreq('H')
        
        return prod_df
    
    def __get_isolated_nans(self, prod_df, col) -> pd.DataFrame():
        mask_repeat_NaN = prod_df.groupby(prod_df[col].notna().cumsum())[col].transform('size').le(2)
        mask = mask_repeat_NaN&prod_df[col].isna()
        return pd.DataFrame(data=prod_df[mask][col])
    
    def __fill_missing_prod_values(self, prod_df, wind_df) -> pd.DataFrame():
        for park in self.parks:
            if park=='skom':
                park_name = 'bess'
            else:
                park_name = park
            
            df_nans = self.__get_isolated_nans(prod_df, f'production_{park}')
            df_nans = df_nans.loc[:self.valid_start_date]
            
            # use all data until valid set, do regression
            wind_df_park = get_columns(wind_df, park_name)
            wind_df_park = wind_df_park.drop(columns=get_columns(wind_df_park, '+').columns)

            X = pd.concat([
                prod_df[f'production_{park}'],
                wind_df_park
            ], axis=1).dropna(axis=0)
            
            X = X.loc[:self.valid_start_date-timedelta(hours=1)]
            
            feature = X[f'wind_speed_{park_name}'].values.reshape(-1, 1)
            target = X[f'production_{park}'].values
            
            linear = LinearRegression().fit(feature, target)
            
            wind_speeds = np.zeros(len(df_nans))
            for i, index in enumerate(df_nans.index):
                wind_speeds[i] = wind_df[f'wind_speed_{park_name}'].loc[index]
            wind_speeds = wind_speeds.reshape(-1, 1)
            
            # predict values for times at df_nans, update df_nans to have values
            linear_preds = np.ravel(linear.predict(wind_speeds))
            
            df_nans[f'production_{park}'] = linear_preds
            
            logging.info(f'Filled {len(linear_preds)} missing production values with linear approximations at {park}.')
            
            prod_df[f'production_{park}'].update(df_nans[f'production_{park}'])
        
        return prod_df
    '''''
    def __infer_missing_prod_value(self, df: pd.DataFrame, time: datetime, park: str, real_prod_exists=False, only_use_seen_data=False):
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        
        interval_size = 2.0
        interval_lower = df[f'wind_speed_{park}'].loc[time_str] - interval_size/2.0
        interval_upper = interval_lower + interval_size
        
        if only_use_seen_data:
            relevant_wind = df.loc[:time_str][f'wind_speed_{park}'][
                df.loc[:time_str][f'wind_speed_{park}'] > interval_lower].copy()
            relevant_wind = relevant_wind[df.loc[:time_str][
                f'wind_speed_{park}'] < interval_upper]
        else:
            relevant_wind = df[f'wind_speed_{park}'][df[f'wind_speed_{park}'] > interval_lower].copy()
            relevant_wind = relevant_wind[df[f'wind_speed_{park}'] < interval_upper]
        relevant_prod = df.loc[relevant_wind.index][f'production_{park}']
        
        alpha = np.nanmean(np.divide(relevant_prod.values, relevant_wind.values))
        pred_prod = alpha*df[f'wind_speed_{park}'].loc[time_str]
        if real_prod_exists:
            real_prod = df[f'production_{park}'].loc[time_str].copy()
            return pred_prod, real_prod
        else:
            return pred_prod

    def __get_model_preds(self, df: pd.DataFrame, park: str):
        idxs = df[f'production_{park}'][
            df[f'production_{park}'].isna()].index

        model_preds = []
        for index in idxs:
            model_preds.append(self.__infer_missing_prod_value(df, index, park))
            
        return model_preds

    def __insert_missing_values(self, df: pd.DataFrame, park: str):
        model_preds = self.__get_model_preds(df, park)
        for i, index in enumerate(df[f'production_{park}'][
            df[f'production_{park}'].isna()].index):
            df[f'production_{park}'].loc[index] = model_preds[i]
        return df'''''
    
    '''def fill_missing_values(self, df: pd.DataFrame=None):
        
        if df is None:
            df = self.read_formatted_data()
        
        df = df.dropna(subset=get_columns(df, 'wind').columns)
        
        # drop additional rows before as well, since these are forecasts done several hours before (up to 66 hours before)
        start_remove = pd.to_datetime('2018-05-12 00:00:00')
        end_remove = pd.to_datetime('2018-05-14 00:00:00')
        df = df.query('index < @start_remove or index > @end_remove')
        
        for park in self.parks:
            if park=='vals': continue
            df = self.__insert_missing_values(df, park)
        
        df = df.asfreq('H')
        
        return df'''


    '''Handling of old data'''
    
    def read_old_formatted_data(self, nan: bool) -> pd.DataFrame:
        filename = 'df_nan' if nan else 'df_no_nan'
        df = pd.read_csv(f'{self.old_formatted_data_path}/{filename}.csv')
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index(['time'])
        return df
    
    def __read_old_wind_data(self) -> pd.DataFrame:
        '''
        Read and concatenate wind speed and wind direction NWP prediction data.
        The data is based on NWP (Numerical Weather Predictions), where the column
        'date_calc' is the time of prediction, and 'date_forecast' is the time of
        prediction. The wind direction given in degrees (0-360), is transformed into 
        cos and sin values (0 deg -> cos is 1, sin is 0, 180 deg -> cos is -1, sin is 0)

        Returns a dataframe containing wind speed and wind direction for each park.
        '''
        logging.info('Reading wind data.')

        # read weather data
        weather = {}
        for park in self.parks:
            weather[park] = pd.read_csv(f'{self.old_raw_data_path}/{park}_weather.csv')
        
        # time columns to datetime objects
        weather[self.parks[0]]['date_calc'] = pd.to_datetime(weather[self.parks[0]]['date_calc'])
        weather[self.parks[0]]['date_forecast'] = pd.to_datetime(weather[self.parks[0]]['date_forecast'])
        
        # rename to distinguish parks
        weather[self.parks[0]].columns = ['date_calc', 'date_forecast', f'wind_speed_{self.parks[0]}', f'wind_direction_{self.parks[0]}']
        for park in self.parks[1:]:
            weather[park].columns = ['date_calc_del', 'date_forecast_del', f'wind_speed_{park}', f'wind_direction_{park}']
        
        # concatenate data
        df = pd.concat(list(weather.values()), axis=1, join="inner")

        # delete excess time columns
        df = df.drop(columns=['date_calc_del', 'date_forecast_del'])
        
        # treat values above limit to be unrealistic, and tranform them to NaN
        limit = 360 # limit >= 360 in order to not set degree (direction) values to NaN
        df = pd.concat([
            df[['date_calc', 'date_forecast']], 
            get_columns(df, 'wind').apply(lambda x: [y if y <= limit else np.nan for y in x])
            ], axis=1)

        logging.info('Done reading wind data.')
        return df
    
    def get_old_wind_data(self) -> pd.DataFrame:
        
        logging.info('Formatting wind data.')
        
        # interpolate to remove a few inf and nan values
        #df = df.interpolate(method ='linear', limit_direction ='forward', limit=1)

        df = self.__read_old_wind_data()
        
        for park in self.parks:
            df = self.__direction_transforms(df, f'wind_direction_{park}')
        
        # convert the date_forecast columns to contain an integer representing number of hours ahead of calculation
        df['date_forecast'] = df['date_forecast'] - df['date_calc']
        timelist = df['date_forecast'].dt.components
        df['date_forecast'] = timelist['days'].values*24+timelist['hours'].values

        # rename date_calc to time
        df = df.rename(columns={'date_calc': 'time'})

        # add date_forecast column as a part of the columns indexing
        df = df.pivot(index='time', columns='date_forecast')
        
        # insert empty rows for the hours without data
        df = df.asfreq(freq='H')

        # in case of more than pred_horizon rows with nan values, we only want to fill up to pred_horizon
        #nan_row_counter = 0

        logging.info('Filling wind prediction dataframe using forecasts.')

        for j in range(len(df.columns.levels[0].values)):
            arr = df[str(df.columns.levels[0].values[j])].values

            for g in range(1, len(arr)):
                for h in range(len(arr[g])-1):
                    if np.isnan(arr[g][h]):
                        arr[g][h] = arr[g-1][h+1]

            df[str(df.columns.levels[0].values[j])] = arr

            logging.info(f'Done with {df.columns.levels[0].values[j]}')

        logging.info('Done filling wind prediction dataframe.')
        
        # data input to decoder
        now_df = df.copy()
        forecast_df = df.copy()
        for col in forecast_df.columns:
            if col[1] > self.pred_horizon or col[1] < 1:
                forecast_df = forecast_df.drop(columns=col)
            if col[1] > 0:
                now_df = now_df.drop(columns=col)
        forecast_df.columns = [''.join(str(col[0])+'+'+str(int(col[1]))+'h').strip() for col in forecast_df.columns.values]
        now_df.columns = [col[0] for col in now_df.columns.values]
        
        # concat wind now data and forecasted data
        df = pd.concat([now_df, forecast_df], axis=1)
        
        logging.info('Done formatting wind data.')
        return df

    def __read_old_prod_data(self) -> pd.DataFrame:
        '''
        Read and concatenate production data.

        Returns a dataframe containing hourly production at each park.
        '''
        logging.info('Reading production data.')

        # read raw production data
        prod = {}
        for park in self.parks:
            prod[park] = pd.read_csv(f'{self.raw_data_path}/{park}_prod.csv')
            prod[park].time = pd.to_datetime(prod[park].time)
            prod[park] = prod[park].set_index(['time'])
            prod[park].columns = [f'production_{park}']
        
        # concatenate formatted production data
        df = pd.concat(list(prod.values()), axis=1)
        
        logging.info('Done reading production data.')
        return df
    
    def get_old_prod_data(self) -> pd.DataFrame:
        
        df = self.__read_old_prod_data()
        
        # due to clearly untrue values, certain production values at yvik are removed
        logging.info('Removing clearly untrue production values at Ytre Vikna.')
        df.production_yvik.loc['2019-12-16 10:00:00':] = np.nan
        
        return df
    

if __name__=='__main__':
    num = NumericalDataHandler()
    
    wind = num.get_wind_data()
    
    #print(wind.columns)
    #print(wind)
    
    #prod = num.get_prod_data()
    
    #print(prod.columns)
    #print(prod)