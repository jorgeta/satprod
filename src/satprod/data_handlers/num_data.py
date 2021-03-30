import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

from satprod.data_handlers.data_utils import sin_transform, cos_transform, get_columns

from tasklog.tasklogger import logging

class NumericalDataHandler():

    def __init__(self, pred_limit: int=6, train_test_split: datetime=datetime(2020,5,1,0)):
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'
        self.parks = ['bess', 'skom', 'vals', 'yvik']
        self.raw_data_path = f'{self.root}/data/num'
        self.formatted_data_path = f'{self.root}/data/formatted'
        
        self.pred_limit = pred_limit
        self.train_test_split = train_test_split
        
        # treat inf values as NaN
        pd.options.mode.use_inf_as_na = True
        
        # stop writing warning to console concerning potential copying error not relevant in this case
        pd.options.mode.chained_assignment = None
    
    def update_formatted_files(self):
        wind_df = self.get_wind_data()
        prod_df = self.get_prod_data()
        df = pd.concat([wind_df,prod_df], axis=1)
        self.write_formatted_data(df, True)
        df = self.fill_missing_values(df)
        # drop skom wind data because they're the same as for bess
        df = df.drop(columns=get_columns(wind_df, 'skom').columns)
        self.write_formatted_data(df, False)
    
    def write_formatted_data(self, df: pd.DataFrame, nan: bool):
        filename = 'df_nan' if nan else 'df_no_nan'
        df.to_csv(f'{self.formatted_data_path}/{filename}.csv')
        
    def read_formatted_data(self, nan: bool) -> pd.DataFrame:
        filename = 'df_nan' if nan else 'df_no_nan'
        df = pd.read_csv(f'{self.formatted_data_path}/{filename}.csv')
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index(['time'])
        return df
    
    def __read_wind_data(self) -> pd.DataFrame:
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
            weather[park] = pd.read_csv(f'{self.raw_data_path}/{park}_weather.csv')
        
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
    
    def __direction_transforms(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        assert column in df.columns, f'{column} needs to be a column in the input dataframe.'
        
        df[f'{column}_cos'] = cos_transform(df[column].values)
        df[f'{column}_sin'] = sin_transform(df[column].values)
        df = df.drop(columns=[column])
        return df
    
    def get_wind_data(self) -> pd.DataFrame:
        
        logging.info('Formatting wind data.')
        
        # interpolate to remove a few inf and nan values
        #df = df.interpolate(method ='linear', limit_direction ='forward', limit=1)

        df = self.__read_wind_data()
        
        for park in self.parks:
            df = self.__direction_transforms(df, f'wind_direction_{park}')
        
        # convert the date_forecast columns to contain an integer represeting number of hours ahead of calculation
        df['date_forecast'] = df['date_forecast'] - df['date_calc']
        timelist = df['date_forecast'].dt.components
        df['date_forecast'] = timelist['days'].values*24+timelist['hours'].values

        # rename date_calc to time
        df = df.rename(columns={'date_calc': 'time'})

        # add date_forecast column as a part of the columns indexing
        df = df.pivot(index='time', columns='date_forecast')
        
        # insert empty rows for the hours without data
        df = df.asfreq(freq='H')

        # in case of more than pred_limit rows with nan values, we only want to fill up to pred_limit
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
            if col[1] > self.pred_limit or col[1] < 1:
                forecast_df = forecast_df.drop(columns=col)
            if col[1] > 0:
                now_df = now_df.drop(columns=col)
        forecast_df.columns = [''.join(str(col[0])+'+'+str(int(col[1]))+'h').strip() for col in forecast_df.columns.values]
        now_df.columns = [col[0] for col in now_df.columns.values]
        
        # concat wind now data and forecasted data
        df = pd.concat([now_df, forecast_df], axis=1)
        
        logging.info('Done formatting wind data.')
        return df

    def __read_prod_data(self) -> pd.DataFrame:
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
    
    def get_prod_data(self) -> pd.DataFrame:
        
        df = self.__read_prod_data()
        
        # due to clearly untrue values, certain production values at yvik are removed
        logging.info('Removing clearly untrue production values at Ytre Vikna.')
        df.production_yvik.loc['2019-12-16 10:00:00':] = np.nan
        
        return df
    
    def __infer_missing_prod_value(self, df: pd.DataFrame, time: datetime, park: str, real_prod_exists=False, only_use_seen_data=False):
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        one_day_ago_str = (time-timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        
        interval_size = 2.0
        interval_lower = df[f'wind_speed_{park}'].loc[time_str] - interval_size/2.0
        interval_upper = interval_lower + interval_size
        
        if only_use_seen_data:
            relevant_wind = df.loc[:one_day_ago_str][f'wind_speed_{park}'][
                df.loc[:time_str][f'wind_speed_{park}'] > interval_lower].copy()
            relevant_wind = relevant_wind[df.loc[:one_day_ago_str][
                f'wind_speed_{park}'] < interval_upper]
        else:
            relevant_wind = df[f'wind_speed_{park}'][df[f'wind_speed_{park}'] > interval_lower].copy()
            relevant_wind = relevant_wind[df[f'wind_speed_{park}'] < interval_upper]
        relevant_prod = df.loc[relevant_wind.index][f'production_{park}']
        
        alpha = np.nanmean(np.divide(relevant_prod.values, np.power(relevant_wind.values, 3)))
        pred_prod = alpha*np.power(df[f'wind_speed_{park}'].loc[time_str], 3)
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
        return df
    
    def fill_missing_values(self, df: pd.DataFrame=None):
        
        if df is None:
            df = self.read_formatted_data(nan=True)
        df = df[df.index < self.train_test_split]
        
        df = df.dropna(subset=get_columns(df, 'wind').columns)
        
        # drop additional rows before as well, since these are forecasts done several hours before (up to 66 hours before)
        start_remove = pd.to_datetime('2018-05-12 00:00:00')
        end_remove = pd.to_datetime('2018-05-14 00:00:00')
        df = df.query('index < @start_remove or index > @end_remove')
        
        for park in self.parks:
            if park=='vals': continue
            df = self.__insert_missing_values(df, park)
        
        df = df.asfreq('H')
        
        return df