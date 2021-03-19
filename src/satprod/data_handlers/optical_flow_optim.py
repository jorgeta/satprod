import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
import json

import os
from datetime import datetime, timedelta

from satprod.configs.config_utils import TimeInterval, ImgType
from satprod.data_handlers.num_data import read_formatted_data_no_nan
from satprod.data_handlers.data_utils import wind_degrees_to_polar, max_min_scale_df, RMSE, MAE
from satprod.data_handlers.optical_flow import OpticalFlow

from tasklog.tasklogger import logging


class OpticalFlowOptim():
    
    def __init__(self, step: int=1, scale: int=100, fps: int=6):
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'
        
        self.step = step
        self.scale = scale
        self.fps = fps
        
    def optim(self):
        folder = 'of_optim_results'
        
        # loop of methods
        for imgType in ['lk_dense']:
            imgType = ImgType(imgType)
            
            os.makedirs(f'{self.root}/data/{folder}/{imgType.value}', exist_ok=True)
            
            lowest_dir_mae = np.inf
            
            grid, params = self.get_parameter_grid(imgType.value)
        
            # loop of parameters
            for param_settings in list(ParameterGrid(grid)):
                for key, value in param_settings.items():
                    params[key] = value
                try:
                    # do optical flow on all days with the method and parameters of this part of the loop.
                    self.of_all_days(imgType.value, params)
            
                    # gather all data for comparison (must be called after of_all_days)
                    data = self.get_all_data(imgType)
                    
                    # compute different measures
                    evaluation = self.evaluate(data)
                    
                    if evaluation['dir_total_mae'] < lowest_dir_mae:
                        lowest_dir_mae = evaluation['dir_total_mae']
                        optimal_params_dir_mae = params.copy()
                        evaluation_with_lowest_dir_mae = evaluation.copy()
                    
                    # summary to console
                    logging.info(imgType)
                    logging.info(params)
                    logging.info(evaluation)
                except:
                    logging.warning(f'The combination {params} of parameters were not accepted by the algorithm.')
                    pass
            
            # save best params with error measures
            dmp = json.dumps(optimal_params_dir_mae)
            f = open(f'{self.root}/data/{folder}/{imgType.value}/optimal_params_dir_mae.json','w')
            f.write(dmp)
            f.close()
            
            dmp = json.dumps(evaluation_with_lowest_dir_mae)
            f = open(f'{self.root}/data/{folder}/{imgType.value}/evaluation_with_lowest_dir_mae.json','w')
            f.write(dmp)
            f.close()
            
    def get_parameter_grid(self, imgType: str):
        imgType = ImgType(imgType)
        if imgType==ImgType.FB_DENSE:
            variable = {
                
            }
            constant = {
                'winsize' : 29,
                'levels' : 2,
                'poly_n' : 13,
                'poly_sigma' : 2.7,
                'pyr_scale' : 0.4,
                'iterations' : 3,
                'flags': 0
            }
            return variable, constant
        elif imgType==ImgType.DTVL1_DENSE:
            variable = { 
                'warps' : [5, 10, 15]
            }
            constant = {
                'medianFiltering' : 5,
                'tau' : 0.125,
                'lambda' : 0.20,
                'scaleStep' : 0.8,
                'gamma' : 0.0,
                'epsilon' : 0.01,
                'theta' : 0.3,
                'useInitialFlow' : False,
                'nscales' : 5,
                'innnerIterations' : 30,
                'outerIterations' : 10,
            }
            return variable, constant
        elif imgType==ImgType.LK_DENSE:
            variable = { 
                
            }
            constant = {
                'k' : 64,
                'sigma' : 0.1,
                'use_post_proc' : True,
                'fgs_lambda' : 500.0,
                'fgs_sigma' : 1.5,
                'grid_step' : 2
            }
            return variable, constant
        elif imgType==ImgType.RLOF_DENSE:
            variable = {
                
            }
            constant = {
                'forwardBackwardThreshold' : 2.5,
                'epicK': 128, 
                'epicSigma': 0.05,
                'epicLambda' : 200.0,
                'fgsLambda' : 500.0,
                'fgsSigma' : 1.5,
                'use_variational_refinement' : False,
                'use_post_proc' : True
            }
            return variable, constant
        else:
            logging.warning(f'{imgType.value} is an invalid image type.')
    
    def evaluate(self, data: pd.DataFrame):
        evaluation = {}
        
        mag_total_mae = 0
        dir_total_mae = 0
        
        mag_total_rmse = 0
        dir_total_rmse = 0
        
        parks = ['bess', 'skom', 'vals', 'yvik']
        for park in parks:
            park_data = data[[col for col in data.columns if park in col]]
            for trig in ['sin', 'cos']:
                dir_data = park_data[[col for col in park_data.columns if trig in col]]
            
                pix = dir_data[f'{park}_deg_pixel_{trig}'].values
                med = dir_data[f'{park}_deg_median_{trig}'].values
                wdi = dir_data[f'wind_direction_{trig}_{park}'].values
                
                evaluation[f'{park}_{trig}_pix_med_rmse'] = RMSE(pix, med)
                evaluation[f'{park}_{trig}_pix_wdi_rmse'] = RMSE(pix, wdi)
                evaluation[f'{park}_{trig}_wdi_med_rmse'] = RMSE(wdi, med)
                dir_total_rmse += RMSE(wdi, med)
                
                evaluation[f'{park}_{trig}_pix_med_mae'] = MAE(pix, med)
                evaluation[f'{park}_{trig}_pix_wdi_mae'] = MAE(pix, wdi)
                evaluation[f'{park}_{trig}_wdi_med_mae'] = MAE(wdi, med)
                dir_total_mae += MAE(wdi, med)
            
            pix = park_data[f'{park}_mag_pixel'].values
            med = park_data[f'{park}_mag_median'].values
            wsp = park_data[f'wind_speed_{park}'].values
            
            evaluation[f'{park}_mag_pix_med_rmse'] = RMSE(pix, med)
            evaluation[f'{park}_mag_pix_wsp_rmse'] = RMSE(pix, wsp)
            evaluation[f'{park}_mag_wsp_med_rmse'] = RMSE(wsp, med)
            mag_total_rmse += RMSE(wsp, med)
            
            evaluation[f'{park}_mag_pix_med_mae'] = MAE(pix, med)
            evaluation[f'{park}_mag_pix_wsp_mae'] = MAE(pix, wsp)
            evaluation[f'{park}_mag_wsp_med_mae'] = MAE(wsp, med)
            mag_total_mae += MAE(wsp, med)
        
        evaluation['mag_total_mae'] = mag_total_mae
        evaluation['mag_total_rmse'] = mag_total_rmse
        evaluation['dir_total_mae'] = dir_total_mae
        evaluation['dir_total_rmse'] = dir_total_rmse
        return evaluation
    
    def of_all_days(self, imgType: str, params=None):
        imgType = ImgType(imgType)
        for date in pd.date_range(start=datetime(2019,6,3), end=datetime(2019,6,9), freq='D'):
            self.start = date + timedelta(hours=3)#datetime(2019,6,day,3)
            self.stop = date + timedelta(hours=21)#datetime(2019,6,day,21)
            self.interval = TimeInterval(self.start, self.stop)
            self.timestr = self.start.strftime('%Y-%m-%d-%H')
            
            self.sat_vid_name = f'{self.timestr}-{str(int(15*self.step))}min-{self.scale}sc-sat'

            self.opticalFlow = OpticalFlow(
                satVidName=self.sat_vid_name, interval=self.interval, step=self.step, scale=self.scale)
            
            self.opticalFlow.denseflow(imgType.value, params=params, play=False, fps=self.fps)
        
    def get_numerical_data(self, date: datetime) -> pd.DataFrame:
        enc_df, _ = read_formatted_data_no_nan(f'{self.root}/data/formatted')
        start = date + timedelta(hours=4)
        stop = date + timedelta(hours=21)
        data = enc_df.loc[start.strftime('%Y-%m-%d %H:%M:%S'):stop.strftime('%Y-%m-%d %H:%M:%S')].copy()
        parks = ['bess', 'skom', 'vals', 'yvik']
        for park in parks:
            data[f'wind_direction_sin_{park}'] = np.sin(np.pi*wind_degrees_to_polar(
                data[f'wind_direction_sin_{park}'], data[f'wind_direction_cos_{park}'])/180)
            data[f'wind_direction_cos_{park}'] = np.cos(np.pi*wind_degrees_to_polar(
                data[f'wind_direction_sin_{park}'], data[f'wind_direction_cos_{park}'])/180)
        return data
    
    def get_of_data(self, date: datetime, imgType: ImgType) -> pd.DataFrame:
        date_for_name = date + timedelta(hours=3, minutes=15)
        name = date_for_name.strftime('%Y-%m-%d-%H') + f'-15min-20sc-{imgType.value}'
        of_res = pd.read_csv(f'{self.root}/data/of_num_results/{name}.csv')
        of_res['time'] = of_res['Unnamed: 0']
        of_res['time'] = pd.to_datetime(of_res['time'])
        of_res = of_res.set_index('time')
        of_res = of_res.drop(columns=['Unnamed: 0'])
        of_res = of_res[3:].asfreq('H')
        
        of_deg_res = of_res[[col for col in of_res.columns if 'deg' in col]]
        for col in of_deg_res.columns:
            sin_tmp = np.sin(of_deg_res[col])
            cos_tmp = np.cos(of_deg_res[col])
            of_deg_res = of_deg_res.drop(columns=[col])
            of_deg_res[f'{col}_sin'] = sin_tmp
            of_deg_res[f'{col}_cos'] = cos_tmp
        of_res = of_res.drop(columns=[col for col in of_res.columns if 'deg' in col])
        of_res = pd.concat([of_res, of_deg_res], axis=1)
        return of_res
    
    def get_all_data(self, imgType: str) -> pd.DataFrame:
        imgType = ImgType(imgType)
        of_res = pd.DataFrame()
        data = pd.DataFrame()
        for date in pd.date_range(start=datetime(2019,6,3), end=datetime(2019,6,9), freq='D'):
            new_of_res = self.get_of_data(date, imgType)
            of_res = pd.concat([of_res, new_of_res], axis=0)
            
            new_data = self.get_numerical_data(date)
            data = pd.concat([data, new_data], axis=0)
            
        of_dir_df = of_res[[col for col in of_res.columns if 'deg' in col]]
        of_mag_df = max_min_scale_df(of_res[[col for col in of_res.columns if 'mag' in col]])
        speed = max_min_scale_df(data[[col for col in data.columns if 'speed' in col]])
        prod = max_min_scale_df(data[[col for col in data.columns if 'prod' in col]])
        direction = data[[col for col in data.columns if 'direction' in col]]
        
        concat = pd.concat([of_dir_df, direction, of_mag_df, speed, prod], axis=1)
        return concat