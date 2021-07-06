import fire
import os
from datetime import datetime, timedelta
import cv2
import logging
import pandas as pd
import numpy as np

from satprod.configs.config_utils import read_yaml, TimeInterval, ImgType, structurize_wind_grid_images
from satprod.data_handlers.img_data import Img, ImgDataset, SatImg, FlowImg
from satprod.data_handlers.video import SatVid, FlowVid
from satprod.optical_flow.optical_flow_optim import OpticalFlowOptim
from satprod.optical_flow.optical_flow import OpticalFlow
from satprod.data_handlers.num_data import NumericalDataHandler
from satprod.pipelines.training import train_model
from satprod.pipelines.evaluation import ModelEvaluation
from satprod.pipelines.dataset import WindDataset
from satprod.pipelines.comparison import ModelComparison
from satprod.configs.job_configs import TrainConfig

from tasklog.tasklogger import init_logger
init_logger()

class App:

    def __init__(self):
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../..'
        self.config = read_yaml(f'{self.root}/config.yaml')

        logging.info(f'Running {self.config.app}')

        self.step = 1
        self.scale = 20
        self.fps = 6
        
        self.of_optim = OpticalFlowOptim(self.step, self.scale, self.fps)
        
        self.num = NumericalDataHandler()
        
    def train(self, parameter_tuning: bool=False, feature_selection: bool=False):
        if feature_selection:
            
            numerical_feature_combinations = [
                ['production'],
                ['production', 'speed'],
                ['production', 'speed', 'direction'],
                ['production', 'speed', 'direction', 'temporal'],
                ['speed'],
                ['speed', 'direction']
                ['speed', 'direction', 'temporal']
            ]
            
            for numerical_features in numerical_feature_combinations:
                self.config.data_config.numerical_features = numerical_features
            
        if parameter_tuning:
            if self.config.model == 'LSTM':
                
                pass
            
            if self.config.model == 'TCN':
                channel_combinations = [
                    [32, 32, 32],
                    [32, 32], 
                    [32],
                ]
                kernel_combinations = [
                    {'kernel_size': 3, 'dilation_base': 2},
                    {'kernel_size': 4, 'dilation_base': 2}
                ]
                dropout_combinations = [
                    0.1, 0.2, 0.3, 0.4, 0.5
                ]
                for channels in channel_combinations:
                    for kernel in kernel_combinations:
                        for dropout in dropout_combinations:
                            self.config.models.tcn.channels = channels
                            self.config.models.tcn.kernel_size = kernel['kernel_size']
                            self.config.models.tcn.dilation_base = kernel['dilation_base']
                            self.config.models.tcn.dropout = dropout
                            
                            train_model(self.config)
            
        if not parameter_tuning and not feature_selection:
            train_model(self.config)
        
    def evaluate(self):
        timestamps = [
            #'2021-06-14-13-02',
            '2021-07-03-04-16',
            '2021-07-03-23-38',
            '2021-07-04-07-37',
            '2021-07-04-15-06',
            '2021-07-05-05-04',
            '2021-07-05-14-34',
            '2021-07-06-01-27',
            '2021-07-03-07-52',
            '2021-07-03-23-53',
            '2021-07-04-08-21',
            '2021-07-04-21-12',
            '2021-07-05-05-34',
            '2021-07-05-15-14',
            '2021-07-06-11-26',
            '2021-07-02-20-40',
            '2021-07-03-11-26',
            '2021-07-04-00-09',
            '2021-07-04-14-19',
            '2021-07-04-21-33',
            '2021-07-05-05-57',
            '2021-07-05-23-45',
            '2021-07-06-12-14',
            '2021-07-03-00-31',
            '2021-07-03-15-16',
            '2021-07-04-07-20',
            '2021-07-04-14-38',
            '2021-07-04-21-50',
            '2021-07-05-13-59',
            '2021-07-06-00-27',
            '2021-07-06-13-13'
        ]
        for timestamp in timestamps:
            model_name = 'TCN'
            #timestamp = '2021-06-16-11-29'
            park = 'skom'
            sorting = 'num'
            ModelEvaluation(timestamp=timestamp, model_name=model_name, park=park, sorting=sorting)
        
    def compare(self, park: str):
        ModelComparison(park, self.config.comparison)
    
    def satellite_video(self, date: datetime=datetime(2019,6,3), play=False):
        """Create and save satvid with name giving information about the test.

        Args:
            date (datetime, optional): [description]. Defaults to datetime(2019,6,3).
            play (bool, optional): [description]. Defaults to False.
        """
        
        self.start = date + timedelta(hours=3)#datetime(2019,6,day,3)
        self.stop = date + timedelta(hours=21)#datetime(2019,6,day,21)
        self.interval = TimeInterval(self.start, self.stop)
        self.timestr = self.start.strftime('%Y-%m-%d-%H')
        
        self.sat_vid_name = f'{self.timestr}-{str(int(15*self.step))}min-{self.scale}sc-sat'
        
        satVid = SatVid(name=self.sat_vid_name, interval=self.interval, step=self.step, scale=self.scale)
        satVid.save()
        if play:
            satVid.play(self.sat_vid_name, 'sat', fps=self.fps)
            
    def grid_video(self, date: datetime=datetime(2019, 5, 17), play=True):
        """Creates a video from the grid speed images.

        Args:
            date (datetime, optional): [description]. Defaults to datetime(2019, 5, 17).
            play (bool, optional): [description]. Defaults to True.
        """
        self.start = date
        self.stop = date + timedelta(hours=23)
        self.interval = TimeInterval(self.start, self.stop)
        self.timestr = self.interval.start.strftime('%Y-%m-%d-%H')
        
        self.grid_vid_name = f'{self.timestr}-{str(int(15*self.step))}min-{self.scale}sc-grid'
        
        vid = FlowVid(ImgType('grid'), name=self.grid_vid_name, interval=self.interval, step=self.step)
        vid.save()
        if play:
            vid.play(self.grid_vid_name, 'grid', fps=2)
    
    def optflow(self, imgType: str, date: datetime=datetime(2019,6,3)):
        
        self.start = date
        self.stop = date + timedelta(hours=23)
        self.interval = TimeInterval(self.start, self.stop)
        self.timestr = self.start.strftime('%Y-%m-%d-%H')
        self.sat_vid_name = f'{self.timestr}-{str(int(15*self.step))}min-{self.scale}sc-sat'
        
        self.of = OpticalFlow(satVidName=self.sat_vid_name, step=self.step, scale=self.scale)
        
        self.of.denseflow(imgType, play=False, save=False, fps=1)
    
    def optflow_all_days(self, imgType: str, 
                        start: datetime=datetime(2018,3,20), 
                        end: datetime=datetime(2019,3,19)):
        
        for date in pd.date_range(start=start, end=end, freq='D'):
            
            self.interval = TimeInterval(date, date+timedelta(hours=23, minutes=59))
            self.timestr = self.interval.start.strftime('%Y-%m-%d-%H')
            
            self.sat_vid_name = f'{self.timestr}-{str(int(15*self.step))}min-{self.scale}sc-sat'
            try:
                satVid = SatVid(name=self.sat_vid_name, interval=self.interval, step=self.step, scale=self.scale)
                satVid.save()
                #satVid.play(self.sat_vid_name, 'sat')
            except:
                continue
                
            try:
                self.of = OpticalFlow(
                    satVidName=self.sat_vid_name, step=self.step, scale=self.scale)
                
                self.of.denseflow(imgType, save=False, play=False, fps=1)
            except:
                pass
            
            try:
                satVid.delete(self.sat_vid_name, 'sat')
            except:
                continue
    
    def wind_grid_structurize(self):
        """Orders the wind speed grid images in the desired folder structure,
        for example 'satprod/data/img/grid/2013/01/01/00;00;00.png'.
        """
        structurize_wind_grid_images()
    
    def update_image_indices(self):
        
        # dummy train config
        train_config = TrainConfig(
            batch_size = 64,
            num_epochs = 30,
            learning_rate = 4e-3,
            scheduler_step_size = 5,
            scheduler_gamma = 0.8,
            train_valid_splits = 1,
            pred_sequence_length = 5,
            random_seed = 0,
            parks = ['bess'],
            num_feature_types = ['speed'],
            img_features = ['grid'],
            img_extraction_method = 'lenet'
        )
        
        dataset = WindDataset(data_config)
        if not dataset.image_indices_recently_updated:
            dataset.update_image_indices()
    

def main():
    fire.Fire(App)

if __name__ == '__main__':
    main()