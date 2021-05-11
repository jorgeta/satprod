import fire
import os
from datetime import datetime, timedelta
import cv2
import logging
import pandas as pd

from satprod.configs.config_utils import read_yaml, TimeInterval, ImgType
from satprod.data_handlers.img_data import Img, ImgDataset, SatImg, FlowImg
from satprod.data_handlers.video import SatVid, FlowVid
from satprod.optical_flow.optical_flow_optim import OpticalFlowOptim
from satprod.optical_flow.optical_flow import OpticalFlow
from satprod.data_handlers.num_data import NumericalDataHandler
from satprod.configs.config_utils import structurize_wind_grid_images
from satprod.pipelines.training import train_model
from satprod.pipelines.evaluation import Evaluate

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
        
    def train(self):
        train_model()
        
    def evaluate(self):
        model_name = 'simple_LSTM'
        timestamp = '2021-05-11-17-40'
        park = 'skom'
        evaluate = Evaluate(timestamp=timestamp, model_name=model_name, park=park)
        evaluate.baseline_comparisons()
        evaluate.plot_fitting_example()
        evaluate.plot_training_curve()
    
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
        """[summary]

        Args:
            imgType (str): [description]
            date (datetime, optional): [description]. Defaults to datetime(2019,6,3).
        """
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
        """[summary]

        Args:
            imgType (str): [description]
            start (datetime, optional): [description]. Defaults to datetime(2018,3,20).
            end (datetime, optional): [description]. Defaults to datetime(2019,3,19).
        """
        for date in pd.date_range(start=datetime(2019,1,1), end=datetime(2019,3,19), freq='D'):
            
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
        structurize_wind_grid_images()
    

def main():
    fire.Fire(App)

if __name__ == '__main__':
    main()