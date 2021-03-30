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
    
    def satellite_video(self, play=False):
        '''
        Create and save satvid with name giving information about the test.
        '''
        date = datetime(2018,6,15)
        self.start = date + timedelta(hours=3)#datetime(2019,6,day,3)
        self.stop = date + timedelta(hours=21)#datetime(2019,6,day,21)
        self.interval = TimeInterval(self.start, self.stop)
        self.timestr = self.start.strftime('%Y-%m-%d-%H')
        
        self.sat_vid_name = f'{self.timestr}-{str(int(15*self.step))}min-{self.scale}sc-sat'
            
        satVid = SatVid(name=self.sat_vid_name, interval=self.interval, step=self.step, scale=self.scale)
        satVid.save()
        if play:
            satVid.play(self.sat_vid_name, 'sat', fps=self.fps)
            
    def grid_video(self, play=True):
        date = datetime(2017, 5, 17)
        self.start = date
        self.stop = date + timedelta(hours=23)
        self.interval = TimeInterval(self.start, self.stop)
        self.timestr = self.interval.start.strftime('%Y-%m-%d-%H')
        
        self.grid_vid_name = f'{self.timestr}-{str(int(15*self.step))}min-{self.scale}sc-grid'
        
        vid = FlowVid(ImgType('grid'), name=self.grid_vid_name, interval=self.interval, step=self.step)
        vid.save()
        if play:
            vid.play(self.grid_vid_name, 'grid', fps=2)
    
    def run_optical_flow(self, imgType: str):
        date = datetime(2018,6,15)
        self.start = date + timedelta(hours=3)#datetime(2019,6,day,3)
        self.stop = date + timedelta(hours=21)#datetime(2019,6,day,21)
        self.interval = TimeInterval(self.start, self.stop)
        self.timestr = self.start.strftime('%Y-%m-%d-%H')
        self.sat_vid_name = f'{self.timestr}-{str(int(15*self.step))}min-{self.scale}sc-sat'
        
        self.of = OpticalFlow(satVidName=self.sat_vid_name, step=self.step, scale=self.scale)
        
        self.of.denseflow(imgType, play=True, save=True, fps=1)
        
    def optical_flow_all_days(self):
        for imgType in ['lk_dense']:
            for date in pd.date_range(start=datetime(2018,9,30), end=datetime(2018,12,31), freq='D'):
                
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