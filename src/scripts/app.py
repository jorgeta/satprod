import fire
import os
from datetime import datetime, timedelta
import cv2
import logging

from satprod.configs.config_utils import read_yaml, TimeInterval, ImgType
from satprod.data_handlers.img_data import Img, ImgDataset, SatImg, FlowImg
from satprod.data_handlers.video import SatVid, FlowVid
from satprod.optical_flow.optical_flow_optim import OpticalFlowOptim
from satprod.optical_flow.optical_flow import OpticalFlow
from satprod.data_handlers.num_data import NumericalDataHandler

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
    
    def satvid(self, day: int, play=False):
        '''
        Create and save satvid with name giving information about the test.
        '''
        date = datetime(2019,6,day)
        self.start = date + timedelta(hours=3)#datetime(2019,6,day,3)
        self.stop = date + timedelta(hours=21)#datetime(2019,6,day,21)
        self.interval = TimeInterval(self.start, self.stop)
        self.timestr = self.start.strftime('%Y-%m-%d-%H')
        
        self.sat_vid_name = f'{self.timestr}-{str(int(15*self.step))}min-{self.scale}sc-sat'
            
        satVid = SatVid(name=self.sat_vid_name, interval=self.interval, step=self.step, scale=self.scale)
        satVid.save()
        if play:
            satVid.play(name=self.sat_vid_name, fps=self.fps)
    
    def of(self, day: int, imgType: str):
        date = datetime(2019,6,day)
        self.start = date + timedelta(hours=3)#datetime(2019,6,day,3)
        self.stop = date + timedelta(hours=21)#datetime(2019,6,day,21)
        self.interval = TimeInterval(self.start, self.stop)
        self.timestr = self.start.strftime('%Y-%m-%d-%H')
        self.sat_vid_name = f'{self.timestr}-{str(int(15*self.step))}min-{self.scale}sc-sat'
        
        self.of = OpticalFlow(satVidName=self.sat_vid_name, interval=self.interval, step=self.step, scale=self.scale)
        
        self.of.denseflow(imgType)


def main():
    fire.Fire(App)

if __name__ == '__main__':
    main()