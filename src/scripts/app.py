import fire
import os
from datetime import datetime
import cv2
import logging

from satprod.configs.config_utils import read_yaml, TimeInterval, ImgType
from satprod.data_handlers.img_data import Img, ImgDataset, SatImg, FlowImg
from satprod.data_handlers.video import Vid, SatVid, FlowVid
from satprod.data_handlers.optical_flow import OpticalFlow

from tasklog.tasklogger import init_logger
init_logger()

class App:

    def __init__(self):
        #self.logger = TaskLogger().logger

        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.config = read_yaml(f'{cd}/../../config.yaml')

        logging.info(f'Running {self.config.app}')
        
        day = 3
        self.start = datetime(2019,6,day,3)
        self.stop = datetime(2019,6,day,21)

        self.interval = TimeInterval(self.start, self.stop)

        self.timestr = self.start.strftime('%Y-%m-%d-%H')

        self.step = 1

        # when scale is lowered from 10, lower min distance in LK
        self.scale = 20
        
        self.fps = 6

        self.satname = f'{self.timestr}-{str(int(15*self.step))}min-{self.scale}sc-sat'
        self.densename = f'{self.timestr}-{str(int(15*self.step))}min-{self.scale}sc-dense'
        self.sparsename = f'{self.timestr}-{str(int(15*self.step))}min-{self.scale}sc-sparse'
        self.sparsemaskname = f'{self.timestr}-{str(int(15*self.step))}min-{self.scale}sc-sparsemask'

        self.of = OpticalFlow(
            satVidName=self.satname, interval=self.interval, step=self.step, scale=self.scale)

    def satvid(self):
        '''
        Create and save satvid with name giving information about the test.
        '''
        
        v = SatVid(name=self.satname, interval=self.interval, step=self.step, scale=self.scale)
        v.play(name=self.satname, fps=self.fps)

    def denseflow(self):
        
        # Parameters for Farneback optical flow
        fb_params = dict ( pyr_scale = 0.5,
                        levels = 3,
                        winsize = 30,
                        iterations = 3,
                        poly_n = 7,
                        poly_sigma = 1.5,
                        flags = 0 )

        self.of.farneback(fb_params)
    
        dv = FlowVid(ImgType.DENSE, self.densename, self.interval, self.step)
        #dv.play(self.densename, fps=self.fps)

    def sparseflow(self):

        # Parameters for ShiTomasi corner detection

        # day 1, 15 min, scale 10
        # day 1, 30 min, scale 10 # best
        # day 1, 15 min, scale 20 # best, but requires 15 min
        # day 1, 30 min, scale 20
        feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.05,
                            minDistance = 14,
                            blockSize = 7 )

        # day 1, 15 min, scale 5
        '''feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.05,
                            minDistance = 7,
                            blockSize = 7 )'''
        

        # Parameters for Lucas Kanade optical flow
        lk_params = dict( winSize  = (30,30),#(15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.of.lukasKanade(feature_params, lk_params)

        sv = FlowVid(ImgType.SPARSE, self.sparsename, self.interval, self.step)
        sv.play(self.sparsename, fps=self.fps)

        #smv = FlowVid(ImgType.SPARSEMASK, self.sparsemaskname, self.interval, self.step)
        #smv.play(self.sparsemaskname, fps=self.fps)


def main():
    fire.Fire(App)

if __name__ == '__main__':
    main()