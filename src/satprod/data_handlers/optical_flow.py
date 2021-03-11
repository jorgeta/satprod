import os
import cv2
import numpy as np
import pandas as pd

from satprod.data_handlers.img_data import ImgDataset
from satprod.configs.config_utils import ImgType, TimeInterval, read_yaml
from satprod.data_handlers.data_utils import scaler

from tasklog.tasklogger import logging

class OpticalFlow():
    '''
    Performs optical flow on a series of satellite images, and saves them
    using the same struture as the satellite images are stored.

    Methods supported:
        Farneback (cv2.calcOpticalFlowFarneback())
        Lukas-Kanade (cv2.calcOpticalFlowPyrLK())

    Possible future methods:
        Horn-Schunck (?)
        cv2.SparseOpticalFlow.calc()
        cv2.DualTVL1OpticalFlow.calc()
    '''

    def __init__(self, satVidName: str, interval: TimeInterval, step: int=1, scale: int=100):
        '''
        Input parameters:
            satVidName: filename of the satellite video to do optical flow on
            interval: TimeInterval giving first and last image of the video,
            scale: scaling used when creating the video satVidName
            step: if 1, use all images, if 2, every second.
                Included in order to know what image is the first one with flow results
        '''

        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'

        self.videopath = os.path.join(self.root, 'data/video')

        self.name = satVidName
        self.step = step
        self.scale = scale

        # Get satellite image paths to mimic them
        data = ImgDataset(ImgType.SAT)

        #self.getDateIdx = data.getDateIdx
        self.start_idx = data.getDateIdx(interval.start)
        self.stop_idx = data.getDateIdx(interval.stop)
        self.img_paths = data.img_paths[self.start_idx+step:self.stop_idx+step]
        self.timestamps = data.timestamps[self.start_idx+step:self.stop_idx+step]
        
        # initialise dataframes for storing results of dense optical flow
        self.direction_df = pd.DataFrame(
            columns = ['vals', 'yvik', 'bess', 'skom'],
            index = self.timestamps
        )
        self.magnitude_df = pd.DataFrame(
            columns = ['vals', 'yvik', 'bess', 'skom'],
            index = self.timestamps
        )
        
        # positions of parks in full scale image (see notebook park_pixel_positions.ipynb)
        self.positions = {'vals': (200,460), 'yvik': (75, 580), 'bess': (135, 590), 'skom': (140, 600)}
        
        # update positions to be correct for the scaled image
        for key, value in self.positions.items():
            self.positions[key] = (
                int(np.round(value[0]*self.scale/100)), int(np.round(value[1]*self.scale/100)))

        logging.info(f'Object for optical flow on {self.name} initialised.')

    def get_degrees(self, ang_img) -> dict:
        '''
        Extract degrees from angle image obtained by Farneback and cartToPolar.
        '''

        degrees = {'vals': 0, 'yvik': 0, 'bess': 0, 'skom': 0}
        for key, value in self.positions.items():
            degrees[key] = int(360-2*ang_img[value[0],value[1]])
        return degrees
    
    def get_magnitude(self, mag_img) -> dict:
        '''
        Extract magnutide/speed from magnitude image obtained by Farneback and cartToPolar.
        '''

        magnitudes = {'vals': 0, 'yvik': 0, 'bess': 0, 'skom': 0}
        for key, value in self.positions.items():
            magnitudes[key] = mag_img[value[0],value[1]]
        return magnitudes

    def farneback(self, fb_params):
        logging.info('Running Farneback optical flow.')
        logging.info(f'Farneback params:\n {fb_params}.')
        logging.info(f'Images every {self.step*15} minutes between {self.timestamps[0]} and {self.timestamps[-1]} are used.')
        
        # initialise and create where the results should be stored
        self.dense_img_paths = []
        for i in range(0, len(self.img_paths), self.step):
            dp = self.img_paths[i].replace('/img/', f'/dense_flow/')
            self.dense_img_paths.append(dp)
            os.makedirs('/'.join(dp.split('/')[:-1]), exist_ok=True)

        # get satellite image video
        cap = cv2.VideoCapture(os.path.join(self.videopath, self.name+'.avi'))

        # get first image in the video, and store it to prvs (previous)
        _, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        
        # initialise hsv image (results are a series of these)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        counter = 0
        while(1):
            # retrieving next image in satellite image video
            _, frame2 = cap.read()
            if frame2 is not None: 
                logging.info(f'Performing dense optical flow at for time {self.timestamps[counter]}')
                
                # grayscaling newest image
                next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

                # perform farneback optical flow
                flow = cv2.calcOpticalFlowFarneback(prvs, next, None, **fb_params)
                
                # create hsv image from algorithm result
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

                # registering wind direction estimate at park positions
                degrees = self.get_degrees(ang*180/np.pi/2)
                for key, value in degrees.items():
                    self.direction_df.loc[f'{self.timestamps[counter]}'][key] = value

                # registering wind speed estimate at park positions
                magnitudes = self.get_magnitude(mag)
                for key, value in magnitudes.items():
                    self.magnitude_df.loc[f'{self.timestamps[counter]}'][key] = value

                # save result image to data folder
                cv2.imwrite(self.dense_img_paths[counter],rgb)
                prvs = next
            else:
                # no images left in the satellite image video, so the algorithm is stopped
                cap.release()
                cv2.destroyAllWindows()
                break
            counter += 1

        # repetition of release and destroy calls for safety
        cap.release()
        cv2.destroyAllWindows()

        # store results at park positions to csv files named after the satellite image video name
        os.makedirs(f'{self.root}/data/direction_dfs', exist_ok=True)
        os.makedirs(f'{self.root}/data/magnitude_dfs', exist_ok=True)
        self.direction_df.to_csv(f'{self.root}/data/direction_dfs/{self.name}.csv')
        self.magnitude_df.to_csv(f'{self.root}/data/magnitude_dfs/{self.name}.csv')

        logging.info('Finished running Farneback optical flow.')

    def lucasKanade(self, feature_params, lk_params):
        logging.info('Running Lucas-Kanade optical flow.')
        logging.info(f'Feature params:\n {feature_params}.')
        logging.info(f'Lucas-Kanade params:\n {lk_params}.')
        logging.info(f'Images every {self.step*15} minutes between {self.timestamps[0]} and {self.timestamps[-1]} are used.')

        # initialise and create where the results should be stored
        self.sparse_img_paths = []
        self.sparsemask_img_paths = []
        for i in range(0, len(self.img_paths), self.step):
            nsp = self.img_paths[i].replace('/img/', f'/sparse_flow/')
            nsmp = self.img_paths[i].replace('/img/', f'/sparse_flow_mask/')
            self.sparse_img_paths.append(nsp)
            self.sparsemask_img_paths.append(nsmp)
            os.makedirs('/'.join(nsp.split('/')[:-1]), exist_ok=True)
            os.makedirs('/'.join(nsmp.split('/')[:-1]), exist_ok=True)

        cap = cv2.VideoCapture(os.path.join(self.videopath, self.name+'.avi'))

        # Create some random colors
        color = np.random.randint(0,255,(100,3))

        # Take first frame and find corners in it
        _, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        counter = 0
        while(1):
            # retrieving next image in satellite image video
            _, frame = cap.read()
            if frame is not None:
                logging.info(f'Performing sparse optical flow at for time {self.timestamps[counter]}')
                
                # grayscaling newest image
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                # Select good points
                if p1 is None:
                    logging.error('Lucas Kanade method failed, could ot find any new good points.')
                    exit()
                
                good_new = p1[st==1]
                good_old = p0[st==1]

                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    frame = cv2.circle(frame,(a,b),1,color[i].tolist(),-1)
                img = cv2.add(frame, mask)

                # write results to data folder
                cv2.imwrite(self.sparse_img_paths[counter],img)
                cv2.imwrite(self.sparsemask_img_paths[counter],mask)

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)
            else:
                break
            
            # handle different density of images over time, only update good features every 60 minutes
            if (self.step == 4) or \
                (self.step == 1 and (counter+1) % 4 == 0) or \
                (self.step == 2 and (counter-1) % 2 == 0):
                mask = np.zeros_like(old_frame)
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
            counter += 1

        cv2.destroyAllWindows()
        cap.release()

        logging.info('Finished running Lucas-Kanade optical flow.')