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

    Coming:
        Horn-Schunck (?)
        cv2.SparseOpticalFlow.calc()
        cv2.DualTVL1OpticalFlow.calc()
    '''

    def __init__(self, satVidName: str, interval: TimeInterval, step: int=1, scale: int=100):
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

        self.dense_img_paths = []
        self.sparse_img_paths = []
        self.sparsemask_img_paths = []
        for i in range(0, len(self.img_paths), step):
            new_dense_path = self.img_paths[i].replace('/img/', f'/dense_flow/')
            new_sparse_path = self.img_paths[i].replace('/img/', f'/sparse_flow/')
            new_sparsemask_path = self.img_paths[i].replace('/img/', f'/sparse_flow_mask/')
            self.dense_img_paths.append(new_dense_path)
            self.sparse_img_paths.append(new_sparse_path)
            self.sparsemask_img_paths.append(new_sparsemask_path)
            os.makedirs('/'.join(new_dense_path.split('/')[:-1]), exist_ok=True)
            os.makedirs('/'.join(new_sparse_path.split('/')[:-1]), exist_ok=True)
            os.makedirs('/'.join(new_sparsemask_path.split('/')[:-1]), exist_ok=True)
        
        self.direction_df = pd.DataFrame(
            columns = ['vals', 'yvik', 'bess', 'skom'],
            index = self.timestamps
        )
        #logging.info(self.direction_df)

        logging.info(f'Object for optical flow on {self.name} initialised.')

    def get_degrees(self, ang_img):

        # positions of parks in full scale image
        positions = {'vals': (200,460), 'yvik': (75, 580), 'bess': (135, 590), 'skom': (140, 600)}
        
        # update positions to be correct for the scaled image
        for key, value in positions.items():
            positions[key] = (int(np.round(value[0]*self.scale/100)), int(np.round(value[1]*self.scale/100)))
        
        # extract degrees from angle image obtained by Farneback and cartToPolar
        degrees = {'vals': 0, 'yvik': 0, 'bess': 0, 'skom': 0}
        for key, value in positions.items():
            degrees[key] = int(360-2*ang_img[value[0],value[1]])
        return degrees

    def farneback(self, fb_params):
        logging.info('Running Farneback optical flow.')
        logging.info(f'Images every {self.step*15} minutes between {self.timestamps[0]} and {self.timestamps[-1]} are used.')
        
        cap = cv2.VideoCapture(os.path.join(self.videopath, self.name+'.avi'))

        _, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        counter = 0
        while(1):
            #print(counter)
            _, frame2 = cap.read()
            if frame2 is not None: 
                logging.info(f'{self.timestamps[counter]}')
                next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                #cv2.imwrite(f'gray_{counter}.png', frame2)

                #flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow = cv2.calcOpticalFlowFarneback(prvs, next, None, **fb_params)
                
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

                #cv2.imwrite(f'ang_{counter}.png', ang*180/np.pi/2)
                degrees = self.get_degrees(ang*180/np.pi/2)
                for key, value in degrees.items():
                    self.direction_df.loc[f'{self.timestamps[counter]}'][key] = value

                #logging.info(degrees)

                '''cv2.imwrite(f'flow0_{counter}.png',scaler(flow[...,0], 0, 255))
                cv2.imwrite(f'flow1_{counter}.png',scaler(flow[...,1], 0, 255))
                cv2.imwrite(f'mag_{counter}.png',scaler(mag, 0, 255))
                cv2.imwrite(f'ang_{counter}.png',scaler(ang, 0, 255))'''

                #cv2.imshow('frame2',rgb)
                #k = cv2.waitKey(30) & 0xff
                #if k == 27:
                    #break
                #elif k == ord('s'):
                #cv2.imwrite(f'{save}/opticalfb_{counter}.png',frame2)
                #cv2.imwrite(f'{save}/opticalhsv_{counter}.png',rgb)
                cv2.imwrite(self.dense_img_paths[counter],rgb)
                prvs = next
            else:
                #cv2.imwrite('opticalfb.png',frame2)
                #cv2.imwrite('opticalhsv.png',rgb)
                cap.release()
                cv2.destroyAllWindows()
                break
            counter += 1

        cap.release()
        cv2.destroyAllWindows()
        self.direction_df.to_csv(f'{self.root}/data/{self.name}.csv')
        logging.info('Finished running Farneback optical flow.')

    def lukasKanade(self, feature_params, lk_params):
        cap = cv2.VideoCapture(os.path.join(self.videopath, self.name+'.avi'))

        '''# params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,#100,
                            qualityLevel = 0.05,#0.3,
                            minDistance = 14,
                            blockSize = 7 )

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),#(15,15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))'''

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
            _, frame = cap.read()
            if frame is not None:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                # Select good points
                if p1 is None:
                    print('ERROR: Lucas Kanade method failed, could ot find any new good points.')
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

                #cv2.imshow('frame',img)
                #cv2.imwrite(f'{save}/opticalfeatures_{counter}.png',img)
                #cv2.imwrite(f'{save}/opticalmask_{counter}.png',mask)
                cv2.imwrite(self.sparse_img_paths[counter],img)
                cv2.imwrite(self.sparsemask_img_paths[counter],mask)

                #k = cv2.waitKey(30) & 0xff
                #if k == 27:
                #    break

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)
            else:
                break
            if (self.step == 4) or \
                (self.step == 1 and (counter+1) % 4 == 0) or \
                (self.step == 2 and (counter-1) % 2 == 0):
                mask = np.zeros_like(old_frame)
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
            counter += 1

        cv2.destroyAllWindows()
        cap.release()