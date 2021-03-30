import os
import cv2
import numpy as np
import pandas as pd
import json
import pickle
from datetime import datetime, timedelta

from satprod.data_handlers.img_data import ImgDataset
from satprod.data_handlers.video import FlowVid
from satprod.configs.config_utils import ImgType, TimeInterval, read_yaml
from satprod.data_handlers.data_utils import scaler, date2interval

from tasklog.tasklogger import logging

class OpticalFlow():
    '''
    Performs optical flow on a series of satellite images, and saves them
    using the same struture as the satellite images are stored.

    Methods supported:
        Farneback (dense)
        Dual TVL1 (dense)
        RLOF (dense)
        Lucas-Kanade (dense) 
        Lucas-Kanade (sparse)
    '''

    def __init__(self, satVidName: str, step: int=1, scale: int=100, mean_limit: int=80):
        '''
        Input parameters:
            satVidName: filename of the satellite video to do optical flow on
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
        self.mean_limit = mean_limit

        # Get satellite image paths to mimic them
        data = ImgDataset(ImgType.SAT)
        
        satvideopath = os.path.join(self.videopath, 'sat')
        with open(os.path.join(satvideopath, self.name+'-timestamps.pickle'), 'rb') as timestamp_list:
            timestamps = pickle.load(timestamp_list)

        self.timestamps = timestamps #data.timestamps[self.start_idx+step:self.stop_idx+step]
        self.interval = TimeInterval(start=timestamps[0], stop=timestamps[-1])
        
        #self.getDateIdx = data.getDateIdx
        self.start_idx = data.getDateIdx(self.interval.start)
        self.stop_idx = data.getDateIdx(self.interval.stop)
        self.img_paths = data.img_paths[self.start_idx:self.stop_idx+1]
        
        # initialise dataframes for storing results of dense optical flow
        self.direction_pixel_df = pd.DataFrame(
            columns = ['vals', 'yvik', 'bess', 'skom'],
            index = self.timestamps
        )
        self.direction_median_df = self.direction_pixel_df.copy()
        self.magnitude_pixel_df = self.direction_pixel_df.copy()
        self.magnitude_median_df = self.direction_pixel_df.copy()
        
        # positions of parks in full scale image (see notebook park_pixel_positions.ipynb)
        self.positions = {'vals': (200,460), 'yvik': (75, 580), 'bess': (135, 590), 'skom': (140, 600)}
        
        # update positions to be correct for the scaled image
        for key, value in self.positions.items():
            self.positions[key] = (
                int(np.round(value[0]*self.scale/100)), int(np.round(value[1]*self.scale/100)))

        logging.info(f'Satellite video: {self.name}')

    def denseflow(self, imgType: str, params: dict=None, save: bool=False, play: bool=False, fps: int=6):
        imgType = ImgType(imgType)
        timestr = '-'.join(self.name.split('-')[:4])
        flow_vid_name = f'{timestr}-{str(int(15*self.step))}min-{self.scale}sc-{imgType.value}'

        # Parameters for Farneback optical flow
        if params is None and imgType==ImgType.FB_DENSE:
            params = {
                'winsize' : 29,
                'levels' : 2,
                'poly_n' : 13,
                'poly_sigma' : 2.7,
                'pyr_scale' : 0.4,
                'iterations' : 3,
                'flags': 0
            }
        
        # Parameters for Dual TVL1 optical flow
        if params is None and imgType==ImgType.DTVL1_DENSE:
            params = { 
                'tau' : 0.25,
                'lambda' : 0.25,
                'theta' : 0.3,
                'nscales' : 5,
                'warps' : 5,
                'epsilon' : 0.01,
                'innnerIterations' : 30,
                'outerIterations' : 10,
                'scaleStep' : 0.8,
                'gamma' : 0.0,
                'medianFiltering' : 5,
                #'useInitialFlow' : False 
            }
        
        # Parameters for Dense Lucas-Kanade optical flow
        if params is None and imgType==ImgType.LK_DENSE:
            params = {
                'k' : 64,
                'sigma' : 0.1,
                'use_post_proc' : True,
                'fgs_lambda' : 500.0,
                'fgs_sigma' : 1.5,
                'grid_step' : 2,
            }
        
        # Parameters for RLOF optical flow
        if params is None and imgType==ImgType.RLOF_DENSE:
            '''optimal params, cause segmentation fault
            params = {
                'forwardBackwardThreshold' : 2.5,
                'epicK': 128, 
                'epicSigma': 0.05,
                'epicLambda' : 200.0,
                'fgsLambda' : 500.0,
                'fgsSigma' : 1.5,
                'use_variational_refinement' : False,
                'use_post_proc' : True
            }'''
            '''default params, cause segmentation fault
            params = {
                'forwardBackwardThreshold' : 1.0,
                'epicK': 128, 
                'epicSigma': 0.05,
                'epicLambda' : 999.0,
                'fgsLambda' : 500.0,
                'fgsSigma' : 1.5,
                'use_variational_refinement' : False,
                'use_post_proc' : True
            }'''
            params = None
        
        if imgType==ImgType.FB_DENSE: self.__farneback(params=params)
        elif imgType==ImgType.DTVL1_DENSE: self.__dualTVL1(params=params)
        elif imgType==ImgType.LK_DENSE: self.__denseLucasKanade(params=params)
        elif imgType==ImgType.RLOF_DENSE: self.__RLOF(params=params)
        else:
            logging.warning(f'{imgType.value} does not correspond to any dense flow algorithms.')
            exit()

        if play:
            flowVid = FlowVid(imgType, flow_vid_name, self.interval, self.step)
            flowVid.save()
            flowVid.play(flow_vid_name, imgType.value, fps=fps)
            flowVid.delete(flow_vid_name, imgType.value)
        if save:
            flowVid = FlowVid(imgType, flow_vid_name, self.interval, self.step)
            flowVid.save()

    def sparseflow(self, feature_params=None, skl_params=None, save: bool=False, play=False, fps: int=6):
        # Parameters for ShiTomasi corner detection
        if feature_params is None:
            feature_params = dict( maxCorners = 100,
                                qualityLevel = 0.05,
                                minDistance = 14,
                                blockSize = 7 )
        
        # Parameters for Lucas Kanade optical flow
        if skl_params is None:
            slk_params = dict( winSize  = (30,30),
                            maxLevel = 2,
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.__sparseLucasKanade(feature_params, slk_params)

        for imgType in [ImgType('lk_sparse'), ImgType('lk_sparsemask')]:
            timestr = '-'.join(self.name.split('-')[:4])
            flow_vid_name = f'{timestr}-{str(int(15*self.step))}min-{self.scale}sc-{imgType.value}'
            
            if play:
                flowVid = FlowVid(imgType, flow_vid_name, self.interval, self.step)
                flowVid.save()
                flowVid.play(flow_vid_name, imgType.value, fps=fps)
                flowVid.delete(flow_vid_name, imgType.value)
            if save:
                flowVid = FlowVid(imgType, flow_vid_name, self.interval, self.step)
                flowVid.save()
    
    def __get_degrees(self, ang_img) -> dict:
        '''
        Extract degrees from angle image obtained by dense optical flow and cartToPolar.
        '''
        area_width = 50 # pixels to every side of the position (before scaling)
        area_constant = int(area_width*self.scale/100) # pixels to every side of the position (after scaling). scale=20 -> area_constant=10
        ang = ang_img*180/np.pi/2
        
        degrees_pixel = {'vals': 0, 'yvik': 0, 'bess': 0, 'skom': 0}
        degrees_median = {'vals': 0, 'yvik': 0, 'bess': 0, 'skom': 0}
        for key, value in self.positions.items():
            degrees_pixel[key] = 360-2*ang[value[0],value[1]]
            degrees_median[key] = 360-2*np.median(np.ravel(ang[value[0]-area_constant:value[0]+area_constant,value[1]-area_constant:value[1]+area_constant]))
        return degrees_pixel, degrees_median

    def __get_magnitude(self, mag_img) -> dict:
        '''
        Extract magnutide/speed from magnitude image obtained by dense optical flow and cartToPolar.
        '''
        area_width = 50 # pixels to every side of the position (before scaling)
        area_constant = int(area_width*self.scale/100) # pixels to every side of the position (after scaling). scale=20 -> area_constant=10
        
        magnitudes_pixel = {'vals': 0, 'yvik': 0, 'bess': 0, 'skom': 0}
        magnitudes_median = {'vals': 0, 'yvik': 0, 'bess': 0, 'skom': 0}
        for key, value in self.positions.items():
            magnitudes_pixel[key] = mag_img[value[0],value[1]]
            magnitudes_median[key] = np.median(np.ravel(mag_img[value[0]-area_constant:value[0]+area_constant,value[1]-area_constant:value[1]+area_constant]))
        return magnitudes_pixel, magnitudes_median

    def __create_image_folder(self, name: str) -> [str]:
        paths = []
        for i in range(0, len(self.img_paths), self.step):
            path = self.img_paths[i].replace('/img/sat/', f'/img/{name}/')
            paths.append(path)
            os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
        return paths
    
    def __farneback(self, params: dict):
        self.__dense(
            method = cv2.calcOpticalFlowFarneback,
            imgType = ImgType.FB_DENSE,
            params = params,
            gray = True
        )
        
    def __dualTVL1(self, params: dict):
        dtvl1 = cv2.optflow.DualTVL1OpticalFlow_create(**params)
        self.__dense(
            method = dtvl1.calc,
            imgType = ImgType.DTVL1_DENSE,
            params = params,
            gray = True
        )
        
    def __denseLucasKanade(self, params: dict):
        self.__dense(
            method = cv2.optflow.calcOpticalFlowSparseToDense,
            imgType = ImgType.LK_DENSE,
            params = params,
            gray = True
        )
        
    def __RLOF(self, params: dict):
        self.__dense(
            method = cv2.optflow.calcOpticalFlowDenseRLOF,
            imgType = ImgType.RLOF_DENSE,
            params = params,
            gray = False
        )

    def __dense(self, method, imgType: ImgType, params=None, gray: bool=False):
        logging.info(f'Running {imgType.value} optical flow.')
        logging.info(f'Params:\n {params}.')
        logging.info(f'Step: {self.step*15} minutes.')
        logging.info(f'{self.timestamps[0]} to {self.timestamps[-1]}.')
        
        # get limited interval for where optical flow is useful
        [of_start_limit, of_stop_limit] = date2interval(self.timestamps[0])
        
        # initialise and create where the results should be stored
        self.flow_img_paths = self.__create_image_folder(imgType.value)
    
        # Read the video and first frame
        cap = cv2.VideoCapture(os.path.join(self.videopath, 'sat', self.name+'.avi'))
        ret, old_frame = cap.read()
        timestamp_counter = 0
        
        # create HSV & make Value a constant
        hsv = np.zeros_like(old_frame)
        hsv[..., 1] = 255
        
        # Preprocessing for exact method
        if gray: old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
        # store the previous flow to use when computing the current
        #old_flow = None

        while True:
            # Read the next frame
            ret, new_frame = cap.read()
            timestamp_counter += 1
            
            if not ret:
                break
            img_dates = (self.timestamps[timestamp_counter-1], self.timestamps[timestamp_counter])
            if img_dates[1].minute != 0: continue
            if img_dates[1] <= of_start_limit+timedelta(hours=1) or \
                img_dates[1] >= of_stop_limit - timedelta(hours=1):
                logging.info(f'Skipping nighttime image at {img_dates}.')
                continue
            if ((img_dates[1]-img_dates[0]).seconds//60) > 30:
                logging.info(f'Skipping due to lack of intermediate images.')
                continue
            if np.mean(new_frame) < self.mean_limit:
                logging.info(f'Skipping due to low image mean: {np.mean(new_frame)}, mean limit: {self.mean_limit}')
                continue
            
            logging.info(f'{imgType.value} optflow for {img_dates}.')
            # Preprocessing for exact method
            if gray: new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            
            try:
                # Calculate Optical Flow
                if imgType==ImgType.DTVL1_DENSE:
                    # does not take params as argument because they were already defined when creating the method
                    flow = method(old_frame, new_frame, None)
                elif imgType==ImgType.RLOF_DENSE:
                    # avoid segmentation fault
                    flow = method(old_frame, new_frame, None)
                else:
                    flow = method(old_frame, new_frame, None, **params)
            except:
                logging.info(f'Flow failed at {img_dates}. Continuing to next image.')
                continue
            #old_flow = np.copy(flow)
            # Encoding: convert the algorithm's output into Polar coordinates
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Use Hue and Value to encode the Optical Flow
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convert HSV image into BGR for demo
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Update the previous frame
            old_frame = new_frame
            
            # registering wind direction estimate at park positions
            degrees_pixel, degrees_median = self.__get_degrees(ang)
            for key, value in degrees_pixel.items():
                self.direction_pixel_df.loc[f'{self.timestamps[timestamp_counter]}'][key] = value
            for key, value in degrees_median.items():
                self.direction_median_df.loc[f'{self.timestamps[timestamp_counter]}'][key] = value

            # registering wind speed estimate at park positions
            magnitudes_pixel, magnitudes_median = self.__get_magnitude(mag)
            for key, value in magnitudes_pixel.items():
                self.magnitude_pixel_df.loc[f'{self.timestamps[timestamp_counter]}'][key] = value
            for key, value in magnitudes_median.items():
                self.magnitude_median_df.loc[f'{self.timestamps[timestamp_counter]}'][key] = value
                
            # save result image to data folder
            cv2.imwrite(self.flow_img_paths[timestamp_counter],bgr)
            
            

        # done using video
        cap.release()
        
        # store results at park positions to csv files named after the satellite image video name
        path = f'{self.root}/data/of_num_results/{imgType.value}'
        os.makedirs(path, exist_ok=True)
        
        filename = self.name.replace('-sat', '')
        
        self.direction_pixel_df.columns = ['vals_deg_pixel', 'yvik_deg_pixel', 'bess_deg_pixel', 'skom_deg_pixel']
        self.direction_median_df.columns = ['vals_deg_median', 'yvik_deg_median', 'bess_deg_median', 'skom_deg_median']
        self.magnitude_pixel_df.columns = ['vals_mag_pixel', 'yvik_mag_pixel', 'bess_mag_pixel', 'skom_mag_pixel']
        self.magnitude_median_df.columns = ['vals_mag_median', 'yvik_mag_median', 'bess_mag_median', 'skom_mag_median']
        
        of_results_df = pd.concat([self.direction_pixel_df, self.direction_median_df, self.magnitude_pixel_df, self.magnitude_median_df], axis=1)
        of_results_df.to_csv(f'{path}/{filename}.csv')
        
        dmp = json.dumps(params)
        f = open(f'{path}/{filename}_params.json','w')
        f.write(dmp)
        f.close()

        #logging.info(f'Finished running {imgType.value} optical flow.')


    def __sparseLucasKanade(self, feature_params, lk_params):
        logging.info('Running lk_sparse optical flow.')
        logging.info(f'Feature params:\n {feature_params}.')
        logging.info(f'Params:\n {lk_params}.')
        logging.info(f'Step: {self.step*15} minutes.')
        logging.info(f'{self.timestamps[0]} to {self.timestamps[-1]}.')

        # initialise and create where the results should be stored
        self.sparse_img_paths = self.__create_image_folder('lk_sparse')
        self.sparsemask_img_paths = self.__create_image_folder('lk_sparsemask')

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
                #logging.info(f'Performing sparse optical flow at for time {self.timestamps[counter]}')
                
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

        #logging.info('Finished running lk_sparse optical flow.')

'''
DUAL TVL1 OPTICAL FLOW PARAMETERS
default parameters:
double 	tau = 0.25, # tau is the timestep used by the numerical solver. It can be set to any value
    less than 0.125 and convergence is guaranteed, but empirically it can be set as high as 0.25
    for faster convergence.
double 	lambda = 0.15, # lambda is the most important parameter; it sets the weight of the 
    smoothness term in the energy. The ideal value of lambda will vary depending on the image 
    sequence, with smaller values corresponding to smoother solutions. The default value for 
    lambda is 0.15.
double 	theta = 0.3, # The parameter theta is called (by the authors of the original paper) the 
    “tightness parameter.” This is the parameter that couples the two stages of the overall 
    solver. In principle it should be very small, but the algorithm is stable for a wide range 
    of values. The default tightness parameter is 0.30.
int 	nscales = 5, # The number of scales in the image pyramid is set by nscales.
int 	warps = 5, # For each scale, the number of warps is the number of times ∇ It +1(x→ + u→0) 
    and It +1(x→ + u→0) are computed per scale. This parameter allows a trade-off between speed 
    (fewer warps) and accuracy (more warps). By default there are five scales and five warps per 
    scale.
double 	epsilon = 0.01, #epsilon is the stopping criterion used by the numerical solver.
int 	innnerIterations = 30, # Additionally there is an iterations criterion, which sets the 
    maximum number of iterations allowed.
int 	outerIterations = 10,
double 	scaleStep = 0.8,
double 	gamma = 0.0,
int 	medianFiltering = 5,
bool 	useInitialFlow = false 
'''


'''
DENSE LUCAS KANADE OPTICAL FLOW PARAMETERS
#cv2.optflow.calcOpticalFlowSparseToDense
flow = cv2.optflow.calcOpticalFlowSparseToDense(
    from, to[, flow[, grid_step[, k[, sigma[, use_post_proc[, fgs_lambda[, fgs_sigma]]]]]]])
grid_step: stride used in sparse match computation. Lower values usually result in higher 
    quality but slow down the algorithm.
k: number of nearest-neighbor matches considered, when fitting a locally affine model. 
    Lower values can make the algorithm noticeably faster at the cost of some quality degradation.
sigma: parameter defining how fast the weights decrease in the locally-weighted affine fitting. 
    Higher values can help preserve fine details, lower values can help to get rid of the noise 
    in the output flow.

Default values:
int grid_step = 8, int k = 128, float sigma = 0.05f,
bool use_post_proc = true, float fgs_lambda = 500.0f,
float fgs_sigma = 1.5f 
'''

'''
DENSE SIMPLE FLOW OPTICAL FLOW PARAMETERS
# calcOpticalFlowSF(from, to, layers, averaging_block_size, max_flow[, flow]) -> flow
# These three parameters—layers, averaging_block_size, and max_flow (respectively)—
# can reasonably be set to 5, 11, and 20.
params = {layers: 5, averaging_block_size: 11, max_flow: 20}
flow = cv2.optflow.calcOpticalFlowSF(frame_0, frame_1, old_flow, **params)
'''
        