from datetime import datetime, timedelta
import os
import cv2
import pandas as pd
import pickle
from satprod.data_handlers.img_data import ImgDataset, SatImg, FlowImg
from satprod.configs.config_utils import ImgType, TimeInterval

from tasklog.tasklogger import logging

class Vid():
    '''
    Parent class for all videos.

    Children:
        SatVid: video of the original satellite images.
            Puts together a list of images for the create function.
        FlowVid: video of the images obtained by using optical flow on satellite image videos.
            Puts together a list of images for the create function.
    '''

    def __init__(self):

        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'

        self.videopath = os.path.join(self.root, 'data', 'video')

    def create(self, name: str, img_array, timestamps: [datetime]):
        '''
        Write image series to file as a video.
        The created videofile is found in 'videopath'.
        
        Input parameters:
            name: filename of the video, should be on the form 'filename.avi',
            img_array: array of images of type SatImg or FlowImg

        Output parameters:
            None
        '''
        
        assert len(img_array) > 0, 'There are no images to create a video from.'

        # Get dimensions of the images, assuming all the images in the array are of same size
        size = img_array[0].dim

        # Define frames per second (not important since this is controlled by a waitcall when played)
        video_fps = 1

        # Create videowriter object
        videopath = os.path.join(self.videopath, img_array[0].imgType.value)
        os.makedirs(videopath, exist_ok=True)
        
        out = cv2.VideoWriter(
            os.path.join(videopath, name+'.avi'), 
            cv2.VideoWriter_fourcc(*'DIVX'), video_fps, size
        )

        for img_obj in img_array:
            out.write(img_obj.img)
        out.release()
        
        with open(os.path.join(videopath, name+'-timestamps.pickle'), 'wb') as timestamp_list:
            pickle.dump(timestamps, timestamp_list)
        

    def play(self, name: str, imgType: str, fps: int = 2):
        '''
        Play the video in a separate frame.
        
        Input parameters:
            name: filename of the video.
            fps: frames per second.

        Output parameters:
            None
        '''

        # Define a delay time so that the video is played with the correct fps rate
        delay_time = int(1000/fps)

        # Retrieve video
        cap = cv2.VideoCapture(os.path.join(self.videopath, imgType, name+'.avi'))

        while(True):
            # Capture frame-by-frame
            _, frame = cap.read()

            # Our operations on the frame come here
            if frame is not None:
                # Display the resulting frame
                cv2.imshow('frame', frame)

                # Control how fast the video is played back
                cv2.waitKey(delay_time)
            else:
                # If no frame is found, quit displaying
                cap.release()
                cv2.destroyAllWindows()
                break
        
        # Destroy all windows if there are any left
        cv2.destroyAllWindows()
    
    def delete(self, name: str, imgType: str):
        '''
        Delete video.

        Input parameters:
            name: filename of video (without ending)

        Output parameters:
            None
        '''
        os.remove(os.path.join(self.videopath, imgType, name+'.avi'))
        os.remove(os.path.join(self.videopath, imgType, name+'-timestamps.pickle'))

    def get_img_array(
        self, interval: TimeInterval, imgType: ImgType, img_paths: [str], timestamps: [datetime],
        scale: int=100, simplify: bool=False, step: int=1):
        '''
        Write image series to file as a video.
        The created videofile is found in 'videopath'.
        
        Input parameters:
            interval: TimeInterval giving first and last image of the video,
            imgType: ImgType
            img_paths: paths to the desired images for the video
            timestamps: corresponding timestamps to the images in the img_paths
            scale: percent of image used, scale=100 means no change,
            simplify: whether to make the background black and clouds plain white,
            step: skip step-1 images in the series
        
        Output parameters:
            img_array: array of images for the video
        '''
        
        img_array = []
        for i, img_path in enumerate(img_paths):
            if i%step!=0: continue
            
            # Read image from file
            img = cv2.imread(img_path)
            
            if imgType==ImgType.SAT:
                # Create SatImg object from image
                satimg = SatImg(img, timestamps[i])

                # Simplify image
                if simplify: satimg.simplify()
                
                # Scale image (downscaling if scale < 100)
                satimg.resize(scale_percent=scale)
            
                img_array.append(satimg)
            else:
                img_array.append(FlowImg(img, timestamps[i], imgType))
        
        return img_array
    
    def get_fitted_interval(self, timestamps: [datetime], interval: TimeInterval):
        start_idx = -1
        stop_idx = -1
        for i, date in enumerate(timestamps):
            if date >= interval.start:
                interval.start = date
                start_idx = i
                break
        assert start_idx >= 0
        for i, date in enumerate(timestamps):
            if date > interval.stop:
                interval.end = date
                stop_idx = i
                break
        assert stop_idx >= start_idx
        return start_idx, stop_idx, interval


class SatVid(Vid):
    '''
    Object closely related to SatImg and ImgDataset.
    The member functions allows for creating, playing and deleting
    videos from a given range of satellite images.
    '''

    def __init__(self, name: str, interval: TimeInterval, scale: int=100, simplify: bool=False, step: int=1):
        super().__init__()
        start = interval.start.strftime('%Y-%m-%d %H:%M')
        stop = interval.stop.strftime('%Y-%m-%d %H:%M')
        logging.info(f'SatVid: name {name}, interval {start, stop}, scale {scale}, simplification {simplify}, and step {step}.')
        
        self.imgType = ImgType.SAT
        
        data = ImgDataset(self.imgType)
        
        # find first and last image within the input interval
        start_idx, stop_idx, interval = self.get_fitted_interval(data.timestamps, interval)
        print(start_idx, stop_idx, interval)
        self.img_paths = data.img_paths[start_idx:stop_idx]
        self.timestamps = data.timestamps[start_idx:stop_idx]
        self.interval = interval
        
        # extract the img paths that are within the time interval
        self.name = name
        self.scale = scale
        self.simplify = simplify
        self.step = step

        self.img_array = self.get_img_array(
            self.interval,
            self.imgType,
            self.img_paths,
            self.timestamps,
            self.scale,
            self.simplify,
            self.step
        )

    def save(self):
        '''
        Create a video of the image array and save it to file.
        '''
        logging.info('Saving satellite image video to file.')
        logging.info(f'Video name: {self.name}')
        start = self.interval.start.strftime('%Y-%m-%d %H:%M')
        stop = self.interval.stop.strftime('%Y-%m-%d %H:%M')
        logging.info(f'Video interval: {start, stop}')
        self.create(self.name, self.img_array, self.timestamps)

class FlowVid(Vid):

    def __init__(self, imgType: ImgType, name: str, interval: TimeInterval, step: int=1):
        super().__init__()
        start = interval.start.strftime('%Y-%m-%d %H:%M')
        stop = interval.stop.strftime('%Y-%m-%d %H:%M')
        logging.info(f'FlowVid: {imgType.value}, name {name}, interval {start, stop}, and step {step}.')
        
        self.imgType = imgType

        data = ImgDataset(self.imgType)
        
        # find first and last image within the input interval
        start_idx, stop_idx, interval = self.get_fitted_interval(data.timestamps, interval)
        
        self.img_paths = data.img_paths[start_idx:stop_idx]
        self.timestamps = data.timestamps[start_idx:stop_idx]
        self.interval = interval
        self.name = name
        self.step = step

        self.img_array = self.get_img_array(
            self.interval,
            self.imgType,
            self.img_paths,
            self.timestamps,
            step=self.step
        )

    def save(self):
        '''
        Create a video of the image array and save it to file.
        '''
        logging.info('Saving flow video to file.')
        logging.info(f'Image type: {self.imgType.value}')
        logging.info(f'Video name: {self.name}')
        start = self.interval.start.strftime('%Y-%m-%d %H:%M')
        stop = self.interval.stop.strftime('%Y-%m-%d %H:%M')
        logging.info(f'Video interval: {start, stop}')
        self.create(self.name, self.img_array, self.timestamps)

if __name__=='__main__':
    interval = TimeInterval(datetime(2018,3,20), datetime(2018,3,20,23,59))
    #satvid = SatVid('test', interval, 20)
    #print(satvid.timestamps)
    #satvid.save()
    
    #satvid.play('test', satvid.imgType.value)
    #satvid.delete('test', satvid.imgType.value)