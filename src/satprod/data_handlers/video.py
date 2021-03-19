from datetime import datetime, timedelta
import os
import cv2
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

        self.videopath = os.path.join(self.root, 'data/video')

    def create(self, name: str, img_array):
        '''
        Write image series to file as a video.
        The created videofile is found in 'videopath'.
        
        Input parameters:
            name: filename of the video, should be on the form 'filename.avi',
            img_array: array of images of type SatImg or FlowImg

        Output parameters:
            None
        '''

        # Get dimensions of the images, assuming all the images in the array are of same size
        size = img_array[0].dim

        # Define frames per second (not important since this is controlled by a waitcall when played)
        video_fps = 1

        # Create videowriter object
        out = cv2.VideoWriter(
            os.path.join(self.videopath, name+'.avi'), cv2.VideoWriter_fourcc(*'DIVX'), video_fps, size
        )

        for img_obj in img_array:
            out.write(img_obj.img)
        out.release()

    def play(self, name: str, fps: int = 2):
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
        cap = cv2.VideoCapture(os.path.join(self.videopath, name+'.avi'))

        while(True):
            # Capture frame-by-frame
            _, frame = cap.read()

            # Our operations on the frame come here
            if frame is not None: 
                # Make grayscale
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
    
    def delete(self, name: str):
        '''
        Delete video.

        Input parameters:
            name: filename of video (without ending)

        Output parameters:
            None
        '''
        os.remove(os.path.join(self.videopath, name+'.avi'))

    def get_img_array(
        self, interval: TimeInterval, imgType: ImgType, getDateIdx, img_paths, timestamps,
        scale: int=100, simplify: bool=False, step: int=1):
        '''
        Write image series to file as a video.
        The created videofile is found in 'videopath'.
        
        Input parameters:
            interval: TimeInterval giving first and last image of the video,
            imgType: ImgType (SAT / DENSE / SPARSE / SPARSEMASK)
            getDateIdx: function for looking up index in dataset of images corresponding to imgType
            img_paths: paths to the desired images for the video
            timestamps: corresponding timestamps to the images in the img_paths
            scale: percent of image used, scale=100 means no change,
            simplify: whether to make the background black and clouds plain white,
            step: skip step-1 images in the series
        
        Output parameters:
            img_array: array of images for the video
        '''

        # Get indices of images using the dates
        if imgType is not ImgType.SAT:
            self.start_idx = getDateIdx(interval.start+timedelta(minutes=15)*step)
        else:
            self.start_idx = getDateIdx(interval.start)
        self.stop_idx = getDateIdx(interval.stop)

        # Create array of the images that will become the video
        img_array = []
        for i in range(self.start_idx, self.stop_idx+1, step):
            # Read image from file
            img = cv2.imread(img_paths[i])
            
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


class SatVid(Vid):
    '''
    Object closely related to SatImg and ImgDataset.
    The member functions allows for creating, playing and deleting
    videos from a given range of satellite images.
    '''

    def __init__(self, name: str, interval: TimeInterval, 
                 scale: int=100, simplify: bool=False, step: int=1):
        super().__init__()
        start = interval.start.strftime('%Y-%m-%d %H:%M')
        stop = interval.stop.strftime('%Y-%m-%d %H:%M')
        logging.info(f'Initialising SatVid object with name {name}, interval {start, stop}, scale {scale}, simplification {simplify}, and step {step}.')
        
        self.imgType = ImgType.SAT
        
        data = ImgDataset(self.imgType)

        self.img_paths = data.img_paths
        self.timestamps = data.timestamps
        self.getDateIdx = data.getDateIdx

        self.name = name
        self.interval = interval
        self.scale = scale
        self.simplify = simplify
        self.step = step

        self.img_array = self.get_img_array(
            self.interval,
            self.imgType,
            self.getDateIdx,
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
        self.create(self.name, self.img_array)

class FlowVid(Vid):

    def __init__(self, imgType: ImgType, name: str, interval: TimeInterval, step: int=1):
        super().__init__()
        start = interval.start.strftime('%Y-%m-%d %H:%M')
        stop = interval.stop.strftime('%Y-%m-%d %H:%M')
        logging.info(f'Initialising FlowVid object with image type {imgType.value}, name {name}, interval {start, stop}, and step {step}.')
        
        self.imgType = imgType

        data = ImgDataset(self.imgType)

        self.img_paths = data.img_paths
        self.timestamps = data.timestamps
        self.getDateIdx = data.getDateIdx

        self.name = name
        self.interval = interval
        self.step = step

        self.img_array = self.get_img_array(
            self.interval,
            self.imgType,
            self.getDateIdx,
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
        self.create(self.name, self.img_array)