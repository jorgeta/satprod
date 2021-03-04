from datetime import datetime, timedelta
import os
import cv2
from satprod.data_handlers.img_data import ImgDataset, SatImg, DenseFlowImg, SparseFlowImg, SparseFlowMaskImg
from satprod.configs.config_utils import ImgType, TimeInterval

class Vid():

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
            img_array: array of images of type SatImg, DenseFlowImg or SparseImg

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


class SatVid(Vid):
    '''
    Object closely related to SatImg and ImgDataset.
    The member functions allows for creating, playing and deleting
    videos from a given range of satellite images.
    '''

    def __init__(self, name: str, interval: TimeInterval, 
                 scale: int=100, simplify: bool=False, step: int=1):
        super().__init__()
        
        self.imgType = ImgType.SAT
        
        data = ImgDataset(self.imgType)

        self.img_paths = data.img_paths
        self.timestamps = data.timestamps
        self.getDateIdx = data.getDateIdx

        del data

        self.name = name
        self.interval = interval
        self.scale = scale
        self.simplify = simplify
        self.step = step

        self.img_array = self.get_img_array()

        self.create(self.name, self.img_array)
    
    def get_img_array(self):
        '''
        Write image series to file as a video.
        The created videofile is found in 'videopath'.
        
        Input parameters:
            interval: TimeInterval giving first and last image of the video,
            scale: percent of image used, scale=100 means no change,
            simplify: whether to make the background black and clouds plain white,
            step: skip step-1 images in the series
        
        Output parameters:
            img_array: array of images for the video
        '''

        # Get indices of images using the dates
        self.start_idx = self.getDateIdx(self.interval.start)
        self.stop_idx = self.getDateIdx(self.interval.stop)

        # Create array of the images that will become the video
        img_array = []
        for i in range(self.start_idx, self.stop_idx+1, self.step):
            # Read image from file
            img = cv2.imread(self.img_paths[i])

            # Create SatImg object from image
            satimg = SatImg(img, self.timestamps[i])

            # Simplify image
            if self.simplify: satimg.simplify()
            
            # Scale image (downscaling if scale < 100)
            satimg.resize(scale_percent=self.scale)
            
            img_array.append(satimg)

        return img_array

class FlowVid(Vid):

    def __init__(self, imgType: ImgType, name: str, interval: TimeInterval, step: int=1):
        super().__init__()
        
        self.imgType = imgType

        data = ImgDataset(self.imgType)

        self.img_paths = data.img_paths
        self.timestamps = data.timestamps
        self.getDateIdx = data.getDateIdx

        del data

        self.name = name
        self.interval = interval
        self.step = step

        self.img_array = self.get_img_array()

        self.create(self.name, self.img_array)
    
    def get_img_array(self):
        '''
        Write image series to file as a video.
        The created videofile is found in 'videopath'.
        
        Input parameters:
            interval: TimeInterval giving first and last image of the video,
            step: skip step-1 images in the series
        
        Output parameters:
            img_array: array of images for the video
        '''

        # Get indices of images using the dates
        self.start_idx = self.getDateIdx(self.interval.start+timedelta(minutes=15)*self.step)
        self.stop_idx = self.getDateIdx(self.interval.stop)

        # Create array of the images that will become the video
        img_array = []
        for i in range(self.start_idx, self.stop_idx+1, self.step):
            # Read image from file
            img = cv2.imread(self.img_paths[i])

            # Create DenseFlowImg/SparseFlowImg object from image and add to list
            if self.imgType==ImgType.DENSE:
                img_array.append(DenseFlowImg(img, self.timestamps[i]))
            elif self.imgType==ImgType.SPARSE:
                img_array.append(SparseFlowImg(img, self.timestamps[i]))
            elif self.imgType==ImgType.SPARSEMASK:
                img_array.append(SparseFlowMaskImg(img, self.timestamps[i]))
            else:
                print('ERROR: Use SatVid object for ImgType.SAT images.')
                exit()

        return img_array


if __name__=='__main__':
    start = datetime(2019,6,3,3)
    stop = datetime(2019,6,3,21)
    fps = 4
    name = '3jun2019-60min-100sc-sat'

    interval = TimeInterval(start, stop)
    v = SatVid(name=name, interval=interval, step=4)
    v.play(name=name, fps=fps)

'''if __name__=='__main__':
    v = SatVid()
    
    start = datetime(2019,6,3,3)#datetime(2018,6,28,6)
    stop = datetime(2019,6,3,21)#datetime(2018,6,28,18)

    name = f'test60min1.avi'
    #v.delete(name)

    scale = 15

    v.create(name, start, stop, True, scale)
    v.play(name, 1.5)

    of = OpticalFlow(root, name)
    #of.lukasKanade(f'.')

    foldername = 'ex60min10scale2'
    os.makedirs(f'{foldername}', exist_ok=True)
    of.farneback(f'{foldername}')

    #v.delete(name)

    for i in range(12):
        start += timedelta(hours=1)
        stop = start + timedelta(hours=2)
        name = f'take_{i}.avi'
        #v.delete(name)
        v.create(name, start, stop, True, scale)
        #v.play(name, 0.5)

        of = OpticalFlow(root, name)
        os.makedirs(f'{foldername}/{i}', exist_ok=True)
        of.lukasKanade(f'{foldername}/{i}')
        v.delete(name)'''