import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

import torch
import torchvision
import torch.utils.data
from torchvision import transforms

import cv2

from satprod.data_handlers.data_utils import datetime2path, path2datetime
from satprod.configs.config_utils import ImgType

from tasklog.tasklogger import logging

class Img():
    '''
    Parent class for all images.
    Takes one image, as a numpy array, as well as 
    the date and time of the image.

    Children:
        SatImg: the original satellite images.
        FlowImg: result image of using optical flow on a SatImg.
    '''

    def __init__(self, img, date: datetime):
        self.img = img
        self.date = date
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.dim = (self.width, self.height)

    def display(self):
        '''
        Display image.
        '''
        plt.imshow(self.img)
        plt.show()

    def asTensor(self):
        '''
        Transfroms image into tensor. (Returns new tensor object, 
        does not change img instance of class.
        '''
        toTensor = transforms.ToTensor()
        return toTensor(self.img)


class SatImg(Img):
    '''
    Adds the ability the be resized and simplified, and the object variable imgType=ImgType.SAT.
    '''

    def __init__(self, img, date: datetime):
        super().__init__(img, date)
        self.imgType = ImgType.SAT
    
    def resize(self, scale_percent=20, interp=cv2.INTER_AREA):
        # find new dimensions for the scaled image
        self.width = int(self.width * scale_percent / 100)
        self.height = int(self.height * scale_percent / 100)
        self.dim = (self.width, self.height)

        # resize image
        self.img = cv2.resize(self.img, self.dim, interpolation = interp)

    def simplify(self, onlyBackground=False):
        '''
        Transforms the image into an image of three colors, depending on how white the
        pixels are. Since clouds are whiter than the background, the background is 
        turned into black. Light clouds are dark gray, and dense clouds are white.

        Input parameters:
            onlyBackground: if True, only the background is made black, the rest isn't changed.
        '''
        # get smoother lines by blurring
        blur = cv2.GaussianBlur(self.img, (15, 15), 2)
        
        # define colors of the new image
        white = [255,255,255]
        black = [0,0,0]
        darkgray = [90,90,90]
        gray = [120,120,120]

        # transform background to black
        lower = np.array(darkgray)
        upper = np.array(white)
        mask = cv2.inRange(blur, lower, upper)
        masked_img = cv2.bitwise_and(self.img, self.img, mask=mask)

        # color the clouds depending on how white they are
        if not onlyBackground:
            height, width, _ = masked_img.shape

            for x in range(0,width):
                for y in range(0,height):
                    channels_xy = masked_img[y,x]
                    if all(channels_xy > gray):
                        masked_img[y,x] = gray
                    else:
                        if all(channels_xy > black):
                            masked_img[y,x] = darkgray
        
        # change the original image to the masked
        self.img = masked_img

class FlowImg(Img):
    '''
    An image represeting an optical flow result.

    Parameters:
        img: array of pixels (also in Img class)
        date: date and time image is taken (also in Img class)
        imgType: represents different of methods
    '''
    def __init__(self, img, date: datetime, imgType: ImgType):
        super().__init__(img, date)
        assert imgType is not ImgType.SAT, 'The image should be a SatImg object, not FlowImg.'
        self.imgType = imgType

class ImgDataset(torch.utils.data.Dataset):
    '''
    Set of all the satellite images, including 
    information on the time they were taken.
    '''
    
    def __init__(self, imgType: ImgType, normalize: bool=False, upscale: bool=False, grayscale: bool=False):
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'

        self.upscale = upscale
        self.normalize = normalize
        self.grayscale = grayscale
        
        self.imgType = imgType
        if self.imgType==ImgType.GRID:
            self.flag = 0
        else:
            self.flag = 1

        # define where to get the images depending on the image type
        folder=f'img/{imgType.value}'

        self.imgroot = os.path.join(self.root, 'data', folder)
        
        try:
            self.img_paths = []
            for img in torchvision.datasets.ImageFolder(self.imgroot).imgs:
                self.img_paths.append(img[0])
            
            self.timestamps = [
                path2datetime(self.img_paths[i][len(self.imgroot):]) for i in range(len(self.img_paths))
            ]
        except:
            logging.warning('Cannot call dataset of dense and sparse optical flow images when there are none.')
            exit()

    def __getitem__(self, idx: int):
        # read image from path, and get its corresponding timestamp
        if self.grayscale:
            img = cv2.imread(self.img_paths[idx], 0)
        else:
            img = cv2.imread(self.img_paths[idx], 1)
        if self.normalize:
            img = img/255.0
        if self.upscale: 
            img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
        
        date = self.timestamps[idx]
        

        # return the correct imagetype
        if self.imgType==ImgType.SAT: 
            return SatImg(img, date)
        else: 
            return FlowImg(img, date, self.imgType)

    def __len__(self):
        return len(self.img_paths)
    
    def getDateIdx(self, date: datetime) -> int:
        try:
            return self.timestamps.index(date)
        except ValueError:
            return np.nan

if __name__=='__main__':
    data = ImgDataset(ImgType('sat'))
    print(data[0].date)