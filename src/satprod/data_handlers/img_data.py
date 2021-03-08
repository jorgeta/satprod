import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

import torch
import torchvision
import torch.utils.data

import cv2

from satprod.data_handlers.data_utils import datetime2path, path2datetime
from satprod.configs.config_utils import ImgType

class Img():
    '''
    Parent class for all images.
    Takes one image, as a numpy array, as well as 
    the date and time of the image.

    Children:
        SatImg: the original satellite images.
        DenseFlowImg: the result of using dense optical flow on the original images.
        SparseFlowImg: the result of using sparse optical flow on the original images.
    '''

    def __init__(self, img, date: datetime):
        self.img = img
        self.date = date
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.dim = (self.width, self.height)

    def display(self):
        plt.imshow(self.img)
        plt.show()

    def asTensor(self):
        toTensor = torchvision.transforms.ToTensor()
        return toTensor(self.img)


class SatImg(Img):
    '''
    Child of the Img parent class.
    Adds the ability the be resized and simplified.
    '''
    
    def resize(self, scale_percent=5, interp=cv2.INTER_AREA):
        width = int(self.width * scale_percent / 100)
        height = int(self.height * scale_percent / 100)
        dim = (width, height)
        self.img = cv2.resize(self.img, dim, interpolation = interp)
        self.width = width
        self.height = height
        self.dim = dim

    def simplify(self, onlyBackground=False):
        blur = cv2.GaussianBlur(self.img, (15, 15), 2)

        white = [255,255,255]
        black = [0,0,0]
        darkgray = [90,90,90]
        gray = [120,120,120]

        lower = np.array(darkgray)
        upper = np.array(white)
        mask = cv2.inRange(blur, lower, upper)
        masked_img = cv2.bitwise_and(self.img, self.img, mask=mask)

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
        
        self.img = masked_img

class FlowImg(Img):
    def __init__(self, img, date: datetime, imgType: ImgType):
        super().__init__(img, date)
        self.imgType = imgType

class ImgDataset(torch.utils.data.Dataset):
    '''
    Set of all the satellite images, including 
    information on the time they were taken.
    '''
    
    def __init__(self, imgType: ImgType):
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'

        self.imgType = imgType

        if self.imgType==ImgType.DENSE: folder='dense_flow'
        elif self.imgType==ImgType.SPARSE: folder='sparse_flow'
        elif self.imgType==ImgType.SPARSEMASK: folder='sparse_flow_mask'
        else: folder='img'

        self.imgroot = os.path.join(self.root, 'data', folder)
        
        try:
            self.img_paths = []
            for img in torchvision.datasets.ImageFolder(self.imgroot).imgs:
                self.img_paths.append(img[0])
            
            self.timestamps = [
                path2datetime(self.img_paths[i][len(self.imgroot):]) for i in range(len(self.img_paths))
            ]
        except:
            print('ERROR: Cannot call dataset of dense and sparse optical flow images when there are none.')
            exit()

    def __getitem__(self, idx: int):
        img = cv2.imread(self.img_paths[idx],1)
        date = self.timestamps[idx]

        if self.imgType!=ImgType.SAT: return FlowImg(img, date, self.imgType)
        else: return SatImg(img, date)

    def __len__(self):
        return len(self.img_paths)
    
    def getDateIdx(self, date: datetime) -> int:
        return self.timestamps.index(date)
