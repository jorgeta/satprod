import torch
import torchvision
from datetime import datetime, timedelta
import os
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
import cv2
from utils import datetime2path, path2datetime
#from src.satprod.data_handlers.utils import datetime2path, path2datetime

root = '..'
imgpath = f'{root}/data/img'

class SatImg():
    '''
    One satellite image, as a numpy array.
    The date and time of the image is also included.
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

    def resize(self, scale_percent=5, interp=cv2.INTER_AREA):
        width = int(self.width * scale_percent / 100)
        height = int(self.height * scale_percent / 100)
        dim = (width, height)
        return SatImg(cv2.resize(self.img, dim, interpolation = interp), self.date)


class SatImgDataset(torch.utils.data.Dataset):
    '''
    Set of all the satellite images, including 
    information on the time they were taken.
    '''
    
    def __init__(self, root: str):
        self.imgroot = os.path.join(root, 'data/img')
        
        self.img_paths = []
        for img in torchvision.datasets.ImageFolder(self.imgroot).imgs:
            self.img_paths.append(img[0])
        
        self.timestamps = [path2datetime(self.img_paths[i][len(self.imgroot):]) for i in range(len(self.img_paths))]

    def __getitem__(self, idx: int):
        img = cv2.imread(self.img_paths[idx],1)
        date = self.timestamps[idx]
        return SatImg(img, date)

    def __len__(self):
        return len(self.img_paths)
    
    def getDateIdx(self, date: datetime) -> int:
        return self.timestamps.index(date)