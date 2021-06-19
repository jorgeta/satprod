import numpy as np
import torch
from torch import nn
from satprod.ml.imgnets import ResNet, VGG, image_feature_extraction

class SIN(nn.Module):
    
    def __init__(self, 
                output_size: int, 
                num_output_features: int, 
                dropout: float = 0.0,
                sequence_length: int = 1,
                resnet_params: dict=None,
                lenet_params: dict=None,
                vgg_params: dict=None,
                img_extraction_method: str=None):
        super(SIN, self).__init__()
        
        self.name = 'SIN' # Simple Image Net
        self.output_size = output_size
        self.num_output_features = num_output_features
        self.sequence_length = sequence_length
        
        self.img_extraction_method = img_extraction_method
        if self.img_extraction_method=='lenet':
            self.lenet = LeNet(**lenet_params)
            self.num_image_features = lenet_params['channels'][-1]
        elif self.img_extraction_method=='resnet':
            self.resnet = ResNet(**resnet_params)
            self.num_image_features = resnet_params['output_size']
        elif img_extraction_method=='vgg':
            self.vgg = VGG(**vgg_params)
            self.num_image_features = vgg_params['output_size']
        else:
            raise NotImplementedError
        
        self.linear = nn.Linear(
            self.num_image_features*self.sequence_length, 
            self.output_size*self.num_output_features
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data_dict: dict):
        x_img = data_dict['X_img']
        
        assert x_img is not None, 'Feature error: The dataset needs to contain past features for the LSTM.'
        
        self.batch_size = x_img.shape[0]
        x = image_feature_extraction(x_img, self)
        
        x = x.view(self.batch_size, self.num_image_features*self.sequence_length)
        
        x = self.linear(self.dropout(x))
        
        x = x.view(self.batch_size, self.output_size, self.num_output_features)
        # x.shape: (batch_size, output_size, num_output_features)
        
        return x