import numpy as np
import torch
from torch import nn
from satprod.ml.imgnets import LeNet, ResNet, DeepSense, image_feature_extraction

class SimpleResNet(nn.Module):
    
    def __init__(self, 
                output_size: int, 
                num_output_features: int, 
                sequence_length: int = 1,
                resnet_params: dict=None):
        super(SimpleResNet, self).__init__()
        
        self.name = 'SimpleResNet'
        self.output_size = output_size
        self.num_output_features = num_output_features
        self.sequence_length = sequence_length
        
        self.resnet = ResNet(**resnet_params)
        self.num_image_features = resnet_params['output_size']
        
        self.linear = nn.Linear(
            self.num_image_features*self.sequence_length, 
            self.output_size*self.num_output_features
        )
        
        self.img_extraction_method = 'resnet'
    
    def forward(self, data_dict: dict):
        x_img = data_dict['X_img']
        
        assert x_img is not None, 'Feature error: The dataset needs to contain past features for the LSTM.'
        
        self.batch_size = x_img.shape[0]
        x_img = image_feature_extraction(x_img, self)
        
        x = self.linear(x_img)
        
        x = x.view(self.batch_size, self.output_size, self.num_output_features)
        # x.shape: (batch_size, output_size, num_output_features)
        
        return x