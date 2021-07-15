import numpy as np
import torch
from torch import nn
from torch.nn.functional import pad
from satprod.ml.imgnets import LeNet, ResNet, VGG, image_feature_extraction
from collections import OrderedDict
from torch.nn.utils import weight_norm

class TCN(nn.Module):
    
    def __init__(self,
                num_past_features: int,
                output_size: int,
                pred_sequence_length: int = 5,
                kernel_size: int = 3,
                dilation_base: int = 2,
                channels: [int] = [10, 10, 10],
                resnet_params: dict=None,
                lenet_params: dict=None,
                vgg_params: dict=None,
                img_extraction_method: str=None,
                dropout: float=0.0,
                only_predict_future_values: bool=False
                ):
        super(TCN, self).__init__()
        self.name = 'TCN'
        
        self.output_size = output_size
        self.pred_sequence_length = pred_sequence_length
        self.kernel_size = kernel_size
        self.img_extraction_method = img_extraction_method
        self.channels = channels
        self.num_past_features = num_past_features
        self.num_image_features = 0
        self.dropout = dropout
        self.only_predict_future_values = only_predict_future_values
        
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
        
        self.dilation_base = dilation_base
        assert self.kernel_size >= self.dilation_base
        
        self.sequence_length = int(
            1+2*(self.kernel_size-1)*(2**len(self.channels)-1)/(self.dilation_base-1)
        )
        
        self.num_layers = len(self.channels)
        self.input_size = self.num_past_features+self.num_image_features
        self.dilations = [self.dilation_base**i for i in range(self.num_layers)]
        
        residual_block_stack = OrderedDict()
        for i in range(self.num_layers):
            residual_block_stack[f'block{i+1}'] = ResidualBlock(
                in_channels=self.channels[i-1] if i>0 else self.input_size,
                out_channels=self.channels[i],
                kernel_size=self.kernel_size,
                dilation=self.dilations[i],
                dropout=self.dropout
            )
        
        self.tcn_stack = nn.Sequential(residual_block_stack)
        
        self.linear = nn.Linear(
            in_features=self.channels[-1],
            out_features=self.output_size
        )
        
    def forward(self, data_dict: dict):
        
        x_prod = data_dict['X_prod']
        x_weather = data_dict['X_weather']
        x_weather_forecasts = data_dict['X_weather_forecasts']
        x_img = data_dict['X_img']
        x_img_forecasts = data_dict['X_img_forecasts']
        
        x = None
        if x_prod is not None:
            if x_weather is not None:
                x = torch.cat([x_weather, x_prod], dim=2)
                if x_weather_forecasts is not None:
                    x_weather_forecasts = pad(x_weather_forecasts, (0, self.output_size))
                    x = torch.cat([x, x_weather_forecasts], dim=1)
            else:
                self.batch_size = x_prod.shape[0]
                x = pad(x_prod, (0, 0, self.pred_sequence_length, 0))
        else:
            if x_weather is not None:
                x = x_weather
                if x_weather_forecasts is not None:
                    x = torch.cat([x, x_weather_forecasts], dim=1)
        if x is not None:
            x = x[:, self.pred_sequence_length:, :]
            self.batch_size = x.shape[0]
        else:
            assert x_img is not None, 'Feature error: The dataset is empty.'
        # x shape: (batch_size, sequence_length, n_features) (64, 29, 4)
        
        if x_img is not None:
            assert x_img_forecasts is not None, 'The model requires image forecasts when images are used.'
            self.batch_size = x_img.shape[0]
            
            x_img = torch.cat([x_img, x_img_forecasts], dim=1)
            x_img = x_img[:, self.pred_sequence_length:, :]
            x_img = image_feature_extraction(x_img, self)
            if x is not None:
                x = torch.cat([x, x_img], dim=2)
            else: 
                x = x_img
        
        x = self.tcn_stack(x.transpose(1,2)).transpose(1, 2)
        
        if self.only_predict_future_values:
            return self.linear(x)[:, -self.pred_sequence_length:, :]
        else:
            return self.linear(x)

class ResidualBlock(nn.Module):
    
    def __init__(self, 
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                dilation: int,
                dropout: float = 0.0
                ):
        super(ResidualBlock, self).__init__()
        
        self.causalconvs = nn.Sequential(
            weight_norm(CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation
            )),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            weight_norm(CausalConv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation
            )),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        self.residual = nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.causalconvs(x) + self.residual(x))
        

class CausalConv1d(nn.Conv1d):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                stride: int=1,
                dilation: int=1,
                groups: int=1,
                bias: bool=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input):
        x = pad(input.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super(CausalConv1d, self).forward(x)
    