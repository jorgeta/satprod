import numpy as np
import torch
from torch import nn
from torch.nn.functional import pad
from satprod.ml.imgnets import LeNet, ResNet, DeepSense, image_feature_extraction
from collections import OrderedDict
from torch.nn.utils import weight_norm

class TCN(nn.Module):
    
    def __init__(self,
                num_past_features: int,
                num_forecast_features: int, 
                output_size: int,
                pred_sequence_length: int = 5,
                kernel_size: int = 3,
                dilation_base: int = 2,
                channels: [int] = [10, 10, 10],
                resnet_params: dict=None,
                lenet_params: dict=None,
                deepsense_params: dict=None,
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
        self.num_forecast_features = num_forecast_features
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
        elif img_extraction_method=='deepsense':
            raise NotImplementedError
            self.deepsense = DeepSense(**deepsense_params)
            self.num_image_features = 0
        else:
            pass
        
        self.dilation_base = dilation_base
        assert self.kernel_size >= self.dilation_base
        
        self.sequence_length = int(
            1+2*(self.kernel_size-1)*(2**len(self.channels)-1)/(2-1)
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
                dilation=self.dilations[i]
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
        
        x_weather_forecasts = pad(x_weather_forecasts, (0,x_prod.shape[2]))
        
        x = torch.cat([x_weather, x_prod], dim=2)
        x = torch.cat([x, x_weather_forecasts], dim=1)
        x = x[:, self.pred_sequence_length:, :]
        # x shape: (batch_size, sequence_length, n_features) (64, 29, 4)
        
        self.batch_size = x.shape[0]
        if x_img is not None:
            assert x_img_forecasts is not None
            self.batch_size = x_img.shape[0]
            
            x_img = torch.cat([x_img, x_img_forecasts], dim=1)
            x_img = x_img[:, self.pred_sequence_length:, :]
            
            x_img = image_feature_extraction(x_img, self)
            
            x = torch.cat([x, x_img], dim=2) if x is not None else x_img
        
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

if __name__=='__main__':
    n_features = 1
    batch_size = 64
    sequence_length = 15
    output_size = 1
    kernel_size = 5
    num_past_features = 4
    num_forecast_features = 15
    pred_sequence_length = 5
    num_unique_forecast_features = 3
    
    resnet_params = {
        'output_size' : 8
    }
    
    lenet_params = {
        'channels' : [1, 16, 32],
        'kernel_size_conv' : [8,4],
        'stride_conv' : [4,4],
        'padding_conv' : [0,0],
        'kernel_size_pool' : [3,2],
        'stride_pool' : [3,1],
        'padding_pool' : [0,0],
        'height' : 100,
        'width' : 100
    }
    
    tcn_params = {
        'num_past_features': num_past_features,
        'num_forecast_features': num_forecast_features,
        'output_size': output_size, # 1 if one park
        'sequence_length': sequence_length,
        'pred_sequence_length': pred_sequence_length,
        'img_extraction_method': 'lenet',
        'lenet_params': lenet_params
    }
    
    tcn = TCN(**tcn_params)
    
    print(tcn)
    
    x = torch.randn(batch_size, sequence_length, num_past_features)
    x_forecasts = torch.randn(batch_size, pred_sequence_length, num_unique_forecast_features)
    x_img = torch.randn(batch_size, sequence_length, 100, 100)
    
    print(tcn(x, x_forecasts, x_img).shape)
    