import numpy as np
import torch
from torch import nn
from satprod.ml.imgnets import LeNet, ResNet, image_feature_extraction

class LSTM(nn.Module):
    
    def __init__(self, 
                input_size: int, 
                hidden_size: int, 
                output_size: int, 
                num_forecast_features: int, 
                num_img_forecast_features: int,
                num_output_features: int, 
                num_layers: int, 
                sequence_length: int,
                linear_size: int=0,
                dropout: float=0.0,
                img_extraction_method: str=None,
                lenet_params: dict=None, 
                resnet_params: dict=None,
                deepsense_params: dict=None):
        super(LSTM, self).__init__()
        
        self.name = 'LSTM'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_size = linear_size
        self.output_size = output_size
        self.num_forecast_features = num_forecast_features
        self.num_img_forecast_features = num_img_forecast_features
        self.num_output_features = num_output_features
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.num_image_features = 0
        
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
        
        self.dropout = nn.Dropout(dropout)
        
        if self.num_layers > 1:
            self.lstm = nn.LSTM(self.input_size+self.num_image_features, 
                                hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            self.lstm = nn.LSTM(self.input_size+self.num_image_features, 
                                hidden_size, num_layers, batch_first=True)
        
        if self.linear_size > 0:
            self.linear1 = nn.Linear(
                self.hidden_size+self.num_forecast_features+self.num_img_forecast_features*self.num_image_features, 
                self.linear_size)
            self.linear2 = nn.Linear(
                self.linear_size, 
                self.output_size*self.num_output_features)
        else:
            self.linear = nn.Linear(
                self.hidden_size+self.num_forecast_features+self.num_img_forecast_features*self.num_image_features, 
                self.output_size*self.num_output_features)
        
        self.init_parameters()
        
        self.tanh = nn.Tanh()
        
    def init_parameters(self):
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_normal_(param)
            else:
                nn.init.normal_(param)
        
        if self.linear_size > 0:
            nn.init.xavier_normal_(self.linear1.weight)
            nn.init.zeros_(self.linear1.bias)
            
            nn.init.xavier_normal_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)
        else:
            nn.init.xavier_normal_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, data_dict: dict):
        
        x_prod = data_dict['X_prod']
        # x shape: (batch_size, sequence_length, num_output_features)
        
        x_weather = data_dict['X_weather']
        # x_weather shape: (batch_size, sequence_length, num_forecast_features)
        
        x_weather_forecasts = data_dict['X_weather_forecasts']
        # x_weather_forecasts shape: (batch_size, pred_sequence_length, num_forecast_features)
        
        x_img = data_dict['X_img']
        # x_img shape: (batch_size, sequence_length, img_height, img_width)
            # or (batch_size, sequence_length, img_height, img_width, bands)
        
        x_img_forecasts = data_dict['X_img_forecasts']
        
        x = None
        if x_prod is not None:
            if x_weather is not None:
                x = torch.cat([x_weather, x_prod], dim=2)
            else:
                x = x_prod
        else:
            if x_weather is not None:
                x = x_weather
        if x is not None:
            self.batch_size = x.shape[0]
        else:
            assert x_img is not None, 'Feature error: The dataset needs to contain past features for the LSTM.'
        
        if x_img is not None:
            self.batch_size = x_img.shape[0]
            x_img = image_feature_extraction(x_img, self)
            x = torch.cat([x, x_img], dim=2) if x is not None else x_img
        
        x, _ = self.lstm(x)
        # x.shape: (batch_size, seq_len, hidden_size)

        x = x[:, -1, :]
        # x.shape: (batch_size, hidden_size)
        if x_weather_forecasts is not None:
            x = torch.cat([x, x_weather_forecasts.view(self.batch_size, -1)], dim=1)
        if x_img_forecasts is not None:
            x_img_forecasts = image_feature_extraction(x_img_forecasts, self)
            x = torch.cat([x, x_img_forecasts.view(self.batch_size, -1)], dim=1)
        
        if self.linear_size > 0:
            x = self.tanh(self.linear1(x))
            # x.shape: (batch_size, linear_size)
        
            x = self.linear2(self.dropout(x))
            # x.shape: (batch_size, output_size*num_output_features)
        else:
            x = self.linear(x)
            # x.shape: (batch_size, output_size*num_output_features)
        
        x = x.view(self.batch_size, self.output_size, self.num_output_features)
        # x.shape: (batch_size, output_size, num_output_features)
        
        return x