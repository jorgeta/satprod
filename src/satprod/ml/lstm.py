import numpy as np
import torch
from torch import nn
from satprod.ml.imgnets import LeNet, ResNet, DeepSense, image_feature_extraction

class LSTM(nn.Module):
    
    def __init__(self, 
                input_size: int, 
                hidden_size: int, 
                output_size: int, 
                num_forecast_features: int, 
                num_output_features: int, 
                num_layers: int, 
                sequence_length: int,
                height: int = 100, 
                width: int = 100,
                linear_size: int=0,
                dropout: float=0.0,
                initialization: str=None, 
                activation=None,
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
        elif img_extraction_method=='deepsense':
            self.deepsense = DeepSense(**deepsense_params)
            self.num_image_features = 0
        else:
            pass
        
        self.dropout = nn.Dropout(dropout)
        
        if self.num_layers > 1:
            self.lstm = nn.LSTM(self.input_size+self.num_image_features, 
                                hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            self.lstm = nn.LSTM(self.input_size+self.num_image_features, 
                                hidden_size, num_layers, batch_first=True)
        
        if self.linear_size > 0:
            self.linear1 = nn.Linear(self.hidden_size+self.num_forecast_features, self.linear_size)
            self.linear2 = nn.Linear(self.linear_size, self.output_size*self.num_output_features)
        else:
            self.linear = nn.Linear(
                self.hidden_size+self.num_forecast_features, 
                self.output_size*self.num_output_features)
        
        if initialization=='xavier':
            self.init_parameters()
        
        self.activation = activation
        
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
            
    '''def init_hidden_state(self, batch_size: int):
        self.hidden = (
            torch.zeros(1, batch_size, self.hidden_size),
            torch.zeros(1, batch_size, self.hidden_size)
        )'''
    
    def forward(self, x, x_forecasts, x_img):
        
        # x shape: (batch_size, sequence_length, num_past_features)
        # x_forecasts shape: (batch_size, num_forecast_features)
        # x_img shape: (batch_size, sequence_length, img_height, img_width)
            # or (batch_size, sequence_length, img_height, img_width, bands)
        
        self.batch_size = x.shape[0]
        #self.init_hidden_state(self.batch_size)
        
        if x_img is not None:
            x_img = image_feature_extraction(x_img, self)
            x = torch.cat([x, x_img], dim=2)
        
        x, _ = self.lstm(x)
        # x.shape: (batch_size, seq_len, hidden_size)

        x = x[:, -1, :]
        # x.shape: (batch_size, hidden_size)
        if x_forecasts is not None:
            x = torch.cat([x, x_forecasts.view(self.batch_size, -1)], dim=1)
        
        if self.linear_size > 0:
            x = self.activation(self.linear1(x))
            # x.shape: (batch_size, linear_size)
        
            x = self.linear2(self.dropout(x))
            # x.shape: (batch_size, output_size*num_output_features)
        else:
            x = self.linear(x)
            # x.shape: (batch_size, output_size*num_output_features)
            
        x = x.view(self.batch_size, self.output_size, self.num_output_features)
        # x.shape: (batch_size, output_size, num_output_features)
        
        return x