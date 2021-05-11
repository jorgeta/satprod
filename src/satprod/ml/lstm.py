import numpy as np
import torch
from torch import nn
from satprod.ml.imgnets import LeNet, ResNet, DeepSense

class LSTM_net(nn.Module):
    
    def __init__(self, 
                input_size: int, 
                hidden_size: int, 
                output_size: int, 
                num_forecast_features: int,
                num_output_features: int, 
                num_layers: int, 
                sequence_len: int,
                channels_1: int,
                channels_2: int,
                kernel_size_conv: [int],
                stride_conv: [int],
                padding_conv: [int],
                kernel_size_pool: [int],
                stride_pool: [int],
                padding_pool: [int],
                height: int = 100, 
                width: int = 100,
                linear_size: int=0,
                dropout: float=0.0,
                initialization: str=None, 
                activation=None, 
                in_channels: int=1, 
                img_extraction_method: str=None):
        super(LSTM_net, self).__init__()
        
        self.name = 'simple_LSTM'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_size = linear_size
        self.output_size = output_size
        self.num_forecast_features = num_forecast_features
        self.num_output_features = num_output_features
        self.num_layers = num_layers
        self.sequence_len = sequence_len
        
        self.dropout = nn.Dropout(dropout)
        
        if self.num_layers > 1:
            self.lstm = nn.LSTM(input_size+channels_2, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            self.lstm = nn.LSTM(input_size+channels_2, hidden_size, num_layers, batch_first=True)
        
        if self.linear_size > 0:
            self.linear1 = nn.Linear(self.hidden_size+self.num_forecast_features, self.linear_size)
            self.linear2 = nn.Linear(self.linear_size, self.output_size*self.num_output_features)
        else:
            self.linear = nn.Linear(self.hidden_size+self.num_forecast_features, self.output_size*self.num_output_features)
        
        if initialization=='xavier':
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
        
        self.activation = activation
        
        self.img_extraction_method = img_extraction_method
        if self.img_extraction_method=='lenet':
            self.lenet = LeNet(
                channels_1,
                channels_2,
                kernel_size_conv,
                stride_conv,
                padding_conv,
                kernel_size_pool,
                stride_pool,
                padding_pool,
                height, 
                width,
                in_channels
            )
        else:
            self.resnet = ResNet(output_size=channels_2)
    
    def forward(self, x, x_forecasts, x_img):
        
        # x shape: (batch_size, sequence_length, num_past_features)
        # x_forecasts shape: (batch_size, num_forecast_features)
        # x_img shape: (batch_size, sequence_length, img_height, img_width)
            # or (batch_size, sequence_length, img_height, img_width, bands)
        batch_size = x.shape[0]
        if x_img is not None:
            if self.img_extraction_method=='lenet':
                x_img_features = []
                for i in range(x_img.shape[1]):
                    img = x_img[:, i, :, :]#.reshape(x_img.shape[0], 1, x_img.shape[2], x_img.shape[3])
                    img = self.lenet(img)
                    x_img_features.append(img)
                
                x_img = torch.stack(x_img_features).view(batch_size, self.sequence_len, -1)
                # x_img shape: (batch_size, sequence_length, channels_2)
                
            else:
                x_img_features = []
                for i in range(x_img.shape[1]):
                    img = x_img[:, i, :, :, :]#.reshape(x_img.shape[0], 1, x_img.shape[2], x_img.shape[3])
                    img = self.resnet(img)
                    x_img_features.append(img)
                
                x_img = torch.stack(x_img_features).view(batch_size, self.sequence_len, -1)
        
            x = torch.cat([x, x_img], dim=2)
        
        x, _ = self.lstm(x)
        # x.shape: (batch_size, seq_len, hidden_size)

        x = x[:, -1, :]
        # x.shape: (batch_size, hidden_size)
        if x_forecasts is not None:
            x = torch.cat([x, x_forecasts], dim=1)
        
        if self.linear_size > 0:
            x = self.activation(self.linear1(x))
            # x.shape: (batch_size, linear_size)
        
            x = self.linear2(self.dropout(x))
            # x.shape: (batch_size, output_size*num_output_features)
        else:
            x = self.linear(x)
            # x.shape: (batch_size, output_size*num_output_features)
            
        x = x.view(batch_size, self.output_size, self.num_output_features)
        # x.shape: (batch_size, output_size, num_output_features)
        
        return x

if __name__ =='__main__':
    lstm = LSTM_net(1,1,1,1,1)
    print(lstm)