import numpy as np
import torch
from torch import nn
from satprod.ml.imgnets import LeNet, ResNet, DeepSense

class TCN_Encoder(nn.Module):
    
    def __init__(self, output_size):
        super(TCN_Encoder, self).__init__()
        
        self.resnet = ResNet(output_size=output_size)
        
        self.convA1 = nn.Conv1d(
            in_channels=self.num_filters_conv[0],
            out_channels=self.num_filters_conv[1],
            kernel_size=kernel_size_conv[0],
            stride=stride_conv[0],
            padding=padding_conv[0]
        )
        
        self.convA2 = nn.Conv1d(
            in_channels=self.num_filters_conv[0],
            out_channels=self.num_filters_conv[1],
            kernel_size=kernel_size_conv[0],
            stride=stride_conv[0],
            padding=padding_conv[0]
        )
        
    def forward(self, x, x_img):
        
        # x shape: (batch_size, sequence_length, num_past_features)
        # x_img shape: (batch_size, sequence_length, img_height, img_width)
            # or (batch_size, sequence_length, img_height, img_width, num_bands)
        
        # concatenate image features and numerical features
        batch_size = x.shape[0]
        if x_img is not None:
            x_img_features = []
            for i in range(x_img.shape[1]):
                img = x_img[:, i, :, :, :]#.reshape(x_img.shape[0], 1, x_img.shape[2], x_img.shape[3])
                img = self.resnet(img)
                x_img_features.append(img)
            
            x_img = torch.stack(x_img_features).view(batch_size, self.sequence_len, -1)
            # x_img shape: (batch_size, sequence_length, channels_2)
            
            x = torch.cat([x, x_img], dim=2)
            # x shape: (batch_size, sequence_length, n_features)
        
        #convolve over the different timesteps in X
        convA1_outputs = []
        convA2_outputs = []
        
        for i in range(1, x.shape[1]):
            convA1_outputs.append(self.convA1(x[:, (i-1):i+1, :]))
        
        for i in range(1, len(convA1_outputs)):
            convA2_outputs.append(self.convA2(convA1_outputs[i-1:i+1]))
            
        return x[:, -1, :], convA1_outputs[-1], convA2_outputs[-1]
        
        

class TCN_Decoder(nn.Module):
    
    def __init__(self):
        super(TCN_Decoder, self).__init__()
        
        self.convB1_first = nn.Conv1d(
            in_channels=self.num_filters_conv[0],
            out_channels=self.num_filters_conv[1],
            kernel_size=kernel_size_conv[0],
            stride=stride_conv[0],
            padding=padding_conv[0]
        )
        
        self.convB2_first = nn.Conv1d(
            in_channels=self.num_filters_conv[0],
            out_channels=self.num_filters_conv[1],
            kernel_size=kernel_size_conv[0],
            stride=stride_conv[0],
            padding=padding_conv[0]
        )
        
        self.convB1 = nn.Conv1d(
            in_channels=self.num_filters_conv[0],
            out_channels=self.num_filters_conv[1],
            kernel_size=kernel_size_conv[0],
            stride=stride_conv[0],
            padding=padding_conv[0]
        )
        
        self.convB2 = nn.Conv1d(
            in_channels=self.num_filters_conv[0],
            out_channels=self.num_filters_conv[1],
            kernel_size=kernel_size_conv[0],
            stride=stride_conv[0],
            padding=padding_conv[0]
        )
        
        self.linear = nn.Linear(
            in_features=100,
            out_features=2
        )
        
    def forward(self, x_forecasts):
        pass

class TCN(nn.Module):
    
    def __init__(self):
        super(TCN, self).__init__()

    def forward(self):
        pass