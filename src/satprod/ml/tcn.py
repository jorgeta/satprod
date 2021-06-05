import numpy as np
import torch
from torch import nn
from satprod.ml.imgnets import LeNet, ResNet, DeepSense, image_feature_extraction
from collections import OrderedDict

class TCN(nn.Module):
    
    def __init__(self, 
                channels: [int],
                num_past_features: int,
                num_unique_forecast_features: int,
                output_size: int,
                pred_sequence_length: int,
                kernel_size: int = 3,
                dilations: [int]=[1, 2, 4],
                resnet_params: dict=None,
                lenet_params: dict=None,
                deepsense_params: dict=None,
                img_extraction_method: str=None):
        super(TCN, self).__init__()
        self.name = 'TCN'
        
        self.channels = channels
        self.output_size = output_size
        self.pred_sequence_length = pred_sequence_length
        self.kernel_size = kernel_size
        self.img_extraction_method = img_extraction_method
        self.num_layers = len(dilations)
        self.dilations = dilations
        self.sequence_length = int((self.kernel_size-1)*np.sum(self.dilations)+1)
        self.num_past_features = num_past_features
        self.num_unique_forecast_features = num_unique_forecast_features
        self.num_image_features = 0
        
        assert len(self.channels)==self.num_layers+1
        assert self.channels[0]==1
        assert self.channels[-1]==1
        
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
        
        cnn_layer_stacks = []
        for j in range(self.num_past_features+self.num_image_features):
            cnn_layer_stack = OrderedDict()
            for i in range(self.num_layers):
                cnn_layer_stack[f'conv{i+1}'] = nn.Conv1d(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i+1],
                    kernel_size=self.kernel_size,
                    dilation=self.dilations[i]
                )
                if i < self.num_layers-1:
                    cnn_layer_stack[f'relu{i+1}'] = nn.ReLU()
            cnn_layer_stacks.append(cnn_layer_stack)
        
        cnn_sequences = []
        for j in range(self.num_past_features+self.num_image_features):
            cnn_sequences.append(nn.Sequential(cnn_layer_stacks[j]))
        
        self.dilated_tcn = nn.ModuleList(cnn_sequences)
        #self.identity = nn.Identity()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(
            in_features = self.num_past_features+self.num_image_features+self.num_unique_forecast_features, 
            out_features = self.output_size
        )
        #print(self.dilated_tcn)
        #print(self.dilated_tcn[0])
        
    def forward(self, x, x_forecasts, x_img, x_last_production, x_img_forecasts, y=None):
        # x shape: (batch_size, sequence_length, num_past_features)
        # x_img shape: (batch_size, sequence_length, img_height, img_width)
            # or (batch_size, sequence_length, img_height, img_width, num_bands)
        
        # concatenate image features and numerical features
        
        self.batch_size = x.shape[0]
        if x_img is not None:
            x_img = image_feature_extraction(x_img, self)
            x = torch.cat([x, x_img], dim=2)
        if x_img_forecasts is not None:
            x_img_forecasts = image_feature_extraction(x_img_forecasts, self)
        
        outputs = []
        outputs.append(self.forward_step(x, x_forecasts[:, 0, :]))
        # output shape: (64, 1)
        
        for i in range(self.pred_sequence_length-1):
            if x_img_forecasts is not None:
                addition = torch.cat([
                    x_forecasts[:, i, :], 
                    outputs[i] if y is None else y[:, i], 
                    x_img_forecasts[:, i, :]
                    ], dim=1).unsqueeze(1)
            else:
                addition = torch.cat([
                    x_forecasts[:, i, :], 
                    outputs[i] if y is None else y[:, i]
                    ], dim=1).unsqueeze(1)

            x = torch.cat([x[:, -self.sequence_length+1:, :], addition], dim=1)
            
            outputs.append(self.forward_step(x, x_forecasts[:, i+1, :]))
        
        output = torch.stack(outputs, dim=1)
        return output
        #convolve over the different timesteps in X
    
    def forward_step(self, x, x_forecast):
        tcn_outputs = []
        for i in range(self.num_past_features+self.num_image_features):
            tcn_outputs.append(self.dilated_tcn[i](x[:, :, i].unsqueeze(1)).squeeze(2))
        
        tcn_output = torch.cat(tcn_outputs, dim=1)
        output_with_forecasts = torch.cat([tcn_output, x_forecast], dim=1)
        
        #identity_output = self.identity(x[:,-1,:])
        #print(tcn_output.shape)
        
        #print(identity_output.shape)
        
        activation_output = tcn_output#+identity_output)
        #print(activation_output.shape)
        return self.linear(output_with_forecasts)
    
