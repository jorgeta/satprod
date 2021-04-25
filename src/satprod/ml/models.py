import numpy as np
import torch
from torch import nn
import os
import pandas as pd
from satprod.pipelines.dataset import WindDataset

class LSTM_net(nn.Module):
    
    def __init__(self, 
                input_size: int, 
                hidden_size: int, 
                output_size: int, 
                num_forecast_features: int,
                num_output_features: int, 
                num_layers: int, 
                sequence_len: int,
                linear_size: int=0,
                dropout: float=0.0,
                initialization: str=None, 
                activation=None):
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
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
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
        
    def forward(self, x, x_forecasts):
        
        # x shape: (batch_size, sequence_length, num_past_features)
        # x_forecasts shape: (batch_size, num_forecast_features)
        batch_size = x.shape[0]
        
        x, _ = self.lstm(x)
        # x.shape: (batch_size, seq_len, hidden_size)

        x = x[:, -1, :]
        # x.shape: (batch_size, hidden_size)
        
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

class Persistence():

    def __init__(self, pred_sequence_length: int):
        self.name = 'persistence'
        self.sequence_length = 1
        self.pred_sequence_length = pred_sequence_length
    
    def predict(self, observation: pd.DataFrame):
        current_production = get_columns(observation, 'production').values
        prediction = np.zeros((self.pred_sequence_length, len(current_production)))
        for i in range(self.pred_sequence_length):
            prediction[i] = current_production
        return prediction

class WindSpeedCubed():
    
    def __init__(self, pred_sequence_length: int, dataset: WindDataset):
        self.name = 'wind_speed_cubed'
        self.sequence_length = 1
        self.pred_sequence_length = pred_sequence_length
        self.dataset = dataset
    
    def predict(self, observation: pd.DataFrame):
        if 'velocity_cubed' in observation.columns:
            wind_speed_forecasts_cubed = get_columns(get_columns(observation,'velocity_cubed'), '+').values
        else:
            wind_speed_forecasts_cubed = get_columns(get_columns(observation,'speed'), '+').pow(3).values
        print(wind_speed_forecasts_cubed)

if __name__ =='__main__':
    lstm = LSTM_net(1,1,1,1,1)
    print(lstm)