import numpy as np
import torch
from torch import nn

class MLR(nn.Module):
    def __init__(self, 
                sequence_length: int,
                num_past_features: int,
                num_forecast_features: int, 
                output_size: int = 1,
                pred_sequence_length: int = 5,
                ):
        super(MLR, self).__init__()
        self.name = 'MLR'
        self.sequence_length = sequence_length
        self.num_past_features = num_past_features
        self.num_forecast_features = num_forecast_features
        self.input_size = self.sequence_length*self.num_past_features + self.num_forecast_features
        self.output_size = output_size
        self.pred_sequence_length = pred_sequence_length
        
        self.linear = nn.Linear(
            self.input_size, 
            self.output_size*self.pred_sequence_length)

    def forward(self, data_dict):
        
        x_prod = data_dict['X_prod']
        x_weather = data_dict['X_weather']
        x_weather_forecasts = data_dict['X_weather_forecasts']
        if x_prod is not None:
            self.batch_size = x_prod.shape[0]
            if x_weather is not None:
                if x_weather_forecasts is not None:
                    x = torch.cat([
                        x_prod.view(self.batch_size, -1), 
                        x_weather.view(self.batch_size, -1), 
                        x_weather_forecasts.view(self.batch_size, -1)
                        ], dim=1)
                else:
                    x = torch.cat([
                        x_prod.view(self.batch_size, -1), 
                        x_weather.view(self.batch_size, -1)
                    ], dim=1)
            else:
                x = x_prod.view(self.batch_size, -1)
        else:
            if x_weather is not None:
                self.batch_size = x_weather.shape[0]
                if x_weather_forecasts is not None:
                    x = torch.cat([
                        x_weather.view(self.batch_size, -1), 
                        x_weather_forecasts.view(self.batch_size, -1)
                    ], dim=1)
                else:
                    x = x_weather.view(self.batch_size, -1)
            else:
                raise Exception('MLR needs either production, wind speed or wind direction to function.')
        return self.linear(x).view(self.batch_size, self.pred_sequence_length, self.output_size)