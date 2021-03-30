import numpy as np
import torch
from torch import nn
import os
import pandas as pd

class LSTM_net(nn.Module):
    
    def __init__(self, input_size, hidden_size, linear_size, output_size, num_output_features, num_layers, sequence_len):
        super(LSTM_net, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_size = linear_size
        self.output_size = output_size
        self.num_output_features = num_output_features
        self.num_layers = num_layers
        self.sequence_len = sequence_len
        
        self.dropout = nn.Dropout(0.2)
        if self.num_layers > 1:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            
        for param in self.lstm.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_normal_(param)
            else:
                nn.init.normal_(param)
        
        self.linear1 = nn.Linear(self.hidden_size, self.linear_size)
        self.linear2 = nn.Linear(self.linear_size, self.output_size*self.num_output_features)
        
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        
        self.tanh = nn.Tanh()
        #self.relu = nn.ReLU()
    
    '''def reset_hidden_state(self, batch_size):
        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size)
        )'''
        
    def forward(self, x):
        
        # x shape: (batch_size, sequence_length, num_features)
        batch_size = x.shape[0]
        
        #x, _ = self.lstm(x.float(), self.hidden)
        #x, _ = self.lstm(x.float(), (self.hidden[0][:,:batch_size,:], self.hidden[1][:,:batch_size,:]))
        x, _ = self.lstm(x)
        
        # x.shape: (batch_size, seq_len, hidden_size)

        x = x[:, -1, :]
        
        # x.shape: (batch_size, hidden_size)
        
        x = self.tanh(self.linear1(x))
        
        # x.shape: (batch_size, linear_size)
        
        x = self.linear2(self.dropout(x))
        
        x = x.view(batch_size, self.output_size, self.num_output_features)
        
        # x.shape: (batch_size, output_size, num_output_features)
        
        return x
    
if __name__ =='__main__':
    lstm = LSTM_net(1,1,1,1,1)
    print(lstm)