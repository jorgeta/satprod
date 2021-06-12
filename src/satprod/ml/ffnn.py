import torch
from torch import nn

class FFNN(nn.Module):
    def __init__(self, 
                input_size: int, 
                linear_size: int, 
                linear_input_sequence_length: int,
                output_size: int, 
                sequence_length: int):
        super(FFNN, self).__init__()
        self.name = 'Feed forward neural network'
        
        '''------------------------'''
        '''---------TODO-----------'''
        '''------------------------'''
        
        self.input_size = input_size
        self.linear_size = linear_size
        self.output_size = output_size
        #self.linear_input_sequence_length = linear_input_sequence_length
        self.sequence_length = sequence_length
        
        self.linear1 = nn.Linear(self.input_size, self.linear_size)
        self.linear2 = nn.Linear(self.linear_size*self.sequence_length, self.output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x, y=None, z=None):
        
        x = x[:, -self.sequence_length:, :]
        
        # x shape: (batch_size, sequence_length, n_features)
        x = self.relu(self.linear1(x))
        
        # x shape: (batch_size, sequence_length, linear_size)
        #x = x[:, -self.linear_input_sequence_length:, :]
        # x shape: (batch_size, linear_input_sequence_length, linear_size)
        x = x.view(x.shape[0], -1)
        
        # x shape: (batch_size, sequence_length*linear_size)
        x = self.linear2(x)
        
        # x shape: (batch_size, output_size)
        return x