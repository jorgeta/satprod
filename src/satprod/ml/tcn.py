import numpy as np
import torch
from torch import nn
from satprod.ml.imgnets import LeNet, ResNet, DeepSense

class TCN_Encoder(nn.Module):
    
    def __init__(self, 
                kernel_size: [int],
                stride: [int],
                padding: [int],
                channels: [int],
                sequence_length: int,
                num_past_features: int,
                resnet_params: dict=None,
                lenet_params: dict=None,
                deepsense_params: dict=None,
                img_extraction_method: str=None):
        super(TCN_Encoder, self).__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.img_extraction_method = img_extraction_method
        self.sequence_length = sequence_length
        self.num_past_features = num_past_features
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
        
        assert len(self.channels)==3
        assert len(self.kernel_size)==2
        assert len(self.padding)==2
        assert len(self.stride)==2
        
        self.convA1 = nn.Conv1d(
            in_channels=self.channels[0],
            out_channels=self.channels[1],
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
            padding=self.padding[0]
        )
        print(self.num_past_features+self.num_image_features)
        self.conv_out_A1= self.compute_conv_dim(
                2*(self.num_past_features+self.num_image_features), self.kernel_size[0], self.stride[0], self.padding[0])
        
        print(f'Conv out A1: {self.conv_out_A1}')
        
        self.convA2 = nn.Conv1d(
            in_channels=self.channels[1],
            out_channels=self.channels[2],
            kernel_size=self.kernel_size[1],
            stride=self.stride[1],
            padding=self.padding[1]
        )
        
        self.conv_out_A2 = self.channels[-1]*self.compute_conv_dim(
                2*self.conv_out_A1, self.kernel_size[1], self.stride[1], self.padding[1])
        
        print(f'Conv out A2: {self.conv_out_A2}')
    
    def image_feature_extraction(self, x, x_img):
        if self.img_extraction_method=='lenet':
            x_img_features = []
            for i in range(x_img.shape[1]):
                img = x_img[:, i, :, :]
                img = self.lenet(img)
                x_img_features.append(img)
            
            x_img = torch.stack(x_img_features).view(self.batch_size, self.sequence_length, -1)
            # x_img shape: (batch_size, sequence_length, channels_2)
            
        else:
            x_img_features = []
            for i in range(x_img.shape[1]):
                img = x_img[:, i, :, :, :]
                img = self.resnet(img)
                x_img_features.append(img)
            
            x_img = torch.stack(x_img_features).view(self.batch_size, self.sequence_length, -1)

        return torch.cat([x, x_img], dim=2)
    
    def compute_conv_dim(self, dim_size, kernel_size, stride, padding):
        return int(np.floor((dim_size - kernel_size + 2 * padding) / stride + 1))
        
    def forward(self, x, x_img):
        # x shape: (batch_size, sequence_length, num_past_features)
        # x_img shape: (batch_size, sequence_length, img_height, img_width)
            # or (batch_size, sequence_length, img_height, img_width, num_bands)
        
        # concatenate image features and numerical features
        self.batch_size = x.shape[0]
        if x_img is not None:
            x = self.image_feature_extraction(x, x_img)
        
        #convolve over the different timesteps in X
        convA1_outputs = []
        convA2_outputs = []
        
        for i in range(1, x.shape[1]):
            convA1_input = torch.cat([x[:, (i-1):i, :], x[:, i:i+1, :]], dim=2)
            print('HER')
            print(convA1_input.shape)
            convA1_output = self.convA1(convA1_input)
            
            print(convA1_output.shape)
            convA1_outputs.append(convA1_output)
        
        for i in range(1, len(convA1_outputs)):
            convA2_input = torch.cat(convA1_outputs[i-1:i+1], dim=2)
            print(convA2_input.shape)
            convA2_output = self.convA2(convA2_input)
            convA2_outputs.append(convA2_output)
        
        print(convA2_outputs[-1].shape)
        return convA2_outputs[-1].view(self.batch_size, -1)

class TCN_Decoder(nn.Module):
    
    def __init__(self,
                channels: [int],
                kernel_size: [int],
                stride: [int],
                padding: [int],
                num_forecast_features: int,
                num_output_features: int,
                num_past_features: int,
                conv_out_A2: [int],
                num_image_features: int,
                pred_sequence_length: int
                ):
        super(TCN_Decoder, self).__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_forecast_features = num_forecast_features
        self.num_output_features = num_output_features
        self.conv_out_A2 = conv_out_A2
        self.num_past_features = num_past_features
        self.num_image_features = num_image_features
        self.pred_sequence_length = pred_sequence_length
        
        assert len(channels)==3
        assert len(kernel_size)==2
        assert len(stride)==2
        assert len(padding)==2
        
        self.convB1 = nn.Conv1d(
            in_channels=self.channels[0],
            out_channels=self.channels[1],
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
            padding=self.padding[0]
        )
        
        self.convB2 = nn.Conv1d(
            in_channels=self.channels[1],
            out_channels=self.channels[2],
            kernel_size=self.kernel_size[1],
            stride=self.stride[1],
            padding=self.padding[1]
        )
        
        self.conv_out_B1 = self.compute_conv_dim(
            self.conv_out_A2*self.channels[-1]+self.num_forecast_features+self.num_output_features, 
            self.kernel_size[0], self.stride[0], self.padding[0]
        )
        
        self.conv_out_B2 = self.compute_conv_dim(
            self.conv_out_B1, self.kernel_size[1], self.stride[1], self.padding[1]
        )
        
        self.linear = nn.Linear(
            in_features=self.conv_out_B2,
            out_features=self.num_output_features
        )
    
    def compute_conv_dim(self, dim_size, kernel_size, stride, padding):
        return int(np.floor((dim_size - kernel_size + 2 * padding) / stride + 1))
        
    def forward(self, encoder_output, x_last_production, x_forecasts):
        self.batch_size = encoder_output.shape[0]
        '''
        torch.Size([64, 8, 14]) a2
        torch.Size([64, 1]) last prod
        torch.Size([64, 5]) forecast
        '''
        print(self.conv_out_A2)
        print(self.conv_out_A2+self.num_forecast_features+self.num_output_features)
        print(self.conv_out_B1)
        print(self.conv_out_B2)
        
        previous_production = [x_last_production]
        for i in range(self.pred_sequence_length):
            convB1_input = torch.cat(
                [
                    x_forecasts[:, i].unsqueeze(1), 
                    previous_production[-1], 
                    encoder_output.view(self.batch_size, -1)
                ], dim=1).unsqueeze(1)
            print(convB1_input.shape)
            convB1_output = self.convB1(convB1_input)
            print(convB1_output.shape)
            convB2_output = self.convB2(convB1_output)
            print(convB2_output.shape)
            exit()
            previous_production.append(self.linear(convB2_output))
        
        # convb1 input: forecast(t+1), last prod, x_t
        print(previous_production[1:])
        exit()
        return previous_production

class TCN(nn.Module):
    
    def __init__(self, encoder_params: dict, decoder_params: dict):
        super(TCN, self).__init__()
        
        self.encoder = TCN_Encoder(**encoder_params)
        self.decoder = TCN_Decoder(
            conv_out_A2=self.encoder.conv_out_A2, 
            num_past_features=self.encoder.num_past_features,
            num_image_features=self.encoder.num_image_features,
            **decoder_params
        )

    def forward(self, x, x_forecasts, x_img, x_last_production):
        encoder_output = self.encoder(x, x_img)
        #print(encoder_output)
        print('Encoder output shape')
        print(encoder_output.shape)
        x = self.decoder(encoder_output, x_last_production, x_forecasts)
        return x
    
if __name__=='__main__':
    parks = ['skom']#,'vals','skom','yvik']
    num_feature_types = ['production', 'speed', 'direction', 'forecast'] #['forecast', 'direction', 'speed', 'production']
    img_features = []#['grid']
    img_extraction_method = 'lenet' # lenet, deepsense, resnet
    
    if len(img_features)==0:
        img_extraction_method = None
    if img_extraction_method is None:
        img_features = []
    
    from satprod.configs.job_configs import TrainConfig
    train_config = TrainConfig(
        batch_size = 64,
        num_epochs = 30,
        learning_rate = 4e-3,
        scheduler_step_size = 5,
        scheduler_gamma = 0.8,
        train_valid_splits = 1,
        pred_sequence_length = 5,
        random_seed = 0,
        parks = parks,
        num_feature_types = num_feature_types,
        img_features = img_features,
        img_extraction_method = img_extraction_method
    )
    
    # get data
    from satprod.pipelines.dataset import WindDataset
    wind_dataset = WindDataset(train_config)
    
    # define image extraction parameters
    if img_extraction_method=='lenet':
        lenet_params = {
            'channels' : [1, 16, 32],
            'kernel_size_conv' : [8,4],
            'stride_conv' : [4,4],
            'padding_conv' : [0,0],
            'kernel_size_pool' : [3,2],
            'stride_pool' : [3,1],
            'padding_pool' : [0,0],
            'height' : wind_dataset.img_height,
            'width' : wind_dataset.img_width
        }
    else:
        lenet_params = None
    
    x = torch.rand((64, 3, 4))
    x_forecasts = torch.rand((64, 5))
    x_img = None #torch.rand((64, 3, 100, 100))
    x_last_production = torch.rand((64, 1))
    
    encoder_params = {
        'kernel_size' : [2,2],
        'stride' : [2,2],
        'padding' : [1,1],
        'channels' : [1,16,32],
        'sequence_length' : 3,
        'num_past_features' : wind_dataset.n_past_features,
        'img_extraction_method' : img_extraction_method,
        'lenet_params' : lenet_params
    }
    
    decoder_params = {
        'channels' : encoder_params['channels'],
        'kernel_size' : [2,2],
        'stride' : [2,2],
        'padding' : [1,1],
        'num_forecast_features' : wind_dataset.n_forecast_features,
        'num_output_features' : wind_dataset.n_output_features,
        'pred_sequence_length' : train_config.pred_sequence_length
    }
    
    net = TCN(encoder_params, decoder_params)
    
    #print(net)
    
    params_in_network = 0
    trainable_params_in_network = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            trainable_params_in_network += len(np.ravel(param.data.numpy()))
            #print(name, param.data.numpy().shape)
        params_in_network += len(np.ravel(param.data.numpy()))
        
    #print(f'Trainable parameters in network: {trainable_params_in_network}.')
    #print(f'Parameters in network: {params_in_network}.')
    
    net(x, x_forecasts, x_img, x_last_production)
    