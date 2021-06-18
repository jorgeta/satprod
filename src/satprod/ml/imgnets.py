import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.transforms import Normalize

class LeNet(nn.Module):
    
    def __init__(self, 
                channels: [int],
                kernel_size_conv: [int],
                stride_conv: [int],
                padding_conv: [int],
                kernel_size_pool: [int],
                stride_pool: [int],
                padding_pool: [int],
                height: int,
                width: int,
                ):
        super(LeNet, self).__init__()
        
        assert height>0
        assert width>0
        assert len(channels)==3
        
        self.name = 'LeNet'
        self.height = height
        self.width = width
        
        self.channels = channels
        
        self.kernel_size_conv = kernel_size_conv #[5, 5]
        self.stride_conv = stride_conv #[1, 1]
        self.padding_conv = padding_conv #[2, 2]

        self.kernel_size_pool = kernel_size_pool#[2, 2]
        self.stride_pool = stride_pool#[2, 2]
        self.padding_pool = padding_pool#[0, 0]
        
        self.conv1 = nn.Conv2d(
            in_channels=self.channels[0],
            out_channels=self.channels[1],
            kernel_size=self.kernel_size_conv[0],
            stride=self.stride_conv[0],
            padding=self.padding_conv[0]
        )
        
        self.pool1 = nn.MaxPool2d(
            kernel_size = kernel_size_pool[0], 
            stride = stride_pool[0], 
            padding = padding_pool[0]
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=self.channels[1],
            out_channels=self.channels[2],
            kernel_size=kernel_size_conv[1],
            stride=stride_conv[1],
            padding=padding_conv[1]
        )
        
        self.pool2 = nn.MaxPool2d(
            kernel_size = kernel_size_pool[1], 
            stride = stride_pool[1], 
            padding = padding_pool[1]
        )
        
        self.calculate_layer_sizes()
    
    def compute_conv_dim(self, dim_size, kernel_size, stride, padding):
        return int(np.floor((dim_size - kernel_size + 2 * padding) / stride + 1))
    
    def calculate_layer_sizes(self):
        
        self.conv_out_height = []
        self.conv_out_width = []
        self.pool_out_height = []
        self.pool_out_width = []
        
        self.conv_out_height.append(
            self.compute_conv_dim(
                self.height, self.kernel_size_conv[0], self.stride_conv[0], self.padding_conv[0]))
        self.conv_out_width.append(
            self.compute_conv_dim(
                self.width, self.kernel_size_conv[0], self.stride_conv[0], self.padding_conv[0]))
        
        self.pool_out_height.append(
            self.compute_conv_dim(
                self.conv_out_height[0], self.kernel_size_pool[0], self.stride_pool[0], self.padding_pool[0]))
        self.pool_out_width.append(
            self.compute_conv_dim(
                self.conv_out_width[0], self.kernel_size_pool[0], self.stride_pool[0], self.padding_pool[0]))
        
        self.conv_out_height.append(
            self.compute_conv_dim(
                self.pool_out_height[0], self.kernel_size_conv[1], self.stride_conv[1], self.padding_conv[1]))
        self.conv_out_width.append(
            self.compute_conv_dim(
                self.pool_out_width[0], self.kernel_size_conv[1], self.stride_conv[1], self.padding_conv[1]))
        
        self.pool_out_height.append(
            self.compute_conv_dim(
                self.conv_out_height[1], self.kernel_size_pool[1], self.stride_pool[1], self.padding_pool[1]))
        self.pool_out_width.append(
            self.compute_conv_dim(
                self.conv_out_width[1], self.kernel_size_pool[1], self.stride_pool[1], self.padding_pool[1]))
        
    def forward(self, x_img):
        
        x_img = x_img.reshape(x_img.shape[0], 1, x_img.shape[1], x_img.shape[2])
        
        x_img = self.conv1(x_img)
        x_img = self.pool1(x_img)
        
        x_img = self.conv2(x_img)
        x_img = self.pool2(x_img)
        
        return x_img.squeeze()

class ResNet(nn.Module):
    
    def __init__(self, output_size, freeze_all_but_last_layer: bool):
        super(ResNet, self).__init__()
        self.name = 'ResNet'
        self.freeze_all_but_last_layer = freeze_all_but_last_layer
        self.resnet18 = resnet18(pretrained=True)
        self.output_size = output_size
        if self.freeze_all_but_last_layer:
            for param in self.resnet18.parameters():
                param.requires_grad = False
        n_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(n_features, self.output_size)
        
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def forward(self, x_img):
        x_img = x_img.view(-1, 3, 224, 224)
        x_img = self.normalize(x_img)
        return self.resnet18(x_img)

class DeepSense(nn.Module):
    def __init__(self):
        super(DeepSense, self).__init__()
        self.name = 'DeepSense'
        
        raise NotImplementedError

def image_feature_extraction(x_img, net):
    if net.img_extraction_method=='lenet':
        x_img_features = []
        for i in range(x_img.shape[1]):
            img = x_img[:, i, :, :]
            img = net.lenet(img)
            x_img_features.append(img)
        
    elif net.img_extraction_method=='resnet':
        x_img_features = []
        for i in range(x_img.shape[1]):
            img = x_img[:, i, :, :, :]
            img = net.resnet(img)
            x_img_features.append(img)
    
    else:
        raise NotImplementedError()

    return torch.stack(x_img_features).view(x_img.shape[0], x_img.shape[1], -1)