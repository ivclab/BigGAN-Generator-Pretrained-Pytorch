from .spectral_normalization import spectral_norm 
from .batch_normalization import HyperBN2d 
from .util import depth_to_space 

import torch 
from torch import nn 

class G_z(nn.Module): 
    def __init__(self, in_features, out_features, eps=1e-4): 
        super().__init__() 
        self.g_linear = spectral_norm(nn.Linear(in_features=in_features, out_features=out_features), eps=eps) 

    def forward(self, x): 
        x = self.g_linear(x) 
        return x 
    
class GBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, latent_dim, eps=1e-4):   
        super().__init__() 
        self.conv0 = spectral_norm(nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding 
        ), eps=eps)
        self.conv1 = spectral_norm(nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding 
        ), eps=eps) 
        self.conv_sc = spectral_norm(nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0
        ), eps=eps) 
        self.hyperbn = HyperBN2d(num_features=in_channels, latent_dim=latent_dim, eps=eps) 
        self.hyperbn_1 = HyperBN2d(num_features=out_channels, latent_dim=latent_dim, eps=eps) 

    def forward(self, x, condition, truncation=1.0): 
        sc = torch.cat((x, x, x, x), dim=1) 
        sc = depth_to_space(sc, r=2)  
        sc = self.conv_sc(sc) 
        x = self.hyperbn(x, condition, truncation) 
        x = torch.relu(x)  
        x = torch.cat((x, x, x, x), dim=1) 
        x = depth_to_space(x, r=2)  
        x = self.conv0(x)    
        x = self.hyperbn_1(x, condition, truncation)  
        x = torch.relu(x)  
        x = self.conv1(x)  
        x = sc + x 
        return x 
