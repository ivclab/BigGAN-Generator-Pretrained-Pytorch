from .spectral_normalization import spectral_norm 

import torch 
from torch import nn 
from torch.nn import functional as F 


class SelfAttention2d(nn.Module): 
    def __init__(self, in_channels, c_bar, c_hat, eps=1e-4): 
        super().__init__() 
        self.theta = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=c_bar, kernel_size=1, bias=False), eps=eps) 
        self.phi = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=c_bar, kernel_size=1, bias=False), eps=eps)
        self.g = spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=c_hat, kernel_size=1, bias=False), eps=eps) 
        self.o_conv = spectral_norm(nn.Conv2d(in_channels=c_hat, out_channels=in_channels, kernel_size=1, bias=False), eps=eps) 
        self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, x): 
        n, c, h, w = x.size()  
        g_x = self.g(x)  
        g_x = F.max_pool2d(g_x, kernel_size=2) 
        g_x = g_x.view(n, -1, h*w//4) 
        phi_x = self.phi(x) 
        phi_x = F.max_pool2d(phi_x, kernel_size=2) 
        phi_x = phi_x.view(n, -1, h*w//4) 
        theta_x = self.theta(x)  
        theta_x = theta_x.view(n, -1, h*w)   
        attn = F.softmax(torch.bmm(theta_x.permute(0, 2, 1), phi_x), dim=-1)   
        y = torch.bmm(g_x, attn.permute(0, 2, 1))  
        y = y.view(n, -1, h, w)  
        o = self.o_conv(y)  
        z = self.gamma * o + x  
        return z 