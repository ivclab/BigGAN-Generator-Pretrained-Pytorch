from .spectral_normalization import spectral_norm 

import torch 
from torch import nn 
from torch.nn import functional as F 

class CrossReplicaBN2d(nn.BatchNorm2d):  
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True): 
        super().__init__(num_features, eps, momentum, affine, track_running_stats) 
        self.register_buffer('standing_means', torch.empty(50, num_features)) 
        self.register_buffer('standing_vars', torch.empty(50, num_features)) 
    
    @torch._jit_internal.weak_script_method 
    def forward(self, input, truncation=1.0): 
        self._check_input_dim(input) 
        exponential_average_factor = 0.0 
        if self.training and self.track_running_stats: 
            if self.num_batches_tracked is not None: 
                self.num_batches_tracked += 1 
                if self.momentum is None: 
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked) 
                else: 
                    exponential_average_factor = self.momentum 
        if not self.training: 
            standing_mean = self.get_standing_stats(self.standing_means, truncation) 
            standing_var = self.get_standing_stats(self.standing_vars, truncation) 
            return F.batch_norm(
                input, standing_mean, standing_var, self.weight, self.bias, 
                self.training or not self.track_running_stats, 
                exponential_average_factor, self.eps 
            )  
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias, 
            self.training or not self.track_running_stats, 
            exponential_average_factor, self.eps 
        ) 

    def get_standing_stats(self, stack, truncation): 
        min = 0.02 - 1e-12  
        max = 1.0 + 1e-12  
        step = 0.02  
        assert(min <= truncation and truncation <= max) 
        idx = round((truncation - step) / step) 
        residual = truncation - idx * step 
        alpha = round(residual / step, 2)  
        ret = torch.sum(torch.cat((alpha*stack[idx:idx+1], (1.0-alpha)*stack[idx+1:idx+2])), dim=0) 
        return ret   

class ScaledCrossReplicaBN2d(CrossReplicaBN2d):   
    pass 

class HyperBN2d(nn.Module):  
    def __init__(self, num_features, latent_dim, eps=1e-4):  
        super().__init__() 
        self.crossreplicabn = CrossReplicaBN2d(num_features=num_features, affine=False, eps=eps) 
        self.gamma = spectral_norm(nn.Linear(in_features=latent_dim, out_features=num_features, bias=False), eps=eps)  
        self.beta = spectral_norm(nn.Linear(in_features=latent_dim, out_features=num_features, bias=False), eps=eps)   

    def forward(self, x, condition, truncation=1.0):  
        return (self.gamma(condition).view(condition.size(0), -1, 1, 1) + 1) * self.crossreplicabn(x, truncation) + self.beta(condition).view(condition.size(0), -1, 1, 1) 

