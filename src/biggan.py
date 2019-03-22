from .spectral_normalization import spectral_norm 
from .batch_normalization import ScaledCrossReplicaBN2d 
from .attention import SelfAttention2d 
from .module import G_z 
from .module import GBlock  

import torch 
from torch import nn 
from torch.nn import functional as F 

class BigGAN128(nn.Module): 
    def __init__(self): 
        super().__init__() 
        ch = 96 
        self.linear = nn.Embedding(num_embeddings=1000, embedding_dim=128) 
        self.g_z = G_z(in_features=20, out_features=4*4*16*ch, eps=1e-4) 
        self.gblock = GBlock(in_channels=16*ch, out_channels=16*ch, kernel_size=3, stride=1, padding=1, latent_dim=148, eps=1e-4) 
        self.gblock_1 = GBlock(in_channels=16*ch, out_channels=8*ch, kernel_size=3, stride=1, padding=1, latent_dim=148, eps=1e-4)
        self.gblock_2 = GBlock(in_channels=8*ch, out_channels=4*ch, kernel_size=3, stride=1, padding=1, latent_dim=148, eps=1e-4)
        self.gblock_3 = GBlock(in_channels=4*ch, out_channels=2*ch, kernel_size=3, stride=1, padding=1, latent_dim=148, eps=1e-4)
        self.attention = SelfAttention2d(in_channels=2*ch, c_bar=ch//4, c_hat=ch) 
        self.gblock_4 = GBlock(in_channels=2*ch, out_channels=1*ch, kernel_size=3, stride=1, padding=1, latent_dim=148, eps=1e-4)
        self.scaledcrossreplicabn = ScaledCrossReplicaBN2d(num_features=ch, eps=1e-4) 
        self.conv_2d = spectral_norm(nn.Conv2d(in_channels=ch, out_channels=3, kernel_size=3, stride=1, padding=1), eps=1e-4)

    def forward(self, z, c, truncation=1.0):  
        z_gz, z_0, z_1, z_2, z_3, z_4 = torch.split(z, split_size_or_sections=20, dim=1)  
        cond = self.linear(c)  
        x = self.g_z(z_gz) 
        x = x.view(z.size(0), 4, 4, -1).permute(0, 3, 1, 2)  
        x = self.gblock(x, torch.cat((z_0, cond), dim=1), truncation) 
        x = self.gblock_1(x, torch.cat((z_1, cond), dim=1), truncation)  
        x = self.gblock_2(x, torch.cat((z_2, cond), dim=1), truncation)  
        x = self.gblock_3(x, torch.cat((z_3, cond), dim=1), truncation) 
        x = self.attention(x)  
        x = self.gblock_4(x, torch.cat((z_4, cond), dim=1), truncation)  
        x = self.scaledcrossreplicabn(x, truncation) 
        x = torch.relu(x) 
        x = self.conv_2d(x)  
        x = torch.tanh(x) 
        return x 

class BigGAN256(nn.Module): 
    def __init__(self): 
        super().__init__() 
        ch = 96 
        self.linear = nn.Embedding(num_embeddings=1000, embedding_dim=128) 
        self.g_z = G_z(in_features=20, out_features=4*4*16*ch, eps=1e-4) 
        self.gblock = GBlock(in_channels=16*ch, out_channels=16*ch, kernel_size=3, stride=1, padding=1, latent_dim=148, eps=1e-4) 
        self.gblock_1 = GBlock(in_channels=16*ch, out_channels=8*ch, kernel_size=3, stride=1, padding=1, latent_dim=148, eps=1e-4)
        self.gblock_2 = GBlock(in_channels=8*ch, out_channels=8*ch, kernel_size=3, stride=1, padding=1, latent_dim=148, eps=1e-4)
        self.gblock_3 = GBlock(in_channels=8*ch, out_channels=4*ch, kernel_size=3, stride=1, padding=1, latent_dim=148, eps=1e-4)
        self.gblock_4 = GBlock(in_channels=4*ch, out_channels=2*ch, kernel_size=3, stride=1, padding=1, latent_dim=148, eps=1e-4)
        self.attention = SelfAttention2d(in_channels=2*ch, c_bar=ch//4, c_hat=ch) 
        self.gblock_5 = GBlock(in_channels=2*ch, out_channels=1*ch, kernel_size=3, stride=1, padding=1, latent_dim=148, eps=1e-4)
        self.scaledcrossreplicabn = ScaledCrossReplicaBN2d(num_features=ch, eps=1e-4) 
        self.conv_2d = spectral_norm(nn.Conv2d(in_channels=ch, out_channels=3, kernel_size=3, stride=1, padding=1), eps=1e-4)

    def forward(self, z, c, truncation=1.0):  
        z_gz, z_0, z_1, z_2, z_3, z_4, z_5 = torch.split(z, split_size_or_sections=20, dim=1)  
        cond = self.linear(c)  
        x = self.g_z(z_gz) 
        x = x.view(z.size(0), 4, 4, -1).permute(0, 3, 1, 2)  
        x = self.gblock(x, torch.cat((z_0, cond), dim=1), truncation) 
        x = self.gblock_1(x, torch.cat((z_1, cond), dim=1), truncation)  
        x = self.gblock_2(x, torch.cat((z_2, cond), dim=1), truncation)  
        x = self.gblock_3(x, torch.cat((z_3, cond), dim=1), truncation) 
        x = self.gblock_4(x, torch.cat((z_4, cond), dim=1), truncation) 
        x = self.attention(x)  
        x = self.gblock_5(x, torch.cat((z_5, cond), dim=1), truncation)  
        x = self.scaledcrossreplicabn(x, truncation) 
        x = torch.relu(x) 
        x = self.conv_2d(x)  
        x = torch.tanh(x) 
        return x 

class BigGAN512(nn.Module): 
    def __init__(self): 
        super().__init__() 
        ch = 96 
        self.linear = nn.Embedding(num_embeddings=1000, embedding_dim=128) 
        self.g_z = G_z(in_features=16, out_features=4*4*16*ch, eps=1e-4) 
        self.gblock = GBlock(in_channels=16*ch, out_channels=16*ch, kernel_size=3, stride=1, padding=1, latent_dim=144, eps=1e-4) 
        self.gblock_1 = GBlock(in_channels=16*ch, out_channels=8*ch, kernel_size=3, stride=1, padding=1, latent_dim=144, eps=1e-4)
        self.gblock_2 = GBlock(in_channels=8*ch, out_channels=8*ch, kernel_size=3, stride=1, padding=1, latent_dim=144, eps=1e-4)
        self.gblock_3 = GBlock(in_channels=8*ch, out_channels=4*ch, kernel_size=3, stride=1, padding=1, latent_dim=144, eps=1e-4)
        self.attention = SelfAttention2d(in_channels=4*ch, c_bar=ch//2, c_hat=2*ch) 
        self.gblock_4 = GBlock(in_channels=4*ch, out_channels=2*ch, kernel_size=3, stride=1, padding=1, latent_dim=144, eps=1e-4)
        self.gblock_5 = GBlock(in_channels=2*ch, out_channels=1*ch, kernel_size=3, stride=1, padding=1, latent_dim=144, eps=1e-4)
        self.gblock_6 = GBlock(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1, latent_dim=144, eps=1e-4)
        self.scaledcrossreplicabn = ScaledCrossReplicaBN2d(num_features=ch, eps=1e-4) 
        self.conv_2d = spectral_norm(nn.Conv2d(in_channels=ch, out_channels=3, kernel_size=3, stride=1, padding=1), eps=1e-4)

    def forward(self, z, c, truncation=1.0):  
        z_gz, z_0, z_1, z_2, z_3, z_4, z_5, z_6 = torch.split(z, split_size_or_sections=16, dim=1)  
        cond = self.linear(c)  
        x = self.g_z(z_gz) 
        x = x.view(z.size(0), 4, 4, -1).permute(0, 3, 1, 2)  
        x = self.gblock(x, torch.cat((z_0, cond), dim=1), truncation) 
        x = self.gblock_1(x, torch.cat((z_1, cond), dim=1), truncation)  
        x = self.gblock_2(x, torch.cat((z_2, cond), dim=1), truncation)  
        x = self.gblock_3(x, torch.cat((z_3, cond), dim=1), truncation) 
        x = self.attention(x)  
        x = self.gblock_4(x, torch.cat((z_4, cond), dim=1), truncation) 
        x = self.gblock_5(x, torch.cat((z_5, cond), dim=1), truncation)  
        x = self.gblock_6(x, torch.cat((z_6, cond), dim=1), truncation)  
        x = self.scaledcrossreplicabn(x, truncation) 
        x = torch.relu(x) 
        x = self.conv_2d(x)  
        x = torch.tanh(x) 
        return x 
