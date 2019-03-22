from src.biggan import BigGAN128
from src.biggan import BigGAN256 
from src.biggan import BigGAN512 

import torch 
import torchvision 

from scipy.stats import truncnorm 

import argparse 

if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('-t', '--truncation', type=float, default=0.4) 
    parser.add_argument('-s', '--size', type=int, choices=[128, 256, 512], default=512) 
    parser.add_argument('-c', '--class_label', type=int, choices=range(0, 1000), default=156) 
    parser.add_argument('-w', '--pretrained_weight', type=str, required=True)
    args = parser.parse_args() 

    truncation = torch.clamp(torch.tensor(args.truncation), min=0.02+1e-4, max=1.0-1e-4).float()  
    c = torch.tensor((args.class_label,)).long()

    if args.size == 128: 
        z = truncation * torch.as_tensor(truncnorm.rvs(-2.0, 2.0, size=(1, 120))).float() 
        biggan = BigGAN128() 
    elif args.size == 256: 
        z = truncation * torch.as_tensor(truncnorm.rvs(-2.0, 2.0, size=(1, 140))).float() 
        biggan = BigGAN256()
    elif args.size == 512: 
        z = truncation * torch.as_tensor(truncnorm.rvs(-2.0, 2.0, size=(1, 128))).float() 
        biggan = BigGAN512() 

    biggan.load_state_dict(torch.load(args.pretrained_weight)) 
    biggan.eval() 
    with torch.no_grad(): 
        img = biggan(z, c, truncation.item())  

    img = 0.5 * (img.data + 1) 
    pil = torchvision.transforms.ToPILImage()(img.squeeze()) 
    pil.show()

