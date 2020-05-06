# BigGAN Generators with Pretrained Weights in Pytorch 
Pytorch implementation of the generator of Large Scale GAN Training for High Fidelity Natural Image Synthesis (BigGAN). 

# Download Pretrained Weights 
The Pretrained weights can be downloaded from the latest release. [link](
https://github.com/ivclab/BigGAN-Generator-Pretrained-Pytorch/releases/latest) 

# Dependencies 
dependencies:
  - python=3.6
  - cudatoolkit=10.0
  - pytorch
  - torchvision
  - scipy

Please also refer to the environment.yml file. 

# Demo 
- To run the code, please download the pretrained weights first.
```shell 
python demo.py -w <PRETRAINED_WEIGHT_PATH> [-s IMAGE_SIZE] [-c CLASS_LABEL] [-t TRUNCATION] 
python demo.py -w ./biggan512-release.pt -s 512 -t 0.3 -c 156 
python demo.py -w ./biggan256-release.pt -s 256 -t 0.02 -c 11 
python demo.py --pretrained_weight ./biggan128-release.pt --size 128 --truncation 0.2 --class_label 821 
``` 
- Valid image size: 128, 256, 512
- Valid class label: 0~999
- Valid truncation: 0.02~1.0


# Results 
|![alt text](./assets/p1.png)|
|:--:|
|*class 156 (512 x 512)*|
|![alt text](./assets/p2.png)|
|*class 11 (512 x 512)*|
|![alt text](./assets/p3.png)|
|*class 821 (512 x 512)*|


# Pretrained Weights 
The pretrained weights are converted from the tensorflow hub modules: 
- https://tfhub.dev/deepmind/biggan-128/2  
- https://tfhub.dev/deepmind/biggan-256/2 
- https://tfhub.dev/deepmind/biggan-512/2  


# References 
paper: https://arxiv.org/abs/1809.11096

https://github.com/ajbrock/BigGAN-PyTorch

# Contact 

Please feel free to leave suggestions or comments to [Tsung-Hung Hsieh](https://github.com/nemothh)(andrewhsiehth@gmail.com).
