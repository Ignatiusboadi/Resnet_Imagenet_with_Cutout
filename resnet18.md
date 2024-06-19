# Image Classification with Custom ResNet

## Overview
This project implements image classification using a custom ResNet architecture on the ImageNet dataset. It employs deep learning techniques in PyTorch to achieve accurate predictions for various object categories.

## Prerequisites
Ensure you have the following installed:
- Python 3.7
- PyTorch
- torchvision
- NumPy
- pandas
- matplotlib
- tqdm


## Model Architecture
Custom ResNet
The model architecture includes:

- Initial convolution layer (7x7 kernel, 64 channels, stride 2)
- Max pooling layer (3x3 kernel, stride 2)
- Four blocks with increasing channel sizes: 64, 128, 256, 512
- Adaptive average pooling and fully connected layer (1000-dimensional output for ImageNet classes)

## Training
### Loss Function and Optimization
- Loss Function: Cross-entropy loss
- Optimizer: Adam optimizer with learning rate of 0.001
- Learning Rate Scheduler: StepLR with step size 7 and gamma 0.1
##Results
###Performance Metrics
Best validation accuracy achieved: 39%% after 5 epoch
