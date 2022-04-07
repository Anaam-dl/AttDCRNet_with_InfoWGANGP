import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
from torch.autograd import Variable
from imgaug import augmenters as iaa
from sklearn.cluster import KMeans
from math import *
import cv2

from imgaug import augmenters as iaa

class avgpool(nn.Module):
    def __init__(self, up_size=0):
        super(avgpool, self).__init__()
        
    def forward(self, x):
        out_man = (x[:,:,::2,::2] + x[:,:,1::2,::2] + x[:,:,::2,1::2] + x[:,:,1::2,1::2]) / 4
        return out_man
    
class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, resample=None, up_size=0):
        super(ResidualBlock, self).__init__()
        if resample == 'up':
            self.bn1 = nn.BatchNorm2d(in_dim)
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.upsample = torch.nn.Upsample(scale_factor=2)#up_size,2
            self.upsample_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
            self.bn2 = nn.BatchNorm2d(out_dim)
            
        elif resample == 'down':
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
            self.pool = avgpool()
            self.pool_conv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=True)
        
        elif resample == None:
            self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias=True)
            
        self.resample = resample

    def forward(self, x):
        
        if self.resample == None:
            shortcut = x
            output = x
            
            output = nn.functional.relu(output)
            output = self.conv1(output)
            output = nn.functional.relu(output)
            output = self.conv2(output)
            
        elif self.resample == 'up':
            shortcut = x
            output = x
            
            shortcut = self.upsample(shortcut) #upsampleconv
            shortcut = self.upsample_conv(shortcut)
            
            output = self.bn1(output)
            output = nn.functional.relu(output)
            output = self.conv1(output)

            output = self.bn2(output)
            output = nn.functional.relu(output)
            output = self.upsample(output) #upsampleconv
            output = self.conv2(output)
                        
        elif self.resample == 'down':
            shortcut = x
            output = x
            
            shortcut = self.pool_conv(shortcut) #convmeanpool
            shortcut = self.pool(shortcut)
            
            output = nn.functional.relu(output)
            output = self.conv1(output)
            
            output = nn.functional.relu(output)
            output = self.conv2(output)    #convmeanpool
            output = self.pool(output)
            
        return output+shortcut

#create_G_architecture
class generator(nn.Module):

    def __init__(self, rand=128):
        super(generator, self).__init__()
        self.rand = rand
        self.linear = nn.Linear(rand  ,2048, bias=True)
        self.layer_up_1 = ResidualBlock(128, 128, 'up', up_size=8)
        self.layer_up_2 = ResidualBlock(128, 128, 'up', up_size=16)
        self.layer_up_3 = ResidualBlock(128, 128, 'up', up_size=32)
        self.layer_up_4 = ResidualBlock(128, 128, 'up', up_size=64)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv_last = nn.Conv2d(128, 1, 3, 1, 1, bias=True)#3

    def forward(self, x):
        x = x.view(-1,self.rand)
        x = self.linear(x)
        x = x.view(-1,128,4,4)
        x = self.layer_up_1(x)
        x = self.layer_up_2(x)
        x = self.layer_up_3(x)
        x = self.layer_up_4(x) 
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.conv_last(x)
        #x = nn.functional.tanh(x)
        x = torch.tanh(x)#!
        return x

# intialization
def uniform(stdev, size):
    return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

def initialize_conv(m,he_init=True):
    fan_in = m.in_channels * m.kernel_size[0]**2
    fan_out = m.out_channels * m.kernel_size[0]**2 / (m.stride[0]**2)

    if m.kernel_size[0]==3:
        filters_stdev = np.sqrt(4./(fan_in+fan_out))
    # Normalized init (Glorot & Bengio)
    else: 
        filters_stdev = np.sqrt(2./(fan_in+fan_out))
        
    filter_values = uniform(
                    filters_stdev,
                    (m.kernel_size[0], m.kernel_size[0], m.in_channels, m.out_channels)
                )
    
    return filter_values

def initialize_linear(m):
    weight_values = uniform(
                np.sqrt(2./(m.in_features+m.out_features)),
                (m.in_features, m.out_features)
            )
    return weight_values

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight = torch.from_numpy(np.transpose(initialize_conv(m)))
        m.weight.data.copy_(weight)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_values = torch.from_numpy(np.transpose(initialize_linear(m)))
        m.weight.data.copy_(weight_values)
        m.bias.data.fill_(0)
        
#sample from categorical distribution
def sample_c(batchsize=32, dis_category=5):
    rand_c = np.zeros((batchsize,dis_category),dtype='float32')
    for i in range(0,batchsize):
        rand = np.random.multinomial(1, dis_category*[1/float(dis_category)], size=1)
        rand_c[i] = rand

    label_c = np.argmax(rand_c,axis=1)
    label_c = torch.LongTensor(label_c.astype('int'))
    rand_c = torch.from_numpy(rand_c.astype('float32'))
    return rand_c, label_c



def create_model(rand=100, dis_category=4):
    netG= generator(rand = rand+dis_category)
    netG.apply(weights_init)
    #netG = netG.cuda()
    return netG

