import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cbam import *
from copy import deepcopy
# =============================================================================

class DCRBlock(nn.Module):
    def __init__(self, in_channel, out_channel, use_cbam=False):
        super(DCRBlock, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.prlu1 = nn.PReLU()
        
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channel)
        self.prlu2 = nn.PReLU()
        
        self.conv3 = nn.Conv2d(self.out_channel, self.out_channel, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_channel)
        self.prlu3 = nn.PReLU()
        
        self.conv4 = nn.Conv2d(self.out_channel, self.out_channel, 1, 1, 0, bias=False)
        self.bn4 = nn.BatchNorm2d(self.out_channel)
        self.prlu4 = nn.PReLU()
        
        self.conv5 = nn.Conv2d(self.out_channel, self.out_channel, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.out_channel)
        self.prlu5 = nn.PReLU()
        
        self.conv6 = nn.Conv2d(self.out_channel, self.out_channel, 1, 1, 0, bias=False)
        self.bn6 = nn.BatchNorm2d(self.out_channel)
        self.prlu6 = nn.PReLU()
        
        if use_cbam:
            self.cbam = CBAM(out_channel, 16)
        else:
            self.cbam = None
        
    def forward(self, x):    
        out_1 = self.prlu1(self.bn1(self.conv1(x)))
        cc_1 = out_1
        out_2 = self.prlu2(self.bn2(self.conv2(out_1)))
        cc_2 = out_2
        out_3 = self.prlu3(self.bn3(self.conv3(out_2)))
        cc_3 = out_3
        out = self.prlu4(self.bn4(self.conv4(out_3)))
        out += cc_1
        out = self.prlu5(self.bn5(self.conv5(out)))
        out += cc_2
        out = self.prlu6(self.bn6(self.conv6(out)))
        if not self.cbam is None:
            out = self.cbam(out)        
        return out + cc_3
        
  
class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.prlu1 = nn.PReLU()
        
        self.conv2 = nn.Conv2d(self.out_channel, self.out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channel)
        self.prlu2 = nn.PReLU()
        
        self.conv3 = nn.Conv2d(self.out_channel, self.out_channel, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_channel)
        self.prlu3 = nn.PReLU()
        
    def forward(self, x): 
        out = self.prlu1(self.bn1(self.conv1(x)))
        out = self.prlu2(self.bn2(self.conv2(out)))
        out = self.prlu3(self.bn3(self.conv3(out)))
        return out

#create Att-DCRNet architecture
class AttDCRNET(nn.Module):
    def __init__(self, num_classes=6, channels=1):
        super(AttDCRNET, self).__init__()
        self.classes = num_classes
        self.channels = channels
        self.convblock1 = ConvBlock(self.channels, 16)
        self.convblock2 = ConvBlock(16, 32)
        
        self.dcrblock1 = DCRBlock(32, 64, use_cbam=True) 
        self.dcrblock2 = DCRBlock(64, 128, use_cbam=True)
        self.dcrblock3 = DCRBlock(128, 256, use_cbam=True)
        
        self.fc1 = nn.Linear(4*4*256, 512, bias=True)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc3 = nn.Linear(512, self.classes, bias=True)
        #self.softmax = nn.LogSoftmax(dim=1)
        self.dropout1 = nn.Dropout(0.5, inplace=True)
        self.dropout2 = nn.Dropout(0.5, inplace=True)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.prlu1 = nn.PReLU()
        
        self.bn2 = nn.BatchNorm1d(512)
        self.prlu2 = nn.PReLU()
        
    # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #out = self.model_half1(x)
        out = self.convblock1(x)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.convblock2(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.dcrblock1(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.dcrblock2(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        out = self.dcrblock3(out)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        
        out = torch.flatten(out,1)
        #out = self.model_half2(out)
        
        out = self.prlu1(self.bn1(self.fc1(out)))
        out = self.dropout1(out)
        out = self.prlu2(self.bn2(self.fc2(out)))
        out = self.dropout2(out)
        out = self.fc3(out)
        return out



