#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:39:50 2023

@author: acxyle

Purpose:
    this section provides the 'anatomic' structure of neural network models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16_bn(nn.Module):
    def __init__(self, num_classes, mode='classification'):
        super().__init__()
        
        self.mode = mode
        
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn_1_1 = nn.BatchNorm2d(64)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn_1_2 = nn.BatchNorm2d(64)
        
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn_2_1 = nn.BatchNorm2d(128)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn_2_2 = nn.BatchNorm2d(128)
        
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn_3_1 = nn.BatchNorm2d(256)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn_3_2 = nn.BatchNorm2d(256)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn_3_3 = nn.BatchNorm2d(256)
        
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn_4_1 = nn.BatchNorm2d(512)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn_4_2 = nn.BatchNorm2d(512)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn_4_3 = nn.BatchNorm2d(512)
        
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn_5_1 = nn.BatchNorm2d(512)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn_5_2 = nn.BatchNorm2d(512)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn_5_3 = nn.BatchNorm2d(512)
        
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)


    def forward(self, x):

        x01 = self.conv_1_1(x)
        x02 = self.bn_1_1(x01)
        x03 = F.relu(x02)
        x04 = self.conv_1_2(x03)
        x05 = self.bn_1_2(x04)
        x06 = F.relu(x05)
        
        x07 = F.max_pool2d(x06, 2, 2)

        x08 = self.conv_2_1(x07)
        x09 = self.bn_2_1(x08)
        x10 = F.relu(x09)
        x11 = self.conv_2_2(x10)
        x12 = self.bn_2_2(x11)
        x13 = F.relu(x12)
        
        x14 = F.max_pool2d(x13, 2, 2)

        x15 = self.conv_3_1(x14)
        x16 = self.bn_3_1(x15)
        x17 = F.relu(x16)
        x18 = self.conv_3_2(x17)
        x19 = self.bn_3_2(x18)
        x20 = F.relu(x19)
        x21 = self.conv_3_3(x20)
        x22 = self.bn_3_3(x21)
        x23 = F.relu(x22)

        x24 = F.max_pool2d(x23, 2, 2)
        
        x25 = self.conv_4_1(x24)
        x26 = self.bn_4_1(x25)
        x27 = F.relu(x26)
        x28 = self.conv_4_2(x27)
        x29 = self.bn_4_2(x28)
        x30 = F.relu(x29)
        x31 = self.conv_4_3(x30)
        x32 = self.bn_4_3(x31)
        x33 = F.relu(x32)
        
        x34 = F.max_pool2d(x33, 2, 2)

        x35 = self.conv_5_1(x34)
        x36 = self.bn_5_1(x35)
        x37 = F.relu(x36)
        x38 = self.conv_5_2(x37)
        x39 = self.bn_5_2(x38)
        x40 = F.relu(x39)
        x41 = self.conv_5_3(x40)
        x42 = self.bn_5_3(x41)
        x43 = F.relu(x42)

        x44 = F.max_pool2d(x43, 2, 2)
        
        x45 = self.avgpool(x44)
        
        x45 = x45.flatten(1)
        
        x46 = self.fc6(x45)
        x47 = F.relu(x46)
        x48 = F.dropout(x47, 0.5)
        x49 = self.fc7(x48)
        x50 = F.relu(x49)
        x51 = F.dropout(x50, 0.5)
        x52 = self.fc8(x51)

        feat = [
                      x01, x02, x03, x04, x05, x06, 
                      x08, x09, x10, x11, x12, x13, 
                      x15, x16, x17, x18, x19, x20, x21, x22, x23,
                      x25, x26, x27, x28, x29, x30, x31, x32, x33, 
                      x35, x36, x37, x38, x39, x40, x41, x42, x43, 
                      x45,
                      x46, x47,
                      x49, x50, 
                      x52
        ]
        
        if self.mode == 'feature_map':
            return feat # for feature_map
        
        if self.mode == 'classification':
            return x52  # for finetune