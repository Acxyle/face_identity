#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:12:39 2023

@author: acxyle
    
    rewrite from https://github.com/cydonia999/VGGFace2-pytorch/blob/master/models/resnet.py
    
    focusing on feature extraction, coorperate with the modification of forward() of nn.Sequential() of container.py as follow:
        
        # -----
        def forward(self, input):    
            
            def _feature_list_forward(feature, module):
                list_or_tensor = module(feature[-1])     # take the last element
                if isinstance(list_or_tensor, list):     # extend/append the new output to the feature list
                    feature.extend(list_or_tensor)
                elif isinstance(list_or_tensor, torch.Tensor):
                    feature.append(list_or_tensor)
                return feature
                
            if isinstance(input, torch.Tensor):     # for 1st forward of every module
                for module in self:
                    input = module(input) if not isinstance(input, list) else _feature_list_forward(input, module)
            elif isinstance(input, list):     # from 2nd forward
                for module in self:
                    input = _feature_list_forward(input, module)
            else:
                raise ValueError
                
            return input
        # -----
    
    Please manually add nn.Softmax(dim)(input) and rewrite the forward() accordingly for normal use
    
"""

import torch
import torch.nn as nn
import math

__all__ = ['ResNet', 
           "resnet18",
           "resnet34",
           "resnet50",
           "resnet101",
           "resnet152"]

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):  # for resnet18() and resnet34()

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x01 = self.conv1(x)
        x02 = self.bn1(x01)
        x03 = self.relu(x02)

        x04 = self.conv2(x03)
        x05 = self.bn2(x04)

        if self.downsample is not None:
            residual = self.downsample(x)

        x06 = x05 + residual
        x07 = self.relu(x06)
        
        feature = [x01, x02, x03, x04, x05, x06, x07]

        return feature


class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        x01 = self.conv1(x)
        x02 = self.bn1(x01)
        x03 = self.relu(x02)

        x04 = self.conv2(x03)
        x05 = self.bn2(x04)
        x06 = self.relu(x05)

        x07 = self.conv3(x06)
        x08 = self.bn3(x07)

        if self.downsample is not None:
            residual = self.downsample(x)

        x09 = x08 + residual
        x10 = self.relu(x09)
        
        feature = [x01, x02, x03, x04, x05, x06, x07, x08, x09, x10]     
        
        return feature


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        
        self.inplanes = 64
        self.include_top = include_top
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers) 
  
    def forward(self, x): 
        x01_ = self.conv1(x)
        x02_ = self.bn1(x01_)
        x03_ = self.relu(x02_)
        x04_ = self.maxpool(x03_)
        
        x_list1 = self.layer1(x04_)
        x_list2 = self.layer2(x_list1)
        x_list3 = self.layer3(x_list2)
        x_list4 = self.layer4(x_list3)      

        x05_ = self.avgpool(x_list4[-1])
        
        if not self.include_top:
            
            all_feature_map = [x01_, x02_, x03_, x04_, *x_list4, x05_]    
                
            return all_feature_map
        
        else:
        
            x05_ = x05_.view(x05_.size(0), -1)
            
            x06_ = self.fc(x05_)
            
            all_feature_map = [x01_, x02_, x03_, x04_, *x_list4, x05_, nn.ReLU()(x06_)]  
            
            return all_feature_map


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

if __name__ == '__main__':
    
    model = resnet18()
    x = torch.randn(1,3,224,224)
    out = model(x)
    for idx, layer in enumerate(out):
        print(idx, layer.shape, layer.numel())