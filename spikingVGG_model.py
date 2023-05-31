#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:44:41 2023

@author: acxyle

    应当说明，默认这个方法下取出的特征并不包括 maxpool，但可以根据需求自定义获取所需要的特征层。
    与 resnet 不同，resnet 下由于网络层数太多，并没有采取手动写出所有层的方法（此code），而是改写了
    pytorch 里 container.py 内 module 中 forward 函数的源码，直接获取所有 module 的输出，然后再设计
    filter 根据需求挑出需要的特征层（默认也是丢弃maxpool，但即便如此 resnet50 网络还是输出了 166 个特征层，
    不利于查看，有待优化）
    
    [Mar 10, 2023 更新] 手动写出全部结构和 forward 的方法不再被使用，新版查看生成模型
"""

import torch
import torch.nn as nn

from copy import deepcopy
from spikingjelly.activation_based import layer
from spikingjelly.activation_based import functional, neuron, surrogate

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

class SpikingVGG16bn_f(nn.Module):      # this version is for VGG1bbn
    def __init__(self, mode='classification', batch_norm=False, num_classes=1000, init_weights=True,
                 spiking_neuron: callable = None, **kwargs):
        super(SpikingVGG16bn_f, self).__init__()
        
        self.mode = mode
        norm_layer = layer.BatchNorm2d
        
        print("------ creating model from SpikingVGG16bn featuremap -----")
        
        self.conv_1_1 = layer.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn_1_1 = norm_layer(64)
        self.neuron_1_1 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_1_2 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_1_2 = norm_layer(64)
        self.neuron_1_2 = spiking_neuron(**deepcopy(kwargs))
        
        self.maxpool_1 = layer.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_2_1 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_2_1 = norm_layer(128)
        self.neuron_2_1 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_2_2 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_2_2 = norm_layer(128)
        self.neuron_2_2 = spiking_neuron(**deepcopy(kwargs))
        
        self.maxpool_2 = layer.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_3_1 = layer.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn_3_1 = norm_layer(256)
        self.neuron_3_1 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_3_2 = layer.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_3_2 = norm_layer(256)
        self.neuron_3_2 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_3_3 = layer.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_3_3 = norm_layer(256)
        self.neuron_3_3 = spiking_neuron(**deepcopy(kwargs))
        
        self.maxpool_3 = layer.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_4_1 = layer.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn_4_1 = norm_layer(512)
        self.neuron_4_1 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_4_2 = layer.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_4_2 = norm_layer(512)
        self.neuron_4_2 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_4_3 = layer.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_4_3 = norm_layer(512)
        self.neuron_4_3 = spiking_neuron(**deepcopy(kwargs))
        
        self.maxpool_4 = layer.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_5_1 = layer.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_5_1 = norm_layer(512)
        self.neuron_5_1 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_5_2 = layer.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_5_2 = norm_layer(512)
        self.neuron_5_2 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_5_3 = layer.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_5_3 = norm_layer(512)
        self.neuron_5_3 = spiking_neuron(**deepcopy(kwargs))
        
        self.maxpool_5 = layer.MaxPool2d(kernel_size=2, stride=2)
        
        self.avgpool = layer.AdaptiveAvgPool2d((7, 7))
        
        self.fc_6 = layer.Linear(512 * 7 * 7, 4096)     # [notice]
        self.fc_6_n = spiking_neuron(**deepcopy(kwargs))
        self.fc_6_d = layer.Dropout()
        
        self.fc_7 = layer.Linear(4096, 4096)
        self.fc_7_n = spiking_neuron(**deepcopy(kwargs))
        self.fc_7_d = layer.Dropout()
        
        self.fc_8 = layer.Linear(4096, num_classes)
        
    def forward(self, x):
        
        x01 = self.conv_1_1(x)
        x02 = self.bn_1_1(x01)
        x03 = self.neuron_1_1(x02)
        
        x04 = self.conv_1_2(x03)
        x05 = self.bn_1_2(x04)
        x06 = self.neuron_1_2(x05)
        
        x07 = self.maxpool_1(x06)
        
        x08 = self.conv_2_1(x07)
        x09 = self.bn_2_1(x08)
        x10 = self.neuron_2_1(x09)
        
        x11 = self.conv_2_2(x10)
        x12 = self.bn_2_2(x11)
        x13 = self.neuron_2_2(x12)
        
        x14 = self.maxpool_2(x13)
        
        x15 = self.conv_3_1(x14)
        x16 = self.bn_3_1(x15)
        x17 = self.neuron_3_1(x16)
        
        x18 = self.conv_3_2(x17)
        x19 = self.bn_3_2(x18)
        x20 = self.neuron_3_2(x19)
        
        x21 = self.conv_3_3(x20)
        x22 = self.bn_3_3(x21)
        x23 = self.neuron_3_3(x22)

        x24 = self.maxpool_3(x23)
        
        x25 = self.conv_4_1(x24)
        x26 = self.bn_4_1(x25)
        x27 = self.neuron_4_1(x26)
        
        x28 = self.conv_4_2(x27)
        x29 = self.bn_4_2(x28)
        x30 = self.neuron_4_2(x29)
        
        x31 = self.conv_4_3(x30)
        x32 = self.bn_4_3(x31)
        x33 = self.neuron_4_3(x32)

        x34 = self.maxpool_4(x33)
        
        x35 = self.conv_5_1(x34)
        x36 = self.bn_5_1(x35)
        x37 = self.neuron_5_1(x36)
        
        x38 = self.conv_5_2(x37)
        x39 = self.bn_5_2(x38)
        x40 = self.neuron_5_2(x39)
        
        x41 = self.conv_5_3(x40)
        x42 = self.bn_5_3(x41)
        x43 = self.neuron_5_3(x42)
  
        x44 = self.maxpool_5(x43)
        
        #x45 = self.avgpool(x44)
        if self.avgpool.step_mode == 's':
            x45 = torch.flatten(x44, 1)
        elif self.avgpool.step_mode == 'm':
            x45 = torch.flatten(x44, 2)
        
        x46 = self.fc_6(x45)
        x47 = self.fc_6_n(x46)
        x48 = self.fc_6_d(x47)
        
        x49 = self.fc_7(x48)
        x50 = self.fc_7_n(x49)
        x51 = self.fc_7_d(x50)
        
        x52 = self.fc_8(x51)
                 
        feature = [
            x01, x02, x03, x04, x05, x06, 
            x08, x09, x10, x11, x12, x13, 
            x15, x16, x17, x18, x19, x20, x21, x22, x23,
            x25, x26, x27, x28, x29, x30, x31, x32, x33, 
            x35, x36, x37, x38, x39, x40, x41, x42, x43, 
            #x45,
            x46, x47,
            x49, x50, 
            x52
            ]

        if self.mode == 'feature':
            return feature
        
        elif self.mode == 'classification':    
            return x52

class SpikingVGG19bn_f(nn.Module):      # this version is for VGG19bn
    def __init__(self, mode='classification', batch_norm=False, num_classes=1000, init_weights=True,
                 spiking_neuron: callable = None, **kwargs):
        super(SpikingVGG19bn_f, self).__init__()
        
        self.mode = mode
        norm_layer = layer.BatchNorm2d
        
        print("------ creating model from SpikingVGG16bn featuremap -----")
        
        self.conv_1_1 = layer.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn_1_1 = norm_layer(64)
        self.neuron_1_1 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_1_2 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_1_2 = norm_layer(64)
        self.neuron_1_2 = spiking_neuron(**deepcopy(kwargs))
        
        self.maxpool_1 = layer.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_2_1 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_2_1 = norm_layer(128)
        self.neuron_2_1 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_2_2 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_2_2 = norm_layer(128)
        self.neuron_2_2 = spiking_neuron(**deepcopy(kwargs))
        
        self.maxpool_2 = layer.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_3_1 = layer.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn_3_1 = norm_layer(256)
        self.neuron_3_1 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_3_2 = layer.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_3_2 = norm_layer(256)
        self.neuron_3_2 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_3_3 = layer.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_3_3 = norm_layer(256)
        self.neuron_3_3 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_3_4 = layer.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_3_4 = norm_layer(256)
        self.neuron_3_4 = spiking_neuron(**deepcopy(kwargs))
        
        self.maxpool_3 = layer.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_4_1 = layer.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn_4_1 = norm_layer(512)
        self.neuron_4_1 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_4_2 = layer.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_4_2 = norm_layer(512)
        self.neuron_4_2 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_4_3 = layer.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_4_3 = norm_layer(512)
        self.neuron_4_3 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_4_4 = layer.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_4_4 = norm_layer(512)
        self.neuron_4_4 = spiking_neuron(**deepcopy(kwargs))
        
        self.maxpool_4 = layer.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_5_1 = layer.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_5_1 = norm_layer(512)
        self.neuron_5_1 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_5_2 = layer.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_5_2 = norm_layer(512)
        self.neuron_5_2 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_5_3 = layer.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_5_3 = norm_layer(512)
        self.neuron_5_3 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_5_4 = layer.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_5_4 = norm_layer(512)
        self.neuron_5_4 = spiking_neuron(**deepcopy(kwargs))
        
        self.maxpool_5 = layer.MaxPool2d(kernel_size=2, stride=2)
        
        self.avgpool = layer.AdaptiveAvgPool2d((7, 7))
        
        
        self.fc_6 = layer.Linear(512 * 7 * 7, 4096)
        self.fc_6_n = spiking_neuron(**deepcopy(kwargs))
        self.fc_6_d = layer.Dropout()
        
        self.fc_7 = layer.Linear(4096, 4096)
        self.fc_7_n = spiking_neuron(**deepcopy(kwargs))
        self.fc_7_d = layer.Dropout()
        
        self.fc_8 = layer.Linear(4096, num_classes)
        
    def forward(self, x):
        
        x01 = self.conv_1_1(x)
        x02 = self.bn_1_1(x01)
        x03 = self.neuron_1_1(x02)
        
        x04 = self.conv_1_2(x03)
        x05 = self.bn_1_2(x04)
        x06 = self.neuron_1_2(x05)
        
        x07 = self.maxpool_1(x06)
        
        x08 = self.conv_2_1(x07)
        x09 = self.bn_2_1(x08)
        x10 = self.neuron_2_1(x09)
        
        x11 = self.conv_2_2(x10)
        x12 = self.bn_2_2(x11)
        x13 = self.neuron_2_2(x12)
        
        x14 = self.maxpool_2(x13)
        
        x15 = self.conv_3_1(x14)
        x16 = self.bn_3_1(x15)
        x17 = self.neuron_3_1(x16)
        
        x18 = self.conv_3_2(x17)
        x19 = self.bn_3_2(x18)
        x20 = self.neuron_3_2(x19)
        
        x21 = self.conv_3_3(x20)
        x22 = self.bn_3_3(x21)
        x23 = self.neuron_3_3(x22)

        x24 = self.conv_3_4(x23)
        x25 = self.bn_3_4(x24)
        x26 = self.neuron_3_4(x25)

        x27 = self.maxpool_3(x26)
        
        x28 = self.conv_4_1(x27)
        x29 = self.bn_4_1(x28)
        x30 = self.neuron_4_1(x29)
        
        x31 = self.conv_4_2(x30)
        x32 = self.bn_4_2(x31)
        x33 = self.neuron_4_2(x32)
        
        x34 = self.conv_4_3(x33)
        x35 = self.bn_4_3(x34)
        x36 = self.neuron_4_3(x35)

        x37 = self.conv_4_4(x36)
        x38 = self.bn_4_4(x37)
        x39 = self.neuron_4_4(x38)
   
        x40 = self.maxpool_4(x39)
        
        x41 = self.conv_5_1(x40)
        x42 = self.bn_5_1(x41)
        x43 = self.neuron_5_1(x42)
        
        x44 = self.conv_5_2(x43)
        x45 = self.bn_5_2(x44)
        x46 = self.neuron_5_2(x45)
        
        x47 = self.conv_5_3(x46)
        x48 = self.bn_5_3(x47)
        x49 = self.neuron_5_3(x48)
  
        x50 = self.conv_5_4(x49)
        x51 = self.bn_5_4(x50)
        x52 = self.neuron_5_4(x51)
   
        x53 = self.maxpool_5(x52)
        
        x54 = self.avgpool(x53)
        if self.avgpool.step_mode == 's':
            x54 = torch.flatten(x54, 1)
        elif self.avgpool.step_mode == 'm':
            x54 = torch.flatten(x54, 2)
        #print(x54.shape)
        
        x55 = self.fc_6(x54)
        x56 = self.fc_6_n(x55)
        x57 = self.fc_6_d(x56)
        
        x58 = self.fc_7(x57)
        x59 = self.fc_7_n(x58)
        x60 = self.fc_7_d(x59)
        
        x61 = self.fc_8(x60)
                 
        feature = [
            x01, x02, x03, x04, x05, x06,
            x08, x09, x10, x11, x12, x13,
            x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26,
            x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39,
            x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, 
            x54,
            x55, x56,
            x58, x59,
            x61
            ]

        if self.mode == 'feature':
            return feature
        
        elif self.mode == 'classification':    
            return x61
        
class SpikingVGG5bn_f(nn.Module):      
    def __init__(self, mode='classification', batch_norm=False, num_classes=1000, init_weights=True,
                 spiking_neuron: callable = None, **kwargs):
        super(SpikingVGG5bn_f, self).__init__()
        
        self.mode = mode
        norm_layer = layer.BatchNorm2d
        
        print("------ creating model from SpikingVGG16bn featuremap -----")
        
        self.conv_1_1 = layer.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn_1_1 = norm_layer(64)
        self.neuron_1_1 = spiking_neuron(**deepcopy(kwargs))
        
        self.maxpool_1 = layer.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_2_1 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_2_1 = norm_layer(128)
        self.neuron_2_1 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_2_2 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_2_2 = norm_layer(128)
        self.neuron_2_2 = spiking_neuron(**deepcopy(kwargs))
        
        self.maxpool_2 = layer.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = layer.AdaptiveAvgPool2d((7, 7))
        
        self.fc_1 = layer.Linear(128 * 7 * 7, 1024)
        self.fc_1_n = spiking_neuron(**deepcopy(kwargs))
        self.fc_1_d = layer.Dropout()
        
        self.fc_2 = layer.Linear(1024, num_classes)
        
    def forward(self, x):
        
        x01 = self.conv_1_1(x)
        x02 = self.bn_1_1(x01)
        x03 = self.neuron_1_1(x02)
        
        x04 = self.maxpool_1(x03)
        
        x05 = self.conv_2_1(x04)
        x06 = self.bn_2_1(x05)
        x07 = self.neuron_2_1(x06)
        
        x08 = self.conv_2_2(x07)
        x09 = self.bn_2_2(x08)
        x10 = self.neuron_2_2(x09)
        
        x11 = self.maxpool_2(x10)
        
        x12 = self.avgpool(x11)
        
        if self.avgpool.step_mode == 's':
            x12 = torch.flatten(x12, 1)
        elif self.avgpool.step_mode == 'm':
            x12 = torch.flatten(x12, 2)
        
        x13 = self.fc_1(x12)
        x14 = self.fc_1_n(x13)
        x15 = self.fc_1_d(x14)
        
        x16 = self.fc_2(x15)
                 
        feature = [
            x01, x02, x03, 
            x05, x06, x07, x08, x09, x10, 
            x12, 
            x13, x14, 
            x16
            ]

        if self.mode == 'feature':
            return feature
        
        elif self.mode == 'classification':    
            return x16
        
class SpikingVGG5_f(nn.Module):      
    def __init__(self, mode='classification', batch_norm=False, num_classes=1000, init_weights=True,
                 spiking_neuron: callable = None, **kwargs):
        super(SpikingVGG5_f, self).__init__()
        
        self.mode = mode
        
        print("------ creating model from SpikingVGG5 featuremap -----")
        
        self.conv_1_1 = layer.Conv2d(3, 64, kernel_size=3, padding=1)
        self.neuron_1_1 = spiking_neuron(**deepcopy(kwargs))
        
        self.maxpool_1 = layer.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_2_1 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
        self.neuron_2_1 = spiking_neuron(**deepcopy(kwargs))
        
        self.conv_2_2 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
        self.neuron_2_2 = spiking_neuron(**deepcopy(kwargs))
        
        self.maxpool_2 = layer.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = layer.AdaptiveAvgPool2d((7, 7))
        
        self.fc_1 = layer.Linear(128 * 7 * 7, 1024)
        self.fc_1_n = spiking_neuron(**deepcopy(kwargs))
        self.fc_1_d = layer.Dropout()
        
        self.fc_2 = layer.Linear(1024, num_classes)
        
    def forward(self, x):
        
        x01 = self.conv_1_1(x)
        x02 = self.neuron_1_1(x01)
        
        x03 = self.maxpool_1(x02)
        
        x04 = self.conv_2_1(x03)
        x05 = self.neuron_2_1(x04)
        
        x06 = self.conv_2_2(x05)
        x07 = self.neuron_2_2(x06)
        
        x08 = self.maxpool_2(x07)
        
        x09 = self.avgpool(x08)
        
        if self.avgpool.step_mode == 's':
            x09 = torch.flatten(x09, 1)
        elif self.avgpool.step_mode == 'm':
            x09 = torch.flatten(x09, 2)
        
        x10 = self.fc_1(x09)
        x11 = self.fc_1_n(x10)
        x12 = self.fc_1_d(x11)
        
        x13 = self.fc_2(x12)
                 
        feature = [
            x01, x02, 
            x04, x05, x06, x07,
            x09, 
            x10, x11,
            x13
            ]

        if self.mode == 'feature':
            return feature
        
        elif self.mode == 'classification':    
            return x13
        
if __name__ == "__main__":
    model = SpikingVGG5_f(mode='feature', spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), num_classes=50, detach_reset=True)
    functional.set_step_mode(model, step_mode='m')
    x = torch.randn(4,1,3,224,224)
    output = model(x)
    for layer_ in output:
        print(layer_.shape)