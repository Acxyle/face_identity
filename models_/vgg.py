#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 15:12:39 2023

@author: acxyle
    
    rewrite from https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
   
    this model is used for feature extraction as the forward process has been changed. 
    
    this model should be used with the feature extraction with modified source code of Sequential().
    
    DO NOT use this code if use (1) module, (2) hook or (3) FX to extract the feature map.
   
"""

import torch
import torch.nn as nn


__all__ = [
    "VGG",
    "vgg5", "vgg5_bn",
    "vgg11", "vgg11_bn",
    "vgg13", "vgg13_bn",
    "vgg16", "vgg16_bn",
    "vgg19", "vgg19_bn",
]


class VGG(nn.Module):
    
    def __init__(self, features:nn.Module, num_classes:int=1000, init_weights:bool=True, dropout:float=0.5) -> None:
        
        super().__init__()
        
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        if len(features) == 8 or len(features) == 11:
            
            self.classifier = nn.Sequential(
                                            nn.Linear(128 * 7 * 7, 1024),
                                            nn.ReLU(True),
                                            nn.Dropout(p=dropout),
                                            nn.Linear(1024, num_classes),
                                            )

        else:
        
            self.classifier = nn.Sequential(
                                            nn.Linear(512 * 7 * 7, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout(p=dropout),
                                            nn.Linear(4096, 4096),
                                            nn.ReLU(True),
                                            nn.Dropout(p=dropout),
                                            nn.Linear(4096, num_classes),
                                            )
        
        
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        x_list = self.features([x])
        x01 = self.avgpool(x_list[-1])
        x01 = torch.flatten(x01, 1)
        x_list2 = self.classifier([x01])
        
        #x02 =  nn.ReLU()(x_list2[-1])
        x02 = nn.Softmax(dim=-1)(x_list2[-1])     
        
        feature = [*x_list[1:], x01, *x_list2[1:-1], nn.ReLU()(x02)]
        
        return feature

def make_layers(cfg, batch_norm:bool=False) -> nn.Sequential:
    
    layers = []
    in_channels = 3
    
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    return nn.Sequential(*layers)


cfgs = {
    'O': [64, 'M', 128, 128, 'M'],
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg:str, batch_norm:bool, **kwargs):

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    return model


def vgg5(**kwargs):
    return _vgg("O", False, **kwargs)

def vgg5_bn(**kwargs):
    return _vgg("O", True, **kwargs)

def vgg11(**kwargs):
    return _vgg("A", False, **kwargs)

def vgg11_bn(**kwargs):
    return _vgg("A", True, **kwargs)

def vgg13(**kwargs):
    return _vgg("B", False, **kwargs)

def vgg13_bn(**kwargs):
    return _vgg("B", True, **kwargs)

def vgg16(**kwargs):
    return _vgg("D", False, **kwargs)

def vgg16_bn(**kwargs):
    return _vgg("D", True, **kwargs)

def vgg19(**kwargs):
    return _vgg("E", False, **kwargs)

def vgg19_bn(**kwargs):
    return _vgg("E", True, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# debug
if __name__ == '__main__':
    model = vgg5_bn()
    #print(model)
    x = torch.randn(1,3,224,224)
    out = model(x)

    for idx, layer in enumerate(out):
        print(idx, layer.shape, layer.numel())