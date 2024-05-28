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
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ['ResNet', 
           "resnet18",
           "resnet34",
           "resnet50",
           "resnet101",
           "resnet152",
           
           "resnext50_32x4d",
           "resnext50_32x8d",
           "resnext50_32x16d",
           "resnext50_32x32d",
           ]

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x01 = self.conv1(x)
        x02 = self.bn1(x01)
        x03 = self.relu(x02)

        x04 = self.conv2(x03)
        x05 = self.bn2(x04)

        if self.downsample is not None:
            identity = self.downsample(x)

        x06 = x05 + identity
        x07 = self.relu(x06)

        return [x01, x02, x03, x04, x05, x06, x07]


class Bottleneck(nn.Module):
    
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x01 = self.conv1(x)
        x02 = self.bn1(x01)
        x03 = self.relu(x02)

        x04 = self.conv2(x03)
        x05 = self.bn2(x04)
        x06 = self.relu(x05)

        x07 = self.conv3(x06)
        x08 = self.bn3(x07)

        if self.downsample is not None:
            identity = self.downsample(x)

        x09 = x08 + identity
        x10 = self.relu(x09)

        return [x01, x02, x03, x04, x05, x06, x07, x08, x09, x10]


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
       x01_ = self.conv1(x)
       x02_ = self.bn1(x01_)
       x03_ = self.relu(x02_)
       x04_ = self.maxpool(x03_)

       x_list1 = self.layer1(x04_)
       x_list2 = self.layer2(x_list1)
       x_list3 = self.layer3(x_list2)
       x_list4 = self.layer4(x_list3)

       x05_ = self.avgpool(x_list4[-1])
       
       x05_ = torch.flatten(x05_, 1)
       
       x06_ = self.fc(x05_)

       return [x01_, x02_, x03_, x04_, *x_list4, x05_, nn.ReLU()(x06_)]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
  
def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    
    model = ResNet(block, layers, **kwargs)

    return model

def resnet18(*, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2], progress, **kwargs)


def resnet34(*, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [3, 4, 6, 3], progress, **kwargs)


def resnet50(*, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3], progress, **kwargs)


def resnet101(*, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3], progress, **kwargs)


def resnet152(*, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 8, 36, 3], progress, **kwargs)


def resnext50_32x4d(*, progress: bool = True, **kwargs: Any) -> ResNet:
 
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3], progress, **kwargs)


def resnext50_32x8d(*, progress: bool = True, **kwargs: Any) -> ResNet:
  
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(Bottleneck, [3, 4, 6, 3], progress, **kwargs)


def resnext50_32x16d(*, progress: bool = True, **kwargs: Any) -> ResNet:
  
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    return _resnet(Bottleneck, [3, 4, 6, 3], progress, **kwargs)


def resnext50_32x32d(*, progress: bool = True, **kwargs: Any) -> ResNet:
  
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 32
    return _resnet(Bottleneck, [3, 4, 6, 3], progress, **kwargs)


# =============================================================================
# if __name__ == '__main__':
#     
#     model = resnext50_32x4d()
#     x = torch.randn(1, 3, 224, 224)
#     out1 = model(x)
#     
#     model = resnext50_32x32d()
#     x = torch.randn(1, 3, 224, 224)
#     out2 = model(x)
#     
#     for _ in range(165):
#         print(_, out1[_].numel(), out2[_].numel())
# =============================================================================
