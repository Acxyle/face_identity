#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:39:36 2024

@author: acxyle-workstation
"""

import torch
#import torchvision

from spikingjelly.activation_based import neuron, functional

import sys
sys.path.append('../')
import models_


# ----------------------------------------------------------------------------------------------------------------------
def get_generator(model_name:str, **kwargs):
    
    generators = {
        'vgg': VGG_layers_generator,
        'resnet': Resnet_layers_generator,
                 }
    
    for key in generators:
        if key in model_name:
            return generators[key]
    

def get_layers_and_units(model_name:str, target_type:str='all', **kwargs):
    """
        ...
    """
    model_name = model_name.lower()
    
    # ---
    generator = get_generator(model_name, **kwargs)
    assert generator is not None
    
    layers, units, shapes = generator(model_name, **kwargs)
        
    return target_layers_selection(model_name, layers, units, shapes, target_type=target_type, **kwargs)
        
    
# ----------------------------------------------------------------------------------------------------------------------
def target_layers_selection(model_name, layers:list[str], units:list[int], shapes:list[tuple], target_type:str='all', **kwargs):
    """
        all: all operations
        act: (spiking) activation function
        feature: manually selected layers (layers are frequently used for feature analysis)
    """
    # ----- init
    act = 'sn' if ('spiking' in model_name or 'sew' in model_name) else 'an'
    
    # ----- determine layers
    if target_type == 'all':
        _layers = layers
        
    elif target_type == 'act':
        _layers = [_ for _ in layers if f'{act}' in _]
        
    elif target_type == 'feature':
        target_layers = [f'{act}', 'Pool']
        if 'vgg' in model_name:
            target_layers += ['fc_3']
        elif 'resnet' in model_name:
            target_layers += ['fc']
            
        _layers = [_ for _ in layers if any(target in _ for target in target_layers)]
        
    else:
        raise ValueError(f'Invalid model name {model_name}')
        
    idx = [layers.index(_) for _ in _layers]
    units = [units[_] for _ in idx]
    shapes = [shapes[_] for _ in idx]
    
    return idx, _layers, units, shapes
    

# ----------------------------------------------------------------------------------------------------------------------
def VGG_layers_generator(model_name, num_classes=50, feature_shape=(3,224,224), **kwargs):
    """
        this script mimics the construction of VGG to genearte the layer names
    """

    # assume the SNN vgg has identical feature map shape with ANN vgg
    if 'spiking' in model_name or 'sew' in model_name:
        model_name = model_name.split('_', 1)[-1]
        act = 'sn'
    else:
        act = 'an'

    # --- config init
    cfgs = {
        'vgg5': [64, 'M', 128, 128, 'M'],
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
           }

    # --- 0. init
    basic_block = ['Conv', 'BN', f'{act}'] if 'bn' in model_name else ['Conv', f'{act}']
    layers = [] 
    l_idx = 1
    b_idx = 1
    
    # --- 1. features
    for v in cfgs[model_name.split('_')[0]]:     # for each block
        if v == 'M':
            layers += [f'L{l_idx}_MaxPool']
            b_idx = 1
            l_idx += 1
        else:
            layers += [f'L{l_idx}_B{b_idx}_{_}' for _ in basic_block] 
            b_idx += 1
    layers += ['AvgPool']
    
    # --- 2. classifier
    layers += ['fc_1', f'{act}_1', 'dp_1', 'fc_2'] if '5' in model_name else ['fc_1', f'{act}_1', 'dp_1', 'fc_2', f'{act}_2', 'dp_2', 'fc_3']
    
    # --- 3. dummy img, model and forward
    x = torch.zeros(1, *feature_shape)
    model = models_.vgg.__dict__[model_name](num_classes=num_classes)
    
    features = model(x)
    assert isinstance(features, list)
    
    functional.reset_net(model)     # SNN clean memory
    
    # --- 4. collect info
    units = []
    shapes = []
    
    for idx, _ in enumerate(features):
        units.append(_.mean(0).detach().cpu().numel())
        shapes.append(_.mean(0).squeeze(0).detach().cpu().numpy().shape)
            
    return layers, units, shapes


# ----------------------------------------------------------------------------------------------------------------------
def Resnet_layers_generator(model_name, T=4, num_classes=50, feature_shape=(3,224,224), **kwargs):     # return layers and units
    """
        ...
    """
    
    base_configs = {
        'resnet18': [2, 2, 2, 2], 
        'resnet34': [3, 4, 6, 3], 
        'resnet50': [3, 4, 6, 3], 
        'resnet101': [3, 4, 23, 3], 
        'resnet152': [3, 8, 36, 3]
    }

    model_dict = {}
    for prefix in ['', 'spiking_', 'sew_']:
        for k, v in base_configs.items():
            model_dict[prefix+k] = v.copy()
            
    # -----
    act = 'sn' if ('spiking' in model_name or 'sew' in model_name) else 'an'
    
    layers = ['Conv_0', 'BN_0', f'{act}_0', 'MaxPool_0']
    
    if 'sew' in model_name:
        bottleneck = ['Conv_1', 'BN_1', f'{act}_1', 'Conv_2', 'BN_2', f'{act}_2', 'Conv_3', 'BN_3', f'{act}_3', 'Residual']
        basicblock = ['Conv_1', 'BN_1', f'{act}_1', 'Conv_2', 'BN_2', f'{act}_2', 'Residual']
    else:
        bottleneck = ['Conv_1', 'BN_1', f'{act}_1', 'Conv_2', 'BN_2', f'{act}_2', 'Conv_3', 'BN_3', 'Residual', f'{act}_3']
        basicblock = ['Conv_1', 'BN_1', f'{act}_1', 'Conv_2', 'BN_2', 'Residual', f'{act}_2']
    
    # ---
    target_blocks = basicblock if '18' in model_name or '34' in model_name else bottleneck
    for l_idx, blocks in enumerate(model_dict[model_name], start=1):   # each layer
        for b_idx in range(blocks):          # each block
            layers += [f'L{l_idx}_B{b_idx+1}_{_}' for _ in target_blocks]
    layers += ['AvgPool', 'fc']     # manually change to fit different model
    
    # --- dummy img, model and forward
    x = torch.zeros(T, 1, *feature_shape) if ('spiking' in model_name or 'sew' in model_name) else torch.zeros(1, *feature_shape)
 
    if model_name in models_.spiking_resnet.__all__:
        model = models_.spiking_resnet.__dict__[model_name](spiking_neuron=neuron.IFNode, num_classes=num_classes)
        functional.set_step_mode(model, step_mode='m')
    elif model_name in models_.sew_resnet.__all__:
        model = models_.sew_resnet.__dict__[model_name](spiking_neuron=neuron.IFNode, num_classes=num_classes, cnf='ADD')
        functional.set_step_mode(model, step_mode='m')
    elif model_name in models_.resnet.__all__:
        model = models_.resnet.__dict__[model_name](num_classes=num_classes)
    else:
        raise ValueError

    features = model(x)
    assert isinstance(features, list)
    
    functional.reset_net(model)
    
    # -----
    units = []
    shapes = []
    for l_idx, _ in enumerate(features):
        units.append(_.mean(0).numel())
        shapes.append(_.mean(0).squeeze(0).detach().cpu().numpy().shape)
    
    return layers, units, shapes

