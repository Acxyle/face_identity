#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:41:07 2024

@author: acxyle-workstation
"""

import os


def rename(root = './Features'):
    
    rename_map = {
    'conv01': 'conv_1',
    'bn01': 'bn_1',
    'act01': 'neuron_1',
    'conv02': 'conv_2',
    'bn02': 'bn_2',
    'act02': 'neuron_2',
    'conv03': 'conv_3',
    'bn03': 'bn_3',
    'act03': 'neuron_3',
    }
    
    files = os.listdir(root)
    
    for file in files:
        for key in rename_map:
            if key in file:
                new_name = os.path.join(f"{'_'.join(file.split('_')[:-1])}_{rename_map[key]}.pkl")
                os.rename(os.path.join(root, file), os.path.join(root, new_name))
                break  


def params_affine_from_spikingjelly04(params):

    print('experiment details:', params['args'])
    print('params.keys():', params.keys())
    print('best_val_acc1: {:.3f}, best_val_acc5: {:.3f}'.format(params['max_test_acc1'], params['test_acc5_at_max_test_acc1']))
        
    # different name space for spikingjelly.0.4 and new versions
    params_replace = {}
    replace_map = {
        'conv1.module.0': 'conv1',
        'conv1.module.1': 'bn1',
        'conv2.module.0': 'conv2',
        'conv2.module.1': 'bn2',
        'conv3.module.0': 'conv3',
        'conv3.module.1': 'bn3',
        'downsample.0.module': 'downsample',  # for sew_resnet
        'downsample.module': 'downsample'     # for spiking_resnet
    }
    
    for layer in params['model']:
        new_layer = layer
        for old, new in replace_map.items():
            if old in layer:
                parts = layer.split(old)
                new_layer = new.join(parts)
                break  # Once a replacement is made, no need to check further
        params_replace[new_layer] = params['model'][layer]

    return params_replace