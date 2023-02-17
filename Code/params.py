#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:58:23 2023

@author: acxyle

Purpose:
    this section provides the params load and affine
"""

def params_affine(params, verbose=False):
    
    tmp = [  "conv_1_1.weight", "conv_1_1.bias", 
           "bn_1_1.weight", "bn_1_1.bias", "bn_1_1.running_mean", "bn_1_1.running_var", "bn_1_1.num_batches_tracked",
           "conv_1_2.weight", "conv_1_2.bias", 
           "bn_1_2.weight", "bn_1_2.bias", "bn_1_2.running_mean", "bn_1_2.running_var", "bn_1_2.num_batches_tracked",
           "conv_2_1.weight", "conv_2_1.bias", 
           "bn_2_1.weight", "bn_2_1.bias", "bn_2_1.running_mean", "bn_2_1.running_var", "bn_2_1.num_batches_tracked",
           "conv_2_2.weight", "conv_2_2.bias", 
           "bn_2_2.weight", "bn_2_2.bias", "bn_2_2.running_mean", "bn_2_2.running_var", "bn_2_2.num_batches_tracked",
           "conv_3_1.weight", "conv_3_1.bias", 
           "bn_3_1.weight", "bn_3_1.bias", "bn_3_1.running_mean", "bn_3_1.running_var", "bn_3_1.num_batches_tracked",
           "conv_3_2.weight", "conv_3_2.bias", 
           "bn_3_2.weight", "bn_3_2.bias", "bn_3_2.running_mean", "bn_3_2.running_var", "bn_3_2.num_batches_tracked",
           "conv_3_3.weight", "conv_3_3.bias", 
           "bn_3_3.weight", "bn_3_3.bias", "bn_3_3.running_mean", "bn_3_3.running_var", "bn_3_3.num_batches_tracked",
           "conv_4_1.weight", "conv_4_1.bias", 
           "bn_4_1.weight", "bn_4_1.bias", "bn_4_1.running_mean", "bn_4_1.running_var", "bn_4_1.num_batches_tracked",
           "conv_4_2.weight", "conv_4_2.bias", 
           "bn_4_2.weight", "bn_4_2.bias", "bn_4_2.running_mean", "bn_4_2.running_var", "bn_4_2.num_batches_tracked",
           "conv_4_3.weight", "conv_4_3.bias", 
           "bn_4_3.weight", "bn_4_3.bias", "bn_4_3.running_mean", "bn_4_3.running_var", "bn_4_3.num_batches_tracked",
           "conv_5_1.weight", "conv_5_1.bias", 
           "bn_5_1.weight", "bn_5_1.bias", "bn_5_1.running_mean", "bn_5_1.running_var", "bn_5_1.num_batches_tracked",
           "conv_5_2.weight", "conv_5_2.bias", 
           "bn_5_2.weight", "bn_5_2.bias", "bn_5_2.running_mean", "bn_5_2.running_var", "bn_5_2.num_batches_tracked",
           "conv_5_3.weight", "conv_5_3.bias", 
           "bn_5_3.weight", "bn_5_3.bias", "bn_5_3.running_mean", "bn_5_3.running_var", "bn_5_3.num_batches_tracked",
           "fc6.weight", "fc6.bias", "fc7.weight", "fc7.bias", "fc8.weight", "fc8.bias"]

    params_replace = {}
    i = 0

    for layer in params:
      params_replace[tmp[i]] = params[layer]
      if verbose == True:
          print('moving [%s]'%(layer), 'to [%s]'%(tmp[i]))
      i += 1

    for i in params_replace:
      for j in params:
        if params[j].type_as(params_replace[i]).equal(params_replace[i]):
            if verbose == True:
                print('%s in [params] == %s in [params_replace]'%(j, i))
            break
        
    
    return params_replace