#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:53:33 2023

@author: acxyle
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def color_column(layers):
    """
    distinguish how many types in layers and return a list of color as the length of layers
    """
    layers_t = []
    for layer in layers:
        layers_t.append(layer.split('_')[0])
    layers_t = list(set(layers_t))
    print(layers_t)
    color = ['teal',  'red','orange', 'lightskyblue', 'tomato']
    layers_c_dict = {}
    for i in range(len(layers_t)):
        layers_c_dict[layers_t[i]] = color[i]
    print(layers_c_dict)
        
    layers_color_list = []
    
    for layer in layers:
        layers_color_list.append(layers_c_dict[layer.split('_')[0]])
    
    print(layers_color_list)
    
    return layers_color_list
    
# plot
root = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/'

alpha = 0.01
layers = [     'conv_1_1','bn_1_1', 'neuron_1_1', 'conv_1_2', 'bn_1_2', 'neuron_1_2',
               'conv_2_1','bn_2_1', 'neuron_2_1', 'conv_2_2', 'bn_2_2', 'neuron_2_2',
               'conv_3_1','bn_3_1', 'neuron_3_1', 'conv_3_2','bn_3_2', 'neuron_3_2', 'conv_3_3','bn_3_3', 'neuron_3_3', 
               'conv_4_1','bn_4_1', 'neuron_4_1', 'conv_4_2','bn_4_2', 'neuron_4_2', 'conv_4_3','bn_4_3', 'neuron_4_3', 
               'conv_5_1','bn_5_1', 'neuron_5_1', 'conv_5_2','bn_5_2', 'neuron_5_2', 'conv_5_3','bn_5_3', 'neuron_5_3', 
               'avgpool',
               'fc_6', 'neuron_6',
               'fc_7', 'neuron_7',
               'fc_8']

dim_list = [64*224*224,64*224*224,64*224*224,64*224*224,64*224*224,64*224*224,
       128*112*112,128*112*112,128*112*112,128*112*112,128*112*112,128*112*112,
       256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,
       512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,
       512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,
       25088,
       4096, 4096,
       4096, 4096,
       50]

count_dict = {}
for layer in layers:    # 建立layer和 selective neuron percent 的 dict
  count_dict[layer] = 0

for f in sorted(os.listdir(root)):

  if f.split('.')[0].split('-')[-1] == 'pvalue':  # 对于每一个pvalue文件

      plist_path = root + f
      print(plist_path)
      key = plist_path.split('/')[-1].split('.')[0].split('-')[0]
      pl = np.loadtxt(plist_path, delimiter=',')

      sig_neuron_ind = [ind for ind, p in enumerate(pl) if p < alpha]
      value = len(sig_neuron_ind)
      count_dict[key]=value


      
ratio = [round(a / b * 100) for a, b in zip(list(count_dict.values()), dim_list)]
print(ratio)
layers_color_list = color_column(layers)

plt.clf()    
plt.figure(figsize=(30,10), dpi=200)
plt.bar(layers, ratio, color=layers_color_list, width=0.5)
plt.xticks(rotation=45)
plt.ylabel('percentage')
plt.title('selective neuron ratio for each layer')
plt.savefig('selective_neuron_percent.png')

plt.show()

