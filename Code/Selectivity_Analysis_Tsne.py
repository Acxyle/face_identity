#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:32:18 2023

@author: acxyle
"""

import torch

import os
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from tqdm import tqdm

import spiking_featuremap_utils


# *********Adjustable***********
dataSet_name = ' '
sample_num = 10
class_num = 50
# ******************************

feature_root = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Results/'
idx_root = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/'

feature_list = [(feature_root+f) for f in sorted(os.listdir(feature_root)) if f.split('.')[-1]=='pkl']
#print(feature_list)
layer_list = [
              'conv_3_3','bn_3_3', 'neuron_3_3', 
              'conv_4_3','bn_4_3', 'neuron_4_3', 
              'conv_5_3','bn_5_3', 'neuron_5_3', 
              'avgpool',
              'fc_6', 'neuron_6',
              'fc_7', 'neuron_7',
              'fc_8'
              ]

idx_list = [(idx_root+f) for f in sorted(os.listdir(idx_root)) if 'neuronIdx'  in f.split('-')[-1]]
#print(idx_list)

label = spiking_featuremap_utils.makeLabels(sample_num, class_num)


for layer in layer_list:

    with open(feature_root + '/' + layer + '.pkl', 'rb') as f:
        fullMatrix = pickle.load(f)
    save_path = idx_root + 'TSNE/' + layer
    spiking_featuremap_utils.make_dir(save_path)

    maskID = np.loadtxt(idx_root  + '/' + layer + '-neuronIdx.csv', delimiter=',')
    maskID = list(map(int, maskID))
    idx_list = []
    for i in range(fullMatrix.shape[1]):
        idx_list.append(i)
    for e in maskID:
        idx_list.remove(int(e))
    maskNonID = idx_list
    print('Length of ID/nonID mask:', len(maskID), len(maskNonID))
    perplexity_ID = min(math.sqrt(len(maskID)), 499)
    perplexity_nonID = min(math.sqrt(len(maskNonID)), 499)
    if perplexity_nonID == 0.:
        perplexity_nonID = 1e-9
    print('perplexity:', perplexity_ID, perplexity_nonID)
    
    valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if not item[1].startswith('not')])
    markers = valid_markers + valid_markers[:class_num - len(valid_markers)]
    
    if len(maskID) == 0:
        tsne_ID = TSNE(perplexity=perplexity_ID).fit_transform(torch.randn(500,1000))
    else:
        tsne_ID = TSNE(perplexity=perplexity_ID).fit_transform(fullMatrix[:, maskID])
    #print(tsne_ID.shape)

    plt.figure()
    for i in range(50): # devide into 50 classes
        plt.scatter(tsne_ID[i * sample_num: i * sample_num + sample_num, 0],
                    tsne_ID[i * sample_num: i * sample_num + sample_num, 1],
                    label[i * sample_num: i * sample_num + sample_num], marker=markers[i])

    plt.title(dataSet_name + ' ' + layer + '_ID')
    plt.text(min(tsne_ID[:,0]), 
             max(tsne_ID[:,1])+0.105*(max(tsne_ID[:,1])-min(tsne_ID[:,1])), 
             '{}/{}'.format(len(maskID), fullMatrix.shape[1]), fontsize=10, bbox={'facecolor':'yellow', 'alpha' : 0.2})
    plt.savefig(save_path + '/tsne_ID.png', bbox_inches='tight', dpi=100)
    
    if len(maskNonID) == 0:
        tsne_nonID = TSNE(perplexity=perplexity_nonID).fit_transform(torch.randn(500,1000))
    else:
        tsne_nonID = TSNE(perplexity=perplexity_nonID).fit_transform(fullMatrix[:, maskNonID])
    plt.figure()
    for i in range(50):
        plt.scatter(tsne_nonID[i * sample_num: i * sample_num + sample_num, 0],
                    tsne_nonID[i * sample_num: i * sample_num + sample_num, 1],
                    label[i * sample_num: i * sample_num + sample_num], marker=markers[i])
        
    plt.title(dataSet_name + ' ' + layer + '_nonID')
    plt.text(min(tsne_nonID[:,0]), 
             max(tsne_nonID[:,1])+0.105*(max(tsne_nonID[:,1])-min(tsne_nonID[:,1])), 
             '{}/{}'.format(len(maskNonID), fullMatrix.shape[1]), fontsize=10, bbox={'facecolor':'yellow', 'alpha' : 0.2})
    plt.savefig(save_path + '/tsne_nonID.png', bbox_inches='tight', dpi=100)
    
    

        