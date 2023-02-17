#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 23:47:42 2023

@author: acxyle
"""

import os
import pickle
import numpy as np
from scipy.io import savemat
from scipy.spatial.distance import pdist, squareform
import seaborn as sn
import matplotlib.pyplot as plt

import spiking_featuremap_utils

def avg_acrossID(matrix, sample_num, class_num):
    col = matrix.shape[1]
    avg_full = np.empty((class_num, col))       # (50, neurons)
    for i in range(class_num):
        subMat = matrix[i * sample_num: i * sample_num + sample_num, :]
        avg_sub = subMat.mean(axis=0)  # to take the mean of each col
        avg_full[i, :] = avg_sub    # (50, neurons)
    return avg_full


# *********Adjustable***********
sample_num = 10
class_num = 50
# ******************************

feature_root = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Results/'
idx_root = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/'
dest = idx_root+'Distance/'
spiking_featuremap_utils.make_dir(dest)

layer_list = [
    'conv_3_3','bn_3_3', 'neuron_3_3', 
    'conv_4_3','bn_4_3', 'neuron_4_3', 
    'conv_5_3','bn_5_3', 'neuron_5_3', 
    'avgpool',
    'fc_6', 'neuron_6',
    'fc_7', 'neuron_7',
    'fc_8'
]


for idx, layer in enumerate(layer_list): # each layer

    with open(feature_root + layer + '.pkl', 'rb') as f:
        fullMatrix = pickle.load(f)
    # Avg: 50*col
    avgMatrix = avg_acrossID(fullMatrix, sample_num, class_num)
    # Mask: ID, nonID
    maskID = np.loadtxt(idx_root  + '/' + layer + '-neuronIdx.csv', delimiter=',')
    maskID = list(map(int, maskID))
    idx_list = []
    for i in range(fullMatrix.shape[1]):
        idx_list.append(i)
    for e in maskID:
        idx_list.remove(int(e))
    maskNonID = idx_list
    print('Length of ID/nonID mask:', len(maskID), len(maskNonID))

    fig, axes = plt.subplots(1, 3, figsize=((30, 10)))
    cbar_ax = fig.add_axes([.91, .1, .03, .8])

    dist_avg = pdist(avgMatrix, 'euclidean')
    m = squareform(dist_avg)
    
    dist_avg_ID = pdist(avgMatrix[:, maskID], 'euclidean')
    m_i = squareform(dist_avg_ID)
    
    dist_avg_NonID = pdist(avgMatrix[:, maskNonID], 'euclidean')
    m_n = squareform(dist_avg_NonID)

    vmax = max(m.max(), m_i.max(), m_n.max())
    vmin = min(m.min(), m_i.min(), m_n.min())

    sn.heatmap(m, ax=axes[0], cbar=0, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
    axes[0].set_title('all neurons')
    sn.heatmap(m_i, ax=axes[1], cbar=1, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
    axes[1].set_title('identity selective neurons')
    sn.heatmap(m_n, ax = axes[2], cbar=2, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
    axes[2].set_title('non identity selective neurons')

    fig.tight_layout(rect=[0, 0, .9, 1])
    
    plt.savefig(dest + layer+'-Corrolation.png', bbox_inches='tight', dpi=100)
    plt.title(layer)
    #plt.show()
    plt.clf()


