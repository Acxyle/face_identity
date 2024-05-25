#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 02:19:13 2024

@author: acxyle-workstation
"""

import os
import warnings
import logging
import numpy as np
import itertools

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from joblib import Parallel, delayed
import scipy

from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind
from statsmodels.stats.multitest import multipletests


from FSA_DRG import FSA_DSM

import sys
sys.path.append('../')
import utils_
from utils_ import utils_similarity

from FSA_RSA import RSA_Human_folds, plot_RSA


# -----
FSA_root = '/home/acxyle-workstation/Downloads/FSA'
FSA_dir = 'VGG/SpikingVGG'
model_depth = 152
FSA_config = f'SpikingVGG16bn_IF_ATan_T4_C2k_fold_'
FSA_model =  f'spiking_vgg16_bn'

_, layers, neurons, _ = utils_.get_layers_and_units(FSA_model, 'act')

root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')

RSA_human_f = RSA_Human_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
used_unit_types = ['qualified', 'sensitive', 'non_sensitive', 'strong_selective']

snn_RSA_dict = {}
for used_unit_type in used_unit_types:
    snn_RSA_dict[used_unit_type] = RSA_human_f.collect_RSA_Similarity_folds(used_unit_type=used_unit_type, used_id_num=50)

snn_RSA_dict_static = {k: [_['similarity'] for f,_ in v.items()] for k,v in snn_RSA_dict.items()}

snn_RSA_dict_static = {
'mean': {k: np.nanmean(v, axis=0) for k,v in snn_RSA_dict_static.items()}, 
'std': {k: np.nanstd(v, axis=0) for k,v in snn_RSA_dict_static.items()}
}


# -----
FSA_root = '/home/acxyle-workstation/Downloads/FSA'
FSA_dir = 'VGG/VGG'
FSA_config = f'VGG16bn_C2k_fold_'
FSA_model =  f'vgg16_bn'

_, layers, neurons, _ = utils_.get_layers_and_units(FSA_model, 'act')

root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
RSA_human_f = RSA_Human_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
ann_RSA_dict = {}
for used_unit_type in used_unit_types:
    ann_RSA_dict[used_unit_type] = RSA_human_f.collect_RSA_Similarity_folds(used_unit_type=used_unit_type, used_id_num=50)

ann_RSA_dict_static = {k: [_['similarity'] for f,_ in v.items()] for k,v in ann_RSA_dict.items()}

ann_RSA_dict_static = {
'mean': {k: np.nanmean(v, axis=0) for k,v in ann_RSA_dict_static.items()}, 
'std': {k: np.nanstd(v, axis=0) for k,v in ann_RSA_dict_static.items()}
}


# -----
fig, ax = plt.subplots(1,2,figsize=(12, 4))

for used_unit_type in used_unit_types:
    
    similarity = np.nan_to_num(ann_RSA_dict_static['mean'][used_unit_type])
    std = np.nan_to_num(ann_RSA_dict_static['std'][used_unit_type])
    color = RSA_human_f.plot_Encode_config.loc[used_unit_type]['color']
    label = RSA_human_f.plot_Encode_config.loc[used_unit_type]['label']
    
    plot_RSA(ax[0], similarity, similarity_std=std, color=color, label=label, smooth=True)
    
    ax[0].set_ylim([-0.05, 0.25])
    ax[0].set_title('VGG')
    ax[0].hlines(0, 0, len(layers)-1, color='gray', linestyle='--', alpha=0.25)
    
    ax[0].set_xticks([0, 14])
    ax[0].set_xticklabels([0, 1])

    similarity = np.nan_to_num(snn_RSA_dict_static['mean'][used_unit_type])
    std = np.nan_to_num(snn_RSA_dict_static['std'][used_unit_type])
    color = RSA_human_f.plot_Encode_config.loc[used_unit_type]['color']
    label = RSA_human_f.plot_Encode_config.loc[used_unit_type]['label']

    plot_RSA(ax[1], similarity, similarity_std=std, color=color, label=label, smooth=True)

    ax[1].set_ylim([-0.05, 0.25])
    ax[1].set_title('SpikingVGG')
    ax[1].hlines(0, 0, len(layers)-1, color='gray', linestyle='--', alpha=0.25)
    
    ax[1].set_xticks([0, 14])
    ax[1].set_xticklabels([0, 1])

legend_lines = [Line2D([0], [0], color='black', lw=4),  
                Line2D([0], [0], color='yellow', lw=4),
                Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='green', lw=4)]

fig.legend(legend_lines, ['ALL', 'Selective', 'Sensitive', 'NS'], loc='lower center', bbox_to_anchor=(0.5, -0.125), fancybox=True, shadow=True, ncol=4)

fig.savefig('RSA_types_layers_VGG_vs_SpikingVGG.svg', bbox_inches='tight')