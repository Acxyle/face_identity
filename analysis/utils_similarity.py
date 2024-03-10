#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 23:31:53 2023

@author: acxyle-workstation

    Warning:
        the entire code needs to be simplfied and upgrade
        the code may not work in some functions because it diretly copy from ipython terminal, please check the dependencies 
    before use it
    
    [task]:
        
        simplify the entire code

"""

import os
import pandas as pd

import scipy.stats as stats
import warnings
import logging
import numpy as np
import scipy.io as sio
import scipy

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from collections import Counter
from joblib import Parallel, delayed
from matplotlib import gridspec
from scipy.stats import gaussian_kde, norm, skew, lognorm, kstest
from scipy.spatial.distance import pdist, squareform

from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind
import itertools
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv

from scipy.integrate import quad

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import utils_
import math

from statsmodels.stats.multitest import multipletests
from scipy.interpolate import RectBivariateSpline
import matplotlib

def multi_model_macaque_style_comparison_human():
    """
        [warning] the data is incomplete
    
        task: 
            1. construct 
                (a) the new averaged feature map ✓
                (b) p_value map
            2. interpolation
            3. merge
            4. 4 types of cells
            
        note:
            1. add auto determination for dict and feature dict
            
    """
    root = '/home/acxyle-workstation/Downloads'
    cell_types = ['qualified', 'legacy', 'selective']
    target_ids = 10
    
    subplotparams = matplotlib.figure.SubplotParams(left=0, right=0.9, bottom=0, top=1, wspace=0.05, hspace=0.05)
    fig, ax = plt.subplots(2,4, figsize=(20,10), subplotpars=subplotparams)
 
    # ----- ANN
    # --- resnet [pending: resnet101_folds, resnet152_folds]
    resnet18_folds = {_:utils_.load(os.path.join(root, f'Face Identity Resnet18_CelebA2622_fold_/RSA/Human/pearson/RSA_static_dict_Human_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    resnet50_folds = {_:utils_.load(os.path.join(root, f'Face Identity Resnet50_CelebA2622_fold_/RSA/Human/pearson/RSA_static_dict_Human_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    
    resnet101 = {_:utils_.load(os.path.join(root, f'Face Identity Resnet101_CelebA2622_fold_0/Analysis/RSA/Human/pearson/{_}/{target_ids}/RSA_results_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    resnet152 = {_:utils_.load(os.path.join(root, f'Face Identity Resnet152_CelebA2622_fold_0/Analysis/RSA/Human/pearson/{_}/{target_ids}/RSA_results_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    
    # --- vgg [pending: vgg5, vgg11, vgg19]
    vgg16_folds = {_:utils_.load(os.path.join(root, f'Face Identity VGG16_fold_/RSA/Human/pearson/RSA_static_dict_Human_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    vgg16bn_folds = {_:utils_.load(os.path.join(root, f'Face Identity VGG16bn_fold_/RSA/Human/pearson/RSA_static_dict_Human_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    
    # ---
    interpolated_temporal_dict = {_: np.array([
                                      _interp_2d(resnet101[_]['similarity_temporal']), 
                                      _interp_2d(resnet152[_]['similarity_temporal']),
                                      *[_interp_2d(resnet18_folds[_][fold_idx]['similarity_temporal']) for fold_idx in range(5)],
                                      *[_interp_2d(resnet50_folds[_][fold_idx]['similarity_temporal']) for fold_idx in range(5)],
                                      *[_interp_2d(vgg16_folds[_][fold_idx]['similarity_temporal']) for fold_idx in range(5)],
                                      *[_interp_2d(vgg16bn_folds[_][fold_idx]['similarity_temporal']) for fold_idx in range(5)],
                                      ]) for _ in cell_types}
    
    interpolated_temporal_dict['non_selective'] = np.array([
                                              _interp_2d(np.nan_to_num(resnet101['non_selective']['similarity_temporal'])), 
                                              _interp_2d(np.nan_to_num(resnet152['non_selective']['similarity_temporal'])),
                                              *[_interp_2d(np.nan_to_num(resnet18_folds['non_selective'][fold_idx]['similarity_temporal'])) for fold_idx in range(5)],
                                              *[_interp_2d(np.nan_to_num(resnet50_folds['non_selective'][fold_idx]['similarity_temporal'])) for fold_idx in range(5)],
                                              *[_interp_2d(np.nan_to_num(vgg16_folds['non_selective'][fold_idx]['similarity_temporal'])) for fold_idx in range(5)],
                                              *[_interp_2d(np.nan_to_num(vgg16bn_folds['non_selective'][fold_idx]['similarity_temporal'])) for fold_idx in range(5)],
                                              ])
    
    
    interpolated_temporal_p_dict = {_: np.array([
                                      _interp_2d(resnet101[_]['similarity_p_perm_temporal']), 
                                      _interp_2d(resnet152[_]['similarity_p_perm_temporal']),
                                      *[_interp_2d(resnet18_folds[_][fold_idx]['similarity_p_perm_temporal']) for fold_idx in range(5)],
                                      *[_interp_2d(resnet50_folds[_][fold_idx]['similarity_p_perm_temporal']) for fold_idx in range(5)],
                                      *[_interp_2d(vgg16_folds[_][fold_idx]['similarity_p_perm_temporal']) for fold_idx in range(5)],
                                      *[_interp_2d(vgg16bn_folds[_][fold_idx]['similarity_p_perm_temporal']) for fold_idx in range(5)],
                                      ]) for _ in cell_types}
    
    interpolated_temporal_p_dict['non_selective'] = np.array([
                                              _interp_2d(np.nan_to_num(resnet101['non_selective']['similarity_p_perm_temporal'], 1)), 
                                              _interp_2d(np.nan_to_num(resnet152['non_selective']['similarity_p_perm_temporal'], 1)),
                                              *[_interp_2d(np.nan_to_num(resnet18_folds['non_selective'][fold_idx]['similarity_p_perm_temporal'], 1)) for fold_idx in range(5)],
                                              *[_interp_2d(np.nan_to_num(resnet50_folds['non_selective'][fold_idx]['similarity_p_perm_temporal'], 1)) for fold_idx in range(5)],
                                              *[_interp_2d(np.nan_to_num(vgg16_folds['non_selective'][fold_idx]['similarity_p_perm_temporal'], 1)) for fold_idx in range(5)],
                                              *[_interp_2d(np.nan_to_num(vgg16bn_folds['non_selective'][fold_idx]['similarity_p_perm_temporal'], 1)) for fold_idx in range(5)],
                                              ])
    
    
    # ---
    ax[0, 0].imshow(np.array([np.mean(interpolated_temporal_dict['qualified'], axis=0)[_*10, :] for _ in range(100)]), aspect='auto', vmin=-0.3, vmax=0.8, cmap='turbo')
    ax[0, 0].axis('off')
    ax[0, 1].imshow(np.array([np.mean(interpolated_temporal_dict['legacy'], axis=0)[_*10, :] for _ in range(100)]), aspect='auto', vmin=-0.3, vmax=0.8, cmap='turbo')
    ax[0, 1].axis('off')
    ax[0, 2].imshow(np.array([np.mean(interpolated_temporal_dict['selective'], axis=0)[_*10, :] for _ in range(100)]), aspect='auto', vmin=-0.3, vmax=0.8, cmap='turbo')
    ax[0, 2].axis('off')
    ax[0, 3].imshow(np.array([np.mean(interpolated_temporal_dict['non_selective'], axis=0)[_*10, :] for _ in range(100)]), aspect='auto', vmin=-0.3, vmax=0.8, cmap='turbo')
    ax[0, 3].axis('off')
    
    # ---
    temporal_dict = {_: {'resnet101': resnet101[_]['similarity_temporal'][resnet101[_]['sig_temporal_FDR'].astype(bool)],
                         'resnet152': resnet152[_]['similarity_temporal'][resnet152[_]['sig_temporal_FDR'].astype(bool)],
                         
                         'resnet18_folds': np.concatenate([resnet18_folds[_][fold_idx]['similarity_temporal'][resnet18_folds[_][fold_idx]['sig_temporal_FDR'].astype(bool)] for fold_idx in range(5)]),
                         'resnet50_folds': np.concatenate([resnet50_folds[_][fold_idx]['similarity_temporal'][resnet50_folds[_][fold_idx]['sig_temporal_FDR'].astype(bool)] for fold_idx in range(5)]),
                         'vgg16_folds': np.concatenate([vgg16_folds[_][fold_idx]['similarity_temporal'][vgg16_folds[_][fold_idx]['sig_temporal_FDR'].astype(bool)] for fold_idx in range(5)]),
                         'vgg16bn_folds': np.concatenate([vgg16bn_folds[_][fold_idx]['similarity_temporal'][vgg16bn_folds[_][fold_idx]['sig_temporal_FDR'].astype(bool)] for fold_idx in range(5)]),
                         } for _ in cell_types}
    
    temporal_dict['non_selective'] = {'resnet101': np.nan_to_num(resnet101['non_selective']['similarity_temporal'][resnet101['non_selective']['sig_temporal_FDR'].astype(bool)]),
                                     'resnet152': np.nan_to_num(resnet152['non_selective']['similarity_temporal'][resnet152['non_selective']['sig_temporal_FDR'].astype(bool)]),
                                     
                                     'resnet18_folds': np.concatenate([np.nan_to_num(resnet18_folds['non_selective'][fold_idx]['similarity_temporal'][resnet18_folds['non_selective'][fold_idx]['sig_temporal_FDR'].astype(bool)]) for fold_idx in range(5)]),
                                     'resnet50_folds': np.concatenate([np.nan_to_num(resnet50_folds['non_selective'][fold_idx]['similarity_temporal'][resnet50_folds['non_selective'][fold_idx]['sig_temporal_FDR'].astype(bool)]) for fold_idx in range(5)]),
                                     'vgg16_folds': np.concatenate([np.nan_to_num(vgg16_folds['non_selective'][fold_idx]['similarity_temporal'][vgg16_folds['non_selective'][fold_idx]['sig_temporal_FDR'].astype(bool)]) for fold_idx in range(5)]),
                                     'vgg16bn_folds': np.concatenate([np.nan_to_num(vgg16bn_folds['non_selective'][fold_idx]['similarity_temporal'][vgg16bn_folds['non_selective'][fold_idx]['sig_temporal_FDR'].astype(bool)]) for fold_idx in range(5)]),
                                        }

    
    # ---
    for idx, _ in enumerate(['qualified', 'legacy', 'selective', 'non_selective']):
        
        inter_p = np.array([np.mean(interpolated_temporal_p_dict[_], axis=0)[i*10, :] for i in range(100)])
        
        interp_p = (inter_p<0.05).astype(float)
        interp_p = scipy.ndimage.gaussian_filter(interp_p, sigma=2)
        
        ax[0, idx].contour(interp_p, levels:=[0.3], cmap='jet', linewidths=3)
    
    # ----- SNN
    # --- spiking_vgg
    svgg16bn_if_4_folds = {_:utils_.load(os.path.join(root, f'Face Identity SpikingVGG16bn_IF_T4_CelebA2622_fold_/RSA/Human/pearson/RSA_static_dict_Human_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    svgg16bn_lif_4_folds = {_:utils_.load(os.path.join(root, f'Face Identity SpikingVGG16bn_LIF_T4_CelebA2622_fold_/RSA/Human/pearson/RSA_static_dict_Human_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    
    svgg16bn_if_16_folds = {_:utils_.load(os.path.join(root, f'Face Identity SpikingVGG16bn_IF_T16_CelebA2622_fold_/RSA/Human/pearson/RSA_static_dict_Human_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    svgg16bn_lif_16_folds = {_:utils_.load(os.path.join(root, f'Face Identity SpikingVGG16bn_LIF_T16_CelebA2622_fold_/RSA/Human/pearson/RSA_static_dict_Human_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    
    # --- spiking_resnet
    # FIXME --- pending results...
    sresnet18_if_4 = {_:utils_.load(os.path.join(root, f'Face Identity SpikingResnet18_IF_FUNCTIONS_CelebA2622/Face Identity SpikingResnet18_IF_ATan_T4_CelebA2622/Analysis/RSA/Human/pearson/{_}/{target_ids}/RSA_results_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    sresnet18_lif_4 = {_:utils_.load(os.path.join(root, f'Face Identity SpikingResnet18_LIF_FUNCTIONS_CelebA2622/Face Identity SpikingResnet18_LIF_ATan_T4_CelebA2622/Analysis/RSA/Human/pearson/{_}/{target_ids}/RSA_results_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    
    #sresnet18_if_16
    sresnet18_lif_16_folds = {_:utils_.load(os.path.join(root, f'Face Identity SpikingResnet18_LIF_T16_CelebA2622_fold_/RSA/Human/pearson/RSA_static_dict_Human_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}

    sresnet50_if_4_0 = {_:utils_.load(os.path.join(root, f'Face Identity SpikingResnet50_IF_T4_CelebA2622_fold_0/Analysis/RSA/Human/pearson/{_}/{target_ids}/RSA_results_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    sresnet50_if_4_3 = {_:utils_.load(os.path.join(root, f'Face Identity SpikingResnet50_IF_T4_CelebA2622_fold_3/Analysis/RSA/Human/pearson/{_}/{target_ids}/RSA_results_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    #sresnet50_lif_4_0 =      # [warning] seems deep sresnet + LIF will have terrible results?
    sresnet101_if_4 = {_:utils_.load(os.path.join(root, f'Face Identity SpikingResnet101_IF_T4_CelebA2622_fold_0/Analysis/RSA/Human/pearson/{_}/{target_ids}/RSA_results_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    
    
    # --- sew_resnet
    sewresnet18_if_4 = {_:utils_.load(os.path.join(root, f'Face Identity SEWResnet18_IF_T4_CelebA2622_fold_0/Analysis/RSA/Human/pearson/{_}/{target_ids}/RSA_results_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    sewresnet18_lif_4 = {_:utils_.load(os.path.join(root, f'Face Identity SEWResnet18_LIF_T4_CelebA2622_fold_0/Analysis/RSA/Human/pearson/{_}/{target_ids}/RSA_results_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    
    sewresnet50_if_4 = {_:utils_.load(os.path.join(root, f'Face Identity SEWResnet50_IF_T4_CelebA2622_fold_0/Analysis/RSA/Human/pearson/{_}/{target_ids}/RSA_results_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    sewresnet50_lif_4 = {_:utils_.load(os.path.join(root, f'Face Identity SEWResnet50_LIF_T4_CelebA2622_fold_0/Analysis/RSA/Human/pearson/{_}/{target_ids}/RSA_results_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    
    sewresnet101_if_4_folds = {_:utils_.load(os.path.join(root, f'Face Identity SEWResnet101_IF_T4_CelebA2622_fold_/RSA/Human/pearson/RSA_static_dict_Human_pearson_{_}_{target_ids}.pkl')) for _ in cell_types}
    
    #---
    snn_interpolated_temporal_dict = {_: np.array([
                                      _interp_2d(sresnet18_if_4[_]['similarity_temporal']), 
                                      _interp_2d(sresnet18_lif_4[_]['similarity_temporal']),
                                      _interp_2d(sresnet50_if_4_0[_]['similarity_temporal']), 
                                      _interp_2d(sresnet50_if_4_3[_]['similarity_temporal']),
                                      _interp_2d(sresnet101_if_4[_]['similarity_temporal']), 
                                      
                                      _interp_2d(sewresnet18_if_4[_]['similarity_temporal']),
                                      _interp_2d(sewresnet18_lif_4[_]['similarity_temporal']), 
                                      _interp_2d(sewresnet50_if_4[_]['similarity_temporal']),
                                      _interp_2d(sewresnet50_lif_4[_]['similarity_temporal']),
                                      
                                      *[_interp_2d(svgg16bn_if_4_folds[_][fold_idx]['similarity_temporal']) for fold_idx in range(5)],
                                      *[_interp_2d(svgg16bn_lif_4_folds[_][fold_idx]['similarity_temporal']) for fold_idx in range(5)],
                                      *[_interp_2d(svgg16bn_if_16_folds[_][fold_idx]['similarity_temporal']) for fold_idx in range(5)],
                                      *[_interp_2d(svgg16bn_lif_16_folds[_][fold_idx]['similarity_temporal']) for fold_idx in range(5)],
                                      
                                      *[_interp_2d(sresnet18_lif_16_folds[_][fold_idx]['similarity_temporal']) for fold_idx in range(1, 5)],
                                      
                                      *[_interp_2d(sewresnet101_if_4_folds[_][fold_idx]['similarity_temporal']) for fold_idx in range(1, 5)],
                                      ]) for _ in cell_types}
    
    snn_interpolated_temporal_dict['non_selective'] = np.array([
                                      _interp_2d(np.nan_to_num(sresnet18_if_4['non_selective']['similarity_temporal'])), 
                                      _interp_2d(np.nan_to_num(sresnet18_lif_4['non_selective']['similarity_temporal'])),
                                      _interp_2d(np.nan_to_num(sresnet50_if_4_0['non_selective']['similarity_temporal'])), 
                                      _interp_2d(np.nan_to_num(sresnet50_if_4_3['non_selective']['similarity_temporal'])),
                                      _interp_2d(np.nan_to_num(sresnet101_if_4['non_selective']['similarity_temporal'])), 
                                      
                                      _interp_2d(np.nan_to_num(sewresnet18_if_4['non_selective']['similarity_temporal'])),
                                      _interp_2d(np.nan_to_num(sewresnet18_lif_4['non_selective']['similarity_temporal'])), 
                                      _interp_2d(np.nan_to_num(sewresnet50_if_4['non_selective']['similarity_temporal'])),
                                      _interp_2d(np.nan_to_num(sewresnet50_lif_4['non_selective']['similarity_temporal'])),
                                      
                                      *[_interp_2d(np.nan_to_num(svgg16bn_if_4_folds['non_selective'][fold_idx]['similarity_temporal'])) for fold_idx in range(5)],
                                      *[_interp_2d(np.nan_to_num(svgg16bn_lif_4_folds['non_selective'][fold_idx]['similarity_temporal'])) for fold_idx in range(5)],
                                      *[_interp_2d(np.nan_to_num(svgg16bn_if_16_folds['non_selective'][fold_idx]['similarity_temporal'])) for fold_idx in range(5)],
                                      *[_interp_2d(np.nan_to_num(svgg16bn_lif_16_folds['non_selective'][fold_idx]['similarity_temporal'])) for fold_idx in range(5)],
                                      
                                      *[_interp_2d(np.nan_to_num(sresnet18_lif_16_folds['non_selective'][fold_idx]['similarity_temporal'])) for fold_idx in range(1, 5)],
                                      
                                      *[_interp_2d(np.nan_to_num(sewresnet101_if_4_folds['non_selective'][fold_idx]['similarity_temporal'])) for fold_idx in range(1, 5)],
                                      ])
    
    snn_interpolated_temporal_p_dict = {_: np.array([
                                      _interp_2d(sresnet18_if_4[_]['similarity_p_perm_temporal']), 
                                      _interp_2d(sresnet18_lif_4[_]['similarity_p_perm_temporal']),
                                      _interp_2d(sresnet50_if_4_0[_]['similarity_p_perm_temporal']), 
                                      _interp_2d(sresnet50_if_4_3[_]['similarity_p_perm_temporal']),
                                      _interp_2d(sresnet101_if_4[_]['similarity_p_perm_temporal']), 
                                      
                                      _interp_2d(sewresnet18_if_4[_]['similarity_p_perm_temporal']),
                                      _interp_2d(sewresnet18_lif_4[_]['similarity_p_perm_temporal']), 
                                      _interp_2d(sewresnet50_if_4[_]['similarity_p_perm_temporal']),
                                      _interp_2d(sewresnet50_lif_4[_]['similarity_p_perm_temporal']),
                                      
                                      *[_interp_2d(svgg16bn_if_4_folds[_][fold_idx]['similarity_p_perm_temporal']) for fold_idx in range(5)],
                                      *[_interp_2d(svgg16bn_lif_4_folds[_][fold_idx]['similarity_p_perm_temporal']) for fold_idx in range(5)],
                                      *[_interp_2d(svgg16bn_if_16_folds[_][fold_idx]['similarity_p_perm_temporal']) for fold_idx in range(5)],
                                      *[_interp_2d(svgg16bn_lif_16_folds[_][fold_idx]['similarity_p_perm_temporal']) for fold_idx in range(5)],
                                      
                                      *[_interp_2d(sresnet18_lif_16_folds[_][fold_idx]['similarity_p_perm_temporal']) for fold_idx in range(1, 5)],
                                      
                                      *[_interp_2d(sewresnet101_if_4_folds[_][fold_idx]['similarity_p_perm_temporal']) for fold_idx in range(1, 5)],
                                      ]) for _ in cell_types}
    
    snn_interpolated_temporal_p_dict['non_selective'] = np.array([
                                      _interp_2d(np.nan_to_num(sresnet18_if_4['non_selective']['similarity_p_perm_temporal'], 1)), 
                                      _interp_2d(np.nan_to_num(sresnet18_lif_4['non_selective']['similarity_p_perm_temporal'], 1)),
                                      _interp_2d(np.nan_to_num(sresnet50_if_4_0['non_selective']['similarity_p_perm_temporal'], 1)), 
                                      _interp_2d(np.nan_to_num(sresnet50_if_4_3['non_selective']['similarity_p_perm_temporal'], 1)),
                                      _interp_2d(np.nan_to_num(sresnet101_if_4['non_selective']['similarity_p_perm_temporal'], 1)), 
                                      
                                      _interp_2d(np.nan_to_num(sewresnet18_if_4['non_selective']['similarity_p_perm_temporal'], 1)),
                                      _interp_2d(np.nan_to_num(sewresnet18_lif_4['non_selective']['similarity_p_perm_temporal'], 1)), 
                                      _interp_2d(np.nan_to_num(sewresnet50_if_4['non_selective']['similarity_p_perm_temporal'], 1)),
                                      _interp_2d(np.nan_to_num(sewresnet50_lif_4['non_selective']['similarity_p_perm_temporal'], 1)),
                                      
                                      *[_interp_2d(np.nan_to_num(svgg16bn_if_4_folds['non_selective'][fold_idx]['similarity_p_perm_temporal'], 1)) for fold_idx in range(5)],
                                      *[_interp_2d(np.nan_to_num(svgg16bn_lif_4_folds['non_selective'][fold_idx]['similarity_p_perm_temporal'], 1)) for fold_idx in range(5)],
                                      *[_interp_2d(np.nan_to_num(svgg16bn_if_16_folds['non_selective'][fold_idx]['similarity_p_perm_temporal'], 1)) for fold_idx in range(5)],
                                      *[_interp_2d(np.nan_to_num(svgg16bn_lif_16_folds['non_selective'][fold_idx]['similarity_p_perm_temporal'], 1)) for fold_idx in range(5)],
                                      
                                      *[_interp_2d(np.nan_to_num(sresnet18_lif_16_folds['non_selective'][fold_idx]['similarity_p_perm_temporal'], 1)) for fold_idx in range(1, 5)],
                                      
                                      *[_interp_2d(np.nan_to_num(sewresnet101_if_4_folds['non_selective'][fold_idx]['similarity_p_perm_temporal'], 1)) for fold_idx in range(1, 5)],
                                      ])
    
    ax[1, 0].imshow(np.array([np.mean(snn_interpolated_temporal_dict['qualified'], axis=0)[_*10, :] for _ in range(100)]), aspect='auto', vmin=-0.3, vmax=0.8, cmap='turbo')
    ax[1, 0].axis('off')
    ax[1, 1].imshow(np.array([np.mean(snn_interpolated_temporal_dict['legacy'], axis=0)[_*10, :] for _ in range(100)]), aspect='auto', vmin=-0.3, vmax=0.8, cmap='turbo')
    ax[1, 1].axis('off')
    ax[1, 2].imshow(np.array([np.mean(snn_interpolated_temporal_dict['selective'], axis=0)[_*10, :] for _ in range(100)]), aspect='auto', vmin=-0.3, vmax=0.8, cmap='turbo')
    ax[1, 2].axis('off')
    ax[1, 3].imshow(np.array([np.mean(snn_interpolated_temporal_dict['non_selective'], axis=0)[_*10, :] for _ in range(100)]), aspect='auto', vmin=-0.3, vmax=0.8, cmap='turbo')
    ax[1, 3].axis('off')
    
    for idx, _ in enumerate(['qualified', 'legacy', 'selective', 'non_selective']):
        
        inter_p = np.array([np.mean(snn_interpolated_temporal_p_dict[_], axis=0)[i*10, :] for i in range(100)])
        
        interp_p = (inter_p<0.05).astype(float)
        interp_p = scipy.ndimage.gaussian_filter(interp_p, sigma=2)
        
        ax[1, idx].contour(interp_p, levels:=[0.3], cmap='jet', linewidths=3)
    
    
    snn_temporal_dict = {_: {
                         'sresnet18_if_4': sresnet18_if_4[_]['similarity_temporal'][sresnet18_if_4[_]['sig_temporal_FDR'].astype(bool)],
                         'sresnet18_lif_4': sresnet18_lif_4[_]['similarity_temporal'][sresnet18_lif_4[_]['sig_temporal_FDR'].astype(bool)],
                         'sresnet50_if_4_0': sresnet50_if_4_0[_]['similarity_temporal'][sresnet50_if_4_0[_]['sig_temporal_FDR'].astype(bool)],
                         'sresnet50_if_4_3': sresnet50_if_4_3[_]['similarity_temporal'][sresnet50_if_4_3[_]['sig_temporal_FDR'].astype(bool)],
                         'sresnet101_if_4': sresnet101_if_4[_]['similarity_temporal'][sresnet101_if_4[_]['sig_temporal_FDR'].astype(bool)],
                         
                         'sewresnet18_if_4': sewresnet18_if_4[_]['similarity_temporal'][sewresnet18_if_4[_]['sig_temporal_FDR'].astype(bool)],
                         'sewresnet18_lif_4': sewresnet18_lif_4[_]['similarity_temporal'][sewresnet18_lif_4[_]['sig_temporal_FDR'].astype(bool)],
                         'sewresnet50_if_4': sewresnet50_if_4[_]['similarity_temporal'][sewresnet50_if_4[_]['sig_temporal_FDR'].astype(bool)],
                         'sewresnet50_lif_4': sewresnet50_lif_4[_]['similarity_temporal'][sewresnet50_lif_4[_]['sig_temporal_FDR'].astype(bool)],
                         
                         'svgg16bn_if_4_folds': np.concatenate([svgg16bn_if_4_folds[_][fold_idx]['similarity_temporal'][svgg16bn_if_4_folds[_][fold_idx]['sig_temporal_FDR'].astype(bool)] for fold_idx in range(5)]),
                         'svgg16bn_lif_4_folds': np.concatenate([svgg16bn_lif_4_folds[_][fold_idx]['similarity_temporal'][svgg16bn_lif_4_folds[_][fold_idx]['sig_temporal_FDR'].astype(bool)] for fold_idx in range(5)]),
                         'svgg16bn_if_16_folds': np.concatenate([svgg16bn_if_16_folds[_][fold_idx]['similarity_temporal'][svgg16bn_if_16_folds[_][fold_idx]['sig_temporal_FDR'].astype(bool)] for fold_idx in range(5)]),
                         'svgg16bn_lif_16_folds': np.concatenate([svgg16bn_lif_16_folds[_][fold_idx]['similarity_temporal'][svgg16bn_lif_16_folds[_][fold_idx]['sig_temporal_FDR'].astype(bool)] for fold_idx in range(5)]),
                         
                         'sresnet18_lif_16_folds': np.concatenate([sresnet18_lif_16_folds[_][fold_idx]['similarity_temporal'][sresnet18_lif_16_folds[_][fold_idx]['sig_temporal_FDR'].astype(bool)] for fold_idx in range(1, 5)]),
                         'sewresnet101_if_4_folds': np.concatenate([sewresnet101_if_4_folds[_][fold_idx]['similarity_temporal'][sewresnet101_if_4_folds[_][fold_idx]['sig_temporal_FDR'].astype(bool)] for fold_idx in range(1, 5)]),

                         } for _ in cell_types}
    
    
    snn_temporal_dict['non_selective'] = {
                                     'sresnet18_if_4': np.nan_to_num(sresnet18_if_4['non_selective']['similarity_temporal'][sresnet18_if_4['non_selective']['sig_temporal_FDR'].astype(bool)]),
                                     'sresnet18_lif_4': np.nan_to_num(sresnet18_lif_4['non_selective']['similarity_temporal'][sresnet18_lif_4['non_selective']['sig_temporal_FDR'].astype(bool)]),
                                     'sresnet50_if_4_0': np.nan_to_num(sresnet50_if_4_0['non_selective']['similarity_temporal'][sresnet50_if_4_0['non_selective']['sig_temporal_FDR'].astype(bool)]),
                                     'sresnet50_if_4_3': np.nan_to_num(sresnet50_if_4_3['non_selective']['similarity_temporal'][sresnet50_if_4_3['non_selective']['sig_temporal_FDR'].astype(bool)]),
                                     'sresnet101_if_4': np.nan_to_num(sresnet101_if_4['non_selective']['similarity_temporal'][sresnet101_if_4['non_selective']['sig_temporal_FDR'].astype(bool)]),
                                     
                                     'sewresnet18_if_4': np.nan_to_num(sewresnet18_if_4['non_selective']['similarity_temporal'][sewresnet18_if_4['non_selective']['sig_temporal_FDR'].astype(bool)]),
                                     'sewresnet18_lif_4': np.nan_to_num(sewresnet18_lif_4['non_selective']['similarity_temporal'][sewresnet18_lif_4['non_selective']['sig_temporal_FDR'].astype(bool)]),
                                     'sewresnet50_if_4': np.nan_to_num(sewresnet50_if_4['non_selective']['similarity_temporal'][sewresnet50_if_4['non_selective']['sig_temporal_FDR'].astype(bool)]),
                                     'sewresnet50_lif_4': np.nan_to_num(sewresnet50_lif_4['non_selective']['similarity_temporal'][sewresnet50_lif_4['non_selective']['sig_temporal_FDR'].astype(bool)]),
                                     
                                     'svgg16bn_if_4_folds': np.concatenate([np.nan_to_num(svgg16bn_if_4_folds['non_selective'][fold_idx]['similarity_temporal'][svgg16bn_if_4_folds['non_selective'][fold_idx]['sig_temporal_FDR'].astype(bool)]) for fold_idx in range(5)]),
                                     'svgg16bn_lif_4_folds': np.concatenate([np.nan_to_num(svgg16bn_lif_4_folds['non_selective'][fold_idx]['similarity_temporal'][svgg16bn_lif_4_folds['non_selective'][fold_idx]['sig_temporal_FDR'].astype(bool)]) for fold_idx in range(5)]),
                                     'svgg16bn_if_16_folds': np.concatenate([np.nan_to_num(svgg16bn_if_16_folds['non_selective'][fold_idx]['similarity_temporal'][svgg16bn_if_16_folds['non_selective'][fold_idx]['sig_temporal_FDR'].astype(bool)]) for fold_idx in range(5)]),
                                     'svgg16bn_lif_16_folds': np.concatenate([np.nan_to_num(svgg16bn_lif_16_folds['non_selective'][fold_idx]['similarity_temporal'][svgg16bn_lif_16_folds['non_selective'][fold_idx]['sig_temporal_FDR'].astype(bool)]) for fold_idx in range(5)]),
                                     
                                     'sresnet18_lif_16_folds': np.concatenate([np.nan_to_num(sresnet18_lif_16_folds['non_selective'][fold_idx]['similarity_temporal'][sresnet18_lif_16_folds['non_selective'][fold_idx]['sig_temporal_FDR'].astype(bool)]) for fold_idx in range(1,5)]),
                                     'sewresnet101_if_4_folds': np.concatenate([np.nan_to_num(sewresnet101_if_4_folds['non_selective'][fold_idx]['similarity_temporal'][sewresnet101_if_4_folds['non_selective'][fold_idx]['sig_temporal_FDR'].astype(bool)]) for fold_idx in range(1,5)]),
                                     
                                        }
    
    temporal_dict = {_:np.concatenate(list(temporal_dict[_].values())) for _ in temporal_dict.keys()}
    snn_temporal_dict = {_:np.concatenate(list(snn_temporal_dict[_].values())) for _ in temporal_dict.keys()}
    
    print('6')

    plt.tight_layout()
    
    # -----
    labels = ['All', 'Selective-I', 'Selective-II', 'Non-Slective']

    # 设置柱状图中每组的宽度
    bar_width = 0.35

    # 设置柱状图的位置
    index = np.arange(len(labels))

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(8,6))
    bar1 = ax.bar(index, t_mean, bar_width, label='ANN', color='orange')
    bar2 = ax.bar(index + bar_width, st_mean, bar_width, label='SNN')

    ax.errorbar(index, t_mean, yerr=([np.std(temporal_dict_list[_]) for _ in range(4)]), fmt='.', capsize=3, linewidth=2, color='black')
    ax.errorbar(index+bar_width, st_mean, yerr=([np.std(snn_temporal_dict_list[_]) for _ in range(4)]), fmt='.', capsize=3, linewidth=2, color='black')

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left', fontsize=24)

    for _ in range(4):
        sigstar([[_-0.0875,_+0.35-0.0875]], [p_values[_]], ax)
        
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=20)


def multi_model_macaque_style_comparison_a2s():
    
    plt.rcParams.update({"font.family": "Times New Roman"})
    
    baseline = utils_.load('/home/acxyle-workstation/Downloads/Face Identity Baseline/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    
    a2s_64_176 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity A2S_Baseline(T64P176)/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    
    a2s_64_224_05 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity A2S_Baseline(T64P224)/0.5_T/Analysis_0.5_T/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    a2s_64_224_all = utils_.load('/home/acxyle-workstation/Downloads/Face Identity A2S_Baseline(T64P224)/all_T/Analysis_all_T/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    
    a2s_256 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity A2S_Baseline(T256)/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    
    # -----
    # --- merge
    similarity_folds_array = np.array([a2s_64_176['similarity'], a2s_64_224_05['similarity'], a2s_64_224_all['similarity'], a2s_256['similarity']])
    folds_mean = np.mean(similarity_folds_array, axis=0)
    folds_std = np.std(similarity_folds_array, axis=0)
    
    similarity_p_folds = np.mean(np.array([a2s_64_176['similarity_p'], a2s_64_224_05['similarity_p'], a2s_64_224_all['similarity_p'], a2s_256['similarity_p']]), axis=0)
    (sig_FDR, p_FDR, alpha_Sadik, alpha_Bonf) = multipletests(similarity_p_folds, alpha=0.01, method='fdr_bh')    

    RSA_dict_across_folds = {
        'similarity': folds_mean,
        'similarity_std': folds_std,
        'similarity_perm': np.max(np.array([a2s_64_176['similarity_perm'], a2s_64_224_05['similarity_perm'], a2s_64_224_all['similarity_perm'], a2s_256['similarity_perm']]), axis=0),
        
        'p_FDR': p_FDR,
        'similarity_p': similarity_p_folds,
        
        'sig_FDR': sig_FDR,
        'sig_Bonf': p_FDR<alpha_Bonf
        }
    
    # ---
    a2s_avg = np.mean([_interp(a2s_256['similarity']), _interp(a2s_64_176['similarity']), 
                       _interp(a2s_64_224_05['similarity']), _interp(a2s_64_224_all['similarity'])], axis=0)

    # ---
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(_interp(baseline['similarity']), color='steelblue', alpha=0.1)
    
    ax.plot(_interp(a2s_256['similarity']), color='steelblue', alpha=0.1)
    ax.plot(_interp(a2s_64_224_05['similarity']), color='steelblue', alpha=0.1)
    ax.plot(_interp(a2s_64_224_all['similarity']), color='steelblue', alpha=0.1)
    
    ax.plot(a2s_avg, color='steelblue', linewidth=5, label='A2S VGG')
    
    ax.plot(_interp(baseline['similarity']), color='gold', linewidth=5, label='VGG')
    
    # ---
    fake_legend_stats_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='steelblue', markersize=5, markeredgecolor='steelblue'),
                                 Line2D([0], [0], marker='o', color='none', markerfacecolor='gold', markersize=5, markeredgecolor='gold')]
    
    fake_legend_stats_labels = [
        f"mean {np.mean(a2s_avg):.3f} std {np.std(a2s_avg):.3f}",
        f"mean {np.mean(_interp(baseline['similarity'])):.3f} std {np.std(_interp(baseline['similarity'])):.3f}"
        ]
    
    fake_legend = ax.legend(fake_legend_stats_handles, fake_legend_stats_labels, framealpha=0.25, ncol=1, handlelength=0, borderpad=0, labelspacing=0, loc='lower center')
    ax.add_artist(fake_legend)
    
    ax.legend()
    
    plt.show()
    
    print('6')


def multi_model_macaque_style_comparison_a2s_ver2_temporal(alpha=0.05, FDR_method='fdr_bh'):
    
    plt.rcParams.update({'font.size': 18})    
    plt.rcParams.update({"font.family": "Times New Roman"})
    
    layers, neurons, shapes = utils_.get_layers_and_units(model_name='resnet18', feature_shape=(3,224,224))
    
    RSA_dict_folds = utils_.load('/home/acxyle-workstation/Downloads/Face Identity Resnet18_CelebA2622_fold_/RSA/Monkey/pearson/RSA_static_dict_Monkey_pearson_None_None.pkl')
    
    
    # -----
    similarity_folds_array = np.array([RSA_dict_folds[fold_idx]['similarity_temporal'] for fold_idx in range(5)])
    folds_mean = np.mean(similarity_folds_array, axis=0)
    folds_std = np.std(similarity_folds_array, axis=0)


    # --- init
    p_temporal_FDR = np.zeros((len(layers), similarity_folds_array.shape[-1]))     # (num_layers, num_time_steps)
    sig_temporal_FDR =  p_temporal_FDR.copy()
    sig_temporal_Bonf = p_temporal_FDR.copy()

    similarity_p_perm_temporal = np.mean(np.array([RSA_dict_folds[fold_idx]['similarity_p_perm_temporal'] for fold_idx in range(5)]), axis=0)

    for _ in range(len(layers)):
        (sig_temporal_FDR[_, :], p_temporal_FDR[_, :], alpha_Sadik_temporal, alpha_Bonf_temporal) = multipletests(similarity_p_perm_temporal[_, :], alpha=alpha, method=FDR_method)      # FDR
        sig_temporal_Bonf[_, :] = p_temporal_FDR[_, :]<alpha_Bonf_temporal     # Bonf correction
            
    RSA_dict_across_folds = {
        'similarity_temporal': folds_mean,
        'similarity_temporal_std': folds_std,

        'sig_temporal_FDR': sig_temporal_FDR,
        'sig_temporal_Bonf': sig_temporal_Bonf,

        'p_temporal_FDR': p_temporal_FDR,

        }
    
    idx, layer_n, _, _ = utils_.activation_function('resnet18', layers, act_only=True)
    RSA_dict_across_folds_neuron = {_:RSA_dict_across_folds[_][idx] for _ in RSA_dict_across_folds.keys()}
    
    # ---
    a2s_vgg16 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity A2S_Resnet18/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    
    # ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ts = np.arange(-50,201,10)
    extent = [ts.min()-5, ts.max()+5, -0.5, RSA_dict_across_folds_neuron['similarity_temporal'].shape[0]-0.5]
    
    tmp = a2s_vgg16['similarity_temporal'] - RSA_dict_across_folds_neuron['similarity_temporal']
    
    img = ax.imshow(tmp, aspect='auto', extent=extent)
    fig.colorbar(img)

    test = scipy.ndimage.gaussian_filter(a2s_vgg16['sig_temporal_Bonf'], sigma=1)
    
    test[test>(1-alpha)] = np.nan
    
    ax.imshow(test, aspect='auto',  cmap='gray', extent=extent, alpha=0.3)
    ax.contour(a2s_vgg16['sig_temporal_Bonf'], levels:=[0.5], origin='upper', cmap='jet', extent=extent, linewidths=3)

    mask = a2s_vgg16['sig_temporal_Bonf']
    

    fake_legend_describe_numpy(tmp, ax, mask.astype(bool))
    
    print('6')
    
def multi_model_macaque_style_comparison_a2s_ver2():
    
    plt.rcParams.update({'font.size': 18})    
    plt.rcParams.update({"font.family": "Times New Roman"})
    
    def _plot(colors, RSA_dict, layers, error_control_measure='sig_FDR', title=None, error_area=True, vlim=None, legend=False):
        
        plot_x = range(len(layers))
        
        if colors == 'ANN':
            colors = ['yellow', 'orange', 'chocolate', 'olive']
        elif colors == 'SNN':
            colors = ['skyblue', 'blue', 'deepskyblue', 'deepskyblue']
        
        # -----
        # --- 1. plot shaded error bars
        if error_area:
            perm_mean = np.mean(RSA_dict['similarity_perm'], axis=1)  
            perm_std = np.std(RSA_dict['similarity_perm'], axis=1)  
            ax.fill_between(plot_x, perm_mean-perm_std, perm_mean+perm_std, color='lightgray', edgecolor='none', alpha=0.5)
            ax.fill_between(plot_x, perm_mean-2*perm_std, perm_mean+2*perm_std, color='lightgray', edgecolor='none', alpha=0.5)
            ax.fill_between(plot_x, perm_mean-3*perm_std, perm_mean+3*perm_std, color='lightgray', edgecolor='none', alpha=0.5)
            ax.plot(plot_x, perm_mean, color='dimgray')

        # --- 2. plot RSA scores with FDR results
        similarity = RSA_dict['similarity']

        if 'similarity_std' in RSA_dict.keys():
            ax.fill_between(plot_x, similarity-RSA_dict['similarity_std'], similarity+RSA_dict['similarity_std'], edgecolor=None, facecolor=colors[0], alpha=0.75)

        for idx, _ in enumerate(RSA_dict[error_control_measure], 0):
             if not _:   
                 ax.scatter(idx, similarity[idx], facecolors='none', edgecolors=colors[1])
             else:
                 ax.scatter(idx, similarity[idx], facecolors=colors[1], edgecolors=colors[1])

        ax.plot(similarity, linestyle='dotted', color=colors[3])

        ax.set_ylabel("Spearman's $\\rho$")
        ax.set_xticks(plot_x)
        ax.set_xticklabels(layers, rotation=90, ha='center')
        ax.set_xlim([0, len(layers)-1])
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(f'{title}')

        handles, labels = ax.get_legend_handles_labels()

        hollow_circle = Line2D([0], [0], marker='o', color=colors[2], linestyle='dotted', markerfacecolor='none', markersize=5, markeredgecolor=colors[0], linewidth=1)
        solid_circle = Line2D([0], [0], marker='o', color=colors[2], linestyle='dotted', markerfacecolor=colors[1], markersize=5, markeredgecolor=colors[0], linewidth=1)

        handles.extend([hollow_circle, solid_circle])
        labels.extend([f"fialed {error_control_measure.split('_')[1]}", f"passed {error_control_measure.split('_')[1]}"])

        similarity_ = similarity[~np.isnan(similarity)]
        if error_area:
            y_radius = np.max(similarity_) - np.min(perm_mean[~np.isnan(perm_mean)])
        else:
            y_radius = np.max(similarity_) - np.min(similarity_)

        if not vlim:
            if error_area:
                ylim_bottom = np.min([np.min(similarity_), np.min(perm_mean[~np.isnan(perm_mean)])])
            else:
                ylim_bottom = np.min(similarity_)
            ax.set_ylim([ylim_bottom-0.025*y_radius, np.max(similarity_)+0.05*y_radius])
        else:
            ax.set_ylim(vlim)
    
    layers, neurons, shapes = utils_.get_layers_and_units(model_name='resnet18', feature_shape=(3,224,224))
    
    RSA_dict_folds = utils_.load('/home/acxyle-workstation/Downloads/Face Identity Resnet18_CelebA2622_fold_/RSA/Monkey/pearson/RSA_static_dict_Monkey_pearson_None_None.pkl')
    
    similarity_folds_array = np.array([RSA_dict_folds[fold_idx]['similarity'] for fold_idx in range(5)])
    folds_mean = np.mean(similarity_folds_array, axis=0)
    folds_std = np.std(similarity_folds_array, axis=0)
    
    similarity_p_folds = np.mean(np.array([RSA_dict_folds[fold_idx]['similarity_p'] for fold_idx in range(5)]), axis=0)
    (sig_FDR, p_FDR, alpha_Sadik, alpha_Bonf) = multipletests(similarity_p_folds, alpha=0.05, method='fdr_bh')    

    RSA_dict_across_folds = {
        'similarity': folds_mean,
        'similarity_std': folds_std,
        'similarity_perm': np.max(np.array([RSA_dict_folds[fold_idx]['similarity_perm'] for fold_idx in range(5)]), axis=0),
        
        'p_FDR': p_FDR,
        'similarity_p': similarity_p_folds,
        
        'sig_FDR': sig_FDR,
        'sig_Bonf': p_FDR<alpha_Bonf
        }
      
    # ---
    idx, layer_n, _, _ = utils_.activation_function('resnet18', layers, act_only=True)
    RSA_dict_across_folds_neuron = {_:RSA_dict_across_folds[_][idx] for _ in RSA_dict_across_folds.keys()}
    
    # ---
    fig, ax = plt.subplots(figsize=(10, 6))

    _plot('ANN', RSA_dict_across_folds_neuron, layer_n)
    
    a2s_vgg16 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity A2S_Resnet18/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    
    _plot('SNN', a2s_vgg16, layer_n, title='A2S Resnet18 v.s. Resnet18')
    
    # ---
    fake_legend_stats_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='deepskyblue', markersize=5, markeredgecolor='steelblue'),
                                 Line2D([0], [0], marker='o', color='none', markerfacecolor='chocolate', markersize=5, markeredgecolor='gold')]
    
    fake_legend_stats_labels = [
        f"A2S Resnet18: {np.mean(a2s_vgg16['similarity']):.3f} (±{np.std(a2s_vgg16['similarity']):.3f})",
        f"Resnet18: {np.mean(RSA_dict_across_folds_neuron['similarity']):.3f} (±{np.std(RSA_dict_across_folds_neuron['similarity']):.3f})"
        ]
    
    fake_legend = ax.legend(fake_legend_stats_handles, fake_legend_stats_labels, framealpha=0.25, ncol=1, handlelength=0, borderpad=0, labelspacing=0, loc='lower center')
    ax.add_artist(fake_legend)
    
    ax.legend()

def multi_model_macaque_style_comparison_resnet():
    """
        [warning] the results are incomplete
        waitng for further results to complete the comparison
    """
    plt.rcParams.update({"font.family": "Times New Roman"})
    
    # ANN Resnet vs Spiking Resnet (vs SEW Resnet)
    # ---
    resnet18 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity Resnet18_CelebA2622_fold_/RSA/Monkey/pearson/RSA_static_dict_Monkey_pearson_None_None.pkl')
    resnet50 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity Resnet50_CelebA2622_fold_/RSA/Monkey/pearson/RSA_static_dict_Monkey_pearson_None_None.pkl')
    
    resnet101 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity Resnet101_CelebA2622_fold_0/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    resnet152 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity Resnet152_CelebA2622_fold_0/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    
    # --- static
    resnet18_avg = []
    resnet50_avg = []
    
    for fold_idx in range(5):
        resnet18_avg.append(resnet18[fold_idx]['similarity'])
        resnet50_avg.append(resnet50[fold_idx]['similarity'])
 
    # ---
    resnet18_avg = np.mean(np.array(resnet18_avg), axis=0)
    resnet50_avg = np.mean(np.array(resnet50_avg), axis=0)
    
    resnet_avg = np.mean([_interp(resnet18_avg), _interp(resnet50_avg), 
                          _interp(resnet101['similarity']), _interp(resnet152['similarity'])], axis=0)
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    for fold_idx in range(5):
        
        ax.plot(_interp(resnet18[fold_idx]['similarity']), color='orange', alpha=0.1)
        ax.plot(_interp(resnet50[fold_idx]['similarity']), color='orange', alpha=0.1)
    
    ax.plot(_interp(resnet101['similarity']), color='orange', alpha=0.1)
    ax.plot(_interp(resnet152['similarity']), color='orange', alpha=0.1)
    
    ax.plot(resnet_avg, color='orange', linewidth=5, label='Resnet')
    
    # -- temporal
    significant_values_resnet = [ 
        resnet101['similarity_temporal'][resnet101['sig_temporal_FDR'].astype(bool)],
        resnet152['similarity_temporal'][resnet152['sig_temporal_FDR'].astype(bool)],
        *[resnet18[_]['similarity_temporal'][resnet18[_]['sig_temporal_FDR'].astype(bool)] for _ in range(5)],
        *[resnet50[_]['similarity_temporal'][resnet50[_]['sig_temporal_FDR'].astype(bool)] for _ in range(5)],
        ]
    
    np.mean(np.concatenate(significant_values_resnet))
    
    print('6')
    
    # FIXME --- pending results...
    sresnet18_if_4 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SpikingResnet18_IF_FUNCTIONS_CelebA2622/Face Identity SpikingResnet18_IF_ATan_T4_CelebA2622/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    #sresnet18_if_16
    
    sresnet18_lif_4 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SpikingResnet18_LIF_FUNCTIONS_CelebA2622/Face Identity SpikingResnet18_LIF_ATan_T4_CelebA2622/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    sresnet18_lif_16 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SpikingResnet18_LIF_T16_CelebA2622_fold_/RSA/Monkey/pearson/RSA_static_dict_Monkey_pearson_None_None.pkl')

    sresnet50_if_4_0 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SpikingResnet50_IF_T4_CelebA2622_fold_0/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    sresnet50_if_4_3 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SpikingResnet50_IF_T4_CelebA2622_fold_3/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    #sresnet50_lif_4_0 =      # [warning] seems deep sresnet + LIF will have terrible results?
    
    sresnet101_if_4 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SpikingResnet101_IF_T4_CelebA2622_fold_0/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    #sresnet101_lif_4
    
    #sresnet152_if_4
    #sresnet152_lif_4

    # --- static
    sresnet18_lif_16_avg = []
    
    for fold_idx in range(5):
        sresnet18_lif_16_avg.append(sresnet18_lif_16[fold_idx]['similarity'])

    # ---
    sresnet18_lif_16_avg = np.mean(np.array(sresnet18_lif_16_avg), axis=0)

    sresnet_avg = np.mean([_interp(sresnet18_lif_16_avg), 
                          _interp(sresnet18_if_4['similarity']), _interp(sresnet18_lif_4['similarity']),
                          _interp(sresnet50_if_4_0['similarity']), _interp(sresnet50_if_4_3['similarity']),
                          _interp(sresnet101_if_4['similarity']),
                          ], axis=0)
    
    for fold_idx in range(5):
        ax.plot(_interp(sresnet18_lif_16[fold_idx]['similarity']), color='blue', alpha=0.1)
    
    ax.plot(_interp(sresnet18_if_4['similarity']), color='blue', alpha=0.1)
    ax.plot(_interp(sresnet18_lif_4['similarity']), color='blue', alpha=0.1)
    
    ax.plot(_interp(sresnet50_if_4_0['similarity']), color='blue', alpha=0.1)
    ax.plot(_interp(sresnet50_if_4_3['similarity']), color='blue', alpha=0.1)

    ax.plot(_interp(sresnet101_if_4['similarity']), color='blue', alpha=0.1)
    
    ax.plot(sresnet_avg, color='blue', linewidth=5, label='SpikingResnet')
    
    # --- temporal
    significant_values_sresnet = [ 
        sresnet18_if_4['similarity_temporal'][sresnet18_if_4['sig_temporal_FDR'].astype(bool)],
        sresnet18_lif_4['similarity_temporal'][sresnet18_lif_4['sig_temporal_FDR'].astype(bool)],
        
        sresnet50_if_4_0['similarity_temporal'][sresnet50_if_4_0['sig_temporal_FDR'].astype(bool)],
        sresnet50_if_4_3['similarity_temporal'][sresnet50_if_4_3['sig_temporal_FDR'].astype(bool)],
        sresnet101_if_4['similarity_temporal'][sresnet101_if_4['sig_temporal_FDR'].astype(bool)],
        
        *[sresnet18_lif_16[_]['similarity_temporal'][sresnet18_lif_16[_]['sig_temporal_FDR'].astype(bool)] for _ in range(5)],
        ]
    
    np.mean(np.concatenate(significant_values_sresnet))
    
    # ---
    sewresnet18_if_4 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SEWResnet18_IF_T4_CelebA2622_fold_0/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    sewresnet18_lif_4 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SEWResnet18_LIF_T4_CelebA2622_fold_0/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    
    sewresnet50_if_4 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SEWResnet50_IF_T4_CelebA2622_fold_0/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    sewresnet50_lif_4 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SEWResnet50_LIF_T4_CelebA2622_fold_0/Analysis/RSA/Monkey/pearson/RSA_results_pearson.pkl')
    
    sewresnet101_if_4 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SEWResnet101_IF_T4_CelebA2622_fold_/RSA/Monkey/pearson/RSA_static_dict_Monkey_pearson_None_None.pkl')

# =============================================================================
#     sewresnet152 = 
# =============================================================================

    # --- static
    sewresnet101_if_4_avg = []
    
    for fold_idx in range(1, 5):
        sewresnet101_if_4_avg.append(sewresnet101_if_4[fold_idx]['similarity'])

    # ---
    sewresnet101_if_4_avg = np.mean(sewresnet101_if_4_avg, axis=0)

    sewresnet_avg = np.mean([_interp(sewresnet101_if_4_avg), 
                          _interp(sewresnet18_if_4['similarity']), _interp(sewresnet18_lif_4['similarity']),
                          _interp(sewresnet50_if_4['similarity']), _interp(sewresnet50_lif_4['similarity'])], axis=0)
    
    for fold_idx in range(1, 5):
        ax.plot(_interp(sewresnet101_if_4[fold_idx]['similarity']), color='cyan', alpha=0.1)
    
    ax.plot(_interp(sewresnet18_if_4['similarity']), color='cyan', alpha=0.1)
    ax.plot(_interp(sewresnet18_lif_4['similarity']), color='cyan', alpha=0.1)
    
    ax.plot(_interp(sewresnet50_if_4['similarity']), color='cyan', alpha=0.1)
    ax.plot(_interp(sewresnet50_lif_4['similarity']), color='cyan', alpha=0.1)
    
    ax.plot(sewresnet_avg, color='cyan', linewidth=5, label='SEWResnet')
    
    # --- temporal
    significant_values_sewresnet = [ 
        sewresnet18_if_4['similarity_temporal'][sewresnet18_if_4['sig_temporal_FDR'].astype(bool)],
        sewresnet18_lif_4['similarity_temporal'][sewresnet18_lif_4['sig_temporal_FDR'].astype(bool)],
        sewresnet50_if_4['similarity_temporal'][sewresnet50_if_4['sig_temporal_FDR'].astype(bool)],
        sewresnet50_lif_4['similarity_temporal'][sewresnet50_lif_4['sig_temporal_FDR'].astype(bool)],
        *[sewresnet101_if_4[_]['similarity_temporal'][sewresnet101_if_4[_]['sig_temporal_FDR'].astype(bool)] for _ in range(1,5)],
        ]
    
    np.mean(np.concatenate(significant_values_sewresnet))
    
    
    ax.legend()
    
    print('6')
    
    ax.set_xticks([0,200,400,600,800,1000])
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    ax.grid(True, axis='y', linestyle='--', linewidth=1, alpha=0.5)
    ax.tick_params(axis='both', labelsize=20)
    
    ax.legend(fontsize=30)
    
    plt.tight_layout()
    fig.savefig('Resnet.svg', transparent=True)


def multi_model_macaque_style_comparison_vgg():
    """
        [warning] - the results are incomplete
    """
    plt.rcParams.update({"font.family": "Times New Roman"})
    
    # ANN VGG vs Spiking VGG - Monkey
    vgg16 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity VGG16_fold_/RSA/Monkey/pearson/RSA_static_dict_Monkey_pearson_None_None.pkl')
    vgg16bn = utils_.load('/home/acxyle-workstation/Downloads/Face Identity VGG16bn_fold_/RSA/Monkey/pearson/RSA_static_dict_Monkey_pearson_None_None.pkl')
    
    svgg16_if_4 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SpikingVGG16bn_IF_T4_CelebA2622_fold_/RSA/Monkey/pearson/RSA_static_dict_Monkey_pearson_None_None.pkl')
    svgg16_if_16 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SpikingVGG16bn_IF_T16_CelebA2622_fold_/RSA/Monkey/pearson/RSA_static_dict_Monkey_pearson_None_None.pkl')
    svgg16_lif_4 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SpikingVGG16bn_LIF_T4_CelebA2622_fold_/RSA/Monkey/pearson/RSA_static_dict_Monkey_pearson_None_None.pkl')
    svgg16_lif_16 = utils_.load('/home/acxyle-workstation/Downloads/Face Identity SpikingVGG16bn_LIF_T16_CelebA2622_fold_/RSA/Monkey/pearson/RSA_static_dict_Monkey_pearson_None_None.pkl')
    
    # --- ANN average
    vgg16_avg = []
    vgg16bn_avg = []
    
    svgg16_if_4_avg = []
    svgg16_if_16_avg = []
    svgg16_lif_4_avg = []
    svgg16_lif_16_avg = []
    
    for fold_idx in range(5):
        vgg16_avg.append(vgg16[fold_idx]['similarity'])
        vgg16bn_avg.append(vgg16bn[fold_idx]['similarity'])
        
        svgg16_if_4_avg.append(svgg16_if_4[fold_idx]['similarity'])
        svgg16_if_16_avg.append(svgg16_if_16[fold_idx]['similarity'])
        svgg16_lif_4_avg.append(svgg16_lif_4[fold_idx]['similarity'])
        svgg16_lif_16_avg.append(svgg16_lif_16[fold_idx]['similarity'])
        
    # --- static
    vgg16_avg = np.mean(np.array(vgg16_avg), axis=0)
    vgg16bn_avg = np.mean(np.array(vgg16bn_avg), axis=0)
        
    vgg_avg = np.mean([_interp(vgg16_avg), _interp(vgg16bn_avg)], axis=0)
    
    # --- 
    svgg16_if_4_avg = np.mean(svgg16_if_4_avg, axis=0)
    svgg16_if_16_avg = np.mean(svgg16_if_16_avg, axis=0)
    svgg16_lif_4_avg = np.mean(svgg16_lif_4_avg, axis=0)
    svgg16_lif_16_avg = np.mean(svgg16_lif_16_avg, axis=0)
    
    svgg_avg = np.mean([_interp(svgg16_if_4_avg), _interp(svgg16_if_16_avg), 
                        _interp(svgg16_lif_4_avg), _interp(svgg16_lif_16_avg)], axis=0)
    
    # --- temporal
    significant_values_vgg = [ 
        *[vgg16[_]['similarity_temporal'][vgg16[_]['sig_temporal_FDR'].astype(bool)] for _ in range(5)],
        *[vgg16bn[_]['similarity_temporal'][vgg16bn[_]['sig_temporal_FDR'].astype(bool)] for _ in range(5)],
        ]
    
    vgg_pool = np.concatenate(significant_values_vgg)
    
    significant_values_svgg = [ 
        *[svgg16_if_4[_]['similarity_temporal'][svgg16_if_4[_]['sig_temporal_FDR'].astype(bool)] for _ in range(5)],
        *[svgg16_if_16[_]['similarity_temporal'][svgg16_if_16[_]['sig_temporal_FDR'].astype(bool)] for _ in range(5)],
        *[svgg16_lif_4[_]['similarity_temporal'][svgg16_lif_4[_]['sig_temporal_FDR'].astype(bool)] for _ in range(5)],
        *[svgg16_lif_16[_]['similarity_temporal'][svgg16_lif_16[_]['sig_temporal_FDR'].astype(bool)] for _ in range(5)],
        ]
    
    svgg_pool = np.concatenate(significant_values_svgg)
    
    # ---
    fig, ax = plt.subplots(figsize=(10,6))
        
    # --- plot
    for fold_idx in range(5):
        
        ax.plot(_interp(vgg16[fold_idx]['similarity']), color='orange', alpha=0.1)
        ax.plot(_interp(vgg16bn[fold_idx]['similarity']), color='orange', alpha=0.1)
 
        # ---
        ax.plot(_interp(svgg16_if_4[fold_idx]['similarity']), color='blue', alpha=0.1)
        ax.plot(_interp(svgg16_if_16[fold_idx]['similarity']), color='blue', alpha=0.1)
        ax.plot(_interp(svgg16_lif_4[fold_idx]['similarity']), color='blue', alpha=0.1)
        ax.plot(_interp(svgg16_lif_16[fold_idx]['similarity']), color='blue', alpha=0.1)
        
    ax.plot(vgg_avg, color='orange', linewidth=5, label='VGG')
    ax.plot(svgg_avg, color='blue', linewidth=5, label='SpikingVGG')
    
    ax.set_xticks([0,200,400,600,800,1000])
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    ax.grid(True, axis='y', linestyle='--', linewidth=1, alpha=0.5)
    ax.tick_params(axis='both', labelsize=20)
    
    ax.legend(fontsize=30)
    
    plt.tight_layout()
    fig.savefig('VGG.svg', transparent=True)
    
    print('6')
    
    
def _interp_2d(input, num_interp_x=1100):
    """
        assume the temporal dimension is constant, only change the layer dimension
    """
    x = np.arange(input.shape[0])
    y = np.arange(input.shape[1])
    
    f = RectBivariateSpline(x, y, input)
    
    x_new = np.linspace(0, input.shape[0]-1, num_interp_x)
    
    output = f(x_new, y)
    
    return output[:1000, :]
    

# --- interplot
def _interp(input, num_interp=1100):
    
    y = input
    x = np.arange(len(input))
    
    f = interp1d(x, y)
    
    return f(np.linspace(0, len(input)-1, num_interp))[:1000]     # ignore the classification head

def _ccc(x, y):
    
    cor = np.corrcoef(x, y)[0][1]

    mean_true = np.mean(x)
    mean_pred = np.mean(y)
    
    var_true = np.var(x)
    var_pred = np.var(y)
    
    sd_true = np.std(x)
    sd_pred = np.std(y)
    
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    ccc = numerator / denominator
    
    return ccc

#FIXME - this process is too complicated, needs to be simplified
def DSM_calculation(metric:str='pearson', feature:np.array=None, vectorize:bool=False, **kwargs):
    """
        by default, the function will remove the empty and nan values
        
        if you prefer to reset all nan values to 0 then calculate
        
        this section only relative to the fisrt_corr
               
        [task]:
            find a way to execute the second half of RSA
            
        [bonus]:
            possible statistics:
                1. pingouin supports: 
                    'bicor': Biweight midcorrelation; 
                    'percbend': Percentage bend correlation; 
                    'shepherd': Shepherd's pi correlation; 
                    'skipped': Skipped correlation
            
        return:
            
    """

    def _mahalanobis(input):
        
        num_sampels = input.shape[0]
        num_features = input.shape[1]
        
        product_list = list(itertools.product(np.arange(num_sampels),np.arange(num_sampels)))
        
        cm = np.cov(input, rowvar=False) + np.eye(num_features)*1e-8
        icm = np.linalg.inv(cm)
        
        mahal_matrix = np.full((num_sampels, num_sampels), np.nan)
        
        for _ in product_list:
            mahal_matrix[_[0], _[1]] = mahalanobis(input[_[0]], input[_[1]], icm)
        
        return mahal_matrix
    
    def _size_and_mask_check(input):
        """
            ...
        """
        if input.size == 0:     # size check
            return None
        mask = ~np.any(np.isnan(input), axis=0)
        if mask.size == 0:     # mask check
            return None
        return mask
    
    if 'euclidean' in metric.lower():
        
        if (mask:=_size_and_mask_check(feature)) is not None:

            feature = feature[:, mask]     # remove the cell/unit with nan values
            eu_distance = pdist(feature, 'euclidean')     
            
            if not vectorize:
                return squareform(eu_distance)
            else:
                return eu_distance

    elif 'pearson' in metric.lower():

        if (mask:=_size_and_mask_check(feature)) is not None:
            
            #FIXME --- nan if all values in one piece are equal, i.e. std = 0, this problem frequently happens for ns and ne
            # --- this version did not make a clear separation for simplification, refer to old version code if encountered
            if np.any([len(np.unique(_))==1 for _ in feature]):
                ...
            
            similarity = np.ma.corrcoef(np.ma.masked_invalid(feature)).data     # (num_ids, num_ids)
                
            return RSM_process(similarity, vectorize, 'arctanh')     
   
    elif 'spearman':
        
        return RSM_process(spearmanr(feature, axis=1, nan_policy='omit').statistic, vectorize)
    
    elif 'mahalanobis':
        
        if (mask:=_size_and_mask_check(feature)) is not None:
            
            mahal_matrix = _mahalanobis(feature[:, mask])
            
            return RSM_process(mahal_matrix, vectorize, 'standardization')
    
    elif 'concordance':
        
        if (mask:=_size_and_mask_check(feature)) is not None:
            
            feature = feature[:, mask]
            
            num_sampels = feature.shape[0]
            
            product_list = list(itertools.product(np.arange(num_sampels),np.arange(num_sampels)))
            
            ccc_matrix = np.full((num_sampels, num_sampels), np.nan)
            
            for _ in product_list:
                ccc_matrix[_[0], _[1]] = _ccc(feature[_[0]], feature[_[1]])
            
            return RSM_process(ccc_matrix, vectorize, 'standardization')
            
    else:
        raise ValueError(f'[Coderror] invalid metric [{metric}]')
    


#FIXME --- semms the entire process should be rewrite to add extra API
def RSM_process(matrix:np.ndarray, vectorize:bool=False, post_process:bool=None, **kwargs):
    """
        need to explain why use np.arctanh(), amplify the values near -1 and 1? but this will deteriorate the data range 
         from 0 to 2
         
         based on experiments, seems the non_linear tranformation like np.arctanh() will not inpact the results of spearman()
         but if the second similarity metric is scaling sensitive, the results may be influenced
         
        [task]:
            add the pre/post process for both RSA-I and RSA-II
    """
    #warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # --- [default] transfer the RSM to DSM by linear subtraction, spearmanr() is invariant to linear tranformation
    if post_process == None:
        DSM = 1 - matrix    # [-1, 1] -> [0, 2] if RSM is [-1, 1]
    
    # --- [lagacy design] from source Matlab code, spearman() is invariant to scaling invariance
    elif post_process == 'arctanh': 
        #matrix = matrix-1e-8
        matrix = np.arctanh(matrix)    
        DSM = 1 - matrix     # (-inf, +inf)
        
    # --- arctanh is mathematically equal to **fisher_z** below, the 1 and np.max(matrix) is not crucial for linear invariant RSA
    elif post_process == 'fisher_z': 
        matrix = matrix-1e-8
        matrix = np.log((1+matrix)/(1-matrix))/2
        DSM = np.max(matrix) - matrix     # [0, +inf)
    
    # --- [standardization / z-score transformation]
    elif post_process == 'standardization':
        matrix = (matrix-np.mean(matrix))/np.std(matrix)
        DSM = np.max(matrix) - matrix
        
    elif post_process == 'normalization':
        matrix = (matrix-np.min(matrix))/(np.max(matrix)-np.min(matrix))
        DSM = np.max(matrix) - matrix
 
    elif post_process == 'square':
        matrix = matrix**2
        DSM = np.max(matrix) - matrix
        
    elif post_process == 'log':
        matrix = matrix + 1
        matrix = np.log(matrix)/np.log(np.e)
        DSM = np.max(matrix) - matrix
        
    elif post_process == 'yeo-johnson':
        matrix, used_lambda = stats.yeojohnson(matrix)
        DSM = np.max(matrix) - matrix
        
    # -----
    if vectorize:
        return RSM_vectorize(DSM)
    else:
        return DSM
    

def RSM_vectorize(matrix:np.ndarray, k=1, m=None):
    
    return matrix[np.triu_indices(matrix.shape[0], k, m)]


"""
    below functions directly from the CKA colab tutorial
"""

def gram_linear(x, **kwargs):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0, **kwargs):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T, rtol=1e-06, atol=1e-05):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)

def describe_numpy(input:np.array=None):
    """
        this function uses a dict format to print the information of a numpy array like Pandas describe
    """
    
    if input is None or input[~np.isnan(input)].size == 0:
        
        print('[Codinfo] [describe_numpy] received input is None or full of nan values, returned None')
        
        return None
    
    array_stats = {
        "count_valid": input[~np.isnan(input)].size,
        "count_all": input.size,
        "mean": np.mean(input[~np.isnan(input)]),
        "std": np.std(input[~np.isnan(input)]),
        "min": np.min(input[~np.isnan(input)]),
        "25%": np.percentile(input[~np.isnan(input)], 25),
        "50% (median)": np.median(input[~np.isnan(input)]),
        "75%": np.percentile(input[~np.isnan(input)], 75),
        "max": np.max(input[~np.isnan(input)])
    }
    
    return array_stats

def fake_legend_describe_numpy(input, ax, mask=None):
    
    similarity_interest = input[mask]
    
    similarity_stats_dict = describe_numpy(similarity_interest)
    
    if similarity_stats_dict is None:
        
        fake_legend_stats_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='orange', markersize=5, markeredgecolor='orange') for _ in range(8)]
        
        fake_legend_stats_labels = [
            f"count: {similarity_interest.size}/{input.size}",
            "mean: - ",
            "std: - ",
            "min: - ",
            "25%: - ",
            "50%: - ",
            "75%: - ",
            "max: - "
            ]

        fake_legend = ax.legend(fake_legend_stats_handles, fake_legend_stats_labels, framealpha=0.25, ncol=2, handlelength=0, borderpad=0, labelspacing=0)
        ax.add_artist(fake_legend)
        
    else:
    
        fake_legend_stats_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='orange', markersize=5, markeredgecolor='orange') for _ in range(8)]
        
        fake_legend_stats_labels = [
            f"count: {similarity_interest.size}/{input.size}",
            f"mean: {similarity_stats_dict['mean']:.3f}",
            f"std: {similarity_stats_dict['std']:.3f}",
            f"min: {similarity_stats_dict['min']:.3f}",
            f"25%: {similarity_stats_dict['25%']:.3f}",
            f"50%: {similarity_stats_dict['50% (median)']:.3f}",
            f"75%: {similarity_stats_dict['75%']:.3f}",
            f"max: {similarity_stats_dict['max']:.3f}"
            ]
        
        fake_legend = ax.legend(fake_legend_stats_handles, fake_legend_stats_labels, framealpha=0.25, ncol=2, handlelength=0, borderpad=0, labelspacing=0)
        ax.add_artist(fake_legend)
    
    return fake_legend




if __name__ == '__main__':
    
    #multi_model_macaque_style_comparison()
    #multi_model_macaque_style_comparison_resnet()
    #multi_model_macaque_style_comparison_human()
    
    #multi_model_macaque_style_comparison_a2s_ver2()
    multi_model_macaque_style_comparison_a2s_ver2_temporal()