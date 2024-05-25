#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:06:15 2024

@author: acxyle-workstation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats

import sys
sys.path.append('../')
import utils_

from FSA_RSA import RSA_Monkey_folds, RSA_Human_folds

from matplotlib.lines import Line2D

from scipy.interpolate import interp1d, RectBivariateSpline


from FSA_RSA import plot_RSA, plot_RSA_temporal


plt.rcParams.update({"font.family": "Times New Roman"})
plt.rcParams.update({'font.size': 16})

def _interp_2d(input, num_interp_x=153):
    """
        assume the temporal dimension is constant, only change the layer dimension
    """
    x = np.arange(input.shape[0])
    y = np.arange(input.shape[1])
    
    f = RectBivariateSpline(x, y, input)
    
    x_new = np.linspace(0, input.shape[0]-1, num_interp_x)
    
    output = f(x_new, y)
    
    return output[:(num_interp_x-1), :]
    

# --- interplot
def _interp(input, num_interp=153):
    
    y = input
    x = np.arange(len(input))
    
    f = interp1d(x, y)
    
    return f(np.linspace(0, len(input)-1, num_interp))[:(num_interp-1)]     # ignore the classification head



def collect_one_model_RSA_results(FSA_dict, **kwargs):
    
    _, layers, neurons, _ = utils_.get_layers_and_units(FSA_dict['model'], 'act')
    root = os.path.join(FSA_dict['root'], FSA_dict['dir'], f"FSA {FSA_dict['config']}")

    RSA_human_ = RSA_Human_folds(root=root, layers=layers, neurons=neurons)
    
    return RSA_human_.collect_RSA_Similarity_folds(**kwargs)
    
    
ann_dict_monkey = {}
ann_dict_human = {}

snn_dict_monkey = {}
snn_dict_human = {}

ann_FSA_dict = {
    'root': '/home/acxyle-workstation/Downloads/FSA',
    'dir': 'VGG/VGG',
    'config': 'VGG16bn_C2k_fold_',
    'model':'vgg16_bn',
        }

#ann_dict_human = collect_one_model_RSA_results(ann_FSA_dict)

ann_dict_human = {}
used_unit_types = ['qualified', 'sensitive', 'non_sensitive', 'selective', 'strong_selective', 'weak_selective', 'non_selective']

for _ in used_unit_types:
    
     tmp = collect_one_model_RSA_results(ann_FSA_dict, used_unit_type=_, used_id_num=50)
     
     tmp = np.mean([v['similarity_temporal'] for k,v in tmp.items()], axis=0)
     
     ann_dict_human[_] = tmp

vmax = np.nanmax([v for k,v in ann_dict_human.items()])

vmin = np.nanmin([v for k,v in ann_dict_human.items()])


snn_FSA_dict = {
    'root': '/home/acxyle-workstation/Downloads/FSA',
    'dir': 'VGG/SpikingVGG',
    'config': 'SpikingVGG16bn_IF_ATan_T4_C2k_fold_',
    'model':'spiking_vgg16_bn',
        }


snn_dict_human = {}

for _ in used_unit_types:
    
     tmp = collect_one_model_RSA_results(snn_FSA_dict, used_unit_type=_, used_id_num=50)
     
     tmp = np.mean([v['similarity_temporal'] for k,v in tmp.items()], axis=0)
     
     snn_dict_human[_] = tmp


snn_dict_human = collect_one_model_RSA_results(snn_FSA_dict)
    
    

# ----- monkey
ann_monkey_record_bank = {}
snn_monkey_record_bank = {}

for model_depth in [18, 50, 101, 152]:
    for fold_idx in range(5):
    
        ann_monkey_record_bank.update(
            {f'resnet_{model_depth}_fold_{fold_idx}': ann_dict_monkey[model_depth][fold_idx]['similarity']}
            )
        
        snn_monkey_record_bank.update(
            {f'sew_resnet_{model_depth}_fold_{fold_idx}': snn_dict_monkey[model_depth][fold_idx]['similarity']}
            )

ann_monkey_record_bank = {k: _interp(v) for k,v in ann_monkey_record_bank.items()}
snn_monkey_record_bank = {k: _interp(v) for k,v in snn_monkey_record_bank.items()}


# ----- human
ann_human_record_bank = {}
snn_human_record_bank = {}

for model_depth in [18, 50, 101, 152]:
    for fold_idx in range(5):
    
        ann_human_record_bank.update(
            {f'resnet_{model_depth}_fold_{fold_idx}': ann_dict_human[model_depth][fold_idx]['similarity']}
            )
        
        snn_human_record_bank.update(
            {f'sew_resnet_{model_depth}_fold_{fold_idx}': snn_dict_human[model_depth][fold_idx]['similarity']}
            )

ann_human_record_bank = {k: _interp(v) for k,v in ann_human_record_bank.items()}
snn_human_record_bank = {k: _interp(v) for k,v in snn_human_record_bank.items()}


# -----    
for primate in ['monkey', 'human']:
    for network_type in ['ann', 'snn']:
        for model_depth in [18, 50, 101, 152]:
            exec(f"{network_type}_{primate}_record_bank_{model_depth} = [v for k, v in {network_type}_{primate}_record_bank.items() if '{model_depth}' in k]")
        

def plot_nn_record_bank_comparison(ax, ann_record_bank, snn_record_bank, alpha=0.2, ylim=None, smooth=True):
    
    for v in ann_record_bank:
        ax.plot(v, color='orange', alpha=alpha)
        if smooth:
            ax.plot(scipy.ndimage.gaussian_filter(np.mean(ann_record_bank, axis=0), sigma=1), color='orange', linewidth=3)
        else:
            ax.plot(np.mean(ann_record_bank, axis=0), color='orange', linewidth=3)
    
    for v in snn_record_bank:
        ax.plot(v, color='blue', alpha=alpha)
        if smooth:
            ax.plot(scipy.ndimage.gaussian_filter(np.mean(snn_record_bank, axis=0), sigma=1), color='blue', linewidth=3)
        else:
            ax.plot(np.mean(snn_record_bank, axis=0), color='blue', linewidth=3)
        
    ax.set_ylim(ylim)
    ax.set_xticks([0, 152])
    ax.set_xticklabels([0, 1])


# -----
fig, ax = plt.subplots(5,2,figsize=(10, 15))

ax[0, 0].set_title('Macaque', fontsize=30)
ax[0, 1].set_title('Human', fontsize=30)

for idx, model_depth in enumerate([18, 50, 101, 152]):  
    
    exec(f"plot_nn_record_bank_comparison(ax[idx, 0], ann_monkey_record_bank_{model_depth}, snn_monkey_record_bank_{model_depth}, ylim=[0,0.7])")
    ax[idx, 0].set_ylabel(model_depth, fontsize=30, color='teal', rotation=0)
    ax[idx, 0].yaxis.set_label_coords(-0.3, 0.8)
    ax[idx, 0].yaxis.labelpad = -10
    
    exec(f"plot_nn_record_bank_comparison(ax[idx, 1], ann_human_record_bank_{model_depth}, snn_human_record_bank_{model_depth}, ylim=[-0.1, 0.42])")

plot_nn_record_bank_comparison(ax[4, 0], list(ann_monkey_record_bank.values()), list(snn_monkey_record_bank.values()), alpha=0.05, ylim=[0,0.7])
ax[4, 0].set_ylabel('All', fontsize=30, color='teal', rotation=0)
ax[4, 0].yaxis.set_label_coords(-0.3, 0.8)

plot_nn_record_bank_comparison(ax[4, 1], list(ann_human_record_bank.values()), list(snn_human_record_bank.values()), alpha=0.05, ylim=[-0.1, 0.42])

legend_lines = [Line2D([0], [0], color='blue', lw=4),  
                Line2D([0], [0], color='orange', lw=4)]

fig.legend(legend_lines, ['SEWResnet', 'Resnet'], loc='lower center', bbox_to_anchor=(0.5, 0.05), fancybox=True, shadow=True, ncol=2)

#plt.tight_layout()
plt.savefig('RSA_resnet_depth_comparison.svg', bbox_inches='tight')