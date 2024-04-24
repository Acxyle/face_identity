#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:53:33 2023

@author: acxyle

    ...
   
"""

import os
import math
import warnings
import logging
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

import utils_


class FSA_ANOVA():
    """
        FSA: Face-Selectivity-Analysis
        
        the 'Features' contains all processed img fr, not spike train
    """
    
    def __init__(self, root='./Face_Identity VGG16', layers=None, neurons=None, num_classes=50, num_samples=10, alpha=0.01, **kwargs):
        
        self.root = os.path.join(root, 'Features')     # <- folder for feature maps, which should be generated before analysis
        self.dest = os.path.join(root, 'Analysis')    # <- folder for analysis results
        utils_.make_dir(self.dest)
        
        self.dest_ANOVA = os.path.join(self.dest, 'ANOVA')
        utils_.make_dir(self.dest_ANOVA)
        
        self.layers = layers
        self.neurons = neurons
        
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_samples = num_samples
        
        self.model_structure = root.split('/')[-1].split(' ')[-1]     # in current code, the 'root' file should list those information in structural name, arbitrary
    
    
    def __call__(self, **kwargs):
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        # --- 1. 
        self.calculation_ANOVA(**kwargs)
        
        ratio_dict = self.calculation_ANOVA_pct(**kwargs)
        
        # --- 2.
        fig, ax = plt.subplots(figsize=(math.floor(len(self.layers)/1.6), 6))
        
        title = f"Sensitive pct {self.model_structure}"
        
        self.plot_ANOVA_pct(fig, ax, ratio_dict, title=title, plot_bar=True, **kwargs)
        
        fig.savefig(os.path.join(self.dest_ANOVA, f'{title}.svg'), bbox_inches='tight')
        plt.close()
    

    def calculation_ANOVA(self, normalize=True, sort=True, num_workers=-1, **kwargs):
        """
            normalize: if True, normalize the feature map
            sort: if True, sort the featuremap from lexicographic order (pytorch) into natural order
            num_workers:
        """
        
        utils_.formatted_print('Executing calculation_ANOVA')
        
        idces_path = os.path.join(self.dest_ANOVA, 'ANOVA_idces.pkl')
        stats_path = os.path.join(self.dest_ANOVA, 'ANOVA_stats.pkl')
        
        if os.path.exists(idces_path) and os.path.exists(stats_path):
            self.ANOVA_idces = self.load_ANOVA_idces()
            self.ANOVA_stats = self.load_ANOVA_stats()
        
        else:
            self.ANOVA_idces = {}
            self.ANOVA_stats = {}     # <- p_values
            
            for idx, layer in enumerate(self.layers):     # for each layer
    
                feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), normalize=normalize, sort=sort, **kwargs)

                if feature.shape[0] != self.num_classes*self.num_samples or feature.shape[1] != self.neurons[idx]:     # check
                    raise AssertionError('[Coderror] feature.shape[0] ({}) != self.num_classes*self.num_samples ({},{}) or feature.shape[1] ({}) != self.neurons[idx] ({})'.format(feature.shape[0], self.num_classes, self.num_samples, feature.shape[1], self.neurons[idx]))
                
                # ----- joblib
                pl = Parallel(n_jobs=num_workers)(delayed(one_way_ANOVA)(feature[:, i]) for i in tqdm(range(feature.shape[1]), desc=f'ANOVA [{layer}]'))
    
                neuron_idx = np.array([idx for idx, p in enumerate(pl) if p < self.alpha])
            
                self.ANOVA_stats[layer] = pl
                self.ANOVA_idces[layer] = neuron_idx
            
            utils_.dump(self.ANOVA_idces, idces_path)
            utils_.dump(self.ANOVA_stats, stats_path)
            
            utils_.formatted_print('[Codinfo] ANOVA results have been saved in {}'.format(self.dest_ANOVA))
            
            
    def calculation_ANOVA_pct(self, ANOVA_path=None, **kwargs):
        
        ratio_path = os.path.join(self.dest_ANOVA, 'ratio.pkl') if ANOVA_path == None else ANOVA_path
        
        if os.path.exists(ratio_path):
            
            ratio_dict = utils_.load(ratio_path, verbose=False)
            
        else:
            
            self.ANOVA_idces = self.load_ANOVA_idces()
            
            ratio_dict = {layer: self.ANOVA_idces[layer].size/self.neurons[idx]*100 for idx, layer in enumerate(self.layers)}
                  
            utils_.dump(ratio_dict, ratio_path)     
            
        return ratio_dict
            
            
    def plot_ANOVA_pct(self, fig, ax, ratio_dict, title='sensitive ratio', line_color=None, plot_bar=False, **kwargs):      
        """
            ...
        """
        
        utils_.formatted_print('Executing plot_ANOVA_pct...')
        
        # -----
        _, pcts = zip(*ratio_dict.items())
        
        # -----
        if line_color is None:
            line_color = 'red'
        
        if plot_bar:
            colors = color_column(self.layers)
            plot_ANOVA_pct(ax, self.layers, pcts, title=title, bar_colors=colors, line_color=line_color, linewidth=1.5, **kwargs)
        
        else:
            plot_ANOVA_pct(ax, self.layers, pcts, title=title, line_color=line_color, linewidth=1.5, **kwargs)
        
    
    def load_ANOVA_idces(self, ANOVA_idces_path=None):
        if not ANOVA_idces_path:
            ANOVA_idces_path = os.path.join(self.dest_ANOVA, 'ANOVA_idces.pkl')
        return utils_.load(ANOVA_idces_path)
        
    
    def load_ANOVA_stats(self, ANOVA_stats_path=None):
        if not ANOVA_stats_path:
            ANOVA_stats_path = os.path.join(self.dest_ANOVA, 'ANOVA_stats.pkl')
        return utils_.load(ANOVA_stats_path)
    
            
# ----------------------------------------------------------------------------------------------------------------------
def one_way_ANOVA(input, num_classes=50, num_samples=10, **kwargs):
    """
        if all values are 0, this will cause 'nan' F_value and 'nan' p_value, nan values will be filtered in folowing 
        selection with threshold 0.01
    """
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore')
        
        d = list(input.reshape(num_classes, num_samples))
        p = stats.f_oneway(*d)[1]     # [0] for F-value, [1] for p-value

    return p


# ----------------------------------------------------------------------------------------------------------------------
def plot_ANOVA_pct(ax, layers, pcts, title=None, bar_colors=None, line_color=None, linewidth=2.5, label=None, **kwargs):
    
    plt.rcParams.update({'font.size': 22})     
    plt.rcParams.update({"font.family": "Times New Roman"})
    
    if bar_colors is not None:
        ax.bar(layers, pcts, color=bar_colors, width=0.5)
    
    if title is not None:
        ax.set_title(title)
        
    if label == None:
        label='sensitive units'
        
    ax.plot(layers, pcts, color=line_color, linestyle='-', linewidth=linewidth, alpha=1, label=label)
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers, rotation='vertical')
    ax.set_ylabel('percentage (%)')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    ax.legend()
     
    
# ----------------------------------------------------------------------------------------------------------------------
def color_column(layers, constant_colors=False, **kwargs):
    """
        randomly generate colors
    """
    
    layers_t = []
    color = []
    
    for layer in layers:
        layers_t.append(layer.split('_')[0])
    layers_t = list(set(layers_t))

    if not constant_colors:
        for item in range(len(layers_t)):
            color.append((np.random.random(), np.random.random(), np.random.random()))
    else:
        assert len(layers_t) == 5
        color = ['teal', 'red', 'orange', 'lightskyblue', 'tomato']     # ['bn', 'fc', 'avgpool', 'activation', 'conv']
    
    layers_c_dict = {}
    for i in range(len(layers_t)):
        layers_c_dict[layers_t[i]] = color[i]     

    layers_color_list = []
        
    for layer in layers:
        layers_color_list.append(layers_c_dict[layer.split('_')[0]])

    return layers_color_list


# ----------------------------------------------------------------------------------------------------------------------
class FSA_ANOVA_folds(FSA_ANOVA):
    
    def __init__(self, num_folds=5, root=None, **kwargs):
        
        assert 'fold' in root
        
        super().__init__(root=root, **kwargs)
        self.root = root     # 'fold_' rather 'Fratures'
        self.num_folds = num_folds
        
        ...
    
    def __call__(self, **kwargs):
        
        # --- 1. calculation
        raio_dict = self.calculation_ANOVA_pct_folds(**kwargs)
        
        # --- 2. plot
        fig, ax = plt.subplots(figsize=(math.floor(len(self.layers)/1.6), 6))
        
        title=f'Sensitive pct {self.model_structure}'
        self.plot_ANOVA_pct_folds(fig, ax, raio_dict, title=title, **kwargs)
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.dest_ANOVA, f'{title}.svg'))
        
        plt.close()

        
    def calculation_ANOVA_pct_folds(self, ANOVA_path=None, **kwargs):
        
        ratio_dict_path = os.path.join(self.dest_ANOVA, 'ratio.pkl') if ANOVA_path == None else ANOVA_path
        
        if os.path.exists(ratio_dict_path):
            
            ratio_dict = utils_.load(ratio_dict_path)
        
        else:
        
            ratio_dict = [utils_.load(os.path.join(self.root, f"-_Single Models/{self.root.split('/')[-1]}{fold_idx}/Analysis/ANOVA/ratio.pkl"), verbose=False) for fold_idx in np.arange(self.num_folds)]
            ratio_dict = {_: [ratio_dict[__][_] for __ in np.arange(self.num_folds)] for _ in self.layers}
            ratio_dict = {stat: {k: getattr(np, stat)(v) for k, v in ratio_dict.items()} for stat in ['mean', 'std']}

            utils_.dump(ratio_dict, ratio_dict_path)
        
        return ratio_dict
    
    
    def plot_ANOVA_pct_folds(self, fig, ax, ratio_dict, line_color=None, title=None, **kwargs):
        
        # --- init
        layers, pcts = zip(*ratio_dict['mean'].items())
        _, stds = zip(*ratio_dict['std'].items())
        
        pcts, stds = map(np.array, [pcts, stds])

        if line_color is None:
            line_color = 'red'
        
        plot_ANOVA_pct(ax, self.layers, pcts, title=title, line_color=line_color, linewidth=1.5, **kwargs)
        
        ax.fill_between(np.arange(len(self.layers)), pcts-stds, pcts+stds, edgecolor=None, facecolor=utils_.lighten_color(utils_.color_to_hex(line_color)), alpha=0.5)
        
        
   
# ----------------------------------------------------------------------------------------------------------------------
#FIXME --- consider to change the API so it can be properly called by Main_script.py
class FSA_ANOVA_Comparison(FSA_ANOVA_folds):
    """
        obtain each results before call this comparison
        this is far from an automatic script, need to manually change to show a lof of different models
    """

    def __init__(self, **kwargs):
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        # ---
        self.color_pool = ['blue', 'green', 'red', 'purple', 'orange', 'chocolate']     # manually change the pool
        
        ...
        
        
    def __call__(self, roots_and_models, **kwargs):
        
        # ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ratio_dict = {}
        title = []
        
        for idx, (root, model) in enumerate(roots_and_models):
            
            super().__init__(root=roots_and_models[idx][0], **kwargs)     # save the comparison results in the fisrt folder
            self.layers, self.neurons, _ = utils_.get_layers_and_units(roots_and_models[idx][1], target_layers='act')
            
            if 'fold' in root:
                
                ratio_dict = self.calculation_ANOVA_pct_folds(os.path.join(root, 'Analysis/ANOVA/ratio.pkl'))

                _label = root.split('/')[-1].split(' ')[-1].replace('_fold_', '').replace('_CelebA2622', '')
                title.append(_label)
                
                self.plot_ANOVA_pct_folds(fig, ax, ratio_dict, line_color=self.color_pool[idx], label=_label)
                
                ...
                
            else:
                
                ratio_dict = self.calculation_ANOVA_pct(os.path.join(root, 'Analysis/ANOVA/ratio.pkl'))
                
                _label=root.split('/')[-1].split(' ')[-1]
                title.append(_label)
                
                self.plot_ANOVA_pct(fig, ax, ratio_dict, line_color=self.color_pool[idx], label=_label)
                ...


        #ax.set_title(title:=' v.s '.join(title))
        ax.set_title(title:='Sensitive pct ANN v.s SNN')
        
        # --- setting
        ...
        # ---
        
        fig.tight_layout()
        fig.savefig(os.path.join(roots_and_models[0][0], f'Analysis/ANOVA/Comparison {title}.svg'))
        
        plt.close()
    
   
# ======================================================================================================================
if __name__ == "__main__":
    
    root_dir = '/home/acxyle-workstation/Downloads'
    
    # -----
    layers, neurons, shapes = utils_.get_layers_and_units('vgg16_bn', target_layers='act')
    
    # ---
    FSA_ANOVA(os.path.join(root_dir, 'Face Identity VGG16bn_VGGFace'), layers=layers, neurons=neurons)()
    
    # ---
    #FSA_ANOVA_folds = FSA_ANOVA_folds(num_folds=5, root=os.path.join(root_dir, 'Face Identity VGG16bn_fold_'), layers=layers, neurons=neurons)
    #FSA_ANOVA_folds()
    
    # -----
# =============================================================================
#     roots_models = [
#         (os.path.join(root_dir, 'Face Identity VGG16_fold_'), 'vgg16'),
#         (os.path.join(root_dir, 'Face Identity VGG16bn_fold_'), 'vgg16_bn'),
#         (os.path.join(root_dir, 'Face Identity SpikingVGG16bn_IF_T4_CelebA2622_fold_'), 'spiking_vgg16_bn'),
#         (os.path.join(root_dir, 'Face Identity SpikingVGG16bn_IF_T16_CelebA2622_fold_'), 'spiking_vgg16_bn'),
#         (os.path.join(root_dir, 'Face Identity SpikingVGG16bn_LIF_T4_CelebA2622_fold_'), 'spiking_vgg16_bn'),
#         (os.path.join(root_dir, 'Face Identity SpikingVGG16bn_LIF_T16_CelebA2622_fold_'), 'spiking_vgg16_bn')
#         ]
#     FSA_ANOVA_Comparison = FSA_ANOVA_Comparison(roots_models)
# =============================================================================
    
    
