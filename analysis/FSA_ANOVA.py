#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:53:33 2023

@author: acxyle

   
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
        
        the 'Features' contains all processed img fr, not spike
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


    def calculation_ANOVA(self, normalize=True, sort=True, num_workers=-1, **kwargs):
        """
            normalize: if True, normalize the feature map
            sort: if True, sort the featuremap from lexicographic order (pytorch) into natural order
            num_workers:
        """
        
        utils_._print('Executing calculation_ANOVA')
        
        idces_path = os.path.join(self.dest_ANOVA, 'ANOVA_idces.pkl')
        stats_path = os.path.join(self.dest_ANOVA, 'ANOVA_stats.pkl')
        
        if os.path.exists(idces_path) and os.path.exists(stats_path):
            self.ANOVA_idces = utils_.load(idces_path)
            self.ANOVA_stats = utils_.load(stats_path)
        
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
            
            utils_._print('[Codinfo] ANOVA results have been saved in {}'.format(self.dest_ANOVA))
            
            
    def calculate_sensitive_pct(self, ):
        
        ratio_path = os.path.join(self.dest_ANOVA, 'ratio.pkl')
        
        if os.path.exists(ratio_path):
            
            ratio_dict = utils_.load(ratio_path, verbose=False)
            
        else:
            
            ratio_dict = {layer: self.ANOVA_idces[layer].size/self.neurons[idx]*100 for idx, layer in enumerate(self.layers)}
                  
            utils_.dump(ratio_dict, ratio_path)     
            
        return ratio_dict
            
            
    def plot_ANOVA_pct(self, title='sensitive ratio', **kwargs):      
        """
            ...
        """
        
        utils_._print('Executing plot_ANOVA_pct...')
        
        plt.rcParams.update({'font.size': 22})     
        plt.rcParams.update({"font.family": "Times New Roman"})
 
        if not hasattr(self, 'ANOVA_idces'):
            self.ANOVA_idces = utils_.load(os.path.join(self.dest_ANOVA, 'ANOVA_idces.pkl'))

        # -----
        ratio_dict = self.calculate_sensitive_pct()
        layers_color_list = color_column(self.layers)
        
        x = list(ratio_dict.keys())
        y = list(ratio_dict.values())
        
        # -----
        fig, ax = plt.subplots(figsize=(math.floor(len(self.layers)/1.6), 6))
        
        ax.plot(x, y, color='red', linestyle='-', linewidth=2.5, alpha=1, label='sensitive units')
        ax.bar(x, y, color=layers_color_list, width=0.5)
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x, rotation='vertical')
        ax.set_ylabel('percentage (%)')
        ax.set_title(f'{title} - {self.model_structure}')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        ax.legend()
        
        fig.savefig(os.path.join(self.dest_ANOVA, f'{title}.svg'), bbox_inches='tight')
        plt.close()
            
    
    # FIXME --- 
    # ------------------------------------------------------------------------------------------------------------------
    def plot_ANOVA_pct_multi_models(self, ):
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        ANOVA_save_root = os.path.join(self.model_root, 'ANOVA')
        utils_.make_dir(ANOVA_save_root)
        
        def _ANOVA_pct_collect():
            
            ANOVA_pct_folds_path = os.path.join(ANOVA_save_root, 'ANOVA_folds_array.pkl')
            
            if os.path.exists(ANOVA_pct_folds_path):
                
                ANOVA_folds_array = utils_.load(ANOVA_pct_folds_path)
            
            else:
            
                ANOVA_folds = {}
        
                for fold_idx in np.arange(1, self.num_fold):
                    
                    root = os.path.join(self.model_root+str(fold_idx))
                    
                    ANOVA_folds[fold_idx] = utils_.load(os.path.join(root, 'Analysis', 'ANOVA', 'ratio.pkl'), verbose=False)
                    
                ANOVA_folds_array = np.array([np.array(_) for _ in list(ANOVA_folds.values())])     # (num_folds, num_layers)
                
                utils_.dump(ANOVA_folds_array, ANOVA_pct_folds_path)
            
            return ANOVA_folds_array
        
        def _ANOVA_pct_plot(ax, layers, ANOVA_folds_array, title):
            
            folds_mean = np.mean(ANOVA_folds_array, axis=0)
            folds_std = np.std(ANOVA_folds_array, axis=0)  
            
            ax.fill_between(np.arange(len(layers)), folds_mean-folds_std, folds_mean+folds_std, edgecolor=None, facecolor='skyblue', alpha=0.75)
            ax.plot(np.arange(len(layers)), folds_mean, color='blue', linewidth=0.5)
            ax.set_xticks(np.arange(len(layers)), layers, rotation='vertical')
            ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
            ax.set_title(f'{self.model_structure} {title}')
            
            plt.tight_layout()
            fig.savefig(os.path.join(ANOVA_save_root, f'{title}_folds.svg'))
            plt.close()
        
        # ---
        ANOVA_folds_array = _ANOVA_pct_collect()
        
        # ---
        fig, ax = plt.subplots(figsize=(18,10))
        _ANOVA_pct_plot(ax, self.layers, ANOVA_folds_array, 'ANOVA_pct')

        # ---
        act_idx, act_layers, act_neurons, _ = utils_.activation_function(self.model_structure, self.layers, self.neurons)
        ANOVA_folds_array = ANOVA_folds_array[:, act_idx]
        
        fig, ax = plt.subplots(figsize=(10,10))
        _ANOVA_pct_plot(ax, act_layers, ANOVA_folds_array, 'ANOVA_pct_act')
        
        
    # FIXME ----- test version ----- need to fix, move the function in Main_analyzer to here
    def plot_ANOVA_model_comparison(self, comparing_models_list):
        """
            in this function, the default model is which underlies this class, and the input is a list of comparing models
        """
        print('[Codinfo] Executing selectivity_neuron_ANOVA_comparison_plot...')
        
        plt.rcParams.update({'font.size': 22})
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        # ----- operation to acquire the ratio of current model
        fig_folder = os.path.join(self.dest, 'Figures')
        utils_.make_dir(fig_folder)
        
        ratio_path = os.path.join(fig_folder, 'ratio.pkl')
        
        layers_color_list = color_column(self.layers)
        
        if os.path.exists(ratio_path):
            ratio = utils_.load(ratio_path)
        else:
            raise RuntimeError('[Coderror] no ratio file detected for current model, plase run plot_ANOVA_pct() first.')
        
        # -----
        def _load_comparing_data(comparing_model):

            ratio_comparing = utils_.load(f'/media/acxyle/Data/ChromeDownload/{comparing_model}/Analysis/ANOVA/Figures/ratio.pkl')
            
            if 'baseline' not in comparing_model.lower():
                comparing_model_config_list = comparing_model.split('_')[1:-2]     # for SNN: 1_2_3_4 - structure_neuron_surrogate_T, for ANN: 1_2 - structure_activation
            else:
                comparing_model_config_list = [comparing_model.split('_')[1]+'(baseline)', 'ReLU']
                
            return ratio_comparing, comparing_model_config_list
        
        # -----
        # 1. for all calculation
        title = 'Sensitive Unit Percentage(s) Comparision'
        fig, ax = plt.subplots(figsize=(math.floor(len(self.layers)/1.6), 10), dpi=100)
        
        # plot fundamental model on the canvas
        self.plot_single_figure_VS(ax=ax, layers=self.layers, ratio=ratio, layers_color_list=layers_color_list, model_structure=self.model_structure,
                                   linewidth=2.5, alpha=1, linestyle='-')
        
        # plot comparing models on the canvas
        for comparing_model in comparing_models_list:     # for each comparing model
            ratio_comparing, comparing_model_config_list = _load_comparing_data(comparing_model)
            comparing_model_structure = comparing_model_config_list[0]
            
            layers, _, _ = utils_.get_layers_and_neurons(comparing_model_structure, self.num_classes)
            layer_idx = [idx for idx, _ in enumerate(self.layers) if _ in layers]
            layers_color_list_comparing = [layers_color_list[_] for _ in layer_idx]
            
            print(f'[Codinfo] adding Comparing models: {comparing_model_structure}')
            
            self.plot_single_figure_VS(ax=ax, layers=layers, ratio=ratio_comparing, layers_color_list=layers_color_list_comparing, model_structure=comparing_model_structure)
        
        ax.set_xticklabels(self.layers, rotation='vertical')
        ax.set_ylabel('percentage (%)')
        ax.set_title(f'{title}')
        ax.legend()
        
        fig.savefig(os.path.join(fig_folder, f'{title}.svg'), bbox_inches='tight')

        plt.close()
        
        # 2. for imaginary neurons only ([question] why contains pool layers?) - this version is for ANN because 'act', SNN should be 'neuron'
        title = 'Sensitive Unit Percentage(s) Comparision - Activation'
        
        # ----- for fundamental model
        neuron_layer = [[idx, layer] for idx, layer in enumerate(self.layers) if 'neuron' in layer or 'pool' in layer or 'fc_3' in layer]
        neuron_layer_idx = [_[0] for _ in neuron_layer]
        neuron_layer = [_[1] for _ in neuron_layer]
        
        ratio_neuron = [ratio[_] for _ in neuron_layer_idx]
        layers_color_list_neuron = [layers_color_list[_] for _ in neuron_layer_idx]
        
        fig, ax = plt.subplots(figsize=(math.floor(len(neuron_layer)/1.6), 10), dpi=100)
        
        # plot fundamental model on the canvas
        self.plot_single_figure_VS(ax=ax, layers=neuron_layer, ratio=ratio_neuron, layers_color_list=layers_color_list_neuron, 
                                   model_structure=self.model_structure, linewidth=2.5, alpha=1, linestyle='-')
                
        # plot comparing models on the canvas
        for comparing_model in comparing_models_list:     # for each comparing model
            ratio_comparing, comparing_model_config_list = _load_comparing_data(comparing_model)
            comparing_model_structure = comparing_model_config_list[0]
            
            layers, _, _ = utils_.get_layers_and_neurons(comparing_model_structure, self.num_classes)
            layers = [[idx,_] for idx, _ in enumerate(layers) if 'neuron' in _ or 'pool' in _ or 'fc_3' in _]     # <- select target layers
            layers_idx = [_[0] for _ in layers]
            layers = [_[1] for _ in layers]
            ratio_comparing = [ratio_comparing[_] for _ in layers_idx]
            
            layer_idx = [idx for idx, _ in enumerate(self.layers) if _ in layers]
            layers_color_list_comparing = [layers_color_list[_] for _ in layer_idx]     # <- determine the colors
            
            print(f'[Codinfo] adding Comparing models: {comparing_model_structure}')
            
            self.plot_single_figure_VS(ax=ax, layers=layers, ratio=ratio_comparing, layers_color_list=layers_color_list_comparing, model_structure=comparing_model_structure)
        
        ax.set_xticklabels(layers, rotation='vertical')
        ax.set_ylabel('percentage (%)')
        ax.set_title(f'{title}')
        ax.legend()
        
        fig.savefig(os.path.join(fig_folder, f'{title}.png'), bbox_inches='tight')
        
        plt.close()
        

    # FIXME
    def plot_single_figure_VS(self, **kwargs):
        """
            every time plot one model      
        """
        ax = kwargs['ax']
        
        # --- default setting for comparing lines
        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = 1.5
        if 'alpha' not in kwargs.keys():
            kwargs['alpha'] = 0.5
        if 'linestyle' not in kwargs.keys():
            kwargs['linestyle'] = '--'
            
        # --- default setting for baseline
        if 'baseline' in kwargs['model_structure']:
            kwargs['linewidth'] = 2.0
            kwargs['alpha'] = 0.75
            kwargs['linestyle'] = '--'
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            ax.plot(kwargs['layers'], kwargs['ratio'], linestyle=kwargs['linestyle'], linewidth=kwargs['linewidth'], alpha=kwargs['alpha'], label=kwargs['model_structure'])
            ax.bar(kwargs['layers'], kwargs['ratio'], color=kwargs['layers_color_list'], width=0.5, alpha=kwargs['alpha'])
            
            
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


# ======================================================================================================================
if __name__ == "__main__":
    
    root_dir = '/home/acxyle-workstation/Downloads'
    
    layers, neurons, shapes = utils_.get_layers_and_units('vgg16', target_layers='act')
    
    FSA_ANOVA = FSA_ANOVA(os.path.join(root_dir, 'Face Identity Baseline'), layers=layers, neurons=neurons)
         
    FSA_ANOVA.calculation_ANOVA()
    FSA_ANOVA.plot_ANOVA_pct()
