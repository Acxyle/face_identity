#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:53:33 2023

@author: acxyle

    #TODO
    build a compact script for (1) ANOVA analysis; (2) .pkl and .csv store; (3) plot; (4) comparisons
    
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

class ANOVA_analyzer():
    """
        in the update of Sept 6, 2023, this code receives the entire folder path as the input rather than the 'Features' path
    """
    def __init__(self, root='.../Face_Identity VGG16/', 
                 alpha=0.01, num_classes=50, num_samples=10, layers=None, neurons=None):
        
        assert root[-1] != '/', f"[Codinfo] root {root} should not end with '/'"
        
        self.root = os.path.join(root, 'Features/')     # <- folder for feature maps, which should be generated before analysis
        self.dest = os.path.join(root, 'Analysis/')    # <- folder for analysis results
        utils_.make_dir(self.dest)
        
        self.dest_ANOVA = os.path.join(self.dest, 'ANOVA')
        utils_.make_dir(self.dest_ANOVA)
        
        self.layers = layers
        self.neurons = neurons
        
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_samples = num_samples
        
        self.model_structure = root.split('/')[-1].split(' ')[-1]     # in current code, the 'root' file should list those information in structural name, arbitrary

        # some tests to check the saved files are valid and matchable with self.layers and self.neurons
        if self.layers == None or self.neurons == None:
            raise RuntimeError('[Coderror] invalid layers and/or neurons')
            
        if not os.path.exists(self.root):
            raise RuntimeWarning(f'[Codwarning] can not find the root of [{self.root}]')
        elif os.listdir(self.root) == []:
            raise RuntimeWarning('[Codwarning] the path of feature directory is valid but found no file')
        
        #if set([f.split('.')[0] for f in os.listdir(self.root)]) != set(self.layers):     # [notice] can mute this message when segmental test
        #    raise AssertionError('[Coderror] the saved .pkl files must be exactly the same with attribute self.layers')
        
        
    def calculation_ANOVA(self, ):
        
        print('[Codinfo] Executing calculation_ANOVA...')
        num_workers = int(os.cpu_count()/2)
        print(f'[Codinfo] Executing parallel computation with num_workers={num_workers}')
        
        idces_path = os.path.join(self.dest_ANOVA, 'ANOVA_idces.pkl')
        stats_path = os.path.join(self.dest_ANOVA, 'ANOVA_stats.pkl')
        
        if os.path.exists(idces_path) and os.path.exists(stats_path):
            self.ANOVA_idces = utils_.load(idces_path, verbose=True)
            self.ANOVA_stats = utils_.load(stats_path, verbose=True)
        
        else:
            ANOVA_idces = {}
            ANOVA_stats = {}     # <- p_values
            
            for idx_l, layer in enumerate(self.layers):     # for each layer
    
                feature = utils_.load(os.path.join(self.root, layer+'.pkl'), verbose=False)
                    
                pl = np.zeros(feature.shape[1])       # p_value_list for all neuron
                    
                if feature.shape[0] != self.num_classes*self.num_samples or feature.shape[1] != self.neurons[idx_l]:     #  feature check
                    raise AssertionError('[Coderror] feature.shape[0] ({}) != self.num_classes*self.num_samples ({},{}) or feature.shape[1] ({}) != self.neurons[idx_l] ({})'.format(feature.shape[0], self.num_classes, self.num_samples, feature.shape[1], self.neurons[idx_l]))
                
                # ----- parallel computing by joblib
                pl = Parallel(n_jobs=num_workers)(delayed(one_way_ANOVA)(feature[:, i]) for i in tqdm(range(feature.shape[1]), desc=f'[{layer}] ANOVA'))
    
                neuron_idx = np.array([idx for idx, p in enumerate(pl) if p < self.alpha])     # p_value_idx_list for neurons that satisfy the threshold
                
                ANOVA_stats.update({layer: pl})
                ANOVA_idces.update({layer: neuron_idx})
            
            utils_.dump(ANOVA_idces, idces_path)
            utils_.dump(ANOVA_stats, stats_path)
            
            print('[Codinfo] Selectivity_Neuron_ANOVA Calculation Done.')
            print('[Codinfo] pvalue.csv and neuron_idx.pkl files have been saved in {}'.format(self.dest_ANOVA))
            
            self.ANOVA_idces = ANOVA_idces
            self.ANOVA_stats = ANOVA_stats


    def plot_ANOVA_pct(self):      
        
        print('[Codinfo] Executing plot_ANOVA_pct...')
        
        plt.rcParams.update({'font.size': 22})     
        plt.rcParams.update({"font.family": "Times New Roman"})

        fig_folder = os.path.join(self.dest_ANOVA, 'Figures/')
        utils_.make_dir(fig_folder)
        
        ratio_path = os.path.join(self.dest_ANOVA, 'ratio.pkl')
        
        #if not hasattr(self, 'ANOVA_stats'):
        #    self.ANOVA_stats = utils_.load(os.path.join(self.dest_ANOVA, 'ANOVA_stats.pkl'))
        
        if not hasattr(self, 'ANOVA_idces'):
            self.ANOVA_idces = utils_.load(os.path.join(self.dest_ANOVA, 'ANOVA_idces.pkl'))
        
        
        def _plot_single_figure(layers, ratio, layers_color_list, title):

            logging.getLogger('matplotlib').setLevel(logging.ERROR)
            
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                
                fig, ax = plt.subplots(figsize=(math.floor(len(layers)/1.6), 10))  
                ax.plot(layers, ratio, color='red', linestyle='-', linewidth=2.5, alpha=1, label=self.model_structure)
                ax.bar(layers, ratio, color=layers_color_list, width=0.5)
                ax.set_xticklabels(labels=layers, rotation='vertical')
                ax.set_ylabel('percentage (%)')
                ax.set_title(f'{title} - {self.model_structure}')
                ax.legend()
                fig.savefig(os.path.join(fig_folder, title+'.png'), bbox_inches='tight')
                fig.savefig(os.path.join(fig_folder, title+'.eps'), bbox_inches='tight')
                plt.close()
        
        # -----
        if os.path.exists(ratio_path):
            
            ratio = utils_.load(ratio_path)
            
        else:
            
            count_dict = {layer:0 for layer in self.layers}

            for layer in self.layers:
                sig_neuron_idx = self.ANOVA_idces[layer]
                value = len(sig_neuron_idx)
                count_dict[layer]=value
                  
            ratio = [round(a/b * 100) for a, b in zip(list(count_dict.values()), self.neurons)]  # ratio = unit_passed_ANOVA/all_unit
            utils_.dump(ratio, ratio_path)     # save the percentages for other plot
        
        layers_color_list = color_column(self.layers)
        
        # --- all
        _plot_single_figure(self.layers, ratio, layers_color_list, 'sensitive unit ratio')
        
        # --- act
        neuron_layer_idx, neuron_layer, _ = utils_.activation_function(self.model_structure, self.layers)
        
        ratio_neuron = [ratio[_] for _ in neuron_layer_idx]
        layers_color_list_neuron = [layers_color_list[_] for _ in neuron_layer_idx]
                
        _plot_single_figure(neuron_layer, ratio_neuron, layers_color_list_neuron, 'sensitive unit ratio neuron')
                                
        
    # FIXME ----- test version, need to remove the use of **kwargs
    def plot_ANOVA_model_comparison(self, comparing_models_list):
        """
            in this function, the default model is which underlies this class, and the input is a list of comparing models
        """
        print('[Codinfo] Executing selectivity_neuron_ANOVA_comparison_plot...')
        
        plt.rcParams.update({'font.size': 22})
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        # ----- operation to acquire the ratio of current model
        fig_folder = os.path.join(self.dest, 'Figures/')
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
        
        fig.savefig(os.path.join(fig_folder, title+'.png'), bbox_inches='tight')
        fig.savefig(os.path.join(fig_folder, title+'.eps'), bbox_inches='tight')    
        #fig.savefig(os.path.join(fig_folder, title+'.svg'), bbox_inches='tight', transparent=True)
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
        
        fig.savefig(os.path.join(fig_folder, title+'.png'), bbox_inches='tight')
        fig.savefig(os.path.join(fig_folder, title+'.eps'), bbox_inches='tight')     # no transparency
        #fig.savefig(os.path.join(fig_folder, title+'.svg'), bbox_inches='tight', transparent=True)
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
def one_way_ANOVA(feature_strip, num_classes=50, num_samples=10):
    """
        if all units have responses 0, so will have 50 groups each has 10 0 values, this will cause 'nan' F_value and 'nan' p_value
        nan values will be filtered in folowing selection with threshold 0.01
    """
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore')
        
        d = [feature_strip[i*num_samples: (i+1)*num_samples] for i in range(num_classes)]
        p = stats.f_oneway(*d)[1]     # [0] for F-value, [1] for p-value

    return p


# ----------------------------------------------------------------------------------------------------------------------
def color_column(layers, constant_colors=False):
    """
        randomly generated colors
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


if __name__ == "__main__":
    
    model_name = 'vgg16_bn'
    
    model_ = vgg.__dict__[model_name](num_classes=50)
    layers, neurons, shapes = utils_.generate_vgg_layers_list_ann(model_, model_name)
    
    analyzer = ANOVA_analyzer(root='/home/acxyle-workstation/Downloads/Face Identity SpikingVGG16bn_LIF_T4_vggface/', 
                              alpha=0.01, num_classes=50, num_samples=10, layers=layers, neurons=neurons)
    
    analyzer.calculation_ANOVA()
    analyzer.plot_ANOVA_pct()

    #analyzer.plot_ANOVA_model_comparison(['Identity_VGG16bn_ReLU_CelebA2622_Neuron'])