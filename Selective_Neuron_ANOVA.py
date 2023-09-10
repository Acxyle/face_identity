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
import pickle
import warnings
import logging
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt

import concurrent.futures
from joblib import Parallel, delayed

import spiking_vgg, spiking_resnet, sew_resnet
from spikingjelly.activation_based import surrogate, neuron, functional

import vgg, resnet
import utils_

class ANOVA_analyzer():
    """
        in the update of Sept 6, 2023, this code receives the entire folder path as the input rather than the 'Features' path
    """
    def __init__(self, root='.../Face_Identity VGG16/', 
                 alpha=0.01, num_classes=50, num_samples=10, layers=None, neurons=None):
        
        self.root = os.path.join(root, 'Features/')     # <- folder for feature maps, which should be generated before analysis
        self.dest = '/'.join([*root.split('/')[:-1], 'Analysis'])     # <- folder for analysis results
        utils_.make_dir(self.dest)
        
        self.dest_ANOVA = os.path.join(self.dest, 'ANOVA')
        utils_.make_dir(self.dest_ANOVA)
        
        self.layers = layers
        self.neurons = neurons
        
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_samples = num_samples
        
        self.model_structure = root.split('/')[-2].split(' ')[1]     # in current code, the 'root' file should list those information in structural name, arbitrary

        # some tests to check the saved files are valid and matchable with self.layers and self.neurons
        if self.layers == None or self.neurons == None:
            raise RuntimeError('[Coderror] invalid layers and/or neurons')
            
        if not os.path.exists(self.root):
            raise RuntimeWarning(f'[Codwarning] can not find the root of [{self.root}]')
        elif os.listdir(self.root) == []:
            raise RuntimeWarning('[Codwarning] the path of feature directory is valid but found no file')
        
        #if set([f.split('.')[0] for f in os.listdir(self.root)]) != set(self.layers):     # [notice] can mute this message when segmental test
        #    raise AssertionError('[Coderror] the saved .pkl files must be exactly the same with attribute self.layers')
        
    def selectivity_neuron_ANOVA(self, ):
        print('[Codinfo] Executing selectivity_neuron_ANOVA...')
        num_workers = int(os.cpu_count()/2)
        print(f'[Codinfo] Executing parallel computation with num_workers={num_workers}')
        
        idces_path = os.path.join(self.dest_ANOVA, 'ANOVA_idces.pkl')
        stats_path = os.path.join(self.dest_ANOVA, 'ANOVA_stats.pkl')
        
        if os.path.exists(idces_path) and os.path.exists(stats_path):
            self.ANOVA_idces = utils_.pickle_load(idces_path)
            self.ANOVA_stats = utils_.pickle_load(stats_path)
        
        else:
            ANOVA_idces = {}
            ANOVA_stats = {}
            
            for idx_l, layer in enumerate(self.layers):     # for each layer
    
                with open(os.path.join(self.root, layer+'.pkl'), 'rb') as pkl:
                    feature = pickle.load(pkl)
                pl = np.zeros(feature.shape[1])       # p_value_list for all neuron
                    
                if feature.shape[0] != self.num_classes*self.num_samples or feature.shape[1] != self.neurons[idx_l]:     #  feature check
                    raise AssertionError('[Coderror] feature.shape[0] ({}) != self.num_classes*self.num_samples ({},{}) or feature.shape[1] ({}) != self.neurons[idx_l] ({})'.format(
                        feature.shape[0], self.num_classes, self.num_samples, feature.shape[1], self.neurons[idx_l]))
                
                # ----- parallel computing by joblib
                pl = Parallel(n_jobs=num_workers)(delayed(self.one_way_ANOVA)(feature, i) for i in tqdm(range(feature.shape[1]), desc=f'[{layer}] ANOVA'))
    
                neuron_idx = np.array([idx for idx, p in enumerate(pl) if p < self.alpha])     # p_value_idx_list for neurons that satisfy the threshold
                
                ANOVA_stats.update({layer: pl})
                ANOVA_idces.update({layer: neuron_idx})
            
            utils_.pickle_dump(idces_path, ANOVA_idces)
            utils_.pickle_dump(stats_path, ANOVA_stats)
            
            print('[Codinfo] Selectivity_Neuron_ANOVA Calculation Done.')
            print('[Codinfo] pvalue.csv and neuron_idx.pkl files have been saved in {}'.format(self.dest_ANOVA))
            
            self.ANOVA_idces = ANOVA_idces
            self.ANOVA_stats = ANOVA_stats
        
    # define ANOVA for parallel calculation
    def one_way_ANOVA(self, feature, i):
        """
            if all units have responses 0, so will have 50 groups each has 10 0 values, this will cause 'nan' F_value and 'nan' p_value
            [following question] how to deal with the unit with nan statistic values?
        """
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            neuron_ = feature[:, i]
            d = [neuron_[i*self.num_samples: i*self.num_samples+self.num_samples] for i in range(self.num_classes)]
            p = stats.f_oneway(*d)[1]  # [0] for F-value, [1] for p-value

        return p
    
    def selectivity_neuron_ANOVA_plot(self):      # this plot rely on both Local initialization and the Saved .csv files
        print('[Codinfo] Executing selectivity_neuron_ANOVA_plot...')
        print('{Codwarning] This function plots from signeuronIdx.csv')
        
        fig_folder = os.path.join(self.dest_ANOVA, 'Figures/')
        utils_.make_dir(fig_folder)
        
        ratio_path = os.path.join(fig_folder, 'ratio.pkl')
        if os.path.exists(ratio_path):
            ratio = utils_.pickle_load(ratio_path)
        else:
            count_dict = {layer:0 for layer in self.layers}     # initializationFalse
            
            # list out all neuronIdx.csv files for test
            for layer in self.layers:
                sig_neuron_idx_list = self.ANOVA_idces[layer]
                sig_neuron_idx = np.loadtxt(sig_neuron_idx_list, delimiter=',')
                value = sig_neuron_idx.size
                count_dict[layer]=value
                #print(layer, value)
                  
            ratio = [round(a / b * 100) for a, b in zip(list(count_dict.values()), self.neurons)]  # ratio = unit_passed_ANOVA/all_unit
            utils_.pickle_dump(ratio_path, ratio)     # save the percentages for other plot
        
        layers_color_list = color_column(self.layers)
        
        # 1. plot all calculations
        self.plot_single_figure(layers=self.layers, ratio=ratio, layers_color_list=layers_color_list, 
                                dest=fig_folder, title='sensitive unit ratio', length=math.floor(len(self.layers)/1.6), width=10, 
                                model_structure=f'{self.model_structure}')
        
        # 2. for imaginary neurons only ([question] why contains pool layers?) - this version is for ANN because 'act', SNN should be 'neuron'     
        neuron_layer = [[idx, layer] for idx, layer in enumerate(self.layers) if 'neuron' in layer or 'pool' in layer or 'fc_3' in layer]
        neuron_layer_idx = [_[0] for _ in neuron_layer]
        neuron_layer = [_[1] for _ in neuron_layer]
        
        ratio_neuron = [ratio[_] for _ in neuron_layer_idx]
        layers_color_list_neuron = [layers_color_list[_] for _ in neuron_layer_idx]
                
        self.plot_single_figure(layers=neuron_layer, ratio=ratio_neuron, layers_color_list=layers_color_list_neuron, 
                                dest=fig_folder, title='sensitive unit ratio neuron', length=math.floor(len(neuron_layer)/1.6), width=10, 
                                model_structure=f'{self.model_structure}')
                                
        print('[Codinfo] selective_neuron_percent.png has been saved into {}'.format(fig_folder))
        
    def plot_single_figure(self, **kwargs):
        
        plt.rcParams.update({'font.size': 22})     # control the all font size
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            fig, ax = plt.subplots(figsize=(kwargs['length'], kwargs['width']), dpi=100)  
            ax.plot(kwargs['layers'], kwargs['ratio'], color='red', linestyle='-', linewidth=2.5, alpha=1, label=kwargs['model_structure'])
            ax.bar(kwargs['layers'], kwargs['ratio'], color=kwargs['layers_color_list'], width=0.5)
            ax.set_xticklabels(labels=kwargs['layers'], rotation='vertical')
            ax.set_ylabel('percentage (%)')
            ax.set_title(f'{kwargs["title"]} - {self.model_structure}')
            ax.legend()
            fig.savefig(os.path.join(kwargs['dest'], kwargs['title']+'.png'), bbox_inches='tight')
            fig.savefig(os.path.join(kwargs['dest'], kwargs['title']+'.eps'), bbox_inches='tight', format='eps')
            #plt.show()
            plt.close()
        
    def selectivity_neuron_ANOVA_plot_Comparison(self, comparing_models_list):
        
        """
            in this function, the default model is which underlies this class, and the input is a list of comparing models
            [notice] in update on Sept 5, 2023, the 'ratio' generation has been removed, so before this section, MUST use 
            selectivity_neuron_ANOVA_plot() to generate the 'ratio.pkl'
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
            ratio = utils_.pickle_load(ratio_path)
        else:
            raise RuntimeError('[Coderror] no ratio file detected for current model, plase do selectivity_neuron_ANOVA_plot() first.')
        # -----
        
        # 1. for all calculation
        title = 'Sensitive Unit Percentage(s) Comparision'
        fig, ax = plt.subplots(figsize=(math.floor(len(self.layers)/1.6), 10), dpi=100)
        
        # plot fundamental model on the canvas
        self.plot_single_figure_VS(ax=ax, layers=self.layers, ratio=ratio, layers_color_list=layers_color_list, model_structure=self.model_structure,
                                   linewidth=2.5, alpha=1, linestyle='-')
        
        # plot comparing models on the canvas
        for comparing_model in comparing_models_list:     # for each comparing model
            ratio_comparing, comparing_model_config_list = self.load_comparing_data(comparing_model)
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
        fig.savefig(os.path.join(fig_folder, title+'.eps'), bbox_inches='tight', format='eps')     # no transparency
        fig.savefig(os.path.join(fig_folder, title+'.svg'), bbox_inches='tight', format='svg', transparent=True)
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
            ratio_comparing, comparing_model_config_list = self.load_comparing_data(comparing_model)
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
        fig.savefig(os.path.join(fig_folder, title+'.eps'), bbox_inches='tight', format='eps')     # no transparency
        fig.savefig(os.path.join(fig_folder, title+'.svg'), bbox_inches='tight', format='svg', transparent=True)
        plt.close()
        
        print('[Codinfo] selective_neuron_percent.png has been saved into {}'.format(fig_folder))
        
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
            
    def load_comparing_data(self, comparing_model):
        """
            this function does not have auto path locating, must precisely provide the location of target ratio.pkl file,
            or precisely build the path structure for multiple models
            
            [notice] in the update on Sept 5, 2023, the folder structures have been changed from {comparing_model}/ANOVA/ratio.pkl
            to {comparing_model}/ANOVA/Figures/ratio.pkl. The entire update refers to github
        """
        ratio_comparing = utils_.pickle_load(f'/media/acxyle/Data/ChromeDownload/{comparing_model}/Analysis/ANOVA/Figures/ratio.pkl')
        
        if 'baseline' not in comparing_model.lower():
            comparing_model_config_list = comparing_model.split('_')[1:-2]     # for SNN: 1_2_3_4 - structure_neuron_surrogate_T, for ANN: 1_2 - structure_activation
        else:
            comparing_model_config_list = [comparing_model.split('_')[1]+'(baseline)', 'ReLU']
        return ratio_comparing, comparing_model_config_list
        
    def load_and_check_p_value(self, alpha, save=False, plot=False, verbose=False):
        print('[Codinfo] Executing load_and_check_p_value...')
        self.load_and_check_p_value_global(alpha, self.dest, self.layers, self.neurons, save, plot, verbose)
     
        
    # [notice] below code is legacy design from early version in March 2023 for outside call to investigate alpha-level
    @staticmethod
    def load_and_check_p_value_global(alpha, p_folder, layers, neurons, save=False, plot=False, verbose=False):
        
        p_value_list = [os.path.join(p_folder, layer+'-pvalue.csv') for layer in layers]
        print("[Codinfo] root [{}] has [{}] p_value files".format(p_folder, len(p_value_list)))
        
        save_path = os.path.join(p_folder, 'alpha({})'.format(alpha))
        count_dict = {layer:0 for layer in layers}     # initialization
        
        if verbose:
            for p in p_value_list:
                print(p)
        for layer in layers:  # each layer
            for p_value_path in p_value_list:
                if layer == p_value_path.split('/')[-1].split('-')[0]:
                    temp = np.loadtxt(p_value_path)
                    sig_neuron_idx = []
                    for idx, p_value in enumerate(temp):      # each neuron
                        if p_value < alpha:
                            sig_neuron_idx.append(idx)
                    
                    count_dict[layer]=len(sig_neuron_idx)
                    print("[Codinfo] [{}] has [{}] sig neuron with alpha [{}]".format(layer.split('/')[-1], len(sig_neuron_idx), alpha))
                    
                    if save==True:
                        if verbose:
                            print('[Codinfo] Creating a new subfolder of [alpha({})] under [{}]'.format(alpha, p_folder))
                        utils_.make_dir(save_path)
                        neuron_idx = np.array(sig_neuron_idx)
                        if not os.path.exists(os.path.join(save_path, layer+'-neuronIdx(alpha:{}).csv'.format(alpha))):
                          np.savetxt(os.path.join(save_path, layer+'-neuronIdx(alpha:{}).csv'.format(alpha)), neuron_idx, delimiter=',')
                          
        ratio = [round(a / b * 100) for a, b in zip(list(count_dict.values()), neurons)]
        
        if verbose:
            for idx, item in enumerate(count_dict.items()):
                print(item, '{} all_neuron | ratio: {}%'.format(neurons[idx], ratio[idx]))
        
        if plot:
            print('[Codinfo] ploting selectivity_neuron_ANOVA...')
            if verbose:
                print('[Codwarning] this figure based on data may comes from outside the initialiazation of this class, so the curve may inaccurate')
            
            layers_color_list = color_column(layers)
            utils_.make_dir(save_path)
            plt.clf()    
            plt.figure(figsize=(math.floor(len(layers)/1.6),10), dpi=200)
            plt.bar(layers, ratio, color=layers_color_list, width=0.5)
            plt.xticks(rotation='vertical')
            plt.ylabel('percentage')
            plt.title('selective neuron ratio for each layer')
            plt.savefig(os.path.join(save_path, 'selective_neuron_percent_.png'))
            print('[Codinfo] selectivi_neuron_percent_.png has been saved into {}'.format(save_path))
        
    @staticmethod
    def selectivity_neuron_ANOVA_plot_arbitrary_alpha(alpha_list, dest, layers, neurons):     
        print('[Codinfo] Executing selectivity_neuron_ANOVA_plot_arbitrary_alpha...')
        print('[Codwarning] This function plot from pvalue.csv')
        dest = os.path.join(dest, 'ANOVA/')
        utils_.make_dir(dest)
        
        p_value_list = [os.path.join(dest, layer+'-pvalue.csv') for layer in layers]
        print("[Codinfo] root [{}] has [{}] p_value files".format(dest, len(p_value_list)))
            
        x_ = [i for i in range(len(layers))]  
        plt.figure(figsize=(math.floor(len(alpha_list)*len(layers)/2),10), dpi=200)
        
        for idx, alpha in enumerate(alpha_list):    # each alpha
            print(idx, alpha)
            count_dict = {layer:0 for layer in layers}
            for layer in p_value_list:      # each layer
                temp = np.loadtxt(layer)
                sig_neuron_idx = []
                for i, p_value in enumerate(temp):    # each neuron
                    if p_value < alpha:
                        sig_neuron_idx.append(i)
                count_dict[layer.split('/')[-1].split('-')[0]] = len(sig_neuron_idx)
                
            ratio = [round(a / b * 100) for a, b in zip(list(count_dict.values()), neurons)]
            print(ratio)
        
            plt.bar((np.array(x_)+0.2*idx).tolist(), ratio, label='alpha: {}'.format(alpha), width=0.2)
            
        plt.xticks((np.array(x_)+math.floor(len(alpha_list)/2)*0.2).tolist(), layers)
        plt.ylabel('percentage')
        plt.title('selective neuron ratio for each layer with different alpha')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(dest, 'selective_neuron_percent_differet_alpha.png'))
        #plt.show()
        print('[Codinfo] selectivi_neuron_percent_differet_alpha.png has been saved into {}'.format(dest))


def color_column(layers):
    """
        in order to increase generalization, color assignment is not in use but randomly generate colors
    """
    layers_t = []
    color = []
    
    for layer in layers:
        layers_t.append(layer.split('_')[0])
    layers_t = list(set(layers_t))
    #print(layers_t)
    
    for item in range(len(layers_t)):
        color.append((np.random.random(), np.random.random(), np.random.random()))
    # ↑↓
    #color = ['teal', 'red', 'orange', 'lightskyblue', 'tomato']    
    
    layers_c_dict = {}
    for i in range(len(layers_t)):
        layers_c_dict[layers_t[i]] = color[i]     
    #print(layers_c_dict)     # ['bn', 'fc', 'avgpool', 'activation', 'conv']
            
    layers_color_list = []
        
    for layer in layers:
        layers_color_list.append(layers_c_dict[layer.split('_')[0]])
    #print(layers_color_list)
        
    return layers_color_list


if __name__ == "__main__":
    
    model_ = vgg.__dict__['vgg16'](num_classes=50)
    layers, neurons, shapes = utils_.generate_vgg_layers_list_ann(model_, 'vgg16')
    
    analyzer = ANOVA_analyzer(root='/home/acxyle-workstation/Downloads/Face_Identity VGG16/', 
                              alpha=0.01, num_classes=50, num_samples=10, layers=layers, neurons=neurons)
    
    #analyzer.selectivity_neuron_ANOVA()
    analyzer.selectivity_neuron_ANOVA_plot()

    #analyzer.selectivity_neuron_ANOVA_plot_VS_baseline('Identity_VGG16bn_ReLU_CelebA2622_Neuron')
