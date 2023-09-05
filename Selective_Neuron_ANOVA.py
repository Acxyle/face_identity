#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:53:33 2023

@author: acxyle

    #TODO
    build a compact script for (1) ANOVA analysis; (2) .pkl and .csv store; (3) plot
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
    def __init__(self, 
                 root='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Results/',
                 dest='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG19bn_Neuron/',
                 alpha=0.01, num_classes=50, num_samples=10, layers=None, neurons=None):
        self.root = root
        
        self.layers = layers
        self.neurons = neurons
        
        self.dest = dest
        utils_.make_dir(self.dest)
        
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_samples = num_samples
        
        config_list = dest.split('/')[-2].split('_')[1:5]
        self.model_structure = config_list[0]
        self.neuron_type = config_list[1]
        self.surrogate_func = config_list[2]
        self.simulation_step = config_list[3]
        
        # some tests to check the saved files are valid and matchable with self.layers and self.neurons
        if self.layers == None or self.neurons == None:
            raise RuntimeError('[Coderror] invalid layers and/or neurons')
            
        if not os.path.exists(self.root):
            raise RuntimeWarning(f'[Codwarning] can not find the root of [{self.root}]')
        elif os.listdir(self.root) == []:
            raise RuntimeWarning('[Codwarning] the path of feature directory is valid but found no file')
        
        #if set([f.split('.')[0] for f in os.listdir(self.root)]) != set(self.layers):     # [notice] can mute this message when segmental test
        #    raise AssertionError('[Coderror] the saved .pkl files must be exactly the same with attribute self.layers')
        
    def selectivity_neuron_ANOVA(self, verbose=False):
        print('[Codinfo] Executing selectivity_neuron_ANOVA...')

        # === for each layer
        for idx_l, layer in enumerate(self.layers):     
            neuron_idx = []
            with open(os.path.join(self.root, layer+'.pkl'), 'rb') as pkl:
                feature = pickle.load(pkl)
            pl = np.zeros(feature.shape[1])       # p_value_list for all neuron
                
            if verbose:
                print(layer, feature.shape)
            if feature.shape[0] != self.num_classes*self.num_samples or feature.shape[1] != self.neurons[idx_l]:     #  feature check
                raise AssertionError('[Coderror] feature.shape[0] ({}) != self.num_classes*self.num_samples ({},{}) or feature.shape[1] ({}) != self.neurons[idx_l] ({})'.format(
                    feature.shape[0], self.num_classes, self.num_samples, feature.shape[1], self.neurons[idx_l]))
            
            # ----- parallel computing by concurrent.futures.Thread/Process
            #executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())
            #job_pool = []
            #for i in tqdm(range(feature.shape[1]), desc=f'[{layer}] ANOVA Submit'):
            #    job = executor.submit(self.one_way_ANOVA, feature, i)
            #    job_pool.append(job)
            #for i in tqdm(range(feature.shape[1]), desc=f'[{layer}] ANOVA Collect'):
            #    pl[i] = job_pool[i].result()
            #executor.shutdown()
            # ----- parallel computing by joblib
            pl = Parallel(n_jobs=os.cpu_count())(delayed(self.one_way_ANOVA)(feature, i) for i in tqdm(range(feature.shape[1]), desc=f'[{layer}] ANOVA'))
            
            if not os.path.exists(os.path.join(self.dest, layer+'-pvalue.csv')):
                np.savetxt(os.path.join(self.dest, layer+'-pvalue.csv'), pl)
            
            sig_neuron_idx = [idx for idx, p in enumerate(pl) if p < self.alpha]     # p_value_idx_list for neurons that satisfy the threshold
              
            if verbose:
                print('[Codinfo]', layer, 'has', len(sig_neuron_idx), 'significant neurons')
                  
            neuron_idx.append(sig_neuron_idx)
            neuron_idx = np.array(neuron_idx)
            if not os.path.exists(os.path.join(self.dest, layer+'-neuronIdx.csv')):
                np.savetxt(os.path.join(self.dest, layer+'-neuronIdx.csv'), neuron_idx, delimiter=',')
        
        print('[Codinfo] Selectivity_Neuron_ANOVA Calculation Done.')
        print('[Codinfo] pvalue.csv and neuron_idx.pkl files have been saved in {}'.format(self.dest))
        
    # define ANOVE for multi-threads excution
    def one_way_ANOVA(self, feature, i):
        neuron = feature[:, i]
        d = [neuron[i*self.num_samples: i*self.num_samples+self.num_samples] for i in range(self.num_classes)]
        p = stats.f_oneway(*d)[1]  # [0] for F-value, [1] for p-value
        
        return p
    
    def selectivity_neuron_ANOVA_plot(self):      # this plot rely on both Local initialization and the Saved .csv files
        print('[Codinfo] Executing selectivity_neuron_ANOVA_plot...')
        print('{Codwarning] This function plots from signeuronIdx.csv')
        
        dest = os.path.join(self.dest, 'ANOVA/')
        utils_.make_dir(dest)
        
        count_dict = {layer:0 for layer in self.layers}     # initializationFalse
        
        # list out all neuronIdx.csv files for test
        for layer in self.layers:
            for f in [p for p in os.listdir(self.dest) if 'neuronIdx' in p]:
                if layer in f:     # [help] add extra judgement but allow more free name space
                    sig_neuron_idx_list = self.dest + f
                    sig_neuron_idx = np.loadtxt(sig_neuron_idx_list, delimiter=',')
                    value = sig_neuron_idx.size
                    count_dict[layer]=value
                    #print(layer, value)
              
        ratio = [round(a / b * 100) for a, b in zip(list(count_dict.values()), self.neurons)]
        utils_.pickle_dump(dest+'ratio.pkl', ratio)     # save the percentages for plot
        
        layers_color_list = color_column(self.layers)
        
        # 1. plot all calculations
        self.plot_single_figure(self.layers, ratio, layers_color_list, dest, 'selective_unit_percent', math.floor(len(self.layers)/1.6), 10, f'{self.model_structure}')
        
        # 2. for imaginary neurons only ([question] why contains pool layers?) - this version is for ANN because 'act', SNN should be 'neuron'
        ratio_neuron = []
        layers_color_list_neuron = []
        count_dict_neuron = {layer:count_dict[layer] for layer in list(count_dict.keys()) if 'neuron' in layer or 'pool' in layer or 'fc_3' in layer}
        for idx,i in enumerate(count_dict.keys()):
            if i in count_dict_neuron:
                ratio_neuron.append(round(count_dict[i]/self.neurons[idx]*100))
                layers_color_list_neuron.append(layers_color_list[idx])   
        
        self.plot_single_figure(list(count_dict_neuron.keys()), ratio_neuron, layers_color_list_neuron, dest, 'selective_unit_percent (dense)', 30, 10, f'{self.model_structure} (neuron)')
        
        print('[Codinfo] selective_neuron_percent.png has been saved into {}'.format(dest))
        
    def plot_single_figure(self, layers, ratio, layers_color_list, dest, mark, length, width, label):
        # [important] control the all font size
        # [notice] the impact is constant unless reopen the terminal
        plt.rcParams.update({'font.size': 22})
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            plt.clf()    
            plt.figure(figsize=(length, width), dpi=100)
            plt.plot(layers, ratio, color='red', linestyle='-', linewidth=2.5, alpha=1, label=label)
            plt.bar(layers, ratio, color=layers_color_list, width=0.5)
            plt.xticks(rotation='vertical')
            plt.ylabel('percentage')
            plt.title(f'selective neuron ratio for each layer ({mark}) - {self.model_structure}')
            plt.legend()
            plt.savefig(os.path.join(dest, mark+'.png'), bbox_inches='tight')
            plt.savefig(os.path.join(dest, mark+'.eps'), bbox_inches='tight', format='eps')
            #plt.show()
            plt.close()
        
    # [notice] this function amis to compare 2 different models performance, now need to manually input the data of the second model
    def selectivity_neuron_ANOVA_plot_VS_baseline(self, baseline_model):
        print('[Codinfo] Executing selectivity_neuron_ANOVA_plot...')
        print('{Codwarning] This function plots from signeuronIdx.csv')
        
        dest = os.path.join(self.dest, 'ANOVA/')
        utils_.make_dir(dest)
        
        ratio_file = os.path.join(dest, 'ratio.pkl')
        
        if os.path.exists(ratio_file):
            ratio = utils_.pickle_load(ratio_file)
        else:
            count_dict = {layer:0 for layer in self.layers}     # initializationFalse
            # list out all neuronIdx.csv files for test
            for layer in self.layers:
                for f in [p for p in os.listdir(self.dest) if 'neuronIdx' in p]:
                    if layer in f:     # [help] add extra judgement but allow more free name space
                        sig_neuron_idx_list = self.dest + f
                        sig_neuron_idx = np.loadtxt(sig_neuron_idx_list, delimiter=',')
                        value = sig_neuron_idx.size
                        count_dict[layer]=value
                        #print(layer, value)
                  
            ratio = [round(a / b * 100) for a, b in zip(list(count_dict.values()), self.neurons)]
            utils_.pickle_dump(dest+'ratio.pkl', ratio)
        
        ratio_baseline, baseline_model_config = self.load_baseline_data(baseline_model)     # load the baseline model
        baseline_model_structure = baseline_model_config.split('_')[0]
        print(f'[Codinfo] Comparing models: {self.model_structure} v.s. {baseline_model_structure}')
        
        layers_color_list = color_column(self.layers)
        
        # 1. for all calculation
        self.plot_single_figure_VS(layers=self.layers, ratio=ratio, 
                                  baseline_model_config=baseline_model_structure, ratio_baseline=ratio_baseline, 
                                  layers_color_list=layers_color_list, dest=dest, save_path='selective unit percent comparison', 
                                  length=math.floor(len(self.layers)/1.6), width=10, config=self.model_structure)
        
        # 2. for imaginary neurons only ([question] why contains pool layers?) - this version is for ANN because 'act', SNN should be 'neuron'
        neuron_layer = [[idx, layer] for idx, layer in enumerate(self.layers) if 'neuron' in layer or 'pool' in layer or 'fc_3' in layer]
        neuron_layer_idx = [_[0] for _ in neuron_layer]
        neuron_layer = [_[1] for _ in neuron_layer]
        
        ratio_neuron = [ratio[_] for _ in neuron_layer_idx]
        ratio_neuron_baseline = [ratio_baseline[_] for _ in neuron_layer_idx]
        layers_color_list_neuron = [layers_color_list[_] for _ in neuron_layer_idx]
                
        self.plot_single_figure_VS(layers=neuron_layer, ratio=ratio_neuron, 
                                  baseline_model_config=baseline_model_structure, ratio_baseline=ratio_neuron_baseline, 
                                  layers_color_list=layers_color_list_neuron, dest=dest, save_path='selective unit percent comparison neuron', 
                                  length=math.floor(len(neuron_layer)/1.6), width=10, config=self.model_structure)
        
        print('[Codinfo] selective_neuron_percent.png has been saved into {}'.format(dest))
        
    def plot_single_figure_VS(self, **kwargs):
        plt.rcParams.update({'font.size': 22})
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
    
            plt.clf()    
            plt.figure(figsize=(kwargs['length'], kwargs['width']), dpi=100)
            plt.plot(kwargs['layers'], kwargs['ratio'], color='red', linestyle='-', linewidth=2.5, alpha=1, label=kwargs['config'])
            plt.bar(kwargs['layers'], kwargs['ratio'], color=kwargs['layers_color_list'], width=0.5)
            
            plt.plot(kwargs['layers'], kwargs['ratio_baseline'], color='blue', linestyle='-', linewidth=2.5, alpha=1, label=f'{kwargs["baseline_model_config"]}')
            plt.bar(kwargs['layers'], kwargs['ratio_baseline'], color=kwargs['layers_color_list'], width=0.5, alpha=0.25)
            
            plt.xticks(rotation='vertical')
            plt.ylabel('percentage')
            plt.title(f'{kwargs["save_path"]}')
            plt.legend()
            plt.savefig(os.path.join(kwargs['dest'], kwargs['save_path']+'.png'), bbox_inches='tight')
            plt.savefig(os.path.join(kwargs['dest'], kwargs['save_path']+'.eps'), bbox_inches='tight', format='eps')     # no transparency
            plt.savefig(os.path.join(kwargs['dest'], kwargs['save_path']+'.svg'), bbox_inches='tight', format='svg', transparent=True)
            #plt.show()
            plt.close()
        
    def load_baseline_data(self, baseline_model):
        ratio_baseline = utils_.pickle_load(f'/media/acxyle/Data/ChromeDownload/{baseline_model}/ANOVA/ratio.pkl')
        baseline_model_config = baseline_model.split('_')[1:-2]
        return ratio_baseline, '_'.join(baseline_model_config)
        
    def load_and_check_p_value(self, alpha, save=False, plot=False, verbose=False):
        print('[Codinfo] Executing load_and_check_p_value...')
        self.load_and_check_p_value_global(alpha, self.dest, self.layers, self.neurons, save, plot, verbose)
     
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

# [notice] in order to increase generalization, color assignment is not in use but randomly generate colors, this can be changed back to fix colors in simple situations
def color_column(layers):
    """
    distinguish how many types in layers and return a list of color as the length of layers
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
    
    spiking_model = spiking_vgg.__dict__['spiking_vgg16_bn'](spiking_neuron=neuron.LIFNode, num_classes=50, surrogate_function=surrogate.ATan(), detach_reset=True)
    functional.set_step_mode(spiking_model, step_mode='m')
    layers, neurons, shapes = utils_.generate_vgg_layers(spiking_model, 'spiking_vgg16_bn')
    
# =============================================================================
#     layers_ = [i for i in layers if 'neuron' in i or 'fc3' in i or 'pool' in i]
#     index_ = [layers.index(i) for i in layers_]
#     neurons_ = [neurons[i] for i in index_]
#     shapes_ = [shapes[i] for i in index_]
#     layers = layers_
#     neurons = neurons_
# =============================================================================
    
    analyzer = ANOVA_analyzer(root='/media/acxyle/Data/ChromeDownload/Identity_SpikingVGG16bn_LIF_CelebA2622_Results/', 
                              dest='/media/acxyle/Data/ChromeDownload/Identity_SpikingVGG16bn_LIF_CelebA2622_Neuron/',
                              alpha=0.01, num_classes=50, num_samples=10, layers=layers, neurons=neurons)
    
    #analyzer.selectivity_neuron_ANOVA()
    #analyzer.selectivity_neuron_ANOVA_plot()

    analyzer.selectivity_neuron_ANOVA_plot_VS_baseline('Identity_VGG16bn_ReLU_CelebA2622_Neuron')
