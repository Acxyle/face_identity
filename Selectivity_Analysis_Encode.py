#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:54:55 2023

@author: acxyle

    complete 5 sections in one script:
        1. obtrain_encode_class_dict() - save dict.pkl
        2. draw_encode_frequency()
        3. draw_encode_frequency_for_each_layer()
        4. draw_merged_encode_frequency_for_each_layer()
        5. draw_single_neuron_response()
    
    Task: Sept 6, 2023
        
        rewrite the code based on my preference - make the 'encode' not based on ANOVA but independent

"""

import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import random
import argparse
import gc
import logging
#from functools import reduce

import spiking_vgg, spiking_resnet, sew_resnet
from spikingjelly.activation_based import surrogate, neuron, functional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from joblib import Parallel, delayed
from scipy.interpolate import interp1d

import vgg, resnet
import utils_


class Encode_feaquency_analyzer():
    """
        in the update on Sept 6, 2023, remove original 2 input files setting
    """
    def __init__(self, root,
                 num_classes=50, num_samples=10, layers=None, neurons=None):
        
        self.root = os.path.join(root, 'Features/')     # <- folder for feature maps, which should be generated before analysis
        self.dest = '/'.join([*root.split('/')[:-1], 'Analysis'])     # <- folder for analysis results
        utils_.make_dir(self.dest)
        
        self.dest_Encode = os.path.join(self.dest, 'Encode')
        utils_.make_dir(self.dest_Encode)
        
        self.layers = layers
        self.neurons = neurons
        
        self.num_classes = num_classes
        self.num_samples = num_samples
        
        self.feature_list = [os.path.join(self.root, _) for _ in sorted(os.listdir(self.root)) if 'pkl' in _]     # feature .pkl list
        idx_folder = os.path.join(self.dest, 'ANOVA/ANOVA_idces')
        self.idx_list = [os.path.join(idx_folder, _) for _ in sorted(os.listdir(idx_folder)) if 'neuronIdx'  in _]    # <- consider to remove this?
        
        if layers == None or neurons == None:
            raise RuntimeError('[Codwarning] invalid layers and neurons')
            
        # FIXME - to have a better function to plot
        self.subplot_row, self.subplot_col = self.generate_subplot_row_and_col(len(self.layers))
        
        self.model_structure = root.split('/')[-2].split(' ')[1]
        
    #FIXME
    def generate_subplot_row_and_col(self, input):
        col = 5
        row = math.floor(input/col) +1
        remainder = input%col
        print(f'[Codinfo] Calculated cal [{col}] and row+1 [{row}] for input [{input}], with remainder [{remainder}]')
        return row, col
    
    # define Encode for parallel calculation
    def encode_calculation(self, feature, i):
        """
            under construction...
        """
        
        feature_of_single_unit = feature[:, i]
        grouped_feature_of_single_unit = feature_of_single_unit.reshape(-1,10)
        
        threshold = np.mean(feature_of_single_unit) + 2*np.std(feature_of_single_unit)
        local_mean = np.mean(grouped_feature_of_single_unit, axis=1)     # array of 50 values
        
        encode_class = np.where(local_mean>threshold)[0]+1  # '>' prevent all 0
        
        #plt.bar(range(len(local_mean)), local_mean)
        #plt.hlines(threshold, 0, 49)
        
        return encode_class
    
    #FIXME - the stuck problem of processpool
    def obtain_encode_class_dict(self, ):
        """
            current version, the idx is based on id_sensitive unit, not absolute idx the correction is a big project
            considering need to change all following functions this code now looks not good, need to rewrite
            
            [task] 1) parallel computation; 2) separate computation and plot; 3) abs Enocde rather sensitive_based unit 
        """
        print('[Codinfo] Executing obtain_encode_class_dict...')
        self.Encode_dict = {}
        self.Sort_dict = {}
        
        num_workers = os.cpu_count()
        print(f'[Codinfo] Executing parallel computation with num_workers={num_workers}')

        # ----- layer check
        feature_list_check = [_.split('/')[-1].split('.')[0] for _ in self.feature_list]
        layers_check = self.layers.copy()
        if not sorted(layers_check) == sorted(feature_list_check):
            raise RuntimeError('[Coderror] detected the features and layers not match')
        # -----
        
        for idx, feature_path in enumerate(self.feature_list):     # for each layer
            
            layer = feature_path.split('/')[-1].split('.')[0]
        
            feature = utils_.pickle_load(feature_path)      # load feature matrix
                
            idx_path = self.idx_list[idx]
            sensitive_idx = np.loadtxt(idx_path, delimiter=',').astype(int)      # 1.1 sensitive_idx
            non_sensitive_idx = np.array(list(set(np.arange(feature.shape[1]))-set(sensitive_idx)))     # 1.2 non_sensitive_idx
            
            unit_encode_dict = {}

            #for i in tqdm(range(feature.shape[1]), desc='unit'):     # for each neuron
            with Parallel(n_jobs=num_workers, require='sharedmem') as parallel:
                pl = parallel(delayed(self.encode_calculation)(feature, i) for i in tqdm(range(feature.shape[1]), desc=f'[{layer}] Encode'))  
                
            for i in range(feature.shape[1]):
                unit_encode_dict[i] = pl[i]

            self.Encode_dict[layer] = unit_encode_dict
            
            # [notice] 2 types of units
            unit_encode_list = [_ for _ in unit_encode_dict.values()]
            unit_encode_num = np.array([len(_) for _ in unit_encode_list])
            
            si_idx = np.where(unit_encode_num==1)[0]     # 2.1 si_idx
            mi_idx = np.where(unit_encode_num>1)[0]     # 2.2 mi_idx
            
            encode_idx = np.hstack((si_idx, mi_idx))     # 2.3 (2.1+2.2) encode_idx
            non_encode_idx = np.array(list(set(np.arange(feature.shape[1]))-set(encode_idx)))     # 2.4 non_encode_idx
            
            sensitive_si_idx = np.intersect1d(si_idx, sensitive_idx)     # 3.1 sensitive_si_idx
            sensitive_mi_idx = np.intersect1d(mi_idx, sensitive_idx)     # 3.2 sensitive_mi_idx
            sensitive_encode_idx = np.intersect1d(sensitive_idx, encode_idx)     # 3.3 sensitive_encode_idx
            sensitive_non_encode_idx = np.intersect1d(sensitive_idx, non_encode_idx)     # 3.4 sensitive_non_encode_idx
            
            non_sensitive_si_idx = np.intersect1d(non_sensitive_idx, si_idx)     # 4.1 ns_s
            non_sensitive_mi_idx = np.intersect1d(non_sensitive_idx, mi_idx)     # 4.2 ns_m
            non_sensitive_encode_idx = np.intersect1d(non_sensitive_idx, encode_idx)     # 4.3 ns_e
            non_sensitive_non_encode_idx = np.intersect1d(non_sensitive_idx, non_encode_idx)     # 4.4 ns_ne
            
            basic_type = {
                'si_idx': si_idx,
                'mi_idx': mi_idx,
                'non_encode_idx': non_encode_idx
                }
            
            advanced_type = {
                'sensitive_si_idx': sensitive_si_idx,
                'sensitive_mi_idx': sensitive_mi_idx,
                'sensitive_encode_idx': sensitive_encode_idx,
                'sensitive_non_encode_idx': sensitive_non_encode_idx,
                
                'non_sensitive_si_idx': non_sensitive_si_idx,
                'non_sensitive_mi_idx': non_sensitive_mi_idx,
                'non_sensitive_encode_idx': non_sensitive_encode_idx,
                'non_sensitive_non_encode_idx': non_sensitive_non_encode_idx,
                }
            
            unit_sort_dict = {
                'basic_type': basic_type,
                'advanced_type': advanced_type
                }
            
            self.Sort_dict[layer] = unit_sort_dict
            
            gc.collect()
            
        print('[Codinfo] Saving Encode_dict.pkl and Sort_dict.pkl...')
        utils_.pickle_dump(os.path.join(self.dest_Encode, 'Encode_dict.pkl'), self.Encode_dict)      # save the relationship between layer (include SI and MI) and encoded classes
        utils_.pickle_dump(os.path.join(self.dest_Encode, 'Sort_dict.pkl'), self.Sort_dict)
        print('[Codinfo] Encode_dict.pkl saved in {}'.format(self.dest_Encode))
    
    #FIXME
    # looks need to find a better way to save the data make us to better understand the encode_id, si_idx, mi_idx, just like the MATLAB version
    def reload_encode_and_sort_dict(self, ):
        print('[Codinfo] Reloading Encode_dict and Sort_dict...')
        
        self.Encode_dict = utils_.pickle_load(os.path.join(self.dest_Encode, 'Encode_dict.pkl'))
        self.Sort_dict = utils_.pickle_load(os.path.join(self.dest_Encode, 'Sort_dict.pkl'))
    
    def calculate_intersection_point(self, y1, y2, num_interpolate=10000):
        x = np.arange(len(y1))
        
        f1 = interp1d(x, y1)
        f2 = interp1d(x, y2)
        
        x_new = np.linspace(0, len(x)-1, num_interpolate)
        intersection_x = None
        for xi in x_new:
            if f1(xi) >= f2(xi):
                intersection_x = xi
                break
            
        intersection_y = f1(intersection_x)
        
        return intersection_x, intersection_y
    
    # ----- under construction...
    def encode_layer_percent_plot(self, fig_folder, fig, ax, curve_dict=None, point_dict=None, title=None, save=True):
        print(f'[Condinfo] plotting {title}')
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        plt.rcParams.update({'font.size': 18})     # control the all font size
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        if curve_dict is not None:
            for curve in list(curve_dict.keys()):     # this is a design of dict-dict
                curve = curve_dict[curve]
                ax.plot(curve['values'], color=curve['color'], linestyle=curve['linestyle'], linewidth=curve['linewidth'], label=curve['label'])
        if point_dict is not None:
            for point in list(point_dict.keys()):
                point = point_dict[point]
                ax.scatter(point['point']['x'], point['point']['y'], color=point['color'], linewidth=1.0, label=point['label'])
                ax.vlines(point['point']['x'], 0, point['point']['y'], color='gray', linewidth=1.0)
        
        ax.legend(framealpha=0.5)
        ax.set_title(f'{title}', fontname='Times New Roman')
        ax.grid(True)
        ax.set_xticks(np.arange(len(self.layers)), self.layers, rotation='vertical', fontname='Times New Roman')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim([0,100])
        
        if save:
            fig.savefig(os.path.join(fig_folder, title+'.png'), bbox_inches='tight')
            fig.savefig(os.path.join(fig_folder, title+'.eps'), bbox_inches='tight', format='eps')     # no transparency
            fig.savefig(os.path.join(fig_folder, title+'.svg'), bbox_inches='tight', format='svg', transparent=True)
    
    # -----
    def encode_layer_percent_plot_dict(self, values=None, point=None, color=None, linestyle=None, linewidth=None, label=None):

        return {
            'values': values,
            'point': point,
            'color': color,
            'linestyle': linestyle,
            'linewidth': linewidth,
            'label': label
            }
    
    def selectivity_encode_layer_percent_plot(self, ):
        """
            this function plot the percentages of different types of units over layers
        """
        
        fig_folder = os.path.join(self.dest_Encode, 'Figures')
        utils_.make_dir(fig_folder)
        
        if not hasattr(self, 'Encode_dict') and not hasattr(self, 'Sort_dict'):
            self.reload_encode_and_sort_dict()
        
        print('[Codinfo] preparing plot...')
        # 3 types of basic units
        # ['si_idx', 'mi_idx', 'non_encode_idx']
        non_encode_unit_percentages = [len(self.Sort_dict[_]['basic_type']['non_encode_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        si_unit_percentages = [len(self.Sort_dict[_]['basic_type']['si_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        mi_unit_percentages = [len(self.Sort_dict[_]['basic_type']['mi_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        encode_unit_percentages = [si_unit_percentages[_]+mi_unit_percentages[_] for _ in range(len(self.layers))]
        
        inter_x, inter_y = self.calculate_intersection_point(encode_unit_percentages, non_encode_unit_percentages)
        inter_x_s, inter_y_s = self.calculate_intersection_point(si_unit_percentages, non_encode_unit_percentages)
        inter_x_m, inter_y_m = self.calculate_intersection_point(mi_unit_percentages, non_encode_unit_percentages)
        
        sensitive_encode_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['sensitive_encode_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        sensitive_non_encode_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['sensitive_non_encode_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        sensitive_si_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['sensitive_si_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        sensitive_mi_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['sensitive_mi_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        
        non_sensitive_encode_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['non_sensitive_encode_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        non_sensitive_non_encode_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['non_sensitive_non_encode_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        non_sensitive_si_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['non_sensitive_si_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        non_sensitive_mi_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['non_sensitive_mi_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        
        sensitive_percentages = [sensitive_encode_unit_percentages[_]+sensitive_non_encode_unit_percentages[_] for _ in range(len(self.layers))]
        non_sensitive_percentages = [non_sensitive_encode_unit_percentages[_]+non_sensitive_non_encode_unit_percentages[_] for _ in range(len(self.layers))]
        # -----
        
        # -----
        figs, axes = plt.subplots(2,2,figsize=(24,12))
        
        fig,ax = plt.subplots(figsize=(12,6))
        curve_dict = {
            'encode_unit': self.encode_layer_percent_plot_dict(values=encode_unit_percentages, label='encode (si+mi)', ),
            'si_unit': self.encode_layer_percent_plot_dict(values=si_unit_percentages, label='si', linestyle='--'),
            'mi_unit': self.encode_layer_percent_plot_dict(values=mi_unit_percentages, label='mi', linestyle='--'),
            'non_encode_unit': self.encode_layer_percent_plot_dict(values=non_encode_unit_percentages, label='non_encode'),
            }
        point_dict = {
            'intersect_e': self.encode_layer_percent_plot_dict(point={'x':inter_x,'y':inter_y}, label=f'intersect_e {inter_x:.2f}'),
            'intersect_s': self.encode_layer_percent_plot_dict(point={'x':inter_x_s,'y':inter_y_s}, label=f'intersect_s {inter_x_s:.2f}'),
            'intersect_m': self.encode_layer_percent_plot_dict(point={'x':inter_x_m,'y':inter_y_m}, label=f'intersect_m {inter_x_m:.2f}'),
            }
        self.encode_layer_percent_plot(fig_folder, fig, ax, curve_dict, point_dict, "encode vs non_encode")
        self.encode_layer_percent_plot(fig_folder, figs, axes[0,0], curve_dict, point_dict, "encode vs non_encode", False)
        
        # -----
        # 8 types of advanced units
        # ['sensitive_si_idx', 'sensitive_mi_idx', 'sensitive_encode_idx', 'sensitive_non_encode_idx', 
        # 'non_sensitive_si_idx', 'non_sensitive_mi_idx', 'non_sensitive_encode_idx', 'non_sensitive_non_encode_idx']
        
        inter_x, inter_y = self.calculate_intersection_point(sensitive_encode_unit_percentages, sensitive_non_encode_unit_percentages)
        inter_x_s, inter_y_s = self.calculate_intersection_point(sensitive_si_unit_percentages, sensitive_non_encode_unit_percentages)
        inter_x_m, inter_y_m = self.calculate_intersection_point(sensitive_mi_unit_percentages, sensitive_non_encode_unit_percentages)

        fig,ax = plt.subplots(figsize=(12,6))
        curve_dict = {
            'sensitive_encode_unit': self.encode_layer_percent_plot_dict(values=sensitive_encode_unit_percentages, label='sensitive_encode(si+mi)', linewidth=1.0),
            'sensitive_si_unit': self.encode_layer_percent_plot_dict(values=sensitive_si_unit_percentages, label='sensitive_si', linestyle='--', linewidth=1.0),
            'sensitive_mi_unit': self.encode_layer_percent_plot_dict(values=sensitive_mi_unit_percentages, label='sensitive_mi', linestyle='--', linewidth=1.0),
            'sensitive_non_encode_unit': self.encode_layer_percent_plot_dict(values=sensitive_non_encode_unit_percentages, label='sensitive_non_encode', linewidth=1.0),
            'sensitive_unit':self.encode_layer_percent_plot_dict(values=sensitive_percentages, label='sensitive', linewidth=3.0)
            }
        point_dict = {
            'intersect_s-e': self.encode_layer_percent_plot_dict(point={'x':inter_x,'y':inter_y}, label=f'intersect_s-e {inter_x:.2f}'),
            'intersect_s-s': self.encode_layer_percent_plot_dict(point={'x':inter_x_s,'y':inter_y_s}, label=f'intersect_s-s {inter_x_s:.2f}'),
            'intersect_s-m': self.encode_layer_percent_plot_dict(point={'x':inter_x_m,'y':inter_y_m}, label=f'intersect_s-m {inter_x_m:.2f}'),
            }
        self.encode_layer_percent_plot(fig_folder, fig, ax, curve_dict, point_dict, "sensitive")
        self.encode_layer_percent_plot(fig_folder, figs, axes[0,1], curve_dict, point_dict, "sensitive", False)
        
        # -----
        fig,ax = plt.subplots(figsize=(12,6))
        curve_dict = {
            'non_sensitive_encode_unit': self.encode_layer_percent_plot_dict(values=non_sensitive_encode_unit_percentages, label='non_sensitive_encode', linewidth=1.0),
            'non_sensitive_si_unit': self.encode_layer_percent_plot_dict(values=non_sensitive_si_unit_percentages, label='non_sensitive_si', linestyle='--', linewidth=1.0),
            'non_sensitive_mi_unit': self.encode_layer_percent_plot_dict(values=non_sensitive_mi_unit_percentages, label='non_sensitive_mi', linestyle='--', linewidth=1.0),
            'non_sensitive_non_encode_unit': self.encode_layer_percent_plot_dict(values=non_sensitive_non_encode_unit_percentages, label='non_sensitive_non_encode', linewidth=1.0),
            'non_sensitive_unit':self.encode_layer_percent_plot_dict(values=non_sensitive_percentages, label='non_sensitive', linewidth=3.0)
            }
        point_dict = None
        self.encode_layer_percent_plot(fig_folder, fig, ax, curve_dict, point_dict, "non_sensitive")
        self.encode_layer_percent_plot(fig_folder, figs, axes[1,0], curve_dict, point_dict, "non_sensitive", False)
        
        # -----
        fig,ax = plt.subplots(figsize=(12,6))
        curve_dict = {
            'sensitive_unit': self.encode_layer_percent_plot_dict(values=sensitive_percentages, color='purple', label='sensitive', linewidth=3.0),
            'non_sensitive_unit': self.encode_layer_percent_plot_dict(values=non_sensitive_percentages, color='purple', label='non_sensitive', linestyle='--'),
            'encode_unit': self.encode_layer_percent_plot_dict(values=encode_unit_percentages, color='blue', label='encode', linewidth=3.0),
            'non_encode_unit': self.encode_layer_percent_plot_dict(values=non_encode_unit_percentages, color='blue', label='non_encode', linestyle='--'),
            }
        point_dict = None
        self.encode_layer_percent_plot(fig_folder, fig, ax, curve_dict, point_dict, "sensitive and encode")
        self.encode_layer_percent_plot(fig_folder, figs, axes[1,1], curve_dict, point_dict, "sensitive and encode", False)
        
        title = 'sensitive_and_encode_all'
        figs.subplots_adjust(hspace=0.5, wspace=0.1)
        figs.savefig(os.path.join(fig_folder, title+'.png'), bbox_inches='tight')
        figs.savefig(os.path.join(fig_folder, title+'.eps'), bbox_inches='tight', format='eps')     # no transparency
        figs.savefig(os.path.join(fig_folder, title+'.svg'), bbox_inches='tight', format='svg', transparent=True)
        
        plt.close('all')
        
    # legacy design
    def recover_encode_class_dict(self, SIMI_dict=None):     # [Warning] this function merges the encoded [classes] again, different with SIMI.py
        print('[Codinfo] Executing recover_encode_class_dict...')
        if SIMI_dict == None:   # directly succeed from self.obtain_encode_clas_dict()
            SIMI_dict_ = self.SIMI_dict
        else:
            SIMI_dict_ = SIMI_dict
        temp_dict = {}
        for k, v in SIMI_dict_.items():  # for each layer
            encode_class = []
            if list(v[2]['SI_idx'].values()) != []:
                for i in list(v[2]['SI_idx'].values()):     # for each neuron
                    for j in i:
                        encode_class.append(j)
            if list(v[3]['MI_idx'].values()) != []:
                for i in list(v[3]['MI_idx'].values()):
                    for j in i:
                        encode_class.append(j)    
            temp_dict.update({k: encode_class})
            
        return temp_dict
    
    # fill-in the white blank
    def draw_encode_frequency(self):        # general figure for encoding frequency
    
        print('[Codinfo] Executing draw_encode_frequency...')
        
        if self.mode == 'reload_encode_dict':
            encode_class_dict = self.reload_and_revocer_encode_class_dict()
        else:
            encode_class_dict = self.recover_encode_class_dict() 
        #print(encode_class_dict)
        
        freq_dic = {}
        for idx, layer in enumerate(self.layers):    # for each layer
            freq = {g:0 for g in range(1,51)}
            encode_class_list = encode_class_dict[layer]   # [list] obtain the encoded classes
    
            for item in encode_class_list:  # for each class
                if item in freq:  # update the dict or add new k-v pair
                    freq[item] += 1 
                    
            freq = {k: v / self.neurons[idx] for k, v in freq.items()}    # convert v from abs avlue to ratio
            #freq = dict(sorted(freq.items(), key=lambda item: item[0]))     # sort
            freq_dic.update({layer: freq})
        
        a = pd.DataFrame.from_dict(freq_dic)
        a_neuron = a[[i for i in self.layers if 'neuron' in i or 'pool' in i or 'fc_3' in i]]
        
        # 1. for all calculations
        plt.figure()
        im = plt.matshow(a, aspect='auto')
        
        plt.colorbar(im, fraction=0.12, pad=0.04)
        plt.xticks([])
        plt.xlabel(f'{a.shape[1]} Layers')
        plt.yticks([])
        plt.ylabel('IDs')
        plt.title('all')
        plt.savefig(self.dest_Encode+'Encoding_frequency_of_layers_all_calculation.png', bbox_inches='tight', dpi=100)
        plt.close()
        
        # 2. for imaginary neurons
        plt.figure()
        im = plt.matshow(a_neuron, aspect='auto')
        plt.colorbar(im, fraction=0.12, pad=0.04)
        plt.xticks([])
        plt.xlabel(f'{a_neuron.shape[1]} Layers')
        plt.yticks([])
        plt.ylabel('IDs')
        plt.title('act')
        plt.savefig(self.dest_Encode+'Encoding_frequency_of_layers.png', bbox_inches='tight', dpi=100)
        plt.close()
        
        # 3. merged plot
        fig, axs = plt.subplots(1, 2)
        im = axs[0].matshow(a, aspect='auto')
        axs[1].matshow(a_neuron, aspect='auto')
        axs[0].set_title('all')
        axs[0].set_xticks([])
        axs[0].set_xlabel(f'{a.shape[1]} Layers')
        axs[0].set_yticks([])
        axs[0].set_ylabel('IDs')
        axs[1].set_title('act')
        axs[1].set_xticks([])
        axs[1].set_xlabel(f'{a_neuron.shape[1]} Layers')
        axs[1].set_yticks([])
        fig.suptitle(f'Encoding Frequency - {self.model_structure}', fontsize=10)
        fig.subplots_adjust(top=0.8)
        fig.colorbar(im, ax=axs, orientation='vertical')
        plt.savefig(self.dest_Encode+'Encoding_frequency_of_layers_merged.png', bbox_inches='tight')
        plt.savefig(self.dest_Encode+'Encoding_frequency_of_layers_merged.eps', bbox_inches='tight', format='eps')
        plt.close()

    def draw_encode_frequency_for_each_layer(self):         # encoding frequency for each layer
        print('[Codinfo] Executing draw_encode_frequency_for_each_layer...')
        if self.mode == 'reload_encode_dict':
            encode_class_dict = self.reload_and_revocer_encode_class_dict()
        else:
            encode_class_dict = self.recover_encode_class_dict()  

        #make dir fo the image batch
        save_folder = os.path.join(self.dest_Encode, 'Each_Layer_Encoding_Performance/')
        utils_.make_dir(save_folder)
        occ_list = []
        for layer in self.layers:    # for each layer
            occurrences = []
            for i in range(self.num_classes):     # for each ID
                occ = encode_class_dict[layer].count(i + 1)  # calculate the frequency of each ID [one value]
                occurrences.append(occ) # store frequency for each ID [list of 50]
            occ_list.append(occurrences)    # merge the frequency info for all layers [list of len(layers)]
            x = np.arange(self.num_classes)+1
            plt.figure()
            plt.bar(x, occurrences, width=0.5)
            plt.xticks(np.arange(0, self.num_classes+1, step=2))
            plt.xlabel('IDs')
            plt.ylabel('Frequrency')
            plt.title('Encoded ID frequency: '+layer+'\n\u03F4: 2std')
            plt.savefig(save_folder+layer+'_encoding_performance.png', bbox_inches='tight', dpi=100)
            plt.close()
    
    def draw_merged_encode_frequency_for_each_layer(self):      # encoding frequency for each layer, basiccaly this is the merged version of the last one
        print('[Codinfo] Executing draw_merged_encode_frequency_for_each_layer...')
        if self.mode == 'reload_encode_dict':
            encode_class_dict = self.reload_and_revocer_encode_class_dict()
        else:
            encode_class_dict = self.recover_encode_class_dict()
        
        fig, axs = plt.subplots(self.subplot_row, self.subplot_col, figsize=((self.subplot_col*2, self.subplot_row*2)))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        x = np.arange(self.num_classes)+1
        cnt_row = 0
        cnt_col = 0
        
        for layer in self.layers:    # for each layer
            occurrences = []
            for i in range(self.num_classes):     # for each ID
                occ = encode_class_dict[layer].count(i + 1)
                occurrences.append(occ)
            axs[cnt_row, cnt_col].bar(x, occurrences, width=0.5)
            axs[cnt_row, cnt_col].set_title(layer, fontsize=8)
            cnt_col += 1        # set subplot location
            if cnt_col == self.subplot_col:
                cnt_col = 0
                cnt_row += 1
        for ax in axs.flat:
            ax.label_outer()
        for ax in axs.flat:
            ax.set_ylabel(ylabel='Freq', fontsize=8, labelpad=0)
            ax.set_xlabel(xlabel='IDs', fontsize=8, labelpad=0)
        plt.tight_layout()
        plt.savefig(self.dest_Encode+'single_layer_encoding_performance.png', bbox_inches='tight', dpi=100)
        plt.close()

    
if __name__ == "__main__":
    
    model_ = vgg.__dict__['vgg16'](num_classes=50)
    layers, neurons, shapes = utils_.generate_vgg_layers_list_ann(model_, 'vgg16')

    root_dir = '/home/acxyle-workstation/Downloads'

    selectivity_analyzer = Encode_feaquency_analyzer(root=os.path.join(root_dir, 'Face Identity Baseline/'), 
                                                     layers=layers, neurons=neurons)
    
    #selectivity_analyzer.obtain_encode_class_dict()
    #selectivity_analyzer.selectivity_encode_layer_percent_plot()
    
    #selectivity_analyzer.draw_encode_frequency()
    #selectivity_analyzer.draw_encode_frequency_for_each_layer()
    #selectivity_analyzer.draw_merged_encode_frequency_for_each_layer()
