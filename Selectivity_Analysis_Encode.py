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
    
    Task: Sept 17, 2023
        
        1) devide computation and plot
        2) further divide the unit types
        2.1) refer to the threshold in biological neuron to remove less active unit
        2.2) use 'ref (mean(mean_values)+2std(mean_values))' as an additional condition to select spontaneous active unites
    
    Task: Sept 22, 2023
    
        1) refer to ANOVA, write function(s) to compare results from different modules
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
from scipy.interpolate import interp1d, RegularGridInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator
from collections import Counter

from scipy.stats import gaussian_kde
from matplotlib import gridspec

import vgg, resnet
import utils_


class Encode_feaquency_analyzer():
    """
        The the basic design, the feature map and encode_dict follows the lexical order. While the figure and Encode_id_unit_dict.pkl
        follows natural order
    
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
        
        self.ANOVA_idces = utils_.pickle_load(os.path.join(self.dest, 'ANOVA/ANOVA_idces.pkl'))   # <- consider to remove this?
        
        if layers == None or neurons == None:
            raise RuntimeError('[Codwarning] invalid layers and neurons')
            
        self.model_structure = root.split('/')[-2].split(' ')[2]
        
    def obtain_encode_class_dict(self, ):
        """
            this function generates the encode_dict, 1-based dict containing encoded classes, and sort_dict, different types of units
        """
        print('[Codinfo] Executing obtain_encode_class_dict...')
        
        encode_dict_path = os.path.join(self.dest_Encode, 'Encode_dict.pkl')
        sort_dict_path = os.path.join(self.dest_Encode, 'Sort_dict.pkl')
        
        if os.path.exists(encode_dict_path) and os.path.exists(sort_dict_path):
            
            self.Encode_dict = utils_.pickle_load(encode_dict_path)
            self.Sort_dict = utils_.pickle_load(sort_dict_path)
        
        else:
        
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
            
            for layer in self.layers:     # for each layer
                
                feature = utils_.pickle_load(os.path.join(self.root, f'{layer}.pkl'))      # load feature matrix
    
                s = self.ANOVA_idces[layer]     # sensitive_idx
                ns = np.array(list(set(np.arange(feature.shape[1]))-set(s)))     # non_sensitive_idx
                
                unit_encode_dict = {}
    
                # ----- the encode ID is 1-based
                with Parallel(n_jobs=num_workers) as parallel:
                    pl = parallel(delayed(encode_calculation)(feature, i) for i in tqdm(range(feature.shape[1]), desc=f'[{layer}] Encode'))  
                # -----
                
                unit_encode_dict = {i: pl[i] for i in range(len(pl))}     # convert the returned list to dict
                
                self.Encode_dict[layer] = unit_encode_dict     # save as dict for further use
                
                gc.collect()

                # ----- 2. encode test
                si = np.array([_ for _ in unit_encode_dict.keys() if len(unit_encode_dict[_]['encode']) == 1])     # 10
                wsi = np.array([_ for _ in unit_encode_dict.keys() if len(unit_encode_dict[_]['weak_encode']) == 1 and len(unit_encode_dict[_]['encode']) == 0])     
                wsi = np.setdiff1d(wsi, si)     # 499
            
                mi = np.array([_ for _ in unit_encode_dict.keys() if len(unit_encode_dict[_]['encode']) > 1])     # 3
                wmi = np.array([_ for _ in unit_encode_dict.keys() if len(unit_encode_dict[_]['weak_encode']) > 1 and len(unit_encode_dict[_]['encode']) == 0])     
                wmi = np.setdiff1d(wmi, mi)     # 949
        
                non_encode = np.array([_ for _ in unit_encode_dict.keys() if len(unit_encode_dict[_]['weak_encode']) == 0 and len(unit_encode_dict[_]['encode']) == 0])
                
                # ----- 3. advanced types
                s_si = np.intersect1d(s, si)     
                s_wsi = np.intersect1d(s, wsi)     
                s_mi = np.intersect1d(s, mi)     
                s_wmi = np.intersect1d(s, wmi)     
                s_non_encode = np.intersect1d(s, non_encode)     
                
                ns_si = np.intersect1d(ns, si)     
                ns_wsi = np.intersect1d(ns, wsi)     
                ns_mi = np.intersect1d(ns, mi)     
                ns_wmi = np.intersect1d(ns, wmi)     
                ns_non_encode = np.intersect1d(ns, non_encode)     
                
                unit_sort_dict = {
                    
                    'basic_type': {
                        'si_idx': si,
                        'wsi_idx': wsi,
                        'mi_idx': mi,
                        'wmi_idx': wmi,
                        'non_encode_idx': non_encode
                        },
                    
                    'advanced_type': {
                        'sensitive_si': s_si,
                        'sensitive_wsi': s_wsi,
                        'sensitive_mi': s_mi,
                        'sensitive_wmi': s_wmi,
                        'sensitive_non_encode': s_non_encode,
                        'non_sensitive_si': ns_si,
                        'non_sensitive_wsi': ns_wsi,
                        'non_sensitive_mi': ns_mi,
                        'non_sensitive_wmi': ns_wmi,
                        'non_sensitive_non_encode': ns_non_encode
                        }
                    }
                
                self.Sort_dict[layer] = unit_sort_dict
                
            print('[Codinfo] Saving Sort_dict.pkl...')
            utils_.pickle_dump(sort_dict_path, self.Sort_dict)
    
            print('[Codinfo] Saving Encode_dict.pkl...')
            utils_.pickle_dump(encode_dict_path, self.Encode_dict)     
            
            print('[Codinfo] Encode_dict.pkl saved in {}'.format(self.dest_Encode))

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
        if intersection_x is not None:
            intersection_y = f1(intersection_x)
        else:
            intersection_y = None
        
        return intersection_x, intersection_y
    
    # ----- under construction...
    def encode_layer_percent_plot(self, fig_folder, fig, ax, layers=None, curve_dict=None, point_dict=None, title=None, save=True):
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
        if layers == None:
            layers = self.layers
        ax.set_xticks(np.arange(len(layers)), layers, rotation='vertical', fontname='Times New Roman')
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
    
    def selectivity_encode_layer_percent_plot_all_operation(self, fig_folder, figs, axes,  
                                                            non_encode_unit_percentages, 
                                                            si_unit_percentages, 
                                                            wsi_unit_percentages, 
                                                            mi_unit_percentages, 
                                                            wmi_unit_percentages, 
                                                            encode_unit_percentages, 
                                                            weak_encode_unit_percentages,
                                                            
                                                            sensitive_non_encode_unit_percentages, 
                                                            sensitive_encode_unit_percentages, 
                                                            sensitive_weak_encode_unit_percentages, 
                                                            sensitive_si_unit_percentages, 
                                                            sensitive_wsi_unit_percentages, 
                                                            sensitive_mi_unit_percentages, 
                                                            sensitive_wmi_unit_percentages,
                                                            
                                                            non_sensitive_non_encode_unit_percentages, 
                                                            non_sensitive_encode_unit_percentages, 
                                                            non_sensitive_weak_encode_unit_percentages, 
                                                            non_sensitive_si_unit_percentages, 
                                                            non_sensitive_wsi_unit_percentages, 
                                                            non_sensitive_mi_unit_percentages, 
                                                            non_sensitive_wmi_unit_percentages,
                                                            
                                                            sensitive_percentages, 
                                                            non_sensitive_percentages,
                                                            
                                                            layers=None, saperate_fig=False):
        
        # ----- Fig 1
        inter_x, inter_y = self.calculate_intersection_point(encode_unit_percentages, non_encode_unit_percentages)
        inter_x_s, inter_y_s = self.calculate_intersection_point(si_unit_percentages, non_encode_unit_percentages)
        inter_x_m, inter_y_m = self.calculate_intersection_point(mi_unit_percentages, non_encode_unit_percentages)
        
        curve_dict = {
            'non_encode_unit': self.encode_layer_percent_plot_dict(values=non_encode_unit_percentages, label='non_encode'),
            'si_unit': self.encode_layer_percent_plot_dict(values=si_unit_percentages, label='si', linestyle='--'),
            'mi_unit': self.encode_layer_percent_plot_dict(values=mi_unit_percentages, label='mi', linestyle='--'),
            'encode_unit': self.encode_layer_percent_plot_dict(values=encode_unit_percentages, label='encode (si+mi)', ),
            
            'wsi_unit': self.encode_layer_percent_plot_dict(values=wsi_unit_percentages, label='wsi', linestyle='dotted'),
            'wmi_unit': self.encode_layer_percent_plot_dict(values=wmi_unit_percentages, label='wmi', linestyle='dotted'),
            'weak_encode_unit': self.encode_layer_percent_plot_dict(values=weak_encode_unit_percentages, color='skyblue', label='weak_encode (wsi+wmi)', linestyle='dotted'),
            }
        
        point_dict = {
            'intersect_e': self.encode_layer_percent_plot_dict(point={'x':inter_x,'y':inter_y}, 
                                                               #label=f'intersect_e {inter_x:.2f}'
                                                               ),
            'intersect_s': self.encode_layer_percent_plot_dict(point={'x':inter_x_s,'y':inter_y_s}, 
                                                               #label=f'intersect_s {inter_x_s:.2f}'
                                                               ),
            'intersect_m': self.encode_layer_percent_plot_dict(point={'x':inter_x_m,'y':inter_y_m}, 
                                                               #label=f'intersect_m {inter_x_m:.2f}'
                                                               ),
            }
        
        # ---
        self.encode_layer_percent_plot(fig_folder, figs, axes[0,0], layers, curve_dict, point_dict, "encode vs non_encode", False)
        
        if saperate_fig:
            fig,ax = plt.subplots(figsize=(12,6))
            self.encode_layer_percent_plot(fig_folder, fig, ax, layers, curve_dict, point_dict, "encode vs non_encode")
        # -----
        
        # ----- Fig 2
        inter_x, inter_y = self.calculate_intersection_point(sensitive_encode_unit_percentages, sensitive_non_encode_unit_percentages)
        inter_x_s, inter_y_s = self.calculate_intersection_point(sensitive_si_unit_percentages, sensitive_non_encode_unit_percentages)
        inter_x_m, inter_y_m = self.calculate_intersection_point(sensitive_mi_unit_percentages, sensitive_non_encode_unit_percentages)

        curve_dict = {
            'sensitive_encode_unit': self.encode_layer_percent_plot_dict(values=sensitive_encode_unit_percentages, label='sensitive_e(si+mi)', linewidth=1.0),
            'sensitive_si_unit': self.encode_layer_percent_plot_dict(values=sensitive_si_unit_percentages, label='sensitive_si', linestyle='--', linewidth=1.0),
            'sensitive_mi_unit': self.encode_layer_percent_plot_dict(values=sensitive_mi_unit_percentages, label='sensitive_mi', linestyle='--', linewidth=1.0),
            
            'sensitive_weak_encode_unit': self.encode_layer_percent_plot_dict(values=sensitive_weak_encode_unit_percentages, label='sensitive_we(wsi+wmi)', linewidth=1.0),
            'sensitive_wsi_unit': self.encode_layer_percent_plot_dict(values=sensitive_wsi_unit_percentages, label='sensitive_wsi', linestyle='dotted', linewidth=1.0),
            'sensitive_wmi_unit': self.encode_layer_percent_plot_dict(values=sensitive_wmi_unit_percentages, label='sensitive_wmi', linestyle='dotted', linewidth=1.0),
            
            'sensitive_non_encode_unit': self.encode_layer_percent_plot_dict(values=sensitive_non_encode_unit_percentages, label='sensitive_non_encode', linewidth=1.0),
            'sensitive_unit':self.encode_layer_percent_plot_dict(values=sensitive_percentages, label='sensitive', linewidth=3.0)
            }
        
        point_dict = {
            'intersect_s-e': self.encode_layer_percent_plot_dict(point={'x':inter_x,'y':inter_y}, 
                                                                 #label=f'intersect_s-e {inter_x:.2f}'
                                                                 ),
            'intersect_s-s': self.encode_layer_percent_plot_dict(point={'x':inter_x_s,'y':inter_y_s}, 
                                                                 #label=f'intersect_s-s {inter_x_s:.2f}'
                                                                 ),
            'intersect_s-m': self.encode_layer_percent_plot_dict(point={'x':inter_x_m,'y':inter_y_m}, 
                                                                 #label=f'intersect_s-m {inter_x_m:.2f}'
                                                                 ),
            }
        
        self.encode_layer_percent_plot(fig_folder, figs, axes[0,1], layers, curve_dict, point_dict, "sensitive", False)
        
        if saperate_fig:
            fig,ax = plt.subplots(figsize=(12,6))
            self.encode_layer_percent_plot(fig_folder, fig, ax, layers, curve_dict, point_dict, "sensitive")
        
        # ----- Fig 3
        curve_dict = {
            'non_sensitive_encode_unit': self.encode_layer_percent_plot_dict(values=non_sensitive_encode_unit_percentages, label='non_sensitive_e', linewidth=1.0),
            'non_sensitive_si_unit': self.encode_layer_percent_plot_dict(values=non_sensitive_si_unit_percentages, label='non_sensitive_si', linestyle='--', linewidth=1.0),
            'non_sensitive_mi_unit': self.encode_layer_percent_plot_dict(values=non_sensitive_mi_unit_percentages, label='non_sensitive_mi', linestyle='--', linewidth=1.0),
            
            'non_sensitive_weak_encode_unit': self.encode_layer_percent_plot_dict(values=non_sensitive_weak_encode_unit_percentages, label='non_sensitive_we', linewidth=1.0),
            'non_sensitive_wsi_unit': self.encode_layer_percent_plot_dict(values=non_sensitive_wsi_unit_percentages, label='non_sensitive_wsi', linestyle='dotted', linewidth=1.0),
            'non_sensitive_wmi_unit': self.encode_layer_percent_plot_dict(values=non_sensitive_wmi_unit_percentages, label='non_sensitive_wmi', linestyle='dotted', linewidth=1.0),
            
            'non_sensitive_non_encode_unit': self.encode_layer_percent_plot_dict(values=non_sensitive_non_encode_unit_percentages, label='non_sensitive_non_encode', linewidth=1.0),
            'non_sensitive_unit':self.encode_layer_percent_plot_dict(values=non_sensitive_percentages, label='non_sensitive', linewidth=3.0)
            }
        
        point_dict = None
        
        # ---
        self.encode_layer_percent_plot(fig_folder, figs, axes[1,0], layers, curve_dict, point_dict, "non_sensitive", False)
        if saperate_fig:
            fig,ax = plt.subplots(figsize=(12,6))
            self.encode_layer_percent_plot(fig_folder, fig, ax, layers, curve_dict, point_dict, "non_sensitive")
        
        # ----- Fig 4
        curve_dict = {
            'sensitive_unit': self.encode_layer_percent_plot_dict(values=sensitive_percentages, color='purple', label='sensitive', linewidth=3.0),
            'non_sensitive_unit': self.encode_layer_percent_plot_dict(values=non_sensitive_percentages, color='purple', label='non_sensitive', linestyle='--'),
            
            'encode_unit': self.encode_layer_percent_plot_dict(values=encode_unit_percentages, color='blue', label='encode', linewidth=3.0),
            'weak_encode_unit': self.encode_layer_percent_plot_dict(values=weak_encode_unit_percentages, color='skyblue', label='weak_encode (wsi+wmi)', linestyle='dotted'),
            
            'non_encode_unit': self.encode_layer_percent_plot_dict(values=non_encode_unit_percentages, color='blue', label='non_encode', linestyle='--'),
            }
        point_dict = None
        
        # ---
        self.encode_layer_percent_plot(fig_folder, figs, axes[1,1], layers, curve_dict, point_dict, "sensitive and encode", False)
        if saperate_fig:
            fig,ax = plt.subplots(figsize=(12,6))
            self.encode_layer_percent_plot(fig_folder, fig, ax, layers, curve_dict, point_dict, "sensitive and encode")
        # -----
        
    def selectivity_encode_layer_percent_plot(self, ):
        """
            this function plot the percentages of different types of units over layers
        """
        
        fig_folder = os.path.join(self.dest_Encode, 'Figures')
        utils_.make_dir(fig_folder)
        
        if not hasattr(self, 'Sort_dict'):
            try:    
                self.Sort_dict = utils_.pickle_load(os.path.join(self.dest_Encode, 'Sort_dict.pkl'))
            except(RuntimeError):
                self.Sort_dict = utils_.pickle_load_gzip(os.path.join(self.dest_Encode, 'Sort_dict.gz'))
                
        print('[Codinfo] preparing plot...')
        # 3 types of basic units: ['si_idx', 'mi_idx', 'non_encode_idx']

        # ----- recover stats
        non_encode_unit_percentages = [len(self.Sort_dict[_]['basic_type']['non_encode_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        
        si_unit_percentages = [len(self.Sort_dict[_]['basic_type']['si_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        wsi_unit_percentages = [len(self.Sort_dict[_]['basic_type']['wsi_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        
        mi_unit_percentages = [len(self.Sort_dict[_]['basic_type']['mi_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        wmi_unit_percentages = [len(self.Sort_dict[_]['basic_type']['wmi_idx'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        
        encode_unit_percentages = [si_unit_percentages[_]+mi_unit_percentages[_] for _ in range(len(self.layers))]
        weak_encode_unit_percentages = [wsi_unit_percentages[_]+wmi_unit_percentages[_] for _ in range(len(self.layers))]
        
        # -----
        sensitive_non_encode_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['sensitive_non_encode'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        
        sensitive_si_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['sensitive_si'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        sensitive_wsi_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['sensitive_wsi'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        
        sensitive_mi_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['sensitive_mi'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        sensitive_wmi_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['sensitive_wmi'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        
        sensitive_encode_unit_percentages = [sensitive_si_unit_percentages[_]+sensitive_mi_unit_percentages[_] for _ in range(len(self.layers))]
        sensitive_weak_encode_unit_percentages = [sensitive_wsi_unit_percentages[_]+sensitive_wmi_unit_percentages[_] for _ in range(len(self.layers))]
        
        # ---
        non_sensitive_non_encode_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['non_sensitive_non_encode'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        
        non_sensitive_si_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['non_sensitive_si'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        non_sensitive_wsi_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['non_sensitive_wsi'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        
        non_sensitive_mi_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['non_sensitive_mi'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        non_sensitive_wmi_unit_percentages = [len(self.Sort_dict[_]['advanced_type']['non_sensitive_wmi'])/self.neurons[idx]*100 for idx, _ in enumerate(self.layers)]
        
        non_sensitive_encode_unit_percentages = [non_sensitive_si_unit_percentages[_]+non_sensitive_mi_unit_percentages[_] for _ in range(len(self.layers))]
        non_sensitive_weak_encode_unit_percentages = [non_sensitive_wsi_unit_percentages[_]+non_sensitive_wmi_unit_percentages[_] for _ in range(len(self.layers))]
        
        # ---
        sensitive_percentages = [sensitive_encode_unit_percentages[_]+sensitive_weak_encode_unit_percentages[_]+sensitive_non_encode_unit_percentages[_] for _ in range(len(self.layers))]
        non_sensitive_percentages = [non_sensitive_encode_unit_percentages[_]+non_sensitive_weak_encode_unit_percentages[_]+non_sensitive_non_encode_unit_percentages[_] for _ in range(len(self.layers))]
        
        # ----- all operation
        figs, axes = plt.subplots(2,2,figsize=(24,12))
        
        self.selectivity_encode_layer_percent_plot_all_operation(fig_folder, figs, axes,  
                                                                non_encode_unit_percentages, 
                                                                si_unit_percentages, 
                                                                wsi_unit_percentages, 
                                                                mi_unit_percentages, 
                                                                wmi_unit_percentages, 
                                                                encode_unit_percentages, 
                                                                weak_encode_unit_percentages,
                                                                
                                                                sensitive_non_encode_unit_percentages, 
                                                                sensitive_encode_unit_percentages, 
                                                                sensitive_weak_encode_unit_percentages, 
                                                                sensitive_si_unit_percentages, 
                                                                sensitive_wsi_unit_percentages, 
                                                                sensitive_mi_unit_percentages, 
                                                                sensitive_wmi_unit_percentages,
                                                                
                                                                non_sensitive_non_encode_unit_percentages, 
                                                                non_sensitive_encode_unit_percentages, 
                                                                non_sensitive_weak_encode_unit_percentages, 
                                                                non_sensitive_si_unit_percentages, 
                                                                non_sensitive_wsi_unit_percentages, 
                                                                non_sensitive_mi_unit_percentages, 
                                                                non_sensitive_wmi_unit_percentages,
                                                                
                                                                sensitive_percentages, 
                                                                non_sensitive_percentages,
                                                                layers=None, saperate_fig=False)
        
        title = 'sensitive_and_encode_all'
        figs.subplots_adjust(hspace=0.5, wspace=0.1)
        figs.savefig(os.path.join(fig_folder, title+'.png'), bbox_inches='tight')
        figs.savefig(os.path.join(fig_folder, title+'.eps'), bbox_inches='tight', format='eps')     # no transparency
        figs.savefig(os.path.join(fig_folder, title+'.svg'), bbox_inches='tight', format='svg', transparent=True)
        plt.close()
        
        # ----- activation function
        act_idx, act_layers, _ = utils_.imaginary_neurons_vgg(self.layers)
        figs_act, axes_act = plt.subplots(2,2,figsize=(24,12))
        
        self.selectivity_encode_layer_percent_plot_all_operation(fig_folder, figs_act, axes_act,  
                                                                 
                                                                [non_encode_unit_percentages[_] for _ in act_idx], 
                                                                [si_unit_percentages[_] for _ in act_idx], 
                                                                [wsi_unit_percentages[_] for _ in act_idx], 
                                                                [mi_unit_percentages[_] for _ in act_idx], 
                                                                [wmi_unit_percentages[_] for _ in act_idx], 
                                                                [encode_unit_percentages[_] for _ in act_idx], 
                                                                [weak_encode_unit_percentages[_] for _ in act_idx], 
                                                                
                                                                [sensitive_encode_unit_percentages[_] for _ in act_idx], 
                                                                [sensitive_non_encode_unit_percentages[_] for _ in act_idx], 
                                                                [sensitive_weak_encode_unit_percentages[_] for _ in act_idx], 
                                                                [sensitive_si_unit_percentages[_] for _ in act_idx], 
                                                                [sensitive_wsi_unit_percentages[_] for _ in act_idx], 
                                                                [sensitive_mi_unit_percentages[_] for _ in act_idx], 
                                                                [sensitive_wmi_unit_percentages[_] for _ in act_idx], 
                                                                
                                                                [non_sensitive_encode_unit_percentages[_] for _ in act_idx], 
                                                                [non_sensitive_non_encode_unit_percentages[_] for _ in act_idx], 
                                                                [non_sensitive_weak_encode_unit_percentages[_] for _ in act_idx], 
                                                                [non_sensitive_si_unit_percentages[_] for _ in act_idx], 
                                                                [non_sensitive_wsi_unit_percentages[_] for _ in act_idx], 
                                                                [non_sensitive_mi_unit_percentages[_] for _ in act_idx], 
                                                                [non_sensitive_wmi_unit_percentages[_] for _ in act_idx], 
                                                                
                                                                [sensitive_percentages[_] for _ in act_idx], 
                                                                [non_sensitive_percentages[_] for _ in act_idx],
                                                                
                                                                layers=act_layers, saperate_fig=False)
        
        title = 'sensitive_and_encode_all_neuron'
        figs_act.subplots_adjust(hspace=0.5, wspace=0.1)
        figs_act.savefig(os.path.join(fig_folder, title+'.png'), bbox_inches='tight')
        figs_act.savefig(os.path.join(fig_folder, title+'.eps'), bbox_inches='tight', format='eps')     # no transparency
        figs_act.savefig(os.path.join(fig_folder, title+'.svg'), bbox_inches='tight', format='svg', transparent=True)
        plt.close()
        
    def draw_encode_frequency_layers(self, freq, ):
        """
            this functions provides the encode frequency of each layer
        """
        print('[Codinfo] Executing encode frequency layers...')
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        fig_folder = os.path.join(self.dest_Encode, 'Figures')
        utils_.make_dir(fig_folder)
        
        frequency_folder = os.path.join(fig_folder, 'Frequency_layer')
        
        if os.path.exists(frequency_folder):
            pass
        
        else:
            utils_.make_dir(frequency_folder)
            for _ in range(freq.shape[1]):
                fig,ax=plt.subplots()
                ax.bar(np.arange(1,51), freq[:,_])
                
                for __ in range(1,4):
                    rect = plt.Rectangle((0, np.mean(freq[:,_])), 50, utils_.generate_threshold(freq[:,_], alpha=0, delta=__), edgecolor=None, facecolor='blue', alpha=0.25)
                    ax.add_patch(rect)
                
                ax.set_title(self.layers[_])
                ax.set_ylim([np.min(freq), 1.1*np.max(freq)])
                fig.savefig(os.path.join(frequency_folder, self.layers[_]+'.png'), bbox_inches='tight')
                plt.close()
        
        
    def generate_freq_map(self, ):
        
        print('[Codinfo] generating freq map...')
        
        if not hasattr(self, 'Encode_dict') and not hasattr(self, 'Sort_dict'):
            print('[Codinfo] Loading Encode_dict and Sort_dict...')
            self.Encode_dict = utils_.pickle_load(os.path.join(self.dest_Encode, 'Encode_dict.pkl'))
            self.Sort_dict = utils_.pickle_load(os.path.join(self.dest_Encode, 'Sort_dict.pkl'))
        
        freq_path = os.path.join(self.dest_Encode, 'freq.pkl')
        
        if os.path.exists(freq_path):
            freq_dict = utils_.pickle_load(freq_path)
        else:
            correct_id = utils_.lexicographic_order(50)+1
            
            freq_dict = {
                'all_encode': self.generate_freq_map_single(correct_id, criteria='strong'),
                'all_weak_encode': self.generate_freq_map_single(correct_id, criteria='weak'),
                'all_all_encode': self.generate_freq_map_single(correct_id),
                
                'sensitive_si': self.generate_freq_map_single(correct_id, ['advanced_type'], ['sensitive_si']),
                'sensitive_mi': self.generate_freq_map_single(correct_id, ['advanced_type'], ['sensitive_mi']),
                'non_sensitive_si': self.generate_freq_map_single(correct_id, ['advanced_type'], ['non_sensitive_si']),
                'non_sensitive_mi': self.generate_freq_map_single(correct_id, ['advanced_type'], ['non_sensitive_mi']),
                
                'sensitive_wsi': self.generate_freq_map_single(correct_id, ['advanced_type'], ['sensitive_wsi']),
                'sensitive_wmi': self.generate_freq_map_single(correct_id, ['advanced_type'], ['sensitive_wmi']),
                'non_sensitive_wsi': self.generate_freq_map_single(correct_id, ['advanced_type'], ['non_sensitive_wsi']),
                'non_sensitive_wmi': self.generate_freq_map_single(correct_id, ['advanced_type'], ['non_sensitive_wmi']),
                
                'sensitive_all_si': self.generate_freq_map_single(correct_id, ['advanced_type'], ['sensitive_si', 'sensitive_wsi']),
                'sensitive_all_mi': self.generate_freq_map_single(correct_id, ['advanced_type'], ['sensitive_mi', 'sensitive_wmi']),
                'non_sensitive_all_si': self.generate_freq_map_single(correct_id, ['advanced_type'], ['non_sensitive_si', 'non_sensitive_wsi']),
                'non_sensitive_all_mi': self.generate_freq_map_single(correct_id, ['advanced_type'], ['non_sensitive_mi', 'non_sensitive_wmi']),
                
                'sensitive_non_encode': self.generate_freq_map_single(correct_id, ['advanced_type'], ['sensitive_non_encode']),
                'non_sensitive_non_encode': self.generate_freq_map_single(correct_id, ['advanced_type'], ['non_sensitive_non_encode']),
                }

            utils_.pickle_dump(freq_path, freq_dict)
        
        return freq_dict
    
    def generate_freq_map_single(self, correct_id, unit_type_1=None, unit_type_2=None, criteria=None):

        freq = []

        for layer_idx, layer in tqdm(enumerate(self.layers), desc=f'{unit_type_2}'):
            
            unit_encode_dict = self.Encode_dict[layer]
            
            if unit_type_1 != None and unit_type_2 != None:
                
                target_units = np.array([])
                for type_1 in unit_type_1:
                    for type_2 in unit_type_2:
                        target_units = np.concatenate((target_units, self.Sort_dict[layer][type_1][type_2]))     
                
                if len(unit_type_2) == 1 and 'w' in unit_type_2[0]:
                    unit_encode_list = [unit_encode_dict[unit]['weak_encode'] for unit in target_units]
                elif len(unit_type_2) == 1 and 'non_encode' not in unit_type_2[0]:
                    unit_encode_list = [unit_encode_dict[unit]['encode'] for unit in target_units]
                elif len(unit_type_2) == 1 and 'non_encode' in unit_type_2[0]:
                    unit_encode_list = [[*unit_encode_dict[unit]['weak_encode'], *unit_encode_dict[unit]['encode']] for unit in target_units]
                elif len(unit_type_2) == 2:
                    unit_encode_list = [[*unit_encode_dict[unit]['weak_encode'], *unit_encode_dict[unit]['encode']] for unit in target_units]
                    
            else:
                target_units = np.arange(self.neurons[layer_idx])
                if criteria == None:
                    unit_encode_list = [[*unit_encode_dict[unit]['weak_encode'], *unit_encode_dict[unit]['encode']] for unit in target_units]
                elif criteria == 'weak':
                    unit_encode_list = [unit_encode_dict[unit]['weak_encode'] for unit in target_units]
                elif criteria == 'strong':
                    unit_encode_list = [unit_encode_dict[unit]['encode'] for unit in target_units]
           
            pool = [id_ for encoded_ids in unit_encode_list for id_ in encoded_ids]     # for all unit
            frequency = Counter(pool)
            
            frequency = {correct_id[_-1]:frequency[_] for _ in range(1,51)}     # map correct_id
            frequency = {_: frequency[_]/self.neurons[layer_idx] for _ in range(1,51)}     # sort correct_id
            
            freq.append(np.array(list(frequency.values())))
            
        freq = np.array(freq).T      # (num_layers, num_classes) -> (num_classes, num_layers)
        
        return freq
    
    def draw_encode_frequency_component(self, x_position, y_position, x_width, x_height,
                                        sub_x_position, sub_y_position, sub_x_step, sub_y_step, sub_width=0.175, sub_height=0.1025,
                                        freq_dict=None, title=None, vmin=0., vmax=1., cmap=None, idx=None, label_on=False, sub_dict=None):
        
        x = 0
        y = 0
        
        ax = plt.gcf().add_axes([x_position, y_position, x_width, x_height])
        freq = freq_dict[f'{title}']
        ax.imshow(freq, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(f'{title}')
        
        if label_on == True:
            ax.set_xticks(np.arange(len(self.layers)))
            ax.set_xticklabels(['' if _ not in idx else self.layers[_] for _ in range(len(self.layers))], rotation='vertical')
            ax.set_yticks(np.arange(0,50,5), np.arange(1,51,5))
        elif label_on == False:
            ax.set_xticks([])
            ax.set_yticks([])
            
        if sub_dict is not None:
            for key in sub_dict:
                freq = freq_dict[key]
                sub_ax = plt.gcf().add_axes([sub_x_position + sub_x_step*x, sub_y_position + sub_y_step*y, sub_width, sub_height])
                sub_ax.imshow(freq, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
                sub_ax.set_title(f'{key}')
                sub_ax.set_xticks([])
                sub_ax.set_yticks([])
                sub_ax.axis('off')
                x+=1
                if x == 2:
                    y = 1
                    x = 0
                    
                    
    # plot 3D - not in use
    def draw_encode_frequency_component_3D(self, x_position, y_position, x_width, x_height,
                                        sub_x_position, sub_y_position, sub_x_step, sub_y_step, sub_width=0.175, sub_height=0.1025,
                                        freq_dict=None, title=None, vmin=0., vmax=1., cmap=None, idx=None, label_on=False, sub_dict=None):
        x = np.arange(len(self.layers))
        y = np.arange(self.num_classes)
        X, Y = np.meshgrid(x, y)

        ax = plt.gcf().add_axes([x_position, y_position, x_width, x_height], projection='3d')
        ax.plot_surface(X, Y, freq_dict[f'{title}'], vmin=vmin, vmax=vmax, cmap=cmap)

        ax.set_ylabel('IDs')
        ax.set_zlabel('Normalized responses')
        ax.set_title(f'{title}')
        ax.set_zlim([vmin, vmax])
        ax.view_init(elev=30, azim=225)
        
        if label_on == True:
            ax.set_xticks(np.arange(len(self.layers)))
            ax.set_xticklabels(['' if _ not in idx else self.layers[_] for _ in range(len(self.layers))], rotation='vertical')
            ax.set_yticks(np.arange(0, 50, 5), np.arange(1, 51, 5))
            
            for label in ax.get_xticklabels():
                label.set_rotation(-50)  # 45 degree angle for x-axis tick labels
            for label in ax.get_yticklabels():
                label.set_rotation(-35)  # -45 degree angle for y-axis tick labels
                
        elif label_on == False:
            ax.set_xticks([])
            ax.set_yticks([])

        if sub_dict is not None:
            
            x = 0
            y = 0
            
            for key in sub_dict:
       
                sub_ax = plt.gcf().add_axes([sub_x_position + sub_x_step*x, sub_y_position + sub_y_step*y, sub_width, sub_height], projection='3d')
                sub_ax.plot_surface(X, Y, freq_dict[key], cmap=cmap, vmin=vmin, vmax=vmax)
                sub_ax.set_title(f'{key}')
                
                sub_ax.set_xticks(idx)
                sub_ax.set_xticklabels(['' for _ in idx])
                
                sub_ax.set_yticks(np.arange(0, 50, 5), np.arange(1, 51, 5))
                sub_ax.set_yticklabels(['' for _ in np.arange(0, 50, 5)])
                
                sub_ax.set_zlim(vmin, vmax)
                sub_ax.view_init(elev=30, azim=225)
                
                x+=1
                if x == 2:
                    y = 1
                    x = 0
        
        # ----- interpolation
        #x_fine_grid = np.linspace(0, freq.shape[1]-1, 1000)  # 10 times denser
        #y_fine_grid = np.linspace(0, freq.shape[0]-1, 1000)  # 10 times denser
        
        #ct_interp_full = CloughTocher2DInterpolator(list(zip(X.ravel(), Y.ravel())), freq.ravel())
        #Z_fine_ct = ct_interp_full(np.meshgrid(y_fine_grid, x_fine_grid)[0], np.meshgrid(y_fine_grid, x_fine_grid)[1])
        
        #fig = plt.figure(figsize=(20, 14))
        #ax = fig.add_subplot(111, projection='3d')
        #ax.plot_surface(np.meshgrid(y_fine_grid, x_fine_grid)[0], np.meshgrid(y_fine_grid, x_fine_grid)[1], Z_fine_ct, cmap='viridis')

        #ax.set_xlabel('X axis')
        #ax.set_ylabel('Y axis')
        #ax.set_zlabel('Z axis')
        #ax.set_title('Interpolation using CloughTocher2DInterpolator')
        #fig.colorbar(surf, shrink=0.5)
        #ax.view_init(elev=30, azim=225)
        
        #plt.tight_layout()
        #fig.savefig(os.path.join(fig_folder, '3D interp.png'), bbox_inches='tight')
        #fig.savefig(os.path.join(fig_folder, '3D interp.eps'), bbox_inches='tight', format='eps')
        #plt.close()
        # -----
        
                
    def draw_encode_frequency(self, draw_encode_frequency_layers=False):        # general figure for encoding frequency
        """
            [Sept 9] abandonded the use of dict_based frequency and the pd.DataFrame.from_dict(freq_dic) plotting
            
        """
        print('[Codinfo] draw encode frequency...')
        
        if not hasattr(self, 'Encode_dict') and not hasattr(self, 'Sort_dict'):
            print('[Codinfo] Loading Encode_dict and Sort_dict...')
            self.Encode_dict = utils_.pickle_load(os.path.join(self.dest_Encode, 'Encode_dict.pkl'))
            self.Sort_dict = utils_.pickle_load(os.path.join(self.dest_Encode, 'Sort_dict.pkl'))
            
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        print('[Codinfo] Executing draw_encode_frequency...')
        
        fig_folder = os.path.join(self.dest_Encode, 'Figures')
        utils_.make_dir(fig_folder)
        
        # -----
        freq_dict = self.generate_freq_map()
        # -----
        
        # ----- exterior plots
        if draw_encode_frequency_layers:
            self.draw_encode_frequency_layers(freq_dict['all_encode'])
        # -----
        
        # -----
        idx, layers, _ = utils_.imaginary_neurons_vgg(self.layers)
        
        vmin = 1.
        vmax = 0.
        for key in list(freq_dict.keys()):
            vmin = np.min([vmin, np.min(freq_dict[key])])
            vmax = np.max([vmax, np.max(freq_dict[key])])
        
        # ---
        encode_type_list = ['all_encode', 'sensitive_si', 'sensitive_mi', 'non_sensitive_si', 'non_sensitive_mi']
        weak_encode_type_list = ['all_weak_encode', 'sensitive_wsi', 'sensitive_wmi', 'non_sensitive_wsi', 'non_sensitive_wmi']
        all_encode_type_list = ['all_all_encode', 'sensitive_all_si', 'sensitive_all_mi', 'non_sensitive_all_si', 'non_sensitive_all_mi']
        
        # ----- raw 2D fig
        fig = plt.figure(figsize=(20, 30))
        cmap = 'turbo'
        
        self.draw_encode_frequency_component(x_position=0.15, y_position=0.7, x_width=0.25, x_height=0.225, 
                                             sub_x_position=0.425, sub_y_position=0.7, sub_x_step=0.145, sub_y_step=0.115, sub_width=0.130, sub_height=0.1025,
                                             freq_dict=freq_dict, title='all_all_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=True, sub_dict=all_encode_type_list[1:])
        
        # ---
        self.draw_encode_frequency_component(x_position=0.15, y_position=0.4, x_width=0.25, x_height=0.225, 
                                             sub_x_position=0.425, sub_y_position=0.4, sub_x_step=0.145, sub_y_step=0.115, sub_width=0.130, sub_height=0.1025,
                                             freq_dict=freq_dict, title='all_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=encode_type_list[1:])
        
        # ---
        self.draw_encode_frequency_component(x_position=0.15, y_position=0.15, x_width=0.25, x_height=0.225, 
                                             sub_x_position=0.425, sub_y_position=0.15, sub_x_step=0.145, sub_y_step=0.115, sub_width=0.130, sub_height=0.1025,
                                             freq_dict=freq_dict, title='all_weak_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=weak_encode_type_list[1:])
        
        # ---
        self.draw_encode_frequency_component(x_position=0.75, y_position=0.4, x_width=0.25, x_height=0.225, 
                                             sub_x_position=None, sub_y_position=None, sub_x_step=None, sub_y_step=None, sub_width=0.130, sub_height=0.1025,
                                             freq_dict=freq_dict, title='sensitive_non_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=None)

        # ---
        self.draw_encode_frequency_component(x_position=0.75, y_position=0.15, x_width=0.25, x_height=0.225, 
                                             sub_x_position=None, sub_y_position=None, sub_x_step=None, sub_y_step=None, sub_width=0.130, sub_height=0.1025,
                                             freq_dict=freq_dict, title='non_sensitive_non_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=None)
        
        # ---
        cax = fig.add_axes([1.02, 0.3, 0.01, 0.45]) 
        
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)   # plot colorbar based on arbitrary vmin and vmax
            
        fig.suptitle(f'layer - ID [{self.model_structure}]', x=0.55, y=0.95, fontsize=28)
        
        fig.savefig(os.path.join(fig_folder, 'layer and ID (2D).png'), bbox_inches='tight')
        fig.savefig(os.path.join(fig_folder, 'layer and ID (2D).eps'), bbox_inches='tight', format='eps')
        plt.close()
        
        
        # ----- raw 3D fig
        fig = plt.figure(figsize=(20, 30))
        cmap = 'turbo'
        
        self.draw_encode_frequency_component_3D(x_position=0.15, y_position=0.7, x_width=0.25, x_height=0.225,
                                            sub_x_position=0.425, sub_y_position=0.7, sub_x_step=0.13, sub_y_step=0.1125, sub_width=0.155, sub_height=0.1025,
                                            freq_dict=freq_dict, title='all_all_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                            label_on=True, sub_dict=all_encode_type_list[1:])
        
        # ---
        self.draw_encode_frequency_component_3D(x_position=0.15, y_position=0.4, x_width=0.25, x_height=0.225, 
                                             sub_x_position=0.425, sub_y_position=0.4, sub_x_step=0.13, sub_y_step=0.1125, sub_width=0.155, sub_height=0.1025,
                                             freq_dict=freq_dict, title='all_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=encode_type_list[1:])
        
        # ---
        self.draw_encode_frequency_component_3D(x_position=0.15, y_position=0.15, x_width=0.25, x_height=0.225, 
                                             sub_x_position=0.425, sub_y_position=0.15, sub_x_step=0.13, sub_y_step=0.1125, sub_width=0.155, sub_height=0.1025,
                                             freq_dict=freq_dict, title='all_weak_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=weak_encode_type_list[1:])
        
        # ---
        self.draw_encode_frequency_component_3D(x_position=0.75, y_position=0.4, x_width=0.25, x_height=0.225, 
                                             sub_x_position=None, sub_y_position=None, sub_x_step=None, sub_y_step=None, sub_width=0.155, sub_height=0.1025,
                                             freq_dict=freq_dict, title='sensitive_non_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=None)

        # ---
        self.draw_encode_frequency_component_3D(x_position=0.75, y_position=0.15, x_width=0.25, x_height=0.225, 
                                             sub_x_position=None, sub_y_position=None, sub_x_step=None, sub_y_step=None, sub_width=0.155, sub_height=0.1025,
                                             freq_dict=freq_dict, title='non_sensitive_non_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=None)
        
        cax = fig.add_axes([1.02, 0.3, 0.01, 0.45])
        
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        fig.suptitle(f'layer - ID (3D) [{self.model_structure}]', x=0.55, y=0.95, fontsize=28)
 
        plt.tight_layout()
        fig.savefig(os.path.join(fig_folder, 'layer and ID (3D).png'), bbox_inches='tight')
        fig.savefig(os.path.join(fig_folder, 'layer and ID (3D).eps'), bbox_inches='tight', format='eps')
        plt.close()
        
    #FIXME - the index problem, try to convert the lexical to natural order from the very beginning
    def generate_encoded_id_unit_idx(self, ):
        
        """
            [interesting] A interesting notice is, perhaps can use the same mean+2std to select what ID has been encoded by one layer, 
            and use all the units to build a sub-net the do other experiments...
        """
        
        print('[Codinfo] Generating_encoded_id_unit_idx...')
        
        correct_id = utils_.lexicographic_order(50)+1
        idx_folder = os.path.join(self.dest_Encode, 'unit_of_interested')
        utils_.make_dir(idx_folder)
        
        Encode_id_unit_dict_path = os.path.join(os.path.join(self.dest_Encode, 'Encode_id_unit_dict.pkl'))
        
        if os.path.exists(Encode_id_unit_dict_path):
            layer_dict = utils_.pickle_load(Encode_id_unit_dict_path)
        else:    
            freq_dict = self.generate_freq_map()
            layer_dict = {}
            for idx, layer in tqdm(enumerate(self.layers), desc='enc'):
                
                encoded_id_dict = {}
                test = list(self.Encode_dict[layer].values())
                for level in range(0,4):
                    encoded_id = np.where(freq_dict['all'][:,idx] >= utils_.generate_threshold(freq_dict['all'][:,idx], delta=level))[0]+1
     
                    test_dict = {}
                    if encoded_id.size>0:
                        for id_ in encoded_id:     # for each id (correct)
                            id__ = np.where(correct_id==id_)[0].item()+1     # id (lexical order)
                            
                            test_ = [[idx_,_] for idx_, _ in enumerate(test) if id__ in _]     # for unit encodes certain id
                            test_idx = [_[0] for _ in test_]
                            test_ = [_[1] for _ in test_]    
                            
                            test_1 = [[idx_,_] for idx_, _ in enumerate(test) if id__ in _ and len(_)==1]     # for unit ONLY encodes certain id
                            test_idx1 = [_[0] for _ in test_1]
                            #test_1 = [_[1] for _ in test_1]    
                            
                            test_u = []
                            [test_u.append(_) for _ in test_ if np.all(~np.isin(_, test_u))]
                            for idx_tmp, i in enumerate(test_u):
                                tmp = []
                                [tmp.append(correct_id[value_tmp-1]) for value_tmp in i]
                                test_u[idx_tmp] = tmp
                            
                            test_dict.update({id_.item(): {
                                'single': test_idx1,
                                'all': test_idx,
                                'combinations': test_u
                                }})
                    
                    encoded_id_dict.update({level: test_dict})
                
                layer_dict.update({layer: encoded_id_dict})
            
            print('[Codinfo] Saving Encoded_id_unit.pkl...')
            utils_.pickle_dump(Encode_id_unit_dict_path, layer_dict) 
            
        return layer_dict
    
    def single_acc(self, feature, idx, label, tqdm_bar=None):
        if len(idx) != 0:
            acc = utils_.SVM_classification(feature[:, idx], label, test_size=0.2, random_state=42)
        else:
            acc = 0. 
            
        if isinstance(tqdm_bar, tqdm):
            tqdm_bar.update(1)
        
        return acc
    
    def SVM(self,):
        """
            test version for merging with previous results
        """
        
        print('[Codinfo] computing SVM...')
        
        SVM_path = os.path.join(self.dest_Encode, 'SVM.pkl')
        
        if os.path.exists(SVM_path):
            layer_SVM = utils_.pickle_load(SVM_path)
            
        else:
            
            if not hasattr(self, 'Sort_dict'):
                self.Sort_dict = utils_.pickle_load(os.path.join(self.dest_Encode, 'Sort_dict.pkl'))
            
            ANOVA_idces = utils_.pickle_load(os.path.join(self.dest, 'ANOVA/ANOVA_idces.pkl'))
            
            # ----- for SVM
            label = utils_.lexicographic_order(50)+1
            label = np.repeat(label, 10)
            
            layer_SVM = {}
            
            for layer in self.layers:
                
                tqdm_bar = tqdm(total=31, desc=f'{layer}')
                
                feature = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))

                Sort_dict = self.Sort_dict[layer]
                
                # -----
                all_acc = self.single_acc(feature, np.arange(feature.shape[1]), label, tqdm_bar)
                
                # -----
                sensitive_idx = ANOVA_idces[layer]
                non_sensitive_idx = np.array(list(set(np.arange(feature.shape[1])) - set(sensitive_idx)))
                
                sensitive_acc = self.single_acc(feature, sensitive_idx, label, tqdm_bar)
                non_sensitive_acc = self.single_acc(feature, non_sensitive_idx, label, tqdm_bar)
                
                # -----
                si_idx = Sort_dict['basic_type']['si_idx']
                mi_idx = Sort_dict['basic_type']['mi_idx']
                
                si_acc = self.single_acc(feature, si_idx, label, tqdm_bar)
                mi_acc = self.single_acc(feature, mi_idx, label, tqdm_bar)
                
                encode_acc = self.single_acc(feature, [*si_idx, *mi_idx], label, tqdm_bar)
                
                # -----
                s_si_acc = self.single_acc(feature, Sort_dict['advanced_type']['sensitive_si'], label, tqdm_bar)
                s_mi_acc = self.single_acc(feature, Sort_dict['advanced_type']['sensitive_mi'], label, tqdm_bar)
                
                ns_si_acc = self.single_acc(feature, Sort_dict['advanced_type']['non_sensitive_si'], label, tqdm_bar)
                ns_mi_acc = self.single_acc(feature, Sort_dict['advanced_type']['non_sensitive_mi'], label, tqdm_bar)
                
                s_encode_acc = self.single_acc(feature, [*Sort_dict['advanced_type']['sensitive_si'], *Sort_dict['advanced_type']['sensitive_mi']], label, tqdm_bar)
                ns_encode_acc = self.single_acc(feature, [*Sort_dict['advanced_type']['non_sensitive_si'], *Sort_dict['advanced_type']['non_sensitive_mi']], label, tqdm_bar)
                
                # -----
                wsi_idx = Sort_dict['basic_type']['wsi_idx']
                wmi_idx = Sort_dict['basic_type']['wmi_idx']
                
                wsi_acc = self.single_acc(feature, wsi_idx, label, tqdm_bar)
                wmi_acc = self.single_acc(feature, wmi_idx, label, tqdm_bar)
                
                weak_encode_acc = self.single_acc(feature, [*wsi_idx, *wmi_idx], label, tqdm_bar)
                
                # -----
                s_wsi_acc = self.single_acc(feature, Sort_dict['advanced_type']['sensitive_wsi'], label, tqdm_bar)
                s_wmi_acc = self.single_acc(feature, Sort_dict['advanced_type']['sensitive_wmi'], label, tqdm_bar)

                ns_wsi_acc = self.single_acc(feature, Sort_dict['advanced_type']['non_sensitive_wsi'], label, tqdm_bar)
                ns_wmi_acc = self.single_acc(feature, Sort_dict['advanced_type']['non_sensitive_wmi'], label, tqdm_bar)
                
                s_weak_encode_acc = self.single_acc(feature, [*Sort_dict['advanced_type']['sensitive_wsi'], *Sort_dict['advanced_type']['sensitive_wmi']], label, tqdm_bar)
                ns_weak_encode_acc = self.single_acc(feature, [*Sort_dict['advanced_type']['non_sensitive_wsi'], *Sort_dict['advanced_type']['non_sensitive_wmi']], label, tqdm_bar)
                
                # -----
                all_si_acc = self.single_acc(feature, [*si_idx, *wsi_idx], label, tqdm_bar)
                all_mi_acc = self.single_acc(feature, [*mi_idx, *wmi_idx], label, tqdm_bar)
                
                all_s_si_acc = self.single_acc(feature, [*Sort_dict['advanced_type']['sensitive_si'], *Sort_dict['advanced_type']['sensitive_wsi']], label, tqdm_bar)
                all_s_mi_acc = self.single_acc(feature, [*Sort_dict['advanced_type']['sensitive_mi'], *Sort_dict['advanced_type']['sensitive_wmi']], label, tqdm_bar)
                
                all_ns_si_acc = self.single_acc(feature, [*Sort_dict['advanced_type']['non_sensitive_si'], *Sort_dict['advanced_type']['non_sensitive_wsi']], label, tqdm_bar)
                all_ns_mi_acc = self.single_acc(feature, [*Sort_dict['advanced_type']['non_sensitive_mi'], *Sort_dict['advanced_type']['non_sensitive_wmi']], label, tqdm_bar)

                all_encode_acc = self.single_acc(feature, [*si_idx, *wsi_idx, *mi_idx, *wmi_idx], label, tqdm_bar)
                
                # -----
                non_encode_acc = self.single_acc(feature, Sort_dict['basic_type']['non_encode_idx'], label, tqdm_bar)
                s_non_encode_acc = self.single_acc(feature, Sort_dict['advanced_type']['sensitive_non_encode'], label, tqdm_bar)
                ns_non_encode_acc = self.single_acc(feature, Sort_dict['advanced_type']['non_sensitive_non_encode'], label, tqdm_bar)

                # -----
                layer_SVM.update({layer:
                    {
                    'all_acc': all_acc,
                    
                    'sensitive_acc': sensitive_acc,
                    'non_sensitive_acc': non_sensitive_acc,
                    
                    # -----
                    'si_acc': si_acc,
                    'mi_acc': mi_acc,
                    'encode_acc': encode_acc,
                    
                    # -----
                    's_si_acc': s_si_acc,
                    's_mi_acc': s_mi_acc,
                    
                    'ns_si_acc': ns_si_acc,
                    'ns_mi_acc': ns_mi_acc,
                    
                    's_encode_acc': s_encode_acc,
                    'ns_encode_acc': ns_encode_acc,
                    
                    # -----
                    'wsi_acc': wsi_acc,
                    'wmi_acc': wmi_acc,
                    'weak_encode_acc': weak_encode_acc,
                    
                    # -----
                    's_wsi_acc': s_wsi_acc,
                    's_wmi_acc': s_wmi_acc,
                    
                    'ns_wsi_acc': ns_wsi_acc,
                    'ns_wmi_acc': ns_wmi_acc,
                    
                    's_weak_encode_acc': s_weak_encode_acc,
                    'ns_weak_encode_acc': ns_weak_encode_acc,
                    
                    # -----
                    'all_si_acc': all_si_acc,
                    'all_mi_acc': all_mi_acc,
                            
                    'all_s_si_acc': all_s_si_acc,
                    'all_s_mi_acc': all_s_mi_acc,
                    
                    'all_ns_si_acc': all_ns_si_acc,
                    'all_ns_mi_acc': all_ns_mi_acc,
                        
                    'all_encode_acc': all_encode_acc,
                    
                    # -----
                    'non_encode_acc': non_encode_acc,
                    's_non_encode_acc': s_non_encode_acc,
                    'ns_non_encode_acc': ns_non_encode_acc,
                    }})
                    
            utils_.pickle_dump(SVM_path, layer_SVM)
            print('[Condinfo] SVM calculation done')
        
        return layer_SVM
            
    def SVM_plot_single_fig(self, ax, acc_plot_list):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        for acc_plot in acc_plot_list:
            
# =============================================================================
#             'solid' - basic types + all_basic
#             
#             '--' - a_s_e
#             
#             'dotted' - s_s_e
#             
#             'dashdot' - s_ns_e
#             
#             (0,(3,1,1,1,)) dense dash dot - a_ns_e
#             
#             (0,(3,1,1,1,1,1)) dense dash dotdot - a
#             
#             (0,(5,1)) dense dash
#             
#             (0,(3,10,1,10)) sparse dash dot
# =============================================================================
            
            # --- 1 - 3
            if acc_plot == 'all_acc_plot':
                ax.plot(self.acc_plot_dict['all_acc_plot'], 'black', label='all')
            
            if acc_plot == 'sensitive_acc_plot':
                ax.plot(self.acc_plot_dict['sensitive_acc_plot'], 'purple', label='sensitive')
            if acc_plot == 'non_sensitive_acc_plot':
                ax.plot(self.acc_plot_dict['non_sensitive_acc_plot'], 'blue', label='non_sensitive')
            
            # --- 22 - 23
            if acc_plot == 'all_si_acc_plot':
                ax.plot(self.acc_plot_dict['all_si_acc_plot'], 'gold', label='all_si')
            if acc_plot == 'all_mi_acc_plot':
                ax.plot(self.acc_plot_dict['all_mi_acc_plot'], 'lime', label='all_mi')
                
            # --- 28
            if acc_plot == 'all_encode_acc_plot':
                ax.plot(self.acc_plot_dict['all_encode_acc_plot'], 'aqua', label='all_encode')
                
            # --- 24 - 27
            if acc_plot == 'all_s_si_acc_plot':
                ax.plot(self.acc_plot_dict['all_s_si_acc_plot'], 'gold', label='all_s_si', linestyle='--')
            if acc_plot == 'all_s_mi_acc_plot':
                ax.plot(self.acc_plot_dict['all_s_mi_acc_plot'], 'lime', label='all_s_mi', linestyle='--')
                
            if acc_plot == 'all_ns_si_acc_plot':
                ax.plot(self.acc_plot_dict['all_ns_si_acc_plot'], 'gold', label='all_ns_si', linestyle=(0,(3,1,1,1,)))
            if acc_plot == 'all_ns_mi_acc_plot':
                ax.plot(self.acc_plot_dict['all_ns_mi_acc_plot'], 'lime', label='all_ns_mi', linestyle=(0,(3,1,1,1)))

            # --- 4- 6
            if acc_plot == 'si_acc_plot':
                ax.plot(self.acc_plot_dict['si_acc_plot'], 'orange', label='si')
            if acc_plot == 'mi_acc_plot':
                ax.plot(self.acc_plot_dict['mi_acc_plot'], 'green', label='mi')
            if acc_plot == 'encode_acc_plot':
                ax.plot(self.acc_plot_dict['encode_acc_plot'], 'skyblue', label='encode')
                
            # --- 7 - 12
            if acc_plot == 's_si_acc_plot':
                ax.plot(self.acc_plot_dict['s_si_acc_plot'], 'orange', label='s_si', linestyle='--')
            if acc_plot == 's_mi_acc_plot':
                ax.plot(self.acc_plot_dict['s_mi_acc_plot'], 'green', label='s_mi', linestyle='--')

            if acc_plot == 'ns_si_acc_plot':
                ax.plot(self.acc_plot_dict['ns_si_acc_plot'], 'orange', label='ns_si', linestyle=(0,(3,1,1,1,)))
            if acc_plot == 'ns_mi_acc_plot':
                ax.plot(self.acc_plot_dict['ns_mi_acc_plot'], 'green', label='ns_mi', linestyle=(0,(3,1,1,1,)))
                
            if acc_plot == 's_encode_acc_plot':
                ax.plot(self.acc_plot_dict['s_encode_acc_plot'], 'skyblue', label='s_encode', linestyle='--')
            if acc_plot == 'ns_encode_acc_plot':
                ax.plot(self.acc_plot_dict['ns_encode_acc_plot'], 'skyblue', label='ns_encode', linestyle=(0,(3,1,1,1,)))
                
            # --- 13 - 15
            if acc_plot == 'wsi_acc_plot':
                ax.plot(self.acc_plot_dict['wsi_acc_plot'], 'tan', label='wsi')
            if acc_plot == 'wmi_acc_plot':
                ax.plot(self.acc_plot_dict['wmi_acc_plot'], 'darkgreen', label='wmi')
            if acc_plot == 'weak_encode_acc_plot':
                ax.plot(self.acc_plot_dict['weak_encode_acc_plot'], 'cornflowerblue', label='weak_encode')
            
            # --- 16 - 21
            if acc_plot == 's_wsi_acc_plot':
                ax.plot(self.acc_plot_dict['s_wsi_acc_plot'], 'tan', label='s_wsi', linestyle='dotted')
            if acc_plot == 's_wmi_acc_plot':
                ax.plot(self.acc_plot_dict['s_wmi_acc_plot'], 'darkgreen', label='s_wmi', linestyle='dotted')
            
            if acc_plot == 'ns_wsi_acc_plot':
                ax.plot(self.acc_plot_dict['ns_wsi_acc_plot'], 'tan', label='ns_wsi', linestyle='dashdot')
            if acc_plot == 'ns_wmi_acc_plot':
                ax.plot(self.acc_plot_dict['ns_wmi_acc_plot'], 'darkgreen', label='ns_wmi', linestyle='dashdot')
            
            if acc_plot == 's_weak_encode_acc_plot':
                ax.plot(self.acc_plot_dict['s_weak_encode_acc_plot'], 'cornflowerblue', label='s_weak_encode', linestyle='dotted')
            if acc_plot == 'ns_weak_encode_acc_plot':
                ax.plot(self.acc_plot_dict['ns_weak_encode_acc_plot'], 'cornflowerblue', label='ns_weak_encode', linestyle='dashdot')
                
            # --- 29 - 31
            if acc_plot == 'non_encode_acc_plot':
                ax.plot(self.acc_plot_dict['non_encode_acc_plot'], 'red', label='non_encode')
            if acc_plot == 's_non_encode_acc_plot':
                ax.plot(self.acc_plot_dict['s_non_encode_acc_plot'], 'red', label='s_non_encode', linestyle= (0,(3,1,1,1,1,1)))
            if acc_plot == 'ns_non_encode_acc_plot':
                ax.plot(self.acc_plot_dict['ns_non_encode_acc_plot'], 'red', label='ns_non_encode', linestyle=(0,(3,10,1,10)))
            # ---
            
        # -----
        ax.set_ylim([0, 100])
        idx, layers, _ = utils_.imaginary_neurons_vgg(self.layers)
        ax.set_xticks(np.arange(len(self.layers)))
        ax.set_xticklabels(['' if _ not in idx else self.layers[_] for _ in range(len(self.layers))], rotation='vertical', fontname='Times New Roman')
        ax.set_yticks(np.arange(1, 101))
        ax.set_yticklabels(['' if (_%10)!=0 else _ for _ in range(1,101)], rotation='vertical', fontname='Times New Roman')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(ncol=3, loc="upper left", framealpha=0.5)
        
    def SVM_plot(self, ):
        
        print('[Codinfo] Executing SVM plot...')
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        if not hasattr(self, 'Sort_dict'):
            self.Sort_dict = utils_.pickle_load(os.path.join(self.dest_Encode, 'Sort_dict.pkl'))
        
        SVM_fig_folder = os.path.join(self.dest_Encode, 'SVM_Figures')
        utils_.make_dir(SVM_fig_folder)
        
        layer_SVM = self.SVM()
      
        self.acc_plot_dict = {
            
            # --- 1 - 3
            'all_acc_plot':[layer_SVM[layer]['all_acc'] for layer in self.layers],
            
            'sensitive_acc_plot':[layer_SVM[layer]['sensitive_acc'] for layer in self.layers],
            'non_sensitive_acc_plot':[layer_SVM[layer]['non_sensitive_acc'] for layer in self.layers],
            
            # --- 4 - 6
            'si_acc_plot':[layer_SVM[layer]['si_acc'] for layer in self.layers],
            'mi_acc_plot':[layer_SVM[layer]['mi_acc'] for layer in self.layers],
  
            'encode_acc_plot':[layer_SVM[layer]['encode_acc'] for layer in self.layers],
            
            # --- 7 - 12
            's_si_acc_plot':[layer_SVM[layer]['s_si_acc'] for layer in self.layers],
            's_mi_acc_plot':[layer_SVM[layer]['s_mi_acc'] for layer in self.layers],
            
            'ns_si_acc_plot':[layer_SVM[layer]['ns_si_acc'] for layer in self.layers],
            'ns_mi_acc_plot':[layer_SVM[layer]['ns_mi_acc'] for layer in self.layers],
            
            's_encode_acc_plot':[layer_SVM[layer]['s_encode_acc'] for layer in self.layers],
            'ns_encode_acc_plot':[layer_SVM[layer]['ns_encode_acc'] for layer in self.layers],
            
            # --- 13 - 15
            'wsi_acc_plot': [layer_SVM[layer]['wsi_acc'] for layer in self.layers],
            'wmi_acc_plot': [layer_SVM[layer]['wmi_acc'] for layer in self.layers],
            
            'weak_encode_acc_plot': [layer_SVM[layer]['weak_encode_acc'] for layer in self.layers],
            
            # --- 16 - 21
            's_wsi_acc_plot': [layer_SVM[layer]['s_wsi_acc'] for layer in self.layers],
            's_wmi_acc_plot': [layer_SVM[layer]['s_wmi_acc'] for layer in self.layers],
                            
            'ns_wsi_acc_plot': [layer_SVM[layer]['ns_wsi_acc'] for layer in self.layers],
            'ns_wmi_acc_plot': [layer_SVM[layer]['ns_wmi_acc'] for layer in self.layers],
            
            's_weak_encode_acc_plot': [layer_SVM[layer]['s_weak_encode_acc'] for layer in self.layers],
            'ns_weak_encode_acc_plot': [layer_SVM[layer]['ns_weak_encode_acc'] for layer in self.layers],
            
            # --- 22 - 28
            'all_si_acc_plot': [layer_SVM[layer]['all_si_acc'] for layer in self.layers],
            'all_mi_acc_plot': [layer_SVM[layer]['all_mi_acc'] for layer in self.layers],
            
            'all_s_si_acc_plot': [layer_SVM[layer]['all_s_si_acc'] for layer in self.layers],
            'all_s_mi_acc_plot': [layer_SVM[layer]['all_s_mi_acc'] for layer in self.layers],
            
            'all_ns_si_acc_plot': [layer_SVM[layer]['all_ns_si_acc'] for layer in self.layers],
            'all_ns_mi_acc_plot': [layer_SVM[layer]['all_ns_mi_acc'] for layer in self.layers],
            
            'all_encode_acc_plot': [layer_SVM[layer]['all_encode_acc'] for layer in self.layers],
            
            # --- 29 - 31
            'non_encode_acc_plot':[layer_SVM[layer]['non_encode_acc'] for layer in self.layers],
            's_non_encode_acc_plot':[layer_SVM[layer]['s_non_encode_acc'] for layer in self.layers],
            'ns_non_encode_acc_plot':[layer_SVM[layer]['ns_non_encode_acc'] for layer in self.layers],
            }
        
        acc_plot_list = list(self.acc_plot_dict.keys())
        
        # ----- All types
        fig, ax = plt.subplots(figsize=(14,10))
        self.SVM_plot_single_fig(ax, acc_plot_list)
        ax.set_title(title:=f'All types [{self.model_structure}]')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.eps'), bbox_inches='tight', format='eps')
        
        # ----- Basic types
        fig, ax = plt.subplots(figsize=(14,10))
        self.SVM_plot_single_fig(ax, [acc_plot_list[_] for _ in [0, 1,2, 3,4,5, 12,13,14, 21,22, 27,28]])
        ax.set_title(title:=f'Basic types [{self.model_structure}]')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.eps'), bbox_inches='tight', format='eps')
        
        # ----- S vs NS and E vs NE
        fig, ax = plt.subplots(figsize=(14,10))
        self.SVM_plot_single_fig(ax, [acc_plot_list[_] for _ in [0, 1, 6,7, 10, 15,16, 19, 23,24, 29]])
        ax.set_title(title:=f'Sensitive types [{self.model_structure}]')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.eps'), bbox_inches='tight', format='eps')
        
        fig, ax = plt.subplots(figsize=(14,10))
        self.SVM_plot_single_fig(ax, [acc_plot_list[_] for _ in [0, 2, 8,9, 11, 17,18, 20, 25,26, 30]])
        ax.set_title(title:=f'Non Sensitive types [{self.model_structure}]')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.eps'), bbox_inches='tight', format='eps')
        
        fig, ax = plt.subplots(figsize=(14,10))
        self.SVM_plot_single_fig(ax, [acc_plot_list[_] for _ in [0, 3,4,5, 6,7,8,9,10,11, 12,13,14, 15,16,17,18,19,20, 21,22,23,24,25,26,27]])
        ax.set_title(title:=f'Encode types [{self.model_structure}]')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.eps'), bbox_inches='tight', format='eps')
        
        fig, ax = plt.subplots(figsize=(14,10))
        self.SVM_plot_single_fig(ax, [acc_plot_list[_] for _ in [0,28,29,30]])
        ax.set_title(title:=f'Non Encode types [{self.model_structure}]')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.eps'), bbox_inches='tight', format='eps')
        
        # ------ Strong, Weak and All
        fig, ax = plt.subplots(figsize=(14,10))
        self.SVM_plot_single_fig(ax, [acc_plot_list[_] for _ in [0, 3,4,5, 6,7,8,9,10,11]])
        ax.set_title(title:=f'Strong all types [{self.model_structure}]')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.eps'), bbox_inches='tight', format='eps')
        
        fig, ax = plt.subplots(figsize=(14,10))
        self.SVM_plot_single_fig(ax, [acc_plot_list[_] for _ in [0, 12,13,14, 15,16,17,18,19,20]])
        ax.set_title(title:=f'Weak all types [{self.model_structure}]')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.eps'), bbox_inches='tight', format='eps')
        
        fig, ax = plt.subplots(figsize=(14,10))
        self.SVM_plot_single_fig(ax, [acc_plot_list[_] for _ in [0, 21,22,23,24,25,26,27]])
        ax.set_title(title:=f'All(Strong+Weak) all types [{self.model_structure}]')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.png'), bbox_inches='tight')
        fig.savefig(os.path.join(SVM_fig_folder, f'{title}.eps'), bbox_inches='tight', format='eps')
        
        plt.close('all')
        
        print('[Codinfo] Image saved')
    
    def layer_response_assemble(self,):
        
        print('[Codinfo] Executing layer_response_assemble...')
        
        if not hasattr(self, 'Sort_dict'):
            self.Sort_dict = utils_.pickle_load(os.path.join(self.dest_Encode, 'Sort_dict.pkl'))
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 14})
        
        fig_folder = os.path.join(self.dest_Encode, 'Layer_units_stats')
        utils_.make_dir(fig_folder)
        
        colorpool_jet = plt.get_cmap('jet', 50)
        colors = [colorpool_jet(i) for i in range(50)]
        
        layers = self.layers[:5]
        
        #with Parallel(n_jobs=1) as parallel:     
        #    parallel(delayed(self.layer_response_assemble_sinlge_layer)(fig_folder, layer, colors) for layer in layers)  
            
        for layer in layers:
            self.layer_response_assemble_sinlge_layer(fig_folder, layer, colors)
            
            gc.collect()
    
    def layer_response_assemble_sinlge_layer(self, fig_folder, layer, colors):
        
        Sort_dict = self.Sort_dict[layer]
        feature = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))
        
        y_lim_min = np.min(feature)
        y_lim_max = np.max(feature)
        
        idx_dict = {
            's_si': Sort_dict['advanced_type']['sensitive_si'],
            's_wsi': Sort_dict['advanced_type']['sensitive_wsi'],
            
            's_mi': Sort_dict['advanced_type']['sensitive_mi'],
            's_wmi': Sort_dict['advanced_type']['sensitive_wmi'],
            
            's_non_encode': Sort_dict['advanced_type']['sensitive_non_encode'],
            
            'ns_si': Sort_dict['advanced_type']['non_sensitive_si'],
            'ns_wsi': Sort_dict['advanced_type']['non_sensitive_wsi'],
            
            'ns_mi': Sort_dict['advanced_type']['non_sensitive_mi'],
            'ns_wmi': Sort_dict['advanced_type']['non_sensitive_wmi'],
            
            'ns_non_encode': Sort_dict['advanced_type']['non_sensitive_non_encode']
            }
        
        fig, ax = plt.subplots(figsize=(26,10))
        gs_main = gridspec.GridSpec(2, 5, figure=fig)
        
        tqdm_bar = tqdm(total=10, desc=f'{layer}')
        
        i_ = 0
        for i in range(2):
            for j in range(5):
                # Define a sub-grid within the current cell of the main grid
                gs_sub = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[4, 1], subplot_spec=gs_main[i, j])

                ax_left = fig.add_subplot(gs_sub[0])
                ax_right = fig.add_subplot(gs_sub[1])
                
                if i_ != 0:
                    ax_left.set_xticks([])
                    ax_left.set_yticks([])

                ax_right.set_xticks([])
                ax_right.set_yticks([])
                
                if idx_dict[list(idx_dict.keys())[i_]].size == 0:
                    pct = len(idx_dict[list(idx_dict.keys())[i_]])/feature.shape[1]*100
                    ax_left.set_title(list(idx_dict.keys())[i_] + f' [{pct:.2f}%]')
                    ax_right.set_title('th')
                    i_ += 1
                else:
                    feature_test = feature[:,idx_dict[list(idx_dict.keys())[i_]]]     # (500, num_units)
                    feature_test_mean = feature_test.reshape(self.num_classes, self.num_samples, -1)     # (50, 10, num_units)
                    
                    # -----
                    x = np.array([[[_] for _ in range(self.num_classes)]*feature_test_mean.shape[2]]).reshape(-1)
                    y = np.mean(feature_test_mean, axis=1).T.reshape(-1)
                    
                    c = np.array(colors)
                    c = np.tile(c, [feature_test_mean.shape[2], 1])
                    #c = np.repeat(c, 10, axis=0)     # <- for each img
                    
                    # -----
                    ax_left.scatter(x, y, color=c, alpha=0.1, marker='.', s=1)     # use small size to replace adjustable alpha
                    # -----
                    
                    pct = len(idx_dict[list(idx_dict.keys())[i_]])/feature.shape[1]*100
                    ax_left.set_title(list(idx_dict.keys())[i_] + f' [{pct:.2f}%]')
                    # -----
                    
                    feature_test_mean = np.mean(feature_test_mean, axis=1)     # (50, num_units)
                    # ----- stats: mean for each ID
                    values = feature_test_mean.reshape(-1)    # (50*num_units)
                    if np.std(values) == 0:
                        pass
                    else:
                        kde_mean = gaussian_kde(values)
                        x_vals_mean = np.linspace(np.min(values), np.max(values)*1.1, 1000)
                        y_vals_mean = kde_mean(x_vals_mean)
                        ax_right.plot(y_vals_mean, x_vals_mean, color='blue')
                        
                    # ----- stats: threshold (mean+2std of all 500 values)
                    values = np.mean(feature_test, axis=0) + 2*np.std(feature_test, axis=0)     # (units,)
                    if np.std(values) == 0:
                        pass
                    else:
                        kde = gaussian_kde(values)
                        x_vals = np.linspace(np.min(values), np.max(values)*1.1, 1000)
                        y_vals = kde(x_vals)
                        ax_right.plot(y_vals, x_vals, color='red')
                    
                        y_vals_max = np.max(y_vals)
                        x_vals_max = x_vals[np.where(y_vals==y_vals_max)[0].item()]
                        
                        ax_left.hlines(x_vals_max, 0, 50, colors='red', alpha=0.75, linestyle='--')
                        ax_right.hlines(x_vals_max, np.min(y_vals), np.max(y_vals), colors='red', alpha=0.75, linestyle='--')
                    
                    # ----- stats: ref (mean+2std of all 50 mean values)
                    values = np.mean(feature_test_mean, axis=0) + 2*np.std(feature_test_mean, axis=0)     # (units,)
                    
                    if np.std(values) == 0:
                        pass
                    else:
                        kde = gaussian_kde(values)
                        x_vals = np.linspace(np.min(values), np.max(values)*1.1, 1000)
                        y_vals = kde(x_vals)
                        ax_right.plot(y_vals, x_vals, color='teal')
                    
                        y_vals_max = np.max(y_vals)
                        x_vals_max = x_vals[np.where(y_vals==y_vals_max)[0].item()]
                        
                        ax_left.hlines(x_vals_max, 0, 50, colors='teal', alpha=0.75, linestyle='--')
                        ax_right.hlines(x_vals_max, np.min(y_vals), np.max(y_vals), colors='teal', alpha=0.75, linestyle='--')
                    
                    scaling_factor = 0.6
                    
                    ax_left.set_ylim([y_lim_min, y_lim_max*scaling_factor])
                    ax_right.set_ylim([y_lim_min, y_lim_max*scaling_factor])
                    ax_right.set_title('th')
                    
                    i_ += 1
                tqdm_bar.update(1)
               
        ax.axis('off')
        ax.plot([],[],color='blue',label='mean')
        ax.plot([],[],color='teal',label='ref')
        ax.plot([],[],color='red',label='threshold')
        
        fig.suptitle(f'{layer} [{self.model_structure}]', y=0.97, fontsize=20)
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, bbox_transform=plt.gcf().transFigure)
        
        plt.tight_layout()
        fig.savefig(os.path.join(fig_folder, f'{layer}.png'), bbox_inches='tight')
        #fig.savefig(os.path.join(fig_folder, f'{layer}.eps'), bbox_inches='tight', format='eps')
        #fig.savefig(os.path.join(fig_folder, f'{layer}.svg'), bbox_inches='tight', format='svg', transparent=True)
        plt.close()
    
    def layer_response_single_boxplot(self, random_select_units=10):
        
        """
            this function provides boxplot of example units of different types
        """
        print('[Codinfo] Executing layer_response_single_boxplot')
        
        if not hasattr(self, 'Sort_dict'):
            self.Sort_dict = utils_.pickle_load(os.path.join(self.dest_Encode, 'Sort_dict.pkl'))
            
        fig_folder = os.path.join(self.dest_Encode, 'Layer_units_samples')
        utils_.make_dir(fig_folder)
        
        colorpool_jet = plt.get_cmap('jet', 50)
        colors = [colorpool_jet(i) for i in range(50)]
        
        layers = self.layers[:]
        
        for layer in layers:
            
            layer_fig_folder = os.path.join(fig_folder, f'{layer}')
            utils_.make_dir(layer_fig_folder)
            
            Sort_dict = self.Sort_dict[layer]
            feature = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))
            
            y_lim_min = np.min(feature)
            y_lim_max = np.max(feature)
            
            idx_dict = {
                's_si': Sort_dict['advanced_type']['sensitive_si'],
                's_wsi': Sort_dict['advanced_type']['sensitive_wsi'],
                
                's_mi': Sort_dict['advanced_type']['sensitive_mi'],
                's_wmi': Sort_dict['advanced_type']['sensitive_wmi'],
                
                's_non_encode': Sort_dict['advanced_type']['sensitive_non_encode'],
                
                'ns_si': Sort_dict['advanced_type']['non_sensitive_si'],
                'ns_wsi': Sort_dict['advanced_type']['non_sensitive_wsi'],
                
                'ns_mi': Sort_dict['advanced_type']['non_sensitive_mi'],
                'ns_wmi': Sort_dict['advanced_type']['non_sensitive_wmi'],
                
                'ns_non_encode': Sort_dict['advanced_type']['non_sensitive_non_encode']
                }
            
            for key in tqdm(list(idx_dict.keys()), desc=f'{layer}'):     # for each type
                test_type = idx_dict[key]
                
                type_layer_fig_folder = os.path.join(layer_fig_folder, f'{key}')
                utils_.make_dir(type_layer_fig_folder)
                
                if test_type.size == 0:
                    pass
                
                else:
                    if test_type.size > random_select_units:
                        test_idces = random.sample(list(test_type), random_select_units)
                    else:
                        test_idces = test_type
                       
                    with Parallel(n_jobs=-1) as parallel:     # for 10 units of one type
                        parallel(delayed(self.layer_response_single_boxplot_single)(type_layer_fig_folder, key, layer, feature, idx, colors, y_lim_min, y_lim_max) for idx in test_idces)  
    
                    gc.collect()
                    
    def layer_response_single_boxplot_single(self, type_layer_fig_folder, key, layer, feature, idx, colors, y_lim_min, y_lim_max):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 14})
        
        fig, ax = plt.subplots(1, 2, figsize=(20,10))
    
        x = np.array([[_]*10 for _ in range(1,51)])
        y = feature[:, idx]
        c = np.repeat(np.array(colors), 10, axis=0)
        
        test_feature = [y.reshape(self.num_classes, self.num_samples)[_] for _ in range(self.num_classes)]
        test_feature_mean = np.array([np.mean(test_feature[_]) for _ in range(self.num_classes)])
        
        ax[0].scatter(x, y, color=c, s=10)
        ax[0].scatter(np.arange(1,51), test_feature_mean, color=colors, marker='d')
        for _ in range(self.num_classes):
            ax[0].vlines(_+1, np.min(y), test_feature_mean[_], linestyle='--')
        ax[0].hlines(np.mean(y)+2*np.std(y), 1, 50, colors='red', linestyle='--', label=r'$V_{th}=\bar{x}+2\sqrt{\frac{1}{500}\sum(x_i-\bar{x})^2}$')
        ax[0].hlines(np.mean(test_feature_mean)+2*np.std(test_feature_mean), 1, 50, colors='teal', linestyle='--', label=r'$ref=\bar{x}+2\sqrt{\frac{1}{50}\sum(x_i-\bar{x_i})^2}, \bar{x_i} = \frac{1}{10}\sum{x_i}$')
        ax[0].set_title('scatters')
        ax[0].legend(framealpha=0.75)

        ax[0].set_ylim([y_lim_min, y_lim_max])
        
        boxes = ax[1].boxplot(test_feature, patch_artist=True, sym='+')
        ax[1].hlines(np.mean(np.array(test_feature))+2*np.std(np.array(test_feature)), 1, 50, colors='red', linestyle='--')
        ax[1].hlines(np.mean(test_feature_mean)+2*np.std(test_feature_mean), 1, 50, colors='teal', linestyle='--')
        
        ax[1].scatter(np.arange(1,51), test_feature_mean, color=colors, marker='d')
        for _ in range(self.num_classes):
            ax[1].vlines(_+1, np.min(y), test_feature_mean[_], linestyle='--')
        
        for i, _ in enumerate(boxes['boxes']):
            _.set(color=colors[i], alpha=0.5)
        for i, _ in enumerate(boxes['fliers']):
            _.set(marker='+', markerfacecolor=colors[i], markeredgecolor=colors[i], markersize=10, alpha=0.75)
        ax[1].set_title('boxplots')
        
        ax[1].set_ylim([y_lim_min, y_lim_max])
        
        fig.suptitle(f'[{layer}] unit: {idx}', y=0.98)
        plt.tight_layout()
        
        fig.savefig(os.path.join(type_layer_fig_folder, f'{key}_{idx}.png'), bbox_inches='tight')
        #fig.savefig(os.path.join(fig_folder, f'{layer}.eps'), bbox_inches='tight', format='eps')
        #fig.savefig(os.path.join(fig_folder, f'{layer}.svg'), bbox_inches='tight', format='svg', transparent=True)
        plt.close()
    
# define Encode for parallel calculation
def encode_calculation(feature, i):
    """
        parallel computation to obtain encode_dict
    """
    
    feature_of_single_unit = feature[:, i]
    grouped_feature_of_single_unit = feature_of_single_unit.reshape(-1,10)
    
    threshold = np.mean(feature_of_single_unit) + 2*np.std(feature_of_single_unit)     # [notice] np.std([500 firing_rates]) is np.sqrt(10) times larger than np.std([50 mean_firing_rates])
    
    local_mean = np.mean(grouped_feature_of_single_unit, axis=1)     # array of 50 values
    ref = np.mean(feature_of_single_unit) + 2*np.std(local_mean)
    
    encode_class = np.where(local_mean>threshold)[0]+1  # '>' prevent all 0
    
    weak_encode_class = np.where(local_mean>ref)[0]+1
    weak_encode_class = np.setdiff1d(weak_encode_class, encode_class)
    
    encode_id = {
        'encode': encode_class,
        'weak_encode': weak_encode_class,
                }
    
    return encode_id

if __name__ == "__main__":
    
    model_name = 'vgg16_bn'
    
    model_ = vgg.__dict__[model_name](num_classes=50)
    layers, neurons, shapes = utils_.generate_vgg_layers_list_ann(model_, model_name)

    root_dir = '/home/acxyle-workstation/Downloads'

    selectivity_analyzer = Encode_feaquency_analyzer(root=os.path.join(root_dir, 'Face Identity VGG16bn/'), 
                                                     layers=layers, neurons=neurons)
    
    #selectivity_analyzer.obtain_encode_class_dict()
    #selectivity_analyzer.selectivity_encode_layer_percent_plot()
    
    #selectivity_analyzer.draw_encode_frequency()
    
    # ---
    #selectivity_analyzer.generate_encoded_id_unit_idx()     # <- currently not in use 
    # ---
    
    selectivity_analyzer.SVM()
    #selectivity_analyzer.SVM_plot()
    
    #selectivity_analyzer.layer_response_assemble()
    #selectivity_analyzer.layer_response_single_boxplot()
