#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:54:55 2023

@author: acxyle

    complete 5 sections in one script:
        1. obtrain_encode_class_dict() - save dict.pkl
        2. plot_Encode_freq()
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
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import logging
import warnings
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from collections import Counter

from scipy.stats import gaussian_kde
from matplotlib import gridspec


from Bio_Cell_Records_Process import Human_Neuron_Records_Process
import utils_
import models_


class Encode_feaquency_analyzer():
    """
        The basic design, the feature map and encode_dict follows the lexical order. While the figure and Encode_id_unit_dict.pkl
        follows natural order
    
        in the update on Sept 6, 2023, remove original 2 input files setting
    """
    def __init__(self, root,
                 num_classes=50, num_samples=10, layers=None, neurons=None):
        
        assert root[-1] != '/', f"[Codinfo] root {root} should not end with '/'"
        
        self.root = os.path.join(root, 'Features')     # <- folder for feature maps, which should be generated before analysis
        self.dest = os.path.join(root, 'Analysis')
        utils_.make_dir(self.dest)
        
        self.dest_Encode = os.path.join(self.dest, 'Encode')
        utils_.make_dir(self.dest_Encode)
        
        self.layers = layers
        self.neurons = neurons
        
        self.num_classes = num_classes
        self.num_samples = num_samples
        
        self.feature_list = [os.path.join(self.root, _) for _ in sorted(os.listdir(self.root)) if 'pkl' in _]     # feature .pkl list
        
        self.ANOVA_idces = utils_.load(os.path.join(self.dest, 'ANOVA/ANOVA_idces.pkl'))   # <- consider to remove this?
        
        if layers == None or neurons == None:
            raise ValueError('[Codwarning] invalid layers and neurons')
            
        self.model_structure = root.split('/')[-1].split(' ')[-1]
        
        
    def calculation_Encode(self, ):
        """
            this function generates the encode_dict, 1-based dict containing encoded classes, and sort_dict, different types of units
        """
        print('[Codinfo] Executing calculation_Encode...')
        
        encode_dict_path = os.path.join(self.dest_Encode, 'Encode_dict.pkl')
        sort_dict_path = os.path.join(self.dest_Encode, 'Sort_dict.pkl')
        
        if (not hasattr(self, 'Sort_dict') and os.path.exists(encode_dict_path)):
            
            #self.Encode_dict = utils_.load(encode_dict_path)     # memory consuming
            self.Sort_dict = utils_.load(sort_dict_path)
        
        else:
            
            # ----- init
            self.Encode_dict = {}
            self.Sort_dict = {}
            
            # --- layer check
            feature_list_check = [_.split('/')[-1].split('.')[0] for _ in self.feature_list]
            layers_check = self.layers.copy()
            if not sorted(layers_check) == sorted(feature_list_check):
                raise RuntimeError('[Coderror] detected the features and layers do not match')
                
            num_workers = os.cpu_count()
            print(f'[Codinfo] Executing parallel computation with num_workers={num_workers}')
                
            # --- running
            for layer in self.layers:     # for each layer
                
                feature = utils_.load(os.path.join(self.root, f'{layer}.pkl'))      # load feature matrix
    
                s = self.ANOVA_idces[layer]     # sensitive_idx
                ns = np.array(list(set(np.arange(feature.shape[1]))-set(s)))     # non_sensitive_idx
                
                unit_encode_dict = {}
    
                # [notice] the encode ID is 1-based
                pl = Parallel(n_jobs=num_workers)(delayed(encode_calculation)(feature, i) for i in tqdm(range(feature.shape[1]), desc=f'[{layer}] Encode'))  
    
                unit_encode_dict = {i: pl[i] for i in range(len(pl))}    
                
                self.Encode_dict[layer] = unit_encode_dict    
                
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
                unit_sort_dict = {
                    
                    'basic_type': {
                        'si': si,
                        'wsi': wsi,
                        'mi': mi,
                        'wmi': wmi,
                        'non_encode': non_encode
                        },
                    
                    'advanced_type': {
                        's_si': np.intersect1d(s, si),
                        's_wsi': np.intersect1d(s, wsi),
                        's_mi': np.intersect1d(s, mi),
                        's_wmi': np.intersect1d(s, wmi),
                        's_non_encode': np.intersect1d(s, non_encode),
                        
                        'ns_si': np.intersect1d(ns, si),
                        'ns_wsi': np.intersect1d(ns, wsi),
                        'ns_mi': np.intersect1d(ns, mi),
                        'ns_wmi': np.intersect1d(ns, wmi),
                        'ns_non_encode': np.intersect1d(ns, non_encode)
                        }
                    }
                
                self.Sort_dict[layer] = unit_sort_dict
                
            utils_.dump(self.Sort_dict, sort_dict_path, verbose=True)
            utils_.dump(self.Encode_dict, encode_dict_path, verbose=True)  
            
            # ---
            if hasattr(self, 'Encode_dict'):
                delattr(self, 'Encode_dict')     # release the memory
            
        
    @staticmethod
    def obtain_Encode_types_pct(layers, neurons, Sort_dict, num_types=5):
        
        if num_types == 5:
            
            Encode_types_pct = {}
            
            for layer in Sort_dict.keys():
                
                target_groups = ['s_si', 's_wsi', 's_mi', 's_wmi']
                
                tmp = {_: Sort_dict[layer]['advanced_type'][_] for _ in target_groups}
                
                all_non = [Sort_dict[layer]['advanced_type'][_] for _ in Sort_dict[layer]['advanced_type'] if _ not in target_groups]
                all_non = [__ for _ in all_non for __ in _]
                
                tmp['n_e'] = np.array(all_non)
                
                Sort_dict[layer] = tmp
            
            # ---
            for type_ in target_groups+['n_e']:
                
                Encode_types_pct[type_] = np.array([len(Sort_dict[_][type_])/neurons[idx]*100 for idx, _ in enumerate(layers)])
            
            
        elif num_types == 23:     # 4 subplots
            
            Encode_types_pct = {}
        
            # -----
            for type_ in ['non_encode', 'si', 'mi', 'wsi', 'wmi']:
                
                Encode_types_pct[type_] = np.array([len(Sort_dict[_]['basic_type'][type_])/neurons[idx]*100 for idx, _ in enumerate(layers)])
            
            # -----
            for type_ in ['s_si', 's_mi', 's_wsi', 's_wmi', 's_non_encode', 'ns_si', 'ns_mi', 'ns_wsi', 'ns_wmi', 'ns_non_encode']:
                
                Encode_types_pct[type_] = np.array([len(Sort_dict[_]['advanced_type'][type_])/neurons[idx]*100 for idx, _ in enumerate(layers)])
            
            # ---
            Encode_types_pct['encode'] = np.array([Encode_types_pct['si'][_]+Encode_types_pct['mi'][_] for _ in range(len(layers))])
            Encode_types_pct['weak_encode'] = np.array([Encode_types_pct['wsi'][_]+Encode_types_pct['wmi'][_] for _ in range(len(layers))])
            
            Encode_types_pct['s_encode'] = np.array([Encode_types_pct['s_si'][_]+Encode_types_pct['s_mi'][_] for _ in range(len(layers))])
            Encode_types_pct['s_weak_encode'] = np.array([Encode_types_pct['s_wsi'][_]+Encode_types_pct['s_wmi'][_] for _ in range(len(layers))])
            
            Encode_types_pct['ns_encode'] = np.array([Encode_types_pct['ns_si'][_]+Encode_types_pct['ns_mi'][_] for _ in range(len(layers))])
            Encode_types_pct['ns_weak_encode'] = np.array([Encode_types_pct['ns_wsi'][_]+Encode_types_pct['ns_wmi'][_] for _ in range(len(layers))])
            
            Encode_types_pct['s'] = np.array([Encode_types_pct['s_encode'][_]+Encode_types_pct['s_weak_encode'][_]+Encode_types_pct['s_non_encode'][_] for _ in range(len(layers))])
            Encode_types_pct['ns'] = np.array([Encode_types_pct['ns_encode'][_]+Encode_types_pct['ns_weak_encode'][_]+Encode_types_pct['ns_non_encode'][_] for _ in range(len(layers))])
            
        
        return Encode_types_pct
    
    
    @staticmethod
    def obtain_Encode_types_curve_dict(Encode_types_pct):
        """
            this function return the cruve config for each key of Encode_types_dict. Filter the keys of the input if need to select special types
        """
        
        curve_dict = {}
        
        for key in Encode_types_pct.keys():
              
            # ----- basic type
            if key == 'non_encode':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='non_encode', color='#808080', linestyle='--', linewidth=3.0)  # Grey
                            
            elif key == 'si':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='si', color='#0000FF', linestyle='--', linewidth=3.0)  # Blue
                            
            elif key == 'mi':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='mi', color='#FFA500', linestyle='--', linewidth=3.0)  # Orange
                            
            elif key == 'encode':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='encode (si+mi)', color='#800080', linewidth=3.0)  # Purple
                            
            elif key == 'wsi':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='wsi', color='#ADD8E6', linestyle='dotted', linewidth=3.0)  # Light Blue
                            
            elif key == 'wmi':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='wmi', color='#FFD580', linestyle='dotted', linewidth=3.0)  # Light Orange
                            
            elif key == 'weak_encode':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='weak_encode (wsi+wmi)', color='#FFC0CB', linestyle='dotted', linewidth=3.0)  # Pink
                            
            # ---                                                  
            elif key == 's':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='s', color='#FF0000', linewidth=3.0)  # Red
                                                                              
            elif key == 'ns': 
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='ns', color='#008000', linewidth=3.0)  # Green
            
            # ----- advanced type
            elif key == 'n_e':  
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='n_e', color='#555555', linewidth=1.0)  # Dark Gray
            
            # ---
            # For combinations, using a blend of the base colors and adjusting linewidth and linestyle to match the base type
            elif key == 's_si':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='s_si', color='#800080', linestyle='--', linewidth=1.0)  # Deep Purple
                        
            elif key == 's_mi':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='s_mi', color='#FF4500', linestyle='--', linewidth=1.0)  # Orange Red
                        
            elif key == 's_encode':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='sensitive_e(si+mi)', color='#C71585', linewidth=1.0)  # Medium Violet Red
                        
            elif key == 's_wsi':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='s_wsi', color='#FF69B4', linestyle='dotted', linewidth=1.0)  # Hot Pink
            
            elif key == 's_wmi':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='s_wmi', color='#FFA07A', linestyle='dotted', linewidth=1.0)  # Light Salmon
              
            elif key == 's_weak_encode':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='sensitive_we(wsi+wmi)', color='#FFB6C1', linestyle='dotted', linewidth=1.0)  # Light Pink
            
            elif key == 's_non_encode':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='s_non_encode', color='#B22222', linestyle='--', linewidth=1.0)  # Firebrick
                            
            # ---
            elif key == 'ns_si':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='ns_si', color='#004080', linestyle='--', linewidth=1.0)  # Dark Blue
                                                                              
            elif key == 'ns_mi':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='ns_mi', color='#808000', linestyle='--', linewidth=1.0)  # Olive
                            
            elif key == 'ns_encode':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='ns_encode', color='#4B0082', linestyle='--', linewidth=1.0)  # Indigo
                                                                
            elif key == 'ns_wsi':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='ns_wsi', color='#80C0E6', linestyle='dotted', linewidth=1.0)  # Sky Blue
                                                                              
            elif key == 'ns_wmi':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='ns_wmi', color='#B0C080', linestyle='dotted', linewidth=1.0)  # Sage
                                                                              
            elif key == 'ns_weak_encode':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='ns_weak_encode', color='#BA55D3', linestyle='dotted', linewidth=1.0)  # Medium Orchid
                                                                                                                                  
            elif key == 'ns_non_encode':
                curve_dict[key] = encode_layer_percent_plot_dict(Encode_types_pct[key], label='ns_non_encode', color='#556B2F', linestyle='--', linewidth=1.0)  # Dark Olive Green
                
 
        return curve_dict
    
    
    # FIXME ----- test version
    def plot_Encode_pct(self, num_types=5):
        """
            this function plot the percentages of different types of units over layers
            
            3 types of basic units: ['si', 'mi', 'non_encode']
        """
        
        plt.rcParams.update({'font.size': 18})     # control the all font size
        plt.rcParams.update({"font.family": "Times New Roman"})

        # -----
        fig_folder = os.path.join(self.dest_Encode, 'Figures')
        utils_.make_dir(fig_folder)
        
        if not hasattr(self, 'Sort_dict'):  
            sort_dict_path = os.path.join(self.dest_Encode, 'Sort_dict.pkl')
            self.Sort_dict = utils_.load(sort_dict_path)
    
        print(f'[Codinfo] Executing plot_Encode_pct() with num_types={num_types}')
        
        # --- label correction
        if 'basic_type' in self.Sort_dict['neuron_1'].keys() and 'si_idx' in self.Sort_dict['neuron_1']['basic_type'].keys():
            
            for layer in self.Sort_dict.keys():
                
                unit_sort_dict = {
                    
                    'basic_type': {
                        'si': self.Sort_dict[layer]['basic_type']['si_idx'],
                        'wsi': self.Sort_dict[layer]['basic_type']['wsi_idx'],
                        'mi': self.Sort_dict[layer]['basic_type']['mi_idx'],
                        'wmi': self.Sort_dict[layer]['basic_type']['wmi_idx'],
                        'non_encode': self.Sort_dict[layer]['basic_type']['non_encode_idx']
                        },
                    
                    'advanced_type': {
                        's_si': self.Sort_dict[layer]['advanced_type']['sensitive_si'],
                        's_wsi': self.Sort_dict[layer]['advanced_type']['sensitive_wsi'],
                        's_mi': self.Sort_dict[layer]['advanced_type']['sensitive_mi'],
                        's_wmi': self.Sort_dict[layer]['advanced_type']['sensitive_wmi'],
                        's_non_encode': self.Sort_dict[layer]['advanced_type']['sensitive_non_encode'],
                        
                        'ns_si': self.Sort_dict[layer]['advanced_type']['non_sensitive_si'],
                        'ns_wsi': self.Sort_dict[layer]['advanced_type']['non_sensitive_wsi'],
                        'ns_mi': self.Sort_dict[layer]['advanced_type']['non_sensitive_mi'],
                        'ns_wmi': self.Sort_dict[layer]['advanced_type']['non_sensitive_wmi'],
                        'ns_non_encode': self.Sort_dict[layer]['advanced_type']['non_sensitive_non_encode'],
                        }
                    }
                
                self.Sort_dict[layer] = unit_sort_dict
        
            os.remove(sort_dict_path)
            utils_.dump(self.Sort_dict, sort_dict_path)
            print('------------------------------------------')
            print('[Codinfo] self.Sort_dict updated')
            print('------------------------------------------')

        if num_types == 5:
            
            # ----- 5 types
            Encode_types_pct = self.obtain_Encode_types_pct(self.layers, self.neurons, self.Sort_dict.copy(), num_types=num_types)
            
            # --- 1. all
            curve_dict = self.obtain_Encode_types_curve_dict(Encode_types_pct)
            
            fig, ax = plt.subplots(figsize=(10,6))
            self.encode_layer_percent_plot(fig_folder, fig, ax, self.layers, curve_dict, None)
            
            ax.set_title(title:=self.model_structure + '_5_types')
            
            fig.savefig(os.path.join(fig_folder, title+'.png'), bbox_inches='tight')
            fig.savefig(os.path.join(fig_folder, title+'.eps'), bbox_inches='tight')     
            plt.close()
            
            # --- 2. act
            act_idx, act_layers, _ = utils_.activation_function_vgg(self.layers)
            Encode_types_pct = {_: [Encode_types_pct[_][idx] for idx in act_idx] for _ in Encode_types_pct.keys()}
 
            curve_dict = self.obtain_Encode_types_curve_dict(Encode_types_pct)
            
            fig, ax = plt.subplots(figsize=(10,6))
            self.encode_layer_percent_plot(fig_folder, fig, ax, act_layers, curve_dict, None)
            
            ax.set_title(title:=self.model_structure + '_5_types_act')
            
            fig.savefig(os.path.join(fig_folder, title+'.png'), bbox_inches='tight')
            fig.savefig(os.path.join(fig_folder, title+'.eps'), bbox_inches='tight')    
            plt.close()
            
        elif num_types == 23:
            
            Encode_types_pct = self.obtain_Encode_types_pct(self.layers, self.neurons, self.Sort_dict.copy(), num_types=num_types)
            
            # ----- all operation
            figs, axes = plt.subplots(2,2,figsize=(24,12))
            
            self.selectivity_encode_layer_percent_plot_all_operation(fig_folder, figs, axes, 
                                                                    Encode_types_pct,
                                                                    layers=self.layers)
            
            figs.suptitle(title:=self.model_structure + '_23_types')
            figs.subplots_adjust(hspace=0.5, wspace=0.1)
            figs.savefig(os.path.join(fig_folder, title+'.png'), bbox_inches='tight')
            figs.savefig(os.path.join(fig_folder, title+'.eps'), bbox_inches='tight')    
            #figs.savefig(os.path.join(fig_folder, title+'.svg'), bbox_inches='tight', transparent=True)
            plt.close()
            
            # ----- activation function
            act_idx, act_layers, _ = utils_.activation_function_vgg(self.layers)
            Encode_types_pct = {_: [Encode_types_pct[_][idx] for idx in act_idx] for _ in Encode_types_pct.keys()}
            
            figs_act, axes_act = plt.subplots(2,2,figsize=(24,12))
            
            self.selectivity_encode_layer_percent_plot_all_operation(fig_folder, figs_act, axes_act,  
                                                                    Encode_types_pct,
                                                                    layers=act_layers)
            
            figs.suptitle(title:=self.model_structure + '_23_types_act')
            figs_act.subplots_adjust(hspace=0.5, wspace=0.1)
            figs_act.savefig(os.path.join(fig_folder, title+'.png'), bbox_inches='tight')
            figs_act.savefig(os.path.join(fig_folder, title+'.eps'), bbox_inches='tight')     
            #figs_act.savefig(os.path.join(fig_folder, title+'.svg'), bbox_inches='tight', transparent=True)
            plt.close()
            
        else:
            
            raise ValueError(f"[Codinfo] num_types {num_types} not supported, choose from '5', '23'.")
    
    
    @staticmethod
    def calculate_intersection_point(y1, y2, num_interpolate=10000):
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
            intersection_y = f1(intersection_x).item()
        else:
            intersection_y = None
        
        return intersection_x, intersection_y
    
    
    # ----- under construction...
    @staticmethod
    def encode_layer_percent_plot(fig_folder, fig, ax, layers=None, curve_dict=None, point_dict=None):
        """
            this function is the basic function to plot pct of different types of units over layers
        """
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        if curve_dict is not None:
            for curve in curve_dict.keys():    
                curve = curve_dict[curve]
                ax.plot(curve['values'], color=curve['color'], linestyle=curve['linestyle'], linewidth=curve['linewidth'], label=curve['label'])
                
                if 'std' in curve.keys():
                    ax.fill_between(np.arange(len(layers)), curve['values']-curve['std'], curve['values']+curve['std'], edgecolor=None, facecolor=utils_.lighten_color(curve['color']), alpha=0.75)
            
                
        if point_dict is not None:
            for point in point_dict.keys():
                point = point_dict[point]
                ax.scatter(point['point']['x'], point['point']['y'], color=point['color'], linewidth=1.0, label=point['label'])
                ax.vlines(point['point']['x'], 0, point['point']['y'], color='gray', linewidth=1.0)
        
        ax.legend(framealpha=0.5)
        ax.grid(True)

        ax.set_xticks(np.arange(len(layers)), layers, rotation='vertical')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim([0,100])
        

    def selectivity_encode_layer_percent_plot_all_operation(self, fig_folder, figs, axes,  
                                                            Encode_types_pct,
                                                            layers=None):

        
        # ----- Fig 1
        inter_x, inter_y = self.calculate_intersection_point(Encode_types_pct['encode'], Encode_types_pct['non_encode'])
        inter_x_s, inter_y_s = self.calculate_intersection_point(Encode_types_pct['si'], Encode_types_pct['non_encode'])
        inter_x_m, inter_y_m = self.calculate_intersection_point(Encode_types_pct['mi'], Encode_types_pct['non_encode'])

        Encode_types_pct_ = {_: Encode_types_pct[_] for _ in ['non_encode', 'si', 'mi', 'encode', 'wsi', 'wmi', 'weak_encode']}
        curve_dict = self.obtain_Encode_types_curve_dict(Encode_types_pct_)
        
        point_dict = {
            'intersect_e': encode_layer_percent_plot_dict(point={'x':inter_x,'y':inter_y}, 
                                                               #label=f'intersect_e {inter_x:.2f}'
                                                               ),
            'intersect_s': encode_layer_percent_plot_dict(point={'x':inter_x_s,'y':inter_y_s}, 
                                                               #label=f'intersect_s {inter_x_s:.2f}'
                                                               ),
            'intersect_m': encode_layer_percent_plot_dict(point={'x':inter_x_m,'y':inter_y_m}, 
                                                               #label=f'intersect_m {inter_x_m:.2f}'
                                                               ),
            }
        
        # ---
        self.encode_layer_percent_plot(fig_folder, figs, axes[0,0], layers, curve_dict, point_dict)
        axes[0,0].set_title('encode vs non_encode')
        

        # ----- Fig 2
        inter_x, inter_y = self.calculate_intersection_point(Encode_types_pct['s_encode'], Encode_types_pct['s_non_encode'])
        inter_x_s, inter_y_s = self.calculate_intersection_point(Encode_types_pct['s_si'], Encode_types_pct['s_non_encode'])
        inter_x_m, inter_y_m = self.calculate_intersection_point(Encode_types_pct['s_mi'], Encode_types_pct['s_non_encode'])
        
        Encode_types_pct_ = {_: Encode_types_pct[_] for _ in ['s_encode', 's_si', 's_mi', 's_weak_encode', 's_wsi', 's_wmi', 's_non_encode', 's']}
        curve_dict = self.obtain_Encode_types_curve_dict(Encode_types_pct_)
  
        point_dict = {
            'intersect_s-e': encode_layer_percent_plot_dict(point={'x':inter_x,'y':inter_y}, 
                                                                 #label=f'intersect_s-e {inter_x:.2f}'
                                                                 ),
            'intersect_s-s': encode_layer_percent_plot_dict(point={'x':inter_x_s,'y':inter_y_s}, 
                                                                 #label=f'intersect_s-s {inter_x_s:.2f}'
                                                                 ),
            'intersect_s-m': encode_layer_percent_plot_dict(point={'x':inter_x_m,'y':inter_y_m}, 
                                                                 #label=f'intersect_s-m {inter_x_m:.2f}'
                                                                 ),
            }
        
        self.encode_layer_percent_plot(fig_folder, figs, axes[0,1], layers, curve_dict, point_dict)
        axes[0,1].set_title('sensitive')

        # ----- Fig 3
        Encode_types_pct_ = {_: Encode_types_pct[_] for _ in ['ns_encode', 'ns_si', 'ns_mi', 'ns_weak_encode', 'ns_wsi', 'ns_wmi', 'ns_non_encode', 'ns']}
        curve_dict = self.obtain_Encode_types_curve_dict(Encode_types_pct_)

        self.encode_layer_percent_plot(fig_folder, figs, axes[1,0], layers, curve_dict, None)
        axes[1,0].set_title('non_sensitive')
        
        # ----- Fig 4
        Encode_types_pct_ = {_: Encode_types_pct[_] for _ in ['s', 'ns', 'encode', 'weak_encode', 'non_encode']}
        curve_dict = self.obtain_Encode_types_curve_dict(Encode_types_pct_)
        
        # ---
        self.encode_layer_percent_plot(fig_folder, figs, axes[1,1], layers, curve_dict, None)
        axes[1,1].set_title('sensitive and encode')
        
    
    def plot_Encode_freq(self, draw_encode_frequency_layers=False):        # general figure for encoding frequency
        """
            [Sept 9, 2023] abandonded the use of dict_based frequency and the pd.DataFrame.from_dict(freq_dic) plotting
            
        """
        print('-------------------------------')
        print('[Codinfo] plotting frequency...')
        print('-------------------------------')
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        fig_folder = os.path.join(self.dest_Encode, 'Figures')
        utils_.make_dir(fig_folder)
        
        # -----
        freq_dict = self.generate_freq_map()

        print('[Codinfo] plotting Encode_freq...')
        
        # ----- exterior plots
        if draw_encode_frequency_layers:
            self.draw_encode_frequency_layers(freq_dict['all_encode'])

        # -----
        idx, layers, _ = utils_.activation_function_vgg(self.layers)
        
        vmin = 1.
        vmax = 0.
        for key in list(freq_dict.keys()):
            vmin = np.min([vmin, np.min(freq_dict[key])])
            vmax = np.max([vmax, np.max(freq_dict[key])])
        
        # ---
        encode_type_list = ['encode', 's_si', 's_mi', 'ns_si', 'ns_mi']
        weak_encode_type_list = ['weak_encode', 's_wsi', 's_wmi', 'ns_wsi', 'ns_wmi']
        all_encode_type_list = ['all_encode', 's_all_si', 's_all_mi', 'ns_all_si', 'ns_all_mi']
        
        # ----- raw 2D fig
        fig = plt.figure(figsize=(20, 30))
        cmap = 'turbo'
        
        self.draw_encode_frequency_component(x_position=0.15, y_position=0.7, x_width=0.25, x_height=0.225, 
                                             sub_x_position=0.425, sub_y_position=0.7, sub_x_step=0.145, sub_y_step=0.115, sub_width=0.130, sub_height=0.1025,
                                             freq_dict=freq_dict, title='all_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=True, sub_dict=all_encode_type_list[1:])
        
        # ---
        self.draw_encode_frequency_component(x_position=0.15, y_position=0.4, x_width=0.25, x_height=0.225, 
                                             sub_x_position=0.425, sub_y_position=0.4, sub_x_step=0.145, sub_y_step=0.115, sub_width=0.130, sub_height=0.1025,
                                             freq_dict=freq_dict, title='encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=encode_type_list[1:])
        
        # ---
        self.draw_encode_frequency_component(x_position=0.15, y_position=0.15, x_width=0.25, x_height=0.225, 
                                             sub_x_position=0.425, sub_y_position=0.15, sub_x_step=0.145, sub_y_step=0.115, sub_width=0.130, sub_height=0.1025,
                                             freq_dict=freq_dict, title='weak_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=weak_encode_type_list[1:])
        
        # ---
        self.draw_encode_frequency_component(x_position=0.75, y_position=0.4, x_width=0.25, x_height=0.225, 
                                             sub_x_position=None, sub_y_position=None, sub_x_step=None, sub_y_step=None, sub_width=0.130, sub_height=0.1025,
                                             freq_dict=freq_dict, title='s_non_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=None)

        # ---
        self.draw_encode_frequency_component(x_position=0.75, y_position=0.15, x_width=0.25, x_height=0.225, 
                                             sub_x_position=None, sub_y_position=None, sub_x_step=None, sub_y_step=None, sub_width=0.130, sub_height=0.1025,
                                             freq_dict=freq_dict, title='ns_non_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=None)
        
        # ---
        cax = fig.add_axes([1.02, 0.3, 0.01, 0.45]) 
        
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)   # plot colorbar based on arbitrary vmin and vmax
            
        fig.suptitle(f'layer - ID [{self.model_structure}]', x=0.55, y=0.95, fontsize=28)
        
        fig.savefig(os.path.join(fig_folder, 'layer and ID (2D).png'), bbox_inches='tight')
        fig.savefig(os.path.join(fig_folder, 'layer and ID (2D).eps'), bbox_inches='tight')
        plt.close()
        
        
        # ----- raw 3D fig
        fig = plt.figure(figsize=(20, 30))
        cmap = 'turbo'
        
        self.draw_encode_frequency_component_3D(x_position=0.15, y_position=0.7, x_width=0.25, x_height=0.225,
                                            sub_x_position=0.425, sub_y_position=0.7, sub_x_step=0.13, sub_y_step=0.1125, sub_width=0.155, sub_height=0.1025,
                                            freq_dict=freq_dict, title='all_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                            label_on=True, sub_dict=all_encode_type_list[1:])
        
        # ---
        self.draw_encode_frequency_component_3D(x_position=0.15, y_position=0.4, x_width=0.25, x_height=0.225, 
                                             sub_x_position=0.425, sub_y_position=0.4, sub_x_step=0.13, sub_y_step=0.1125, sub_width=0.155, sub_height=0.1025,
                                             freq_dict=freq_dict, title='encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=encode_type_list[1:])
        
        # ---
        self.draw_encode_frequency_component_3D(x_position=0.15, y_position=0.15, x_width=0.25, x_height=0.225, 
                                             sub_x_position=0.425, sub_y_position=0.15, sub_x_step=0.13, sub_y_step=0.1125, sub_width=0.155, sub_height=0.1025,
                                             freq_dict=freq_dict, title='weak_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=weak_encode_type_list[1:])
        
        # ---
        self.draw_encode_frequency_component_3D(x_position=0.75, y_position=0.4, x_width=0.25, x_height=0.225, 
                                             sub_x_position=None, sub_y_position=None, sub_x_step=None, sub_y_step=None, sub_width=0.155, sub_height=0.1025,
                                             freq_dict=freq_dict, title='s_non_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=None)

        # ---
        self.draw_encode_frequency_component_3D(x_position=0.75, y_position=0.15, x_width=0.25, x_height=0.225, 
                                             sub_x_position=None, sub_y_position=None, sub_x_step=None, sub_y_step=None, sub_width=0.155, sub_height=0.1025,
                                             freq_dict=freq_dict, title='ns_non_encode', vmin=vmin, vmax=vmax, cmap=cmap, idx=idx, 
                                             label_on=False, sub_dict=None)
        
        cax = fig.add_axes([1.02, 0.3, 0.01, 0.45])
        
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        fig.suptitle(f'layer - ID (3D) [{self.model_structure}]', x=0.55, y=0.95, fontsize=28)
 
        fig.savefig(os.path.join(fig_folder, 'layer and ID (3D).png'), bbox_inches='tight')
        fig.savefig(os.path.join(fig_folder, 'layer and ID (3D).eps'), bbox_inches='tight')
        plt.close()
        
        # ---
        if hasattr(self, 'Encode_dict'):
            delattr(self, 'Encode_dict')     # release the memory

    def draw_encode_frequency_layers(self, freq, ):
        """
            this functions provides the encode frequency of each layer
        """
        print('[Codinfo] Executing encode frequency layers...')
 
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
        
        freq_path = os.path.join(self.dest_Encode, 'freq.pkl')
        
        # ---
        if os.path.exists(freq_path):
            
            freq_dict = utils_.load(freq_path)
            
            if 'all_all_encode' in freq_dict.keys():
                
                freq_dict = {
                    'encode': freq_dict['all_encode'],
                    'weak_encode': freq_dict['all_weak_encode'],
                    'all_encode': freq_dict['all_all_encode'],
                    
                    's_si': freq_dict['sensitive_si'],
                    's_mi': freq_dict['sensitive_mi'],
                    'ns_si': freq_dict['non_sensitive_si'],
                    'ns_mi': freq_dict['non_sensitive_mi'],
                    
                    's_wsi': freq_dict['sensitive_wsi'],
                    's_wmi': freq_dict['sensitive_wmi'],
                    'ns_wsi': freq_dict['non_sensitive_wsi'],
                    'ns_wmi': freq_dict['non_sensitive_wmi'],
                    
                    's_all_si': freq_dict['sensitive_all_si'],
                    's_all_mi': freq_dict['sensitive_all_mi'],
                    'ns_all_si': freq_dict['non_sensitive_all_si'],
                    'ns_all_mi': freq_dict['non_sensitive_all_mi'],
                    
                    's_non_encode': freq_dict['sensitive_non_encode'],
                    'ns_non_encode': freq_dict['non_sensitive_non_encode']
                    }
                
                os.remove(freq_path)
                utils_.dump(freq_dict, freq_path)
                print('------------------------------------------')
                print('[Codinfo] freq_dict updated')
                print('------------------------------------------')
                
        else:
            
            # --- init
            if not hasattr(self, 'Sort_dict'):
                self.Sort_dict = utils_.load(os.path.join(self.dest_Encode, 'Sort_dict.pkl'))

            if not hasattr(self, 'Encode_dict'):
                self.Encode_dict = utils_.load(os.path.join(self.dest_Encode, 'Encode_dict.pkl'))

            freq_dict = {
                'encode': self.generate_freq_map_single('strong'),
                'weak_encode': self.generate_freq_map_single('weak'),
                'all_encode': self.generate_freq_map_single(),
                
                's_si': self.generate_freq_map_single(['s_si']),
                's_mi': self.generate_freq_map_single(['s_mi']),
                'ns_si': self.generate_freq_map_single(['ns_si']),
                'ns_mi': self.generate_freq_map_single(['ns_mi']),
                
                's_wsi': self.generate_freq_map_single(['s_wsi']),
                's_wmi': self.generate_freq_map_single(['s_wmi']),
                'ns_wsi': self.generate_freq_map_single(['ns_wsi']),
                'ns_wmi': self.generate_freq_map_single(['ns_wmi']),
                
                's_all_si': self.generate_freq_map_single(['s_si', 's_wsi']),
                's_all_mi': self.generate_freq_map_single(['s_mi', 's_wmi']),
                'ns_all_si': self.generate_freq_map_single(['ns_si', 'ns_wsi']),
                'ns_all_mi': self.generate_freq_map_single(['ns_mi', 'ns_wmi']),
                
                's_non_encode': self.generate_freq_map_single(['s_non_encode']),
                'ns_non_encode': self.generate_freq_map_single(['ns_non_encode']),
                }

            utils_.dump(freq_dict, freq_path)
        
        return freq_dict
    
    
    # FIXME --- seems this function can be simplfied
    def generate_freq_map_single(self, unit_type=None):
        """
            this function provides the encode performace for the correct id
        """
        
        # ---
        correct_id = utils_.lexicographic_order(50)

        freq = []

        for layer_idx, layer in tqdm(enumerate(self.layers), desc=f"{unit_type}"):
            
            unit_encode_dict = self.Encode_dict[layer]
            
            if unit_type == None:
                
                target_units = np.arange(self.neurons[layer_idx])
                
                unit_encode_list = [[*unit_encode_dict[unit]['weak_encode'], *unit_encode_dict[unit]['encode']] for unit in target_units]
               
            else:
                
                if isinstance(unit_type, str):
                    
                    target_units = np.arange(self.neurons[layer_idx])
                
                    if unit_type == 'weak':
                        unit_encode_list = [unit_encode_dict[unit]['weak_encode'] for unit in target_units]
                        
                    elif unit_type == 'strong':
                        unit_encode_list = [unit_encode_dict[unit]['encode'] for unit in target_units]
                
                if isinstance(unit_type, list):
                
                    target_units = np.array([])
                    
                    for type_ in unit_type:
                        
                        target_units = np.concatenate((target_units, self.Sort_dict[layer]['advanced_type'][type_]))     
                    
                    # -----
                    if len(unit_type) == 2:     # this will also collect the 'weak_encoded' ids of 'encode' units
                    
                        unit_encode_list = [[*unit_encode_dict[unit]['weak_encode'], *unit_encode_dict[unit]['encode']] for unit in target_units]
                        
                    elif len(unit_type) == 1 and 'non_encode' in unit_type[0]:     # both are empty list
                        
                        unit_encode_list = []
                        
                        for unit in target_units:
                            
                            assert len(unit_encode_dict[unit]['weak_encode']) == 0 and len(unit_encode_dict[unit]['encode']) == 0
                            
                            unit_encode_list.append([*unit_encode_dict[unit]['weak_encode'], *unit_encode_dict[unit]['encode']])
                            
                        #unit_encode_list = [[*unit_encode_dict[unit]['weak_encode'], *unit_encode_dict[unit]['encode']] for unit in target_units]
                    
                    elif len(unit_type) == 1 and 'w' in unit_type[0]:     # assert the 'weak_encode' unit strongly encode no id
                        
                        unit_encode_list = []
                    
                        for unit in target_units:
                            
                            assert len(unit_encode_dict[unit]['encode']) == 0
                            
                            unit_encode_list.append(unit_encode_dict[unit]['weak_encode'])
                            
                        #unit_encode_list = [unit_encode_dict[unit]['weak_encode'] for unit in target_units]
                        
                    elif len(unit_type) == 1:     # this will only collect the strongly encoded ids of 'encode' units
                    
                        unit_encode_list = []
                    
                        for unit in target_units:
                            
                            assert len(unit_encode_dict[unit]['encode']) > 0
                            
                            unit_encode_list.append(unit_encode_dict[unit]['encode'])
                    
                        #unit_encode_list = [unit_encode_dict[unit]['encode'] for unit in target_units]
                        
            pool = [id_ for encoded_ids in unit_encode_list for id_ in encoded_ids]     # for all unit
            frequency = Counter(pool)
            
            frequency = {correct_id[_-1]: frequency[_] for _ in range(1,51)}     # map correct_id
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
    
    @staticmethod
    def single_acc(feature, idx, label, tqdm_bar):
        if len(idx) != 0:
            acc = utils_.SVM_classification(feature[:, idx], label, test_size=0.2, random_state=42)
        else:
            acc = 0. 
            
        tqdm_bar.update(1)
        
        return acc
    
    #FIXME
    def SVM(self, num_types=5):
        """
            test version for merging with previous results
            
            [notice] manually select the used cell types, now use new 5 types
        """
        
        print('[Codinfo] computing SVM...')
        
        SVM_path = os.path.join(self.dest_Encode, 'SVM.pkl')
        
        #FIXME --- tmp
        if os.path.exists(SVM_path):
            
            layer_SVM = utils_.load(SVM_path)
            
        else:
            
            if not hasattr(self, 'Sort_dict'):
                self.Sort_dict = utils_.load(os.path.join(self.dest_Encode, 'Sort_dict.pkl'))
            
            #ANOVA_idces = utils_.pickle_load(os.path.join(self.dest, 'ANOVA/ANOVA_idces.pkl'))
            
            # ----- for SVM
            label = utils_.lexicographic_order(50)+1
            label = np.repeat(label, 10)
            
            layer_SVM = {}
            
            for layer in self.layers:
                
                # --- depends
                tqdm_bar = tqdm(total=num_types+1, desc=f'{layer}')
                
                feature = utils_.load(os.path.join(self.root, layer+'.pkl'), verbose=False)

                Sort_dict = self.Sort_dict[layer]
                
                # -----
                #sensitive_idx = ANOVA_idces[layer]
                #non_sensitive_idx = np.array(list(set(np.arange(feature.shape[1])) - set(sensitive_idx)))
                
                #si = Sort_dict['basic_type']['si']
                #mi = Sort_dict['basic_type']['mi']
                #wsi = Sort_dict['basic_type']['wsi']
                #wmi = Sort_dict['basic_type']['wmi']
                # -----
                
                # -----
                layer_SVM.update({layer:
                    {
                        
                    # --- all units
                    'all_acc': self.single_acc(feature, np.arange(feature.shape[1]), label, tqdm_bar),     # reference line
                    
                    # --- sensitive conditions
                    #'sensitive_acc': self.single_acc(feature, sensitive_idx, label, tqdm_bar),
                    #'non_sensitive_acc': self.single_acc(feature, non_sensitive_idx, label, tqdm_bar),
                    
                    # --- encode conditions
                    #'si_acc': self.single_acc(feature, si, label, tqdm_bar),
                    #'mi_acc': self.single_acc(feature, mi, label, tqdm_bar),
                    #'encode_acc': self.single_acc(feature, [*si, *mi], label, tqdm_bar),
                    
                    # --- weak encode conditions
                    #'wsi_acc': self.single_acc(feature, wsi, label, tqdm_bar),
                    #'wmi_acc': self.single_acc(feature, wmi, label, tqdm_bar),
                    #'weak_encode_acc': self.single_acc(feature, [*wsi, *wmi], label, tqdm_bar),
                    
                    # --- non encode conditions
                    #'non_encode_acc': self.single_acc(feature, Sort_dict['basic_type']['non_encode'], label, tqdm_bar),
                    #'s_non_encode_acc': self.single_acc(feature, Sort_dict['advanced_type']['s_non_encode'], label, tqdm_bar),
                    #'ns_non_encode_acc': self.single_acc(feature, Sort_dict['advanced_type']['ns_non_encode'], label, tqdm_bar),
                    
                    # --- sensitive + encode conditions
                    's_si_acc': self.single_acc(feature, Sort_dict['advanced_type']['s_si'], label, tqdm_bar),
                    's_mi_acc': self.single_acc(feature, Sort_dict['advanced_type']['s_mi'], label, tqdm_bar),
                    #'s_encode_acc': self.single_acc(feature, [*Sort_dict['advanced_type']['s_si'], *Sort_dict['advanced_type']['s_mi']], label, tqdm_bar),
                    
                    #'ns_si_acc': self.single_acc(feature, Sort_dict['advanced_type']['ns_si'], label, tqdm_bar),
                    #'ns_mi_acc': self.single_acc(feature, Sort_dict['advanced_type']['ns_mi'], label, tqdm_bar),
                    #'ns_encode_acc': self.single_acc(feature, [*Sort_dict['advanced_type']['ns_si'], *Sort_dict['advanced_type']['ns_mi']], label, tqdm_bar),
                    
                    # --- sensitive + weak encode conditions
                    's_wsi_acc': self.single_acc(feature, Sort_dict['advanced_type']['s_wsi'], label, tqdm_bar),
                    's_wmi_acc': self.single_acc(feature, Sort_dict['advanced_type']['s_wmi'], label, tqdm_bar),
                    #'s_weak_encode_acc': self.single_acc(feature, [*Sort_dict['advanced_type']['s_wsi'], *Sort_dict['advanced_type']['s_wmi']], label, tqdm_bar),
                    
                    #'ns_wsi_acc': self.single_acc(feature, Sort_dict['advanced_type']['ns_wsi'], label, tqdm_bar),
                    #'ns_wmi_acc': self.single_acc(feature, Sort_dict['advanced_type']['ns_wmi'], label, tqdm_bar),
                    #'ns_weak_encode_acc': self.single_acc(feature, [*Sort_dict['advanced_type']['ns_wsi'], *Sort_dict['advanced_type']['ns_wmi']], label, tqdm_bar),
                    
                    # --- encode / weak encode conditions
                    #'all_si_acc': self.single_acc(feature, [*si, *wsi], label, tqdm_bar),
                    #'all_mi_acc': self.single_acc(feature, [*mi, *wmi], label, tqdm_bar),
                    #'all_encode_acc': self.single_acc(feature, [*si, *wsi, *mi, *wmi], label, tqdm_bar),
                            
                    # --- sensitive + encode/ weak encode conditions
                    #'all_s_si_acc': self.single_acc(feature, [*Sort_dict['advanced_type']['s_si'], *Sort_dict['advanced_type']['s_wsi']], label, tqdm_bar),
                    #'all_s_mi_acc': self.single_acc(feature, [*Sort_dict['advanced_type']['s_mi'], *Sort_dict['advanced_type']['s_wmi']], label, tqdm_bar),
                    #'all_s_strong_encode_acc': self.single_acc(feature, [*Sort_dict['advanced_type']['s_si'], *Sort_dict['advanced_type']['s_wsi'], *Sort_dict['advanced_type']['s_mi'], *Sort_dict['advanced_type']['s_wmi']], label, tqdm_bar),
                    
                    #'all_ns_si_acc': self.single_acc(feature, [*Sort_dict['advanced_type']['ns_si'], *Sort_dict['advanced_type']['ns_wsi']], label, tqdm_bar),
                    #'all_ns_mi_acc': self.single_acc(feature, [*Sort_dict['advanced_type']['ns_mi'], *Sort_dict['advanced_type']['ns_wmi']], label, tqdm_bar),
                    #'all_ns_strong_encode_acc': self.single_acc(feature, [*Sort_dict['advanced_type']['ns_si'], *Sort_dict['advanced_type']['ns_wsi'], *Sort_dict['advanced_type']['ns_mi'], *Sort_dict['advanced_type']['ns_wmi']], label, tqdm_bar),
                    
                    # --- this new non_encode means all other types except s_si, s_wsi, s_mi and s_wmi
                    'n_e_acc': self.single_acc(feature, [*Sort_dict['advanced_type']['s_non_encode'], *Sort_dict['advanced_type']['ns_non_encode'],
                                                            *Sort_dict['advanced_type']['ns_si'], *Sort_dict['advanced_type']['ns_wsi'], 
                                                            *Sort_dict['advanced_type']['ns_mi'], *Sort_dict['advanced_type']['ns_wmi']], label, tqdm_bar)
                    
                    }})
                    
            utils_.dump(layer_SVM, SVM_path)
            
            print('[Codinfo] SVM calculation done')
        
        return layer_SVM
            
    
    def SVM_plot(self, plot_types=5):
        
        print('[Codinfo] Executing SVM plot...')
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        SVM_fig_folder = os.path.join(self.dest_Encode, 'SVM_Figures')
        utils_.make_dir(SVM_fig_folder)
        
        # ----- depends
        layer_SVM = self.SVM()
        # -----
        
        # ----- simplified 5 types
        if plot_types == 5:

            # --- all
            self.acc_plot_dict = {
                'all_acc_plot':[layer_SVM[layer]['all_acc'] for layer in self.layers],
                
                's_si_acc_plot':[layer_SVM[layer]['s_si_acc'] for layer in self.layers],
                's_wsi_acc_plot':[layer_SVM[layer]['s_wsi_acc'] for layer in self.layers],
                
                's_mi_acc_plot':[layer_SVM[layer]['s_mi_acc'] for layer in self.layers],
                's_wmi_acc_plot':[layer_SVM[layer]['s_wmi_acc'] for layer in self.layers],
                
                'n_e_acc_plot':[layer_SVM[layer]['n_e_acc'] for layer in self.layers],
                }
            
            fig, ax = plt.subplots(figsize=(14,10))
            self.SVM_plot_single_fig(ax)
            
            ax.set_title(title:=f'5 types [{self.model_structure}]')
            fig.savefig(os.path.join(SVM_fig_folder, f'{title}.png'), bbox_inches='tight')
            fig.savefig(os.path.join(SVM_fig_folder, f'{title}.eps'), bbox_inches='tight', format='eps')
    
            plt.close('all')
            
            # ---
            _, layers, _ = utils_.activation_function_vgg(self.layers)
            layer_neuron_SVM = {_: layer_SVM[_] for _ in layers}
            
            # --- act
            self.acc_plot_dict = {
                'all_acc_plot':[layer_neuron_SVM[layer]['all_acc'] for layer in layers],
                
                's_si_acc_plot':[layer_neuron_SVM[layer]['s_si_acc'] for layer in layers],
                's_wsi_acc_plot':[layer_neuron_SVM[layer]['s_wsi_acc'] for layer in layers],
                
                's_mi_acc_plot':[layer_neuron_SVM[layer]['s_mi_acc'] for layer in layers],
                's_wmi_acc_plot':[layer_neuron_SVM[layer]['s_wmi_acc'] for layer in layers],
                
                'n_e_acc_plot':[layer_neuron_SVM[layer]['n_e_acc'] for layer in layers],
                }
            
            fig, ax = plt.subplots(figsize=(8, 6))
            self.SVM_plot_single_fig(ax)
            
            ax.set_title(title:=f'5 types act [{self.model_structure}]')
            fig.savefig(os.path.join(SVM_fig_folder, f'{title}.png'), bbox_inches='tight')
            fig.savefig(os.path.join(SVM_fig_folder, f'{title}.eps'), bbox_inches='tight', format='eps')
    
            plt.close('all')
        
        # --- full types
        elif plot_types == 31:
            
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
         
        else:
            raise ValueError(f"[Codinfo] plot_types {plot_types} is invalid, choose from '5', '31'.")
        
        # -----
        print('[Codinfo] Image saved')
    
    
    def SVM_plot_single_fig(self, ax):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        for acc_plot in list(self.acc_plot_dict.keys()):
 
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
                ax.plot(self.acc_plot_dict['s_non_encode_acc_plot'], 'red', label='s_non_encode', linestyle=(0,(3,1,1,1,1,1)))
            if acc_plot == 'ns_non_encode_acc_plot':
                ax.plot(self.acc_plot_dict['ns_non_encode_acc_plot'], 'red', label='ns_non_encode', linestyle=(0,(3,10,1,10)))
            
            # --- [new ne]
            if acc_plot == 'n_e_acc_plot':
                ax.plot(self.acc_plot_dict['n_e_acc_plot'], 'red', label='n_e', linestyle=(0,(3,1,1,1,1,1)))
            # ---
            
        # -----
        num_layers = len(self.acc_plot_dict['all_acc_plot'])
        
        idx, act_layers, _ = utils_.activation_function_vgg(self.layers)
            
        # -----
        if num_layers == len(self.layers):
        
            ax.set_xticks(np.arange(len(self.layers)))
            ax.set_xticklabels(['' if _ not in idx else self.layers[_] for _ in range(len(self.layers))], rotation='vertical')
            
            ax.set_ylim([0, 100])
            ax.set_yticks(np.arange(1, 101))
            ax.set_yticklabels(['' if (_%10)!=0 else _ for _ in range(1,101)], rotation='vertical')
        
        elif num_layers == len(idx):
            
            ax.set_xticks(np.arange(len(idx)))
            ax.set_xticklabels(act_layers, rotation='vertical')
            
        else:
            raise RuntimeError(f"[Codinfo] num_layers {num_layers} is in valid, select from len(self.layers) '{len(self.layers)}', len(act_layers) '{len(idx)}'.")
        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(ncol=3, loc="upper left", framealpha=0.5)
    
    
    # ------------------------------------------------------------------------------------------------------------------
    #FIXME --- test version
    def plot_stacked_responses(self, num_types=5, start_layer_idx=-12):
        """
            this function is memory consuming
        """
        
        assert start_layer_idx < 0, f'[Coderror] start_layer_idx {start_layer_idx} must be negative in current design'
        
        print(f'[Codinfo] Executing plot_stacked_responses... | num_types: {num_types} | num_layers: {np.abs(start_layer_idx)}')
        
        if not hasattr(self, 'Sort_dict'):
            self.Sort_dict = utils_.load(os.path.join(self.dest_Encode, 'Sort_dict.pkl'))
        
        self.fig_folder = os.path.join(self.dest_Encode, 'Layer_stacked_responses')
        utils_.make_dir(self.fig_folder)
        
        layers = self.layers[start_layer_idx:]     # from final conv to the end
        
        # -----
        self.fig_folder = os.path.join(self.fig_folder, str(num_types))
        utils_.make_dir(self.fig_folder)
        
        if np.abs(start_layer_idx) < 13:
            
            Parallel(n_jobs=12)(delayed(self.plot_stacked_responses_single_layer)(layer, num_types) for layer in layers)
    
        else:
        
            for layer in layers:
                
                self.plot_stacked_responses_single_layer(layer, num_types)
                
                #mem = psutil.virtual_memory()
                #swap = psutil.swap_memory()
                #print(f'\nCPU+SWAP used: {(mem.used+swap.used)/1024/1024/1024:.3f} / 256')
                #print(f'\nCPU used pct: {mem.percent} %')
                #print(sys.getrefcount(self.plot_stacked_responses_single_layer))
                
                gc.collect()
            
        delattr(self, 'fig_folder')
    
    #FIXME
    def plot_stacked_responses_single_layer(self, layer, num_types=5, 
                                       num_classes=50, num_samples=10):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 14})
        
        feature = utils_.load(os.path.join(self.root, layer+'.pkl'), verbose=False)

        if num_types == 5:
            
            # select the classes
            idx_dict = {
                's_si': self.Sort_dict[layer]['advanced_type']['s_si'],
                's_wsi': self.Sort_dict[layer]['advanced_type']['s_wsi'],
                
                's_mi': self.Sort_dict[layer]['advanced_type']['s_mi'],
                's_wmi': self.Sort_dict[layer]['advanced_type']['s_wmi'],
                
                'n_e': np.concatenate((self.Sort_dict[layer]['advanced_type']['s_non_encode'], 
                        self.Sort_dict[layer]['advanced_type']['ns_si'], 
                        self.Sort_dict[layer]['advanced_type']['ns_wsi'], 
                        self.Sort_dict[layer]['advanced_type']['ns_mi'], 
                        self.Sort_dict[layer]['advanced_type']['ns_wmi'], 
                        self.Sort_dict[layer]['advanced_type']['ns_non_encode'])).astype(np.int64)
                }
            
            # init the canvas
            fig, ax = plt.subplots(figsize=(26, 6))
            gs_main = gridspec.GridSpec(1, 5, figure=fig)

            plot_single(fig, gs_main, layer, num_types, idx_dict, feature, num_classes, num_samples)
            
        elif num_types == 10:
            
            # select the classes
            idx_dict = {
                's_si': self.Sort_dict[layer]['advanced_type']['s_si'],
                's_wsi': self.Sort_dict[layer]['advanced_type']['s_wsi'],
                
                's_mi': self.Sort_dict[layer]['advanced_type']['s_mi'],
                's_wmi': self.Sort_dict[layer]['advanced_type']['s_wmi'],
                
                's_non_encode': self.Sort_dict[layer]['advanced_type']['s_non_encode'],
                
                'ns_si': self.Sort_dict[layer]['advanced_type']['ns_si'],
                'ns_wsi': self.Sort_dict[layer]['advanced_type']['ns_wsi'],
                
                'ns_mi': self.Sort_dict[layer]['advanced_type']['ns_mi'],
                'ns_wmi': self.Sort_dict[layer]['advanced_type']['ns_wmi'],
                
                'ns_non_encode': self.Sort_dict[layer]['advanced_type']['ns_non_encode']
                }
            
            # init the canvas
            fig, ax = plt.subplots(figsize=(26,10))
            gs_main = gridspec.GridSpec(2, 5, figure=fig)
            
            plot_single(fig, gs_main, layer, num_types, idx_dict, feature, num_classes, num_samples)
                    
        ax.axis('off')
        ax.plot([],[],color='blue',label='mean')
        ax.plot([],[],color='teal',label='ref')
        ax.plot([],[],color='red',label='threshold')
        
        fig.suptitle(f'{layer} [{self.model_structure}]', y=0.97, fontsize=20)
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, bbox_transform=plt.gcf().transFigure)
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.fig_folder, f'{layer} num_types {num_types}.png'), bbox_inches='tight')
        #fig.savefig(os.path.join(self.fig_folder, f'{layer} {num_types}.eps'), bbox_inches='tight', format='eps')
        #fig.savefig(os.path.join(self.fig_folder, f'{layer} {num_types}.svg'), bbox_inches='tight', format='svg', transparent=True)
        plt.close() 
    

    def plot_sample_responses(self, random_select_units=10, start_layer_idx=-12):
        """
            this function provides boxplot of example units of different types
        """
        print('[Codinfo] Executing plot_sample_responses')
        
        if not hasattr(self, 'Sort_dict'):
            self.Sort_dict = utils_.load(os.path.join(self.dest_Encode, 'Sort_dict.pkl'))
            
        fig_folder = os.path.join(self.dest_Encode, 'Layer_sample_responses')
        utils_.make_dir(fig_folder)

        layers = self.layers[start_layer_idx:]
        
        for layer in layers:
            
            self.layer_fig_folder = os.path.join(fig_folder, f'{layer}')
            utils_.make_dir(self.layer_fig_folder)
            
            Sort_dict = self.Sort_dict[layer]
            feature = utils_.load(os.path.join(self.root, layer+'.pkl'), verbose=False)
            
            idx_dict = {
                's_si': Sort_dict['advanced_type']['s_si'],
                's_wsi': Sort_dict['advanced_type']['s_wsi'],
                
                's_mi': Sort_dict['advanced_type']['s_mi'],
                's_wmi': Sort_dict['advanced_type']['s_wmi'],
                
                's_non_encode': Sort_dict['advanced_type']['s_non_encode'],
                
                'ns_si': Sort_dict['advanced_type']['ns_si'],
                'ns_wsi': Sort_dict['advanced_type']['ns_wsi'],
                
                'ns_mi': Sort_dict['advanced_type']['ns_mi'],
                'ns_wmi': Sort_dict['advanced_type']['ns_wmi'],
                
                'ns_non_encode': Sort_dict['advanced_type']['ns_non_encode']
                }
            
            idx_dict_keys = list(idx_dict.keys())
            
            for key in tqdm(idx_dict_keys, desc=f'{layer}'):     # for each type
            
                test_idces = idx_dict[key]
                
                if test_idces.size == 0:
                    pass
                
                else:
                    if test_idces.size > random_select_units:
                        test_idces = np.random.choice(test_idces, random_select_units)

                    Parallel(n_jobs=-1)(delayed(self.plot_sample_responses_single_layer)(key, layer, feature, idx) for idx in test_idces)  
    
                    gc.collect()
                    
                    
    def plot_sample_responses_single_layer(self, key, layer, feature, idx, plot_option='response'):
        """
            threshold: r'$V_{th}=\bar{x}+2\sqrt{\frac{1}{500}\sum(x_i-\bar{x})^2}$'
            ref: r'$ref=\bar{x}+2\sqrt{\frac{1}{50}\sum(x_i-\bar{x_i})^2}, \bar{x_i} = \frac{1}{10}\sum{x_i}$'
        """
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 14})
        
        colorpool_jet = plt.get_cmap('jet', 50)
        colors = [colorpool_jet(i) for i in range(50)]
        
        y_lim_min = np.min(feature)
        y_lim_max = np.max(feature)
        
        self.type_layer_fig_folder = os.path.join(self.layer_fig_folder, f'{key}')
        utils_.make_dir(self.type_layer_fig_folder)
        
        #x = np.array([[_]*10 for _ in range(1,51)])
        #y = feature[:, idx]    #
        #c = np.repeat(np.array(colors), 10, axis=0)
        
        x = np.repeat(np.arange(1, 51), 10)     # (500)
        y = feature[:, idx]     # (500,)
        c = np.repeat(np.array(colors), 10, axis=0)     # (500)
        
        test_feature = [y.reshape(self.num_classes, self.num_samples)[_,:] for _ in range(self.num_classes)]
        test_feature_mean = np.mean(test_feature, axis=1)     # (50,)
        
        def _plot_sample_responses_response(ax):
            
            ax.scatter(x, y, color=c, s=10)
            
            ax.scatter(np.arange(1,51), test_feature_mean, color=colors, marker='d')
            for _ in range(self.num_classes):
                ax.vlines(_+1, np.min(y), test_feature_mean[_], linestyle='--')
                
            ax.set_title('responses')
            ax.set_ylim([y_lim_min, y_lim_max])
            
        def _plot_sample_responses_boxplot(ax):
            
            boxes = ax.boxplot(test_feature, patch_artist=True, sym='+')
            
            for i, _ in enumerate(boxes['boxes']):
                _.set(color=colors[i], alpha=0.5)
            for i, _ in enumerate(boxes['fliers']):
                _.set(marker='+', markerfacecolor=colors[i], markeredgecolor=colors[i], markersize=10, alpha=0.75)
                
            ax.set_title('boxplot')
            ax.set_ylim([y_lim_min, y_lim_max])
        
        # -----
        fig, ax = plt.subplots(figsize=(10,10))
        
        if plot_option == 'response':
            
            _plot_sample_responses_response(ax)
            
        elif plot_option == 'boxplot':
            
            _plot_sample_responses_boxplot(ax)
            
        elif plot_option == 'all':
 
            _plot_sample_responses_response(ax)
            _plot_sample_responses_response(ax)
            
        else:
            
            raise ValueError(f"[Codinfo] plot_option {plot_option} is invalid, choose from 'response', 'boxplot', 'all'.")
        
        # ---
        ax.hlines(np.mean(y)+2*np.std(y), 1, 50, colors='red', linestyle='--', label='threshold')
        ax.hlines(np.mean(test_feature_mean)+2*np.std(test_feature_mean), 1, 50, colors='teal', linestyle='--', label='ref')
        
        ax.legend(framealpha=0.75)
        
        fig.suptitle(f'[{layer}] unit: {idx}', y=0.98)
        plt.tight_layout()
        fig.savefig(os.path.join(self.type_layer_fig_folder, f'{key}_{idx}.png'), bbox_inches='tight')
        #fig.savefig(os.path.join(self.type_layer_fig_folder, f'{key}_{idx}.eps'), bbox_inches='tight')
        plt.close()
        
        
    # ------------------------------------------------------------------------------------------------------------------
    #FIXME 
    def plot_unit_responses_PDF(self, ):
        """
            [Jan 3, 2023] Task: need to add more section to compare the distribution of different types of units
        """
        
        save_path = os.path.join(self.dest_Encode, 'unit_responses_PDF')
        utils_.make_dir(save_path)
        
        save_path = os.path.join(save_path, 'all')
        utils_.make_dir(save_path)
        
        feature_path_list = [os.path.join(self.root, layer+'.pkl') for layer in self.layers]
        
        Parallel(n_jobs=int(os.cpu_count()/2))(delayed(NN_unit_FR_stats_plot_single)(layer, save_path, feature_path_list[idx], self.model_structure+f'_{layer}') for idx, layer in tqdm(enumerate(self.layers), desc=f'[{self.model_structure}] unit PDF'))
        
        #for idx, layer in tqdm(enumerate(self.layers), desc=f'[{self.model_structure}] unit PDF'):
        #    NN_unit_FR_stats_plot_single(layer, save_path, feature_path_list[idx], self.model_structure+f'_{layer}')
        
        print('[Codinfo] plot_unit_responses_PDF() finished.')
        
        
    def plot_pct_pie_chart(self, target_layers=None):
        
        if not hasattr(self, 'Sort_dict'):
            self.Sort_dict = utils_.load(os.path.join(self.dest_Encode, 'Sort_dict.pkl'))

        save_path = os.path.join(self.dest_Encode, 'pct_pie_chart')
        utils_.make_dir(save_path)
        
        if target_layers is None:
            
            target_layers = self.layers
        
        for target_layer in tqdm(target_layers, desc='Layers'):
                
            NN_pct = self.Sort_dict[target_layer]['advanced_type']
            
            tmp = [NN_pct[_] for _ in ['s_non_encode', 'ns_si', 'ns_wsi', 'ns_mi', 'ns_wmi', 'ns_non_encode']]
            tmp = [__ for _ in tmp for __ in _]
            
            NN_pct_new = {}
    
            for _ in ['s_si', 's_wsi', 's_mi', 's_wmi']:

                NN_pct_new.update({_: NN_pct[_]})
            
            NN_pct_new.update({'n_e': np.array(tmp)})
            
            values = [len(NN_pct_new[_]) for _ in NN_pct_new.keys()]
    
            labels = [f's_SI ({values[0]})', f'w_SI ({values[1]})', f's_MI ({values[2]})', f'w_MI ({values[3]})', f'NE ({values[4]})']
            
            colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']
            
            explode = [0.5 * (1. - (value - min(values)) / (max(values) - min(values))) for value in values]
            
            title = f'{np.sum(values)} Units'
            
            fig, ax = plt.subplots(figsize=(10,6))
            
            utils_.plot_pie_chart(fig, ax, values, labels, title, colors, explode)
            
            fig.savefig(os.path.join(save_path, f'{target_layer}_pct_pie_chart.png'))
            fig.savefig(os.path.join(save_path, f'{target_layer}_pct_pie_chart.eps'))
            
            plt.close()
        
        print('[Codinfo] plot_pct_pie_chart() finished.')
            
        
# ----------------------------------------------------------------------------------------------------------------------
def encode_layer_percent_plot_dict(values=None, point=None, color=None, linestyle=None, linewidth=None, label=None):

    return {
        'values': values,
        'point': point,
        'color': color,
        'linestyle': linestyle,
        'linewidth': linewidth,
        'label': label
        }
        
        
# ----------------------------------------------------------------------------------------------------------------------
def plot_single(fig, gs_main, layer, num_types, idx_dict, feature, num_classes, num_samples):
    """
        [notice] no auto-adjust for figure size, the proper figsize must be manually appointed
    """
    colorpool_jet = plt.get_cmap('jet', 50)
    colors = [colorpool_jet(i) for i in range(50)]
    
    tqdm_bar = tqdm(total=num_types, desc=f'{layer}')
    idx_dict_keys = list(idx_dict.keys())
    
    y_lim_min = np.min(feature)
    y_lim_max = np.max(feature)
    y_lin_range = y_lim_max - y_lim_min
    
    num_cols = gs_main.ncols
    num_rows = gs_main.nrows
    
    for i in range(num_rows):
        for j in range(num_cols):
            
            gs_sub = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[4, 1], subplot_spec=gs_main[i, j])

            ax_left = fig.add_subplot(gs_sub[0])
            ax_right = fig.add_subplot(gs_sub[1])
            
            if (i+j) != 0:
                ax_left.set_xticks([])
                ax_left.set_yticks([])

            if idx_dict[idx_dict_keys[j]].size == 0:
                ax_left.set_title(idx_dict_keys[j] + ' [0.00%]')
                ax_right.set_title('th')

            else:
                feature_test = feature[:, idx_dict[idx_dict_keys[j]]]     # (500, num_units)
                feature_test_groups = feature_test.reshape(num_classes, num_samples, -1)     # (50, 10, num_units)
                
                feature_test_mean = np.mean(feature_test_groups, axis=1)     # (50, num_units)
                
                num_units = len(idx_dict[idx_dict_keys[j]])
                
                # -----
                x = np.tile(np.arange(num_classes), num_units)     # (0,1,...,49,0,1,...)
                y = feature_test_mean.T.reshape(-1)     # every 50 ids for unit by unit
                
                c = np.tile(np.array(colors), [num_units, 1])

                # -----
                ax_left.scatter(x, y, color=c, alpha=0.1, marker='.', s=1)     # use small size to replace adjustable alpha
                # -----
                
                pct = num_units/feature.shape[1]*100
                ax_left.set_title(idx_dict_keys[j] + f' [{pct:.2f}%]')
                # -----
                
                # ----- stats: mean firing rate for each id
                values = feature_test_mean.reshape(-1)    # (50*num_units)
                
                plot_single_subsubplot(ax_left, ax_right, values, color='blue')
                
                # ----- stats: threshold (mean+2std of all 500 values)
                values = np.mean(feature_test, axis=0) + 2*np.std(feature_test, axis=0)     # (num_units,)

                plot_single_subsubplot(ax_left, ax_right, values, color='red')
                
                # ----- stats: ref (mean+2std of all 50 mean values)
                values = np.mean(feature_test_mean, axis=0) + 2*np.std(feature_test_mean, axis=0)     # (num_units,)
                
                plot_single_subsubplot(ax_left, ax_right, values, color='teal')
                
                # -----
                scaling_factor = 0.1
                
                ax_left.set_ylim([y_lim_min - y_lin_range*scaling_factor, y_lim_max + y_lin_range*scaling_factor])
                ax_right.set_ylim([y_lim_min - y_lin_range*scaling_factor, y_lim_max + y_lin_range*scaling_factor])
                ax_right.set_title('th')
                
            tqdm_bar.update(1)


def plot_single_subsubplot(ax_left, ax_right, values, color, scaling_factor=0.1):
    
    if np.std(values) == 0:
        pass
    else:
        kde = gaussian_kde(values)
        
        min_values = np.min(values)
        max_values = np.max(values)
        
        values_range = max_values - min_values
        
        x_vals = np.linspace(min_values - scaling_factor*values_range, max_values + scaling_factor*values_range, 101)
        y_vals = kde(x_vals)
        ax_right.plot(y_vals, x_vals, color=color)
    
        y_vals_max = np.max(y_vals)
        x_vals_max = x_vals[np.where(y_vals==y_vals_max)[0].item()]
        
        ax_left.hlines(x_vals_max, 0, 50, colors=color, alpha=0.75, linestyle='--')
        ax_right.hlines(x_vals_max, np.min(y_vals), np.max(y_vals), colors=color, alpha=0.75, linestyle='--')


# ----------------------------------------------------------------------------------------------------------------------
#TODO - consider the distribution of different types of units
def NN_unit_FR_stats_plot_single(layer, save_path, feature_path, model_structure):
    
    with warnings.catch_warnings():
        
        warnings.simplefilter(action='ignore')
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        feature = utils_.load(feature_path, verbose=False)
        
        fig = Human_Neuron_Records_Process.neuron_FR_stats_plot(model_structure, 'unit', feature)
        
        plt.tight_layout()
        fig.savefig(os.path.join(save_path, layer+'.png'))
        fig.savefig(os.path.join(save_path, layer+'.eps'))
    
    plt.close()
    
    
# ----------------------------------------------------------------------------------------------------------------------
# FIXME - consider to put the correct id labels here rather than locations
def encode_calculation(feature, i):
    """
        parallel computation to obtain encode_dict
        
        return:
            
            encode: the classes with lexicographic label
            weak_encode: ...
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
    
    model_ = models_.vgg.__dict__[model_name](num_classes=50)
    layers, neurons, shapes = utils_.generate_vgg_layers_list_ann(model_, model_name)

    root_dir = '/home/acxyle-workstation/Downloads'

    
    
    for folder in [
                    #'Face Identity Baseline', 
                    #'Face Identity VGG16', 
                    #'Face Identity VGG16bn',
                    'Face Identity SpikingVGG16bn_IF_T4_CelebA2622_fold_3', 
                    #'Face Identity SpikingVGG16bn_IF_T16_CelebA2622',
                    #'Face Identity SpikingVGG16bn_LIF_T4_CelebA2622', 
                    #'Face Identity SpikingVGG16bn_LIF_T16_CelebA2622',
                    #'Face Identity SpikingVGG16bn_LIF_T4_vggface'
                    ]:
        
        selectivity_analyzer = Encode_feaquency_analyzer(root=os.path.join(root_dir, folder), 
                                                         layers=layers, neurons=neurons)
        
        #selectivity_analyzer.calculation_Encode()
        #selectivity_analyzer.plot_Encode_pct()
        
        #selectivity_analyzer.plot_Encode_freq()
        
        # ---
        #selectivity_analyzer.generate_encoded_id_unit_idx()     # <- currently not in use 
        # ---

        #selectivity_analyzer.SVM()
        #selectivity_analyzer.SVM_plot()
    
        #selectivity_analyzer.plot_stacked_responses()
        #selectivity_analyzer.plot_sample_responses()
        
        selectivity_analyzer.plot_unit_responses_PDF()
        #selectivity_analyzer.plot_pct_pie_chart()
    
