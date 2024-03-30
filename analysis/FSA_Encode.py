#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:54:55 2023

@author: acxyle

    ...
    
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import gc
import logging
import warnings
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from collections import Counter

from scipy.stats import gaussian_kde
from matplotlib import gridspec
import pandas as pd


from Bio_Cell_Records_Process import Human_Neuron_Records_Process
import utils_

# ----------------------------------------------------------------------------------------------------------------------
...

# ----------------------------------------------------------------------------------------------------------------------
class FSA_Encode():
    """
        FSA: Face-Selectivity-Analysis
    
        Main functions:
            1) calculate and display **Percentage** of Encode units along layers
            2) calculate and display **Frequency** of encoded identities along layers
    """
    
    def __init__(self, root, layers=None, neurons=None, num_classes=50, num_samples=10, **kwargs):
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
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

        self.ANOVA_idces = utils_.load(os.path.join(self.dest, 'ANOVA/ANOVA_idces.pkl'))   # <- consider to remove this?
        
        self.model_structure = root.split('/')[-1].split(' ')[-1]
     
        self.basic_types = ['s_si', 's_wsi', 's_mi', 's_wmi', 's_non_encode', 'ns_si', 'ns_wsi', 'ns_mi', 'ns_wmi', 'ns_non_encode']   
     
        
    # FIXME ---
    def calculation_Encode(self, normalize=True, sort=True, num_workers=-1, **kwargs):
        """
            ...
        """
        
        utils_._print('Executing calculation_Encode...')
        
        sort_dict_path = os.path.join(self.dest_Encode, 'Sort_dict.pkl')
        encode_dict_path = os.path.join(self.dest_Encode, 'Encode_dict.pkl')

        if (not hasattr(self, 'Sort_dict') and os.path.exists(sort_dict_path)):
            
            self.Sort_dict = self.load_Sort_dict()
        
        else:
            
            # ----- init
            self.Encode_dict = {}
            self.Sort_dict = {}
            
            # --- running
            for layer in self.layers:     # for each layer
                
                feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), normalize=normalize, sort=sort, **kwargs)      # load feature matrix
                
                # ----- 1. ANOVA
                s = self.ANOVA_idces[layer]     # sensitive_idx
                ns = np.setdiff1d(np.arange(feature.shape[1]), s)     # non_sensitive_idx
                
                # ----- 2. Encode
                pl = Parallel(n_jobs=num_workers)(delayed(calculation_Encode)(feature[:, i]) for i in tqdm(range(feature.shape[1]), desc=f'[{layer}] Encode'))  
    
                unit_encode_dict = {i: pl[i] for i in range(len(pl))}    
                
                self.Encode_dict[layer] = unit_encode_dict    
                
                # ----- 2. encode test
                si = []
                wsi = []
                mi = []
                wmi = []
                non_encode = []
                
                for k, v in unit_encode_dict.items():
                    if len(v['encode']) == 1:
                        si.append(k)
                    elif len(v['encode']) == 0 and len(v['weak_encode']) == 1:
                        wsi.append(k)
                    elif len(v['encode']) > 1:
                        mi.append(k)
                    elif len(v['encode']) == 0 and len(v['weak_encode']) > 1:
                        wmi.append(k)
                    elif len(v['encode']) == 0 and len(v['weak_encode']) == 0:
                        non_encode.append(k)
                
                # ----- 3. basic types
                unit_sort_dict = {
                    
                    's_si': np.intersect1d(s, si),
                    's_wsi': np.intersect1d(s, wsi),
                    's_mi': np.intersect1d(s, mi),
                    's_wmi': np.intersect1d(s, wmi),
                    's_non_encode': np.intersect1d(s, non_encode),
                    
                    'ns_si': np.intersect1d(ns, si),
                    'ns_wsi': np.intersect1d(ns, wsi),
                    'ns_mi': np.intersect1d(ns, mi),
                    'ns_wmi': np.intersect1d(ns, wmi),
                    'ns_non_encode': np.intersect1d(ns, non_encode),
                    }
                
                self.Sort_dict[layer] = unit_sort_dict
                
            utils_.dump(self.Sort_dict, sort_dict_path, verbose=True)
            utils_.dump(self.Encode_dict, encode_dict_path, verbose=True)  
            
    
    def calculation_Sort_dict(self, used_unit_types, **kwargs):
        
        basic_types = [_ for _ in used_unit_types if _ in self.basic_types]
        advanced_types = list(set(used_unit_types) - set(basic_types))
        
        advanced_Sort_dict = self.calculation_Sort_dict_advanced(advanced_types, **kwargs)
        
        Sort_dict = {k: {_:v[_] for _ in basic_types} for k,v in self.Sort_dict.items()}
        Sort_dict = {layer: {**Sort_dict[layer], **advanced_Sort_dict[layer]} for layer in self.layers}
        
        return Sort_dict
        
    
    def calculation_Sort_dict_advanced(self, used_unit_types:list[str], **kwargs):
        """
            ...     
        """
        
        advanced_type = _unit_types(used_unit_types)
        advanced_Sort_dict = {layer: {_: np.concatenate([self.Sort_dict[layer][__] for __ in advanced_type[_]]) for _ in used_unit_types} for layer in self.Sort_dict.keys()}
        
        return advanced_Sort_dict
    
    
    # FIXME --- 
    def calculation_units_pct(self, unit_types=['s_si', 's_wsi', 's_mi', 's_wmi', 'non_selective']):
        """
            
            input: unit_types
        
            return: pct of units of each type along layers
            
        """
        # ---
        Sort_dict = self.calculation_Sort_dict(unit_types)
        
        unit_pct = {_: np.array([len(Sort_dict[layer][_])/self.neurons[idx]*100 for idx, layer in enumerate(self.layers)]) for _ in unit_types}
        
        return unit_pct
    
    
    def calculation_curve_dict(self, units_pct):
        """
            this function return the cruve config for each key of Encode_types_dict. 
            Filter the keys of the input if need to select special types
        """
        
        style_config_df = self.plot_Encode_config
        
        curve_dict = {}
        
        for key in units_pct:
            
            if key in style_config_df.index:
                
                config = style_config_df.loc[key]
                
                curve_dict[key] = seal_plot_config(
                                    units_pct[key],
                                    label=config['label'],
                                    color=config['color'],
                                    linestyle=config['linestyle'],
                                    linewidth=config['linewidth']
                )

        return curve_dict
    
    
    @property
    def plot_Encode_config(self, ):
        
        style_config = {
                        'type': [
                            'qualified',
                            's_si', 's_wsi', 's_mi', 's_wmi', 's_non_encode',
                            'ns_si', 'ns_wsi', 'ns_mi', 'ns_wmi', 'ns_non_encode',

                            'si', 'wsi', 'mi', 'wmi', 'non_encode',
                            'sensitive', 'non_sensitive', 'strong_encode', 'weak_encode', 'encode', 
                            
                            'strong_selective', 'weak_selective', 'selective', 'non_selective',
                            's_strong_encode', 's_weak_encode', 's_encode',
                            'ns_strong_encode', 'ns_weak_encode', 'ns_encode',
                                ],
                        
                        'label': [
                            'qualified(all)',
                            's_si', 's_wsi', 's_mi', 's_wmi', 's_non_encode',
                            'ns_si', 'ns_wsi', 'ns_mi', 'ns_wmi', 'ns_non_encode',

                            'si(s+ns)', 'wsi(s+ns)', 'mi(s+ns)', 'wmi(s+ns)', 'non_encode(s+ns)',
                            'sensitive', 'non_sensitive', 'strong_encode', 'weak_encode', 'encode', 
                            
                            'strong_selective', 'weak_selective', 'selective', 'non_selective',
                            's_strong_encode', 's_weak_encode', 's_encode',
                            'ns_strong_encode', 'ns_weak_encode', 'ns_encode',
                                ],
                        
                        'color': [
                            '#000000',
                            '#0000FF', '#00BFFF', '#FF4500', '#FFA07A', '#B22222',
                            '#00008B', '#87CEEB', '#CD5C5C', '#FA8072', '#696969',

                            '#0000CD', '#ADD8E6', '#FF6347', '#FFDAB9', '#808080',
                            '#FF0000', '#008000', '#800080', '#FFC0CB', '#8A2BE2', 
                            
                            '#FFD700', '#FFA500', '#FF8C00', '#D3D3D3',
                            '#9400D3', '#FF69B4', '#8A2BE2', 
                            '#4B0082', '#C71585', '#7B68EE',
                                ],
                        
                        'linestyle': [
                            None,
                            '-', '-', '-', '-', '-',
                            'dotted', 'dotted', 'dotted', 'dotted', 'dotted',

                            '--', (0, (3, 1, 1, 1,)), '--', (0, (3, 1, 1, 1,)), (3,(3,5,1,5)),
                            '-', '-', '-', 'dotted', '-', 

                            '--', '--', '-', '-',
                            '--', 'dotted', '-',
                            '--', 'dotted', '-',
                                ],
                        
                        'linewidth': [
                            3.0, 
                            2.0, 2.0, 2.0, 2.0, 2.0, 
                            2.5, 2.5, 2.5, 2.5, 2.5, 

                            3.0, 3.0, 3.0, 3.0, 3.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 

                            2.0, 2.0, 2.0, 2.0, 
                            3.5, 3.5, 3.5,
                            3.5, 3.5, 3.5,
                                ]
                    }
        
        style_config_df = pd.DataFrame(style_config).set_index('type')
        
        return style_config_df
    
    
    def plot_Encode_pct_single(self, unit_types:list[str]=['s_si', 's_wsi', 's_mi', 's_wmi', 'non_selective'], **kwargs):
        """
            ...
        """

        # --- init
        fig_folder = os.path.join(self.dest_Encode, 'Figures')
        utils_.make_dir(fig_folder)
        
        self.Sort_dict = self.load_Sort_dict()

        units_pct = self.calculation_units_pct(unit_types, **kwargs)
        curve_dict = self.calculation_curve_dict(units_pct, **kwargs)
        
        # --- plot
        fig, ax = plt.subplots(figsize=(10,6))
        
        self.plot_units_pct(fig, ax, self.layers, curve_dict, **kwargs)
        
        ax.set_title(title:=self.model_structure)
        
        fig.savefig(os.path.join(fig_folder, f'{title}-{unit_types}.svg'), bbox_inches='tight')    
        plt.close()
    
    
    def plot_Encode_pct_comprehensive(self, **kwargs):
        """
            ...
        """
        # --- init
        fig_folder = os.path.join(self.dest_Encode, 'Figures')
        utils_.make_dir(fig_folder)
        
        self.Sort_dict = self.load_Sort_dict()
        
        # --- plot, comprehensive
        plot_dict = {
            (0,0): (['non_encode', 'si', 'mi', 'encode', 'wsi', 'wmi', 'weak_encode'], 'encode v.s. non_encode'),
            (0,1): (['strong_selective', 's_si', 's_mi', 'weak_selective', 's_wsi', 's_wmi', 'non_selective', 'sensitive'], 'sensitive'),
            (1,0): (['ns_encode', 'ns_si', 'ns_mi', 'ns_weak_encode', 'ns_wsi', 'ns_wmi', 'ns_non_encode', 'non_sensitive'], 'non_sensitive'),
            (1,1): (['sensitive', 'non_sensitive', 'encode', 'weak_encode', 'non_encode'], 'sensitive and encode')
            }
        
        fig, ax = plt.subplots(2,2,figsize=(18,10))

        for k,v in plot_dict.items():
            
            units_pct = self.calculation_units_pct(v[0], **kwargs)
            curve_dict = self.calculation_curve_dict(units_pct, **kwargs)
            
            self.plot_units_pct(fig, ax[k], self.layers, curve_dict)
            ax[k].set_title(v[1])
        
        fig.tight_layout()
        fig.savefig(os.path.join(fig_folder, f'{self.model_structure}-comprehensive.svg'), bbox_inches='tight')    
        plt.close()
    
    
    @staticmethod
    def plot_units_pct(fig, ax, layers=None, curve_dict=None, **kwargs):
        """
            this function is the basic function to plot pct of different types of units over layers
        """
        #logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        if curve_dict is not None:
            for curve in curve_dict.keys():    
                curve = curve_dict[curve]
                ax.plot(curve['values'], color=curve['color'], linestyle=curve['linestyle'], linewidth=curve['linewidth'], label=curve['label'])
                
                if 'std' in curve.keys():
                    ax.fill_between(np.arange(len(layers)), curve['values']-curve['std'], curve['values']+curve['std'], edgecolor=None, facecolor=utils_.lighten_color(curve['color']), alpha=0.75)
            
        ax.legend(framealpha=0.5)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

        ax.set_xticks(np.arange(len(layers)), layers, rotation='vertical')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim([0, 100])
        
        
    def calculation_freq_map(self, used_unit_type=['strong_encode', 'weak_encode', 'encode', 's_all_si', 's_all_mi', 'ns_all_si', 'ns_all_mi'], **kwargs):
        """
            ...
            this should make the use of advanced Sort_dict
        """
        
        freq_path = os.path.join(self.dest_Encode, 'freq.pkl')
        
        # ---
        if os.path.exists(freq_path):
            
            freq_dict = utils_.load(freq_path)
              
        else:
            
            self.Sort_dict = self.load_Sort_dict()
            self.Encode_dict = self.load_Encode_dict()
            
            used_unit_type = used_unit_type + self.basic_types
            
            Sort_dict = self.calculation_Sort_dict(used_unit_type)
            
            # ---
            freq_layer = {}
            
            for idx, layer in tqdm(enumerate(self.layers)):     # layer
                
                sort_dict = Sort_dict[layer]
                encode_dict = self.Encode_dict[layer]
                
                # ---
                freq = {}
                for k, units in sort_dict.items():

                    encode_type = 'encode' if k in ['s_si', 's_mi', 'ns_si', 'ns_mi'] else 'weak_encode' if k in  ['s_wsi', 's_wmi', 'ns_wsi', 'ns_wmi'] else None
            
                    if units.size > 0:  
                        if encode_type:
                            id_pool = np.concatenate([encode_dict[unit][encode_type] for unit in units])
                        else:
                            id_pool = np.concatenate([[*encode_dict[unit]['encode'], *encode_dict[unit]['weak_encode']] for unit in units])
                    else:
                        id_pool = np.array([])
            
                    frequency = Counter(id_pool)
                    freq[k] = np.array([frequency[_]/self.neurons[idx]  for _ in range(50)]) 
           
                freq_layer[layer] = freq
                
            # ---
            freq_dict = {k: np.vstack([freq_layer[layer][k] for layer in self.layers]).T for k in sort_dict.keys()}
            
            utils_.dump(freq_dict, freq_path)
        
        return freq_dict
        

    def plot_Encode_freq(self, cmap='turbo', **kwargs):        # general figure for encoding frequency
        """
            ...
        """

        utils_._print('plotting Encode_freq...')
        utils_.make_dir(fig_folder:=os.path.join(self.dest_Encode, 'Figures'))
        
        # -----
        freq_dict = self.calculation_freq_map()

        # -----
        vmin = min(np.min(freq_dict[key]) for key in freq_dict)
        vmax = max(np.max(freq_dict[key]) for key in freq_dict)
        
        fig = plt.figure(figsize=(20, 30))
        
        self.plot_Encode_freq_2D(fig, freq_dict, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
        
        fig.savefig(os.path.join(fig_folder, 'layer and ID (2D).svg'), bbox_inches='tight')
        plt.close()
        
        
        # ----- raw 3D fig
        fig = plt.figure(figsize=(20, 30))
        
        self.plot_Encode_freq_3D(fig, freq_dict, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)

        fig.savefig(os.path.join(fig_folder, 'layer and ID (3D).svg'), bbox_inches='tight')
        plt.close()
        

    def plot_Encode_freq_2D(self, fig, freq_dict, vmin=0., vmax=1., cmap='turbo', **kwargs):
        
        
        def _plot_Encode_freq_2D(freq_dict, x_position, y_position, x_width=0.25, x_height=0.225, 
                                 sub_x_position=None, sub_y_position=None, sub_x_step=0.145, sub_y_step=0.115, sub_width=0.130, sub_height=0.1025,
                                title=None, vmin=0., vmax=1., cmap=None, label_on=False, sub_dict=None, **kwargs):
            
            x = 0
            y = 0
            
            ax = plt.gcf().add_axes([x_position, y_position, x_width, x_height])
            freq = freq_dict[f'{title}']
            ax.imshow(freq, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title(f'{title}')
            
            if label_on:
                ax.set_xticks(np.arange(len(self.layers)))
                ax.set_xticklabels(self.layers, rotation='vertical')
                ax.set_yticks(np.arange(0,50,5), np.arange(1,51,5))
                
            else:
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

        # --- 
        plot_info = {
            'title': ['encode', 'strong_encode', 'weak_encode', 's_non_encode', 'ns_non_encode'],
            'x_position': [0.15, 0.15, 0.15, 0.75, 0.75],
            'y_position': [0.7, 0.4, 0.15, 0.4, 0.15],
            
            'sub_x_position': [0.425, 0.425, 0.425, None, None],
            'sub_y_position': [0.7, 0.4, 0.15, None, None],
            
            'label_on': [True, False, False, False, False],
            'sub_dict': [
                        ['s_all_si', 's_all_mi', 'ns_all_si', 'ns_all_mi'],
                        ['s_si', 's_mi', 'ns_si', 'ns_mi'],
                        ['s_wsi', 's_wmi', 'ns_wsi', 'ns_wmi'],
                        None,
                        None
                        ]
        }
        
        plot_info_df = pd.DataFrame(plot_info).set_index('title')
        
        # ---
        for _ in plot_info_df.index:
            
            plot_type = plot_info_df.loc[_]
            
            _plot_Encode_freq_2D(
            
            x_position=plot_type["x_position"],
            y_position=plot_type["y_position"],
            
            sub_x_position=plot_type["sub_x_position"],
            sub_y_position=plot_type["sub_y_position"],
            
            freq_dict=freq_dict,
            
            title=_,
            
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            
            label_on=plot_type["label_on"],
            sub_dict=plot_type["sub_dict"]
            )
        
        cax = fig.add_axes([1.02, 0.3, 0.01, 0.45])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        
        fig.suptitle(f'Layer - ID [{self.model_structure}]', x=0.55, y=0.95, fontsize=28)
        
    
    def plot_Encode_freq_3D(self, fig, freq_dict, vmin=0., vmax=1., cmap='turbo', **kwargs):
        
        def _plot_Encode_freq_3D(x_position, y_position, x_width=0.25, x_height=0.225,
                                sub_x_position=None, sub_y_position=None, sub_x_step=0.13, sub_y_step=0.1125, sub_width=0.175, sub_height=0.1025,
                                freq_dict=None, title=None, vmin=0., vmax=1., cmap=None, label_on=False, sub_dict=None, **kwargs):
     
            X, Y = np.meshgrid(np.arange(len(self.layers)), np.arange(self.num_classes))

            ax = plt.gcf().add_axes([x_position, y_position, x_width, x_height], projection='3d')
            ax.plot_surface(X, Y, freq_dict[f'{title}'], vmin=vmin, vmax=vmax, cmap=cmap)

            ax.set_ylabel('IDs')
            ax.set_zlabel('Normalized responses')
            ax.set_title(f'{title}')
            ax.set_zlim([vmin, vmax])
            ax.view_init(elev=30, azim=225)
            
            if label_on == True:
                ax.set_xticks(np.arange(len(self.layers)))
                ax.set_xticklabels(self.layers, rotation='vertical')
                ax.set_yticks(np.arange(0, 50, 5), np.arange(1, 51, 5))
                
                for label in ax.get_xticklabels():
                    label.set_rotation(-50)
                for label in ax.get_yticklabels():
                    label.set_rotation(-35)
                    
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
                    
                    sub_ax.set_xticks(np.arange(len(self.layers)))
                    sub_ax.set_xticklabels([])
                    
                    sub_ax.set_yticks(np.arange(0, 50, 5), np.arange(1, 51, 5))
                    sub_ax.set_yticklabels(['' for _ in np.arange(0, 50, 5)])
                    
                    sub_ax.set_zlim(vmin, vmax)
                    sub_ax.view_init(elev=30, azim=225)
                    
                    x+=1
                    if x == 2:
                        y = 1
                        x = 0
            
            # --- interpolation
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

        
        plot_info_3D = {
            'title': ['encode', 'strong_encode', 'weak_encode', 's_non_encode', 'ns_non_encode'],
            'x_position': [0.15, 0.15, 0.15, 0.75, 0.75],
            'y_position': [0.7, 0.4, 0.15, 0.4, 0.15],
            
            'sub_x_position': [0.425, 0.425, 0.425, None, None],
            'sub_y_position': [0.7, 0.4, 0.15, None, None],
            'sub_x_step': [0.13, 0.13, 0.13, None, None],
            'sub_y_step': [0.1125, 0.1125, 0.1125, None, None],
            
            'label_on': [True, False, False, False, False],
            'sub_dict': [
                        ['s_all_si', 's_all_mi', 'ns_all_si', 'ns_all_mi'],
                        ['s_si', 's_mi', 'ns_si', 'ns_mi'],
                        ['s_wsi', 's_wmi', 'ns_wsi', 'ns_wmi'],
                        None,
                        None
                        ]
        }
        
        plot_info_df_3D = pd.DataFrame(plot_info_3D).set_index('title')

        for _ in plot_info_df_3D.index:
            
            plot_type = plot_info_df_3D.loc[_]
            
            _plot_Encode_freq_3D(
                
                x_position=plot_type["x_position"],
                y_position=plot_type["y_position"],
                
                sub_x_position=plot_type["sub_x_position"],
                sub_y_position=plot_type["sub_y_position"],
                sub_x_step=plot_type.get("sub_x_step"),
                sub_y_step=plot_type.get("sub_y_step"),
                
                freq_dict=freq_dict,
                
                title=_,
                
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                
                label_on=plot_type["label_on"],
                sub_dict=plot_type["sub_dict"]
            )
        
        cax = fig.add_axes([1.02, 0.3, 0.01, 0.45])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        fig.suptitle(f'layer - ID (3D) [{self.model_structure}]', x=0.55, y=0.95, fontsize=28)
        
    
    
            
        
    def load_Sort_dict(self, sort_dict_path=None):
        if not sort_dict_path:
            sort_dict_path = os.path.join(self.dest_Encode, 'Sort_dict.pkl')
        return utils_.load(sort_dict_path)
        
    
    def load_Encode_dict(self, encode_dict_path=None):
        if not encode_dict_path:
            encode_dict_path = os.path.join(self.dest_Encode, 'Encode_dict.pkl')
        return utils_.load(encode_dict_path)
    
    
    # --- legacy
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
        

# ----------------------------------------------------------------------------------------------------------------------
def _unit_types(used_unit_types):
    """
        add any combination here if wanted
        
        see historical version for tested combinations
    """
    
    k_d = {
        'qualified': ['s_si', 's_wsi', 's_mi', 's_wmi', 's_non_encode', 'ns_si', 'ns_wsi', 'ns_mi', 'ns_wmi', 'ns_non_encode'],
        
        'selective': ['s_si', 's_wsi', 's_mi', 's_wmi'],
        'strong_selective': ['s_si','s_mi'],
        'weak_selective': ['s_wsi', 's_wmi'],
        'non_selective': ['s_non_encode', 'ns_si', 'ns_wsi', 'ns_mi', 'ns_wmi', 'ns_non_encode'],
        
        'sensitive': ['s_si', 's_wsi', 's_mi', 's_wmi', 's_non_encode'],
        'non_sensitive': ['ns_si', 'ns_wsi', 'ns_mi', 'ns_wmi', 'ns_non_encode'],
        
        'encode': ['s_si', 'ns_si', 's_wsi', 'ns_wsi', 's_mi', 'ns_mi', 's_wmi', 'ns_wmi'],
        'strong_encode': ['s_si', 'ns_si', 's_mi', 'ns_mi'],
        'weak_encode': ['s_wsi', 'ns_wsi', 's_wmi', 'ns_wmi'],
        'non_encode': ['s_non_encode', 'ns_non_encode'],
        
        # --- legacy design
        'si': ['s_si', 'ns_si'],
        'mi': ['s_mi', 'ns_mi'],
        'wsi': ['s_wsi', 'ns_wsi'],
        'wmi': ['s_wmi', 'ns_wmi'],
        
        'ns_strong_encode': ['ns_si', 'ns_mi'],
        'ns_weak_encode': ['ns_wsi', 'ns_wmi'],
        'ns_encode': ['ns_si', 'ns_wsi', 'ns_mi', 'ns_wmi'],
        
        's_strong_encode': ['s_si', 's_mi'],
        's_weak_encode': ['s_wsi', 's_wmi'],
        's_encode': ['s_si', 's_wsi', 's_mi', 's_wmi'],
        
        's_all_si': ['s_si', 's_wsi'],
        's_all_mi': ['s_mi', 's_wmi'],
        'ns_all_si': ['ns_si', 'ns_wsi'],
        'ns_all_mi': ['ns_mi', 'ns_wmi'],
        # ...
        
    }
    
    for _ in used_unit_types:
        assert _ in k_d.keys(), f'please assign cell_type: [{_}]'
    
    return {_: k_d[_] for _ in used_unit_types}


def seal_plot_config(values=None, point=None, color=None, linestyle=None, linewidth=None, label=None):

    return {
        'values': values,
        'point': point,
        'color': color,
        'linestyle': linestyle,
        'linewidth': linewidth,
        'label': label
        }


# ----------------------------------------------------------------------------------------------------------------------
class FSA_Responses(FSA_Encode):
    """
        ...
        
        Main functions:
            1) calculate and display responses of a single unit and multiple units
            2) calculate and display **Percentage** of units of a single layer
        
    """
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.dest_Responses = os.path.join(self.dest_Encode, 'Responses')
        utils_.make_dir(self.dest_Responses)
        
        self.Sort_dict = self.load_Sort_dict()
        
        ...
    
    
    def plot_unit_responses(self, random_select_units=10, start_layer_idx=-5, cmap='jet', **kwargs):
        """
            ...
            
            threshold: r'$V_{th}=\bar{x}+2\sqrt{\frac{1}{500}\sum(x_i-\bar{x})^2}$'
            ref: r'$ref=\bar{x}+2\sqrt{\frac{1}{50}\sum(x_i-\bar{x_i})^2}, \bar{x_i} = \frac{1}{10}\sum{x_i}$'
            
        """
        # --- init ---
        utils_.make_dir(fig_folder:=os.path.join(self.dest_Responses, 'Single Units'))

        colors = [plt.get_cmap(cmap, 50)(i) for i in range(50)]
        
        # ---
        def _plot_unit_responses_layer(unit_idx, input, **kwargs):

            # -----
            plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})
            
            local_means, global_mean, threshold, ref = calculation_unit_responses(input, **kwargs)
            
            # -----
            fig, ax = plt.subplots(figsize=(6, 6))
            
            plot_unit_responses(ax, input, local_means, colors=colors, vmin=vmin, vmax=vmax, **kwargs)

            # ---
            ax.hlines(0., 0, 49, colors='gray', linestyle='-')
            ax.hlines(threshold, 0, 49, colors='red', linestyle='--', label='threshold')
            ax.hlines(ref, 0, 49, colors='teal', linestyle='--', label='ref')
            #ax.set_ylim([vmin, vmax])
            
            # ---
            handles, labels = ax.get_legend_handles_labels()
            
            mean = Line2D([0], [0], marker='d', markersize=5, markeredgecolor='none', color='gray', linestyle='--', linewidth=1)
            median = Line2D([0], [0], marker='_', markersize=5, color='orange', linewidth=0)
            outlier = Line2D([0], [0], marker='+', markersize=5, markeredgecolor='gray', linewidth=0)

            handles.extend([mean, median, outlier])
            labels.extend(['mean', 'median', 'outlier'])
            
            ax.legend(handles, labels, framealpha=0.75)
            
            # ---
            fig.suptitle(f'[{layer}] unit: {unit_idx}', y=0.975)
            fig.tight_layout()
            
            fig.savefig(os.path.join(type_layer_fig_folder, f'{cell_type}_{unit_idx}.svg'))
            plt.close()
        
        # ---
        for layer in self.layers[start_layer_idx:]:
             
            utils_.make_dir(layer_fig_folder:=os.path.join(fig_folder, f'{layer}'))
            feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), normalize=True, sort=True, **kwargs)
            
            vmin = np.min(feature)
            vmax = np.max(feature)
            
            for cell_type, v  in tqdm(self.Sort_dict[layer].items(), desc=f'{layer}'):     # for each type
                
                utils_.make_dir(type_layer_fig_folder:=os.path.join(layer_fig_folder, f'{cell_type}'))
            
                if v.size != 0:
 
                    if v.size > random_select_units:
                        
                        v = np.random.choice(v, random_select_units)

                    Parallel(n_jobs=random_select_units)(delayed(_plot_unit_responses_layer)(unit_idx, feature[:, unit_idx], **kwargs) for unit_idx in v)  
                    

    def plot_stacked_responses(self, num_types=5, start_layer_idx=-5, used_unit_types=['s_si', 's_wsi', 's_mi', 's_wmi', 'non_selective'], **kwargs):
        """
            this function is memory consuming
        """
                
        assert start_layer_idx < 0, f'[Coderror] start_layer_idx {start_layer_idx} must be negative in current design'

        utils_._print(f'Executing plot_stacked_responses... | num_types: {num_types} | num_layers: {np.abs(start_layer_idx)}')
        
        # ---
        utils_.make_dir(fig_folder:=os.path.join(self.dest_Responses, 'Stacked Responses'))
        utils_.make_dir(type_fig_folder:=os.path.join(fig_folder, str(num_types)))
        
        # ---
        self.Sort_dict = self.load_Sort_dict()
        
        if num_types == 5:
            Sort_dict = self.calculation_Sort_dict(used_unit_types)
        else:
            Sort_dict = self.Sort_dict
        
        # ---
        figsize = (26, 6) if num_types == 5 else (26, 10)
        gs_rows, gs_cols = (1, 5) if num_types == 5 else (2, 5)
        

        for layer in self.layers[start_layer_idx:]:
            fig, ax = plt.subplots(figsize=figsize)
            gs_main = gridspec.GridSpec(gs_rows, gs_cols, figure=fig)
        
            feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), normalize=True, sort=True, **kwargs)
            plot_stacked_responses(fig, gs_main, layer, Sort_dict[layer], feature)
        
            ax.axis('off')
            ax.plot([], [], color='blue', label='mean')
            ax.plot([], [], color='teal', linestyle='--', label='ref')
            ax.plot([], [], color='red', linestyle='--', label='threshold')
        
            fig.suptitle(f'{layer} [{self.model_structure}]', y=0.97, fontsize=20)
            fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, bbox_transform=plt.gcf().transFigure)
        
            fig.tight_layout()
            fig.savefig(os.path.join(type_fig_folder, f'{layer} num_types {num_types}.png'), bbox_inches='tight')
            plt.close()

            #mem = psutil.virtual_memory()
            #swap = psutil.swap_memory()
            #print(f'\nCPU+SWAP used: {(mem.used+swap.used)/1024/1024/1024:.3f} / 256')
            #print(f'\nCPU used pct: {mem.percent} %')
            #print(sys.getrefcount(self.plot_stacked_responses_single_layer))
        

    def plot_responses_PDF(self, used_unit_types:list[str]=['qualified', 'strong_selective', 'weak_selective', 'non_selective'], start_layer_idx=-5, **kwargs):
        """
            ...
            
            this funtion is equivalent to the PDF plot of self.plot_stacked_responses() but use the function from bio data
            process with more details
        """
        
        utils_.make_dir(fig_folder:=os.path.join(self.dest_Responses, 'Responses PDF'))
        
        Sort_dict = self.calculation_Sort_dict(used_unit_types)
        
        for unit_type in used_unit_types:

            utils_.make_dir(save_path:=os.path.join(fig_folder, unit_type))

            for layer in tqdm(self.layers[start_layer_idx:], desc=f'{unit_type} PDF'):
    
                #warnings.simplefilter(action='ignore')
                #logging.getLogger('matplotlib').setLevel(logging.ERROR)
                
                feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), normalize=True, sort=True, verbose=False, **kwargs)[:, Sort_dict[layer][unit_type]]
                
                fig = Human_Neuron_Records_Process.plot_FR_PDF(self.model_structure, 'unit', feature, layer=layer, unit_type=unit_type, **kwargs)
                
                fig.tight_layout()
                fig.savefig(os.path.join(save_path, f'{layer}.svg'))
                
                plt.close()
        

    def plot_pct_pie_chart(self, used_unit_types=['s_si', 's_wsi', 's_mi', 's_wmi', 'non_selective'], start_layer_idx=-5, **kwargs):
        
        utils_.make_dir(save_path:=os.path.join(self.dest_Encode, 'Pie Chart'))
        
        Sort_dict = self.calculation_Sort_dict(used_unit_types)
        
        for layer in tqdm(self.layers[start_layer_idx:], desc='Pir Chart'):
            
            sort_dict = Sort_dict[layer]
            pcts = [_.size/self.neurons[self.layers.index(layer)]*100 for idx, _ in enumerate(sort_dict.values())]
            
            # ---
            labels = [f'{_}' for idx, _ in enumerate(used_unit_types)]
            colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']
            explode = [0.5 * (1. - (value - min(pcts)) / (max(pcts) - min(pcts))) for value in pcts]
            
            # ---
            fig, ax = plt.subplots(figsize=(10,6))
            utils_.plot_pie_chart(fig, ax, pcts, labels, title=f'{self.neurons[self.layers.index(layer)]} Units', colors=colors, explode=explode, **kwargs)
            fig.savefig(os.path.join(save_path, f'{layer}_pct_pie_chart.svg'), transparent=True)
            plt.close()
        


def plot_unit_responses(ax, input, local_means, colors=None, num_classes=50, num_samples=10, **kwargs):
    """
        ...
    """
 
    # ---
    boxes = ax.boxplot(list(input.reshape(num_classes, num_samples)), patch_artist=True, sym='+', positions=np.arange(50))
    
    for i, _ in enumerate(boxes['boxes']):
        _.set(color=colors[i], alpha=0.5)
        
    for i, _ in enumerate(boxes['fliers']):
        _.set(marker='+', markerfacecolor=colors[i], markeredgecolor=colors[i], markersize=5, alpha=0.75)
    
    # ---
    ax.scatter(np.repeat(np.arange(num_classes), num_samples), input, color=np.repeat(np.array(colors), num_samples, axis=0), s=8, alpha=0.5, edgecolor='none')
    ax.scatter(np.arange(num_classes), local_means, color=colors, marker='d', s=12)
    
    for _ in range(num_classes):
        ax.vlines(_, np.min(input), local_means[_], linestyle='--', alpha=0.75)
    
    ax.set_xticks(ticks:=[0, 9, 19, 29, 39, 49])
    ax.set_xticklabels(ticks)

         
# ----------------------------------------------------------------------------------------------------------------------
def plot_stacked_responses(fig, gs_main, layer, sort_dict, feature, num_classes=50, num_samples=10, scaling_factor=0.1, **kwargs):
    """
        ...
        
        manually appointed fig size
        
    """
    
    def _plot_stacked_responses_single(input, vmin=0., vmax=1., color=None, linestyle=None, scaling_factor=0.1, **kwargs):
        
        if np.std(input) != 0:

            v_radius = vmax - vmin
            
            dummy_x = np.linspace(vmin-scaling_factor*v_radius, vmax+scaling_factor*v_radius, 101)
            dummy_y = gaussian_kde(input)(dummy_x)
            
            ax_right.plot(dummy_y, dummy_x, linestyle=linestyle, color=color)
            
            if len(y_peak:=np.where(dummy_y==np.max(dummy_y))[0]) == 1:
                x_vals_max = dummy_x[y_peak.item()]
            else:
                x_vals_max = dummy_x[y_peak[0]]
            
            ax_left.hlines(x_vals_max, 0, 50, colors=color, alpha=0.75, linestyle='--')
            ax_right.hlines(x_vals_max, np.min(dummy_y), np.max(dummy_y), colors=color, alpha=0.75, linestyle='--')
        
 
    # ---
    colors = [plt.get_cmap('jet', 50)(i) for i in range(50)]

    tqdm_bar = tqdm(total=len(sort_dict.keys()), desc=f'{layer}')

    y_min = np.min(feature)
    y_max = np.percentile(feature, 99.) if np.max(feature) > 1. else 1.

    y_lin_range = y_max - y_min
    
    for i in range(num_rows:=gs_main.nrows):
        for j in range(num_cols:=gs_main.ncols):
            
            unit_type = list(sort_dict.keys())[i*num_cols+j]
            
            gs_sub = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[4, 1], subplot_spec=gs_main[i, j])

            ax_left = fig.add_subplot(gs_sub[0])
            ax_right = fig.add_subplot(gs_sub[1])
            ax_right.set_yticks([])
            
            if (i+j) != 0:
                ax_left.set_xticks([])
                ax_left.set_yticks([])

            if (num_units:=sort_dict[unit_type].size) == 0:
                ax_left.set_title(f'{unit_type} [0.00%]')
                ax_right.set_title('')

            else:
                
                local_means, global_mean, threshold, ref = [np.array(_) for _ in zip(*[calculation_unit_responses(_, **kwargs) for _ in feature[:, sort_dict[unit_type]].T])]

                # ---
                ax_left.scatter(np.tile(np.arange(num_classes), num_units), local_means.reshape(-1), color=np.tile(np.array(colors), [num_units, 1]), alpha=0.1, marker='.', s=1)
                ax_left.set_title(f'{unit_type} [{num_units/feature.shape[1]*100:.2f}%]')
                
                # ---
                _plot_stacked_responses_single(local_means.reshape(-1), vmin=y_min, vmax=y_max, color='blue', **kwargs)
                _plot_stacked_responses_single(threshold, vmin=y_min, vmax=y_max, color='red', linestyle='dotted', **kwargs)
                _plot_stacked_responses_single(ref, vmin=y_min, vmax=y_max, color='teal', linestyle='dotted', **kwargs)
                
                # ---
                ax_left.set_ylim([y_min-y_lin_range*scaling_factor, y_max+y_lin_range*scaling_factor])
                ax_right.set_ylim([y_min-y_lin_range*scaling_factor, y_max+y_lin_range*scaling_factor])
                ax_right.set_title('PDF')
                
            tqdm_bar.update(1)


def calculation_Encode(input, **kwargs):
    """
        ...
    """
    
    local_means, global_mean, threshold, ref = calculation_unit_responses(input, **kwargs)
    
    encode = np.where(local_means>threshold)[0]     # '>' prevent all 0
    weak_encode = np.setdiff1d(np.where(local_means>ref)[0], encode)
    
    return {'encode': encode, 'weak_encode': weak_encode}


def calculation_unit_responses(input, num_classes=50, num_samples=10, n=2, **kwargs):

    global_mean = np.mean(input)    
    local_means = np.mean(input.reshape(num_classes, num_samples), axis=1)

    threshold = global_mean + n*np.std(input)     # total variance
    ref = global_mean + n*np.std(local_means)     # between-group variance

    return local_means, global_mean, threshold, ref


# ----------------------------------------------------------------------------------------------------------------------
class FSA_SVM(FSA_Encode):
    """
        by default, the SVM kernel is RBF
    """
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.Sort_dict = self.load_Sort_dict()
        
        self.dest_SVM = os.path.join(self.dest_Encode, 'SVM')
        utils_.make_dir(self.dest_SVM)
        
        ...
        
    
    def process_SVM(self, used_unit_types=['qualified', 'strong_selective', 'weak_selective', 'non_selective'], **kwargs):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        # ----- calculation
        layer_SVM = self.calculation_SVM(used_unit_types)
        
        # ----- plot
        fig, ax = plt.subplots(figsize=(10,6))
        
        self.plot_SVM(ax, layer_SVM, **kwargs)
        
        ax.set_title(title:=f'SVM | {self.model_structure}')
        fig.savefig(os.path.join(self.dest_SVM, f'{title}.svg'), bbox_inches='tight')
        plt.close()
    

    def calculation_SVM(self, used_unit_types=['s_si', 's_wsi', 's_mi', 's_wmi', 'non_selective'], **kwargs):
        """
            ...
        """
        
        utils_._print('computing SVM...')

        if os.path.exists(SVM_path:=os.path.join(self.dest_SVM, f'SVM {used_unit_types}.pkl')):
            
            layer_SVM = utils_.load(SVM_path)
            
        else:
            
            # --- init
            Sort_dict = self.calculation_Sort_dict(used_unit_types)
            
            layer_SVM = {}
            
            for layer in self.layers:
                
                # --- depends
                feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), normalize=True, sort=True, **kwargs)

                tqdm_bar = tqdm(total=len(used_unit_types), desc=f'SVM {layer}')
                
                # ---
                layer_SVM[layer] = {k: calculation_SVM(feature[:, v], np.repeat(np.arange(50), 10), tqdm_bar) for k,v in Sort_dict[layer].items()}
            
            layer_SVM = {'acc': {_: np.array([v[_] for k,v in layer_SVM.items()]) for _ in used_unit_types}}
            
            utils_.dump(layer_SVM, SVM_path, verbose=False)

        return layer_SVM
            
    
    def plot_SVM(self, ax, layer_SVM, **kwargs):
        """
            ...
        """
        
        # --- init
        SVM_type_conifg = self.plot_Encode_config

        # --- all
        for k,v in layer_SVM['acc'].items():
            
            plot_config = SVM_type_conifg.loc[k]

            ax.plot(layer_SVM['acc'][k], color=plot_config['color'], linestyle=plot_config['linestyle'], label=k)
            
            if 'std' in layer_SVM:
                ax.fill_between(np.arange(len(layers)), layer_SVM['acc'][k]-layer_SVM['std'][k], layer_SVM['acc'][k]+layer_SVM['std'][k], 
                                edgecolor=None, facecolor=utils_.lighten_color(plot_config['color']), alpha=0.75, **kwargs)

        # -----
        ax.set_xticks(np.arange(len(layers)))
        ax.set_xticklabels(layers, rotation='vertical')
        
        ax.set_ylim([0, 100])
        ax.set_yticks(np.arange(1, 101))
        ax.set_yticklabels(['' if (_%10)!=0 else _ for _ in range(1,101)])
        
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        ax.legend(ncol=2, framealpha=0.5)
        

def calculation_SVM(input, label, tqdm_bar, **kwargs):
   
    tqdm_bar.update(1)
    
    return utils_.SVM_classification(input, label, test_size=0.2, random_state=42, **kwargs) if input.size != 0 else 0.


# ======================================================================================================================
if __name__ == "__main__":

    layers, neurons, shapes = utils_.get_layers_and_units('vgg16', target_layers='act')

    root_dir = '/home/acxyle-workstation/Downloads'

    #selectivity_analyzer = FSA_Encode(root=os.path.join(root_dir, 'Face Identity Baseline'), layers=layers, neurons=neurons)
    #selectivity_analyzer.calculation_Encode()
    #selectivity_analyzer.plot_Encode_pct_single()
    #selectivity_analyzer.plot_Encode_pct_comprehensive()
    #selectivity_analyzer.plot_Encode_freq()


    #selectivity_analyzer = FSA_Responses(root=os.path.join(root_dir, 'Face Identity Baseline'), layers=layers, neurons=neurons)
    #selectivity_analyzer.plot_unit_responses()
    #selectivity_analyzer.plot_stacked_responses()
    #selectivity_analyzer.plot_responses_PDF()
    #selectivity_analyzer.plot_pct_pie_chart()
    
    
    selectivity_analyzer = FSA_SVM(root=os.path.join(root_dir, 'Face Identity Baseline'), layers=layers, neurons=neurons)
    selectivity_analyzer.process_SVM()
    
