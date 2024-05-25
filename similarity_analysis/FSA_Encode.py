#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:54:55 2023

@author: Jinge Wang, Runnan Cao

    refer to: https://github.com/JingeW/ID_selective
              https://osf.io/824s7/
    
@modified: acxyle
    
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from collections import Counter

from scipy.stats import gaussian_kde
from matplotlib import gridspec
import pandas as pd


import sys
sys.path.append('../')
import utils_

from bio_records_process.human_feature_process import human_feature_process

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
        
        self.root = os.path.join(root, 'Features')     # <- folder for feature maps, which should be generated before analyhss
        self.dest = os.path.join(root, 'Analysis')
        utils_.make_dir(self.dest)
        
        self.dest_Encode = os.path.join(self.dest, 'Encode')
        utils_.make_dir(self.dest_Encode)
        
        self.layers = layers
        self.neurons = neurons
        
        self.num_classes = num_classes
        self.num_samples = num_samples

        self.model_structure = root.split('/')[-1].split(' ')[-1]
     
        #self.bahsc_types = ['a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne', 
        #                    'na_hs', 'na_ls', 'na_hm', 'na_lm', 'na_ne']   
     
        
    # FIXME ---
    def calculation_Encode(self, normalize=True, sort=True, num_workers=-1, **kwargs):
        """
            ...
        """
        
        utils_.formatted_print('Executing calculation_Encode...')
        
        sort_dict_path = os.path.join(self.dest_Encode, 'Sort_dict.pkl')
        encode_dict_path = os.path.join(self.dest_Encode, 'Encode_dict.pkl')

        if (not hasattr(self, 'Sort_dict') and os.path.exists(sort_dict_path)):
            
            self.Sort_dict = self.load_Sort_dict()
        
        else:
            
            # ----- init
            self.Encode_dict = {}
            self.Sort_dict = {}
            
            self.ANOVA_idces = utils_.load(os.path.join(self.dest, 'ANOVA/ANOVA_idces.pkl'), verbose=False) 
            
            # --- running
            for layer in self.layers:     # for each layer
                
                feature = utils_.load_feature(os.path.join(self.root, f'{layer}.pkl'), normalize=normalize, sort=sort, verbose=False, **kwargs)      # load feature matrix
                
                # ----- 1. ANOVA
                a = self.ANOVA_idces[layer]     # anova_idx
                na = np.setdiff1d(np.arange(feature.shape[1]), a)     # non_anova_idx
                
                # ----- 2. Encode
                pl = Parallel(n_jobs=num_workers)(delayed(calculation_Encode)(feature[:, i]) for i in tqdm(range(feature.shape[1]), desc=f'[{layer}] Encode'))  
    
                unit_encode_dict = {i: pl[i] for i in range(len(pl))}    
                
                self.Encode_dict[layer] = unit_encode_dict    
                
                # ----- 2. encode test
                hs = []
                ls = []
                hm = []
                lm = []
                non_encode = []
                
                for k, v in unit_encode_dict.items():
                    if len(v['encode']) == 1:
                        hs.append(k)
                    elif len(v['encode']) == 0 and len(v['weak_encode']) == 1:
                        ls.append(k)
                    elif len(v['encode']) > 1:
                        hm.append(k)
                    elif len(v['encode']) == 0 and len(v['weak_encode']) > 1:
                        lm.append(k)
                    elif len(v['encode']) == 0 and len(v['weak_encode']) == 0:
                        non_encode.append(k)
                
                # ----- 3. bahsc types
                unit_sort_dict = {
                    
                    'a_hs': np.intersect1d(a, hs),
                    'a_ls': np.intersect1d(a, ls),
                    'a_hm': np.intersect1d(a, hm),
                    'a_lm': np.intersect1d(a, lm),
                    'a_ne': np.intersect1d(a, non_encode),
                    
                    'na_hs': np.intersect1d(na, hs),
                    'na_ls': np.intersect1d(na, ls),
                    'na_hm': np.intersect1d(na, hm),
                    'na_lm': np.intersect1d(na, lm),
                    'na_ne': np.intersect1d(na, non_encode),
                    }
                
                self.Sort_dict[layer] = unit_sort_dict
                
            utils_.dump(self.Sort_dict, sort_dict_path, verbose=True)
            utils_.dump(self.Encode_dict, encode_dict_path, verbose=True)  
            
    
    def calculation_Sort_dict(self, used_unit_types, **kwargs):
        
        self.bahsc_types = ['a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne', 
                            'na_hs', 'na_ls', 'na_hm', 'na_lm', 'na_ne']   
            
        bahsc_types = [_ for _ in used_unit_types if _ in self.bahsc_types]
        advanced_types = [_ for _ in used_unit_types if _ not in bahsc_types]
        
        advanced_Sort_dict = self.calculation_Sort_dict_advanced(advanced_types, **kwargs)
        
        Sort_dict = {k: {_:v[_] for _ in bahsc_types} for k,v in self.Sort_dict.items()}
        Sort_dict = {layer: {k: advanced_Sort_dict[layer][k] if k in advanced_Sort_dict[layer] else Sort_dict[layer][k] for k in used_unit_types} for layer in self.layers}

        return {layer: {type_: v.astype(int) for type_, v in dict_.items()} for layer, dict_ in Sort_dict.items()}
        
    
    def calculation_Sort_dict_advanced(self, used_unit_types:list[str], **kwargs):
        """
            ...     
        """
        
        advanced_type = _unit_types(used_unit_types)
        advanced_Sort_dict = {layer: {_: np.concatenate([self.Sort_dict[layer][__] for __ in advanced_type[_]]) for _ in used_unit_types} for layer in self.Sort_dict.keys()}
        
        return advanced_Sort_dict
    
    
    # FIXME --- 
    def calculation_units_pct(self, used_unit_types=['a_hs', 'a_ls', 'a_hm', 'a_lm', 'non_selective'], **kwargs):
        """
            ...
        """
        # ---
        Sort_dict = self.calculation_Sort_dict(used_unit_types)
        
        unit_pct = {_: np.array([len(Sort_dict[layer][_])/self.neurons[idx]*100 for idx, layer in enumerate(self.layers)]) for _ in used_unit_types}
        
        return unit_pct
    
    
    def calculation_curve_dict(self, units_pct, Encode_path=None, **kwargs):
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
                            'a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne',
                            'na_hs', 'na_ls', 'na_hm', 'na_lm', 'na_ne',
                            
                            'a_s', 'a_m',

                            'hs', 'ls', 'hm', 'lm', 'non_encode',
                            'anova', 'non_anova', 'high_encode', 'weak_encode', 'encode', 
                            
                            'high_selective', 'low_selective', 'selective', 'non_selective',
                            'a_h_encode', 'a_l_encode', 'a_encode',
                            'na_h_encode', 'na_l_encode', 'na_encode',
                                ],
                        
                        'label': [
                            'qualified(all)',
                            'a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne',
                            'na_hs', 'na_ls', 'na_hm', 'na_lm', 'na_ne',

                            'a_s', 'a_m',

                            'hs', 'ls', 'hm', 'lm', 'non_encode',
                            'anova', 'non_anova', 'high_encode', 'weak_encode', 'encode', 
                            
                            'high_selective', 'low_selective', 'selective', 'non_selective',
                            'a_h_encode', 'a_l_encode', 'a_encode',
                            'na_h_encode', 'na_l_encode', 'na_encode',
                                ],
                        
                        'color': [
                            '#000000',
                            '#0000FF', '#00BFFF', '#FF4500', '#FFA07A', '#008000',
                            '#00008B', '#87CEEB', '#CD5C5C', '#FA8072', '#696969',

                            '#0000FF', '#FF4500', 

                            '#0000CD', '#ADD8E6', '#FF6347', '#FFDAB9', '#808080',
                            '#FF0000', '#707000', '#800080', '#FFC0CB', '#8A2BE2', 
                            
                            '#FFD700', '#FFA500', '#FF8C00', '#999999',
                            '#9400D3', '#FF69B4', '#8A2BE2', 
                            '#4B0082', '#C71585', '#7B68EE',
                                ],
                        
                        'linestyle': [
                            None,
                            '-', '-', '-', '-', '-',
                            'dotted', 'dotted', 'dotted', 'dotted', 'dotted',
                            
                            '--', (0, (3, 1, 1, 1,)),
                            
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

                            3.0, 3.0,

                            3.0, 3.0, 3.0, 3.0, 3.0,
                            3.0, 3.0, 3.0, 3.0, 3.0, 

                            2.0, 2.0, 2.0, 2.0, 
                            3.5, 3.5, 3.5,
                            3.5, 3.5, 3.5,
                                ]
                    }
        
        style_config_df = pd.DataFrame(style_config).set_index('type')
        
        return style_config_df
    
    
    def plot_Encode_pct(self, used_unit_types=['a_hs', 'a_ls', 'a_hm', 'a_lm', 'non_selective'], **kwargs):
        """
            ...
        """

        # --- init
        fig_folder = os.path.join(self.dest_Encode, 'Figures')
        utils_.make_dir(fig_folder)
        
        self.Sort_dict = self.load_Sort_dict()

        units_pct = self.calculation_units_pct(used_unit_types, **kwargs)
        curve_dict = self.calculation_curve_dict(units_pct, **kwargs)
        
        # --- plot
        fig, ax = plt.subplots(figsize=(10,6))
        
        self.plot_units_pct(fig, ax, self.layers, curve_dict, **kwargs)
        
        ax.set_title(title:=self.model_structure)
        
        fig.savefig(os.path.join(fig_folder, f'{title}-{used_unit_types}.svg'), bbox_inches='tight')    
        plt.close()
        
    
    def plot_Encode_pct_bar_chart(self, units_pct=None, used_unit_types=None, **kwargs):

        utils_.make_dir(fig_folder:=os.path.join(self.dest_Encode, 'Figures'))
        style_config_df = self.plot_Encode_config
        
        if used_unit_types is None:
            #used_unit_types = ['anova', 'non_anova', 'selective', 'high_selective', 'low_selective', 'non_selective']
            used_unit_types = ['a_hs', 'a_ls', 'a_hm', 'a_lm', 'non_selective']

        # --- init
        if units_pct is None:
            self.Sort_dict = self.load_Sort_dict()
            units_pct = self.calculation_units_pct(used_unit_types, **kwargs)

        #x = np.arange(len(self.layers))
        #fig, ax = plt.subplots(figsize=(len(self.layers),6))
        #ax.bar(x, units_pct['non_anova'], color='green', label='N-Sen')
        #ax.bar(x, units_pct['anova'], bottom=units_pct['non_anova'], alpha=0.5, color='red', label='Senhstive')
        #ax.bar(x+[0.25]*len(self.layers), units_pct['high_selective'], bottom=units_pct['non_selective']+units_pct['low_selective'], width=0.3, color='yellow', label='Selective')
        #ax.bar(x+[-0.25]*len(self.layers), units_pct['non_selective'], color='gray', width=0.3, label='N-Sel')
        
        x = np.arange(len(self.layers))
        bottoms = np.zeros(len(self.layers))
        fig, ax = plt.subplots(figsize=(len(self.layers), 6))

        for _ in used_unit_types:
            
            config = style_config_df.loc[_]
            color = config['color']
            label = config['label']
            
            ax.bar(x, units_pct[_], bottom=bottoms, alpha=0.5, color=color, label=label)
            bottoms += units_pct[_]
        
        ax.legend(ncol=6, bbox_to_anchor=(1, -0.05))
        ax.set_title(f'Pcts of subsets@{self.model_structure}')

        fig.savefig(os.path.join(fig_folder, f'{self.model_structure} pcts of subsets {used_unit_types}.svg'), bbox_inches='tight')    
        plt.close()
    
    
    def plot_Encode_pct_comprehenhsve(self, **kwargs):
        """
            ...
        """
        # --- init
        fig_folder = os.path.join(self.dest_Encode, 'Figures')
        utils_.make_dir(fig_folder)
        
        self.Sort_dict = self.load_Sort_dict()
        
        # --- plot, comprehenhsve
        plot_dict = {
            (0,0): (['non_encode', 'hs', 'hm', 'encode', 'ls', 'lm', 'weak_encode'], 'encode v.s. non_encode'),
            (0,1): (['high_selective', 'a_hs', 'a_hm', 'low_selective', 'a_ls', 'a_lm', 'non_selective', 'anova'], 'anova'),
            (1,0): (['na_encode', 'na_hs', 'na_hm', 'na_l_encode', 'na_ls', 'na_lm', 'na_ne', 'non_anova'], 'non_anova'),
            (1,1): (['anova', 'non_anova', 'encode', 'weak_encode', 'non_encode'], 'anova and encode')
            }
        
        fig, ax = plt.subplots(2,2,figsize=(18,10))

        for k,v in plot_dict.items():
            
            units_pct = self.calculation_units_pct(v[0], **kwargs)
            curve_dict = self.calculation_curve_dict(units_pct, **kwargs)
            
            self.plot_units_pct(fig, ax[k], self.layers, curve_dict)
            ax[k].set_title(v[1])
        
        fig.tight_layout()
        fig.savefig(os.path.join(fig_folder, f'{self.model_structure}-comprehenhsve.svg'), bbox_inches='tight')    
        plt.close()
    
    
    @staticmethod
    def plot_units_pct(fig, ax, layers=None, curve_dict=None, color=None, label=None, text=True, **kwargs):
        """
            this function is the bahsc function to plot pct of different types of units over layers, manually change
        """
        #logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        if curve_dict is not None:
            for curve in curve_dict.keys():    
                curve = curve_dict[curve]
                
                if color is not None and label is not None:
                    curve['color'], curve['label'] = color, label

                ax.plot(curve['values'], color=curve['color'], linestyle=curve['linestyle'], linewidth=curve['linewidth'], label=curve['label'])
                    
                if 'stds' in curve.keys():
                    ax.fill_between(np.arange(len(layers)), curve['values']-curve['stds'], curve['values']+curve['stds'], edgecolor=None, facecolor=utils_.lighten_color(utils_.color_to_hex(curve['color']), 40), alpha=0.75)
            
        ax.legend(framealpha=0.5)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
            
        if text:
            ax.set_xticks(np.arange(len(layers)), layers, rotation='vertical')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim([0, 100])
        
        
    def calculation_freq_map(self, used_unit_type=['high_encode', 'weak_encode', 'encode', 'a_s', 'a_m', 'na_s', 'na_m'], **kwargs):
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
            
            used_unit_type = used_unit_type + self.bahsc_types
            
            Sort_dict = self.calculation_Sort_dict(used_unit_type)
            
            # ---
            freq_layer = {}
            
            for idx, layer in tqdm(enumerate(self.layers)):     # layer
                
                sort_dict = Sort_dict[layer]
                encode_dict = self.Encode_dict[layer]
                
                # ---
                freq = {}
                for k, units in sort_dict.items():

                    encode_type = 'encode' if k in ['a_hs', 'a_hm', 'na_hs', 'na_hm'] else 'weak_encode' if k in  ['a_ls', 'a_lm', 'na_ls', 'na_lm'] else None
            
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

        utils_.formatted_print('plotting Encode_freq...')
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
        
        
        def _plot_Encode_freq_2D(freq_dict, x_pohstion, y_pohstion, x_width=0.25, x_height=0.225, 
                                 sub_x_pohstion=None, sub_y_pohstion=None, sub_x_step=0.145, sub_y_step=0.115, sub_width=0.130, sub_height=0.1025,
                                title=None, vmin=0., vmax=1., cmap=None, label_on=False, sub_dict=None, **kwargs):
            
            x = 0
            y = 0
            
            ax = plt.gcf().add_axes([x_pohstion, y_pohstion, x_width, x_height])
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
                    sub_ax = plt.gcf().add_axes([sub_x_pohstion + sub_x_step*x, sub_y_pohstion + sub_y_step*y, sub_width, sub_height])
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
            'title': ['encode', 'high_encode', 'weak_encode', 'a_ne', 'na_ne'],
            'x_pohstion': [0.15, 0.15, 0.15, 0.75, 0.75],
            'y_pohstion': [0.7, 0.4, 0.15, 0.4, 0.15],
            
            'sub_x_pohstion': [0.425, 0.425, 0.425, None, None],
            'sub_y_pohstion': [0.7, 0.4, 0.15, None, None],
            
            'label_on': [True, False, False, False, False],
            'sub_dict': [
                        ['a_s', 'a_m', 'na_s', 'na_m'],
                        ['a_hs', 'a_hm', 'na_hs', 'na_hm'],
                        ['a_ls', 'a_lm', 'na_ls', 'na_lm'],
                        None,
                        None
                        ]
        }
        
        plot_info_df = pd.DataFrame(plot_info).set_index('title')
        
        # ---
        for _ in plot_info_df.index:
            
            plot_type = plot_info_df.loc[_]
            
            _plot_Encode_freq_2D(
            
            x_pohstion=plot_type["x_pohstion"],
            y_pohstion=plot_type["y_pohstion"],
            
            sub_x_pohstion=plot_type["sub_x_pohstion"],
            sub_y_pohstion=plot_type["sub_y_pohstion"],
            
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
        
        def _plot_Encode_freq_3D(x_pohstion, y_pohstion, x_width=0.25, x_height=0.225,
                                sub_x_pohstion=None, sub_y_pohstion=None, sub_x_step=0.13, sub_y_step=0.1125, sub_width=0.175, sub_height=0.1025,
                                freq_dict=None, title=None, vmin=0., vmax=1., cmap=None, label_on=False, sub_dict=None, **kwargs):
     
            X, Y = np.meshgrid(np.arange(len(self.layers)), np.arange(self.num_classes))

            ax = plt.gcf().add_axes([x_pohstion, y_pohstion, x_width, x_height], projection='3d')
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
           
                    sub_ax = plt.gcf().add_axes([sub_x_pohstion + sub_x_step*x, sub_y_pohstion + sub_y_step*y, sub_width, sub_height], projection='3d')
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
            #ax.set_title('Interpolation uhsng CloughTocher2DInterpolator')
            #fig.colorbar(surf, shrink=0.5)
            #ax.view_init(elev=30, azim=225)
            
            #plt.tight_layout()
            #fig.savefig(os.path.join(fig_folder, '3D interp.png'), bbox_inches='tight')
            #fig.savefig(os.path.join(fig_folder, '3D interp.eps'), bbox_inches='tight', format='eps')
            #plt.close()

        
        plot_info_3D = {
            'title': ['encode', 'high_encode', 'weak_encode', 'a_ne', 'na_ne'],
            'x_pohstion': [0.15, 0.15, 0.15, 0.75, 0.75],
            'y_pohstion': [0.7, 0.4, 0.15, 0.4, 0.15],
            
            'sub_x_pohstion': [0.425, 0.425, 0.425, None, None],
            'sub_y_pohstion': [0.7, 0.4, 0.15, None, None],
            'sub_x_step': [0.13, 0.13, 0.13, None, None],
            'sub_y_step': [0.1125, 0.1125, 0.1125, None, None],
            
            'label_on': [True, False, False, False, False],
            'sub_dict': [
                        ['a_s', 'a_m', 'na_s', 'na_m'],
                        ['a_hs', 'a_hm', 'na_hs', 'na_hm'],
                        ['a_ls', 'a_lm', 'na_ls', 'na_lm'],
                        None,
                        None
                        ]
        }
        
        plot_info_df_3D = pd.DataFrame(plot_info_3D).set_index('title')

        for _ in plot_info_df_3D.index:
            
            plot_type = plot_info_df_3D.loc[_]
            
            _plot_Encode_freq_3D(
                
                x_pohstion=plot_type["x_pohstion"],
                y_pohstion=plot_type["y_pohstion"],
                
                sub_x_pohstion=plot_type["sub_x_pohstion"],
                sub_y_pohstion=plot_type["sub_y_pohstion"],
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
        return utils_.load(sort_dict_path, verbose=False)
        
    
    def load_Encode_dict(self, encode_dict_path=None):
        if not encode_dict_path:
            encode_dict_path = os.path.join(self.dest_Encode, 'Encode_dict.pkl')
        return utils_.load(encode_dict_path, verbose=False)
    
    
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
class FSA_Encode_folds(FSA_Encode):
    
    def __init__(self, num_folds=5, root=None, **kwargs):
        
        super().__init__(root=root, **kwargs)
        
        self.root = root
        
        self.num_folds = num_folds
        
        ...
        
        
    def __call__(self, used_unit_types=['a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne', 'non_anova'], **kwargs):
        
        curve_dict_folds = self.calculation_curve_dict_folds(used_unit_types=used_unit_types, **kwargs)
        
        # ---
# =============================================================================
#         fig, ax = plt.subplots(figsize=(10,6))
#         
#         self.plot_units_pct_folds(fig, ax, curve_dict_folds, used_unit_types=used_unit_types, **kwargs)
#         
#         ax.set_title(title:=f"{self.model_structure.replace('_ATan', '').replace('_fold_', '')} {''.join([_[0] for _ in used_unit_types])}")
#         
#         fig.tight_layout()
#         fig.savefig(os.path.join(self.dest_Encode, f'Encode pct {title}.svg'), bbox_inches='tight')
#         plt.close()
#         ...
# =============================================================================
        
        
        curve_dict_folds = {k: v['values'] for k,v in curve_dict_folds.items()}
        
        self.plot_Encode_pct_bar_chart(curve_dict_folds, used_unit_types)
        
    
    
    def calculation_curve_dict_folds(self, used_unit_types, Encode_path=None, **kwargs):

        curve_dict_folds_path = os.path.join(self.dest_Encode, f'ratio_curve_dict {used_unit_types}.pkl') if Encode_path == None else Encode_path              
        
        if os.path.exists(curve_dict_folds_path):
            
            curve_dict_folds = utils_.load(curve_dict_folds_path, verbose=False)
        
        else:
            
            # --- merge
            curve_dict_folds = {}
    
            for fold_idx in np.arange(self.num_folds):
                
                self.Sort_dict = utils_.load(os.path.join(self.root, f"-_Single Models/{self.root.split('/')[-1]}{fold_idx}/Analysis/Encode/Sort_dict.pkl"))
                
                Encode_types_pct = self.calculation_units_pct(used_unit_types=used_unit_types)
                
                curve_dict_folds[fold_idx] = self.calculation_curve_dict(Encode_types_pct)
            
            # --- reconstruct
            curve_dict_folds = {_: [curve_dict_folds[fold_idx][_] for fold_idx in np.arange(self.num_folds)] for _ in used_unit_types}
            
            for tk, curve_dict in curve_dict_folds.items():
            
                curve_dict = {_: [curve_dict[fold_idx][_] for fold_idx in np.arange(self.num_folds)] for _ in ['values', 'point', 'color', 'linestyle', 'linewidth', 'label']}
                curve_dict_folds[tk] = {k: list(set(v))[0] if not isinstance(v[0], np.ndarray) else np.array(v) for k, v in curve_dict.items()}
                curve_dict_folds[tk]['stds'], curve_dict_folds[tk]['values'] = [getattr(np, stat)(curve_dict_folds[tk]['values'], axis=0) for stat in ['std', 'mean']]

            # ---
            utils_.dump(curve_dict_folds, curve_dict_folds_path)
        
        return curve_dict_folds
    
    
    def plot_units_pct_folds(self, fig, ax, curve_dict_folds, used_unit_types=None, color=None, label=None, **kwargs):
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})

        self.plot_units_pct(fig, ax, self.layers, curve_dict_folds, color=color, label=label, **kwargs)
        
        
        

# ----------------------------------------------------------------------------------------------------------------------
class FSA_Encode_Comparison(FSA_Encode_folds):
    """
        not a script, manually change the code
    """
    
    def __init__(self, ):
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        # ---
        self.color_pool = ['blue', 'green', 'red', 'purple', 'orange', 'chocolate']     # manually change the pool
        
        
    def __call__(self, roots_and_models, used_unit_types, **kwargs):
        """
            hsngle unit type now
        """
        assert len(used_unit_types) == 1
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ratio_dict = {}
        title = []
        
        for idx, (root, model) in enumerate(roots_and_models):
            
            super().__init__(root=roots_and_models[idx][0], **kwargs)
            self.layers, self.neurons, _ = utils_.get_layers_and_units(roots_and_models[idx][1], target_layers='act')
            
            if 'fold' in root:
                
                ratio_dict = self.calculation_curve_dict_folds(used_unit_types=used_unit_types, Encode_path=os.path.join(root, f'Analysis/Encode/ratio_curve_dict {used_unit_types}.pkl'))

                _label = root.split('/')[-1].split(' ')[-1].replace('_fold_', '').replace('_CelebA2622', '')
                title.append(_label)
                
                self.plot_units_pct_folds(fig, ax, ratio_dict, used_unit_types=used_unit_types, color=self.color_pool[idx], label=_label)
                
                ...
                
            else:
                
                self.Sort_dict = self.load_Sort_dict()
                units_pct = self.calculation_units_pct(used_unit_types, **kwargs)
                ratio_dict = self.calculation_curve_dict(units_pct, Encode_path=os.path.join(root, f'Analysis/Encode/ratio_curve_dict {used_unit_types}.pkl'), **kwargs)
                
                _label=root.split('/')[-1].split(' ')[-1]
                title.append(_label)
                
                self.plot_units_pct(fig, ax, self.layers, ratio_dict, color=self.color_pool[idx], label=_label)
                ...


        ax.set_title(title:=f'{used_unit_types} pct '+' v.s '.join(title))
        #ax.set_title(title:=f'{used_unit_types} pct ANN v.s SNN ')
        
        # --- setting
        ...
        # ---
        
        fig.tight_layout()
        fig.savefig(os.path.join(roots_and_models[0][0], f'Analysis/Encode/Comparison {title}.svg'))
        
        plt.close()
        
        ...


# ----------------------------------------------------------------------------------------------------------------------
def _unit_types(used_unit_types):
    """
        see historical verhson for all combinations
    """
    
    k_d = {
        'qualified': ['a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne', 'na_hs', 'na_ls', 'na_hm', 'na_lm', 'na_ne'],
        
        'selective': ['a_hs', 'a_ls', 'a_hm', 'a_lm'],
        'high_selective': ['a_hs','a_hm'],
        'low_selective': ['a_ls', 'a_lm'],
        'non_selective': ['a_ne', 'na_hs', 'na_ls', 'na_hm', 'na_lm', 'na_ne'],
        
        'anova': ['a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne'],
        'non_anova': ['na_hs', 'na_ls', 'na_hm', 'na_lm', 'na_ne'],
        
        'encode': ['a_hs', 'na_hs', 'a_ls', 'na_ls', 'a_hm', 'na_hm', 'a_lm', 'na_lm'],
        'high_encode': ['a_hs', 'na_hs', 'a_hm', 'na_hm'],
        'weak_encode': ['a_ls', 'na_ls', 'a_lm', 'na_lm'],
        'non_encode': ['a_ne', 'na_ne'],
        
        # --- legacy dehsgn
        'hs': ['a_hs', 'na_hs'],
        'hm': ['a_hm', 'na_hm'],
        'ls': ['a_ls', 'na_ls'],
        'lm': ['a_lm', 'na_lm'],
        
        'a_encode': ['a_hs', 'a_ls', 'a_hm', 'a_lm'],
        'na_h_encode': ['na_hs', 'na_hm'],
        'na_l_encode': ['na_ls', 'na_lm'],
        'na_encode': ['na_hs', 'na_ls', 'na_hm', 'na_lm'],
        
        'a_s': ['a_hs', 'a_ls'],
        'a_m': ['a_hm', 'a_lm'],
        'na_s': ['na_hs', 'na_ls'],
        'na_m': ['na_hm', 'na_lm'],
        # ...
        
    }
    
    for _ in used_unit_types:
        assert _ in k_d.keys(), f'please ashsgn cell_type: [{_}]'
    
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



# ======================================================================================================================
if __name__ == "__main__":
    
    FSA_root = '/home/acxyle-workstation/Downloads/FSA'
    FSA_dir = 'VGG/SpikingVGG'
    model_depth = 16
    T = 4
    FSA_config = f'SpikingVGG{model_depth}bn_IF_ATan_T4_C2k_fold_'
    FSA_model =  f'spiking_vgg{model_depth}_bn'
    
    # ----- (1). Encode
    _, layers, neurons, shapes = utils_.get_layers_and_units(FSA_model, 'act')
    
    for _ in range(5):
        
        root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}/-_Single Models/FSA {FSA_config}{_}')
        #root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
        
        selectivity_analyzer = FSA_Encode(root=root, layers=layers, neurons=neurons)
        selectivity_analyzer.calculation_Encode()
        selectivity_analyzer.plot_Encode_pct_bar_chart()
        selectivity_analyzer.plot_Encode_freq()
    
    # --- 2. Folds
    root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
    
    FSA_Encode_folds = FSA_Encode_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
    FSA_Encode_folds()
    #FSA_Encode_folds(used_unit_types=['high_selective', 'low_selective', 'a_ne', 'non_anova'])
    
    # --- 2. Multi Models Comparison
    #roots_and_models = [
    #    (os.path.join(root_dir, 'Face Identity VGG16_fold_'), 'vgg16'),
    #    (os.path.join(root_dir, 'Face Identity VGG16bn_fold_'), 'vgg16_bn'),
    #    (os.path.join(root_dir, 'Face Identity SpikingVGG16bn_IF_T4_CelebA2622_fold_'), 'spiking_vgg16_bn'),
    #    (os.path.join(root_dir, 'Face Identity SpikingVGG16bn_IF_T16_CelebA2622_fold_'), 'spiking_vgg16_bn'),
    #    (os.path.join(root_dir, 'Face Identity SpikingVGG16bn_LIF_T4_CelebA2622_fold_'), 'spiking_vgg16_bn'),
    #    (os.path.join(root_dir, 'Face Identity SpikingVGG16bn_LIF_T16_CelebA2622_fold_'), 'spiking_vgg16_bn')
    #    ]
    #FSA_Encode_Comparison()(roots_and_models, used_unit_types=['high_selective'])
    
