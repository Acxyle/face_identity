#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 22:02:44 2023

@author: acxyle

    This code contains (1) TSNE, (2.1) Distance, (2.2) Pearsons Correlation
"""


import os
import math
import warnings
import logging

import numpy as np
import seaborn as sn
from tqdm import tqdm
import matplotlib as mpl
from scipy.io import savemat
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from matplotlib.transforms import ScaledTranslation

import vgg, resnet
import spiking_vgg, spiking_resnet, sew_resnet
from spikingjelly.activation_based import surrogate, neuron, functional

import utils_


class Selectiviy_Analysis_DR():
    
    def __init__(self, 
                 root='/Identity_Spikingjelly_VGG_Results/',
                 num_samples=10, num_classes=50, layers=None, neurons=None,  data_name=None):
        
        if layers == None or neurons == None:
            raise RuntimeError('[Codwarning] invalid layers and neurons')
            
        self.layers = layers
        self.neurons = neurons
        
        self.root = os.path.join(root, 'Features')
        self.dest = os.path.join(root, 'Analysis')
        utils_.make_dir(self.dest)
        
        self.dest_DR = os.path.join(self.dest, 'Dimension_Reduction')
        utils_.make_dir(self.dest_DR)
        
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        if data_name != None:
            self.data_name = data_name
            
        self.model_structure = root.split('/')[-2].split(' ')[2]
        
    #FIXME
    def selectivity_analysis_tsne(self, parallel=False, verbose=False):
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        print('[Codinfo] Executing selectivity_analysis_Tsne...')
        label = utils_.makeLabels(self.num_samples, self.num_classes)
        
        self.save_path_DR = os.path.join(self.dest_DR, 'TSNE')
        utils_.make_dir(self.save_path_DR)
        
        self.save_path_fig_DR = os.path.join(self.save_path_DR, 'Figures')
        utils_.make_dir(self.save_path_fig_DR)
        
        
        valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if not item[1].startswith('not')])
        markers = valid_markers + valid_markers[:self.num_classes - len(valid_markers)]
        
        # === put calculation here
        self.tsne_analysis_calculation(label, markers, in_layer_plot=True)
        
        # === put plot here
        self.tsne_analysis_plot(label, markers)
        
    def tsne_analysis_calculation(self, label, markers, in_layer_plot):
        
        save_path = os.path.join(self.save_path_DR, 'tsne_all.pkl')
        
        if os.path.exists(save_path):
            
            self.tsne_layer_dict = utils_.pickle_load(save_path)
        
        else:
        
            Sort_dict = utils_.pickle_load(os.path.join(self.dest, 'Encode/Sort_dict.pkl'))
            
            self.tsne_layer_dict = {}
            
            layers = self.layers[:]
            
            # ----- sequantial calculation
            # ----- parallel can only be used for later layers because of RAM limit
            for layer in tqdm(layers, desc='TSNE Sequential'):
                
                feature = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))
                
                advanced_units_types = Sort_dict[layer]['advanced_type']
                
                tsne_layer = self.tsne_layer_calculation(layer=layer, 
                                            feature=feature, 
                                            mask_dict=advanced_units_types, 
                                            label=label, 
                                            markers=markers,
                                            plot=in_layer_plot)  
                
                self.tsne_layer_dict.update(tsne_layer)
                
            utils_.pickle_dump(save_path, self.tsne_layer_dict)
        
    def tsne_layer_calculation(self, layer, feature, mask_dict, label, markers, plot=True):
        """
            operation for each layer
        """
        
        perplexity_dict = self.calculate_perplexity(mask_dict)
        perplexity_dict.update(self.calculate_perplexity({'all': np.arange(feature.shape[1])}))
        
        mask_dict.update({'all': np.arange(feature.shape[1])})

        tsne_dict, tsne_coordinate = self.tsne_layer_process(layer, feature, label, mask_dict, perplexity_dict, markers)
        
        tsne_all = {layer: {
                         'tsne_dict': tsne_dict,
                         'tsne_coordinate': tsne_coordinate
                         }}

        if plot:
            
            # in-layer comparison
            fig, ax = plt.subplots(3, 3, figsize=(18,18), dpi=100)
            
            row_idx = 0
            column_idx = 0
            
            for idx, type_ in enumerate(list(tsne_dict.keys())):
                
                norm_lim = True
                
                self.tsne_in_layer_plot(ax=ax[row_idx, column_idx], 
                               tsne=tsne_dict[type_], 
                               mask=mask_dict[type_], 
                               label=label, 
                               markers=markers, 
                               title=type_, 
                               layer=layer, 
                               tsne_coordinate=tsne_coordinate,
                               norm_lim=norm_lim)
                
                column_idx += 1
                if column_idx == 3:
                    row_idx += 1
                    column_idx = 0
            
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                
                fig.savefig(self.save_path_fig_DR + f'/tsne_{layer}.png', bbox_inches='tight', dpi=100)     # change dpi whilst data change
                fig.savefig(self.save_path_fig_DR + f'/tsne_{layer}.eps', bbox_inches='tight', dpi=100, format='eps')
                plt.close()  
                
        return tsne_all
            
    def tsne_layer_process(self, layer, feature, label, mask_dict, perplexity_dict, markers):
        """
            this function calculate the tSNE and saved results, also plot the reduced feature map
        """
        
        tsne_dict = {}
        
        for type_ in perplexity_dict.keys():
            
            mask = mask_dict[type_]
            
            if mask.size == 0:
                tsne_dict.update({type_: None})
                
            else:
                feature_ = feature[:, mask]
                perplexity = perplexity_dict[type_]
                tsne = self.tsne_layer_process_single(feature_, perplexity)
                tsne_dict.update({type_: tsne})
            
        # --- obtain tsne_coordinate: 4 corner points
        tsne_values_all = list(tsne_dict.values())
        
        tsne_x = [_[:,0] for _ in tsne_values_all if _ is not None]
        tsne_y = [_[:,1] for _ in tsne_values_all if _ is not None]
        tsne_x = np.hstack(tsne_x)
        tsne_y = np.hstack(tsne_y)

        tsne_coordinate = {
            'x_min': np.min(tsne_x),
            'x_max': np.max(tsne_x),
            'y_min': np.min(tsne_y),
            'y_max': np.max(tsne_y)
            }
        
        return tsne_dict, tsne_coordinate
  
    def tsne_layer_process_single(self, feature: np.array, perplexity):
        """
            this function calculates the tsne dimension reduction.
            on behalf of TSNE, need to increase SWAP for shallow layer NC_Lab_desktop and NC_Lab_Workstation, both 128G Memory
            a) when performing tSNE to very high dimensional data, there's a lot of problems for algorithms and computing:
                1. curse of dimensionality (human experience about 'distance' may invalid for high dimensional data)
                2. noise
                3. overfitting
                4. interpretability
                ...
            thus, it is reasonable to reduce the dimension of the raw feature
            b) a commonly used way to reduce the dimension firstly is PCA, for the feature of 500*3m, usually can use PCA to reduce it to 1~500 dimensions (citation) before the tSNE, 
        the disadvantage is the dimensions after PCA can not exceeds min(n_classes, n_features))
            c) according to Van der Maaten and Hinton, they suggested to try different values of perplexity, to see the trade-off between local and glocal relationships
        """
        # [notice] the first 2 methods have no mask judgement, manually add the conditional statement if necessary
        # --- method 1, set a threshold for data size
        #if len(mask) > 1700000:
        #    print(f'[Codinfo] too large input [{len(mask)}], skipped.')
        #    return
        #else:
        #    if len(mask) == 0:
        #        return
        #    elif len(mask) == 1:
        #        tsne = feature[:, mask]
        #        return tsne
        #    else:
        #        if np.std(np.array(feature[:, mask])) != 0.:     # [notice] make sure all faetures are not identical
        #            tsne = TSNE(perplexity=perplexity, n_jobs=-1).fit_transform(feature[:,mask]) 
        #            return tsne
        #        else:     # all values in different units are identical
        #            return
        # --- method 2, use PCA to reduce all feature as (500,500)
        #test_value = int(self.num_classes*self.num_samples)     
        #if feature[:, mask].shape[1] > test_value:     
        #    np_log = math.ceil(test_value*(math.log(len(mask)/test_value)+1.))
        #    pca = PCA(n_components=min(test_value, np_log))
        #    x = pca.fit_transform(feature[:, mask])
        #    tsne = TSNE(perplexity=perplexity, n_jobs=-1).fit_transform(x)
                
        # --- method 3, manually change the SWAP for large data load
        if feature.size == 0:     # no value
            return
        elif feature.shape[1] == 1:     # dim=1
            tsne = np.repeat(feature, 2, axis=1)
            return tsne
        else:
            if np.std(feature) != 0.:     # [notice] make sure all faetures are not identical
                tsne = TSNE(perplexity=perplexity, n_jobs=-1).fit_transform(feature) 
                return tsne
            else:     # all values in different units are identical
                return
        
    def tsne_in_layer_plot(self, ax, tsne, mask, label, markers, title, layer, tsne_coordinate, norm_lim=True):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        if tsne is not None:
            
            if norm_lim == True:
                tsne_x_min = tsne_coordinate['x_min']
                tsne_x_max = tsne_coordinate['x_max']
                tsne_y_min = tsne_coordinate['y_min']
                tsne_y_max = tsne_coordinate['y_max']
            else:
                tsne_x_min = np.min(tsne[:,0])
                tsne_x_max = np.max(tsne[:,0])
                tsne_y_min = np.min(tsne[:,1])
                tsne_y_max = np.max(tsne[:,1])
                
            width = tsne_x_max - tsne_x_min
            height = tsne_y_max - tsne_y_min

            for i in range(self.num_classes):     # for each class
                self.tsne_in_layer_plot_scatter(ax, tsne, i, label, markers)
                
            ax.set_xlim((tsne_x_min-0.025*width, tsne_x_min+1.025*width))
            ax.set_ylim((tsne_y_min-0.025*height, tsne_y_min+1.025*height))
            
            ax.vlines(0, tsne_x_min-0.5*width, tsne_x_min+1.5*width, colors='gray',  linestyles='--', linewidth=2.0)
            ax.hlines(0, tsne_y_min-0.5*height, tsne_y_min+1.5*height, colors='gray',  linestyles='--', linewidth=2.0)
        
        ax.set_title(f'{title} \n {mask.size}/{self.neurons[self.layers.index(layer)]}({mask.size/self.neurons[self.layers.index(layer)]*100:.2f}%)')
        ax.grid(False)
        
        
        
        
    def tsne_in_layer_plot_scatter(self, ax, tsne, i, label, markers):
        
        try:
            if tsne.shape[1]==1:
                ax.scatter(tsne[i*self.num_samples: (i+1)*self.num_samples, 0], np.zeros_like(tsne[i*self.num_samples: (i+1)*self.num_samples, 0]),
                            label[i*self.num_samples: (i+1)* self.num_samples], marker=markers[i])
                ax.set_xticklabels([])
            elif tsne.shape[1]==2:
                ax.scatter(tsne[i*self.num_samples: (i+1)*self.num_samples, 0], tsne[i*self.num_samples: (i+1)*self.num_samples, 1],
                            label[i*self.num_samples: (i+1)* self.num_samples], marker=markers[i])

        except AttributeError as e:
            if "'NoneType' object has no attribute 'shape'" in str(e):
                pass
            else:
                raise
    
    def calculate_perplexity(self, mask_dict):
        
        mask_types = list(mask_dict.keys())
        perplexity_dict = {}
        
        for key in mask_types:
            
            mask = mask_dict[key]
            
            if len(mask) > 0:     # if units number > 0
                perplexity = min(math.sqrt(len(mask)), self.num_classes*self.num_samples-1)     # min(square_root(unit numbers), 499)
            else:     # no unit
                perplexity = 1e-9
            
            perplexity_dict.update({key: perplexity})
            
        return perplexity_dict
    
    # -----
    def tsne_analysis_plot(self, label, markers, ):   
        
        print('[Codinfo] Executing tsne_analysis_plot...')
        
        if not hasattr(self, 'tsne_layer_dict'):
            self.tsne_layer_dict = utils_.pickle_load(os.path.join(self.save_path_DR, 'tsne_all.pkl'))
        
        Sort_dict = utils_.pickle_load(os.path.join(self.dest, 'Encode/Sort_dict.pkl'))
        
        idces, layers, _ = utils_.imaginary_neurons_vgg(self.layers)
        
        self.tsne_analysis_plot_single(layers, idces, Sort_dict, label, markers, norm_lim=True, suptitle='neuron_norm_lim')

        self.tsne_analysis_plot_single(layers, idces, Sort_dict, label, markers, norm_lim=False, suptitle='neuron')
        
        
    def tsne_analysis_plot_single(self, layers, idces, Sort_dict, label, markers, norm_lim, suptitle):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        fig, ax = plt.subplots(len(layers), 7, figsize=(6*7, 6*len(layers)), dpi=100)
        
        row_idx = 0
        column_idx = 0
        
        for idx, layer in zip(idces, layers):

            tsne_dict = self.tsne_layer_dict[layer]['tsne_dict']
            tsne_coordinate = self.tsne_layer_dict[layer]['tsne_coordinate']
            
            mask_dict = Sort_dict[layer]['advanced_type']
            mask_dict.update({'all': np.arange(self.neurons[idx])})
            
            for idx, type_ in enumerate(['all', 'sensitive_si_idx', 'sensitive_mi_idx', 'sensitive_non_encode_idx',
                                         'non_sensitive_si_idx', 'non_sensitive_mi_idx', 'non_sensitive_non_encode_idx']):
                
                self.tsne_in_layer_plot(ax=ax[row_idx, column_idx], 
                               tsne=tsne_dict[type_], 
                               mask=mask_dict[type_], 
                               label=label, 
                               markers=markers, 
                               title=layer+' '+type_, 
                               layer=layer, 
                               tsne_coordinate=tsne_coordinate,
                               norm_lim=norm_lim)
                
                column_idx += 1
                if column_idx == 7:
                    row_idx += 1
                    column_idx = 0
        
        fig.suptitle(f'{self.model_structure}', y=0.895, fontsize=28)
        
        fig.savefig(self.save_path_DR + f'/tsne_all_{suptitle}.png', bbox_inches='tight', dpi=100)     # change dpi whilst data change
        #fig.savefig(self.save_path_DR + f'/tsne_all_{suptitle}.eps', bbox_inches='tight', dpi=100, format='eps')
        plt.close() 
    
# ==================================================================================================================
#FIXME
"""
    in my expectation, the EU distance and Pearsons' Correlation should be merged in one function but with different 
    args to produce different metrics, so that in future can add other values for RSA
"""
class Selectiviy_Analysis_SM():
    
    def __init__(self, 
                 root='/Identity_Spikingjelly_VGG_Results/',
                 num_samples=10, num_classes=50, layers=None, neurons=None,  data_name=None):
        
        if layers == None or neurons == None:
            raise RuntimeError('[Codwarning] invalid layers and neurons')
            
        self.layers = layers
        self.neurons = neurons
        
        self.root = os.path.join(root, 'Features')
        self.dest = os.path.join(root, 'Analysis')
        utils_.make_dir(self.dest)
        
        self.dest_DSM = os.path.join(self.dest, '(Dis)Similarity_Matrix')
        utils_.make_dir(self.dest_DSM)
        
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        if data_name != None:
            self.data_name = data_name
            
        self.model_structure = root.split('/')[-2].split(' ')[2]
        
    
    def selectivity_analysis_similarity_metrics(self, metrics: list[str]):
        """
            metrics should be a list of metrics,
            now have: (1) Euclidean Distance; (2) Pearson Correlation Coefficient
        """
        print(f'[Codinfo] Executing similarity_metrics | {self.model_structure}...')
        
        # ----- load different types of units
        self.Sort_dict = utils_.pickle_load(os.path.join(self.dest, 'Encode/Sort_dict.pkl'))
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        # =====
        for self.metric in metrics:     # for each metric
            
            self.metric_folder = os.path.join(os.path.join(self.dest_DSM, f'{self.metric}'))
            utils_.make_dir(self.metric_folder)
            
            dict_path = os.path.join(self.metric_folder, f'{self.metric}.pkl')
            
            if os.path.exists(dict_path):
                metric_dict = utils_.pickle_load(dict_path)
            
            else:
                metric_dict = self.selectivity_analysis_similarity(in_layer_plot=True)
                
                utils_.pickle_dump(dict_path, metric_dict)
                #savemat(os.path.join(self.metric_folder, f'{self.metric}.mat'), metric_dict)
            
            # ----- plot
            if self.metric == 'euclidean':
                self.selectivity_analysis_plot(metric_dict, sup_v=None)
            elif self.metric == 'pearson':
                self.selectivity_analysis_plot(metric_dict, sup_v=(0,2))
               
    def selectivity_analysis_similarity(self, in_layer_plot:bool=False):
        
        print(f'[Codinfo] Executing selectivity_analysis_metric [{self.metric}]...')
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        self.dest_metric = os.path.join(self.dest_DSM, f'{self.metric}')
        utils_.make_dir(self.dest_metric)
        
        metric_dict = {}     # use a dict to store the info of each layer
        
        layers = self.layers[:]
        
        for layer in tqdm(layers):     # for each layer
            
            feature = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))     # (500, num_units)
            
            mean_FR = self.calculate_mean_FR(feature)
            sorted_idx = utils_.lexicographic_order(self.num_classes)     # correct labels
            mean_FR = self.restore_order(mean_FR, sorted_idx)     # (50, num_units)
            
            # -----
            units_type_dict = self.Sort_dict[layer]['advanced_type']
            units_type_dict.update({'all': np.arange(mean_FR.shape[1])})
            # -----
    
            # ----- generate similarity metrics
            metric_type_dict = {}
            
            for type_ in [_ for _ in units_type_dict.keys() if 'sensitive_encode_idx' not in _]:
                
                similarity_dict = selectivity_analysis_calculation(self.metric, mean_FR[:, units_type_dict[type_].astype(int)])
                metric_type_dict[type_] = similarity_dict
            # -----
            
            metric_dict[layer] = metric_type_dict
            
            if in_layer_plot:
                
                if self.metric == 'euclidean':     # for any values
                    self.selectivity_analysis_similarity_in_layer_plot(layer, metric_type_dict)     
                elif self.metric == 'pearson':     # for similarity values
                    self.selectivity_analysis_similarity_in_layer_plot(layer, metric_type_dict, (0,2))    
            
        return metric_dict
    
    #FIXME
    def selectivity_analysis_similarity_in_layer_plot(self, layer, metric_type_dict, v:tuple=None):
        
        plot_folder = os.path.join(self.metric_folder, 'in_layer_Figures')
        utils_.make_dir(plot_folder)
        
        metric_values_pool = np.array([metric_type_dict[_]['matrix'] for _ in metric_type_dict.keys() if metric_type_dict[_] != None])
        
        if v == None:
            vmin = np.min(metric_values_pool)
            vmax = np.max(metric_values_pool)
        else:
            vmin = v[0]
            vmax = v[1]
        
        fig, ax = plt.subplots(1,7,figsize=(35,5))

        for idx, key in enumerate(metric_type_dict.keys()):
            
            ax[idx].set_title(f'{key}')
            
            if metric_type_dict[key] != None:
                ax[idx].imshow(metric_type_dict[key]['matrix'], origin='lower', vmin=vmin, vmax=vmax)
                #sn.heatmap(similarity_matrix, ax=ax[idx]], cbar=0, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)     # legacy use
                
                if metric_type_dict[key]['contains_nan'] == True:
                    ax[idx].set_title(f'{key} [contains NaN]')
                
                ax[idx].set_xlabel(f"{metric_type_dict[key]['num_units']/(self.neurons[self.layers.index(layer)])*100:.2f}%")
            
            else:
                
                ax[idx].set_xlabel("0.00%")
                
            ax[idx].set_xticks([])
            ax[idx].set_yticks([])


        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            fig.tight_layout(rect=[0, 0, .9, 1])
            fig.suptitle(layer, y=1.025)
            fig.savefig(os.path.join(plot_folder, f'{layer}.png'), bbox_inches='tight', dpi=100)
            plt.close()
        
    def selectivity_analysis_plot(self, metric_dict, sup_v:tuple=None):

        plt.rcParams.update({"font.size": 30})
        
        # ----- not applicable for all metrics
        metric_dict_ = {_:{__: metric_dict[_][__]['matrix'] if metric_dict[_][__] != None else None for __ in metric_dict[_].keys()} for _ in metric_dict.keys()}
        metric_dict_pool = np.concatenate([_ for _ in [np.concatenate([metric_dict_[key][__] for __ in metric_dict_[key].keys() if metric_dict_[key][__] is not None]).reshape(-1) for key in metric_dict_.keys()]])   # in case of inhomogeneous shape
        
        if sup_v is None:
            sup_vmin = np.min(metric_dict_pool)
            sup_vmax = np.max(metric_dict_pool)
        else:
            sup_vmin = sup_v[0]
            sup_vmax = sup_v[1]
        
        tqdm_bar = tqdm(total=6, desc='Plot')
        
        # -----
        self.selectivity_analysis_plot_single(metric_dict, sup_vmin, sup_vmax, layer_type='imaginary_neuron', plot_type='')
        tqdm_bar.update(1)
        
        # -----
        self.selectivity_analysis_plot_single(metric_dict, sup_vmin, sup_vmax, layer_type='imaginary_neuron', plot_type='suplim')
        tqdm_bar.update(1)
        
        # -----
        self.selectivity_analysis_plot_single(metric_dict, sup_vmin, sup_vmax, layer_type='imaginary_neuron', plot_type='norm')
        tqdm_bar.update(1)
        
        # -----
        self.selectivity_analysis_plot_single(metric_dict, sup_vmin, sup_vmax, layer_type='all', plot_type='')
        tqdm_bar.update(1)
        
        # -----
        self.selectivity_analysis_plot_single(metric_dict, sup_vmin, sup_vmax, layer_type='all', plot_type='suplim')
        tqdm_bar.update(1)
        
        # -----
        self.selectivity_analysis_plot_single(metric_dict, sup_vmin, sup_vmax, layer_type='all', plot_type='norm')
        tqdm_bar.update(1)
    
    def selectivity_analysis_plot_single(self, metric_dict, sup_vmin, sup_vmax, cmap='turbo', layer_type:str=None, plot_type:str=None):
        
        plot_folder = os.path.join(self.metric_folder, 'Figures')
        utils_.make_dir(plot_folder)
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        if layer_type == None or 'all' in layer_type.lower():
            layers = self.layers
        elif 'neuron' in layer_type.lower() or 'unit' in layer_type.lower():
            _, layers, _ = utils_.imaginary_neurons_vgg(self.layers)
        
        fig,ax = plt.subplots(7, len(layers), figsize=(5*len(layers), 35))

        for idx, layer in enumerate(layers):
            
            metric_layer_pool = np.concatenate([_['vector'] for _ in metric_dict[layer].values() if _ is not None])
            vmin = np.min(metric_layer_pool)
            vmax = np.max(metric_layer_pool)
            vnorm = vmax-vmin
            
            for idx_, type_ in enumerate(metric_dict[layer].keys()):
                
                if idx_ == 0:
                    ax[idx_, idx].set_title(layer)
                    
                if idx == 0:
                    ax[idx_, idx].set_ylabel(type_)
                
                if plot_type == None or plot_type == '':
                    if metric_dict[layer][type_] is not None:
                        ax[idx_, idx].imshow(metric_dict[layer][type_]['matrix'], origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
                        ax[idx_, idx].set_xlabel(f"{metric_dict[layer][type_]['num_units']/(self.neurons[self.layers.index(layer)])*100:.2f}%")
                    else:
                        ax[idx_, idx].set_xlabel("0.00%")
                        
                elif plot_type == 'suplim':
                    if metric_dict[layer][type_] is not None:
                        ax[idx_, idx].imshow(metric_dict[layer][type_]['matrix'], origin='lower', vmin=sup_vmin, vmax=sup_vmax, cmap=cmap)
                        ax[idx_, idx].set_xlabel(f"{metric_dict[layer][type_]['num_units']/(self.neurons[self.layers.index(layer)])*100:.2f}%")
                        
                        cax = fig.add_axes([1.01, 0.1, 0.01, 0.75])
                        norm = plt.Normalize(vmin=sup_vmin, vmax=sup_vmax)
                        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                        
                    else:
                        ax[idx_, idx].set_xlabel("0.00%")
                        
                elif plot_type == 'norm':
                    if metric_dict[layer][type_] is not None:
                        ax[idx_, idx].imshow((metric_dict[layer][type_]['matrix']-vmin)/vnorm, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
                        ax[idx_, idx].set_xlabel(f"{metric_dict[layer][type_]['num_units']/(self.neurons[self.layers.index(layer)])*100:.2f}%")
                        
                        cax = fig.add_axes([1.01, 0.1, 0.01, 0.75])
                        norm = plt.Normalize(vmin=0, vmax=1)
                        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                        
                    else:
                        ax[idx_, idx].set_xlabel("0.00%")
                
                ax[idx_, idx].set_xticks([])
                ax[idx_, idx].set_yticks([])

        
        fig.suptitle(f'{self.model_structure} | {self.metric} | {layer_type} | {plot_type}', y=1.015, fontsize=50)
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            fig.tight_layout()
            fig.savefig(os.path.join(plot_folder, f'{layer_type}_{plot_type}.png'), bbox_inches='tight', dpi=100)
            plt.close()

    #FIXME parellel process
    def selectivity_analysis_correlation(self):
        
        print('[Codinfo] Executing selectivity_analysis_correlation...')
        dest = os.path.join(self.dest,'Correlation/')
        utils_.make_dir(dest)
        
        save_folder_id, cor_meaenFR_dict, cor_meanFR_id_selective_dict, cor_meanFR_non_id_selective_dict = self.correlation_state(dest, 'ID/')
        #dest_faces, cor_full_dict, cor_full_ID_dict, cor_full_nonID_dict = self.correlation_state(dest, 'faces/')
        
        for layer in tqdm(self.layers):
            
            feature = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))
            
            mean_FR = self.calculate_mean_FR(feature)     # Order correction
            sorted_idx = utils_.lexicographic_order(self.num_classes)
            mean_FR = self.restore_order(mean_FR, sorted_idx)
            
            mask_id = np.loadtxt(self.dest  + '/' + layer + '-neuronIdx.csv', delimiter=',')
            mask_id = list(map(int, mask_id))
            mask_non_id = sorted(list(set(np.arange(feature.shape[1])) - set(mask_id)))
            
            self.correlation_single(mean_FR, layer, mask_id, mask_non_id, cor_meaenFR_dict, cor_meanFR_id_selective_dict, cor_meanFR_non_id_selective_dict, save_folder_id)
            #[notice] no restore for idx here
            #cor_full_dict, cor_full_ID_dict, cor_full_nonID_dict = self.correlation_single(feature, layer, mask_id, mask_non_id, cor_full_dict, cor_full_ID_dict, cor_full_nonID_dict, dest_faces)
            
        print('[Codinfo] Saving Correlation Matrix...')
        
        self.correlation_save('id_all', cor_meaenFR_dict, 'id_id_selective', cor_meanFR_id_selective_dict, 'id_non_id_selective', cor_meanFR_non_id_selective_dict, dest)
        #self.correlation_save('full', cor_full_dict, 'full_ID', cor_full_ID_dict, 'full_nonID', cor_full_nonID_dict, dest)
   
    def correlation_state(self, dest, data_type):
        save_path = dest+data_type
        utils_.make_dir(save_path)
        
        cor_dict = {}
        cor_sensitive_dict = {}
        cor_nonsensitive_dict = {}
        
        return save_path, cor_dict, cor_sensitive_dict, cor_nonsensitive_dict
    
    def correlation_save(self, cor_name, cor_dict, cor_sensitive_name, cor_sensitive_dict, cor_nonsensitive_name, cor_nonsensitive_dict, dest):
        
        savemat(os.path.join(dest, f'correlation_matrix_{cor_name}.mat'), cor_dict)
        savemat(os.path.join(dest, f'correlation_matrix_{cor_sensitive_name}.mat'), cor_sensitive_dict)
        savemat(os.path.join(dest, f'correlation_matrix_{cor_nonsensitive_name}.mat'), cor_nonsensitive_dict)
        
        #utils_.pickle_dump(os.path.join(dest, f'/correlation_matrix_{cor_name}.pkl'), cor_dict)
        #utils_.pickle_dump(os.path.join(dest, f'/correlation_matrix_{cor_sensitive_name}.pkl'), cor_sensitive_dict)
        #utils_.pickle_dump(os.path.join(dest, f'/correlation_matrix_{cor_nonsensitive_name}.pkl'), cor_nonsensitive_dict)
    
    def correlation_single(self, feature, layer, mask_id, mask_non_id, cor_dict, cor_sensitive_dict, cor_nonsensitive_dict, dest):
        cor_all, cor_sensitive, cor_nonsensitive = self.correlation_calculate_single(feature, layer, mask_id, mask_non_id)
        
        cor_dict.update({layer: cor_all})
        cor_sensitive_dict.update({layer: cor_sensitive})
        cor_nonsensitive_dict.update({layer: cor_nonsensitive})
        
        self.correlation_plot_single(cor_all, cor_sensitive, cor_nonsensitive, mask_id, mask_non_id, layer, dest)
    
    def restore_order(self, mean_FR, sorted_idx):
        mean_FR = [[mean_FR[_,:], sorted_idx[_]] for _ in range(self.num_classes)]
        mean_FR = sorted(mean_FR, key=lambda x:x[1])
        mean_FR = np.array([mean_FR[_][0] for _ in range(self.num_classes)])
        
        return mean_FR
   
    def correlation_calculate_single(self, matrix, layer, mask_id, mask_non_id):
        """
            this function calculates the correlation coefficient,
            
            if np.std(ont_unit_feature) = 0, it will cause corrcoef as nan.
        """
        cor_all = np.corrcoef(matrix)
        cor_id = np.corrcoef(matrix[:, mask_id])
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            cor_nonID = np.corrcoef(matrix[:, mask_non_id])
            cor_nonID[np.isnan(cor_nonID)]=0
        
        return cor_all, cor_id, cor_nonID
        
    def correlation_plot_single(self, cor_all, cor_id, cor_nonID, mask_id, mask_non_id, layer, save_path):
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        fig, axes = plt.subplots(1, 3, figsize=((30, 10)))
        cbar_ax = fig.add_axes([.91, .1, .03, .8])
    
        vmax = max(cor_all.max(), cor_id.max(), cor_nonID.max())
        vmin = min(cor_all.min(), cor_id.min(), cor_nonID.min())
    
        sn.heatmap(cor_all, ax=axes[0], cmap='jet', cbar=0, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
        axes[0].set_title(f'all neurons ({len(mask_id)+len(mask_non_id)})')
        sn.heatmap(cor_id, ax=axes[1], cmap='jet',  cbar=1, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
        axes[1].set_title(f'identity selective neurons ({len(mask_id)})')
        sn.heatmap(cor_nonID, ax = axes[2], cmap='jet',  cbar=2, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
        axes[2].set_title(f'non identity selective neurons ({len(mask_non_id)})')
    
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            fig.tight_layout(rect=[0, 0, .9, 1])
            plt.title(layer)
            plt.savefig(save_path+layer+'-Correlation.png', bbox_inches='tight', dpi=100)
            plt.close()

    def calculate_mean_FR(self, matrix):            
        return np.array([matrix[_*self.num_samples:(_+1)*self.num_samples, :] for _ in range(self.num_classes)]).mean(axis=1)
   
    
#FIXME
def selectivity_analysis_calculation(metric: str, feature: np.array):
    """
        based on [metric] to calculate
    """
    
    similarity_dict = {}
    
    if 'euclidean' in metric.lower():
        similarity_value = pdist(feature, 'euclidean')     # (1225,)
        similarity_matrix = squareform(similarity_value)     # (50, 50)
        
        similarity_dict.update({
            'vector': similarity_value,     # for RSA
            'matrix': similarity_matrix,     # for plot
            'contains_nan': False,     # by default, pdist() can receive null input and generate 0 rather NaN as output
            'num_units': feature.shape[1]
            })
    
    elif 'pearson' in metric.lower():
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            if feature.shape[1] == 0:
                similarity_dict = None
            
            else:
                similarity_matrix = np.corrcoef(feature)
                
                if np.any(np.isnan(similarity_matrix)):     # when detecting NaN value, i.e.,  the values of one class are identical
                    similarity_dict.update({'contains_nan': True})
                    similarity_matrix[np.isnan(similarity_matrix)] = 0
                    
                else:
                    similarity_dict.update({'contains_nan': False})
                    
                similarity_matrix = 1 - similarity_matrix     # SM [-1, 1] -> DSM [0, 2]
                similarity_matrix = (similarity_matrix + similarity_matrix.T)/2     # correct as symmetric
                for _ in range(similarity_matrix.shape[0]):     # correct diagnal values as 0
                    similarity_matrix[_,_] = 0
                    
                similarity_value = squareform(similarity_matrix)     # (1225,)
    
                similarity_dict.update({
                    'vector': similarity_value,     # for RSA
                    'matrix': similarity_matrix,     # for plot
                    'num_units': feature.shape[1]
                    })
    
    return similarity_dict

    
if __name__ == '__main__':
    
# =============================================================================
#     layers = ['Conv_5_3', 'Pool_5', 'FC_6', 'FC_7']
#     neurons = [100352, 25088, 4096, 4096]
# 
#     root_dir = '/media/acxyle/Data/ChromeDownload/'
# 
#     selectivity_additional_analyzer = Selectiviy_Analysis_DR_and_RSA(
#                 root=os.path.join(root_dir, 'Identity_VGG16bn_ReLU_CelebA2622_Results/'), 
#                 dest=os.path.join(root_dir, 'Identity_VGG16bn_ReLU_CelebA2622_Neuron/'), 
#                 layers=layers, neurons=neurons)
#     selectivity_additional_analyzer.selectivity_analysis_tsne()
# =============================================================================

    model_name = 'vgg16_bn'
    
    model_ = vgg.__dict__[model_name](num_classes=50)
    layers, neurons, shapes = utils_.generate_vgg_layers_list_ann(model_, model_name)

    root_dir = '/home/acxyle-workstation/Downloads/'

    selectivity_additional_analyzer = Selectiviy_Analysis_SM(root=os.path.join(root_dir, 'Face Identity SpikingVGG16bn_LIF_T4_vggface/'), 
                layers=layers, neurons=neurons)
    
    metrics_list = ['euclidean', 'pearson']
    
    selectivity_additional_analyzer.selectivity_analysis_similarity_metrics(metrics_list)
                                                                                         

