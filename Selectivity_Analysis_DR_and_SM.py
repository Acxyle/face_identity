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
from joblib import Parallel, delayed
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
import utils_similarity


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
class Selectiviy_Analysis_SM():
    """
        the Euclidean distance and Pearsons' Correlation should be merged in one function but with different 
        args to produce different metrics, so that in future can add other values for RSA
    """
    def __init__(self, 
                 root='/Identity_Spikingjelly_VGG_Results/',
                 num_samples=10, num_classes=50, layers=None, neurons=None,  data_name=None):
        #super().__init__()
        
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
            
        self.model_structure = root.split('/')[-1].split(' ')[-1]
        
    
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
        for metric in metrics:     # for each metric
            
            metric_folder = os.path.join(self.dest_DSM, f'{metric}')
            utils_.make_dir(metric_folder)
            
            metric_dict = self.selectivity_analysis_similarity(metric, in_layer_plot=True)
            
            # ----- plot
            if metric == 'euclidean':
                self.selectivity_analysis_plot(metric, metric_dict, sup_v=None)
            elif metric == 'pearson':
                self.selectivity_analysis_plot(metric, metric_dict, sup_v=(0,1))
               
    def selectivity_analysis_similarity(self, metric, in_layer_plot:bool=False):
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        metric_folder = os.path.join(self.dest_DSM, f'{metric}')
        utils_.make_dir(metric_folder)
        
        layers = self.layers[:]
        
        dict_path = os.path.join(metric_folder, f'{metric}.pkl')
        
        if os.path.exists(dict_path):
            
            metric_dict = utils_.pickle_load(dict_path)
        
        else:
            
            print(f'[Codinfo] Executing selectivity_analysis_metric [{metric}]...')
            
            # ----- load different types of units
            self.Sort_dict = utils_.pickle_load(os.path.join(self.dest, 'Encode/Sort_dict.pkl'))
            
            metric_dict = {}     # use a dict to store the info of each layer
            
            tqdm_bar = tqdm(total=len(layers), desc='NN_DM')
            
            for layer in layers:     # for each layer
                
                feature = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))     # (500, num_units)
                
                mean_FR = self.calculate_mean_FR(feature)
                sorted_idx = utils_.lexicographic_order(self.num_classes)     # correct labels
                mean_FR = self.restore_order(mean_FR, sorted_idx)     # (50, num_units)
                # ----- 0. add type of all neuron into the dict
                # --- 1 - 10
                units_type_dict = self.Sort_dict[layer]['advanced_type']
                # --- 11
                units_type_dict.update({'all': np.arange(mean_FR.shape[1])})
                # --- 12 - 13
                units_type_dict.update({'selective': 
                                        np.concatenate(np.array([self.Sort_dict[layer]['advanced_type']['sensitive_si'],
                                        self.Sort_dict[layer]['advanced_type']['sensitive_wsi'],
                                        self.Sort_dict[layer]['advanced_type']['sensitive_mi'],
                                        self.Sort_dict[layer]['advanced_type']['sensitive_wmi']], dtype=object))
                                        })
                
                units_type_dict.update({'non_selective': 
                                        np.concatenate(np.array([self.Sort_dict[layer]['advanced_type']['sensitive_non_encode'],
                                        self.Sort_dict[layer]['advanced_type']['non_sensitive_si'],
                                        self.Sort_dict[layer]['advanced_type']['non_sensitive_wsi'],
                                        self.Sort_dict[layer]['advanced_type']['non_sensitive_mi'],
                                        self.Sort_dict[layer]['advanced_type']['non_sensitive_wmi'],
                                        self.Sort_dict[layer]['advanced_type']['non_sensitive_non_encode']], dtype=object))
                                        })
                
                # --- 14 - 15
                units_type_dict.update({'sensitive': 
                                        np.concatenate(np.array([self.Sort_dict[layer]['advanced_type']['sensitive_si'],
                                        self.Sort_dict[layer]['advanced_type']['sensitive_wsi'],
                                        self.Sort_dict[layer]['advanced_type']['sensitive_mi'],
                                        self.Sort_dict[layer]['advanced_type']['sensitive_wmi'],
                                        self.Sort_dict[layer]['advanced_type']['sensitive_non_encode']], dtype=object))
                                        })
                
                units_type_dict.update({'non_sensitive': 
                                        np.concatenate(np.array([self.Sort_dict[layer]['advanced_type']['non_sensitive_si'],
                                        self.Sort_dict[layer]['advanced_type']['non_sensitive_wsi'],
                                        self.Sort_dict[layer]['advanced_type']['non_sensitive_mi'],
                                        self.Sort_dict[layer]['advanced_type']['non_sensitive_wmi'],
                                        self.Sort_dict[layer]['advanced_type']['non_sensitive_non_encode']], dtype=object))
                                        })
                
                # --- 16 - 17
                units_type_dict.update({'encode': 
                                        np.concatenate(np.array([self.Sort_dict[layer]['basic_type']['si_idx'],
                                        self.Sort_dict[layer]['basic_type']['wsi_idx'],
                                        self.Sort_dict[layer]['basic_type']['mi_idx'],
                                        self.Sort_dict[layer]['basic_type']['wmi_idx']], dtype=object))
                                        })
                
                units_type_dict.update({'non_encode': self.Sort_dict[layer]['basic_type']['non_encode_idx']})
                
                # --- 18 -19
                units_type_dict.update({'all_sensitive_si': 
                                        np.concatenate(np.array([self.Sort_dict[layer]['advanced_type']['sensitive_si'],
                                        self.Sort_dict[layer]['advanced_type']['sensitive_wsi']], dtype=object))
                                        })
                
                units_type_dict.update({'all_sensitive_mi': 
                                        np.concatenate(np.array([self.Sort_dict[layer]['advanced_type']['sensitive_mi'],
                                        self.Sort_dict[layer]['advanced_type']['sensitive_wmi']], dtype=object))
                                        })
    
                # --- 1. generate similarity metrics
                pl = Parallel(n_jobs=int(os.cpu_count()/3))(delayed(utils_similarity.selectivity_analysis_calculation)(metric, mean_FR[:, units_type_dict[type_].astype(int)]) for type_ in [_ for _ in units_type_dict.keys()])
                metric_type_dict = {type_: pl[idx] for idx, type_ in enumerate([_ for _ in units_type_dict.keys()])}
                metric_dict[layer] = metric_type_dict
                
                # --- 2. in layer plot
                if in_layer_plot:
                    if metric == 'euclidean':     # for any values
                        self.selectivity_analysis_similarity_in_layer_plot(layer, metric, metric_type_dict)     
                    elif metric == 'pearson':     # for similarity values
                        self.selectivity_analysis_similarity_in_layer_plot(layer, metric, metric_type_dict, (0,1))    
                
                tqdm_bar.update(1)
                # -----
                
            utils_.pickle_dump(dict_path, metric_dict)
            #savemat(os.path.join(metric_folder, f'{metric}.mat'), metric_dict)
            
        return metric_dict
    
    #FIXME
    def selectivity_analysis_similarity_in_layer_plot(self, layer, metric, metric_type_dict, v:tuple=None):
        
        metric_folder = os.path.join(self.dest_DSM, f'{metric}')
        utils_.make_dir(metric_folder)
        
        plot_folder = os.path.join(metric_folder, 'in_layer_Figures')
        utils_.make_dir(plot_folder)
        
        metric_values_pool = np.array([metric_type_dict[_]['matrix'] for _ in metric_type_dict.keys() if metric_type_dict[_] != None])
        
        if v == None:
            vmin = np.min(metric_values_pool)
            vmax = np.max(metric_values_pool)
        else:
            vmin = v[0]
            vmax = v[1]
        
        types = len(metric_type_dict)
        
        fig, ax = plt.subplots(1, types, figsize=(types*5,5))

        for idx, key in enumerate(metric_type_dict.keys()):
            
            ax[idx].set_title(f'{key}')
            
            if metric_type_dict[key] != None:
                ax[idx].imshow(metric_type_dict[key]['matrix'], origin='lower', vmin=vmin, vmax=vmax)
                
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
        
    def selectivity_analysis_plot(self, metric, metric_dict, sup_v:tuple=None):

        plt.rcParams.update({"font.size": 30})
        
        # ----- not applicable for all metrics
        metric_dict_ = {_:{__: metric_dict[_][__]['matrix'] if metric_dict[_][__] != None else None for __ in metric_dict[_].keys()} for _ in metric_dict.keys()}     # assemble all types of all layers
        metric_dict_pool = np.concatenate([_ for _ in [np.concatenate([metric_dict_[key][__] for __ in metric_dict_[key].keys() if metric_dict_[key][__] is not None]).reshape(-1) for key in metric_dict_.keys()]])   # in case of inhomogeneous shape
        
        if sup_v is None:
            sup_vmin = np.min(metric_dict_pool)
            sup_vmax = np.max(metric_dict_pool)
        else:
            sup_vmin = sup_v[0]
            sup_vmax = sup_v[1]
        
        tqdm_bar = tqdm(total=6, desc='Plot')
        
        # -----
        self.selectivity_analysis_plot_single(metric,  metric_dict, sup_vmin, sup_vmax, layer_type='imaginary_neuron', plot_type='')
        tqdm_bar.update(1)
        
        # -----
        self.selectivity_analysis_plot_single(metric,  metric_dict, sup_vmin, sup_vmax, layer_type='imaginary_neuron', plot_type='suplim')
        tqdm_bar.update(1)
        
        # -----
        self.selectivity_analysis_plot_single(metric,  metric_dict, sup_vmin, sup_vmax, layer_type='imaginary_neuron', plot_type='norm')
        tqdm_bar.update(1)
        
        # -----
        self.selectivity_analysis_plot_single(metric,  metric_dict, sup_vmin, sup_vmax, layer_type='all', plot_type='')
        tqdm_bar.update(1)
        
        # -----
        self.selectivity_analysis_plot_single(metric,  metric_dict, sup_vmin, sup_vmax, layer_type='all', plot_type='suplim')
        tqdm_bar.update(1)
        
        # -----
        self.selectivity_analysis_plot_single(metric,  metric_dict, sup_vmin, sup_vmax, layer_type='all', plot_type='norm')
        tqdm_bar.update(1)
    
    def selectivity_analysis_plot_single(self, metric, metric_dict, sup_vmin, sup_vmax, cmap='turbo', layer_type:str=None, plot_type:str=None):
        
        metric_folder = os.path.join(self.dest_DSM, f'{metric}')
        utils_.make_dir(metric_folder)
        
        plot_folder = os.path.join(metric_folder, 'Figures')
        utils_.make_dir(plot_folder)
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        if layer_type == None or 'all' in layer_type.lower():
            layers = self.layers
        elif 'neuron' in layer_type.lower() or 'unit' in layer_type.lower():
            _, layers, _ = utils_.imaginary_neurons_vgg(self.layers)
        
        num_types = len(metric_dict[list(metric_dict.keys())[0]])
        
        fig, ax = plt.subplots(num_types, len(layers), figsize=(3*len(layers), 3*num_types))

        for idx, layer in enumerate(layers):     # for each layer
            
            for idx_, type_ in enumerate(metric_dict[layer].keys()):     # for each type of cells
                
                if idx_ == 0:
                    ax[idx_, idx].set_title(layer)
                    
                if idx == 0:
                    ax[idx_, idx].set_ylabel(type_)
                
                # 1. by default, the vmin and vmax normalize the vlim of different types of units
                if plot_type == None or plot_type == '':
                    if metric_dict[layer][type_] is not None:
                        if metric_dict[layer][type_]['num_units'] != 0:
                            
                            # FIXME, looks like this method is not reasonable
                            metric_layer_pool = np.concatenate([_['vector'] for _ in metric_dict[layer].values() if _ is not None])
                            vmin = np.min(metric_layer_pool)
                            vmax = np.max(metric_layer_pool)
                            
                            ax[idx_, idx].imshow(metric_dict[layer][type_]['matrix'], origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
                            ax[idx_, idx].set_xlabel(f"{metric_dict[layer][type_]['num_units']/(self.neurons[self.layers.index(layer)])*100:.2f}%")
                        else:
                            ax[idx_, idx].axis('off')
                    else:
                        ax[idx_, idx].axis('off')
                        
                # 2. ylim setted by exterier values
                elif plot_type == 'suplim':
                    if metric_dict[layer][type_] is not None:
                        if metric_dict[layer][type_]['num_units'] != 0:
                            ax[idx_, idx].imshow(metric_dict[layer][type_]['matrix'], origin='lower', aspect='auto', vmin=sup_vmin, vmax=sup_vmax, cmap=cmap)
                            ax[idx_, idx].set_xlabel(f"{metric_dict[layer][type_]['num_units']/(self.neurons[self.layers.index(layer)])*100:.2f}%")
                            
                            cax = fig.add_axes([1.01, 0.1, 0.01, 0.75])
                            norm = plt.Normalize(vmin=sup_vmin, vmax=sup_vmax)
                            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                        else:
                            ax[idx_, idx].axis('off')
                    else:
                        ax[idx_, idx].axis('off')
                        
                # 3. only care about one type and the relationship between classes
                elif plot_type == 'norm':
                    if metric_dict[layer][type_] is not None:
                        if metric_dict[layer][type_]['num_units'] != 0:
                            
                            vmin = np.min(metric_dict[layer][type_]['vector'])
                            vmax = np.max(metric_dict[layer][type_]['vector'])
                            vnorm = vmax - vmin    
                            
                            if vnorm == 0:     # when vmin = vmin -> all values are equal -> (usually) all distance values = 0 -> all raw values = 0
                                ax[idx_, idx].set_xlabel(f"{metric_dict[layer][type_]['num_units']/(self.neurons[self.layers.index(layer)])*100:.2f}%")
                                
                            else:
                                ax[idx_, idx].imshow((metric_dict[layer][type_]['matrix']-vmin)/vnorm, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
                                ax[idx_, idx].set_xlabel(f"{metric_dict[layer][type_]['num_units']/(self.neurons[self.layers.index(layer)])*100:.2f}%")
                                
                            cax = fig.add_axes([1.01, 0.1, 0.01, 0.75])
                            norm = plt.Normalize(vmin=0, vmax=1)
                            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                        else:
                            ax[idx_, idx].axis('off')
                    else:
                        ax[idx_, idx].axis('off')
                
                ax[idx_, idx].set_xticks([])
                ax[idx_, idx].set_yticks([])
                
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.1, hspace=0.1)
        
        if plot_type == None or plot_type == '':
            fig.suptitle(f'{self.model_structure} | {metric} | {layer_type} | normalized for all types in one layer', y=1.015, fontsize=50)
        elif plot_type == 'suplim':
            fig.suptitle(f'{self.model_structure} | {metric} | {layer_type} | normalized for all types in all layer', y=1.015, fontsize=50)
        elif plot_type == 'norm':
            fig.suptitle(f'{self.model_structure} | {metric} | {layer_type} | normalized for each type in each layer (only care classes similarity) | balnk: all values = 0', y=1.015, fontsize=50)
            
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            fig.tight_layout()
            fig.savefig(os.path.join(plot_folder, f'{layer_type}_{plot_type}.png'), bbox_inches='tight', dpi=100)
            plt.close()

    def restore_order(self, mean_FR, sorted_idx):
        mean_FR = [[mean_FR[_,:], sorted_idx[_]] for _ in range(self.num_classes)]
        mean_FR = sorted(mean_FR, key=lambda x:x[1])
        mean_FR = np.array([mean_FR[_][0] for _ in range(self.num_classes)])
        
        return mean_FR
    
    def calculate_mean_FR(self, matrix):            
        return np.array([matrix[_*self.num_samples:(_+1)*self.num_samples, :] for _ in range(self.num_classes)]).mean(axis=1)
  
    
if __name__ == '__main__':

    model_name = 'vgg16_bn'
    
    model_ = vgg.__dict__[model_name](num_classes=50)
    layers, neurons, shapes = utils_.generate_vgg_layers_list_ann(model_, model_name)

    root_dir = '/home/acxyle-workstation/Downloads/'

    selectivity_additional_analyzer = Selectiviy_Analysis_SM(root=os.path.join(root_dir, 'Face Identity VGG16bn'), 
                layers=layers, neurons=neurons)
    
    metrics_list = ['euclidean', 'pearson']
    
    selectivity_additional_analyzer.selectivity_analysis_similarity_metrics(metrics_list)
                                                                                         

