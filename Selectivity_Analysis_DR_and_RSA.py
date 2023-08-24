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


class Selectiviy_Analysis_DR_and_RSA():
    
    def __init__(self, 
                 root='/Identity_Spikingjelly_VGG_Results/',
                 dest='/Identity_Spikingjelly_VGG_Neuron/',
                 num_samples=10, num_classes=50, data_name='', layers=None, neurons=None, status=False):
        
        if layers == None or neurons == None:
            raise RuntimeError('[Codwarning] invalid layers and neurons')
            
        self.layers = layers
        self.neurons = neurons
        
        self.root = root
        self.dest = dest
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.data_name = data_name
        
        if status:     # flag for local test
            self.selectivity_analysis_Tsne(self)
            self.selectivity_analysis_distance(self)
        
    #FIXME
    def selectivity_analysis_tsne(self, parallel=False, verbose=False):
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        print('[Codinfo] Executing selectivity_analysis_Tsne...')
        label = utils_.makeLabels(self.num_samples, self.num_classes)
        
        save_path = os.path.join(self.dest, 'TSNE')
        utils_.make_dir(save_path)
        
        save_path_fig = os.path.join(save_path, 'Figures')
        utils_.make_dir(save_path_fig)
        save_path_result = os.path.join(save_path, 'Results')
        utils_.make_dir(save_path_result)
        
        tsne_x1_list = []
        tsne_y1_list = []
        tsne_x2_list = []
        tsne_y2_list = []
        tsne_all_list = []
        tsne_ID_list = []
        tsne_nonID_list = []
        title_list = []
        mask_id_list = []
        mask_non_id_list = []
        
        valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if not item[1].startswith('not')])
        markers = valid_markers + valid_markers[:self.num_classes - len(valid_markers)]
        
        #=== under construction
        # ----- sequantial calculation
        for layer in tqdm(self.layers, desc='TSNE Sequential'):
            
            feature = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))
            
            mask_id, mask_non_id = self.generate_masks(self.dest, layer, feature.shape[1])
            mask_id_list.append(len(mask_id))
            mask_non_id_list.append(len(mask_non_id))
            
            self.tsne_layer_calculation(layer, feature, mask_id, mask_non_id, label, markers, save_path_result, save_path_fig, 
                                                 tsne_x1_list, tsne_y1_list, tsne_x2_list, tsne_y2_list, 
                                                 tsne_all_list, tsne_ID_list, tsne_nonID_list, title_list)    
        # ----- parallel calculation
        # working...
        # =====
        
        # in-model comparison
        fig_sup, ax_sup = plt.subplots(len(self.layers), 3, figsize=(18, 6*len(self.layers)), dpi=100)
        
        tsne_x1_sup = min(tsne_x1_list)
        tsne_y1_sup = min(tsne_y1_list)
        tsne_x2_sup = max(tsne_x2_list)
        tsne_y2_sup = max(tsne_y2_list)
        
        dx1, dy1 = 5, 0.25
        
        for idx, layer in enumerate(self.layers):
            
            tsne_all = tsne_all_list[idx]
            tsne_ID = tsne_ID_list[idx]
            tsne_nonID = tsne_nonID_list[idx]
            title = title_list[idx]
            
            if ax_sup.ndim == 1:
                self.tsne_plot(ax_sup[0], tsne_all, [*mask_id, *mask_non_id], label, markers, title+' all', layer, tsne_x1_sup, tsne_y1_sup, tsne_x2_sup, tsne_y2_sup)
                self.tsne_plot(ax_sup[1], tsne_ID, mask_id, label, markers, title+' selective', layer, tsne_x1_sup, tsne_y1_sup, tsne_x2_sup, tsne_y2_sup)
                self.tsne_plot(ax_sup[2], tsne_nonID, mask_non_id, label, markers, title+' non-selective', layer, tsne_x1_sup, tsne_y1_sup, tsne_x2_sup, tsne_y2_sup)
                
                trans1 = ax_sup[0].transAxes + ScaledTranslation(dx1, dy1, fig_sup.dpi_scale_trans)
                trans2 = ax_sup[1].transAxes + ScaledTranslation(dx1, dy1, fig_sup.dpi_scale_trans)
                
                ax_sup[0].text(0, 1, '{}/{}'.format(mask_id_list[idx], mask_id_list[idx]+mask_non_id_list[idx]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2},  
                         transform=trans1, verticalalignment='top', horizontalalignment='left')
                ax_sup[1].text(0, 1, '{}/{}'.format(mask_non_id_list[idx], mask_id_list[idx]+mask_non_id_list[idx]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2}, 
                         transform=trans2, verticalalignment='top', horizontalalignment='left')
                
            else:
                self.tsne_plot(ax_sup[idx, 0], tsne_all, [*mask_id, *mask_non_id], label, markers, title+' all', layer, tsne_x1_sup, tsne_y1_sup, tsne_x2_sup, tsne_y2_sup)
                self.tsne_plot(ax_sup[idx, 1], tsne_ID, mask_id, label, markers, title+' selective', layer, tsne_x1_sup, tsne_y1_sup, tsne_x2_sup, tsne_y2_sup)
                self.tsne_plot(ax_sup[idx, 2], tsne_nonID, mask_non_id, label, markers, title+' non-selective', layer, tsne_x1_sup, tsne_y1_sup, tsne_x2_sup, tsne_y2_sup)
            
                trans1 = ax_sup[idx, 0].transAxes + ScaledTranslation(dx1, dy1, fig_sup.dpi_scale_trans)
                trans2 = ax_sup[idx, 1].transAxes + ScaledTranslation(dx1, dy1, fig_sup.dpi_scale_trans)
                
                ax_sup[idx, 0].text(0, 1, '{}/{}'.format(mask_id_list[idx], mask_id_list[idx]+mask_non_id_list[idx]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2},  
                         transform=trans1, verticalalignment='top', horizontalalignment='left')
                ax_sup[idx, 1].text(0, 1, '{}/{}'.format(mask_non_id_list[idx], mask_id_list[idx]+mask_non_id_list[idx]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2}, 
                         transform=trans2, verticalalignment='top', horizontalalignment='left')
        
        fig_sup.savefig(save_path_fig + '/tsne_all.png', bbox_inches='tight', dpi=100)     # change dpi whilst data change
        fig_sup.savefig(save_path_fig + '/tsne_all.eps', bbox_inches='tight', dpi=100, format='eps')
        plt.close()
        
    def tsne_layer_calculation(self, layer, feature, mask_id, mask_non_id, label, markers, save_path_result, save_path_fig, 
                                        tsne_x1_list, tsne_y1_list, tsne_x2_list, tsne_y2_list,
                                        tsne_all_list, tsne_ID_list, tsne_nonID_list, title_list, verbose=False):
    
        perplexity_ID = self.calculate_perplexity(mask_id)
        perplexity_nonID = self.calculate_perplexity(mask_non_id)
                
        if verbose:
            print('layer: {}, mask_id: {}, nonmask_id: {}, perplexity: {:.3f} {:.3f}'.format(layer, len(mask_id), len(mask_non_id), perplexity_ID, perplexity_nonID))
        
        # in-layer comparison
        fig, ax = plt.subplots(1, 3, figsize=(18,6), dpi=100)
        
        plt.text(0.4, 0.925, '{}/{}'.format(len(mask_id), feature.shape[1]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2}, transform=fig.transFigure, verticalalignment='top', horizontalalignment='left')
        plt.text(0.675, 0.925, '{}/{}'.format(len(mask_non_id), feature.shape[1]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2}, transform=fig.transFigure, verticalalignment='top', horizontalalignment='left')
        
        tsne_x1, tsne_y1, tsne_x2, tsne_y2, tsne_all, tsne_ID, tsne_nonID, title = self.tsne_layer_process(ax, feature, mask_id, perplexity_ID, mask_non_id, perplexity_nonID, label, markers, layer, save_path_result, layer)
        
        tsne_x1_list.append(tsne_x1)
        tsne_y1_list.append(tsne_y1)
        tsne_x2_list.append(tsne_x2)
        tsne_y2_list.append(tsne_y2)
        tsne_all_list.append(tsne_all)
        tsne_ID_list.append(tsne_ID)
        tsne_nonID_list.append(tsne_nonID)
        title_list.append(title)
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            fig.savefig(save_path_fig + f'/tsne_{layer}.png', bbox_inches='tight', dpi=100)     # change dpi whilst data change
            fig.savefig(save_path_fig + f'/tsne_{layer}.eps', bbox_inches='tight', dpi=100, format='eps')
            plt.close()  
            
    def obtain_tsne_range(self, tsne):     # return the max and min values of x and y axes
        return min(tsne[:, 0]), min(tsne[:, 1]), max(tsne[:, 0]), max(tsne[:, 1])
    
    def tsne_save(self, save_path, file_name, file):
        """
            this function saves tsne file in different format,    
            need to manually modify the code to enable/disable different types of saved file
        """
        if file is not None:
            savemat(save_path+f'/tSNE_{file_name}.mat', {f'{file_name}':file})      # for matlab
            utils_.pickle_dump(save_path+f'/tSNE_{file_name}.pkl', file)     # for python
    
    def tsne_layer_process(self, ax, feature, mask_id, perplexity_ID, mask_non_id, perplexity_nonID, label, markers, layer, save_path_result, title):
        """
            this function calculate the tSNE and saved results, also plot the reduced feature map
        """
        tsne_all = self.tsne_layer_process_single(feature, max(perplexity_ID, perplexity_nonID))      # tSNE from all units
        self.tsne_save(save_path_result, layer+'_all', tsne_all)
        
        tsne_ID = self.tsne_layer_process_single(feature, perplexity_ID, mask_id)     # tSNE from id_selective unit only
        self.tsne_save(save_path_result, layer+'_id_selective', tsne_ID)
        
        tsne_nonID = self.tsne_layer_process_single(feature, perplexity_nonID, mask_non_id)     # tSNE from non_id_selective unit only
        self.tsne_save(save_path_result, layer+'_non_id_selective', tsne_nonID)
        
        if tsne_ID is not None and tsne_nonID is not None:
            if tsne_ID.shape[1] == 2 and tsne_nonID.shape[1] == 2:
                tsne_x1 = min(min(tsne_ID[:, 0]), min(tsne_nonID[:, 0]))
                tsne_y1 = min(min(tsne_ID[:, 1]), min(tsne_nonID[:, 1]))
                tsne_x2 = max(max(tsne_ID[:, 0]), max(tsne_nonID[:, 0]))
                tsne_y2 = max(max(tsne_ID[:, 1]), max(tsne_nonID[:, 1]))
            elif tsne_ID.shape[1] == 2 and tsne_nonID.shape[1] == 1:
                tsne_x1, tsne_y1, tsne_x2, tsne_y2 = self.obtain_tsne_range(tsne_ID)
            elif tsne_ID.shape[1] == 1 and tsne_nonID.shape[1] == 2:
                tsne_x1, tsne_y1, tsne_x2, tsne_y2 = self.obtain_tsne_range(tsne_nonID)
        elif tsne_nonID is None:
            tsne_x1, tsne_y1, tsne_x2, tsne_y2 = self.obtain_tsne_range(tsne_ID)
        elif tsne_ID is None:
            tsne_x1, tsne_y1, tsne_x2, tsne_y2 = self.obtain_tsne_range(tsne_nonID)
        else:
            raise RuntimeError('[Codinfo] some unexpected cases happened')
        
        self.tsne_plot(ax[0], tsne_all, [*mask_id, *mask_non_id], label, markers, title+' all', layer, tsne_x1, tsne_y1, tsne_x2, tsne_y2)
        self.tsne_plot(ax[1], tsne_ID, mask_id, label, markers, title+' selective', layer, tsne_x1, tsne_y1, tsne_x2, tsne_y2)
        self.tsne_plot(ax[2], tsne_nonID, mask_non_id, label, markers, title+' non-selective', layer, tsne_x1, tsne_y1, tsne_x2, tsne_y2)
        
        return tsne_x1, tsne_y1, tsne_x2, tsne_y2, tsne_all, tsne_ID, tsne_nonID, title
        
    def tsne_layer_process_single(self, feature, perplexity, mask=None):
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
        if mask == None:
            if np.std(np.array(feature)) != 0.:     # make sure all faetures are not identical
                return TSNE(perplexity=perplexity, n_jobs=-1).fit_transform(feature)
            else:     # all values in different units are identical
                raise RuntimeError('[Coderror] all values in feature map are identical')
        else:
            if len(mask) == 0:
                return
            elif len(mask) == 1:
                tsne = feature[:, mask]
                return tsne
            else:
                if np.std(np.array(feature[:, mask])) != 0.:     # [notice] make sure all faetures are not identical
                    tsne = TSNE(perplexity=perplexity, n_jobs=-1).fit_transform(feature[:,mask]) 
                    return tsne
                else:     # all values in different units are identical
                    return
        
    def tsne_plot(self, ax, tsne, mask, label, markers, title, layer, tsne_x1, tsne_y1, tsne_x2, tsne_y2):
        
        width = tsne_x2-tsne_x1
        height = tsne_y2-tsne_y1
        for i in range(self.num_classes):
            self.tsne_plot_scatter(ax, tsne, i, label, markers, mask, tsne_x1, tsne_y1, tsne_x2, tsne_y2)
            
        ax.set_title(self.data_name + f' {title}')
        ax.grid(True)
        ax.vlines(0, tsne_x1-0.5*width, tsne_x1+1.5*width, colors='gray',  linestyles='--', linewidth=2.0)
        ax.hlines(0, tsne_y1-0.5*height, tsne_y1+1.5*height, colors='gray',  linestyles='--', linewidth=2.0)
        ax.set_xlim((tsne_x1-0.025*width, tsne_x1+1.025*width))
        ax.set_ylim((tsne_y1-0.025*height, tsne_y1+1.025*height))
        
    def tsne_plot_scatter(self, ax, tsne, i, label, markers, mask, tsne_x1, tsne_y1, tsne_x2, tsne_y2):
        
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
    
    def generate_masks(self, dest, layer, feature_col):
        
        sensitive_unit_idx = np.loadtxt(dest + layer + '-neuronIdx.csv', delimiter=',')     # [warning] simple call
        if sensitive_unit_idx.size == 1:
            sensitive_unit_idx = np.array([sensitive_unit_idx])
        sensitive_unit_idx = list(map(int, sensitive_unit_idx))
        all_idx = [i for i in range(feature_col)]
        non_sensitive_unit_idx = list(set(all_idx)-set(sensitive_unit_idx))
        
        return sensitive_unit_idx, non_sensitive_unit_idx
    
    def calculate_perplexity(self, mask):
        
        if len(mask) >= 1:     # if units number > 0
            perplexity = min(math.sqrt(len(mask)), self.num_classes*self.num_samples-1)     # min(square_root(unit numbers), 499)
        else:     # no unit
            perplexity = 1e-9
        return perplexity

    #FIXME distance scale
    def selectivity_analysis_distance(self):
        print('[Codinfo] Executing selectivity_analysis_distance...')
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        dest = self.dest+'Distance/'
        utils_.make_dir(dest)  
        
        dis_dict = {}
        
        for layer in tqdm(self.layers):     # for each layer
        
            feature = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))

            mean_FR = self.calculate_mean_FR(feature)
            sorted_idx = utils_.lexicographic_order(self.num_classes)
            mean_FR = self.restore_order(mean_FR, sorted_idx)
            
            mask_id, mask_non_id = self.generate_masks(self.dest, layer, feature.shape[1])
        
            dist_avg = pdist(mean_FR, 'euclidean')
            m = squareform(dist_avg)
            
            dist_avg_ID = pdist(mean_FR[:, mask_id], 'euclidean')
            m_i = squareform(dist_avg_ID)
            
            dist_avg_NonID = pdist(mean_FR[:, mask_non_id], 'euclidean')
            m_n = squareform(dist_avg_NonID)
            
            dis_dict[layer]={'all': m, 
                             'id_selective': m_i,
                             'non_id_selective': m_n}
        
            vmax = max(m.max(), m_i.max(), m_n.max())
            vmin = min(m.min(), m_i.min(), m_n.min())
            
            fig, axes = plt.subplots(1, 3, figsize=((30, 10)))
            cbar_ax = fig.add_axes([.91, .1, .03, .8])
        
            sn.heatmap(m, ax=axes[0], cbar=0, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
            axes[0].set_title('all neurons')
            sn.heatmap(m_i, ax=axes[1], cbar=1, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
            axes[1].set_title('identity selective neurons')
            sn.heatmap(m_n, ax = axes[2], cbar=2, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
            axes[2].set_title('non identity selective neurons')
        
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                fig.tight_layout(rect=[0, 0, .9, 1])
                plt.title(layer)
                plt.savefig(dest + layer+'-EDistance.png', bbox_inches='tight', dpi=100)
                plt.close()
        
        # [notice] in fact, the Correlation uses the results of pdist()
        savemat(os.path.join(dest, 'distance.mat'), dis_dict)
        utils_.pickle_dump(os.path.join(dest, 'distance.pkl'), dis_dict)
    
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

    neuron_ = neuron.LIFNode

    spiking_model = spiking_resnet.__dict__['spiking_resnet18'](spiking_neuron=neuron_, num_classes=50, surrogate_function=surrogate.ATan(), detach_reset=True, mode='feature')
    functional.set_step_mode(spiking_model, step_mode='m') 
    layers, neurons, shapes = utils_.generate_resnet_layers_list(spiking_model, 'spiking_resnet18')
    
    layers_ = [i for i in layers if 'neuron' in i or 'fc' in i or 'pool' in i]
    index_ = [layers.index(i) for i in layers_]
    neurons_ = [neurons[i] for i in index_]
    layers = layers_
    neurons = neurons_
    
    root_dir = '/media/acxyle/Data/ChromeDownload/'

    selectivity_additional_analyzer = Selectiviy_Analysis_DR_and_RSA(
                root=os.path.join(root_dir, 'Identity_SpikingResnet18_LIF_CelebA2622_Results/'), 
                dest=os.path.join(root_dir, 'Identity_SpikingResnet18_LIF_CelebA2622_Neuron/'), 
                layers=layers, neurons=neurons)
    #selectivity_additional_analyzer.selectivity_analysis_tsne()
    #selectivity_additional_analyzer.selectivity_analysis_distance()
    selectivity_additional_analyzer.selectivity_analysis_correlation()
    

