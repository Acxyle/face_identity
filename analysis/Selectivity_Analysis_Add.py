#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 22:02:44 2023

@author: acxyle

    write this section as class for further modification like importing other analysis algorithm with same structures.
"""


import torch

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

class Selectiviy_Analysis_Additional():
    def __init__(self, root='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Results/',
                 dest='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/',
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
        
        
    
    #FIXME
    def selectivity_analysis_tsne(self, verbose=False):
        
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
        mask_ID_list = []
        mask_nonID_list = []
        
        valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if not item[1].startswith('not')])
        markers = valid_markers + valid_markers[:self.num_classes - len(valid_markers)]
        
        #=== under construction
        # [question] parallel for shallow layers, but shallow layers do not need TSNE analysis, interesting delima
        # ----- parallel calculation
        for layer in tqdm(self.layers):
            
            feature = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))
            
            _, mask_ID, mask_nonID = self.generate_masks(self.dest, layer, feature.shape[1])
            mask_ID_list.append(len(mask_ID))
            mask_nonID_list.append(len(mask_nonID))
            
            self.tsne_layer_calculation_parallel(layer, feature, mask_ID, mask_nonID, label, markers, save_path_result, save_path_fig, 
                                                 tsne_x1_list, tsne_y1_list, tsne_x2_list, tsne_y2_list, 
                                                 tsne_all_list, tsne_ID_list, tsne_nonID_list, title_list)    
        # -----
        
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
                self.tsne_plot(ax_sup[0], tsne_all, [*mask_ID, *mask_nonID], label, markers, title+' all', layer, tsne_x1_sup, tsne_y1_sup, tsne_x2_sup, tsne_y2_sup)
                self.tsne_plot(ax_sup[1], tsne_ID, mask_ID, label, markers, title+' selective', layer, tsne_x1_sup, tsne_y1_sup, tsne_x2_sup, tsne_y2_sup)
                self.tsne_plot(ax_sup[2], tsne_nonID, mask_nonID, label, markers, title+' non-selective', layer, tsne_x1_sup, tsne_y1_sup, tsne_x2_sup, tsne_y2_sup)
                
                trans1 = ax_sup[0].transAxes + ScaledTranslation(dx1, dy1, fig_sup.dpi_scale_trans)
                trans2 = ax_sup[1].transAxes + ScaledTranslation(dx1, dy1, fig_sup.dpi_scale_trans)
                
                ax_sup[0].text(0, 1, '{}/{}'.format(mask_ID_list[idx], mask_ID_list[idx]+mask_nonID_list[idx]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2},  
                         transform=trans1, verticalalignment='top', horizontalalignment='left')
                ax_sup[1].text(0, 1, '{}/{}'.format(mask_nonID_list[idx], mask_ID_list[idx]+mask_nonID_list[idx]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2}, 
                         transform=trans2, verticalalignment='top', horizontalalignment='left')
                
            else:
                self.tsne_plot(ax_sup[idx, 0], tsne_all, [*mask_ID, *mask_nonID], label, markers, title+' all', layer, tsne_x1_sup, tsne_y1_sup, tsne_x2_sup, tsne_y2_sup)
                self.tsne_plot(ax_sup[idx, 1], tsne_ID, mask_ID, label, markers, title+' selective', layer, tsne_x1_sup, tsne_y1_sup, tsne_x2_sup, tsne_y2_sup)
                self.tsne_plot(ax_sup[idx, 2], tsne_nonID, mask_nonID, label, markers, title+' non-selective', layer, tsne_x1_sup, tsne_y1_sup, tsne_x2_sup, tsne_y2_sup)
            
                trans1 = ax_sup[idx, 0].transAxes + ScaledTranslation(dx1, dy1, fig_sup.dpi_scale_trans)
                trans2 = ax_sup[idx, 1].transAxes + ScaledTranslation(dx1, dy1, fig_sup.dpi_scale_trans)
                
                ax_sup[idx, 0].text(0, 1, '{}/{}'.format(mask_ID_list[idx], mask_ID_list[idx]+mask_nonID_list[idx]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2},  
                         transform=trans1, verticalalignment='top', horizontalalignment='left')
                ax_sup[idx, 1].text(0, 1, '{}/{}'.format(mask_nonID_list[idx], mask_ID_list[idx]+mask_nonID_list[idx]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2}, 
                         transform=trans2, verticalalignment='top', horizontalalignment='left')
        
        fig_sup.savefig(save_path_fig + '/tsne_all.png', bbox_inches='tight', dpi=100)     # change dpi whilst data change
        fig_sup.savefig(save_path_fig + '/tsne_all.eps', bbox_inches='tight', dpi=100, format='eps')
        plt.close()
        
    def tsne_layer_calculation_parallel(self, layer, feature, mask_ID, mask_nonID, label, markers, save_path_result, save_path_fig, 
                                        tsne_x1_list, tsne_y1_list, tsne_x2_list, tsne_y2_list,
                                        tsne_all_list, tsne_ID_list, tsne_nonID_list, title_list, verbose=False):
    
        
        
        perplexity_ID = self.calculate_perplexity(mask_ID)
        perplexity_nonID = self.calculate_perplexity(mask_nonID)
                
        if verbose:
            print('layer: {}, mask_ID: {}, nonmask_ID: {}, perplexity: {:.3f} {:.3f}'.format(layer, len(mask_ID), len(mask_nonID), perplexity_ID, perplexity_nonID))
        
        # in-layer comparison
        fig, ax = plt.subplots(1, 3, figsize=(18,6), dpi=100)
        
        plt.text(0.4, 0.925, '{}/{}'.format(len(mask_ID), feature.shape[1]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2}, transform=fig.transFigure, verticalalignment='top', horizontalalignment='left')
        plt.text(0.675, 0.925, '{}/{}'.format(len(mask_nonID), feature.shape[1]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2}, transform=fig.transFigure, verticalalignment='top', horizontalalignment='left')
        
        tsne_x1, tsne_y1, tsne_x2, tsne_y2, tsne_all, tsne_ID, tsne_nonID, title = self.tsne_layer_process(ax, feature, mask_ID, perplexity_ID, mask_nonID, perplexity_nonID, label, markers, layer, save_path_result, layer)
        
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
            
        
    def obtain_tsne_range(self, tsne):
        return min(tsne[:, 0]), min(tsne[:, 1]), max(tsne[:, 0]), max(tsne[:, 1])
    
    #FIXME
    def tsne_save(self, save_path, file_name, file):
        if file is not None:
            savemat(save_path+f'/tSNE_{file_name}.mat', {f'{file_name}':file})      # for matlab
            utils_.pickle_dump(save_path+f'/tSNE_{file_name}.pkl', file)     # for python
    
    def tsne_layer_process(self, ax, feature, mask_ID, perplexity_ID, mask_nonID, perplexity_nonID, label, markers, layer, save_path_result, title):
        """
            this function calculate the tSNE and saved results, also plot the reduced feature map
        """
        tsne_all = self.tsne_layer_process_single(feature, [*mask_ID, *mask_nonID], max(perplexity_ID, perplexity_nonID))      # tSNE from all units
        self.tsne_save(save_path_result, layer+'_all', tsne_all)
        
        tsne_ID = self.tsne_layer_process_single(feature, mask_ID, perplexity_ID)     # tSNE from id_selective unit only
        self.tsne_save(save_path_result, layer+'_id_selective', tsne_ID)
        
        tsne_nonID = self.tsne_layer_process_single(feature, mask_nonID, perplexity_nonID)     # tSNE from non_id_selective unit only
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
        
        self.tsne_plot(ax[0], tsne_all, [*mask_ID, *mask_nonID], label, markers, title+' all', layer, tsne_x1, tsne_y1, tsne_x2, tsne_y2)
        self.tsne_plot(ax[1], tsne_ID, mask_ID, label, markers, title+' selective', layer, tsne_x1, tsne_y1, tsne_x2, tsne_y2)
        self.tsne_plot(ax[2], tsne_nonID, mask_nonID, label, markers, title+' non-selective', layer, tsne_x1, tsne_y1, tsne_x2, tsne_y2)
        
        return tsne_x1, tsne_y1, tsne_x2, tsne_y2, tsne_all, tsne_ID, tsne_nonID, title
        
    def tsne_layer_process_single(self, feature, mask, perplexity):
        if len(mask) == 0:
            return
        elif len(mask) == 1:
            tsne = feature[:, mask]
            return tsne
        else:
            if np.std(np.array(feature[:, mask])) != 0.:     # [notice] make sure all faetures are not identical
                tsne = TSNE(perplexity=perplexity, n_jobs=-1).fit_transform(feature[:, mask]) 
                return tsne
            else:
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
        sig_neuron_idx = np.loadtxt(dest + layer + '-neuronIdx.csv', delimiter=',')     # [warning] simple call
        if sig_neuron_idx.size == 1:
            sig_neuron_idx = np.array([sig_neuron_idx])
        sig_neuron_idx = list(map(int, sig_neuron_idx))
        all_idx = [i for i in range(feature_col)]
        non_sig_neuron_idx = list(set(all_idx)-set(sig_neuron_idx))
        
        return all_idx, sig_neuron_idx, non_sig_neuron_idx
    
    def calculate_perplexity(self, mask):
        """
            a) when performing tSNE to very high dimensional data, there's a lot of problems for algorithms and computing:
                1. curse of dimensionality (human experience about 'distance' may invalid for high dimensional data)
                2. noise
                3. overfitting
                4. interpretability
            thus, it is reasonable to reduce the dimension of the raw feature
            
            b) a commonly used way to reduce the dimension firstly is PCA, for the feature of 500*3m, usually can use PCA to reduce it to 50-200 dimensions (citation) before the tSNE
    
            c) according to Van der Maaten and Hinton, they suggested to try different values of perplexity, to see the trade-off between local and glocal
        """
        if len(mask) >= 1:     # if units number > 0
            perplexity = min(math.sqrt(len(mask)), self.num_classes*self.num_samples-1)     # perplexity is the min(square_root(unit numbers), 499)
            if perplexity == 0.:
                perplexity = 1e-9
        else:
            perplexity = 1e-9
        return perplexity

    #FIXME distance scale
    def selectivity_analysis_distance(self):
        print('[Codinfo] Executing selectivity_analysis_ditance...')
        dest = self.dest+'Distance/'
        utils_.make_dir(dest)  
        
        for layer in tqdm(self.layers):     # for each layer
        
            feature = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))

            avgMatrix = self.avg_across_values(feature)
            _, mask_ID, mask_nonID = self.generate_masks(self.dest, layer, feature.shape[1])
        
            fig, axes = plt.subplots(1, 3, figsize=((30, 10)))
            cbar_ax = fig.add_axes([.91, .1, .03, .8])
        
            dist_avg = pdist(avgMatrix, 'euclidean')
            m = squareform(dist_avg)
            
            dist_avg_ID = pdist(avgMatrix[:, mask_ID], 'euclidean')
            m_i = squareform(dist_avg_ID)
            
            dist_avg_NonID = pdist(avgMatrix[:, mask_nonID], 'euclidean')
            m_n = squareform(dist_avg_NonID)
        
            vmax = max(m.max(), m_i.max(), m_n.max())
            vmin = min(m.min(), m_i.min(), m_n.min())
        
            sn.heatmap(m, ax=axes[0], cbar=0, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
            axes[0].set_title('all neurons')
            sn.heatmap(m_i, ax=axes[1], cbar=1, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
            axes[1].set_title('identity selective neurons')
            sn.heatmap(m_n, ax = axes[2], cbar=2, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
            axes[2].set_title('non identity selective neurons')
        
            fig.tight_layout(rect=[0, 0, .9, 1])
            
            plt.title(layer)
            plt.savefig(dest + layer+'-EDistance.png', bbox_inches='tight', dpi=100)
            plt.close()
        
    
    #FIXME parellel process
    def selectivity_analysis_correlation(self):
        
        print('[Codinfo] Executing selectivity_analysis_correlation...')
        dest = self.dest+'Correlation/'
        utils_.make_dir(dest)
        
        dest_ID, cor_avg_dict, cor_avg_ID_dict, cor_avg_nonID_dict = self.correlation_state(dest, 'ID/')
        #dest_faces, cor_full_dict, cor_full_ID_dict, cor_full_nonID_dict = self.correlation_state(dest, 'faces/')
        
        for layer in tqdm(self.layers):
            
            fullMatrix = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))
            avgMatrix = self.avg_across_values(fullMatrix) 
            
            # [notice] below is the idx correction operation to fix the idx disorder
            #FIXME  avoid dependency on saved file
            tmp_sorted_ID = utils_.pickle_load(os.path.join(os.getcwd(),'tmp_sorted_ID.pkl'))  # x[0] real order, x[1] torch order
            tmp_sorted_ID = sorted(tmp_sorted_ID, key=lambda x: x[1])
            tmp_list_ID = [avgMatrix[i] for i in range(self.num_classes)]
            avgMatrix = self.restore_order(tmp_list_ID, tmp_sorted_ID, self.num_classes)
            
            #tmp_sorted_face = [i for i in tmp_sorted_ID for _ in range(self.num_samples)]
            #tmp_list_face = [fullMatrix[i] for i in range(self.num_classes*self.num_samples)]
            #fullMatrix = self.restore_order(tmp_list_face, tmp_sorted_face, self.num_classes*self.num_samples)

            mask_ID = np.loadtxt(self.dest  + '/' + layer + '-neuronIdx.csv', delimiter=',')
            mask_ID = list(map(int, mask_ID))
            mask_nonID = sorted(list(set(np.arange(fullMatrix.shape[1])) - set(mask_ID)))
            
            cor_avg_dict, cor_avg_ID_dict, cor_avg_nonID_dict = self.correlation_single(avgMatrix, layer, mask_ID, mask_nonID, cor_avg_dict, cor_avg_ID_dict, cor_avg_nonID_dict, dest_ID)
            #cor_full_dict, cor_full_ID_dict, cor_full_nonID_dict = self.correlation_single(fullMatrix, layer, mask_ID, mask_nonID, cor_full_dict, cor_full_ID_dict, cor_full_nonID_dict, dest_faces)
            
        print('[Codinfo] Saving Correlation Matrix...')
        
        self.correlation_save('avg', cor_avg_dict, 'avg_ID', cor_avg_ID_dict, 'avg_nonID', cor_avg_nonID_dict, dest)
        #self.correlation_save('full', cor_full_dict, 'full_ID', cor_full_ID_dict, 'full_nonID', cor_full_nonID_dict, dest)
   
    def correlation_state(self, dest, data_type):
        save_path = dest+data_type
        utils_.make_dir(save_path)
        
        cor_dict = {}
        cor_sensitive_dict = {}
        cor_nonsensitive_dict = {}
        
        return save_path, cor_dict, cor_sensitive_dict, cor_nonsensitive_dict
    
    def correlation_save(self, cor_name, cor_dict, cor_sensitive_name, cor_sensitive_dict, cor_nonsensitive_name, cor_nonsensitive_dict, dest):
        savemat(dest+f'/CorMatrix_{cor_name}.mat', cor_dict)
        savemat(dest+f'/CorMatrix_{cor_sensitive_name}.mat', cor_sensitive_dict)
        savemat(dest+f'/CorMatrix_{cor_nonsensitive_name}.mat', cor_nonsensitive_dict)
    
    def correlation_single(self, matrix, layer, mask_ID, mask_nonID, cor_dict, cor_sensitive_dict, cor_nonsensitive_dict, dest):
        cor, cor_sensitive, cor_nonsensitive = self.correlation_calculate_single(matrix, layer, mask_ID, mask_nonID)
        cor_dict.update({layer: cor})
        cor_sensitive_dict.update({layer: cor_sensitive})
        cor_nonsensitive_dict.update({layer: cor_nonsensitive})
        self.correlation_plot_single(cor, cor_sensitive, cor_nonsensitive, mask_ID, mask_nonID, layer, dest)
        
        return cor_dict, cor_sensitive_dict, cor_nonsensitive_dict
    
    def restore_order(self, list_matrix_restore, class_label_retore, num_classes):
        
        for idx, e in enumerate(list_matrix_restore):
            e = [e]
            e.append(class_label_retore[idx])
            list_matrix_restore[idx] = e
            
        list_matrix_restore = sorted(list_matrix_restore,key=lambda x:int(x[1][0]))
        list_matrix_restore = [list_matrix_restore[i][0] for i in range(num_classes)]
        list_matrix_restore = np.array(list_matrix_restore)
        
        matrix = list_matrix_restore
        
        return matrix
   
    def correlation_calculate_single(self, matrix, layer, mask_ID, mask_nonID):
        
        cor = np.corrcoef(matrix)
        cor_ID = np.corrcoef(matrix[:, mask_ID])
        cor_nonID = np.corrcoef(matrix[:, mask_nonID])
        
        return cor, cor_ID, cor_nonID
        
    def correlation_plot_single(self, cor, cor_ID, cor_nonID, mask_ID, mask_nonID, layer, save_path):
        
        fig, axes = plt.subplots(1, 3, figsize=((30, 10)))
        cbar_ax = fig.add_axes([.91, .1, .03, .8])
    
        vmax = max(cor.max(), cor_ID.max(), cor_nonID.max())
        vmin = min(cor.min(), cor_ID.min(), cor_nonID.min())
    
        sn.heatmap(cor, ax=axes[0], cmap='jet', cbar=0, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
        axes[0].set_title(f'all neurons ({len(mask_ID)+len(mask_nonID)})')
        sn.heatmap(cor_ID, ax=axes[1], cmap='jet',  cbar=1, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
        axes[1].set_title(f'identity selective neurons ({len(mask_ID)})')
        sn.heatmap(cor_nonID, ax = axes[2], cmap='jet',  cbar=2, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
        axes[2].set_title(f'non identity selective neurons ({len(mask_nonID)})')
    
        fig.tight_layout(rect=[0, 0, .9, 1])
        
        plt.title(layer)
        plt.savefig(save_path+layer+'-Correlation.png', bbox_inches='tight', dpi=100)
        plt.close()

    def avg_across_values(self, matrix):            
        avg_ = np.array([matrix[i*self.num_samples:(i+1)*self.num_samples, :] for i in range(self.num_classes)]).mean(axis=1)
        return avg_ 
   
    
if __name__ == '__main__':
    
    model_name = 'vgg16_bn'
    
    model_ = vgg.__dict__[model_name](num_classes=50)
    layers, neurons, shapes = utils_.generate_vgg_layers_list_ann(model_, model_name)
   
    root_dir = '/home/acxyle-workstation/Downloads/'

    selectivity_additional_analyzer = Selectiviy_Analysis_Additional(
                root=os.path.join(root_dir, 'Face Identity SpikingVGG16bn_LIF_T16_CelebA2622/Features'), 
                dest=os.path.join(root_dir, 'Face Identity SpikingVGG16bn_LIF_T16_CelebA2622/C'), 
                layers=layers, neurons=neurons)
    #selectivity_additional_analyzer.selectivity_analysis_tsne()
    #selectivity_additional_analyzer.selectivity_analysis_distance()
    selectivity_additional_analyzer.selectivity_analysis_correlation()
    
