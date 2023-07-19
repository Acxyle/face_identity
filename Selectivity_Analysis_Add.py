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
import pickle

import numpy as np
import seaborn as sn
from tqdm import tqdm
import scipy.io as sio
import matplotlib as mpl
from scipy.io import savemat
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from matplotlib.transforms import ScaledTranslation

#import vgg, resnet
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
        
        if status:     # local test
            self.selectivity_analysis_Tsne(self)
            self.selectivity_analysis_distance(self)
        
    def selectivity_analysis_tsne(self, verbose=False):
        
        print('[Codinfo] Executing selectivity_analysis_Tsne...')
        label = utils_.makeLabels(self.num_samples, self.num_classes)
        save_path = os.path.join(self.dest, 'TSNE')
        utils_.make_dir(save_path)
        
        tsne_x1_list = []
        tsne_y1_list = []
        tsne_x2_list = []
        tsne_y2_list = []
        tsne_ID_list = []
        tsne_nonID_list = []
        title_list = []
        mask_ID_list = []
        mask_nonID_list = []
        
        for layer in tqdm(self.layers):
            self.feature = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))
            _, maskID, maskNonID = self.generate_masks(self.dest, layer, self.feature.shape[1])
            mask_ID_list.append(len(maskID))
            mask_nonID_list.append(len(maskNonID))
            
            perplexity_ID = self.calculate_perplexity(maskID)
            perplexity_nonID = self.calculate_perplexity(maskNonID)
                    
            if verbose:
                print('layer: {}, maskID: {}, nonmaskID: {}, perplexity: {:.3f} {:.3f}'.format(layer, len(maskID), len(maskNonID), perplexity_ID, perplexity_nonID))
            
            valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if not item[1].startswith('not')])
            markers = valid_markers + valid_markers[:self.num_classes - len(valid_markers)]
            
            # in-layer comparison
            fig, ax = plt.subplots(1, 2, figsize=(12,6), dpi=100)
            
            plt.text(0.075, 0.925, '{}/{}'.format(len(maskID), self.feature.shape[1]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2},  
                     transform=fig.transFigure, verticalalignment='top', horizontalalignment='left')
            plt.text(0.5, 0.925, '{}/{}'.format(len(maskNonID), self.feature.shape[1]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2}, 
                     transform=fig.transFigure, verticalalignment='top', horizontalalignment='left')
            
            tsne_x1, tsne_y1, tsne_x2, tsne_y2, tsne_ID, tsne_nonID, title = self.tsne_layer(ax, maskID, perplexity_ID, maskNonID, perplexity_nonID, label, markers, layer, save_path, layer)
            
            tsne_x1_list.append(tsne_x1)
            tsne_y1_list.append(tsne_y1)
            tsne_x2_list.append(tsne_x2)
            tsne_y2_list.append(tsne_y2)
            tsne_ID_list.append(tsne_ID)
            tsne_nonID_list.append(tsne_nonID)
            title_list.append(title)
            
            #plt.show()
            fig.savefig(save_path + f'/tsne_{layer}.png', bbox_inches='tight', dpi=100)     # change dpi whilst data change
            fig.savefig(save_path + f'/tsne_{layer}.eps', bbox_inches='tight', dpi=100, format='eps')
            plt.close()
            
        # in-model comparison
        fig_sup, ax_sup = plt.subplots(len(self.layers), 2, figsize=(12, 6*len(self.layers)), dpi=100)
        
        tsne_x1_sup = min(tsne_x1_list)
        tsne_y1_sup = min(tsne_y1_list)
        tsne_x2_sup = max(tsne_x2_list)
        tsne_y2_sup = max(tsne_y2_list)
        
        dx1, dy1 = -0.4, 0.25
        
        for idx, layer in enumerate(self.layers):
            
            tsne_ID = tsne_ID_list[idx]
            tsne_nonID = tsne_nonID_list[idx]
            title = title_list[idx]
            
            self.tsne_plot(ax_sup[idx, 0], tsne_ID, maskID, label, markers, title+' Selective', layer, save_path, tsne_x1_sup, tsne_y1_sup, tsne_x2_sup, tsne_y2_sup)
            self.tsne_plot(ax_sup[idx, 1], tsne_nonID, maskNonID, label, markers, title+' non-Selective', layer, save_path, tsne_x1_sup, tsne_y1_sup, tsne_x2_sup, tsne_y2_sup)
            
            trans1 = ax_sup[idx, 0].transAxes + ScaledTranslation(dx1, dy1, fig.dpi_scale_trans)
            trans2 = ax_sup[idx, 1].transAxes + ScaledTranslation(dx1, dy1, fig.dpi_scale_trans)
            
            ax_sup[idx, 0].text(0, 1, '{}/{}'.format(mask_ID_list[idx], mask_ID_list[idx]+mask_nonID_list[idx]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2},  
                     transform=trans1, verticalalignment='top', horizontalalignment='left')
            ax_sup[idx, 1].text(0, 1, '{}/{}'.format(mask_nonID_list[idx], mask_ID_list[idx]+mask_nonID_list[idx]), fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2}, 
                     transform=trans2, verticalalignment='top', horizontalalignment='left')
        
        fig_sup.savefig(save_path + '/tsne_all.png', bbox_inches='tight', dpi=100)     # change dpi whilst data change
        fig_sup.savefig(save_path + '/tsne_all.eps', bbox_inches='tight', dpi=100, format='eps')
        plt.close()
            
    def tsne_layer(self, ax, maskID, perplexity_ID, maskNonID, perplexity_nonID, label, markers, layer, save_path, title):
        
        tsne_ID = self.tsne_layer_single(maskID, perplexity_ID, label, markers, layer, save_path, layer)
        tsne_nonID = self.tsne_layer_single(maskNonID, perplexity_nonID, label, markers, layer, save_path, layer)
        
        if tsne_ID is not None and tsne_nonID is not None:
            tsne_x1 = min(min(tsne_ID[:, 0]), min(tsne_nonID[:, 0]))
            tsne_y1 = min(min(tsne_ID[:, 1]), min(tsne_nonID[:, 1]))
            tsne_x2 = max(max(tsne_ID[:, 0]), max(tsne_nonID[:, 0]))
            tsne_y2 = max(max(tsne_ID[:, 1]), max(tsne_nonID[:, 1]))
        else:
            tsne_x1 = min(tsne_ID[:, 0])
            tsne_y1 = min(tsne_ID[:, 1])
            tsne_x2 = max(tsne_ID[:, 0])
            tsne_y2 = max(tsne_ID[:, 1])
        
        self.tsne_plot(ax[0], tsne_ID, maskID, label, markers, title+' Selective', layer, save_path, tsne_x1, tsne_y1, tsne_x2, tsne_y2)
        self.tsne_plot(ax[1], tsne_nonID, maskNonID, label, markers, title+' non-Selective', layer, save_path, tsne_x1, tsne_y1, tsne_x2, tsne_y2)
        
        return tsne_x1, tsne_y1, tsne_x2, tsne_y2, tsne_ID, tsne_nonID, title
        
    def tsne_layer_single(self, mask, perplexity, label, markers, layer, save_path, title):
        if len(mask) == 0:
            return
        elif len(mask) == 1:
            tsne = self.feature[:, mask]
            return tsne
        else:
            if np.std(np.array(self.feature[:, mask])) != 0.:     # [notice] make sure all faetures are not identical
                tsne = TSNE(perplexity=perplexity, n_jobs=-1).fit_transform(self.feature[:, mask]) 
                return tsne
            else:
                return
        
    def tsne_plot(self, ax, tsne, mask, label, markers, title, layer, save_path, tsne_x1, tsne_y1, tsne_x2, tsne_y2):
        
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
        if len(mask) >= 1:
            perplexity = min(math.sqrt(len(mask)), self.num_classes*self.num_samples-1)
            if perplexity == 0.:
                perplexity = 1e-9
        else:
            perplexity = None
        return perplexity

    #FIXME distance scale
    def selectivity_analysis_distance(self):
        print('[Codinfo] Executing selectivity_analysis_ditance...')
        dest = self.dest+'Distance/'
        utils_.make_dir(dest)  
        
        for layer in tqdm(self.layers):     # for each layer
        
            feature = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))

            avgMatrix = self.avg_across_values(feature)
            _, maskID, maskNonID = self.generate_masks(self.dest, layer, feature.shape[1])
        
            fig, axes = plt.subplots(1, 3, figsize=((30, 10)))
            cbar_ax = fig.add_axes([.91, .1, .03, .8])
        
            dist_avg = pdist(avgMatrix, 'euclidean')
            m = squareform(dist_avg)
            
            dist_avg_ID = pdist(avgMatrix[:, maskID], 'euclidean')
            m_i = squareform(dist_avg_ID)
            
            dist_avg_NonID = pdist(avgMatrix[:, maskNonID], 'euclidean')
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
            tmp_sorted_ID = utils_.pickle_load(os.path.join(os.getcwd(),'tmp_sorted_ID.pkl'))  # x[0] real worder, x[1] torch order
            tmp_sorted_ID = sorted(tmp_sorted_ID, key=lambda x: x[1])
            tmp_list_ID = [avgMatrix[i] for i in range(self.num_classes)]
            avgMatrix = self.restore_order(tmp_list_ID, tmp_sorted_ID, self.num_classes)
            
            #tmp_sorted_face = [i for i in tmp_sorted_ID for _ in range(self.num_samples)]
            #tmp_list_face = [fullMatrix[i] for i in range(self.num_classes*self.num_samples)]
            #fullMatrix = self.restore_order(tmp_list_face, tmp_sorted_face, self.num_classes*self.num_samples)

            maskID = np.loadtxt(self.dest  + '/' + layer + '-neuronIdx.csv', delimiter=',')
            maskID = list(map(int, maskID))
            maskNonID = sorted(list(set(np.arange(fullMatrix.shape[1])) - set(maskID)))
            
            cor_avg_dict, cor_avg_ID_dict, cor_avg_nonID_dict = self.correlation_single(avgMatrix, layer, maskID, maskNonID, cor_avg_dict, cor_avg_ID_dict, cor_avg_nonID_dict, dest_ID)
            #cor_full_dict, cor_full_ID_dict, cor_full_nonID_dict = self.correlation_single(fullMatrix, layer, maskID, maskNonID, cor_full_dict, cor_full_ID_dict, cor_full_nonID_dict, dest_faces)
            
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
    
    def correlation_single(self, matrix, layer, maskID, maskNonID, cor_dict, cor_sensitive_dict, cor_nonsensitive_dict, dest):
        cor, cor_sensitive, cor_nonsensitive = self.correlation_calculate_single(matrix, layer, maskID, maskNonID)
        cor_dict.update({layer: cor})
        cor_sensitive_dict.update({layer: cor_sensitive})
        cor_nonsensitive_dict.update({layer: cor_nonsensitive})
        self.correlation_plot_single(cor, cor_sensitive, cor_nonsensitive, maskID, maskNonID, layer, dest)
        
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
   
    def correlation_calculate_single(self, matrix, layer, maskID, maskNonID):
        
        cor = np.corrcoef(matrix)
        cor_ID = np.corrcoef(matrix[:, maskID])
        cor_nonID = np.corrcoef(matrix[:, maskNonID])
        
        return cor, cor_ID, cor_nonID
        
    def correlation_plot_single(self, cor, cor_ID, cor_nonID, maskID, maskNonID, layer, save_path):
        
        fig, axes = plt.subplots(1, 3, figsize=((30, 10)))
        cbar_ax = fig.add_axes([.91, .1, .03, .8])
    
        vmax = max(cor.max(), cor_ID.max(), cor_nonID.max())
        vmin = min(cor.min(), cor_ID.min(), cor_nonID.min())
    
        sn.heatmap(cor, ax=axes[0], cmap='jet', cbar=0, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
        axes[0].set_title(f'all neurons ({len(maskID)+len(maskNonID)})')
        sn.heatmap(cor_ID, ax=axes[1], cmap='jet',  cbar=1, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
        axes[1].set_title(f'identity selective neurons ({len(maskID)})')
        sn.heatmap(cor_nonID, ax = axes[2], cmap='jet',  cbar=2, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
        axes[2].set_title(f'non identity selective neurons ({len(maskNonID)})')
    
        fig.tight_layout(rect=[0, 0, .9, 1])
        
        plt.title(layer)
        plt.savefig(save_path+layer+'-Correlation.png', bbox_inches='tight', dpi=100)
        plt.close()

    def avg_across_values(self, matrix):            
        avg_ = np.array([matrix[i*self.num_samples:(i+1)*self.num_samples, :] for i in range(self.num_classes)]).mean(axis=1)
        return avg_ 
   
    
if __name__ == '__main__':
    
# =============================================================================
#     neuron_ = neuron.LIFNode
#     neuron_name = 'LIF'
#     T = 16
#     
#     spiking_model = spiking_vgg.__dict__['spiking_vgg16_bn'](spiking_neuron=neuron_, num_classes=50, surrogate_function=surrogate.ATan(), detach_reset=True)
#     functional.set_step_mode(spiking_model, step_mode='m')
#     layers, neurons, shapes = utils_.generate_vgg_layers(spiking_model, 'spiking_vgg16_bn')
#     
#     #layers_ = ['L1_B2_neuron', 'L2_B2_neuron', 'L3_B3_neuron', 'L4_B3_neuron', 'L5_B3_neuron']
#     layers_ = [i for i in layers if 'neuron' in i or 'pool' in i or 'fc_3' in i]
#     index_ = [layers.index(i) for i in layers_]
#     neurons_ = [neurons[i] for i in index_]
#     shapes_ = [shapes[i] for i in index_]
#     layers = layers_
#     neurons = neurons_
# 
#     root_dir = '/media/acxyle/Data/ChromeDownload/'
# 
#     selectivity_additional_analyzer = Selectiviy_Analysis_Additional(
#                 root=os.path.join(root_dir, f'Identity_SpikingVGG16bn_{neuron_name}_ATan_T{T}_CelebA2622_Results/'), 
#                 dest=os.path.join(root_dir, f'Identity_SpikingVGG16bn_{neuron_name}_ATan_T{T}_CelebA2622_Neuron/'), 
#                 layers=layers, neurons=neurons)
#     selectivity_additional_analyzer.selectivity_analysis_tsne()
#     #selectivity_additional_analyzer.selectivity_analysis_distance()
#     #selectivity_additional_analyzer.selectivity_analysis_correlation()
# =============================================================================
    
    neuron_ = neuron.IzhikevichNode
    neuron_name = 'Izhikevich'
    
    spiking_model = spiking_resnet.__dict__['spiking_resnet18'](spiking_neuron=neuron_, num_classes=50, surrogate_function=surrogate.ATan(), detach_reset=True, mode='feature')
    functional.set_step_mode(spiking_model, step_mode='m') 
    layers, neurons, shapes = utils_.generate_resnet_layers_list(spiking_model, 'spiking_resnet18')
    
    layers_ = [i for i in layers if 'neuron' in i or 'pool' in i or 'fc' in i]
    index_ = [layers.index(i) for i in layers_]
    neurons_ = [neurons[i] for i in index_]
    shapes_ = [shapes[i] for i in index_]
    layers = layers_
    neurons = neurons_

    root_dir = '/media/acxyle/Data/ChromeDownload/'

    selectivity_additional_analyzer = Selectiviy_Analysis_Additional(
                root=os.path.join(root_dir, f'Identity_spiking_resnet18_{neuron_name}_ATan_T4_CelebA2622_Results/'), 
                dest=os.path.join(root_dir, f'Identity_spiking_resnet18_{neuron_name}_ATan_T4_CelebA2622_Neuron/'), 
                layers=layers, neurons=neurons)
    selectivity_additional_analyzer.selectivity_analysis_tsne()
    #selectivity_additional_analyzer.selectivity_analysis_distance()
    #selectivity_additional_analyzer.selectivity_analysis_correlation()
