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

import vgg, resnet
import spiking_vgg, spiking_resnet, sew_resnet
from spikingjelly.activation_based import surrogate, neuron, functional

import utils_

class Selectiviy_Analysis_Additional():
    def __init__(self, feature_root='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Results/',
                 idx_root='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/',
                 num_samples=10, num_classes=50, data_name='', layers=None, neurons=None, status=False):
        
        if layers == None or neurons == None:
            raise RuntimeError('[Codwarning] invalid layers and neurons')
            
        self.layers = layers
        self.neurons = neurons
        
        self.feature_root = feature_root
        self.idx_root = idx_root
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.data_name = data_name
        
        if status:     # local test
            self.selectivity_analysis_Tsne(self)
            self.selectivity_analysis_distance(self)
        
    def generate_masks(self, idx_root, layer, feature_col):
        
        sig_neuron_idx = np.loadtxt(idx_root + layer + '-neuronIdx.csv', delimiter=',')     # [warning] simple call
        if sig_neuron_idx.size == 1:
            sig_neuron_idx = np.array([sig_neuron_idx])
        sig_neuron_idx = list(map(int, sig_neuron_idx))
        #print(sig_neuron_idx)

        all_idx = [i for i in range(feature_col)]
        non_sig_neuron_idx = list(set(all_idx)-set(sig_neuron_idx))
        #print('Length of ID/nonID mask:', len(sig_neuron_idx), len(non_sig_neuron_idx))
        return all_idx, sig_neuron_idx, non_sig_neuron_idx
    
    def selectivity_analysis_Tsne(self, verbose=False):
        print('[Codinfo] Executing selectivity_analysis_Tsne...')
        label = utils_.makeLabels(self.num_samples, self.num_classes)
        
        """
         [warning] if suffers problem of RAM size, can change TSNE parameters to reduce memory consumption but accuracy may impaired
         eg. TSNE(n_iter=1000->500, perplexity=100->10, early_exaggeration=12->4) 
         Except above changes on TSNE(), other preprocesses can be done on other component or input data
         eg. pca = sklearn.decomposition.PCA(n_components=None->100)
             TSNE().fit_transform(pca(input))
        In this example, use PCA to recude the dimension of input then feed into embedding function: TSNE().fit_transform(input), this can
        reduce the RAM consumption, but probability distribution may changed
        
         [warning] t_sne.py source code location: ./anaconda3/envs/spikingjelly/lib/python3.8/site-packages/sklearn/manifold/
         [warning] source code may changed for check and print running info
        """
        
        for layer in tqdm(self.layers):
        
            with open(os.path.join(self.feature_root, layer+'.pkl'), 'rb') as f:
                feature = pickle.load(f)
            save_path = self.idx_root + 'TSNE/' + layer
            utils_.make_dir(save_path)     # make folder for each layer
        
            _, maskID, maskNonID = self.generate_masks(self.idx_root, layer, feature.shape[1])
            
            if len(maskID) >= 1:
                perplexity_ID = min(math.sqrt(len(maskID)), self.num_classes*self.num_samples-1)
                if perplexity_ID == 0.:
                    perplexity_ID = 1e-9
            if len(maskNonID) >= 1:
                perplexity_nonID = min(math.sqrt(len(maskNonID)), self.num_classes*self.num_samples-1)
                if perplexity_nonID == 0.:
                    perplexity_nonID = 1e-9
            
            if verbose:
                print('layer: {}, maskID: {}, nonmaskID: {}, perplexity: {:.3f} {:.3f}'.format(layer, len(maskID), len(maskNonID), perplexity_ID, perplexity_nonID))
            
            valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if not item[1].startswith('not')])
            markers = valid_markers + valid_markers[:self.num_classes - len(valid_markers)]
            self.single_tsne(maskID, perplexity_ID, feature, label, markers, layer, save_path, 'ID')
            self.single_tsne(maskNonID, perplexity_nonID, feature, label, markers, layer, save_path, 'nonID')

    def single_tsne(self, mask, perplexity, feature, label, markers, layer, save_path, prefix):
        if len(mask) == 0:
            tsne = TSNE(perplexity=perplexity).fit_transform(torch.randn(self.num_samples*self.num_classes, 1000))
        elif len(mask) == 1:
            tsne = TSNE(n_components=1, perplexity=perplexity).fit_transform(feature[:, mask])
        elif np.std(np.array(feature[:, mask])) != 0.:     
            test_value = int(self.num_classes*self.num_samples)     
            if feature[:, mask].shape[1] > test_value:     
                np_log = math.ceil(test_value*(math.log(len(mask)/test_value)+1.))
                pca = PCA(n_components=min(test_value, np_log))
                x = pca.fit_transform(feature[:, mask])
                tsne = TSNE(perplexity=perplexity, n_jobs=-1).fit_transform(x)
            else:
                tsne = TSNE(perplexity=perplexity, n_jobs=-1).fit_transform(feature[:, mask])
        else:
            return
            
        self.tsne_plot(tsne, mask, label, markers, prefix, feature.shape, layer, save_path)
        
    
    def tsne_plot(self, tsne, mask, label, markers, prefix, feature_shape, layer, save_path):
        plt.figure()
        if tsne.shape[1] == 2:
            for i in range(self.num_classes):
                plt.scatter(tsne[i*self.num_samples: (i+1)*self.num_samples, 0],
                            tsne[i*self.num_samples: (i+1)*self.num_samples, 1],
                            label[i*self.num_samples: (i+1)* self.num_samples], marker=markers[i])
                plt.text(min(tsne[:,0]), 
                         max(tsne[:,1])+0.105*(max(tsne[:,1])-min(tsne[:,1])), 
                         '{}/{}'.format(len(mask), feature_shape[1]), fontsize=10, bbox={'facecolor':'yellow', 'alpha' : 0.2})
                
        elif tsne.shape[1] == 1:
            for i in range(self.num_classes):
                plt.scatter(tsne[i*self.num_samples: (i+1)*self.num_samples, 0],
                            tsne[i*self.num_samples: (i+1)*self.num_samples, 0],
                            label[i*self.num_samples: (i+1)* self.num_samples], marker=markers[i])
            
        plt.title(self.data_name + ' ' + layer + f'_{prefix}')
        
        plt.savefig(save_path + f'/tsne_{prefix}.png', bbox_inches='tight', dpi=100)     # change dpi whilst data change
        plt.close()
    

    def avg_acrossID(self, matrix, num_samples, num_classes):            
        avg_full = []
        for i in range(num_classes):
            submat = matrix[i*num_samples:(i+1)*num_samples,:]
            avg_sub = submat.mean(axis=0)
            avg_full.append(avg_sub)
        avg_full = np.array(avg_full)
            
        return avg_full
    
    
    def selectivity_analysis_distance(self):
        print('[Codinfo] Executing selectivity_analysis_ditance...')
        dest = self.idx_root+'Distance/'
        utils_.make_dir(dest)  
        
        for layer in tqdm(self.layers):     # for each layer
        
            with open(self.feature_root + layer + '.pkl', 'rb') as f:
                feature = pickle.load(f)

            avgMatrix = self.avg_acrossID(feature, self.num_samples, self.num_classes)
            _, maskID, maskNonID = self.generate_masks(self.idx_root, layer, feature.shape[1])
        
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
        
    # [notice] test version
    def selectivity_analysis_Correlation(self):
        print('[Codinfo] Executing selectivity_analysis_correlation...')
        dest = self.idx_root+'Correlation/'
        dest_ID = dest+'ID/'
        dest_faces = dest+'faces/'
        
        utils_.make_dir(dest)
        utils_.make_dir(dest_ID)
        utils_.make_dir(dest_faces)
        
        #cor_full_dict = {}
        #cor_full_ID_dict = {}
        #cor_full_nonID_dict = {}
        
        cor_avg_dict = {}
        cor_avg_ID_dict = {}
        cor_avg_nonID_dict = {}
        
        for layer in tqdm(self.layers):
            with open(self.feature_root + layer + '.pkl', 'rb') as f:
                fullMatrix = pickle.load(f)  # 500,neurons
            f.close()
            avgMatrix = self.avg_acrossID(fullMatrix, self.num_samples, self.num_classes)  # -> (50,neurons)
            
            # ----- ID
            # [notice] below is the idx correction operation to fix the idx disorder
            tmp_list = [avgMatrix[i] for i in range(50)]
            with open(os.path.join(os.getcwd(),'tmp_sorted_ID.pkl'),'rb') as f:
                tmp_sorted_ID = pickle.load(f)
            f.close()
            
            tmp_sorted_ID = sorted(tmp_sorted_ID, key=lambda x: x[1])
            for idx, e in enumerate(tmp_list):
                e = [e]
                e.append(tmp_sorted_ID[idx])
                tmp_list[idx] = e
            tmp_list = sorted(tmp_list,key=lambda x:int(x[1][0]))
            tmp_list = [tmp_list[i][0] for i in range(50)]
            tmp_list = np.array(tmp_list)
            avgMatrix = tmp_list
            # -----
            
            # ----- Face
            #tmp_list_face = [fullMatrix[i] for i in range(500)]
            #tmp_sorted_face = [i for i in tmp_sorted_ID for _ in range(10)]
            #tmp_list_face = [fullMatrix[i] for i in range(500)]
            #for idx,e in enumerate(tmp_list_face):
            #    e = [e]
            #    e.append(tmp_sorted_face[idx])
            #    tmp_list_face[idx] = e
            #tmp_list_face = sorted(tmp_list_face,key=lambda x:int(x[1][0]))
            #tmp_list_face = [tmp_list_face[i][0]for i in range(500)]
            #tmp_list_face = np.array(tmp_list_face)
            #fullMatrix = tmp_list_face
            # -----

            maskID = np.loadtxt(self.idx_root  + '/' + layer + '-neuronIdx.csv', delimiter=',')
            maskID = list(map(int, maskID))
        
            idx_list = np.arange(fullMatrix.shape[1])
            maskNonID = sorted(list(set(idx_list) - set(maskID)))
        
            # calculation of correlation coefficient
        
            # 1. full correlation matrix
            #cor_full = np.corrcoef(fullMatrix)
            #cor_full_dict.update({layer: cor_full})
        
            # 2. full ID/nonID correlation matrix
            #cor_full_ID = np.corrcoef(fullMatrix[:, maskID])
            #cor_full_ID_dict.update({layer: cor_full_ID})
        
            #cor_full_nonID = np.corrcoef(fullMatrix[:, maskNonID])
            #cor_full_nonID_dict.update({layer: cor_full_nonID})
        
            # 3. avg correlation matrix
            cor_avg = np.corrcoef(avgMatrix)
            cor_avg_dict.update({layer: cor_avg})
        
            # 4. avg ID/nonID correlation matri
            cor_avg_ID = np.corrcoef(avgMatrix[:, maskID])
            cor_avg_ID_dict.update({layer: cor_avg_ID})
        
            cor_avg_nonID = np.corrcoef(avgMatrix[:, maskNonID])
            cor_avg_nonID_dict.update({layer: cor_avg_nonID})
            
            # 5. plot_ID
            fig, axes = plt.subplots(1, 3, figsize=((30, 10)))
            cbar_ax = fig.add_axes([.91, .1, .03, .8])
        
            vmax = max(cor_avg.max(), cor_avg_ID.max(), cor_avg_nonID.max())
            vmin = min(cor_avg.min(), cor_avg_ID.min(), cor_avg_nonID.min())
        
            sn.heatmap(cor_avg, ax=axes[0], cmap='jet', cbar=0, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
            axes[0].set_title(f'all neurons ({avgMatrix.shape[1]})')
            sn.heatmap(cor_avg_ID, ax=axes[1], cmap='jet',  cbar=1, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
            axes[1].set_title(f'identity selective neurons ({len(maskID)})')
            sn.heatmap(cor_avg_nonID, ax = axes[2], cmap='jet',  cbar=2, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
            axes[2].set_title(f'non identity selective neurons ({len(maskNonID)})')
        
            fig.tight_layout(rect=[0, 0, .9, 1])
            
            plt.title(layer)
            plt.savefig(dest_ID+layer+'-Correlation.png', bbox_inches='tight', dpi=100)
            plt.close()
            
            # 6. plot_face
            #fig, axes = plt.subplots(1, 3, figsize=((30, 10)))
            #cbar_ax = fig.add_axes([.91, .1, .03, .8])
        
            #vmax = max(cor_full.max(), cor_full_ID.max(), cor_full_nonID.max())
            #vmin = min(cor_full.max(), cor_full_ID.max(), cor_full_nonID.max())
        
            #sn.heatmap(cor_full, ax=axes[0], cmap='jet', cbar=0, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
            #axes[0].set_title(f'all neurons ({avgMatrix.shape[1]})')
            #sn.heatmap(cor_full_ID, ax=axes[1], cmap='jet',  cbar=1, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
            #axes[1].set_title(f'identity selective neurons ({len(maskID)})')
            #sn.heatmap(cor_full_nonID, ax = axes[2], cmap='jet',  cbar=2, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax)
            #axes[2].set_title(f'non identity selective neurons ({len(maskNonID)})')
        
            #fig.tight_layout(rect=[0, 0, .9, 1])
            
            #plt.title(layer)
            #plt.savefig(dest_faces+layer+'-Correlation.png', bbox_inches='tight', dpi=100)
            #plt.close()
            
        print('[Codinfo] Saving Correlation Matrix...')
        #savemat(dest + '/' + 'CorMatrix_full.mat', cor_full_dict)
        #savemat(dest + '/' + 'CorMatrix_full_ID.mat', cor_full_ID_dict)
        #savemat(dest + '/' + 'CorMatrix_full_nonID.mat', cor_full_nonID_dict)
        
        savemat(dest + '/' + 'CorMatrix_avg.mat', cor_avg_dict)
        savemat(dest + '/' + 'CorMatrix_avg_ID.mat', cor_avg_ID_dict)
        savemat(dest + '/' + 'CorMatrix_avg_nonID.mat', cor_avg_nonID_dict)
   
if __name__ == '__main__':
    
    model_ = resnet.__dict__['resnet50'](num_classes=50)
    layers, neurons, shapes = utils_.generate_resnet_layers_list_ann(model_, 'resnet50')
    
    layers_ = [i for i in layers if 'neuron' in i or 'fc' in i or 'pool' in i]
    index_ = [layers.index(i) for i in layers_]
    neurons_ = [neurons[i] for i in index_]
    
    layers = layers_
    neurons = neurons_

    selectivity_additional_analyzer = Selectiviy_Analysis_Additional(
                feature_root='/media/acxyle/Data/ChromeDownload/Identity_Resnet50_Original_Results/', 
                idx_root='/media/acxyle/Data/ChromeDownload/Identity_Resnet50_Original_Neuron/',
                layers=layers, neurons=neurons)
    selectivity_additional_analyzer.selectivity_analysis_Correlation()