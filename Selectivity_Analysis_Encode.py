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
    
    Task: Sept 6, 2023
        
        rewrite the code based on my preference - make the 'encode' not based on ANOVA but independent

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
#from functools import reduce

import spiking_vgg, spiking_resnet, sew_resnet
from spikingjelly.activation_based import surrogate, neuron, functional

import vgg, resnet
import utils_


class Encode_feaquency_analyzer():
    """
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
        idx_folder = os.path.join(self.dest, 'ANOVA/ANOVA_idces')
        self.idx_list = [os.path.join(idx_folder, _) for _ in sorted(os.listdir(idx_folder)) if 'neuronIdx'  in _]    # <- consider to remove this?
        
        if layers == None or neurons == None:
            raise RuntimeError('[Codwarning] invalid layers and neurons')
            
        # FIXME - to have a better function to plot
        self.subplot_row, self.subplot_col = self.generate_subplot_row_and_col(len(self.layers))
        
        self.model_structure = root.split('/')[-2].split(' ')[1]
        
    #FIXME
    def generate_subplot_row_and_col(self, input):
        col = 5
        row = math.floor(input/col) +1
        remainder = input%col
        print(f'[Codinfo] Calculated cal [{col}] and row+1 [{row}] for input [{input}], with remainder [{remainder}]')
        return row, col
    
    #FIXME 
    def obtain_encode_class_dict(self, single_neuron_test=False, verbose=False):
        """
            current version, the idx is based on id_sensitive unit, not absolute idx the correction is a big project
            considering need to change all following functions this code now looks not good, need to rewrite
            [task 2] add parallel calculation
        """
        print('[Codinfo] Executing obtain_encode_class_dict...')
        self.SIMI_dict = {}
        
        fig, axs = plt.subplots(self.subplot_row, self.subplot_col, figsize=((self.subplot_col*2, self.subplot_row*2)))   # organize the shape of subplots
        
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        x = np.arange(self.num_classes)+1
        cnt_row = 0
        cnt_col = 0
        save_folder = os.path.join(self.dest_Encode, 'Selected_Neuron_Encoding_Performance/')     # what's the meaning of this operation?
        utils_.make_dir(save_folder)
        
        # ----- layer check
        feature_list_check = [_.split('/')[-1].split('.')[0] for _ in self.feature_list]
        layers_check = self.layers.copy()
        if not sorted(layers_check) == sorted(feature_list_check):
            raise RuntimeError('[Coderror] detected the features and layers not match')
        # -----
        
        # [notice] not use this way because it makes the code hard to read
        for idx, feature_path in enumerate(self.feature_list):     # for each layer
            
            feature = utils_.pickle_load(feature_path)      # load feature matrix
            layer = feature_path.split('/')[-1].split('.')[0]
                
            idx_path = self.idx_list[idx]
            sensitive_unit_idx = np.loadtxt(idx_path, delimiter=',')      # fix the problem of only using the id_sensitive_unit
            
            if sensitive_unit_idx.size == 0:
                self.SIMI_dict.update({layer: [{'neuron_amount': 0},{'selective_neuron_amount': 0},{'SI_idx': {}}, {'MI_idx': {}}]})
            elif sensitive_unit_idx.size != 0:
                if sensitive_unit_idx.size == 1:
                    sensitive_unit_idx = np.array([sensitive_unit_idx])
                sensitive_unit_idx = list(map(int, sensitive_unit_idx))    
                sensitive_units_feature = feature[:, sensitive_unit_idx]  # obtain sensitive_units_feature
                
                # [notice] from here, the  idx is based on id_sensitive unit
                _, sensitive_unit_idx = sensitive_units_feature.shape
                if verbose:
                    print('[Codinfo] 3. identity selective neuron calculate: ', sensitive_unit_idx, 'sensitive_units_feature in total')
                SI_idx = {}     # SI idx: encoded classes - the location in the sensitive_units_feature (sensitive unit)
                MI_idx = {}     # MI idx: encoded classes - like above
                # ===== loop for neurons of one layer
                for i in range(sensitive_unit_idx):  # for each neuron
                    neuron = sensitive_units_feature[:, i]
                    global_mean = np.mean(neuron)
                    global_std = np.std(neuron)
                    threshold = global_mean + 2 * global_std
                    d = [neuron[i*self.num_samples: i*self.num_samples+self.num_samples] for i in range(self.num_classes)]
                    d = np.array(d)
                    local_mean = np.mean(d, axis=1)     # array of 50 values
                    
                    # [notice] need to re-check this process
                    encode_class = [i + 1 for i, mean in enumerate(local_mean) if mean > threshold]     # the list of what classes encoded by this neuron
                    if not encode_class == []:
                        if len(encode_class) == 1:
                            SI_idx.update({i:encode_class})
                        else:
                            MI_idx.update({i:encode_class})
                            
                if verbose:
                    print('\n[Codinfo] layer: {}, {} neurons (SI: {}, MI: {}) pass the threshold (all neuron: {}, selective neuron: {}).'.format(
                    layer, len(list(SI_idx.keys()))+len(list(MI_idx.keys())), len(list(SI_idx.keys())), len(list(MI_idx.keys())), feature.shape[1], sensitive_unit_idx))
                    
                self.SIMI_dict.update({layer: [{'neuron_amount': feature.shape[1]},{'selective_neuron_amount': sensitive_unit_idx},{'SI_idx': SI_idx}, {'MI_idx': MI_idx}]})
                # =====
                
                if single_neuron_test:
                    if sensitive_unit_idx != 0:
                        check_neuron = random.choice(range(sensitive_unit_idx))        # random select a neuron in one layer
                        if verbose:
                            print('[Codinfo] Now check', layer, ': #', check_neuron, '\n')
                        neuron_vector = sensitive_units_feature[:, check_neuron]
                        ID_vector_list = [neuron_vector[i*self.num_samples: i*self.num_samples+self.num_samples] for i in range(self.num_classes)] # list for each ID [list 50]
                        ID_vector_list = np.array(ID_vector_list)
                        mean_list = np.mean(ID_vector_list, axis=1)   # 50 means
                        x = np.arange(self.num_classes) + 1   # add 1 from each element

                        # === merged subplot
                        axs[cnt_row, cnt_col].bar(x, mean_list, width=0.5)      # subplot
                        axs[cnt_row, cnt_col].set_title(layer + ' # ' + str(check_neuron), fontsize=8)
                        cnt_col += 1     # if initial sensitive_unit_idx == 1, this will throw error message while using old version generate_subplot_row_and_col
                        if cnt_col == self.subplot_col:
                            cnt_col = 0
                            cnt_row += 1  
                        # === independent fig
                        fig2, axes2 = plt.subplots(1)
                        axes2.bar(x, mean_list, width=0.5)
                        axes2.set_xticks(np.arange(0, self.num_classes+1, step=2))
                        axes2.set_xlabel('IDs', fontsize=8)
                        axes2.set_ylabel('local mean', fontsize=8)
                        axes2.set_title(layer + ' # ' + str(check_neuron))
                        fig2.savefig(save_folder+layer+' # '+str(check_neuron)+'_encoding_performance.png', bbox_inches='tight', dpi=100)
                        plt.close(fig2)
                                
            if single_neuron_test:
                for ax in axs.flat:
                    ax.label_outer()
                for ax in axs.flat:
                    ax.set_xlabel(xlabel='IDs', fontsize=8, labelpad=0)
                    ax.set_ylabel(ylabel='local mean', fontsize=8, labelpad=0)
                plt.tight_layout()
                fig.savefig(self.dest_Encode+'single_neuron_encoding_performance_All.png', bbox_inches='tight', dpi=100)
                plt.close(fig)
            
        print('[Codinfo] Saving SIMI ID_neuron_encode_class_dict.pkl ...')
        utils_.pickle_dump(os.path.join(self.dest_Encode, 'ID_neuron_encode_class_dict.pkl'), self.SIMI_dict)      # save the relationship between layer (include SI and MI) and encoded classes
        print('[Codinfo] SIMI ID_neuron_encode_dict.pkl saved in {}'.format(self.dest_Encode))
    
    #FIXME
    # looks need to find a better way to save the data make us to better understand the encode_id, si_idx, mi_idx, just like the MATLAB version
    def reload_and_revocer_encode_class_dict(self, save_path=None):
        print('[Codinfo] Executing reload_and_recover_encode_class_dict...')
        if save_path == None:
            save_path_ = self.dest_Encode
            
        encode_class_dict = utils_.pickle_load(os.path.join(save_path_, 'ID_neuron_encode_class_dict.pkl'))
        temp_dict = self.recover_encode_class_dict(encode_class_dict)
            
        return temp_dict
    
    def recover_encode_class_dict(self, SIMI_dict=None):     # [Warning] this function merges the encoded [classes] again, different with SIMI.py
        print('[Codinfo] Executing recover_encode_class_dict...')
        if SIMI_dict == None:   # directly succeed from self.obtain_encode_clas_dict()
            SIMI_dict_ = self.SIMI_dict
        else:
            SIMI_dict_ = SIMI_dict
        temp_dict = {}
        for k, v in SIMI_dict_.items():  # for each layer
            encode_class = []
            if list(v[2]['SI_idx'].values()) != []:
                for i in list(v[2]['SI_idx'].values()):     # for each neuron
                    for j in i:
                        encode_class.append(j)
            if list(v[3]['MI_idx'].values()) != []:
                for i in list(v[3]['MI_idx'].values()):
                    for j in i:
                        encode_class.append(j)    
            temp_dict.update({k: encode_class})
            
        return temp_dict
    
    # fill-in the white blank
    def draw_encode_frequency(self):        # general figure for encoding frequency
    
        print('[Codinfo] Executing draw_encode_frequency...')
        
        if self.mode == 'reload_encode_dict':
            encode_class_dict = self.reload_and_revocer_encode_class_dict()
        else:
            encode_class_dict = self.recover_encode_class_dict() 
        #print(encode_class_dict)
        
        freq_dic = {}
        for idx, layer in enumerate(self.layers):    # for each layer
            freq = {g:0 for g in range(1,51)}
            encode_class_list = encode_class_dict[layer]   # [list] obtain the encoded classes
    
            for item in encode_class_list:  # for each class
                if item in freq:  # update the dict or add new k-v pair
                    freq[item] += 1 
                    
            freq = {k: v / self.neurons[idx] for k, v in freq.items()}    # convert v from abs avlue to ratio
            #freq = dict(sorted(freq.items(), key=lambda item: item[0]))     # sort
            freq_dic.update({layer: freq})
        
        a = pd.DataFrame.from_dict(freq_dic)
        a_neuron = a[[i for i in self.layers if 'neuron' in i or 'pool' in i or 'fc_3' in i]]
        
        # 1. for all calculations
        plt.figure()
        im = plt.matshow(a, aspect='auto')
        
        plt.colorbar(im, fraction=0.12, pad=0.04)
        plt.xticks([])
        plt.xlabel(f'{a.shape[1]} Layers')
        plt.yticks([])
        plt.ylabel('IDs')
        plt.title('all')
        plt.savefig(self.dest_Encode+'Encoding_frequency_of_layers_all_calculation.png', bbox_inches='tight', dpi=100)
        plt.close()
        
        # 2. for imaginary neurons
        plt.figure()
        im = plt.matshow(a_neuron, aspect='auto')
        plt.colorbar(im, fraction=0.12, pad=0.04)
        plt.xticks([])
        plt.xlabel(f'{a_neuron.shape[1]} Layers')
        plt.yticks([])
        plt.ylabel('IDs')
        plt.title('act')
        plt.savefig(self.dest_Encode+'Encoding_frequency_of_layers.png', bbox_inches='tight', dpi=100)
        plt.close()
        
        # 3. merged plot
        fig, axs = plt.subplots(1, 2)
        im = axs[0].matshow(a, aspect='auto')
        axs[1].matshow(a_neuron, aspect='auto')
        axs[0].set_title('all')
        axs[0].set_xticks([])
        axs[0].set_xlabel(f'{a.shape[1]} Layers')
        axs[0].set_yticks([])
        axs[0].set_ylabel('IDs')
        axs[1].set_title('act')
        axs[1].set_xticks([])
        axs[1].set_xlabel(f'{a_neuron.shape[1]} Layers')
        axs[1].set_yticks([])
        fig.suptitle(f'Encoding Frequency - {self.model_structure}', fontsize=10)
        fig.subplots_adjust(top=0.8)
        fig.colorbar(im, ax=axs, orientation='vertical')
        plt.savefig(self.dest_Encode+'Encoding_frequency_of_layers_merged.png', bbox_inches='tight')
        plt.savefig(self.dest_Encode+'Encoding_frequency_of_layers_merged.eps', bbox_inches='tight', format='eps')
        plt.close()

    def draw_encode_frequency_for_each_layer(self):         # encoding frequency for each layer
        print('[Codinfo] Executing draw_encode_frequency_for_each_layer...')
        if self.mode == 'reload_encode_dict':
            encode_class_dict = self.reload_and_revocer_encode_class_dict()
        else:
            encode_class_dict = self.recover_encode_class_dict()  

        #make dir fo the image batch
        save_folder = os.path.join(self.dest_Encode, 'Each_Layer_Encoding_Performance/')
        utils_.make_dir(save_folder)
        occ_list = []
        for layer in self.layers:    # for each layer
            occurrences = []
            for i in range(self.num_classes):     # for each ID
                occ = encode_class_dict[layer].count(i + 1)  # calculate the frequency of each ID [one value]
                occurrences.append(occ) # store frequency for each ID [list of 50]
            occ_list.append(occurrences)    # merge the frequency info for all layers [list of len(layers)]
            x = np.arange(self.num_classes)+1
            plt.figure()
            plt.bar(x, occurrences, width=0.5)
            plt.xticks(np.arange(0, self.num_classes+1, step=2))
            plt.xlabel('IDs')
            plt.ylabel('Frequrency')
            plt.title('Encoded ID frequency: '+layer+'\n\u03F4: 2std')
            plt.savefig(save_folder+layer+'_encoding_performance.png', bbox_inches='tight', dpi=100)
            plt.close()
    
    def draw_merged_encode_frequency_for_each_layer(self):      # encoding frequency for each layer, basiccaly this is the merged version of the last one
        print('[Codinfo] Executing draw_merged_encode_frequency_for_each_layer...')
        if self.mode == 'reload_encode_dict':
            encode_class_dict = self.reload_and_revocer_encode_class_dict()
        else:
            encode_class_dict = self.recover_encode_class_dict()
        
        fig, axs = plt.subplots(self.subplot_row, self.subplot_col, figsize=((self.subplot_col*2, self.subplot_row*2)))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        x = np.arange(self.num_classes)+1
        cnt_row = 0
        cnt_col = 0
        
        for layer in self.layers:    # for each layer
            occurrences = []
            for i in range(self.num_classes):     # for each ID
                occ = encode_class_dict[layer].count(i + 1)
                occurrences.append(occ)
            axs[cnt_row, cnt_col].bar(x, occurrences, width=0.5)
            axs[cnt_row, cnt_col].set_title(layer, fontsize=8)
            cnt_col += 1        # set subplot location
            if cnt_col == self.subplot_col:
                cnt_col = 0
                cnt_row += 1
        for ax in axs.flat:
            ax.label_outer()
        for ax in axs.flat:
            ax.set_ylabel(ylabel='Freq', fontsize=8, labelpad=0)
            ax.set_xlabel(xlabel='IDs', fontsize=8, labelpad=0)
        plt.tight_layout()
        plt.savefig(self.dest_Encode+'single_layer_encoding_performance.png', bbox_inches='tight', dpi=100)
        plt.close()

    
if __name__ == "__main__":
    
    model_ = vgg.__dict__['vgg16'](num_classes=50)
    layers, neurons, shapes = utils_.generate_vgg_layers_list_ann(model_, 'vgg16')

    root_dir = '/home/acxyle-workstation/Downloads'

    selectivity_analyzer = Encode_feaquency_analyzer(root=os.path.join(root_dir, 'Face Identity Baseline/'), 
                                                     layers=layers, neurons=neurons)
    
    selectivity_analyzer.obtain_encode_class_dict(single_neuron_test=True)
    selectivity_analyzer.draw_encode_frequency()
    selectivity_analyzer.draw_encode_frequency_for_each_layer()
    selectivity_analyzer.draw_merged_encode_frequency_for_each_layer()
