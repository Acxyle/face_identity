#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:54:55 2023

@author: acxyle

    #TODO
    complete 5 sections in one script:
        1. obtrain_encode_class_dict() - save dict.pkl
        2. draw_encode_frequency()
        3. draw_encode_frequency_for_each_layer()
        4. draw_merged_encode_frequency_for_each_layer()
        5. draw_single_neuron_response()
    #TODO
    bonus:
        make a parser version?        
    #TODO
    progress:
        move this code to server for utility test
        DONE

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
from functools import reduce

import spiking_vgg, spiking_resnet, sew_resnet
from spikingjelly.activation_based import surrogate, neuron, functional

import resnet
import utils_

parser = argparse.ArgumentParser(description="Selectivity Analyzer - Encoding Frequency", add_help=True)
parser.add_argument("--mode", type=str, default='general', help="[Codelp] 1. general; 2. reload_encode_dict")
parser.add_argument("--num_classes", type=int, default=50, help="{Codelp] set the number of classes")
parser.add_argument("--num_samples", type=int, default=10, help="[Codelp] set the sample number of each class")
parser.add_argument("--save_path", type=str, default=None, help="[Codelp] set the folder where save the output data, this script will generate it automatically")
parser.add_argument("--model", type=str, default='spikingVGG16bn')
args = parser.parse_args()

class Encode_feaquency_analyzer():
    
    def __init__(self, feature_root, idx_root, save_path=None, num_classes=50, num_samples=10, layers=None, neurons=None, mode=None):

# =============================================================================
#         self.feature_list = [(feature_root+f) for f in sorted(os.listdir(feature_root)) if f.split('.')[-1]=='pkl']     # feature .pkl list
#         self.idx_list = [(idx_root+f) for f in sorted(os.listdir(idx_root)) if 'neuronIdx'  in f.split('-')[-1]]    # feature -idx.csv list
# =============================================================================
        
        if layers == None or neurons == None:
            raise RuntimeError('[Codwarning] invalid layers and neurons')
            
        self.layers = layers
        self.neurons = neurons
        
        if save_path == None:
            self.save_path = idx_root+'Frequency/'
        utils_.make_dir(self.save_path)
        
        self.num_classes = num_classes
        self.num_samples = num_samples

        self.subplot_row, self.subplot_col = self.generate_subplot_row_and_col(len(self.layers))
        self.mode = mode     # 1. general; 2. reload_encode_dict | expected that 'mode' can leads to following tasks 
        
        #if self.mode == 'general':
         #   self.obtain_encode_class_dict(self, single_neuron_test=True)
         #   self.draw_encode_frequency(self)
         #   self.draw_encode_frequency_for_each_layer(self)
         #   self.draw_merged_encode_frequency_for_each_layer(self)

        #elif self.mode == 'reload_encode_dict':
         #   self.draw_encode_frequency(self)
         #   self.draw_encode_frequency_for_each_layer(self)
         #   self.draw_merged_encode_frequency_for_each_layer(self)
            
    
    def generate_subplot_row_and_col(self, input):     # [Warning] 原版查看早版本代码，原版的目标更加合理
        col = 6
        row = math.floor(input/col) +1
        return row, col

    def obtain_encode_class_dict(self, verbose=False, single_neuron_test=False):
        """
        #TODO
        [action required] 
        1. the virgin version only record the encode information between layer and identity 
        while missed the Single Identity neuron and Multiple Identity neuron, which 
        leads to recalculate in following lines, so fix this.
        2. make a verbose version for running check
        """
        print('[Codinfo] Executing obtain_encode_class_dict...')
        self.SIMI_dict = {}
        
        #FIXME
        if single_neuron_test:
            #print(self.subplot_row, self.subplot_col)
            fig, axs = plt.subplots(self.subplot_row, self.subplot_col, figsize=((self.subplot_col*2, self.subplot_row*2)))   # organize the shape of subplots
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            x = np.arange(self.num_classes)+1
            cnt_row = 0
            cnt_col = 0
            save_folder = os.path.join(self.save_path, 'Selected_Neuron_Encoding_Performance/')
            utils_.make_dir(save_folder)
        
        for layer in tqdm(self.layers):
            for feature_path in self.feature_list:     # [a check] for each layer 
                if layer == feature_path.split('/')[-1].split('.')[0]:
                    if verbose:
                        print('[Codinfo] 1. layer name: ', layer)
                    with open(feature_path, 'rb') as pkl:
                        feature = pickle.load(pkl)      # load feature matrix
                    for idx_path in self.idx_list:       
                        if layer == idx_path.split('/')[-1].split('-')[0]:     # select the correspondant neuron_idx file
                                if verbose:
                                    print('[Codinfo] 2. idx file: ', idx_path.split('/')[-1])
                                sig_neuron_idx = np.loadtxt(idx_path, delimiter=',')
                                #print('\n', sig_neuron_idx)
                                if sig_neuron_idx.size == 0:
                                    self.SIMI_dict.update({layer: [{'neuron_amount': 0},{'selective_neuron_amount': 0},{'SI_idx': {}}, {'MI_idx': {}}]})
                                elif sig_neuron_idx.size != 0:
                                    if sig_neuron_idx.size == 1:
                                        sig_neuron_idx = np.array([sig_neuron_idx])
                                    sig_neuron_idx = list(map(int, sig_neuron_idx))    
                                    sig_neuron = feature[:, sig_neuron_idx]  # obtain sig_neuron
                                    _, col = sig_neuron.shape
                                    if verbose:
                                        print('[Codinfo] 3. identity selective neuron calculate: ', col, 'sig_neuron in total')
                                    SI_idx = {}     # SI idx: encoded classes
                                    MI_idx = {}     # MI idx: encoded classes
                                    # ===== loop for neurons of one layer
                                    for i in range(col):  # for each neuron
                                        neuron = sig_neuron[:, i]
                                        global_mean = np.mean(neuron)
                                        global_std = np.std(neuron)
                                        threshold = global_mean + 2 * global_std
                                        d = [neuron[i*self.num_samples: i*self.num_samples+self.num_samples] for i in range(self.num_classes)]
                                        d = np.array(d)
                                        local_mean = np.mean(d, axis=1)     # array of 50 values
                                        encode_class = [i + 1 for i, mean in enumerate(local_mean) if mean > threshold]     # the list of what classes encoded by this neuron
                                        if not encode_class == []:
                                            if len(encode_class) == 1:
                                                SI_idx.update({i:encode_class})
                                            else:
                                                MI_idx.update({i:encode_class})
                                    if verbose:
                                        print('\n[Codinfo] layer: {}, {} neurons (SI: {}, MI: {}) pass the threshold (all neuron: {}, selective neuron: {}).'.format(
                                        layer, len(list(SI_idx.keys()))+len(list(MI_idx.keys())), len(list(SI_idx.keys())), len(list(MI_idx.keys())), feature.shape[1], col))
                                    self.SIMI_dict.update({layer: [{'neuron_amount': feature.shape[1]},{'selective_neuron_amount': col},{'SI_idx': SI_idx}, {'MI_idx': MI_idx}]})
                                    # =====
                                    if single_neuron_test:
                                        if col != 0:
                                            check_neuron = random.choice(range(col))        # random select a neuron in one layer
                                            if verbose:
                                                print('[Codinfo] Now check', layer, ': #', check_neuron, '\n')
                                            neuron_vector = sig_neuron[:, check_neuron]
                                            ID_vector_list = [neuron_vector[i*self.num_samples: i*self.num_samples+self.num_samples] for i in range(self.num_classes)] # list for each ID [list 50]
                                            ID_vector_list = np.array(ID_vector_list)
                                            mean_list = np.mean(ID_vector_list, axis=1)   # 50 means
                                            x = np.arange(self.num_classes) + 1   # add 1 from each element
                                            #FIXME
                                            # === merged subplot
                                            axs[cnt_row, cnt_col].bar(x, mean_list, width=0.5)      # subplot
                                            axs[cnt_row, cnt_col].set_title(layer + ' # ' + str(check_neuron), fontsize=8)
                                            cnt_col += 1     # if initial col == 1, this will throw error message while using old version generate_subplot_row_and_col
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
                        fig.savefig(self.save_path+'single_neuron_encoding_performance_All.png', bbox_inches='tight', dpi=100)
                        plt.close(fig)
            
        print('[Codinfo] Saving SIMI ID_neuron_encode_class_dict.pkl ...')
        with open(os.path.join(self.save_path, 'ID_neuron_encode_class_dict.pkl'), 'wb') as f:      # save the relationship betwwen layer (include SI and MI) and encoded classes
            pickle.dump(self.SIMI_dict, f)
        print('[Codinfo] SIMI ID_neuron_encode_dict.pkl saved in {}'.format(self.save_path))
    
    def reload_and_revocer_encode_class_dict(self, save_path=None):
        print('[Codinfo] Executing reload_and_recover_encode_class_dict...')
        if save_path == None:
            save_path_ = self.save_path
        with open(os.path.join(save_path_, 'ID_neuron_encode_class_dict.pkl'), 'rb') as f:
            encode_class_dict = pickle.load(f)
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
    
    #TODO
    # fill-in the white blank
    def draw_encode_frequency(self):        # general figure for encoding frequency
    
        # [notice]
        model_name = 'Spiking_VGG16_bn'
    
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
        plt.xlabel(f'{a.shape[1]} Layers')
        plt.ylabel('IDs')
        plt.title('Encoding Frequency for Each Layer (all calculation)')
        plt.savefig(self.save_path+'Encoding_frequency_of_layers_all_calculation.png', bbox_inches='tight', dpi=100)
        plt.close()
        
        # 2. for imaginary neurons
        plt.figure()
        im = plt.matshow(a_neuron, aspect='auto')
        plt.colorbar(im, fraction=0.12, pad=0.04)
        plt.xlabel(f'{a_neuron.shape[1]} Layers')
        plt.ylabel('IDs')
        plt.title('Encoding Frequency for Each Layer')
        plt.savefig(self.save_path+'Encoding_frequency_of_layers.png', bbox_inches='tight', dpi=100)
        plt.close()
        
        # 3. merged plot
        fig, axs = plt.subplots(1, 2)
        im = axs[0].matshow(a, aspect='auto')
        im_neuron = axs[1].matshow(a_neuron, aspect='auto')
        axs[0].set_title('all calculation')
        axs[0].set_ylabel('IDs')
        axs[0].set_xlabel(f'{a.shape[1]} Layers')
        axs[1].set_title('artificial neuron')
        axs[1].set_yticks([])
        axs[1].set_xlabel(f'{a_neuron.shape[1]} Layers')
        #fig.suptitle(f'Encoding Frequency - {model_name}', fontsize=10)
        fig.subplots_adjust(top=0.8)
        fig.colorbar(im, ax=axs, orientation='vertical')
        plt.savefig(self.save_path+'Encoding_frequency_of_layers_merged.png', bbox_inches='tight')
        plt.savefig(self.save_path+'Encoding_frequency_of_layers_merged.eps', bbox_inches='tight', format='eps')
        plt.close()

    def draw_encode_frequency_for_each_layer(self):         # encoding frequency for each layer
        print('[Codinfo] Executing draw_encode_frequency_for_each_layer...')
        if self.mode == 'reload_encode_dict':
            encode_class_dict = self.reload_and_revocer_encode_class_dict()
        else:
            encode_class_dict = self.recover_encode_class_dict()  
        #TODO
        #make dir fo the image batch
        save_folder = os.path.join(self.save_path, 'Each_Layer_Encoding_Performance/')
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
        plt.savefig(self.save_path+'single_layer_encoding_performance.png', bbox_inches='tight', dpi=100)
        plt.close()

    
if __name__ == "__main__":
    
    model_ = spiking_vgg.__dict__['spiking_vgg16_bn'](spiking_neuron=neuron.LIFNode, num_classes=50, surrogate_function=surrogate.ATan(), detach_reset=True)
    functional.set_step_mode(model_, step_mode='m')
    layers, neurons, shapes = utils_.generate_vgg_layers(model_, 'spiking_vgg16_bn')

    selectivity_analyzer = Encode_feaquency_analyzer(feature_root='/media/acxyle/Data/ChromeDownload/Identity_SpikingVGG16bn_LIF_CelebA9326_Results/', 
                                                     idx_root='/media/acxyle/Data/ChromeDownload/Identity_SpikingVGG16bn_LIF_CelebA9326_Neuron/', 
                                                     save_path=args.save_path, layers=layers, neurons=neurons, mode='reload_encode_dict')
    selectivity_analyzer.draw_encode_frequency()