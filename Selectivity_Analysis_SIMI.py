#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 15:23:50 2023

@author: acxyle
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import logging

import spiking_vgg, spiking_resnet, sew_resnet
from spikingjelly.activation_based import surrogate, neuron, functional

import vgg, resnet
import utils_


class Selectivity_Analysis_SIMI():
    def __init__(self, root='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Results/', 
                 dest='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/',
                 num_classes=50, num_samples=10, layers=None, neurons=None, status=False):
        
        if layers == None or neurons == None:
            raise RuntimeError('[Codwarning] invalid layers and neurons')
            
        self.layers = layers
        self.neurons = neurons
        
        self.root = root
        self.dest = dest
        
        # [notice] close this for local test
        self.feature_list = [(self.root+f) for f in sorted(os.listdir(self.root)) if f.split('.')[-1]=='pkl']
        self.idx_list = [(self.dest+f) for f in sorted(os.listdir(self.dest)) if 'neuronIdx'  in f.split('-')[-1]]

        self.num_classes = num_classes
        self.num_samples = num_samples

        self.save_path = dest+'SIMI/'
        
        utils_.make_dir(self.save_path)
        
        # [notice] load the encoding neuron
        self.load_ID_neuron_encode_class_dict()
        
        if status == True:
            self.SIMI_SVM(self)
            self.bins_percent_of_SIMI(self)
            self.single_neuron_boxplot(self)
        
        # [notice] auto obtain config
        #FIXME
        config_list = dest.split('/')[-2].split('_')[1:6]
        if len(config_list)==5:
            self.model_structure = '_'.join([*config_list[:2]])
            self.neuron_type = config_list[2]
            self.surrogate_func = config_list[3]
            self.simulation_step = config_list[4]
        elif len(config_list)==4:
            self.model_structure = config_list[0]
            self.neuron_type = config_list[1]
            self.surrogate_func = None
            self.simulation_step = None
            
        
    def load_ID_neuron_encode_class_dict(self):
        self.ID_neuron_encode_class_dict = utils_.pickle_load(os.path.join(self.dest, 'Frequency/ID_neuron_encode_class_dict.pkl'))
    
    def recover_SIMI_dict(self):
        SIMI_dict = {}
        for k, v in self.ID_neuron_encode_class_dict.items():  # for each layer
            SI_idx = list(v[2]['SI_idx'].keys())
            MI_idx = list(v[3]['MI_idx'].keys())   
            SIMI_dict.update({k: [SI_idx, MI_idx]})
        
        return SIMI_dict
    
    def SIMI_SVM(self, verbose=False):

        print('[Codinfo] Executing SIMI_SVM Ver2.0 with ID_acc, nonID_acc and all_acc...')
        SIMI_dict = self.recover_SIMI_dict()
        
        SIMI_acc_dict = {}
        SI_acc_dict = {}
        MI_acc_dict = {}
        ID_acc_dict = {}
        nonID_acc_dict = {}
        all_acc_dict = {}
        
        label = utils_.makeLabels(self.num_samples, self.num_classes)
        
        for layer in tqdm(self.layers):     # each layer
            feature_path = os.path.join(self.root, layer+'.pkl')
            for idx_path in self.idx_list:
                if layer == idx_path.split('/')[-1].split('-')[0]: 
                    

                    feature = utils_.pickle_load(feature_path)
                    
                    all_neuron_idx = [i for i in range(feature.shape[1])]
                    
                    sig_neuron_idx = np.loadtxt(idx_path, delimiter=',')
                    
                    if sig_neuron_idx.size == 1:
                        sig_neuron_idx = np.array([sig_neuron_idx])
                    sig_neuron_idx = list(map(int, sig_neuron_idx))
                    non_sig_neuron_idx = list(set(all_neuron_idx)-set(sig_neuron_idx))     # [notice] useful trick to speed up
                    SI_idx = SIMI_dict[layer][0]
                    MI_idx = SIMI_dict[layer][1]
                    SIMI_idx = SI_idx + MI_idx
            
                    all_acc = utils_.SVM_classification(feature, label)*100
                    all_acc_dict.update({layer: all_acc})
                    
                    ID_acc = utils_.SVM_classification(feature[:, sig_neuron_idx], label)*100
                    ID_acc_dict.update({layer: ID_acc})
                    
                    nonID_acc = utils_.SVM_classification(feature[:, non_sig_neuron_idx], label)*100
                    nonID_acc_dict.update({layer: nonID_acc})
                    
                    SIMI_acc = utils_.SVM_classification(feature[:, SIMI_idx], label)*100
                    SIMI_acc_dict.update({layer: SIMI_acc})
                    
                    SI_acc = utils_.SVM_classification(feature[:, SI_idx], label)*100
                    SI_acc_dict.update({layer: SI_acc})
                    
                    MI_acc = utils_.SVM_classification(feature[:, MI_idx], label)*100
                    MI_acc_dict.update({layer: MI_acc})
                    
                    #TODO
                    # 这里之后也改成表格输出
                    if verbose:
                        print('[Codinfo] layer: {}, total neurons: {}, ID: {}, non ID: {}, SI: {}, MI: {}, SI+MI: {}'.format(
                            layer, feature.shape[1], len(sig_neuron_idx), len(non_sig_neuron_idx), len(SI_idx), len(MI_idx), len(SIMI_idx)),
                              '\n[Codinfo] all_acc: {:.2f}%, ID_acc: {:.2f}%, nonID_acc: {:.2f}%, SIMI_acc: {:.2f}%, SI_acc: {:.2f}%, MI_acc: {:.2f}%, '.format(
                            all_acc, ID_acc, nonID_acc, SIMI_acc, SI_acc, MI_acc))
        
        # [notice] save for future use
        utils_.pickle_dump(os.path.join(self.dest, 'SIMI/SIMI.pkl'), [all_acc_dict,ID_acc_dict, nonID_acc_dict, SIMI_acc_dict, SI_acc_dict, MI_acc_dict])
    
    def segment_dict_neuron(self, input_dict):
        input_dict = {i:input_dict[i] for i in list(input_dict.keys()) if 'neuron' in i or 'pool' in i or 'fc' in i}
        return input_dict
    
    def plot_SVM_all(self):
        
        print('[Codinfo] Executing SIMI plotting for selected neuron...')
        
        [all_acc_dict,ID_acc_dict, nonID_acc_dict, SIMI_acc_dict, SI_acc_dict, MI_acc_dict] = utils_.pickle_load(os.path.join(self.dest, 'SIMI/SIMI.pkl'))
        layer_list = self.layers
        fig, ax = plt.subplots(1, 1, figsize=(int(len(self.layers)/2),10), dpi=200)
        plt.rcParams.update({'font.size': 20})
        
        self.plot_single_decoding(ax, layer_list, all_acc_dict,ID_acc_dict, nonID_acc_dict, SIMI_acc_dict, SI_acc_dict, MI_acc_dict, vertical=True, title='Decoding_Performance_(all)')
        plt.close()

    #FIXME
    def plt_SVM_all_calculation_comparison(self):
        print('[Codinfo] Executing SIMI plotting for selected neuron...')
        
        
        [all_acc_dict,ID_acc_dict, nonID_acc_dict, SIMI_acc_dict, SI_acc_dict, MI_acc_dict] = utils_.pickle_load(os.path.join(self.dest, 'SIMI/SIMI.pkl'))
        
        # for camparison
        dest_ANN = '/media/acxyle/Data/ChromeDownload/Identity_VGG16bn_ReLU_CelebA2622_Neuron/'
        with open(dest_ANN + 'SIMI/SIMI.pkl', 'rb') as f:
            [all_acc_dict_ann,ID_acc_dict_ann, nonID_acc_dict_ann, SIMI_acc_dict_ann, SI_acc_dict_ann, MI_acc_dict_ann] = pickle.load(f)
        f.close()
        
        layer_list = self.layers
        x = layer_list
        y_a_ann = [all_acc_dict_ann[k] for k in layer_list]
        y_i_ann = [ID_acc_dict_ann[k] for k in layer_list]
        y_n_ann = [nonID_acc_dict_ann[k] for k in layer_list]
        y_ann = [SIMI_acc_dict_ann[k] for k in layer_list]
        y_s_ann = [SI_acc_dict_ann[k] for k in layer_list]
        y_m_ann = [MI_acc_dict_ann[k] for k in layer_list]
        # --
        
        layer_list = self.layers
        x = layer_list
        y_a = [all_acc_dict[k] for k in layer_list]
        y_i = [ID_acc_dict[k] for k in layer_list]
        y_n = [nonID_acc_dict[k] for k in layer_list]
        y = [SIMI_acc_dict[k] for k in layer_list]
        y_s = [SI_acc_dict[k] for k in layer_list]
        y_m = [MI_acc_dict[k] for k in layer_list]
        
        # for vertical lines
        n_idx = [x.index(_) for _ in [i for i in x if 'neuron' in i]]
        nd_idx = [x.index(_) for _ in [i for i in x if 'drop' in i]]
        nf_idx = [x.index(_) for _ in [i for i in x if 'fc_3' in i]]
        
        plt.figure(figsize=(int(len(self.layers)/2),10), dpi=200)
        plt.rcParams.update({'font.size': 20})
        model_name = self.dest.split('/')[-2].split('_')[1]
        
        plt.plot(x, y, 'blue', label='ID-Encoding(SI+MI)')
        plt.plot(x, y_s, 'green', label='SI')
        plt.plot(x, y_m, 'purple', label='MI')
        plt.plot(x, y_a, 'red', label='all')
        plt.plot(x, y_i, 'orange', label='ID-Sensitive(passed ANOVA)')
        plt.plot(x, y_n, 'teal', label='non-ID-Sensitive(failed ANOVA)')
        
        # ---
        plt.plot(x, y_ann, 'blue', alpha=0.25)
        #plt.plot(x, y_s_ann, 'green', alpha=0.25)
        #plt.plot(x, y_m_ann, 'purple', alpha=0.25)
        plt.plot(x, y_a_ann, 'red', alpha=0.25)
        plt.plot(x, y_i_ann, 'orange', alpha=0.25)
        plt.plot(x, y_n_ann, 'teal', alpha=0.25)
        #--
        
        for _ in n_idx:
            plt.vlines(_, 0, y_i[_], colors='blue', linestyles='--', linewidth=2.0, alpha=0.5)
        for _ in nf_idx:
            plt.vlines(_, 0, y_i[_], colors='teal', linestyles='--', linewidth=1.0, alpha=0.5)
        for _ in nd_idx:
            plt.vlines(_, 0, y_i[_], colors='gray', linestyles='--', linewidth=1.0, alpha=0.5)
        
        plt.ylim((0, 100))
        plt.legend()
        plt.xticks(rotation='vertical')
        plt.ylabel('Classification Accuracy')
        plt.title(f'Neuron Decoding Performance (all calculation) - {model_name} (vs VGG16bn)')
        plt.savefig(self.dest + 'SIMI/SIMI_acc_all_calculation.png', bbox_inches='tight')
        plt.savefig(self.dest + 'SIMI/SIMI_acc_all_calculation.eps', format='eps', bbox_inches='tight')
        plt.close()
        
    def plot_SVM_neuron(self):
        print('[Codinfo] Executing SIMI plotting for selected neuron...')

        [all_acc_dict, ID_acc_dict, nonID_acc_dict, SIMI_acc_dict, SI_acc_dict, MI_acc_dict] = utils_.pickle_load(os.path.join(self.dest, 'SIMI/SIMI.pkl'))
        
        #FIXME
        all_acc_dict = self.segment_dict_neuron(all_acc_dict)
        ID_acc_dict = self.segment_dict_neuron(ID_acc_dict)
        nonID_acc_dict = self.segment_dict_neuron(nonID_acc_dict)
        SIMI_acc_dict = self.segment_dict_neuron(SIMI_acc_dict)
        SI_acc_dict = self.segment_dict_neuron(SI_acc_dict)
        MI_acc_dict = self.segment_dict_neuron(MI_acc_dict)
        
        fig, ax = plt.subplots(1, 1, figsize=(10,8), dpi=100)
        plt.rcParams.update({'font.size': 14})
        
        # [notice] the selection for imaginary neurons
        layer_list = [i for i in self.layers if 'neuron' in i or 'pool' in i or 'fc' in i]
        
        self.plot_single_decoding(ax, layer_list, all_acc_dict,ID_acc_dict, nonID_acc_dict, SIMI_acc_dict, SI_acc_dict, MI_acc_dict, vertical=False, title='Decoding_Performance')
        plt.close()
    
    def plot_single_decoding(self, ax, layer_list, all_acc_dict,ID_acc_dict, nonID_acc_dict, SIMI_acc_dict, SI_acc_dict, MI_acc_dict, vertical=False, title='Decoding'):
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        x = layer_list
        y_a = [all_acc_dict[k] for k in layer_list]
        y_i = [ID_acc_dict[k] for k in layer_list]
        y_n = [nonID_acc_dict[k] for k in layer_list]
        y = [SIMI_acc_dict[k] for k in layer_list]
        y_s = [SI_acc_dict[k] for k in layer_list]
        y_m = [MI_acc_dict[k] for k in layer_list]
        
        ax.plot(x, y, 'blue', label='ID-Encoding(SI+MI)')
        ax.plot(x, y_s, 'green', label='SI')
        ax.plot(x, y_m, 'purple', label='MI')
        ax.plot(x, y_a, 'red', label='all')
        ax.plot(x, y_i, 'orange', label='ID-Sensitive(passed ANOVA)')
        ax.plot(x, y_n, 'teal', label='non-ID-Sensitive(failed ANOVA)')
        
        # for vertical lines
        if vertical == True:
            n_idx = [x.index(_) for _ in [i for i in x if 'neuron' in i]]
            nd_idx = [x.index(_) for _ in [i for i in x if 'drop' in i]]
            nf_idx = [x.index(_) for _ in [i for i in x if 'fc' in i]]
            
            for _ in n_idx:
                ax.vlines(_, 0, y_i[_], colors='blue', linestyles='--', linewidth=2.0, alpha=0.5)
            for _ in nf_idx:
                ax.vlines(_, 0, y_i[_], colors='teal', linestyles='--', linewidth=1.0, alpha=0.5)
            for _ in nd_idx:
                ax.vlines(_, 0, y_i[_], colors='gray', linestyles='--', linewidth=1.0, alpha=0.5)
        
        ax.set_ylim((0, 100))
        ax.legend()
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(layer_list, rotation='vertical')
        ax.set_ylabel('Classification Accuracy')
        ax.set_title(f'{title} - {self.model_structure}')
        plt.savefig(self.dest + f'SIMI/{title}.png', bbox_inches='tight')
        plt.savefig(self.dest + f'SIMI/{title}.eps', format='eps', bbox_inches='tight')
    
    #FIXME
    def plot_SVM_neuron_comparison(self):
        print('[Codinfo] Executing SIMI plotting for selected neuron...')

        [all_acc_dict,ID_acc_dict, nonID_acc_dict, SIMI_acc_dict, SI_acc_dict, MI_acc_dict] = utils_.pickle_load(os.path.join(self.dest, 'SIMI/SIMI.pkl'))
        
        # for camparison
        dest_ANN = '/media/acxyle/Data/ChromeDownload/Identity_VGG16bn_ReLU_CelebA2622_Neuron/'
        with open(dest_ANN + 'SIMI/SIMI.pkl', 'rb') as f:
            [all_acc_dict_ann,ID_acc_dict_ann, nonID_acc_dict_ann, SIMI_acc_dict_ann, SI_acc_dict_ann, MI_acc_dict_ann] = pickle.load(f)
        f.close()
        # --
        
        all_acc_dict = self.segment_dict_neuron(all_acc_dict)
        ID_acc_dict = self.segment_dict_neuron(ID_acc_dict)
        nonID_acc_dict = self.segment_dict_neuron(nonID_acc_dict)
        SIMI_acc_dict = self.segment_dict_neuron(SIMI_acc_dict)
        SI_acc_dict = self.segment_dict_neuron(SI_acc_dict)
        MI_acc_dict = self.segment_dict_neuron(MI_acc_dict)
        
        # [notice] the selection for imaginary neurons
        layer_list = [i for i in self.layers if 'neuron' in i or 'pool' in i or 'fc' in i]
        x = layer_list
        y_a = [all_acc_dict[k] for k in layer_list]
        y_i = [ID_acc_dict[k] for k in layer_list]
        y_n = [nonID_acc_dict[k] for k in layer_list]
        y = [SIMI_acc_dict[k] for k in layer_list]
        y_s = [SI_acc_dict[k] for k in layer_list]
        y_m = [MI_acc_dict[k] for k in layer_list]
        
        plt.figure(figsize=(10,8), dpi=100)
        plt.rcParams.update({'font.size': 14})
        model_name = self.dest.split('/')[-2].split('_')[1]
        
        plt.plot(x, y, 'blue', label='SI+MI')
        plt.plot(x, y_s, 'green', label='SI')
        plt.plot(x, y_m, 'purple', label='MI')
        plt.plot(x, y_a, 'red', label='all neuron')
        plt.plot(x, y_i, 'orange', label='ID selective')
        plt.plot(x, y_n, 'teal', label='non ID selective')
        
        plt.ylim((0, 100))
        plt.legend()
        plt.xticks(rotation='vertical')
        plt.ylabel('Classification Accuracy (%)')
        plt.title(f'Neuron Decoding Performance - {model_name}')
        plt.savefig(self.dest + 'SIMI/SIMI_acc.png', bbox_inches='tight')
        plt.savefig(self.dest + 'SIMI/SIMI_acc.eps', format='eps', bbox_inches='tight')
        plt.close()
        
    def plot_bins_number_and_percent_of_SIMI(self):
        '''
            [notice] should put into 'Encode' Section

        '''
        print('[Codinfo] Executing SIMI percent plot...')
        
        SIMI_dict = self.recover_SIMI_dict()
        
        # 1. all 
        total_neuron = {}
        for idx, layer in enumerate(self.layers):
            total_neuron[layer] = self.neurons[idx]
        
        y_list = [SIMI_dict[k] for k in self.layers]     # according to layer_sequence order to save the number of SI and MI
        y1 = [len(item[0]) for item in y_list]     # y1 records the number of SI for each layer
        y2 = [len(item[1]) for item in y_list]     # y2 records the number of MI for each layer
        t = [total_neuron[k] for k in self.layers]
        percent_si = [i / j * 100 for i, j in zip(y1, t)]
        percent_mi = [i / j * 100 for i, j in zip(y2, t)]
        x = np.arange(len(self.layers))
        
        fig, ax = plt.subplots(1, 1, figsize=(int(len(self.layers)/2),10), dpi=200)
        self.plt_single_stack(ax, x, self.layers, y1, y2, 'Number', f'number (all) - {self.model_structure}')
        plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize=(int(len(self.layers)/2),10), dpi=200)
        self.plt_single_stack(ax, x, self.layers, percent_si, percent_mi, 'Percentage', f'percentage (all) - {self.model_structure}')
        plt.close()
        
        # --- save data
        if not os.path.exists(self.save_path+'stack_num&pct_neuron.pkl'):
            utils_.pickle_dump(os.path.join(self.save_path, 'stack_num&pct_neuron.pkl'), [y1, y2, percent_si, percent_mi])
        
        # 2. imaginary neuron only
        layers_neuron = [i for i in self.layers if 'neuron' in i or 'pool' in i or 'fc' in i]
        
        x_layer = layers_neuron
        x = np.arange(len(layers_neuron))
        y_list = [SIMI_dict[k] for k in layers_neuron] 
        y1 = [len(item[0]) for item in y_list]
        y2 = [len(item[1]) for item in y_list]
        t = [total_neuron[k] for k in layers_neuron]
        percent_si = [i / j * 100 for i, j in zip(y1, t)]
        percent_mi = [i / j * 100 for i, j in zip(y2, t)]
        
        fig, ax = plt.subplots(1, 1, figsize=(10,8), dpi=100)
        self.plt_single_stack(ax, x, x_layer, y1, y2, 'Number', f'number - {self.model_structure}')
        plt.close()
        
        fig, ax = plt.subplots(1,1, figsize=(10,8), dpi=100)
        self.plt_single_stack(ax, x, x_layer, percent_si, percent_mi, 'Percentage', f'percentage - {self.model_structure}')
        plt.close()
    
    def plt_single_stack(self, ax, x, x_label, statistic_si, statistic_mi, title, y_label, shift:float=0.):
        
        plt.rcParams.update({'font.size': 12})
        
        p1 = ax.bar(x-shift, statistic_si, width=0.25)
        p2 = ax.bar(x-shift, statistic_mi, bottom=statistic_si, width=0.25)
        ax.plot(x, statistic_si, linestyle='-', linewidth=2.5, alpha=1)
        statistic_sm = [statistic_si[i]+statistic_mi[i] for i in range(len(statistic_si))]
        ax.plot(x, statistic_sm, linestyle='-', linewidth=2.5, alpha=1)
        
        if 'percentage' in y_label.lower():
            ax.set_ylim((0, 100))
        ax.set_ylabel(f'{y_label}')
        ax.set_xticks(x, x_label, rotation='vertical')
        ax.legend((p1[0], p2[0]), ('Singele_Identity(SI) Unit', 'Multiple_Identity(MI) Unit'), frameon=False)
        ax.set_title(f'{title}')
        plt.savefig(self.save_path + f'/{y_label}.png', bbox_inches='tight')
        plt.savefig(self.save_path + f'/{y_label}.eps', format='eps', bbox_inches='tight')
        #ax.show()
    
    #FIXME
    def bins_percent_of_SIMI_comparison(self):
        print('[Codinfo] Executing SIMI percent plot...')
        
        # [notice]
        model_name = 'Spiking_VGG16_bn'
        
        SIMI_dict = self.recover_SIMI_dict()
        
        '''
            [notice] this should be put into 'Encode' Section, because there is not too much relationship with SIMI
            [notice] here is quite a legacy from Jinge Wang's code, this is not very clear and looks redundant, rewrite is required
        '''
        total_neuron = {}
        for idx, layer in enumerate(self.layers):
            total_neuron[layer] = self.neurons[idx]
        
        x = self.layers
        y_list = [SIMI_dict[k] for k in self.layers]     # according to layer_sequence order to save the number of SI and MI
        y1 = [len(item[0]) for item in y_list]     # y1 records the number of SI for each layer
        y2 = [len(item[1]) for item in y_list]     # y2 records the number of MI for each layer
        t = [total_neuron[k] for k in self.layers]
        percent_si = [i / j * 100 for i, j in zip(y1, t)]
        percent_mi = [i / j * 100 for i, j in zip(y2, t)]
        pct_encode = [percent_si[i]+percent_mi[i] for i in range(len(percent_si))]
        
# =============================================================================
#         # --- save data
#         #if not os.path.exists(self.save_path+'stack_pct.pkl'):
#         #    with open(self.save_path+'stack_pct.pkl', 'wb') as f:
#         #        pickle.dump([percent_si, percent_mi], f, protocol=-1)
#         #    f.close()
#         # ---
#         # --- load data
#         with open('/media/acxyle/Data/ChromeDownload/Identity_VGG16bn_ReLU_CelebA2622_Neuron/SIMI/stack_pct.pkl', 'rb') as f1:
#             [pct_si_vgg16bn, pct_mi_vgg16bn] = pickle.load(f1)
#         f1.close()
#         pct_encode_vgg16bn = [pct_si_vgg16bn[i]+pct_mi_vgg16bn[i] for i in range(len(pct_si_vgg16bn))]
#         # ---
# =============================================================================
        
        plt.rcParams.update({'font.size': 12})
        # 1. all calculation
        # --- number
# =============================================================================
#         plt.figure(figsize=(int(len(self.layers)/2),10), dpi=200)
#         p1 = plt.bar(x, y1, width=0.5)
#         p2 = plt.bar(x, y2, width=0.5)
#         plt.ylabel('Num of neurons')
#         plt.xticks(rotation='vertical')     # [Warning] default vertical, waitingte rewrite a adjustable function
#         plt.legend((p1[0], p2[0]), ('SI', 'MI'))
#         plt.title('Stack plot for SI/MI num in each layer')
#         plt.savefig(self.save_path + '/stack_plt_num.png', bbox_inches='tight')
#         plt.savefig(self.save_path + '/stack_plt_num.eps', format='eps', bbox_inches='tight')
#         plt.close()
# =============================================================================
        
        # --- percentages
        plt.figure(figsize=(int(len(self.layers)/2),10), dpi=200)
        
        x = np.arange(len(self.layers))
        
        p3 = plt.bar(x-0.125, percent_si, width=0.25)
        p4 = plt.bar(x-0.125, percent_mi, bottom=percent_si, width=0.25)
        plt.plot(x, percent_si, linestyle='-', linewidth=2.5, alpha=1)
        plt.plot(x, pct_encode, linestyle='-', linewidth=2.5, alpha=1)
        
# =============================================================================
#         p5 = plt.bar(x+0.125, pct_si_vgg16bn, width=0.25, alpha=0.2)
#         p6 = plt.bar(x+0.125, pct_mi_vgg16bn, bottom=pct_si_vgg16bn, width=0.25, alpha=0.2)
#         plt.plot(x, pct_si_vgg16bn, linestyle='--', linewidth=2.5, alpha=0.2)
#         plt.plot(x, pct_encode_vgg16bn, linestyle='--', linewidth=2.5, alpha=0.2)
# =============================================================================
        
        plt.ylim((0, 100))
        plt.ylabel('Percentage')
        plt.xticks(x, self.layers, rotation='vertical')
        plt.legend((p3[0], p4[0]), ('Singele_Identity(SI) Unit', 'Multiple_Identity(MI) Unit'), frameon=False)
        plt.title('Stack pct. for SI/MI in each layer(all calculation) - SpikingVGG16bn[solid] (vs VGG16bn[transparent])')
        #plt.savefig(self.save_path + '/stack_plt_percentage.png', bbox_inches='tight')
        #plt.savefig(self.save_path + '/stack_plt_percentage.eps', format='eps', bbox_inches='tight')
        plt.show()
        plt.close()
        
        # 2. imaginary neuron only
        layers_neuron = [i for i in self.layers if 'neuron' in i or 'pool' in i or 'fc_3' in i]
        x_layer = layers_neuron
        x = np.arange(len(layers_neuron))
        y_list = [SIMI_dict[k] for k in layers_neuron] 
        y1 = [len(item[0]) for item in y_list]
        y2 = [len(item[1]) for item in y_list]
        t = [total_neuron[k] for k in layers_neuron]
        percent_si = [i / j * 100 for i, j in zip(y1, t)]
        percent_mi = [i / j * 100 for i, j in zip(y2, t)]
        pct_encode = [percent_si[i]+percent_mi[i] for i in range(len(percent_si))]
        
# =============================================================================
#         # --- save data
#         #if not os.path.exists(self.save_path+'stack_pct_neuron.pkl'):
#         #    with open(self.save_path+'stack_pct_neuron.pkl', 'wb') as f:
#         #        pickle.dump([percent_si, percent_mi], f, protocol=-1)
#         #    f.close()
#         # ---
#         # --- load data
#         with open('/media/acxyle/Data/ChromeDownload/Identity_VGG16bn_ReLU_CelebA2622_Neuron/SIMI/stack_pct_neuron.pkl', 'rb') as f1:
#             [pct_si_vgg16bn, pct_mi_vgg16bn] = pickle.load(f1)
#         f1.close()
#         pct_encode_vgg16bn = [pct_si_vgg16bn[i]+pct_mi_vgg16bn[i] for i in range(len(pct_si_vgg16bn))]
#         # ---
# =============================================================================
        
        # --- number
# =============================================================================
#         plt.figure()
#         p1 = plt.bar(x, y1, width=0.5)
#         p2 = plt.bar(x, y2, width=0.5)
#         plt.ylabel('Num of neurons')
#         plt.xticks(rotation='vertical')     # [Warning] default vertical, waitingte rewrite a adjustable function
#         plt.legend((p1[0], p2[0]), ('SI', 'MI'))
#         plt.title('Stack plot for SI/MI num in each layer')
#         plt.savefig(self.save_path + '/stack_plt_num_neuron.png', bbox_inches='tight')
#         plt.savefig(self.save_path + '/stack_plt_num_neuron.eps', format='eps', bbox_inches='tight')
#         plt.close()
# =============================================================================

        # --- percentages
        plt.figure()
        p3 = plt.bar(x-0.125, percent_si, width=0.25)
        p4 = plt.bar(x-0.125, percent_mi, bottom=percent_si, width=0.25)
        plt.plot(x, percent_si, linestyle='-', linewidth=2.5, alpha=1)
        plt.plot(x, pct_encode, linestyle='-', linewidth=2.5, alpha=1)
        
# =============================================================================
#         p5 = plt.bar(x+0.125, pct_si_vgg16bn, width=0.25, alpha=0.2)
#         p6 = plt.bar(x+0.125, pct_mi_vgg16bn, bottom=pct_si_vgg16bn, width=0.25, alpha=0.2)
#         plt.plot(x, pct_si_vgg16bn, linestyle='--', linewidth=2.5, alpha=0.2)
#         plt.plot(x, pct_encode_vgg16bn, linestyle='--', linewidth=2.5, alpha=0.2)
# =============================================================================
        
        plt.ylim((0, 100))
        plt.ylabel('Percentage')
        plt.xticks(x, x_layer, rotation='vertical')
        plt.legend((p3[0], p4[0]), ('Singele_Identity(SI) Unit', 'Multiple_Identity(MI) Unit'), frameon=False)
        plt.title('Stack pct. for SI/MI in each layer(imaginary neuon)\nSpikingVGG16bn[solid] (vs VGG16bn[transparent])')
        #plt.savefig(self.save_path + '/stack_plt_percentage_neuron.png', bbox_inches='tight')
        #plt.savefig(self.save_path + '/stack_plt_percentage_neuron.eps', format='eps', bbox_inches='tight')
        plt.show()
        plt.close()
    
    def paint_neuron_encode_boxplot(self, neuron_list, boxplot):
        for idx, c in enumerate(boxplot['boxes']):
            c.set(color='gray', alpha=0.5)
            
        for idx_ in neuron_list:
            for idx, c in enumerate(boxplot['boxes']):
                if idx+1 == idx_:
                    c.set(color='red', alpha=0.5)
                
    def draw_boxplot(self, layer, idx, feature, mark, ax, col, ymin, ymax):
        #print(mark, col)
        x = np.arange(1,51)
        
        neuron_ = random.sample(idx, 1)[0]     # randomly select an unit
        I = feature[:, neuron_].squeeze()     
        
        if 'SI' in mark:
            #neuron_=11167
            #I = feature[:, neuron_].squeeze() 
            i_list = self.ID_neuron_encode_class_dict[layer][2]['SI_idx'][neuron_]
        elif 'MI' in mark:
            #neuron_=47344
            #I = feature[:, neuron_].squeeze() 
            i_list = self.ID_neuron_encode_class_dict[layer][3]['MI_idx'][neuron_]
        else:
            #neuron_=47271
            #I = feature[:, neuron_].squeeze() 
            i_list = []
        
        I = [I[i*self.num_samples:(i+1)*self.num_samples] for i in range(self.num_classes)]
        b = ax[col].boxplot(I, patch_artist=True, sym='+')
        self.paint_neuron_encode_boxplot(i_list, b)
        
        # [Encode dot]
        ax[col].scatter(list(set(x)-set(i_list)), [np.mean(I[_]) for _ in range(len(I)) if _ not in np.array(i_list)-1], linewidth=2, alpha=0.8, label=r'$\overline{x}<V_{th}$')
        ax[col].scatter(i_list, [np.mean(I[_]) for _ in range(len(I)) if _ in np.array(i_list)-1], color='red', linewidth=2, alpha=0.8, label=r'$\overline{x}>V_{th}$')
        
        for i in range(len(I)):
            if i in i_list:
                ax[col].vlines(i, 0, np.mean(I[i-1]), colors='orange', linestyles='--', linewidth=1.0, alpha=0.5)
            else:
                ax[col].vlines(i, 0, np.mean(I[i-1]), colors='blue', linestyles='--', linewidth=1.0, alpha=0.5)
 
        ax[col].plot(x, [np.mean(I)+2*np.std(I)]*50, color='red', alpha=0.75, label=r'$V_{th}$')
        
        ax[col].set_title(mark+' #'+str(neuron_), fontsize=16)
        ax[col].grid(axis='y')
        ax[col].set_xticks(x[np.where(x%10==0)], x[np.where(x%10==0)])
        ax[col].set_ylim([ymin, ymax])
        ax[col].set_xlim([min(x)-1, max(x)+1])
        ax[col].legend(fontsize=14)
        
        ax[col].tick_params(axis='both', labelsize=14)
        
        #plt.annotate(r'%.3f' % (min_val_loss), xy=(index_min_val_loss, min_val_loss), xycoords='data', xytext=(-100, +100), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->", color="blue", connectionstyle="arc3,rad=.2", alpha = 0.25))
    
    def single_neuron_boxplot(self):  
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        print('[Codinfo] Excuting single_neuron_boxplot...')
        
        save_path = self.dest+'Sigle_neuron_selectivity/'
        utils_.make_dir(save_path)
        
        for layer in tqdm(self.layers):
            feature_path = os.path.join(self.root, layer+'.pkl')
            for idx_path in self.idx_list:     # 问题同上
                if layer == idx_path.split('/')[-1].split('-')[0]: 
                    with open(feature_path, 'rb') as pkl:
                        feature = pickle.load(pkl)  # obtain full feture
                    pkl.close()
                            
                    sig_neuron_idx = np.loadtxt(idx_path, delimiter=',')
                    
                    if sig_neuron_idx.size == 1:
                        sig_neuron_idx = np.array([sig_neuron_idx])
                    sig_neuron_idx = list(map(int, sig_neuron_idx))
                    if sig_neuron_idx != []:
                        feature = feature[:, sig_neuron_idx] # [Notice] this "feature" is signeuron feature, write like this to save RAM
                        ymax = feature.max() 
                        ymin = feature.min()
                        
                        # target: recover the encode_classes_dict
                        SI_idx = list(self.ID_neuron_encode_class_dict[layer][2]['SI_idx'].keys())       #[Warning] this neuron_idx is also based on sig_neuron_feature
                        MI_idx = list(self.ID_neuron_encode_class_dict[layer][3]['MI_idx'].keys())
                                
                        SIMI_idx = SI_idx+MI_idx
                        non_sig_neuron_idx = list(set([i for i in range(feature.shape[1])])-set(SIMI_idx))
                
                        fig, axs = plt.subplots(1, 3, figsize=((30, 10)))       # make the blank fig with 3 subplots
                                
                        if SI_idx != []:    # add an empty detection
                            self.draw_boxplot(layer, SI_idx, feature, layer+' SI', axs, 0, ymin, ymax)
                        if MI_idx != []:
                            self.draw_boxplot(layer, MI_idx, feature, layer+' MI', axs, 1, ymin, ymax)
                        if non_sig_neuron_idx != []:
                            self.draw_boxplot(layer, non_sig_neuron_idx, feature, layer+ ' non Encode', axs, 2, ymin, ymax)
                                
                        plt.savefig(save_path+layer+'_single_neuron_selectivity.png', bbox_inches='tight', dpi=100)
                        plt.savefig(save_path+layer+'_single_neuron_selectivity.eps', bbox_inches='tight', dpi=100, format='eps')
                        plt.close()
                        
                        #plt.show()


if __name__ == "__main__":
    
    neuron_ = neuron.ParametricLIFNode
    neuron_name = 'ParametricLIF'
    
    spiking_model = spiking_resnet.__dict__['spiking_resnet18'](spiking_neuron=neuron_, num_classes=50, surrogate_function=surrogate.ATan(), detach_reset=True, mode='feature')
    functional.set_step_mode(spiking_model, step_mode='m') 
    layers, neurons, shapes = utils_.generate_resnet_layers_list(spiking_model, 'spiking_resnet18')
    
    #layers_ = [i for i in layers if 'fc' in i]
    #index_ = [layers.index(i) for i in layers_]
    #neurons_ = [neurons[i] for i in index_]
    #shapes_ = [shapes[i] for i in index_]
    #layers = layers_
    #neurons = neurons_

    root_dir = '/media/acxyle/Data/ChromeDownload/'

    SIMI_Analyzer = Selectivity_Analysis_SIMI(
                 root=os.path.join(root_dir, f'Identity_spiking_resnet18_{neuron_name}_ATan_T4_CelebA2622_Results/'), 
                 dest=os.path.join(root_dir, f'Identity_spiking_resnet18_{neuron_name}_ATan_T4_CelebA2622_Neuron/'), 
                 layers=layers, neurons=neurons, status=False)
    
    SIMI_Analyzer.SIMI_SVM()
    SIMI_Analyzer.plot_SVM_all()
    SIMI_Analyzer.plot_SVM_neuron()
    SIMI_Analyzer.plot_bins_number_and_percent_of_SIMI()
    SIMI_Analyzer.single_neuron_boxplot()
    
