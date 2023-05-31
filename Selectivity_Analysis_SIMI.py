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

import spiking_vgg, spiking_resnet, sew_resnet
from spikingjelly.activation_based import surrogate, neuron, functional

import resnet
import utils_


class Selectivity_Analysis_SIMI():
    def __init__(self, feature_root='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Results/', 
                 idx_root='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/',
                 num_classes=50, num_samples=10, layers=None, neurons=None, status=False):
        
        if layers == None or neurons == None:
            raise RuntimeError('[Codwarning] invalid layers and neurons')
            
        self.layers = layers
        self.neurons = neurons
        
        self.feature_root = feature_root
        self.idx_root = idx_root
        
        # [notice] close this for local test
        self.feature_list = [(self.feature_root+f) for f in sorted(os.listdir(self.feature_root)) if f.split('.')[-1]=='pkl']
        self.idx_list = [(self.idx_root+f) for f in sorted(os.listdir(self.idx_root)) if 'neuronIdx'  in f.split('-')[-1]]

        self.num_classes = num_classes
        self.num_samples = num_samples

        self.save_path = idx_root+'SIMI/'
        
        utils_.make_dir(self.save_path)
        
        if status == True:
            self.SIMI_SVM(self)
            self.bins_percent_of_SIMI(self)
            self.single_neuron_boxplot(self)
        
    def load_ID_neuron_encode_class_dict(self):
        with open(self.idx_root +'Frequency/ID_neuron_encode_class_dict.pkl', 'rb') as f:
            ID_neuron_encode_class_dict = pickle.load(f)
        f.close()
        
        return ID_neuron_encode_class_dict
    
    def recover_SIMI_dict(self):
        ID_neuron_encode_class_dict = self.load_ID_neuron_encode_class_dict()
        SIMI_dict = {}
        for k, v in ID_neuron_encode_class_dict.items():  # for each layer
            SI_idx = list(v[2]['SI_idx'].keys())
            MI_idx = list(v[3]['MI_idx'].keys())   
            SIMI_dict.update({k: [SI_idx, MI_idx]})
        
        return SIMI_dict
    
    def SIMI_SVM(self, verbose=False):
        '''
        #TODO
        the generated results should be saved for further plotting 
        '''
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
            feature_path = os.path.join(self.feature_root, layer+'.pkl')
            for idx_path in self.idx_list:
                if layer == idx_path.split('/')[-1].split('-')[0]: 
                    #idx_path = os.path.join(self.idx_root,layer+'-neuronIdx.csv')     # 这种指定方法简单直接，但是遇到需要根据不同条件diy文件名时则难以泛化
                    with open(feature_path, 'rb') as pkl:
                        feature = pickle.load(pkl)
                    pkl.close()
                    
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
                    
                    # 这里之后也改成表格输出
                    if verbose:
                        print('[Codinfo] layer: {}, total neurons: {}, ID: {}, non ID: {}, SI: {}, MI: {}, SI+MI: {}'.format(
                            layer, feature.shape[1], len(sig_neuron_idx), len(non_sig_neuron_idx), len(SI_idx), len(MI_idx), len(SIMI_idx)),
                              '\n[Codinfo] all_acc: {:.2f}%, ID_acc: {:.2f}%, nonID_acc: {:.2f}%, SIMI_acc: {:.2f}%, SI_acc: {:.2f}%, MI_acc: {:.2f}%, '.format(
                            all_acc, ID_acc, nonID_acc, SIMI_acc, SI_acc, MI_acc))
        
        # [notice] save the very time consuming data for future use
        with open(self.idx_root + 'SIMI/SIMI.pkl',  'wb') as f:
            pickle.dump([all_acc_dict,ID_acc_dict, nonID_acc_dict, SIMI_acc_dict, SI_acc_dict, MI_acc_dict], f, protocol=-1)
        f.close()
        
        # 1. all calculation
        self.plot_SVM_all_calculation()
        
        # 2. imaginary neuron
        self.plot_SVM_neuron()
    
    def segment_dict_neuron(self, input_dict):
        input_dict = {i:input_dict[i] for i in list(input_dict.keys()) if 'neuron' in i or 'pool' in i or 'fc' in i}
        return input_dict

    def plot_SVM_all_calculation(self):
        print('[Codinfo] Executing SIMI plotting for selected neuron...')
        with open(self.idx_root + 'SIMI/SIMI.pkl', 'rb') as f:
            [all_acc_dict,ID_acc_dict, nonID_acc_dict, SIMI_acc_dict, SI_acc_dict, MI_acc_dict] = pickle.load(f)
        f.close()
        
        layer_list = self.layers
        x = layer_list
        y_a = [all_acc_dict[k] for k in layer_list]
        y_i = [ID_acc_dict[k] for k in layer_list]
        y_n = [nonID_acc_dict[k] for k in layer_list]
        y = [SIMI_acc_dict[k] for k in layer_list]
        y_s = [SI_acc_dict[k] for k in layer_list]
        y_m = [MI_acc_dict[k] for k in layer_list]
        
        plt.figure(figsize=(int(len(self.layers)/2),10), dpi=200)
        plt.rcParams.update({'font.size': 20})
        model_name = self.idx_root.split('/')[-2].split('_')[1]
        
        plt.plot(x, y, 'blue', label='SI+MI')
        plt.plot(x, y_s, 'green', label='SI')
        plt.plot(x, y_m, 'purple', label='MI')
        plt.plot(x, y_a, 'red', label='all neuron')
        plt.plot(x, y_i, 'orange', label='ID selective')
        plt.plot(x, y_n, 'teal', label='non ID selective')
        
        plt.ylim((0, 100))
        plt.legend()
        plt.xticks(rotation='vertical')
        plt.ylabel('Classification Accuracy')
        plt.title(f'Neuron Decoding Performance (all calculation) - {model_name}')
        plt.savefig(self.idx_root + 'SIMI/SIMI_acc_all_calculation.png', bbox_inches='tight')
        plt.savefig(self.idx_root + 'SIMI/SIMI_acc_all_calculation.eps', format='eps', bbox_inches='tight')
        plt.close()
        
    def plot_SVM_neuron(self):
        print('[Codinfo] Executing SIMI plotting for selected neuron...')
        with open(self.idx_root + 'SIMI/SIMI.pkl', 'rb') as f:
            [all_acc_dict,ID_acc_dict, nonID_acc_dict, SIMI_acc_dict, SI_acc_dict, MI_acc_dict] = pickle.load(f)
        f.close()
        
        all_acc_dict = self.segment_dict_neuron(all_acc_dict)
        ID_acc_dict = self.segment_dict_neuron(ID_acc_dict)
        nonID_acc_dict = self.segment_dict_neuron(nonID_acc_dict)
        SIMI_acc_dict = self.segment_dict_neuron(SIMI_acc_dict)
        SI_acc_dict = self.segment_dict_neuron(SI_acc_dict)
        MI_acc_dict = self.segment_dict_neuron(MI_acc_dict)
        
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
        model_name = self.idx_root.split('/')[-2].split('_')[1]
        
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
        plt.savefig(self.idx_root + 'SIMI/SIMI_acc.png', bbox_inches='tight')
        plt.savefig(self.idx_root + 'SIMI/SIMI_acc.eps', format='eps', bbox_inches='tight')
        plt.close()
    
    def bins_percent_of_SIMI(self):
        print('[Codinfo] Executing SIMI percent plot...')
        
        # [notice]
        model_name = 'Spiking_VGG16_bn'
        
        SIMI_dict = self.recover_SIMI_dict()
        
        '''
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
        
        plt.rcParams.update({'font.size': 12})
        # 1. all calculation
        # --- number
        plt.figure(figsize=(int(len(self.layers)/2),10), dpi=200)
        p1 = plt.bar(x, y1, width=0.5)
        p2 = plt.bar(x, y2, width=0.5)
        plt.ylabel('Num of neurons')
        plt.xticks(rotation='vertical')     # [Warning] default vertical, waitingte rewrite a adjustable function
        plt.legend((p1[0], p2[0]), ('SI', 'MI'))
        plt.title('Stack plot for SI/MI num in each layer')
        plt.savefig(self.save_path + '/stack_plt_num.png', bbox_inches='tight')
        plt.savefig(self.save_path + '/stack_plt_num.eps', format='eps', bbox_inches='tight')
        plt.close()
        
        # --- percentages
        plt.figure(figsize=(int(len(self.layers)/2),10), dpi=200)
        p3 = plt.bar(x, percent_si, width=0.5)
        p4 = plt.bar(x, percent_mi, bottom=percent_si, width=0.5)
        plt.ylim((0, 100))
        plt.ylabel('Percentage')
        plt.xticks(rotation='vertical')
        plt.legend((p3[0], p4[0]), ('Singele_Identity(SI) Neuron', 'multiple_Identity(MI) Neuron'), frameon=False)
        plt.title('Stack plot for SI/MI percentage in each layer')
        plt.savefig(self.save_path + '/stack_plt_percentage.png', bbox_inches='tight')
        plt.savefig(self.save_path + '/stack_plt_percentage.eps', format='eps', bbox_inches='tight')
        plt.close()
        
        # 2. imaginary neuron only
        layers_neuron = [i for i in self.layers if 'neuron' in i or 'pool' in i or 'fc_3' in i]
        x = layers_neuron
        y_list = [SIMI_dict[k] for k in layers_neuron] 
        y1 = [len(item[0]) for item in y_list]
        y2 = [len(item[1]) for item in y_list]
        t = [total_neuron[k] for k in layers_neuron]
        percent_si = [i / j * 100 for i, j in zip(y1, t)]
        percent_mi = [i / j * 100 for i, j in zip(y2, t)]
        
        # --- number
        plt.figure()
        p1 = plt.bar(x, y1, width=0.5)
        p2 = plt.bar(x, y2, width=0.5)
        plt.ylabel('Num of neurons')
        plt.xticks(rotation='vertical')     # [Warning] default vertical, waitingte rewrite a adjustable function
        plt.legend((p1[0], p2[0]), ('SI', 'MI'))
        plt.title('Stack plot for SI/MI num in each layer')
        plt.savefig(self.save_path + '/stack_plt_num_neuron.png', bbox_inches='tight')
        plt.savefig(self.save_path + '/stack_plt_num_neuron.eps', format='eps', bbox_inches='tight')
        plt.close()

        # --- percentages
        plt.figure()
        p3 = plt.bar(x, percent_si, width=0.5)
        p4 = plt.bar(x, percent_mi, bottom=percent_si, width=0.5)
        plt.ylim((0, 100))
        plt.ylabel('Percentage')
        plt.xticks(rotation='vertical')
        plt.legend((p3[0], p4[0]), ('Singele_Identity(SI) Neuron', 'multiple_Identity(MI) Neuron'), frameon=False)
        plt.title('Stack plot for SI/MI percentage in each layer')
        plt.savefig(self.save_path + '/stack_plt_percentage_neuron.png', bbox_inches='tight')
        plt.savefig(self.save_path + '/stack_plt_percentage_neuron.eps', format='eps', bbox_inches='tight')
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
        neuron = random.sample(idx, 1)[0]
        
        I = feature[:, neuron].squeeze()
        ID_neuron_encode_class_dict = self.load_ID_neuron_encode_class_dict()
        if 'SI' in mark:
            i_list = ID_neuron_encode_class_dict[layer][2]['SI_idx'][neuron]
        elif 'MI' in mark:
            i_list = ID_neuron_encode_class_dict[layer][3]['MI_idx'][neuron]
        
        I = [I[i*self.num_samples:(i+1)*self.num_samples] for i in range(self.num_classes)]
        b = ax[col].boxplot(I, patch_artist=True, sym='+')
        self.paint_neuron_encode_boxplot(i_list, b)
        ax[col].set_title(mark+' #'+str(neuron))
        ax[col].grid(axis='y')
        ax[col].set_ylim([ymin, ymax])
    
    def single_neuron_boxplot(self):  
        print('[Codinfo] Excuting single_neuron_boxplot...')
        ID_neuron_encode_class_dict = self.load_ID_neuron_encode_class_dict()
        save_path = self.idx_root+'Sigle_neuron_selectivity/'
        utils_.make_dir(save_path)
        
        for layer in tqdm(self.layers):
            feature_path = os.path.join(self.feature_root, layer+'.pkl')
            for idx_path in self.idx_list:     # FIXME 问题同上
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
                        SI_idx = list(ID_neuron_encode_class_dict[layer][2]['SI_idx'].keys())       #[Warning] this neuon_idx is also based on sig_neuron_feature
                        MI_idx = list(ID_neuron_encode_class_dict[layer][3]['MI_idx'].keys())
                                
                        SIMI_idx = SI_idx+MI_idx
                        non_sig_neuron_idx = list(set([i for i in range(feature.shape[1])])-set(SIMI_idx))
                
                        fig, axs = plt.subplots(1, 3, figsize=((30, 10)))       # make the blank fig with 3 subplots
                                
                        if SI_idx != []:    # add an empty detection
                            self.draw_boxplot(layer, SI_idx, feature, layer+' SI', axs, 0, ymin, ymax)
                        if MI_idx != []:
                            self.draw_boxplot(layer, MI_idx, feature, layer+' MI', axs, 1, ymin, ymax)
                        if non_sig_neuron_idx != []:
                            neuron_ni = random.sample(non_sig_neuron_idx, 1)
                            nonID = feature[:, neuron_ni].squeeze()
                            nonID = [nonID[i*self.num_samples:(i+1)*self.num_samples] for i in range(self.num_classes)]
                            b_ni = axs[2].boxplot(nonID, patch_artist=True, sym='+')
                                
                        for idx, b in enumerate(b_ni['boxes']):
                            b.set(color='gray', alpha=0.5)
                        axs[2].set_title(layer+' nonID #'+str(neuron_ni[0]))
                        axs[2].grid(axis='y')
                        axs[2].set_ylim([ymin, ymax])
                        plt.savefig(save_path+layer+'_single_neuron_selectivity.png', bbox_inches='tight', dpi=100)
                        plt.savefig(save_path+layer+'_single_neuron_selectivity.eps', bbox_inches='tight', dpi=100, format='eps')
                        plt.close()


if __name__ == "__main__":
    
    spiking_model = spiking_vgg.__dict__['spiking_vgg16_bn'](spiking_neuron=neuron.LIFNode, num_classes=50, surrogate_function=surrogate.ATan(), detach_reset=True)
    functional.set_step_mode(spiking_model, step_mode='m')
    layers, neurons, shapes = utils_.generate_vgg_layers(spiking_model, 'spiking_vgg16_bn')

    layers = [i for i in layers if 'L2_maxpool' in i]
    neurons = [neurons[4]]

    SIMI_Analyzer = Selectivity_Analysis_SIMI(
                 feature_root='/media/acxyle/Data/ChromeDownload/Identity_SpikingVGG16bn-LIF-CelebA2622_Results/', 
                 idx_root='/media/acxyle/Data/ChromeDownload/Identity_SpikingVGG16bn_LIF_CelebA2622_Neuron/',
                 layers=layers, neurons=neurons, status=False)
    
    #SIMI_Analyzer.plot_SVM_all_calculation()
    #SIMI_Analyzer.plot_SVM_neuron()
    #SIMI_Analyzer.bins_percent_of_SIMI()
    SIMI_Analyzer.single_neuron_boxplot()
    