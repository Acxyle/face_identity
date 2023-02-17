#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 22:13:05 2023

@author: acxyle

此代码的输出是 SI 和 MI 的 list

#TODO
当前 dict 使用的是 encode Ver 1.0， 基于 Ver 2.0 应可以省略第一部分

"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import random

import spiking_featuremap_utils

feature_root = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Results/'
idx_root = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/'

feature_list = [(feature_root+f) for f in sorted(os.listdir(feature_root)) if f.split('.')[-1]=='pkl']
#print(feature_list)

idx_list = [(idx_root+f) for f in sorted(os.listdir(idx_root)) if 'neuronIdx'  in f.split('-')[-1]]
#print(idx_list)

spiking_featuremap_utils.make_dir(os.path.join(idx_root, 'SIMI/'))

def generate_SIMI_dict():
    SIMI_dict = {}
    num_per_class = 10
    for feature_path in tqdm(feature_list):
        print('\n',feature_path.split('/')[-1].split('.')[0])
        with open(feature_path, 'rb') as pkl:
            feature = pickle.load(pkl)
            for idx_path in idx_list:
                if feature_path.split('/')[-1].split('.')[0] == idx_path.split('/')[-1].split('-')[0]:
                    print(idx_path)
                    sig_neuron_idx = list(map(int, np.loadtxt(idx_path, delimiter=',')))    
                    sig_neuron = feature[:,sig_neuron_idx]  # obtain sig_neuron
    
                    row, col = sig_neuron.shape
                    print(feature_path.split('/')[-1].split('.')[0] + ':', col, 'selective neuron intotal')
                    SI_idx = []
                    MI_idx = []
                    for i in range(col):    # for each neuron. [Notice] This idx is from sig_neuron, thus recover sig_neuron first instead of full_neuron
                        neuron = sig_neuron[:, i]
                        global_mean = np.mean(neuron)
                        global_std = np.std(neuron)
                        threshold = global_mean + 2 * global_std
                        d = [neuron[i * num_per_class:i * num_per_class + num_per_class] for i in range(int(row / num_per_class))]
                        d = np.array(d)
                        local_mean = np.mean(d, axis=1)
                        encode_class = [i + 1 for i, mean in enumerate(local_mean) if mean > threshold]
                        if not encode_class == []:
                            if len(encode_class) == 1:
                                SI_idx.append(i)
                            else:
                                MI_idx.append(i)
                    print(len(SI_idx)+len(MI_idx), 'neuron pass the threhold')
                    print('SI:', len(SI_idx))
                    print('MI:', len(MI_idx), '\n')
                    SIMI_dict.update({feature_path.split('/')[-1].split('.')[0]: [{'SI_idx': SI_idx}, {'MI_idx': MI_idx}]})
                    
    with open(idx_root + '/SIMI_cnt.pkl', 'wb') as f:
        pickle.dump(SIMI_dict, f, protocol=-1)

def SIMI_SVM():
    """
    revocer the SIMI feature
    """
    with open(idx_root + '/SIMI_cnt.pkl', 'rb') as f:
        SIMI_dict = pickle.load(f)
    f.close()
    print(SIMI_dict.keys())
    
    SIMI_acc_dict = {}
    SI_acc_dict = {}
    MI_acc_dict = {}
    label = spiking_featuremap_utils.makeLabels(10, 50)
    
    for feature_path in tqdm(feature_list):
        print('\n',feature_path.split('/')[-1].split('.')[0])
        with open(feature_path, 'rb') as pkl:
            feature = pickle.load(pkl)
        pkl.close()
        SI_idx = SIMI_dict[feature_path.split('/')[-1].split('.')[0]][0]['SI_idx']
        MI_idx = SIMI_dict[feature_path.split('/')[-1].split('.')[0]][1]['MI_idx']
        SIMI_idx = SI_idx + MI_idx
        
        SIMI_acc = spiking_featuremap_utils.SVM_classification(feature[:, SIMI_idx], label)
        SIMI_acc_dict.update({feature_path.split('/')[-1].split('.')[0]: SIMI_acc})
        
        SI_acc = spiking_featuremap_utils.SVM_classification(feature[:, SI_idx], label)
        SI_acc_dict.update({feature_path.split('/')[-1].split('.')[0]: SI_acc})
        
        MI_acc = spiking_featuremap_utils.SVM_classification(feature[:, MI_idx], label)
        MI_acc_dict.update({feature_path.split('/')[-1].split('.')[0]: MI_acc})
        
        print('\nID_Accuracy: %d %%' % (100 * SIMI_acc))
    
    layer_list = spiking_featuremap_utils.layer_list_vgg16bn
    x = layer_list
    y = [SIMI_acc_dict[k] for k in layer_list]
    y_s = [SI_acc_dict[k] for k in layer_list]
    y_m = [MI_acc_dict[k] for k in layer_list]
    
    plt.figure(figsize=(30,10), dpi=200)
    
    plt.plot(x, y, 'b', label='SI+MI')
    plt.plot(x, y_s, 'green', label='SI')
    plt.plot(x, y_m, 'purple', label='MI')
    plt.ylim((0, 1))
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylabel('Classification Accuracy')
    plt.title('SIMI_neuron Decoding Accuracy')
    plt.savefig(idx_root + 'SIMI/SIMI_acc.png')

if __name__ == "__main__":
    #generate_SIMI_dict()
    SIMI_SVM()