#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:33:39 2023

@author: acxyle

 --- notice
the function in this file is not real function, just divide for different use
 --- waiting to rewrite for code reuse

"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import random

import spiking_featuremap_utils

feature_root = '/media/acxyle/Data/ChromeDownload/Identity_VGG_Results/'
idx_root = '/media/acxyle/Data/ChromeDownload/Identity_VGG_Neuron/'

feature_list = [(feature_root+f) for f in sorted(os.listdir(feature_root)) if f.split('.')[-1]=='pkl']
#print(feature_list)

idx_list = [(idx_root+f) for f in sorted(os.listdir(idx_root)) if 'neuronIdx'  in f.split('-')[-1]]
#print(idx_list)

layers = spiking_featuremap_utils.layer_list_vgg16bn

dim_list = spiking_featuremap_utils.neuron_list_vgg16bn

save_path = feature_root + '/Freq'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def obtrain_encode_class_dict():
    encode_class_dict = {}
    num_per_class = 10
    for feature_path in tqdm(feature_list):
        print(feature_path.split('/')[-1].split('.')[0])
        with open(feature_path, 'rb') as pkl:
            feature = pickle.load(pkl)
            for idx_path in idx_list:
                if feature_path.split('/')[-1].split('.')[0] == idx_path.split('/')[-1].split('-')[0]:
                    print(idx_path)
                    sig_neuron_idx = list(map(int, np.loadtxt(idx_path, delimiter=',')))    
                    sig_neuron = feature[:,sig_neuron_idx]  # obtain sig_neuron
                    print(sig_neuron.shape)
                    row, col = sig_neuron.shape
                    cnt = 0
                    encode_class = []
                    for i in range(col):  # loop in neurons
                        neuron = sig_neuron[:, i]
                        global_mean = np.mean(neuron)
                        global_std = np.std(neuron)
                        threshold = global_mean + 2 * global_std
                        d = [neuron[i * num_per_class: i * num_per_class + num_per_class] for i in range(int(row / num_per_class))]
                        d = np.array(d)
                        local_mean = np.mean(d, axis=1)
                        for j, mean in enumerate(local_mean):       # for each ID
                            if mean > threshold:
                                encode_class.append(j + 1)       # record which class has been encoded
                    encode_class_dict.update({feature_path.split('/')[-1].split('.')[0]: encode_class})
                    #print(encode_class_dict)
    with open(os.path.join(feature_root, 'encode_class_dict.pkl'), 'wb') as f: # save the relationship betwwen layer and encoded classes
        pickle.dump(encode_class_dict, f)

def load_encode_class_dict():
    with open(os.path.join(feature_root, 'encode_class_dict.pkl'), 'rb') as f:
        encode_class_dict = pickle.load(f)
        encode_class_dict['fc_6'] = encode_class_dict.pop('fc6')
        encode_class_dict['fc_7'] = encode_class_dict.pop('fc7')
        encode_class_dict['fc_8'] = encode_class_dict.pop('fc8')
        for k in encode_class_dict.keys():
            print(k)

def draw_encode_frequency():
    # general figure for encoding frequency
    freq_dic = {}
    for idx, layer in enumerate(layers):    # for each layer
        freq = {}
        encode_class_list = encode_class_dict[layer]    # obtain the encoded classes
        for item in encode_class_list:  # for each class
            if (item in freq):  # update the dict or add new k-v pair
                freq[item] += 1 
            else:
                freq[item] = 1
        freq = {k: v / dim_list[idx] for k, v in freq.items()}    # convert v of  abs avlue to ratio
        freq = dict(sorted(freq.items(), key=lambda item: item[0]))     # sort
        freq_dic.update({layer: freq})
    a = pd.DataFrame.from_dict(freq_dic)
    
    plt.figure()
    im = plt.matshow(a, aspect='auto')
    plt.colorbar(im, fraction=0.12, pad=0.04)
    plt.xlabel('Layers')
    plt.ylabel('IDs')
    plt.title('Ecode Frequency for Each Layer')
    plt.savefig(save_path + '/' + 'Ecode_Frequency_for_Each_Layer.png', bbox_inches='tight', dpi=100)

def draw_encode_frequency_for_each_layer():
    # encoding frequency for each layer
    occ_list = []
    for layer in layers:    # for each layer
        occurrences = []
        for i in range(50):     # for each ID
            occ = encode_class_dict[layer].count(i + 1)  # calculate the frequency of each ID [one value]
            occurrences.append(occ) # store frequency for each ID [list of 50]
        occ_list.append(occurrences)    # merge the frequency info for all layers [list of len(layers)]
        x = np.arange(1, 51)
        plt.figure()
        plt.bar(x, occurrences, width=0.5)
        plt.xticks(np.arange(0, 51, step=2))
        plt.xlabel('IDs')
        plt.ylabel('Frequrency')
        plt.title('Encoded ID frequency: ' + layer + '\nTh: 2std')
        plt.savefig(save_path + '/' + layer + '_sigFreq.png', bbox_inches='tight', dpi=100)

def draw_merged_encode_frequency_for_each_layer():
    # encoding frequency for each layer, basiccaly this is the merged version of the last one
    fig, axs = plt.subplots(9, 5, figsize=((20, 20)))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    x = np.arange(1, 51)
    
    cnt_row = 0
    cnt_col = 0
    
    for layer in layers:    # for each layer
        occurrences = []
        for i in range(50):     # for each ID
            occ = encode_class_dict[layer].count(i + 1)
            occurrences.append(occ)
        axs[cnt_row, cnt_col].bar(x, occurrences, width=0.5)
        axs[cnt_row, cnt_col].set_title(layer, fontsize=14)
        
        cnt_col += 1        # set subplot location
        if cnt_col > 4:
            cnt_col = 0
            cnt_row += 1
            
    for ax in axs.flat:
        ax.label_outer()
    for ax in axs.flat:
        ax.set(xlabel='IDs', ylabel='Freq')
    plt.savefig(save_path + '/' + 'sigFreqAll.png', bbox_inches='tight', dpi=100)
    
    
def draw_single_neuron_response():
    
    save_path = feature_root + '/Mean_check'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    fig, axs = plt.subplots(9, 5, figsize=((20, 20)))   #一个有多少张子图的fig
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    x = np.arange(1, 51)
    cnt_row = 0
    cnt_col = 0

    encode_class_dict = {}
    num_per_class = 10
    for feature_path in tqdm(feature_list):
        print(feature_path.split('/')[-1].split('.')[0])
        with open(feature_path, 'rb') as pkl:
            feature = pickle.load(pkl)
            for idx_path in idx_list:
                if feature_path.split('/')[-1].split('.')[0] == idx_path.split('/')[-1].split('-')[0]:
                    print(idx_path)
                    sig_neuron_idx = list(map(int, np.loadtxt(idx_path, delimiter=',')))    
                    sig_neuron = feature[:,sig_neuron_idx]  # obtain sig_neuron
                    print(sig_neuron.shape)
                    
                    layer = feature_path.split('/')[-1].split('.')[0]      # 抽取 layer 名称
                    num_per_class = 10
                
                    row, col = sig_neuron.shape
                    check_neuron = random.choice(range(col))
                    print('Now check', layer, ': #', check_neuron, '\n')
                    neuron_vector = sig_neuron[:, check_neuron]
                    ID_vector_list = [neuron_vector[i * num_per_class: i * num_per_class + num_per_class] for i in range(int(row / num_per_class))] # list for each ID [list 50]
                    ID_vector_list = np.array(ID_vector_list)
                    mean_list = np.mean(ID_vector_list, axis=1) # 50个 mean
                    std_list = np.std(ID_vector_list, axis=1)   
                    se_list = std_list / np.sqrt(num_per_class) # standard error， 什么地方用到它？
                    x = np.arange(50) + 1
                
                    axs[cnt_row, cnt_col].bar(x, mean_list, width=0.5)      # 画子图
                    axs[cnt_row, cnt_col].set_title(layer + ' # ' + str(check_neuron), fontsize=14)
                    cnt_col += 1
                    if cnt_col > 4:
                        cnt_col = 0
                        cnt_row += 1    
                    
        for ax in axs.flat:
            ax.label_outer()
        for ax in axs.flat:
            ax.set(xlabel='IDs', ylabel='local mean')
        plt.savefig(save_path + '/' + 'meanCheckAll.png', bbox_inches='tight', dpi=100)
    
if __name__ == "__main__":
    draw_single_neuron_response()
