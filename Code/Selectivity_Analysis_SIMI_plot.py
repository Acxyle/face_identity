#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 01:56:48 2023

@author: acxyle
"""

import os
import pickle
import matplotlib.pyplot as plt
import random
import numpy as np

import spiking_featuremap_utils

path = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/SIMI_cnt.pkl'
f = open(path, 'rb')
SIMI_dict = pickle.load(f)
f.close()

layer_sequence = spiking_featuremap_utils.layer_list_vgg16bn

dim_list = spiking_featuremap_utils.neuron_list_vgg16bn


def bins_percent_of_SIMI():
    total_neuron = {}
    
    for idx, layer in enumerate(layer_sequence):
        total_neuron[layer] = dim_list[idx]
    
    save_path = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/SIMI'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    x = layer_sequence
    y_list = [SIMI_dict[k] for k in layer_sequence]     # according to layer_sequence order to save the number of SI and MI
    # print(y_list)
    y1 = [len(list(item[0].values())[0]) for item in y_list]     # y1 records the number of SI for each layer
    y2 = [len(list(item[1].values())[0]) for item in y_list]     # y2 records the number of MI for each layer
    #y = [i + j for i, j in zip(y1, y2)]
    t = [total_neuron[k] for k in layer_sequence]
    percent_si = [i / j * 100 for i, j in zip(y1, t)]
    percent_mi = [i / j * 100 for i, j in zip(y2, t)]
    # print(y1)
    # print(y2)
    plt.figure(figsize=(30,10), dpi=200)
    p1 = plt.bar(x, y1, width=0.5)
    p2 = plt.bar(x, y2, width=0.5)
    plt.ylabel('Num of neurons')
    plt.xticks(rotation=45)
    plt.legend((p1[0], p2[0]), ('SI', 'MI'))
    plt.title('Stack plot for SI/MI num in each layer')
    plt.savefig(save_path + '/stack_plt_num.png', bbox_inches='tight')
    
    plt.figure(figsize=(30,10), dpi=200)
    p3 = plt.bar(x, percent_si, width=0.5)
    p4 = plt.bar(x, percent_mi, bottom=percent_si, width=0.5)
    plt.ylim((0, 100))
    plt.ylabel('Percentage')
    plt.xticks(rotation=90)
    plt.legend((p3[0], p4[0]), ('Singele_Identity(SI) Neuron', 'multiple_Identity(MI) Neuron'), frameon=False)
    plt.title('Stack plot for SI/MI percentage in each layer')
    plt.savefig(save_path + '/stack_plt_percentage.png', bbox_inches='tight')

# =====
feature_root = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Results/'
idx_root = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/'

feature_list = [(feature_root+f) for f in sorted(os.listdir(feature_root)) if f.split('.')[-1]=='pkl']
#print(feature_list)


def recover_encode_classes(neuron):
    global_mean = np.mean(neuron)
    global_std = np.std(neuron)
    threshold = global_mean + 2 * global_std
    d = [neuron[i*10:(i+1)*10] for i in range(50)]
    d = np.array(d)
    local_mean = np.mean(d, axis=1)
    encode_class = [i + 1 for i, mean in enumerate(local_mean) if mean > threshold]
    
    return encode_class

def paint_neuron_encode_boxplot(neuron_list, boxplot):
    
    for idx, c in enumerate(boxplot['boxes']):
        c.set(color='gray', alpha=0.5)
        
    for idx_ in neuron_list:
        for idx, c in enumerate(boxplot['boxes']):
            if idx+1 == idx_:
                c.set(color='red', alpha=0.5)
            
def draw_boxplot(idx, feature, mark, ax, col, ymin, ymax):
    #print(mark, col)
    neuron = random.sample(idx, 1)
    
    I = feature[:, neuron].squeeze()
    i_list = recover_encode_classes(I)
    I = [I[i*10:(i+1)*10] for i in range(50)]
    b = ax[col].boxplot(I, patch_artist=True, sym='+')
    paint_neuron_encode_boxplot(i_list, b)
    ax[col].set_title(mark+' #'+str(neuron[0]))
    ax[col].grid(axis='y')
    ax[col].set_ylim([ymin, ymax])


def single_neuron_boxplot():    # [Notice] Time consuming calculation
    #FIXME
    #need to consider if the classes number is 0
    with open(idx_root + '/SIMI_cnt.pkl', 'rb') as f:
        SIMI_dict = pickle.load(f)
    f.close()
    print(SIMI_dict.keys())
    
    for layer in layer_sequence:
        print(layer)
        save_path = idx_root+'Sigle_neuron_selectivity/'+layer
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for feature_path in feature_list:
            if feature_path.split('/')[-1].split('.')[0] == layer:
                with open(feature_path, 'rb') as pkl:
                    feature = pickle.load(pkl)  # obtain full feture
                pkl.close()
                
                sig_neuron = np.loadtxt(idx_root+layer+'-neuronIdx.csv', delimiter=',')
                feature = feature[:,list(map(int, sig_neuron))] # [Notice] this "feature" is signeuron feature, write like this to save RAM
                ymax = feature.max() 
                ymin = feature.min()
                
                SI_idx = SIMI_dict[layer][0]['SI_idx']
                MI_idx = SIMI_dict[layer][1]['MI_idx']
                
                SIMI_idx = SI_idx+MI_idx
                NonID_idx = [i for i in range(feature.shape[1])]
                for e in SIMI_idx:    # time consuming action
                    NonID_idx.remove(e)
                
                print(len(NonID_idx), len(SI_idx), len(MI_idx))
                
                neuron_ni = random.sample(NonID_idx, 1)
                
                NonID = feature[:, neuron_ni].squeeze()
                NonID = [NonID[i*10:(i+1)*10] for i in range(50)]

                fig, axs = plt.subplots(1, 3, figsize=((30, 10)))       # make the blank fig with 3 subplots
                
                if SI_idx != []:    # add an empty detection
                    draw_boxplot(SI_idx, feature, layer+' SI', axs, 0, ymin, ymax)
                if MI_idx != []:
                    draw_boxplot(MI_idx, feature, layer+' MI', axs, 1, ymin, ymax)
                if NonID_idx != []:
                    b_ni = axs[2].boxplot(NonID, patch_artist=True, sym='+')
                
                for idx, b in enumerate(b_ni['boxes']):
                    b.set(color='gray', alpha=0.5)
                axs[2].set_title(layer+' NonID #'+str(neuron_ni[0]))
                axs[2].grid(axis='y')
                axs[2].set_ylim([ymin, ymax])
                plt.savefig(save_path + '/' + 'single_neuron_selectivity.png', bbox_inches='tight', dpi=100)
                #plt.show()


if __name__ == "__main__":
    #bins_percent_of_SIMI()
    single_neuron_boxplot()