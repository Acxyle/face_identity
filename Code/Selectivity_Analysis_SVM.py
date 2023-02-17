#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:54:43 2023

@author: acxyle
"""

'''Comparison decoding ability(classification accuracy)
    between ID-selective and Non_ID-selective neuron
    
    这玩意儿怎么好像比 ANOVA 还费时间
    这玩意儿计算时的顺序也不太对
    
    TODO:
        1. Seperate ID and Non-ID feature for each layer
            use binary mask
        2. Pick a certain layer to do the classification
            tried Conv5_3, FC8
            and the rest of layers
        3. Pick a method to do the classification
            SVM
        4. Compare and plot the accuracy between ID and Non-ID selective neuron 
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

import spiking_featuremap_utils


def Selectivity_Analysis_SVM(feature_root = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Results/', 
                             Idx_root = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/',
                             dest = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/SVM/'
                             ):
    
    if not os.path.exists(dest):
        os.makedirs(dest)

    fullMatrix_list = [(feature_root+f) for f in sorted(os.listdir(feature_root)) if f.split('.')[-1]=='pkl']

    target_list = spiking_featuremap_utils.layer_list_vgg16bn

    full_acc_dict = {}
    ID_acc_dict = {}
    nonID_acc_dict = {}
    
    for layer in target_list:
        for feature_path in tqdm(fullMatrix_list):    # for each layer
            if layer == feature_path.split('/')[-1].split('.')[0]:
                with open(feature_path, 'rb') as pkl:         # original feature map
                    full_matrix = pickle.load(pkl)
                sig_neuron_idx = np.loadtxt(Idx_root+layer+'-neuronIdx.csv', delimiter=',')
                mask = np.array([(i in sig_neuron_idx) for i in range(full_matrix.shape[1])])
                ID_matrix = full_matrix[:, mask]
                nonID_matrix = full_matrix[:, ~mask]
    
                label = []
                for i in range(50):
                    label += [i + 1] * 10
                label = np.array(label)
                label = np.expand_dims(label, axis=1)
    
                full_acc = spiking_featuremap_utils.SVM_classification(full_matrix, label.ravel())
                full_acc_dict.update({layer: full_acc})
                ID_acc = spiking_featuremap_utils.SVM_classification(ID_matrix, label.ravel())
                ID_acc_dict.update({layer: ID_acc})
                nonID_acc = spiking_featuremap_utils.SVM_classification(nonID_matrix, label.ravel())
                nonID_acc_dict.update({layer: nonID_acc})
                
                print("{} - full_feature: {}, acc: {:.2f}% | Selective: {}, acc: {:.2f}% | Non-selective: {}, acc: {:.2f}%".format(
                    layer, full_matrix.shape[1], 100*full_acc, ID_matrix.shape[1] , 100*ID_acc, nonID_matrix.shape[1], 100*nonID_acc))


    x = target_list
    y = [full_acc_dict[k] for k in target_list]
    y1 = [ID_acc_dict[k] for k in target_list]
    y2 = [nonID_acc_dict[k] for k in target_list]
    plt.figure()
    plt.plot(x, y, 'b', label='full')
    plt.plot(x, y1, 'r', label='ID')
    plt.plot(x, y2, 'b', label='NonID')
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylabel('Classification Accuracy')
    plt.title('Decoding Accuracy')
    plt.savefig(dest + '/' + 'acc.png')


if __name__ == "__main__":
    Selectivity_Analysis_SVM()
