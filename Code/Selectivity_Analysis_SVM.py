#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:54:43 2023

@author: acxyle
"""

'''Comparison decoding ability(classification accuracy)
    between ID-selective and Non_ID-selective neuron

    TODO:
        1. Seperate ID and Non-ID feature for each layer
            use binary mask
        2. Pick a certain layer to do the classification
            tried Conv5_3, FC8
            and the rest of layers
        3. Pick a method to do the classification
            SVM
        4. Compare and plot the accuracy between ID and Non-ID selective neuron
    === 看看更新后的SVM能不能用 
'''

import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
import pickle

import utils


def main():
    feature_root = '/media/acxyle/Data/ChromeDownload/Identity_VGG_Results/'
    Idx_root = '/media/acxyle/Data/ChromeDownload/Identity_VGG_Neuron/'
    dest = '/media/acxyle/Data/ChromeDownload/Identity_VGG_Neuron/SVM/'
    if not os.path.exists(dest):
        os.makedirs(dest)

    fullMatrix_list = [(feature_root+f) for f in sorted(os.listdir(feature_root)) if f.split('.')[-1]=='pkl']
    #print(fullMatrix_list)
    target_list = [
                   'conv_2_1','conv_2_2', 
                   'conv_3_1','conv_3_2','conv_3_3',
                   'conv_4_1','conv_4_2','conv_4_3',
                   'conv_5_1','conv_5_2','conv_5_3',
                   'fc_6', 'ReLU_6',
                   'fc_7', 'ReLU_7',
                   'fc_8']
    #target_list = ['fc_7', 'fc_8']
    # target_list = []
    # sample_num = 50 * 10

    full_acc_dict = {}
    ID_acc_dict = {}
    nonID_acc_dict = {}

    for feature_path in fullMatrix_list:    # for each layer
        layer = feature_path.split('/')[-1].split('.')[0]
        # if not layer in target_list:
        if layer in target_list:
            print('Loading feature matrix of layer', layer + ':')
            with open(feature_path, 'rb') as pkl:         # original feature map
                full_matrix = pickle.load(pkl)
            print('Loading Idx matrix of layer', layer + ':')
            sig_neuron_idx = np.loadtxt(Idx_root+layer+'-neuronIdx.csv', delimiter=',')

            # Gnerate a boolean mask to return the rest of cols
            mask = np.array([(i in sig_neuron_idx) for i in range(full_matrix.shape[1])])
            ID_matrix = full_matrix[:, mask]
            nonID_matrix = full_matrix[:, ~mask]
            print("full_feature: {}, Selective: {}, Non-selective: {}".format(full_matrix.shape[1], ID_matrix.shape[1], nonID_matrix.shape[1]))

            label = []
            for i in range(50):
                label += [i + 1] * 10
            label = np.array(label)
            label = np.expand_dims(label, axis=1)
            print(label.shape)

            full_acc = utils.SVM_classification(full_matrix, label)
            print('full_Accuracy: %d %%' % (100 * full_acc))
            full_acc_dict.update({layer: full_acc})

            ID_acc = utils.SVM_classification(ID_matrix, label)
            print('ID_Accuracy: %d %%' % (100 * ID_acc))
            ID_acc_dict.update({layer: ID_acc})

            nonID_acc = utils.SVM_classification(nonID_matrix, label)
            print('nonID_Accuracy: %d %%' % (100 * nonID_acc))
            nonID_acc_dict.update({layer: nonID_acc})


    x = target_list
    y = [full_acc_dict[k] for k in target_list]
    y1 = [ID_acc_dict[k] for k in target_list]
    y2 = [nonID_acc_dict[k] for k in target_list]
    plt.figure()
    plt.plot(x, y, 'b', label='full')
    plt.plot(x, y1, 'r', label='ID')
    plt.plot(x, y2, 'b', label='full')
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylabel('Classification Accuracy')
    plt.title('Decoding Accuracy')
    plt.savefig(dest + '/' + 'acc.png')


if __name__ == "__main__":
    main()
