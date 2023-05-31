#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 01:04:18 2023

@author: acxyle
"""

import os
import pickle
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

import utils_

# -----


root = '/media/acxyle/Data/ChromeDownload/'

# =============================================================================
# root_dict_keys = ['B', 'A1', 'A2', 'S1', 'S2', 'S3', 'S4']
# root_dict = {
# 'B': 'Identity_VGG_Feature_Original/neuron_idx/',
# 'A1': 'Identity_VGG16_ReLU_CelebA2622_Neuron/',
# 'A2': 'Identity_VGG16bn_ReLU_CelebA2622_Neuron/',
# 'S1': 'Identity_SpikingVGG16bn_IF_CelebA2622_Neuron/',
# 'S2': 'Identity_SpikingVGG16bn_LIF_CelebA2622_Neuron/',
# 'S3': 'Identity_SpikingVGG16bn_IF_CelebA9326_Neuron/',
# 'S4': 'Identity_SpikingVGG16bn_LIF_CelebA9326_Neuron/'
# }
# =============================================================================

root_dict_keys = ['B', 'A1', 'A2', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
root_dict = {
'B': 'Identity_Resnet50_Original_Neuron',
'A1': 'Identity_Resnet50_ReLU_CelebA2622_Neuron',
'A2': 'Identity_Resnet18_ReLU_CelebA2622_Neuron',
'S1': 'Identity_SpikingResnet18_IF_CelebA2622_Neuron',
'S2': 'Identity_SpikingResnet18_LIF_CelebA2622_Neuron',
'S3': 'Identity_SEWResnet18_IF_CelebA2622_Neuron',
'S4': 'Identity_SEWResnet18_LIF_CelebA2622_Neuron',
'S5': 'Identity_SEWResnet50_IF_CelebA2622_Neuron',
'S6': 'Identity_SEWResnet50_LIF_CelebA2622_Neuron'
    }

def RSA_human_scores_analysis():

    neurons = ['vKeep', 'IDNeuron', 'nonIDNeuron']
    #neurons = ['nonIDNeuron']
    ids = ['top10', 'top50']     # [notice] initially there is another ID group - 'selected'
    
    for neuron in neurons:
        for id_ in ids:
            suffix = f'RSA_human/saved_params_{neuron}_{id_}.pkl'
            RSA_human_single_analysis(suffix, root, root_dict, root_dict_keys)

def RSA_human_single_analysis(suffix, root, root_dict, root_dict_keys):
    params_list = RSA_human_get_params(root, root_dict, root_dict_keys, suffix)
    RSA_human_scores_merge_and_comparison(params_list, root_dict_keys, suffix)
    RSA_human_temporal_scores_merge_and_comparison(params_list, root_dict_keys, suffix)

# [notice] from 0 to 6: rFNID, rFNID_T, rPermID, pFNID_FDR, pFNID_T_FDR, sigFN_T, sigFDR_T
def RSA_human_get_params(root, root_dict, root_dict_keys, suffix):
    params_list = []
    for i in root_dict_keys:
        pth = os.path.join(root, root_dict[i], suffix)
        with open(pth, 'rb') as f:
            params = pickle.load(f)
        f.close()
        params_list.append(params)
    return params_list

# not applicatble for np.array
def remove_nan_values_in_list(input_list):
    for idx, i in enumerate(input_list):
        input_list[idx] = i[~np.isnan(i)]
    return input_list

def RSA_human_scores_merge_and_comparison(params_list, root_dict_keys, suffix):
    
    used_ID = suffix.split('.')[0].split('_')[-1]
    neuron_type = suffix.split('.')[0].split('_')[-2]
    
    # [notice] process of NaN value in nonIDNeuron (last layer)
    # [notice] remove the last layer because has no nonIDNeuron
    if neuron_type == 'nonIDNeuron':
        rFNID_list = np.array([i[0] for i in params_list], dtype=object)
        nan_list = [np.isnan(i) for i in rFNID_list]
        rFNID_list = [i[~nan_list[idx]] for idx, i in enumerate(rFNID_list)]
        rFNID_list
        
        rFNIDPerm_list = np.array([i[2] for i in params_list], dtype=object)
        rFNIDPerm_list = [i[~nan_list[idx]] for idx, i in enumerate(rFNIDPerm_list)]
        
        pFN_FDR_list = np.array([i[3] for i in params_list], dtype=object)
        pFN_FDR_list = [i[~nan_list[idx]] for idx, i in enumerate(pFN_FDR_list)]
    else:
       rFNID_list = np.array([i[0] for i in params_list], dtype=object)     # 1. (all) rFNID
       rFNIDPerm_list = np.array([i[2] for i in params_list], dtype=object)
       pFN_FDR_list = np.array([i[3] for i in params_list], dtype=object)
   
    rPermID_Mean_SD = []
    for idx, _ in enumerate(rFNIDPerm_list):     # [notice] for each model
        rPermID_Mean_SD.append(np.mean(_, axis=1) + np.std(_, axis=1)) 
       
    # FDR test  [notice] now a problem is ragged data
    rFNID_list_FDR = []   
    rFNID_list_failed_FDR = []
    for idx, _ in enumerate(rFNID_list):
        rFNID_list_FDR.append(_[pFN_FDR_list[idx] <= 0.05])
        rFNID_list_failed_FDR.append(_[pFN_FDR_list[idx] > 0.05])
    rFNID_list_FDR = np.array(rFNID_list_FDR, dtype=object)     # 2. (solid) rFNID
    rFNID_list_failed_FDR = np.array(rFNID_list_failed_FDR, dtype=object)     # 3. (hollow) rFNID

    # -----
    mean_values = np.array([np.mean(i) for i in rFNID_list])
    max_values = np.array([np.max(i) for i in rFNID_list])
    min_values = np.array([np.min(i) for i in rFNID_list])
    
    # -----
    fig, ax = plt.subplots(figsize=(5,5))
    box_positions = np.arange(1,len(max_values)+1)
    bp = ax.boxplot(rFNID_list, widths=0.6, positions=box_positions, patch_artist=True)
    for box in bp['boxes']:
        box.set_linestyle('--')
        box.set_facecolor('white')
        box.set_alpha(0.5)
        
    if 0 not in rFNID_list_FDR.shape:
        rFNID_list_FDR =  np.squeeze(rFNID_list_FDR)
        box_positions = np.arange(1,len(max_values)+1)+0.2
        bp = ax.boxplot(rFNID_list_FDR, widths=0.2, positions=box_positions, patch_artist=True)
        for box in bp['boxes']:
            box.set_linestyle('-')
            box.set_facecolor('black')
            box.set_alpha(0.5)
        box.set_label('passed_FDR')
    
    if 0 not in rFNID_list_failed_FDR.shape:
        rFNID_list_failed_FDR =  np.squeeze(rFNID_list_failed_FDR)
        box_positions = np.arange(1,len(max_values)+1)-0.2
        bp = ax.boxplot(rFNID_list_failed_FDR, widths=0.2, positions=box_positions, patch_artist=True)
        for box in bp['boxes']:
            box.set_linestyle(':')
            box.set_facecolor('gray')
            box.set_alpha(0.5)
        box.set_label('falied_FDR')
        
    rPermID_Mean_SD_mean = [np.mean(i) for i in rPermID_Mean_SD]
    
    ax.plot(np.arange(1,len(max_values)+1), rPermID_Mean_SD_mean, label='error area')
    ax.fill_between(range(1,len(max_values)+1), np.array([-0.5]*(len(max_values))), rPermID_Mean_SD_mean, color='blue', alpha=0.2)
    ax.plot(np.arange(1, len(mean_values) + 1), mean_values, 'ro')
    group_positions = np.arange(1, len(mean_values) + 1)
    ax.hlines(mean_values[1:], group_positions[1:] - 0.3, group_positions[1:] + 0.3, colors='r', linestyles='solid', label='Mean', alpha=0.8)
    ax.hlines(mean_values[0], 0.8, len(max_values)+0.2, colors='pink', linestyles='--', label='Baseline')
    ax.set_xticks(range(1,len(max_values)+1))
    ax.set_xticklabels(root_dict_keys, rotation=0)
    ax.set_xlabel('Models')
    ax.set_ylabel('Values')
    ax.set_xlim(0.5, len(max_values)+0.5)
    if 0 < min(min_values) and min(min_values) < 0.1:
        min_values = [0]*(len(max_values))
    ax.set_ylim(1.2*min(min_values), 1.2*max(max_values))
    ax.legend()
    ax.set_title(f'RSA_human_Corr ({neuron_type} {used_ID}) (ResNet type)')
    plt.tight_layout(pad=1)
    plt.savefig(save_root+f'RSA_human_Corr ({neuron_type} {used_ID}) Comparison (ResNet type).png')
    plt.savefig(save_root+f'RSA_human_Corr ({neuron_type} {used_ID}) Comparison (ResNet type).svg', format='svg', transparent=True)
    plt.show() 

def assemble_qualified_value(rFNID_T, sig_T):
    tmp = []
    for idx, i in enumerate(sig_T):
        if len(i) != 0:
            tmp.append(rFNID_T[idx][i])
    values = [j for _ in tmp for j in _]
    
    return values

def RSA_human_temporal_scores_merge_and_comparison(params_list, root_dict_keys, suffix):
    FDR_list = []
    sig_list = []
    
    used_ID = suffix.split('.')[0].split('_')[-1]
    neuron_type = suffix.split('.')[0].split('_')[-2]

    if neuron_type == 'nonIDNeuron':
       for params in params_list:
           rFNID_T = params[1][:-1]
           sig_T = params[5][:-1]
           sigFDR_T = params[6][:-1]
           
           values1 = assemble_qualified_value(rFNID_T, sig_T)
           sig_list.append(values1)
           values2 = assemble_qualified_value(rFNID_T, sigFDR_T)
           FDR_list.append(values2)
    else:
        for params in params_list:
            rFNID_T = params[1]
            sig_T = params[5]
            sigFDR_T = params[6]
            values1 = assemble_qualified_value(rFNID_T, sig_T)
            sig_list.append(values1)
            values2 = assemble_qualified_value(rFNID_T, sigFDR_T)
            FDR_list.append(values2)
    # ---
    mean_values_FDR = []
    for _ in FDR_list:
        mean_values_FDR.append(np.mean(_))
    mean_values_FDR = np.array(mean_values_FDR)    
    
    fig, ax = plt.subplots(figsize=(5,5))
    box_positions = np.arange(1,len(mean_values_FDR)+1)-0.1
    bp = ax.boxplot(FDR_list, widths=0.2, positions=box_positions, patch_artist=True)
    
    for box in bp['boxes']:
        box.set_linestyle('--')
        box.set_facecolor('white')
        box.set_alpha(0.5)
    box.set_label('passed FDR')
    
    ax.plot(np.arange(1, len(mean_values_FDR) + 1)-0.1, mean_values_FDR, color='red', marker='d', linestyle='', alpha=0.5)
    group_positions = np.arange(1, len(mean_values_FDR) + 1)-0.1
    ax.hlines(mean_values_FDR[1:], group_positions[1:] - 0.1, group_positions[1:] + 0.1, colors='r', linestyles='solid')
    ax.hlines(mean_values_FDR[0], 0.8, len(mean_values_FDR)+0.2, colors='orange', linestyles='--', label='FDR Baseline')
    
    # ---
    mean_values_sig = []
    for _ in sig_list:
        mean_values_sig.append(np.mean(_))
    mean_values_sig = np.array(mean_values_sig)    
    
    box_positions = np.arange(1,len(mean_values_FDR)+1)+0.1
    bp = ax.boxplot(sig_list, widths=0.2, positions=box_positions, patch_artist=True)
    
    for box in bp['boxes']:
        box.set_linestyle('-')
        box.set_facecolor('black')
        box.set_alpha(0.5)
    box.set_label('passed sig')
    
    ax.plot(np.arange(1, len(mean_values_sig) + 1)+0.1, mean_values_sig, color='red', marker='o', linestyle='', alpha=0.5)
    group_positions = np.arange(1, len(mean_values_sig) + 1)+0.1
    ax.hlines(mean_values_sig[1:], group_positions[1:] - 0.1, group_positions[1:] + 0.1, colors='r', linestyles='solid', label='Mean')
    ax.hlines(mean_values_sig[0], 0.8, len(mean_values_sig)+0.2, colors='pink', linestyles='--', label='sig Baseline')
    
    # ---
    
    ax.set_xticks(range(1,len(mean_values_FDR)+1))
    ax.set_xticklabels(root_dict_keys, rotation=0)
    ax.set_xlabel('Models')
    ax.set_ylabel('Values')
    ax.legend()
    ax.set_title(f'RSA_human_Corr_Temporal ({neuron_type} {used_ID}) (ResNet type)')
    
    plt.tight_layout(pad=1)
    plt.savefig(save_root+f'RSA_human_Corr_Temporal ({neuron_type} {used_ID}) Comparison (ResNet type).png')
    plt.savefig(save_root+f'RSA_human_Corr_Temporal ({neuron_type} {used_ID}) Comparison (ResNet type).svg', format='svg', transparent=True)
    
    plt.show()
    
    

    
def RSA_monkey_scores_analysis():
    
    suffix = 'RSA_monkey/saved_params.pkl'
    params_list = []
    for i in root_dict_keys:
        pth = os.path.join(root, root_dict[i], suffix)
        with open(pth, 'rb') as f:
            params = pickle.load(f)
        f.close()
        params_list.append(params)
    # [notice] from 0 to 4: rFNID, rFNID_T, rFNIDPerm, pFN_FDR, sig_T
    RSA_monkey_scores_merge_and_comparison(params_list, root_dict_keys)
    RSA_monkey_temporal_scores_merge_and_comparison(params_list, root_dict_keys)

def RSA_monkey_scores_merge_and_comparison(params_list, root_dict_keys):

    rFNID_list = np.array([i[0] for i in params_list], dtype=object)
    rFNIDPerm_list = np.array([i[2] for i in params_list], dtype=object)
    pFN_FDR_list = np.array([i[3] for i in params_list], dtype=object)
    
    rPermID_Mean_SD = []
    for idx, _ in enumerate(rFNIDPerm_list):
        rPermID_Mean_SD.append(np.mean(_, axis=1) + np.std(_, axis=1)) 
    
    # FDR test
    for idx, _ in enumerate(rFNID_list):
        rFNID_list[idx] = _[pFN_FDR_list[idx] <= 0.05]
        
    rFNID_list = rFNID_list.T
    #mean_values = np.mean(rFNID_list, axis=0)
    mean_values = []
    for _ in rFNID_list:
        mean_values.append(np.mean(_))
    mean_values = np.array(mean_values) 
        
    fig, ax = plt.subplots(figsize=(5,5))
    ax.boxplot(rFNID_list)
    
    rPermID_Mean_SD = np.array(rPermID_Mean_SD, dtype=object)
    rPermID_Mean_SD_mean = [np.mean(i) for i in rPermID_Mean_SD]
    
    ax.plot(np.arange(1,len(rPermID_Mean_SD)+1), rPermID_Mean_SD_mean, label='error area')
    ax.fill_between(range(1,len(rPermID_Mean_SD+1)+1), np.array([0]*len(mean_values)), rPermID_Mean_SD_mean, color='blue', alpha=0.3)
    ax.plot(np.arange(1, len(mean_values)+1), mean_values, 'ro')
    group_positions = np.arange(1, len(mean_values) + 1)
    ax.hlines(mean_values[1:], group_positions[1:] - 0.2, group_positions[1:] + 0.2, colors='r', linestyles='solid', label='Mean')
    ax.hlines(mean_values[0], 0.8, len(mean_values)+0.2, colors='pink', linestyles='--', label='Baseline')
    ax.set_xticklabels(root_dict_keys, rotation=0)
    ax.set_xlabel('Models')
    ax.set_ylabel('Values')
    ax.set_ylim(0, 1.1*max([np.max(i) for i in rFNID_list]))
    ax.legend()
    ax.set_title('RSA_monkey_Corr (ResNet type)')
    
    plt.tight_layout(pad=1)
    plt.savefig(save_root+'RSA_monkey_Corr Comparison (ResNet type).png')
    plt.savefig(save_root+'RSA_monkey_Corr Comparison (ResNet type).svg', format='svg', transparent=True)
    
    plt.show()


def RSA_monkey_temporal_scores_merge_and_comparison(params_list, root_dict_keys):
    q_list = []
    
    for params in params_list:
        rFNID_T = params[1]
        sig_T = params[4]
        values = assemble_qualified_value(rFNID_T, sig_T)
        q_list.append(values)
    
    mean_values = []
    for _ in q_list:
        mean_values.append(np.mean(_))
    mean_values = np.array(mean_values)    
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.boxplot(q_list)
    
    ax.plot(np.arange(1, len(mean_values) + 1), mean_values, 'ro')
    group_positions = np.arange(1, len(mean_values) + 1)
    ax.hlines(mean_values[1:], group_positions[1:] - 0.2, group_positions[1:] + 0.2, colors='r', linestyles='solid', label='Mean')
    ax.hlines(mean_values[0], 0.8, len(mean_values)+0.2, colors='pink', linestyles='--', label='Baseline')
    
    ax.set_xticklabels(root_dict_keys, rotation=0)
    ax.set_xlabel('Models')
    ax.set_ylabel('Values')
    ax.legend()
    ax.set_title('RSA_monkey_Corr_Temporal (ResNet type)')
    
    plt.tight_layout(pad=1)
    plt.savefig(save_root+'RSA_monkey_Corr_Temporal Comparison (ResNet type).png')
    plt.savefig(save_root+'RSA_monkey_Corr_Temporal Comparison (ResNet type).svg', format='svg')
    
    plt.show()
    
    
if __name__ == "__main__":
    save_root = 'Comparison_ResNet_monkey/'
    utils_.make_dir(save_root)
    RSA_monkey_scores_analysis()
    
    save_root = 'Comparison_ResNet_human/'
    utils_.make_dir(save_root)
    RSA_human_scores_analysis()
    