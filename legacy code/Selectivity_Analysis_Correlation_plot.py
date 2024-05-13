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
def obtain_root_dict(root = '/media/acxyle/Data/ChromeDownload/', model_type=None):
    if model_type == 'VGG':
        root_dict_keys = ['B', 'A1', 'A2', 'S1', 'S2', 'S3', 'S4']
        root_dict = {
        'B': 'Identity_VGG_Feature_Original/neuron_idx/',
        'A1': 'Identity_VGG16_ReLU_CelebA2622_Neuron/',
        'A2': 'Identity_VGG16bn_ReLU_CelebA2622_Neuron/',
        'S1': 'Identity_SpikingVGG16bn_IF_CelebA2622_Neuron/',
        'S2': 'Identity_SpikingVGG16bn_LIF_CelebA2622_Neuron/',
        'S3': 'Identity_SpikingVGG16bn_IF_CelebA9326_Neuron/',
        'S4': 'Identity_SpikingVGG16bn_LIF_CelebA9326_Neuron/'
        }
    elif model_type == 'Resnet':
        root_dict_keys = ['B', 'A1', 'A2', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
        root_dict = {
        'B': 'Identity_Resnet50_Original_Neuron',
        'A1': 'Identity_Resnet50_ReLU_CelebA2622_Neuron',
        'A2': 'Identity_Resnet18_ReLU_CelebA2622_Neuron',
        'S1': 'Identity_SpikingResnet18_IF_CelebA2622_Neuron',
        'S2': 'Identity_SpikingResnet18_LIF_CelebA2622_Neuron',
        'S3': 'Identity_spiking_resnet18_ParametricLIF_ATan_T4_CelebA2622_Neuron',
        'S4': 'Identity_spiking_resnet18_QIF_ATan_T4_CelebA2622_Neuron',
        'S5': 'Identity_spiking_resnet18_EIF_ATan_T4_CelebA2622_Neuron',
        'S6': 'Identity_spiking_resnet18_Izhikevich_ATan_T4_CelebA2622_Neuron'
            }
    else:
        raise RuntimeError('[Codinfo] the model is not in use')
        
    return root, root_dict_keys, root_dict

def RSA_human_scores_analysis(model_type):

    neurons = ['vKeep', 'IDNeuron', 'nonIDNeuron']
    #neurons = ['nonIDNeuron']
    ids = ['top50', 'top10']     # [notice] initially there is another ID group - 'selected'
    
    root, root_dict_keys, root_dict = obtain_root_dict(model_type=model_type)
    
    fig, ax = plt.subplots(2,3,figsize=(18.5,10))
    fig1, ax1 = plt.subplots(2,3,figsize=(18.5,10))
    
    c_row, c_col = 0, 0
    for id_ in ids:
        for neuron in neurons:
            suffix = f'RSA_human/saved_params_{neuron}_{id_}.pkl'
            RSA_human_single_analysis_static(ax[c_row, c_col], suffix, root, root_dict, root_dict_keys, model_type)
            RSA_human_single_analysis_temporal(ax1[c_row, c_col], suffix, root, root_dict, root_dict_keys, model_type)
            c_col += 1
            if c_col == 3:
                c_row += 1
                c_col = 0
                
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(right=0.8)  # Adjust this value to create space on the right
    fig.suptitle(f'RSA_human_Corr Static Comparison ({model_type} type)', x=0.5, y=0.97, fontsize=18, ha='center')
    text_content = 'B: Resnet50_Pretrained\nA1: Resnet50_ReLU\nA2: Resnet18_ReLU\nS1: SpikingResnet18_IF\nS2: SpikingResnet18_LIF\nS3: SpikingResnet18_ParametricLIF\nS4: SpikingResnet18_QIF\nS5: SpikingResnet18_EIF\nS6: SpikingResnet18_Izhikevich'
    fig.text(0.81, 0.2, text_content, fontsize=14, bbox={'facecolor':'yellow', 'alpha': 0.2}, verticalalignment='center')
    fig.savefig(save_root+f'RSA_human_Corr Static Comparison ({model_type} Type).png')
    fig.savefig(save_root+f'RSA_human_Corr Static Comparison ({model_type} Type).svg', format='svg', transparent=True)
    #plt.show() 
    
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.subplots_adjust(right=0.8)  # Adjust this value to create space on the right
    fig1.suptitle(f'RSA_human_Corr Temporal Comparison ({model_type} type)', x=0.5, y=0.97, fontsize=18, ha='center')
    text_content = 'B: Resnet50_Pretrained\nA1: Resnet50_ReLU\nA2: Resnet18_ReLU\nS1: SpikingResnet18_IF\nS2: SpikingResnet18_LIF\nS3: SpikingResnet18_ParametricLIF\nS4: SpikingResnet18_QIF\nS5: SpikingResnet18_EIF\nS6: SpikingResnet18_Izhikevich'
    fig1.text(0.81, 0.2, text_content, fontsize=14, bbox={'facecolor':'yellow', 'alpha': 0.2}, verticalalignment='center')
    fig1.savefig(save_root+f'RSA_human_Corr Temporal Comparison ({model_type} Type).png')
    fig1.savefig(save_root+f'RSA_human_Corr Temporal Comparison ({model_type} Type).svg', format='svg', transparent=True)

def RSA_human_single_analysis_static(ax, suffix, root, root_dict, root_dict_keys, model_type):
    params_list = RSA_human_get_params(root, root_dict, root_dict_keys, suffix)
    RSA_human_scores_merge_and_comparison(ax, params_list, root_dict_keys, suffix, model_type)
    
def RSA_human_single_analysis_temporal(ax, suffix, root, root_dict, root_dict_keys, model_type):
    params_list = RSA_human_get_params(root, root_dict, root_dict_keys, suffix)
    RSA_human_temporal_scores_merge_and_comparison(ax, params_list, root_dict_keys, suffix, model_type)

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

def RSA_human_scores_merge_and_comparison(ax, params_list, root_dict_keys, suffix, model_type):
    
    used_ID = suffix.split('.')[0].split('_')[-1]
    neuron_type = suffix.split('.')[0].split('_')[-2]
    
    # [notice] process of NaN value in nonIDNeuron (last layer)
    # [notice] remove the last layer because has no nonIDNeuron
    if neuron_type == 'nonIDNeuron':
        rFNID_list = np.array([i[0] for i in params_list], dtype=object)
        #rFNID_list = rFNID_list.astype(float)
        nan_list = [np.isnan(i) for i in rFNID_list]
        rFNID_list = [i[~nan_list[idx]] for idx, i in enumerate(rFNID_list)]
        rFNID_list = np.array(rFNID_list, dtype=object)
        
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
        _ = _.astype(float)
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
    #fig, ax = plt.subplots(figsize=(5,5))
    box_positions = np.arange(1,len(max_values)+1)
    bp = ax.boxplot(rFNID_list.T, widths=0.6, positions=box_positions, patch_artist=True)
    for box in bp['boxes']:
        box.set_linestyle('--')
        box.set_facecolor('white')
        box.set_alpha(0.5)
        
    if 0 not in rFNID_list_FDR.shape:
        rFNID_list_FDR =  np.squeeze(rFNID_list_FDR)
        box_positions = np.arange(1,len(max_values)+1)+0.2
        bp = ax.boxplot(rFNID_list_FDR.T, widths=0.2, positions=box_positions, patch_artist=True)
        for box in bp['boxes']:
            box.set_linestyle('-')
            box.set_facecolor('black')
            box.set_alpha(0.5)
        box.set_label('passed_FDR')
    
    if 0 not in rFNID_list_failed_FDR.shape:
        rFNID_list_failed_FDR =  np.squeeze(rFNID_list_failed_FDR)
        box_positions = np.arange(1,len(max_values)+1)-0.2
        bp = ax.boxplot(rFNID_list_failed_FDR.T, widths=0.2, positions=box_positions, patch_artist=True)
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
    ax.set_ylim(-0.2, 0.5)
    #ax.set_ylim(1.2*min(min_values), 1.2*max(max_values))
    ax.legend()
    ax.set_title(f'{neuron_type} {used_ID}')
    #plt.tight_layout(pad=1)
    #plt.savefig(save_root+f'RSA_human_Corr ({neuron_type} {used_ID}) Comparison (VGG Type).png')
    #plt.savefig(save_root+f'RSA_human_Corr ({neuron_type} {used_ID}) Comparison (VGG Type).svg', format='svg', transparent=True)
    #plt.show() RSA_human_scores_analysis

def assemble_qualified_value(rFNID_T, sig_T):
    tmp = []
    for idx, i in enumerate(sig_T):
        if len(i) != 0:
            tmp.append(rFNID_T[idx][i])
    values = [j for _ in tmp for j in _]
    
    return values

def RSA_human_temporal_scores_merge_and_comparison(ax, params_list, root_dict_keys, suffix, model_type):
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
    
    #fig, ax = plt.subplots(figsize=(5,5))
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
    ax.set_ylim(0, 0.85)
    ax.legend()
    ax.set_title(f'{neuron_type} {used_ID}')
    
    #plt.show()
    
"""
    Monkey
"""
def RSA_monkey_scores_analysis(model_type):
    root, root_dict_keys, root_dict = obtain_root_dict(model_type=model_type)
    suffix = 'RSA_monkey/saved_params.pkl'
    params_list = []
    for i in root_dict_keys:
        pth = os.path.join(root, root_dict[i], suffix)
        with open(pth, 'rb') as f:
            params = pickle.load(f)
        f.close()
        params_list.append(params)
    # [notice] from 0 to 4: rFNID, rFNID_T, rFNIDPerm, pFN_FDR, sig_T
    
    #TODO
    # name the outside canvas here
    
    RSA_monkey_scores_merge_and_comparison(model_type, params_list, root_dict_keys)
    RSA_monkey_temporal_scores_merge_and_comparison(model_type, params_list, root_dict_keys)

def RSA_monkey_scores_merge_and_comparison(model_type, params_list, root_dict_keys):

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
    #ax.set_ylim(0, 1.1*max([np.max(i) for i in rFNID_list]))
    ax.set_ylim(0, 0.8)     # [notice] manually control the scale
    ax.legend()
    ax.set_title(f'RSA_monkey_Corr ({model_type} Type)')
    ax.text(1.25, 2.25, 
            'B: Resnet50_Pretrained\nA1: Resnet50_ReLU\nA2: Resnet18_ReLU\nS1: SpikingResnet18_IF\nS2: SpikingResnet18_LIF\nS3: SpikingResnet18_ParametricLIF\nS4: SpikingResnet18_QIF\nS5: SpikingResnet18_EIF\nS6: SpikingResnet18_Izhikevich', 
            fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2},  transform=fig.dpi_scale_trans, verticalalignment='top'
            )
    
    plt.tight_layout(pad=1)
    plt.savefig(save_root+f'RSA_monkey_Corr Comparison ({model_type} Type).svg', format='svg', transparent=True)
    
    plt.show()

def RSA_monkey_temporal_scores_merge_and_comparison(model_type, params_list, root_dict_keys):
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
    ax.set_ylim(0, 0.8)     # [notice] manually control the scale
    ax.set_title(f'RSA_monkey_Corr_Temporal ({model_type} Type)')
    
    ax.text(1.25, 2.25, 
            'B: Resnet50_Pretrained\nA1: Resnet50_ReLU\nA2: Resnet18_ReLU\nS1: SpikingResnet18_IF\nS2: SpikingResnet18_LIF\nS3: SpikingResnet18_ParametricLIF\nS4: SpikingResnet18_QIF\nS5: SpikingResnet18_EIF\nS6: SpikingResnet18_Izhikevich', 
            fontsize=10, bbox={'facecolor':'yellow', 'alpha': 0.2},  transform=fig.dpi_scale_trans, verticalalignment='top'
            )
    
    plt.tight_layout(pad=1)
    plt.savefig(save_root+f'RSA_monkey_Corr_Temporal Comparison ({model_type} Type).png')
    plt.savefig(save_root+f'RSA_monkey_Corr_Temporal Comparison ({model_type} Type).svg', format='svg')
    
    plt.show()
    
    
if __name__ == "__main__":
    # [notice] model type should be VGG or Resnet
    model_type = 'Resnet'
    
    #save_root = f'Comparison_{model_type}_monkey/'
    #utils_.make_dir(save_root)
    #RSA_monkey_scores_analysis(model_type)
    
    save_root = f'Comparison_{model_type}_human/'
    utils_.make_dir(save_root)
    RSA_human_scores_analysis(model_type)
    