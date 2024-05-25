
import os
import numpy as np
from scipy import stats

import sys
sys.path.append('../')
import utils_

import matplotlib.pyplot as plt

from FSA_CKA import CKA_Similarity_Monkey_folds, CKA_Similarity_Human_folds
from FSA_RSA import RSA_Monkey_folds, RSA_Human_folds
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind


# ------------ RSA
# =============================================================================
# first_corr = 'pearson'
# second_corr = 'spearman'
# used_cell_type = 'selective'
# 
# _, layers, neurons, _ = utils_.get_layers_and_units('vgg16', 'act')
# root = '/home/acxyle-workstation/Downloads/FSA/VGG/VGG/FSA VGG16bn_C2k_fold_'
# RSA_H = RSA_Human_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
# RSA_VGG16 = RSA_H.collect_RSA_Similarity_folds(first_corr, second_corr, used_cell_type, 50)
# 
# group_A = np.array([RSA_VGG16[_]['similarity'] for _ in range(5)]).reshape(-1)
# 
# _, layers, neurons, _ = utils_.get_layers_and_units('spiking_vgg16_bn', 'act')
# root = '/home/acxyle-workstation/Downloads/FSA/VGG/SpikingVGG/FSA SpikingVGG16bn_IF_ATan_T4_C2k_fold_'
# RSA_H = RSA_Human_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
# RSA_SVGG16 = RSA_H.collect_RSA_Similarity_folds(first_corr, second_corr, used_cell_type, 50)
# 
# group_B = np.array([RSA_SVGG16[_]['similarity'] for _ in range(5)]).reshape(-1)
# 
# stats.ttest_rel(group_A, group_B)
# =============================================================================


# ---------- CKA 
# =============================================================================
# kernel = 'linear'
# cka_config = f"CKA_results_{kernel}"
# used_unit_type = 'selective'
# used_id_num = 50
# 
# _, layers, neurons, _ = utils_.get_layers_and_units('vgg16', 'act')
# root = '/home/acxyle-workstation/Downloads/FSA/VGG/VGG/FSA VGG16bn_C2k_fold_'
# CKA_H = CKA_Similarity_Human_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
# CKA_VGG16 = CKA_H.collect_CKA_Similarity_folds(kernel, cka_config, used_unit_type, used_id_num)
# 
# group_A = np.array([CKA_VGG16[_]['cka_score'] for _ in range(5)]).reshape(-1)
# 
# _, layers, neurons, _ = utils_.get_layers_and_units('spiking_vgg16_bn', 'act')
# root = '/home/acxyle-workstation/Downloads/FSA/VGG/SpikingVGG/FSA SpikingVGG16bn_IF_ATan_T4_C2k_fold_'
# CKA_H = CKA_Similarity_Human_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
# CKA_SVGG16 = CKA_H.collect_CKA_Similarity_folds(kernel, cka_config, used_unit_type, used_id_num)
# 
# group_B = np.array([CKA_SVGG16[_]['cka_score'] for _ in range(5)]).reshape(-1)
# 
# stats.ttest_rel(group_A, group_B, nan_policy='omit')
# =============================================================================



# =============================================================================
# first_corr = 'pearson'
# second_corr = 'spearman'
# 
# RSA_dict = {}
# 
# def RSA_data_collect(used_unit_type = 'qualified', used_id_num = 50):
#     # ----- ANOVA - temporal - RSA - monkey
#     FSA_root = '/home/acxyle-workstation/Downloads/FSA'
#     FSA_dir = 'VGG/VGG'
#     FSA_config = 'VGG16bn_C2k_fold_'
#     FSA_model = 'vgg16_bn'
#     
#     # -----
#     _, layers, neurons, _ = utils_.get_layers_and_units(FSA_model, 'act')
#     root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
#     
#     #RSA_M_f = RSA_Monkey_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
#     #RSA_VGG = RSA_M_f.collect_RSA_Similarity_folds(first_corr, second_corr)
#     #RSA_VGG_t = np.array([v['similarity_temporal'] for k,v in RSA_VGG.items()]).reshape(-1, 26)
#     
#     RSA_H_f = RSA_Human_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
#     RSA_VGG = RSA_H_f.collect_RSA_Similarity_folds(first_corr, second_corr, used_unit_type, used_id_num)
#     RSA_VGG_t = np.array([v['similarity'] for k,v in RSA_VGG.items()]).reshape(-1)
#     
#     
#     # ------------------
#     FSA_root = '/home/acxyle-workstation/Downloads/FSA'
#     FSA_dir = 'VGG/SpikingVGG'
#     FSA_config = 'SpikingVGG16bn_IF_ATan_T4_C2k_fold_'
#     FSA_model = 'spiking_vgg16_bn'
#     
#     # -----
#     _, layers, neurons, _ = utils_.get_layers_and_units(FSA_model, 'act')
#     root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
#     
#     #RSA_M_f = RSA_Monkey_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
#     #RSA_SVGG = RSA_M_f.collect_RSA_Similarity_folds(first_corr, second_corr)
#     #RSA_SVGG_t = np.array([v['similarity_temporal'] for k,v in RSA_SVGG.items()]).reshape(-1, 26)
#     
#     RSA_H_f = RSA_Human_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
#     RSA_SVGG = RSA_H_f.collect_RSA_Similarity_folds(first_corr, second_corr, used_unit_type, used_id_num)
#     RSA_SVGG_t = np.array([v['similarity'] for k,v in RSA_SVGG.items()]).reshape(-1)
#     
#     return (RSA_VGG_t, RSA_SVGG_t)
# 
# for _ in ['qualified', 'selective', 'strong_selective', 'weak_selective', 'non_selective']:
#     
#     RSA_dict[_] = np.nan_to_num(RSA_data_collect(used_unit_type=_))
# =============================================================================
    
#stats.f_oneway(*list(RSA_dict.values()))
    
#t_groups = list(np.vstack((RSA_VGG_t, RSA_SVGG_t)).T)

#print(stats.f_oneway(*t_groups))


# ----- ANOVA - temporal - RSA - monkey
kernel = 'linear'
cka_config = f"CKA_results_{kernel}"

def CKA_data_collect(used_unit_type = 'qualified', used_id_num = 50):
    FSA_root = '/home/acxyle-workstation/Downloads/FSA'
    FSA_dir = 'VGG/VGG'
    FSA_config = 'VGG16bn_C2k_fold_'
    FSA_model = 'vgg16_bn'
    
    # -----
    _, layers, neurons, _ = utils_.get_layers_and_units(FSA_model, 'act')
    root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
    
    #CKA_M_f = CKA_Similarity_Monkey_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
    #CKA_VGG = CKA_M_f.collect_CKA_Similarity_folds(kernel, cka_config)
    #CKA_VGG_t = np.array([v['cka_score_temporal'] for k,v in CKA_VGG.items()]).reshape(-1,26)
    
    CKA_H_f = CKA_Similarity_Human_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
    CKA_VGG = CKA_H_f.collect_CKA_Similarity_folds(kernel, cka_config, used_unit_type, used_id_num)
    CKA_VGG_t = np.array([v['cka_score'] for k,v in CKA_VGG.items()]).reshape(-1,)
    
    
    # ------------------
    FSA_root = '/home/acxyle-workstation/Downloads/FSA'
    FSA_dir = 'VGG/SpikingVGG'
    FSA_config = 'SpikingVGG16bn_IF_ATan_T4_C2k_fold_'
    FSA_model = 'spiking_vgg16_bn'
    
    # -----
    _, layers, neurons, _ = utils_.get_layers_and_units(FSA_model, 'act')
    root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
    
    #CKA_M_f = CKA_Similarity_Monkey_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
    #CKA_SVGG = CKA_M_f.collect_CKA_Similarity_folds(kernel, cka_config)
    #CKA_VGG_t = np.array([v['cka_score_temporal'] for k,v in CKA_VGG.items()]).reshape(-1,26)
    
    CKA_H_f = CKA_Similarity_Human_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
    CKA_SVGG = CKA_H_f.collect_CKA_Similarity_folds(kernel, cka_config, used_unit_type, used_id_num)
    CKA_SVGG_t = np.array([v['cka_score'] for k,v in CKA_SVGG.items()]).reshape(-1,)
    
    return CKA_VGG_t, CKA_SVGG_t

ANN_CKA_dict = {}
SNN_CKA_dict = {}
for _ in ['qualified', 'selective', 'strong_selective', 'weak_selective', 'non_selective']:
     ANN_CKA_dict[_], SNN_CKA_dict[_] = np.nan_to_num(CKA_data_collect(used_unit_type=_))

#t_groups = list(np.vstack((CKA_VGG_t, CKA_SVGG_t)).T)

#print(stats.f_oneway(*t_groups))
