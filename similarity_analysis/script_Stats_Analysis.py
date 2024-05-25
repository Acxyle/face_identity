#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:06:15 2024

@author: acxyle-workstation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import sys
sys.path.append('../')
import utils_

from FSA_RSA import RSA_Monkey_folds, RSA_Human_folds

"""
    关于统计的问题：1.原始数据怎么构建统计量；2 使用什么检验
    
    统计量：[all, mean, max, 75%] * [all, sig], 至少8种选择
        可以提供多个数据且不需要考虑 大小 和 顺序 的检验: [all]*[sig],如果没有sig就 [all]*[all]
        需要强调顺序（层或者时间）但不需要考虑大小，看具体目的，优先考虑 mean 和 max
        需要强调一一对应，看具体目的，优先考虑 
"""



# =============================================================================
# from scipy.stats import chi2_contingency
# 
# # 构建列联表
# data = np.array([
#     [1, 2, 3, 90],  # ANN
#     [20, 30, 40, 20]   # SNN
# ])
# 
# # 执行卡方检验
# chi2, p, dof, expected = chi2_contingency(data)
# 
# print("Chi-squared Test Statistic:", chi2)
# print("P-value:", p)
# print("Degrees of freedom:", dof)
# print("Expected frequencies:\n", expected)
# =============================================================================



def collect_one_model_RSA_results(layer_depth):

    FSA_root = '/home/acxyle-workstation/Downloads/FSA'
    FSA_dir = 'Resnet/SEWResnet'
    FSA_config = f'SEWResnet{layer_depth}_IF_ATan_T4_C2k_fold_'
    FSA_model = f'sew_resnet{layer_depth}'
    
    _, layers, neurons, _ = utils_.get_layers_and_units(FSA_model, 'act')
    root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
    
    RSA_monkey_ = RSA_Monkey_folds(root=root, layers=layers, neurons=neurons)
    RSA_human_ = RSA_Human_folds(root=root, layers=layers, neurons=neurons)
    
    return RSA_monkey_.collect_RSA_Similarity_folds(), RSA_human_.collect_RSA_Similarity_folds()
    
    
RSA_dict_monkey = {}
RSA_dict_human = {}


for layer_depth in [18, 50, 101, 152]:
    
    RSA_dict_monkey[layer_depth], RSA_dict_human[layer_depth] = collect_one_model_RSA_results(layer_depth)

RSA_dict = {
    'Monkey': RSA_dict_monkey,
    'Human': RSA_dict_human
    }

print('6')

# --- Human ANN (resnet) results
# 1.1 all
pool_tmp_18 = [RSA_dict_human[18][_]['similarity_temporal'][RSA_dict_human[18][_]['sig_temporal_FDR'].astype(bool)] for _ in range(5)]
pool_tmp_50 = [RSA_dict_human[50][_]['similarity_temporal'][RSA_dict_human[50][_]['sig_temporal_FDR'].astype(bool)] for _ in range(5)]
pool_tmp_101 = [RSA_dict_human[101][_]['similarity_temporal'][RSA_dict_human[101][_]['sig_temporal_FDR'].astype(bool)] for _ in range(5)]
pool_tmp_152 = [RSA_dict_human[152][_]['similarity_temporal'][RSA_dict_human[152][_]['sig_temporal_FDR'].astype(bool)] for _ in range(5)]

pool_tmp_18 = np.concatenate(pool_tmp_18)
pool_tmp_50 = np.concatenate(pool_tmp_50)
pool_tmp_101 = np.concatenate(pool_tmp_101)
pool_tmp_152 = np.concatenate(pool_tmp_152)

 # 有一堆假设但是对输入数据格式没有要求
stats.f_oneway(pool_tmp_18, pool_tmp_50, pool_tmp_101, pool_tmp_152)    # 参

# all: F_onewayResult(statistic=54.496574632694795, pvalue=1.610322378304944e-33)
# sig: not applicable
# t sig: F_onewayResult(statistic=105.65489124144122, pvalue=2.513156905186266e-62)
# t all: F_onewayResult(statistic=132.04103961999286, pvalue=3.373458089361664e-85)

stats.kruskal(pool_tmp_18, pool_tmp_50, pool_tmp_101, pool_tmp_152)     # 非参
# KruskalResult(statistic=155.64141469051802, pvalue=1.5983166160310095e-33)

# 1.2 mean
pool_tmp_18 = [np.mean(RSA_dict_human[18][_]['similarity']) for _ in range(5)]
pool_tmp_50 = [np.mean(RSA_dict_human[50][_]['similarity']) for _ in range(5)]
pool_tmp_101 = [np.mean(RSA_dict_human[101][_]['similarity']) for _ in range(5)]
pool_tmp_152 = [np.mean(RSA_dict_human[152][_]['similarity']) for _ in range(5)]

stats.friedmanchisquare(pool_tmp_18, pool_tmp_50, pool_tmp_101, pool_tmp_152)     # 合理但是要求数据相等

# all: FriedmanchisquareResult(statistic=9.959999999999994, pvalue=0.018909228242395843)
# sig: not applicable
# t sig: FriedmanchisquareResult(statistic=12.11999999999999, pvalue=0.00698319445326556)
# t all: FriedmanchisquareResult(statistic=11.159999999999997, pvalue=0.010891421468579314)


# --- Human SNN (sew_resnet) results
# ---2.1 all
# all: 
# sig:
# t sig: F_onewayResult(statistic=2.6527026418079265, pvalue=0.04706521249104896)
#        KruskalResult(statistic=3.7236986546936497, pvalue=0.29288687105253747)
# t all: 
    
# --- 2.2 mean
# all: FriedmanchisquareResult(statistic=15.0, pvalue=0.0018166489665723214)
# sig: -
# t sig: FriedmanchisquareResult(statistic=4.200000000000003, pvalue=0.24066188520961498)
# t all: KruskalResult(statistic=4.2074718987540765, pvalue=0.2399148703968015)