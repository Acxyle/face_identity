#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:35:00 2023

@author: acxyle

#TODO
    write a script to do feature analysis
    
    [Induction]
    1. use spiking_model.py to seperate layers;
    2. use spiking_intermediate_output.py to visualize and validate the features in diferent levels;
    3. use spiking_featuremap.py to extract and save the feature.pkl
    4. use Selectivity_Analyzer to execute analysis
    
#TODO
    [Jan 3, 2023] add the k-folds comparisons
    
#TODO
    [Jan 12, 2023] add the normalized pct curve
    
#TODO
    [Jan 22, 2023] add the process for A2S models
    
"""

import os
import sys
import scipy
import datetime
import time
import psutil
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Bio_Cell_Records_Process import Human_Neuron_Records_Process, Monkey_Neuron_Records_Process

import Selectivity_Analysis_ANOVA
import Selectivity_Analysis_Encode
import Selectivity_Analysis_DR_and_SM
import Selectivity_Analysis_RSA
import Selectivity_Analysis_Feature
import Selectivity_Analysis_CKA

import utils_
import utils_similarity
sys.path.append('../')
import models_

from spikingjelly import visualizing

# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Selectivity Analyzer Ver 4.0", add_help=True)

parser.add_argument("--num_classes", type=int, default=50, help="[Codelp] set the number of classes")
parser.add_argument("--num_samples", type=int, default=10, help="[Codelp] set the sample number of each class")
parser.add_argument("--alpha", type=float, default=0.01, help='[Codelp] assign the alpha value for ANOVA')

parser.add_argument("--root_dir", type=str, default="/home/acxyle-workstation/Downloads", help="[Codelp] root directory for features and neurons")
parser.add_argument("--model", type=str, default='spiking_vgg16_bn')     # trigger

# -----
args = parser.parse_args()


# ----------------------------------------------------------------------------------------------------------------------
# FIXME - under construction...
class Multi_Model_Analysis(Monkey_Neuron_Records_Process, Human_Neuron_Records_Process):
    
    def __init__(self, 
                 feature_root_general='Face Identity VGG16bn_fold_',
                 num_fold=5):

        self.feature_root_general = feature_root_general
        self.num_fold = num_fold
        self.model_structure = feature_root_general.split('_')[0].split(' ')[-1]
        
        self.model_root = os.path.join(args.root_dir, self.feature_root_general)
        utils_.make_dir(self.model_root)
        
        self.layers, self.neurons, _ = utils_.get_layers_and_units(model_name=args.model, feature_shape=(3,224,224))
        _, self.layers, self.neurons, _ = utils_.activation_function(args.model, self.layers, self.neurons, act_only=True)
        
    
    def plot_RSA_pct_multi_models(self, alpha=0.05, FDR_method:str='fdr_bh'):
        
        
        from statsmodels.stats.multitest import multipletests
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        utils_.make_dir(RSA_save_root:=os.path.join(self.model_root, 'RSA'))
        
        # -----
        def _RSA_layer_wise_pct_collect(RSA_save_root_primate, primate='Monkey', criterion='pearson', unit_type:str=None, used_id_num:int=None):
            
            RSA_static_dict_folds_path = os.path.join(RSA_save_root_primate, f'RSA_static_dict_{primate}_{criterion}_{unit_type}_{used_id_num}.pkl')
            
            if os.path.exists(RSA_static_dict_folds_path):
                
                RSA_dict_folds = utils_.load(RSA_static_dict_folds_path, verbose=False)
            
            else:
                
                def __RSA_layer_wise_pct_collect_single_condition(RSA_dict_path):
                    
                    RSA_dict_folds = {}
                    
                    for fold_idx in np.arange(1, self.num_fold):
                        
                        root = os.path.join(self.model_root+str(fold_idx))
                        
                        RSA_dict = utils_.load(os.path.join(root, f'Analysis/RSA/{primate}/{criterion}', f'{RSA_dict_path}.pkl'), verbose=True)
                        
                        RSA_dict_folds[fold_idx] = RSA_dict
                
                    return RSA_dict_folds
                
                if 'monkey' in primate.lower():
                
                    RSA_dict_folds = __RSA_layer_wise_pct_collect_single_condition(f'RSA_results_{criterion}')
                
                elif 'human' in primate.lower():
                    
                    assert unit_type is not None and used_id_num is not None, f"[Codwarning] for primate '{primate}', the unit_type can not be '{unit_type}' and used_id_num can not be '{used_id_num}'"
                    
                    if unit_type == 'legacy':
                        
                        RSA_dict_folds = __RSA_layer_wise_pct_collect_single_condition(f'-_mismatched_comparison/Human Selective V.S. NN Strong Selective/{used_id_num}/RSA_results_{criterion}_{used_id_num}_mismatched_Selective_Human_Strong_Selective_NN')
                                                                                                                                                                              
                    else:
                        
                        RSA_dict_folds = __RSA_layer_wise_pct_collect_single_condition(f'{unit_type}/{used_id_num}/RSA_results_{criterion}_{unit_type}_{used_id_num}')

                utils_.dump(RSA_dict_folds, RSA_static_dict_folds_path)
                
            return RSA_dict_folds
        
        # -----
        def _process(primate='Monkey', criteria=['euclidean', 'pearson']):
            
            utils_.make_dir(RSA_save_root_primate:=os.path.join(RSA_save_root, f'{primate}'))
            
            if 'monkey' in primate.lower():
                
                Monkey_Neuron_Records_Process.__init__(self, )
                
                for criterion in criteria:
                
                    _static(RSA_save_root_primate, primate=f'{primate}', criterion=f'{criterion}')
                    
                    for route in ['sig', 'p']:
                        
                        _temporal(RSA_save_root_primate, primate=f'{primate}', criterion=f'{criterion}', route=route)
                
            elif 'human' in primate.lower():
                
                Human_Neuron_Records_Process.__init__(self, )
                
                for criterion in criteria:
                    
                    if 'euclidean' == criterion:
                        
                        for used_id_num in [10, 50]:
                            
                            _static(RSA_save_root_primate, primate=f'{primate}', criterion=f'{criterion}', unit_type='legacy', used_id_num=used_id_num)
                            
                            for route in ['sig', 'p']:
                                
                                _temporal(RSA_save_root_primate, primate=f'{primate}', criterion=f'{criterion}', unit_type='legacy', used_id_num=used_id_num, route=route)
                                
                    elif 'pearson' == criterion:

                        for unit_type in ['legacy', 'qualified', 'selective', 'non_selective']:
                            
                            for used_id_num in [10, 50]:
                                
                                _static(RSA_save_root_primate, primate=f'{primate}', criterion=f'{criterion}', unit_type=unit_type, used_id_num=used_id_num)
                                
                                for route in ['sig', 'p']:
                                    
                                    _temporal(RSA_save_root_primate, primate=f'{primate}', criterion=f'{criterion}', unit_type=unit_type, used_id_num=used_id_num, route=route)
                    else:
                        
                        raise ValueError
            
            else:
                
                raise ValueError
            
        # FIXME --- need to upgrade and simplify
        def _temporal(RSA_save_root_primate, primate='Monkey', criterion='pearson', unit_type:str=None, used_id_num:int=None, route:str='p'):
            
            utils_.make_dir(RSA_save_root_primate_criteria:=os.path.join(RSA_save_root_primate, f'{criterion}'))
            
            RSA_dict_folds = _RSA_layer_wise_pct_collect(RSA_save_root_primate_criteria, f'{primate}', f'{criterion}', unit_type=unit_type, used_id_num=used_id_num)
            
            # ---
            similarity_folds_array = np.array([RSA_dict_folds[fold_idx]['similarity_temporal'] for fold_idx in range(1, self.num_fold)])
            folds_mean = np.mean(similarity_folds_array, axis=0)
            folds_std = np.std(similarity_folds_array, axis=0)
            
            if route == 'p':
                
                # --- init
                p_temporal_FDR = np.zeros((len(self.layers), similarity_folds_array.shape[-1]))     # (num_layers, num_time_steps)
                sig_temporal_FDR =  p_temporal_FDR.copy()
                sig_temporal_Bonf = p_temporal_FDR.copy()
                
                similarity_p_perm_temporal = np.mean(np.array([RSA_dict_folds[fold_idx]['similarity_p_perm_temporal'] for fold_idx in range(1, self.num_fold)]), axis=0)
                
                for _ in range(len(self.layers)):
                    (sig_temporal_FDR[_, :], p_temporal_FDR[_, :], alpha_Sadik_temporal, alpha_Bonf_temporal) = multipletests(similarity_p_perm_temporal[_, :], alpha=alpha, method=FDR_method)      # FDR
                    sig_temporal_Bonf[_, :] = p_temporal_FDR[_, :]<alpha_Bonf_temporal     # Bonf correction
            
            elif route == 'sig':
                
                sig_temporal_FDR = np.mean(np.array([scipy.ndimage.gaussian_filter(RSA_dict_folds[fold_idx]['sig_temporal_FDR'], sigma=1) for fold_idx in range(1, self.num_fold)]), axis=0)
                sig_temporal_Bonf = np.mean(np.array([scipy.ndimage.gaussian_filter(RSA_dict_folds[fold_idx]['sig_temporal_Bonf'], sigma=1) for fold_idx in range(1, self.num_fold)]), axis=0)
                p_temporal_FDR =  np.mean(np.array([RSA_dict_folds[fold_idx]['p_temporal_FDR'] for fold_idx in range(1, self.num_fold)]), axis=0)
            
            RSA_dict_across_folds = {
                'similarity_temporal': folds_mean,
                'similarity_temporal_std': folds_std,
                
                'sig_temporal_FDR': sig_temporal_FDR,
                'sig_temporal_Bonf': sig_temporal_Bonf,
                
                'p_temporal_FDR': p_temporal_FDR,
                
                }
            
            if 'monkey' in primate.lower():
                extent = [self.ts.min()-5, self.ts.max()+5, -0.5, folds_mean.shape[0]-0.5]
            elif 'human' in primate.lower():
                extent = [-250, 1001, -0.5, folds_mean.shape[0]-0.5]
            
            fig = plt.figure(figsize=(10, folds_mean.shape[0]/4))
            ax = fig.add_axes([0.125, 0.075, 0.75, 0.85])
            
            title = f'RSA temporal {primate} {criterion} {unit_type} {used_id_num} {self.model_structure}'
            Selectivity_Analysis_RSA.plot_temporal_correlation(self.layers, fig, ax, RSA_dict_across_folds, title=f'{title}', vlim=None, extent=extent)
            
            mask = RSA_dict_across_folds['sig_temporal_Bonf']
            if not utils_._is_binary(mask):
                mask = mask>(1-alpha)
            utils_similarity.fake_legend_describe_numpy(RSA_dict_across_folds['similarity_temporal'], ax, mask.astype(bool))
             
            fig.savefig(os.path.join(RSA_save_root_primate_criteria, f'{title}_merged_by_{route}.png'))
            plt.close()
            
            # ---
            idx, layer_n, _, _ = utils_.activation_function(self.model_structure, self.layers)
            RSA_dict_across_folds_neuron = {_:RSA_dict_across_folds[_][idx] for _ in RSA_dict_across_folds.keys()}
            
            folds_mean = folds_mean[idx]
            
            if 'monkey' in primate.lower():
                extent = [self.ts.min()-5, self.ts.max()+5, -0.5, folds_mean.shape[0]-0.5]
            elif 'human' in primate.lower():
                extent = [-250, 1001, -0.5, folds_mean.shape[0]-0.5]
            
            fig = plt.figure(figsize=(10, folds_mean.shape[0]/4))
            ax = fig.add_axes([0.125, 0.075, 0.75, 0.85])
            
            title = f'RSA act temporal {primate} {criterion} {unit_type} {used_id_num} {self.model_structure}'
            Selectivity_Analysis_RSA.plot_temporal_correlation(layer_n, fig, ax, RSA_dict_across_folds_neuron, title=f'{title}', vlim=None, extent=extent)
            
            mask = RSA_dict_across_folds_neuron['sig_temporal_Bonf']
            if not utils_._is_binary(mask):
                mask = mask>(1-alpha)
            utils_similarity.fake_legend_describe_numpy(RSA_dict_across_folds_neuron['similarity_temporal'], ax, mask.astype(bool))
             
            fig.savefig(os.path.join(RSA_save_root_primate_criteria, f'{title}_merged_by_{route}.png'))
            plt.close()
            

        def _static(RSA_save_root_primate, primate='Monkey', criterion='pearson', unit_type:str=None, used_id_num:int=None):
            
            utils_.make_dir(RSA_save_root_primate_criteria:=os.path.join(RSA_save_root_primate, f'{criterion}'))
    
            RSA_dict_folds = _RSA_layer_wise_pct_collect(RSA_save_root_primate_criteria, f'{primate}', f'{criterion}', unit_type=unit_type, used_id_num=used_id_num)
                
            # --- merge
            similarity_folds_array = np.array([RSA_dict_folds[fold_idx]['similarity'] for fold_idx in range(1, self.num_fold)])
            folds_mean = np.mean(similarity_folds_array, axis=0)
            folds_std = np.std(similarity_folds_array, axis=0)
            
            similarity_p_folds = np.mean(np.array([RSA_dict_folds[fold_idx]['similarity_p'] for fold_idx in range(1, self.num_fold)]), axis=0)
            (sig_FDR, p_FDR, alpha_Sadik, alpha_Bonf) = multipletests(similarity_p_folds, alpha=alpha, method=FDR_method)    
    
            RSA_dict_across_folds = {
                'similarity': folds_mean,
                'similarity_std': folds_std,
                'similarity_perm': np.max(np.array([RSA_dict_folds[fold_idx]['similarity_perm'] for fold_idx in range(1, self.num_fold)]), axis=0),
                
                'p_FDR': p_FDR,
                'similarity_p': similarity_p_folds,
                
                'sig_FDR': sig_FDR,
                'sig_Bonf': p_FDR<alpha_Bonf
                }
            
            # ---
            fig, ax = plt.subplots(figsize=(int(len(self.layers)/3), 6))
            title = f'RSA {primate} {criterion} {unit_type} {used_id_num} {self.model_structure}'
            Selectivity_Analysis_RSA.plot_static_correlation(self.layers, ax, RSA_dict_across_folds, title=title, legend=False)
            utils_similarity.fake_legend_describe_numpy(RSA_dict_across_folds['similarity'], ax, RSA_dict_across_folds['sig_FDR'].astype(bool))
            
            idx, layer_n, _, _ = utils_.activation_function(self.model_structure, self.layers)
            RSA_dict_across_folds_neuron = {_:RSA_dict_across_folds[_][idx] for _ in RSA_dict_across_folds.keys()}
            
            plt.tight_layout()
            fig.savefig(os.path.join(RSA_save_root_primate_criteria, f'{title}.png'))
            plt.close()
            
            fig, ax = plt.subplots(figsize=(int(len(layer_n)/3), 6))
            title = f'RSA {primate} {criterion} {unit_type} {used_id_num} act {self.model_structure}'
            Selectivity_Analysis_RSA.plot_static_correlation(layer_n, ax, RSA_dict_across_folds_neuron, title=title, legend=False)
            utils_similarity.fake_legend_describe_numpy(RSA_dict_across_folds_neuron['similarity'], ax, RSA_dict_across_folds_neuron['sig_FDR'].astype(bool))
            
            plt.tight_layout()
            fig.savefig(os.path.join(RSA_save_root_primate_criteria, f'{title}.png'))
            plt.close()
            
        # -----
        _process(primate='Monkey')
        _process(primate='Human')
        
    
    
    def plot_SVM_acc_multi_models(self, num_types=5):
    
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        utils_.make_dir(SVM_save_root:=os.path.join(self.model_root, 'SVM'))
        
        def _SVM_curve_acc_collect():
            
            curve_dict_folds_path = os.path.join(SVM_save_root, 'SVM_folds_dict.pkl')                
            
            if os.path.exists(curve_dict_folds_path):
                
                curve_dict_folds = utils_.load(curve_dict_folds_path, verbose=False)
            
            else:
                
                curve_dict_folds = {}
        
                for fold_idx in np.arange(1, self.num_fold):
                    
                    root = os.path.join(self.model_root+str(fold_idx))

                    curve_dict_folds[fold_idx] = utils_.load(os.path.join(root, f'Analysis/Encode/SVM_types_{num_types}.pkl'), verbose=True)
                    
                utils_.dump(curve_dict_folds, curve_dict_folds_path)
            
            return curve_dict_folds
            
        def _plot_SVM_single(title, target_layers, _acc_plot_dict, _std_plot_dict):
            
            fig, ax = plt.subplots(figsize=(int(len(target_layers)/3), 10))
            Selectivity_Analysis_Encode.Encode_feaquency_analyzer.SVM_plot_single_fig(ax, self.model_structure, target_layers, _acc_plot_dict, _std_plot_dict)

            ax.set_title(title:=f'{title} [{self.model_structure}]')
            fig.savefig(os.path.join(SVM_save_root, f'{title}.png'), bbox_inches='tight')
            plt.close()
        
        curve_dict_folds = _SVM_curve_acc_collect()
        
        unit_types = ['all', 's_si', 's_mi', 's_wsi', 's_wmi', 'n_e']
        
        SVM_acc_dict = {}
        
        for layer in self.layers:

            SVM_acc_dict[layer] = {unit_type: [curve_dict_folds[fold_idx][layer][f'{unit_type}_acc'] for fold_idx in range(1, self.num_fold)] for unit_type in unit_types}
        
        acc_plot_dict = {f'{_}_acc': np.array([np.mean(SVM_acc_dict[layer][f'{_}']) for layer in self.layers]) for _ in unit_types}
        std_plot_dict = {f'{_}_acc': np.array([np.std(SVM_acc_dict[layer][f'{_}']) for layer in self.layers]) for _ in unit_types}
        
        _plot_SVM_single('5 types', self.layers, acc_plot_dict, std_plot_dict)
        
        # --- act
        act_idx, act_layers, act_neurons, _ = utils_.activation_function(self.model_structure, self.layers)
        
        acc_plot_dict = {_: acc_plot_dict[_][act_idx] for _ in acc_plot_dict.keys()}
        std_plot_dict = {_: std_plot_dict[_][act_idx] for _ in std_plot_dict.keys()}
        
        _plot_SVM_single('5 types act', act_layers, acc_plot_dict, std_plot_dict)

    
    def plot_Encode_pct_multi_models(self, ):
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})

        utils_.make_dir(Encode_save_root:=os.path.join(self.model_root, 'Encode'))
        
        def _Encode_curve_pct_collect():

            curve_dict_folds_path = os.path.join(Encode_save_root, 'Encode_folds_dict.pkl')                
            
            if os.path.exists(curve_dict_folds_path):
                
                curve_dict_folds = utils_.load(curve_dict_folds_path, verbose=False)
            
            else:
                
                curve_dict_folds = {}
        
                for fold_idx in np.arange(1, self.num_fold):
                    
                    root = os.path.join(self.model_root+str(fold_idx))
                    
                    Sort_dict = utils_.load(os.path.join(root, 'Analysis/Encode/Sort_dict.pkl'), verbose=True)
                    
                    Encode_types_pct = Selectivity_Analysis_Encode.Encode_feaquency_analyzer.obtain_Encode_types_pct(self.layers, self.neurons, Sort_dict)
                    
                    curve_dict_folds[fold_idx] = Selectivity_Analysis_Encode.Encode_feaquency_analyzer.obtain_Encode_types_curve_dict(Encode_types_pct)
                    
                utils_.dump(curve_dict_folds, curve_dict_folds_path)
            
            return curve_dict_folds
        
        curve_dict_folds = _Encode_curve_pct_collect()
        
        #TODO --- rebuild the curve dict and simplify
        types = list(curve_dict_folds[1].keys())
        
        curve_folds = {}
        
        for type_ in types:
             
            values_folds_array = np.array([curve_dict_folds[i][type_]['values'] for i in range(1, self.num_fold)])
            
            folds_mean = np.mean(values_folds_array, axis=0)
            folds_std = np.std(values_folds_array, axis=0)  
            
            color = curve_dict_folds[1][type_]['color']
            if color == 'black':
                color = '#555555'
            
            curve_folds[type_] = {'color': color,
                                  'linestyle': curve_dict_folds[1][type_]['linestyle'],
                                  'linewidth': curve_dict_folds[1][type_]['linewidth'],
                                  'label': curve_dict_folds[1][type_]['label'],
                                  'values': folds_mean,
                                  'std': folds_std}
            
        # -----
        fig, ax = plt.subplots(figsize=(10,6))
        Selectivity_Analysis_Encode.Encode_feaquency_analyzer.encode_layer_percent_plot(fig, ax, self.layers, curve_folds, None)
        
        title = 'Encode_pct_5_types'
        ax.set_title(f'{self.model_structure} {title}')
        fig.savefig(os.path.join(Encode_save_root, title+'.png'), bbox_inches='tight')
        fig.savefig(os.path.join(Encode_save_root, title+'.eps'), bbox_inches='tight')     
        plt.close()
        
        # -----
        act_idx, act_layers, act_neurons, _ = utils_.activation_function(self.model_structure, self.layers, self.neurons)
        
        for type_ in types:
            
            curve_folds[type_]['values'] = curve_folds[type_]['values'][act_idx]
            curve_folds[type_]['std'] = curve_folds[type_]['std'][act_idx]
        
        fig, ax = plt.subplots(figsize=(10,6))
        Selectivity_Analysis_Encode.Encode_feaquency_analyzer.encode_layer_percent_plot(fig, ax, act_layers, curve_folds, None)
        
        title = 'Encode_pct_act_5_types'
        ax.set_title(f'{self.model_structure} {title}')
        fig.savefig(os.path.join(Encode_save_root, title+'.png'), bbox_inches='tight')
        fig.savefig(os.path.join(Encode_save_root, title+'.eps'), bbox_inches='tight')     
        plt.close()
        
        
    # ------------------------------------------------------------------------------------------------------------------
    def plot_ANOVA_pct_multi_models(self, ):
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        ANOVA_save_root = os.path.join(self.model_root, 'ANOVA')
        utils_.make_dir(ANOVA_save_root)
        
        def _ANOVA_pct_collect():
            
            ANOVA_pct_folds_path = os.path.join(ANOVA_save_root, 'ANOVA_folds_array.pkl')
            
            if os.path.exists(ANOVA_pct_folds_path):
                
                ANOVA_folds_array = utils_.load(ANOVA_pct_folds_path)
            
            else:
            
                ANOVA_folds = {}
        
                for fold_idx in np.arange(1, self.num_fold):
                    
                    root = os.path.join(self.model_root+str(fold_idx))
                    
                    ANOVA_folds[fold_idx] = utils_.load(os.path.join(root, 'Analysis', 'ANOVA', 'ratio.pkl'), verbose=False)
                    
                ANOVA_folds_array = np.array([np.array(_) for _ in list(ANOVA_folds.values())])     # (num_folds, num_layers)
                
                utils_.dump(ANOVA_folds_array, ANOVA_pct_folds_path)
            
            return ANOVA_folds_array
        
        def _ANOVA_pct_plot(ax, layers, ANOVA_folds_array, title):
            
            folds_mean = np.mean(ANOVA_folds_array, axis=0)
            folds_std = np.std(ANOVA_folds_array, axis=0)  
            
            ax.fill_between(np.arange(len(layers)), folds_mean-folds_std, folds_mean+folds_std, edgecolor=None, facecolor='skyblue', alpha=0.75)
            ax.plot(np.arange(len(layers)), folds_mean, color='blue', linewidth=0.5)
            ax.set_xticks(np.arange(len(layers)), layers, rotation='vertical')
            ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
            ax.set_title(f'{self.model_structure} {title}')
            
            plt.tight_layout()
            fig.savefig(os.path.join(ANOVA_save_root, f'{title}_folds.png'))
            fig.savefig(os.path.join(ANOVA_save_root, f'{title}_folds.eps'))
            plt.close()
        
        # ---
        ANOVA_folds_array = _ANOVA_pct_collect()
        
        # ---
        fig, ax = plt.subplots(figsize=(18,10))
        _ANOVA_pct_plot(ax, self.layers, ANOVA_folds_array, 'ANOVA_pct')

        # ---
        act_idx, act_layers, act_neurons, _ = utils_.activation_function(self.model_structure, self.layers, self.neurons)
        ANOVA_folds_array = ANOVA_folds_array[:, act_idx]
        
        fig, ax = plt.subplots(figsize=(10,10))
        _ANOVA_pct_plot(ax, act_layers, ANOVA_folds_array, 'ANOVA_pct_act')


def remove_early_layer_features():
    
    for fold_idx in tqdm(['IF', 'LIF']):
        
        feature_root = f'/home/acxyle-workstation/Downloads/Face Identity SpikingResnet18_{fold_idx}_T4_CelebA2622/Features'
        
        layers, neurons, shapes = utils_.get_layers_and_units(args.model)
        
        #utils_.describe_model(layers, neurons, shapes)
        
        feature_files = os.listdir(feature_root)
        
        for feature_file in feature_files:
            
            if feature_file.split('.')[0] not in layers[-12:]:
                
                os.remove(os.path.join(feature_root, feature_file))


# ----------------------------------------------------------------------------------------------------------------------
def single_model_analysis(args, feature_folder):

    start_time = time.time()
    
    # --- init
    feature_root = os.path.join(args.root_dir, feature_folder)
    
    # --- determine num_units
    layers, neurons, shapes = utils_.get_layers_and_units(args.model, target_layers='act')
    
    # ---
    utils_._print(f'Listing model [{feature_folder}] layers and neuron numbers')
    utils_.describe_model(layers, neurons, shapes)
    
    # ----- 1. ANOVA
    ANOVA_analyzer = Selectivity_Analysis_ANOVA.ANOVA_analyzer(
                                                               feature_root, 
                                                               alpha=args.alpha, num_classes=args.num_classes, num_samples=args.num_samples, 
                                                               layers=layers, neurons=neurons)
    
    ANOVA_analyzer.calculation_ANOVA()
    ANOVA_analyzer.plot_ANOVA_pct()
    
    del ANOVA_analyzer     # release memory space
    
    # ----- 2. Encode
    Encode_analyzer = Selectivity_Analysis_Encode.Encode_feaquency_analyzer(
                                                                            feature_root, 
                                                                            layers=layers, neurons=neurons)
    
    Encode_analyzer.calculation_Encode()
    
    #Encode_analyzer.plot_Encode_pct(num_types=23)
    #Encode_analyzer.plot_Encode_pct(num_types=5)
    
    #Encode_analyzer.plot_Encode_freq()
    
    # ---
    #Encode_analyzer.SVM_analysis()
    
    # ---
    #Encode_analyzer.plot_stacked_responses(num_types=5)
    #Encode_analyzer.plot_stacked_responses(num_types=10)

    #Encode_analyzer.plot_sample_responses()
    
    # FIXME ---
    #Encode_analyzer.plot_unit_responses_PDF()
    
    #Encode_analyzer.plot_pct_pie_chart()
    
    del Encode_analyzer

    # ----- 3. DR SM
    #DR_analyzer = Selectivity_Analysis_DR_and_SM.Selectiviy_Analysis_DR(feature_root, layers=layers, neurons=neurons)
    #DR_analyzer.selectivity_analysis_tsne()
    #del DR_analyzer
    
    # ---
    SM_analyzer = Selectivity_Analysis_DR_and_SM.Selectivity_Analysis_SM(feature_root, layers=layers, neurons=neurons)
    for metric in ['euclidean', 'pearson']:
        SM_analyzer.selectivity_analysis_similarity_metrics(metric)
    
    del SM_analyzer

    # ----- 4. RSA
    RSA_monkey = Selectivity_Analysis_RSA.Selectivity_Analysis_Correlation_Monkey(NN_root=feature_root, layers=layers, neurons=neurons)

    for first_corr in ['euclidean', 'pearson', 'spearman', 'mahalanobis', 'concordance']:
        for second_corr in ['pearson', 'spearman', 'concordance']:
            RSA_monkey.monkey_neuron_analysis(first_corr=first_corr, second_corr=second_corr)

    del RSA_monkey
    
    RSA_human = Selectivity_Analysis_RSA.Selectivity_Analysis_Correlation_Human(NN_root=feature_root, layers=layers, neurons=neurons)
    
    #
    # 
    # 
    # 
    for firsct_corr in ['euclidean', 'pearson', 'mahalanobis', 'spearman', 'concordance']:
        for second_corr in ['pearson', 'spearman', 'concordance']:
            for used_cell_type in ['legacy', 'qualified', 'selective', 'non_selective']:
                for used_id_num in [50, 10]:
                    RSA_human.human_neuron_analysis(first_corr=firsct_corr, second_corr=second_corr, used_cell_type=used_cell_type, used_id_num=used_id_num)

    del RSA_human
    
    # ----- 5. Feature
    #selectivity_feature_analyzer = Selectivity_Analysis_Feature.Selectivity_Analysis_Feature(feature_root, 'TSNE')
    #selectivity_feature_analyzer.feature_analysis()
    #del selectivity_feature_analyzer
    
    
    # ----- 6. CKA
    CKA_monkey = Selectivity_Analysis_CKA.Selectivity_Analysis_CKA_Monkey(NN_root=feature_root, layers=layers, neurons=neurons)
    
    CKA_monkey.monkey_neuron_analysis(kernel='linear')
    for threshold in [0.5, 1.0, 2.0, 10.0]:
        CKA_monkey.monkey_neuron_analysis(kernel='rbf', threshold=threshold)
    
    del CKA_monkey
    
    CKA_human = Selectivity_Analysis_CKA.Selectivity_Analysis_Correlation_Human(NN_root=feature_root, layers=layers, neurons=neurons)
    
    # 
    for used_cell_type in ['qualified', 'selective', 'non_selective', 'legacy']:
        for used_id_num in [50, 10]:
            CKA_human.human_neuron_analysis(kernel='linear', used_cell_type=used_cell_type, used_id_num=used_id_num)
            for threshold in [0.5, 1.0, 2.0, 10.0]:
                CKA_human.human_neuron_analysis(kernel='rbf', threshold=threshold, used_cell_type=used_cell_type, used_id_num=used_id_num)
    
    del CKA_human
    
    # --- 
    end_time = time.time()
    elapsed = end_time - start_time
    
    utils_._print(f"All results are saved in {os.path.join(feature_root, 'Analysis')}")
    utils_._print('Elapsed Time: {}:{:0>2}:{:0>2} '.format(int(elapsed/3600), int((elapsed%3600)/60), int((elapsed%3600)%60)))
    utils_._print('Experiment Done.')    



#FIXME
def Main_Analyzer(args):

    utils_._print('Starting Selective Analysis Experiment...')
    print(args)

    # ['ATan', 'piecewise_exp', 'QPeusudoSPike', 'sigmoid', 'softsign', ''     # good
    #'piecewiseQuadratic', 'Erf',]     # piecewiseQuadratic, erf is poor IF but good LIF, 

# =============================================================================
#     for _neuron in ['IF']:
#         for _surrogate in ['ATan']:
#             for _T in [16]:
#                 for _fold_idx in [1,2,3,4]:
#                     single_model_analysis(args, f'Face Identity SpikingResnet18_{_neuron}_{_surrogate}_T{_T}_CelebA2622_fold_{_fold_idx}')
# =============================================================================
    
    for fold_idx in [1,2,3,4]:
        single_model_analysis(args, f'Face Identity SpikingVGG16bn_LIF_T4_CelebA2622_fold_/-_Single Models/Face Identity SpikingVGG16bn_LIF_T4_CelebA2622_fold_{fold_idx}')

if __name__ == "__main__":
    
    Main_Analyzer(args)
    
    #remove_early_layer_features()
    
# =============================================================================
#     multi_model_analysis = Multi_Model_Analysis(feature_root_general='Face Identity SpikingResnet18_IF_ATan_T16_CelebA2622_fold_')
#     
#     multi_model_analysis.plot_ANOVA_pct_multi_models()
#     multi_model_analysis.plot_Encode_pct_multi_models()
#     multi_model_analysis.plot_RSA_pct_multi_models()
#     multi_model_analysis.plot_SVM_acc_multi_models()
# =============================================================================

