#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:35:00 2023

@author: acxyle

    ...
    
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

import FSA_ANOVA, FSA_Encode, FSA_DRG, FSA_RSA, FSA_CKA
import Selectivity_Analysis_Feature

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
parser.add_argument("--model", type=str, default='vgg16_bn')     # trigger

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
            """
                this function uses 2 routes to obtain the target p values, route 'p' uses the mean values of all p values 
                then calculate the FDR test again; route 'sig' uses the smoothed mean values of sig results(T/F) 
            """
            
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
    


# ----------------------------------------------------------------------------------------------------------------------
def single_model_analysis(args, feature_folder):
    """
        used_id_num=10 is proved meaningless for CKA
    """
    start_time = time.time()
    
    # --- init
    feature_root = os.path.join(args.root_dir, feature_folder)
    
    # --- determine num_units
    layers, neurons, shapes = utils_.get_layers_and_units(args.model, target_layers='act')
    
    # ---
    utils_._print(f'Listing model [{feature_folder}] layers and neuron numbers')
    utils_.describe_model(layers, neurons, shapes)
    
    # ----- 1. ANOVA
    ANOVA_analyzer = FSA_ANOVA.FSA_ANOVA(root=feature_root, layers=layers, neurons=neurons, alpha=args.alpha, num_classes=args.num_classes, num_samples=args.num_samples)
         
    ANOVA_analyzer.calculation_ANOVA()
    ANOVA_analyzer.plot_ANOVA_pct()
    
    del ANOVA_analyzer     # release memory space
    
    # ----- 2. Encode
    Encode_analyzer = FSA_Encode.FSA_Encode(root=feature_root, layers=layers, neurons=neurons)
    
    Encode_analyzer.calculation_Encode()
    Encode_analyzer.plot_Encode_pct_single()
    Encode_analyzer.plot_Encode_pct_comprehensive()
    Encode_analyzer.plot_Encode_freq()
    
    del Encode_analyzer

    Responses_analyzer = FSA_Encode.FSA_Responses(root=feature_root, layers=layers, neurons=neurons)
    Responses_analyzer.plot_unit_responses()
    Responses_analyzer.plot_stacked_responses(num_types=5)
    Responses_analyzer.plot_responses_PDF()
    Responses_analyzer.plot_pct_pie_chart()
    
    del Responses_analyzer
    
    SVM_analyzer = FSA_Encode.FSA_SVM(root=feature_root, layers=layers, neurons=neurons)
    SVM_analyzer.process_SVM()    

    del SVM_analyzer

    # ----- 3. DR, DSM, Gram
    DR_analyzer = FSA_DRG.FSA_DR(feature_root, layers=layers, neurons=neurons)
    DR_analyzer.DR_TSNE()
    del DR_analyzer
    
    DSM_analyzer = FSA_DRG.FSA_DSM(feature_root, layers=layers, neurons=neurons)
    DSM_analyzer.process_DSM(metric='pearson')
    del DSM_analyzer
    
    Gram_analyzer = FSA_DRG.FSA_Gram(feature_root, layers=layers, neurons=neurons)
    Gram_analyzer.calculation_Gram(kernel='linear', normalize=True)
    for threshold in [0.5, 1.0, 2.0, 10.0]:
        Gram_analyzer.calculation_Gram(kernel='rbf', threshold=threshold)
    del Gram_analyzer

    # ----- 4. RSA
    #for monkey experiments
    RSA_monkey = FSA_RSA.RSA_Monkey(NN_root=feature_root, layers=layers, neurons=neurons)
    
    for first_corr in ['euclidean', 'pearson', 'spearman', 'mahalanobis', 'concordance']:
        for second_corr in ['pearson', 'spearman', 'concordance']:
            RSA_monkey.monkey_neuron_analysis(first_corr=first_corr, second_corr=second_corr)
            
    del RSA_monkey
            
    # for human experiments 
    RSA_human = FSA_RSA.RSA_Human(NN_root=feature_root, layers=layers, neurons=neurons)
    
    for firsct_corr in ['euclidean', 'mahalanobis', 'spearman']:
        for second_corr in ['pearson', 'spearman', 'concordance']:
            for used_cell_type in ['legacy', 'qualified', 'selective', 'non_selective']:
                for used_id_num in [50, 10]:
                    RSA_human.process_RSA_human(first_corr=firsct_corr, second_corr=second_corr, used_cell_type=used_cell_type, used_id_num=used_id_num)
    
    del RSA_human
    
    # ----- 5. CKA
    CKA_monkey = FSA_CKA.CKA_Similarity_Monkey(feature_root, layers=layers, neurons=neurons)
    CKA_monkey.process_CKA_monkey(kernel='linear', normalize=True)
    for threshold in [0.5, 1.0, 2.0, 10.0]:
        CKA_monkey.process_CKA_monkey(kernel='rbf', threshold=threshold)
    
    
    CKA_human = FSA_CKA.CKA_Similarity_Human(feature_root, layers=layers, neurons=neurons)
    for used_unit_type in ['legacy', 'qualified', 'selective', 'non_selective']:
        CKA_human.process_CKA_human(kernel='linear', used_unit_type=used_unit_type)
        for threshold in [0.5, 1.0, 10.0]:
            CKA_human.process_CKA_human(kernel='rbf', threshold=threshold, used_unit_type=used_unit_type)
    
    
    # ----- 6. Feature
    #selectivity_feature_analyzer = Selectivity_Analysis_Feature.Selectivity_Analysis_Feature(feature_root, 'TSNE')
    #selectivity_feature_analyzer.feature_analysis()
    #del selectivity_feature_analyzer
    
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
    
    for fold_idx in [0,1,2,3,4]:
        single_model_analysis(args, f'Face Identity VGG16bn_fold_/-_Single Models/Face Identity VGG16bn_fold_{fold_idx}')

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

