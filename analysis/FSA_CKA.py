#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:47:40 2024

@author: acxyle-workstation

    [task] (1) monkey; (2) human; (3) ANN-SNN

"""

import torch

import os
import pickle
import warnings
import logging
import numpy as np

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from joblib import Parallel, delayed
import scipy

from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind

from statsmodels.stats.multitest import multipletests
import itertools

import utils_
import utils_similarity

from Bio_Cell_Records_Process import Human_Neuron_Records_Process, Monkey_Neuron_Records_Process
from FSA_DRG import FSA_Gram

import sys
sys.path.append('../')
import models_

  
# ----------------------------------------------------------------------------------------------------------------------
class CKA_Similarity_base():
    
    def __init__(self, ):
        
        self.ts
        
        self.save_root
        self.layers
        
        self.primate_DM
        self.primate_DM_perm
        
        # --- if use permutation every time, those two can be ignored, 5-fold experiment is suggested
        # --- empirically, the max fluctuation of mean scores between experiments could be ±0.03 with num_perm = 1000
        self.primate_DM_temporal
        self.primate_DM_temporal_perm
    
    
    def calculation_CKA_Similarity(self, kernel='linear', FDR_test=True, alpha=0.05, FDR_method='fdr_bh', save=True, primate=None, **kwargs):
        """
            calculation and save the RSA results for monkey
        """
        
        if kernel == 'rbf' and 'threshold' in kwargs:
            save_path = os.path.join(self.save_root, f"CKA_results_{kernel}_{kwargs['threshold']}.pkl")
        elif kernel == 'linear':
            save_path = os.path.join(self.save_root, f"CKA_results_{kernel}.pkl")
        else:
            raise ValueError(f'[Coderror] Invalid kernel [{kernel}]')
        
        if os.path.exists(save_path):
            
            cka_dict = utils_.load(save_path, verbose=True)
            
        else:

            pl = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(self.calculation_CKA_layer)(layer, kernel=kernel, FDR_test=FDR_test, **kwargs) for layer in tqdm(self.layers, desc=f'CKA {primate}'))
            
            if not FDR_test:
                
                pl_k = ['cka_fr', 'cka_psth']
                
                assert set(pl[0].keys()) == set(pl_k)
                
                extracted_data = [np.array([_[__] for _ in pl]) for __ in pl_k]

                cka_dict = dict(zip(pl_k, extracted_data))
            
            else:
                
                pl_k = ['cka_fr', 'cka_fr_perm', 'p_perm', 'cka_psth', 'cka_psth_perm', 'p_temporal_perm']
            
                assert set(pl[0].keys()) == set(pl_k)
                
                cka_score, cka_score_perm, p, cka_score_temporal, cka_score_temporal_perm, p_temporal = [np.array([_[__] for _ in pl]) for __ in pl_k]
                
                # --- static
                (sig_FDR, p_FDR, alpha_Sadik, alpha_Bonf) = multipletests(p, alpha=alpha, method=FDR_method)    # FDR (flase discovery rate) correction
                sig_Bonf = p_FDR<alpha_Bonf
                
                # --- temporal
                time_steps = self.primate_Gram_temporal.shape[0]
                
                p_temporal_FDR = np.zeros((len(self.layers), time_steps))     # (num_layers, num_time_steps)
                sig_temporal_FDR =  np.zeros_like(p_temporal_FDR, dtype=bool)
                sig_temporal_Bonf = np.zeros_like(p_temporal_FDR, dtype=bool)
                
                for _ in range(len(self.layers)):
                    
                    (sig_temporal_FDR[_, :], p_temporal_FDR[_, :], alpha_Sadik_temporal, alpha_Bonf_temporal) = multipletests(p_temporal[_, :], alpha=alpha, method=FDR_method)      # FDR
                    sig_temporal_Bonf[_, :] = p_temporal_FDR[_, :]<alpha_Bonf_temporal     # Bonf correction
                
                # --- seal results
                cka_dict = {
                    'cka_score': cka_score,
                    'cka_score_perm': cka_score_perm,
                    'p': p,
                    
                    'cka_score_temporal': cka_score_temporal,
                    'cka_score_temporal_perm': cka_score_temporal_perm,
                    'p_temporal': p_temporal,
                    
                    'p_FDR': p_FDR,
                    'sig_FDR': sig_FDR,
                    'sig_Bonf': sig_Bonf,
                    
                    'p_temporal_FDR': p_temporal_FDR,
                    'sig_temporal_FDR': sig_temporal_FDR,
                    'sig_temporal_Bonf': sig_temporal_Bonf,
                    }
                
                if save:
                    
                    utils_.dump(cka_dict, save_path)

        return cka_dict
    
    
    def calculation_CKA_layer(self, layer, kernel='linear', FDR_test=True, num_perm=1000, **kwargs):    
        """
            ...
        """
        
        # --- static
        cka_fr = cka(self.primate_Gram, self.Gram_dict[layer])
        
        # --- temporal
        time_steps = self.primate_Gram_temporal.shape[0]
        cka_psth = np.array([cka(self.primate_Gram_temporal[t], self.Gram_dict[layer]) for t in range(time_steps)])

        if FDR_test:
            
            cka_fr_perm = [cka(self.primate_Gram_perm[_], self.Gram_dict[layer]) for _ in range(num_perm)]
            p_perm = np.mean(cka_fr_perm > cka_fr)
            
            cka_psth_perm = [[cka(self.primate_Gram_temporal_perm[t, _, :, :], self.Gram_dict[layer]) for _ in range(num_perm)] for t in range(time_steps)]
            p_temporal_perm = [np.mean(cka_psth_perm[_] > cka_psth[_]) for _ in range(len(cka_psth))]
            
            results = {
                'cka_fr': cka_fr,
                'cka_fr_perm': cka_fr_perm,
                'p_perm': p_perm,
                'cka_psth': cka_psth,
                'cka_psth_perm': cka_psth_perm,
                'p_temporal_perm': p_temporal_perm
                }
            
        else:
        
            results = {
                'cka_fr': cka_fr,
                'cka_psth': cka_psth
                }
        
        return results
            

    def plot_CKA(self, cka_dict, kernel='linear', error_control_measure='sig_FDR', error_area=True, legend=False, vlim:list[float]=None, **kwargs):
        """
            this function plots CKA score
            
            input:
                layers: define the x axis
                cka_fr: CKA data
                title: 
                
        """
        
        utils_._print('Executing static plotting...')
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            if 'threshold' in kwargs:
                title = f"CKA score {self.model_structure} {kernel} {kwargs['threshold']}"
            elif kernel == 'linear':
                title = f'CKA score {self.model_structure} {kernel}'
            else:
                raise ValueError

            fig, ax = plt.subplots(figsize=(10,6))
            
            plot_CKA(self.layers, ax, cka_dict, title=title, vlim=vlim, legend=legend)
            utils_similarity.fake_legend_describe_numpy(cka_dict['cka_score'], ax, cka_dict[error_control_measure].astype(bool))

            fig.tight_layout(pad=1)
 
            fig.savefig(os.path.join(self.save_root, f'{title}.svg'), bbox_inches='tight')   
            plt.close()
            

    def plot_CKA_temporal(self, cka_dict, extent:list[float]=None, error_control_measure='sig_temporal_Bonf', vlim:list[float]=None, kernel='linear', **kwargs):
        """
            function
            
            self.ts should be determined by derived class
            
            input:
                ...
        """
        
        utils_._print('Executing temporal plotting')
        
        extent = [self.ts.min()-5, self.ts.max()+5, -0.5, cka_dict['cka_score_temporal'].shape[0]-0.5]
 
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
    
        logging.getLogger('matplotlib').setLevel(logging.ERROR)    
    
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            if 'threshold' in kwargs:
                title = f"CKA temporal score {self.model_structure} {kernel} {kwargs['threshold']}"
            elif kernel == 'linear':
                title = f'CKA temporal score {self.model_structure} {kernel}'
            else:
                raise ValueError
            
            fig, ax = plt.subplots(figsize=(np.array(cka_dict['cka_score_temporal'].T.shape)/3.7))
            
            plot_CKA_temporal(self.layers, fig, ax, cka_dict, title=title, vlim=vlim, extent=extent)
            utils_similarity.fake_legend_describe_numpy(cka_dict['cka_score_temporal'], ax, cka_dict[error_control_measure].astype(bool))

            fig.savefig(os.path.join(self.save_root, f'{title}.svg'))     

            plt.close()


class CKA_Similarity_Monkey(Monkey_Neuron_Records_Process, FSA_Gram, CKA_Similarity_base):
    """
        ...
        
        CKA results is not invariant to normalization process      
    """
    
    def __init__(self,  **kwargs):
        
        Monkey_Neuron_Records_Process.__init__(self, **kwargs)
        FSA_Gram.__init__(self, **kwargs)
 
        utils_.make_dir(CKA_root:=os.path.join(self.dest, 'CKA'))
        self.save_root = os.path.join(CKA_root, 'Monkey')
        utils_.make_dir(self.save_root)


    def process_CKA_monkey(self, kernel, normalize=False, FDR_test=True, **kwargs):
        """
            ...
        """
        
        # --- monkey init
        monkey_Gram_dict = self.monkey_neuron_Gram_process(kernel=kernel, **kwargs)
        
        self.primate_Gram = monkey_Gram_dict['monkey_Gram']
        self.primate_Gram_temporal = monkey_Gram_dict['monkey_Gram_temporal']
        
        if FDR_test:
            
            assert set(['monkey_Gram_perm', 'monkey_Gram_temporal_perm']).issubset(monkey_Gram_dict.keys())
            
            self.primate_Gram_perm = monkey_Gram_dict['monkey_Gram_perm']
            self.primate_Gram_temporal_perm = monkey_Gram_dict['monkey_Gram_temporal_perm']
        
        # --- NN init --- monkey only use the entire cells/units
        self.Gram_dict = {k:v['qualified'] for k,v in self.calculation_Gram(kernel=kernel, normalize=normalize, **kwargs).items()}
        
        # ----- calculation
        cka_dict = self.calculation_CKA_Similarity(kernel=kernel, FDR_test=FDR_test, primate='Monkey', **kwargs)
        
        # ----- plot
        self.plot_CKA(cka_dict, kernel=kernel, **kwargs)
        self.plot_CKA_temporal(cka_dict, kernel=kernel, **kwargs)
        

# ----------------------------------------------------------------------------------------------------------------------
class CKA_Similarity_Human(Human_Neuron_Records_Process, FSA_Gram, CKA_Similarity_base):
    """
        unlike RSA, the config of 10-ids has worse performance
    """
    
    def __init__(self, **kwargs):
        
        # ---
        Human_Neuron_Records_Process.__init__(self, **kwargs)
        FSA_Gram.__init__(self, **kwargs)
        
        utils_.make_dir(CKA_root:=os.path.join(self.dest, 'CKA'))
        self.save_root_primate = os.path.join(CKA_root, 'Human')
        utils_.make_dir(self.save_root_primate)
        
        
    def process_CKA_human(self, kernel='linear', used_unit_type='qualified', used_id_num=50, FDR_test=True, **kwargs):
        """
            ...
        """
        # --- additional parameters
        utils_._print(f'Used kernel: {kernel} | Used types: {used_unit_type} | Used ID: {used_id_num}')
        ...
        
        utils_.make_dir(save_root_kernel:=os.path.join(self.save_root_primate, f'{kernel}'))
        utils_.make_dir(save_root_cell_type:=os.path.join(save_root_kernel, used_unit_type))

        self.save_root = os.path.join(save_root_cell_type, str(used_id_num))
        utils_.make_dir(self.save_root)
        
        # --- init
        self.used_id = self.human_corr_select_sub_identities(used_id_num)

        NN_Gram_dict = self.calculation_Gram(kernel=kernel, used_unit_type=used_unit_type, **kwargs)
        
        if used_unit_type == 'legacy':
            human_Gram_dict = self.human_neuron_Gram_process(kernel, 'selective', **kwargs)
            self.Gram_dict = {_: NN_Gram_dict[_]['strong_selective'][np.ix_(self.used_id, self.used_id)] for _ in NN_Gram_dict.keys()}
        else:
            human_Gram_dict = self.human_neuron_Gram_process(kernel, used_unit_type, **kwargs)
            self.Gram_dict = {_: NN_Gram_dict[_][used_unit_type][np.ix_(self.used_id, self.used_id)] for _ in NN_Gram_dict.keys()}
            
        self.primate_Gram = human_Gram_dict['human_Gram'][np.ix_(self.used_id, self.used_id)]
        self.primate_Gram_temporal = np.array([_[np.ix_(self.used_id, self.used_id)] for _ in human_Gram_dict['human_Gram_temporal']])

        if FDR_test:
            
            assert set(['human_Gram_perm', 'human_Gram_temporal_perm']).issubset(human_Gram_dict.keys())
            
            self.primate_Gram_perm = np.array([_[np.ix_(self.used_id, self.used_id)] for _ in human_Gram_dict['human_Gram_perm']])
            self.primate_Gram_temporal_perm = np.array([np.array([__[np.ix_(self.used_id, self.used_id)] for __ in _]) for _ in human_Gram_dict['human_Gram_temporal_perm']])
        
        # ----- calculation
        cka_dict = self.calculation_CKA_Similarity(kernel=kernel, used_unit_type=used_unit_type, used_id_num=used_id_num, primate='Human', **kwargs)
        
        # ----- plot
        self.plot_CKA(cka_dict, kernel=kernel, used_unit_type=used_unit_type, used_id_num=used_id_num, **kwargs)

        self.plot_CKA_temporal(cka_dict, kernel=kernel, used_unit_type=used_unit_type, used_id_num=used_id_num, **kwargs)

            
# ----------------------------------------------------------------------------------------------------------------------
class CKA_Similarity_Comparison(CKA_Similarity_base):
    """
        ...
    """
    
    def __init__(self, N1_root, N2_root, primate_config='Monkey', route='p', **kwargs):
        
        self.layers = layers
        self.primate_config = primate_config.replace('/', '_')
        
        self.model_structure_1 = N1_root.split('/')[-1].split(' ')[-1]
        self.model_structure_2 = N2_root.split('/')[-1].split(' ')[-1]
        
        self.save_root = os.path.join(root_dir, N1_root, f'CKA/Similarity v.s. {self.model_structure_2}')
        utils_.make_dir(self.save_root)
        
        cka_dict_1 = self.folds_model_merge(N1_root, primate_config=primate_config, route=route, **kwargs)
        cka_dict_2 = self.folds_model_merge(N2_root, primate_config=primate_config, route=route, **kwargs)
        
        # ---
        self.plot_CKA(cka_dict_1, cka_dict_2)
        
        self.plot_CKA_temporal(cka_dict_1, cka_dict_2, route=route)
        
        
    # FIXME --- this function should not only extract monkey values but all
    def folds_model_merge(self, NN, primate_config='Monkey', alpha=0.05, FDR_method='fdr_bh', num_folds=5, route='sig', **kwargs):
        """
            this function uses 2 routes to merge the FDR results of all folds. 
            
            Route 'p' uses the mean values of all p values then conduct the FDR test again, the output is boolean
            
            Route 'sig' uses the smoothed mean values of sig results(T/F), the output is float
            
            **Example primate_config:**
                primate_config = 'Monkey'
                primate_config = 'Human/linear/qualified/50'     # 'Human/{kernel}/{used_unit_type}/{used_id_num}'
            
            **Question: what cause inf or nan CKA/RSA values?
        """
        
        
        
        CKA_dict = {_: utils_.load(f"{NN}/-_Single Models/{NN.split('/')[-1]}{_}/Analysis/CKA/{primate_config}/CKA_results_linear.pkl", verbose=False) for _ in tqdm(range(num_folds), desc='CKA results')}
        CKA_dict = {_: np.array([CKA_dict[fold_idx][_] for fold_idx in range(num_folds)]) for _ in list(CKA_dict[0].keys())}
        
        folds_std = np.std(CKA_dict['cka_score'], axis=0)

        if route == 'p':
            
            CKA_dict = {k: np.mean(v, axis=0) for k,v in CKA_dict.items()}
            
            (CKA_dict['sig_Bonf'], CKA_dict['p_FDR'], alpha_Sadik, alpha_Bonf) = multipletests(CKA_dict['p'], alpha=alpha, method=FDR_method)    
            CKA_dict['sig_FDR'] = CKA_dict['p_FDR']<alpha_Bonf

            for _ in range(len(self.layers)):
                (CKA_dict['sig_temporal_FDR'][_, :], CKA_dict['p_temporal_FDR'][_, :], alpha_Sadik_temporal, alpha_Bonf_temporal) = multipletests(CKA_dict['p_temporal'][_, :], alpha=alpha, method=FDR_method)      # FDR
                CKA_dict['sig_temporal_Bonf'][_, :] = CKA_dict['p_temporal_FDR'][_, :]<alpha_Bonf_temporal     # Bonf correction
            
        elif route == 'sig':
            
            gaussian_keys = ['sig_FDR', 'sig_Bonf', 'sig_temporal_FDR', 'sig_temporal_Bonf']
            
            for k, v in CKA_dict.items():
                
                if k in gaussian_keys:
                    
                    CKA_dict[k] = np.mean(np.array([scipy.ndimage.gaussian_filter(v[fold_idx], sigma=1) for fold_idx in range(num_folds)]), axis=0)
                    
                else:
                    
                    CKA_dict[k] = np.mean(v, axis=0)
                    
        else:
            
            raise ValueError
        
        CKA_dict_across_folds = CKA_dict
        CKA_dict_across_folds['cka_score_std'] = folds_std
        
        return CKA_dict_across_folds
    
        
    def plot_CKA(self, cka_dict_1, cka_dict_2, kernel='linear', error_control_measure='sig_FDR', legend=False, vlim:list[float]=None, save=True, **kwargs):
        """
            this function plots static CKA scores and save
            
            input:
                layers: define the x axis
                cka_fr: CKA data
                title: 
                
        """
        
        utils_._print('Executing static plotting...')
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            if 'threshold' in kwargs:
                title = f"CKA score {kernel} {kwargs['threshold']} {self.model_structure_1} {self.model_structure_2}\n{self.primate_config}"
            elif kernel == 'linear':
                title = f'CKA score {kernel} {self.model_structure_1} {self.model_structure_2}\n{self.primate_config}'
            else:
                raise ValueError

            fig, ax = plt.subplots(figsize=(10,6))
            
            plot_CKA(self.layers, ax, cka_dict_1, legend=legend, color_set=['yellow', 'orange', 'chocolate'], error_area=False)
            
            plot_CKA(self.layers, ax, cka_dict_2, title='CKA score VGG16bn v.s. SpikingVGG16bn', vlim=vlim, legend=legend)
            
            fake_legend_stats_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='blue', markersize=5, markeredgecolor='orange'),
                                         Line2D([0], [0], marker='o', color='none', markerfacecolor='orange', markersize=5, markeredgecolor='orange'),]
            
            fake_legend_stats_labels = [
                f"SpikingVGG16bn_LIF_T4: {np.mean(cka_dict_2['cka_score']):.3f}(±{np.mean(cka_dict_2['cka_score_std']):.3f})",
                f"VGG16bn: {np.mean(cka_dict_1['cka_score']):.3f}(±{np.mean(cka_dict_1['cka_score_std']):.3f})",
                ]

            fake_legend = ax.legend(fake_legend_stats_handles, fake_legend_stats_labels, framealpha=0.25, ncol=1, handlelength=0, borderpad=0, labelspacing=0, loc='lower center')
            ax.add_artist(fake_legend)
            
            plt.tight_layout(pad=1)
            if save:
                plt.savefig(os.path.join(self.save_root, f'{title}.svg'), bbox_inches='tight')   
            plt.close()
        
        
    def plot_CKA_temporal(self, cka_dict_1, cka_dict_2, route='sig', extent:list[float]=None, error_control_measure='sig_temporal_Bonf', vlim:list[float]=None, kernel='linear', **kwargs):
        
        ...
        
        utils_._print('Executing temporal plotting')
        
        self.ts = np.arange(-50, 201, 10)
        
        if route == 'p':
        
            cka_dict_diff = {
                'cka_score': cka_dict_2['cka_score'] - cka_dict_1['cka_score'],
                'cka_score_temporal': cka_dict_2['cka_score_temporal'] - cka_dict_1['cka_score_temporal'],
                'sig_temporal_FDR': np.logical_and(cka_dict_2['sig_temporal_FDR'], cka_dict_1['sig_temporal_FDR']).astype(float),
                'sig_temporal_Bonf': np.logical_and(cka_dict_2['sig_temporal_Bonf'], cka_dict_1['sig_temporal_Bonf']).astype(float),
                }
        
        elif route == 'sig':
            
            cka_dict_diff = {
                'cka_score': cka_dict_2['cka_score'] - cka_dict_1['cka_score'],
                'cka_score_temporal': cka_dict_2['cka_score_temporal'] - cka_dict_1['cka_score_temporal'],
                'sig_temporal_FDR': np.mean([cka_dict_2['sig_temporal_FDR'], cka_dict_1['sig_temporal_FDR']], axis=0),
                'sig_temporal_Bonf': np.mean([cka_dict_2['sig_temporal_Bonf'], cka_dict_1['sig_temporal_Bonf']], axis=0),
                }
        
        extent = [self.ts.min()-5, self.ts.max()+5, -0.5, cka_dict_1['cka_score_temporal'].shape[0]-0.5]
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
    
        logging.getLogger('matplotlib').setLevel(logging.ERROR)    
    
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            if 'threshold' in kwargs:
                title = f"CKA score temporal {kernel} {kwargs['threshold']} {self.model_structure_1} {self.model_structure_2}\n{self.primate_config}"
            elif kernel == 'linear':
                title = f'CKA score temporal {kernel} {self.model_structure_1} {self.model_structure_2}\n{self.primate_config}'
            else:
                raise ValueError
            
            fig, ax = plt.subplots(figsize=(np.array(cka_dict_1['cka_score_temporal'].T.shape)/3.7))
            
            plot_CKA_temporal(self.layers, fig, ax, cka_dict_diff, title=title, vlim=vlim, extent=extent)
            utils_similarity.fake_legend_describe_numpy(cka_dict_diff['cka_score_temporal'], ax, cka_dict_diff[error_control_measure].astype(bool))

            plt.savefig(os.path.join(self.save_root, f'{title}.svg'))     

            plt.close()

  
# ----------------------------------------------------------------------------------------------------------------------
def plot_CKA(layers, ax, cka_dict, error_control_measure='sig_FDR', title=None, error_area=True, vlim:list[float]=None, legend=False, color_set=None, **kwargs):
    """
        #TODO 
        add the std error area
    """
    if color_set is None:
        color_set = ['skyblue', 'blue', 'deepskyblue']
        #color_set = ['yellow', 'orange', 'chocolate']

    plot_x = range(len(layers))
    
    # --- 1. plot shaded error bars
    if error_area:
        perm_mean = np.mean(cka_dict['cka_score_perm'], axis=1)  
        perm_std = np.std(cka_dict['cka_score_perm'], axis=1)  
        ax.fill_between(plot_x, perm_mean-perm_std, perm_mean+perm_std, color='lightgray', edgecolor='none', alpha=0.5)
        ax.fill_between(plot_x, perm_mean-2*perm_std, perm_mean+2*perm_std, color='lightgray', edgecolor='none', alpha=0.5)
        ax.fill_between(plot_x, perm_mean-3*perm_std, perm_mean+3*perm_std, color='lightgray', edgecolor='none', alpha=0.5, label='perm 1~3 std')
        ax.plot(plot_x, perm_mean, color='dimgray', label='perm mean')
    
    # --- 2. plot RSA scores with FDR results
    similarity = cka_dict['cka_score']
    
    if 'cka_score_std' in cka_dict.keys():
        ax.fill_between(plot_x, similarity-cka_dict['cka_score_std'], similarity+cka_dict['cka_score_std'], edgecolor=None, facecolor=color_set[0], alpha=0.75)

    for idx, _ in enumerate(cka_dict[error_control_measure], 0):
         if not _:   
             ax.scatter(idx, similarity[idx], facecolors='none', edgecolors=color_set[1])
         else:
             ax.scatter(idx, similarity[idx], facecolors=color_set[1], edgecolors=color_set[1])
             
    ax.plot(similarity, linestyle='dotted', color=color_set[2])

    ax.set_ylabel("CKA score")
    ax.set_xticks(plot_x)
    ax.set_xticklabels(layers, rotation=90, ha='center')
    ax.set_xlim([0, len(layers)-1])
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f'{title}')
    
    handles, labels = ax.get_legend_handles_labels()

    hollow_circle = Line2D([0], [0], marker='o', color=color_set[2], linestyle='dotted', markerfacecolor='none', markersize=5, markeredgecolor=color_set[1], linewidth=1)
    solid_circle = Line2D([0], [0], marker='o', color=color_set[2], linestyle='dotted', markerfacecolor=color_set[1], markersize=5, markeredgecolor=color_set[1], linewidth=1)

    handles.extend([hollow_circle, solid_circle])
    labels.extend([f"fialed {error_control_measure.split('_')[1]}", f"passed {error_control_measure.split('_')[1]}"])
    
    if legend:
        ax.legend(handles, labels, framealpha=0.5)
    
    similarity_ = similarity[~np.isnan(similarity)]
    if error_area:
        y_radius = np.max(similarity_[np.isfinite(similarity_)]) - np.min(perm_mean[~np.isnan(perm_mean)])
    else:
        y_radius = np.max(similarity_[np.isfinite(similarity_)]) - np.min(similarity_[np.isfinite(similarity_)])
    
    if not vlim:
        if error_area:
            ylim_bottom = np.min([np.min(similarity_[np.isfinite(similarity_)]), np.min(perm_mean[~np.isnan(perm_mean)])])
        else:
            ylim_bottom = np.min(similarity_[np.isfinite(similarity_)])
        ax.set_ylim([ylim_bottom-0.025*y_radius, np.max(similarity_[np.isfinite(similarity_)])+0.05*y_radius])
    else:
        ax.set_ylim(vlim)


def plot_CKA_temporal(layers, fig, ax, cka_dict, error_control_measure='sig_temporal_Bonf', title=None, vlim:list[float]=None, extent:list[float]=None):
      
    def _is_binary(input:np.ndarray):
        
        if input.dtype == int:
            return np.all((input==0)|(input==1))
        elif input.dtype == float:
            input = np.nan_to_num(input, 0.)
            return np.all((input==0.)|(input==1.))
        elif input.dtype == bool:
            return True
    
    # FIXME --- the contour() and contourf() are not identical
    def _mask_contour(input):
        
        from matplotlib.ticker import FixedLocator, FixedFormatter
        
        # ---
        c_ax1 = fig.add_axes([0.91, 0.125, 0.03, 0.35])
        c_b1 = fig.colorbar(x, cax=c_ax1)
        c_b1.ax.tick_params(labelsize=16)
        
        # ---
        ax.contour(input, levels:=[-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999], origin='upper', cmap='jet', extent=extent, linewidths=3)
        ax.contourf(input, levels, origin='upper', cmap='gray', extent=extent, alpha=0.3)
        
        dummy_y = ax.contourf(input, levels, cmap='jet')
        for collection in dummy_y.collections:
            collection.set_visible(False)

        c_ax2 = fig.add_axes([0.91, 0.525, 0.03, 0.35])
        c_b2 = fig.colorbar(dummy_y, cax=c_ax2)
        c_b2.ax.tick_params(labelsize=16)

        original_ticks = c_b2.get_ticks()
        original_labels = [str(tick) for tick in original_ticks]

        c_b2.ax.yaxis.set_major_locator(FixedLocator([-0.2, 0., 0.2, 0.4, 0.6, 0.8, 0.925, 0.975, 0.9945]))
        c_b2.ax.yaxis.set_major_formatter(FixedFormatter(original_labels))
        
    def _p_contour(input, alpha=0.05):

        input = scipy.ndimage.gaussian_filter(input.astype(float), sigma=1)
        input[input>(1-alpha)] = np.nan
        
        ax.imshow(input, aspect='auto',  cmap='gray', extent=extent, alpha=0.5)
        ax.contour(input, levels:=[0.5], origin='upper', cmap='jet', extent=extent, linewidths=3)
        
        c_b2 = fig.colorbar(x, cax=fig.add_axes([0.91, 0.125, 0.03, 0.75]))
        c_b2.ax.tick_params(labelsize=16)
        
    # ---
    if vlim:
        x = ax.imshow(cka_dict['cka_score_temporal'], aspect='auto', vmin=vlim[0], vmax=vlim[1], extent=extent)
    else:
        x = ax.imshow(cka_dict['cka_score_temporal'], aspect='auto', extent=extent)

    ax.set_yticks(np.arange(cka_dict['cka_score_temporal'].shape[0]), list(reversed(layers)), fontsize=12)
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_title(f'{title}', fontsize=16)
    
    # FIXEME --- need to upgrade to merged model --- significant correlation (Bonferroni/FDR)
    if error_control_measure == 'sig_temporal_FDR':
        if _is_binary(mask:=cka_dict[error_control_measure]):
            #ax.imshow(mask, aspect='auto',  cmap='gray', extent=extent, interpolation='none', alpha=0.25)
            _p_contour(mask)

        else:
            _mask_contour(mask)
            
        
    elif error_control_measure == 'sig_temporal_Bonf':
        if _is_binary(mask:=cka_dict[error_control_measure]):
            #ax.imshow(mask, aspect='auto',  cmap='gray', extent=extent, interpolation='none', alpha=0.25)
            _p_contour(mask)

        else:
            _mask_contour(mask)


# ----------------------------------------------------------------------------------------------------------------------
class CKA_base():
    
    def __init__(self, used_unit_types, **kwargs):
        
        self.used_unit_types = used_unit_types
        
        ...
        
        
    def __call__(self, **kwargs):
        
        # --- 1.
        cka_dict = self.calculation_CKA(**kwargs)
        
        # --- 2.
        for k, v in cka_dict.items():
            
            self.plot_CKA(v, k, **kwargs)
        
        
    def calculation_CKA(self, **kwargs):
        
        product_list = list(itertools.product(self.N1_layers, self.N2_layers))
        
        return {_type: np.array([cka(self.N1_G_dict[_[0]][_type], self.N2_G_dict[_[1]][_type]) for _ in product_list]).reshape(len(self.N1_layers), len(self.N2_layers)) for _type in tqdm(self.used_unit_types)}
        
    
    def plot_CKA(self, cka_results, _type, **kwargs):
         
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        fig,ax = plt.subplots()
        img = ax.imshow(cka_results, origin='lower', cmap='magma')
        
        ax.set_title(_type)
        ax.set_xlabel(f'{self.N2_structure}')
        ax.set_ylabel(f'{self.N1_structure}')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(img, shrink=0.75, pad=0.04)
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.N1_root, f'{_type}.svg'), bbox_inches='tight')
        
        plt.close()
            

# ----------------------------------------------------------------------------------------------------------------------
class CKA_Comparison(FSA_Gram, CKA_base):
    
    def __init__(self, N1_root, N1_model, N2_root, N2_model, used_unit_types=['qualified', 'strong_selective', 'weak_selective', 'non_selective'], num_folds=5, **kwargs):
        
        CKA_base.__init__(self, used_unit_types=used_unit_types, **kwargs)
        
        def _load_folds(nn_root, _model, _type='act', **kwargs):
            
            nn_grams = utils_.load(os.path.join(nn_root, 'Analysis/Gram/Gram_linear.pkl'))
            
            nn_layers, nn_neurons, _ = utils_.get_layers_and_units(_model, _type)
            
            nn_grams_dict = {layer: {_: nn_grams[layer][_] for _ in used_unit_types} for layer in nn_layers}
            
            nn_structure = nn_root.split('/')[-1].split(' ')[-1]
            
            return nn_layers, nn_neurons, nn_grams_dict, nn_structure
        
        self.N1_layers, _, self.N1_G_dict, self.N1_structure = _load_folds(N1_root, N1_model)
        self.N2_layers, _, self.N2_G_dict, self.N2_structure = _load_folds(N2_root, N2_model)
        
        self.N1_root = os.path.join(N1_root, f'Analysis/CKA/v.s. {self.N2_structure}')
        utils_.make_dir(self.N1_root)
        #self.N2_root = os.path.join(N2_root, 'Analysis')
        

# ----------------------------------------------------------------------------------------------------------------------
class CKA_Comparison_folds(FSA_Gram, CKA_base):
    
    def __init__(self, N1_root, N1_model, N2_root, N2_model, used_unit_types=['qualified', 'strong_selective', 'weak_selective', 'non_selective'], num_folds=5, **kwargs):
        
        CKA_base.__init__(self, used_unit_types=used_unit_types, **kwargs)
        
        def _load_folds(nn_root, _model, _type='act', **kwargs):
            
            nn_grams = utils_.load(os.path.join(nn_root, 'CKA/Grams_linear.pkl'))
            
            nn_layers, nn_neurons, _ = utils_.get_layers_and_units(_model, _type)
            
            nn_grams_dict = {layer: {_: np.mean([nn_grams[fold_idx][layer][_] for fold_idx in range(num_folds)], axis=0) for _ in self.used_unit_types} for layer in nn_layers}
            
            nn_structure = nn_root.split('/')[-1].split(' ')[-1]
            
            return nn_layers, nn_neurons, nn_grams_dict, nn_structure

        self.N1_layers, _, self.N1_G_dict, self.N1_structure = _load_folds(N1_root, N1_model)
        self.N2_layers, _, self.N2_G_dict, self.N2_structure = _load_folds(N2_root, N2_model)
        
        self.N1_root = os.path.join(N1_root, f'CKA/v.s. {self.N2_structure}')
        utils_.make_dir(self.N1_root)
        #self.N2_root = N2_root
        

# ----------------------------------------------------------------------------------------------------------------------
"""
    **CKA colab tutorial**, refer to: https://cka-similarity.github.io/
"""

def gram_linear(x, **kwargs):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0, **kwargs):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T, rtol=1e-06, atol=1e-05):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  
  #FIXME --- is this reasonable to set this cka score as 0?
  if normalization_x == 0 or normalization_y == 0:
      return np.float64(0)
  else:
      return scaled_hsic / (normalization_x * normalization_y)


# ======================================================================================================================
if __name__ == '__main__':
    
    layers, neurons, shapes = utils_.get_layers_and_units('vgg16_bn', target_layers='act')

    root_dir = '/home/acxyle-workstation/Downloads/'
    
    used_unit_types = ['qualified', 'selective', 'non_selective', 'strong_selective', 'weak_selective']
    
    # --- 1.
    #CKA_monkey = CKA_Similarity_Monkey(root=os.path.join(root_dir, 'Face Identity Baseline'), layers=layers, neurons=neurons)
    #CKA_monkey.process_CKA_monkey(kernel='linear', normalize=True)
    #for threshold in [0.5, 1.0, 2.0, 10.0]:
    #    CKA_monkey.process_CKA_monkey(kernel='rbf', threshold=threshold)
    
    # --- 2.
    for _ in range(5):
        CKA_human = CKA_Similarity_Human(root=os.path.join(root_dir, f'Face Identity VGG16bn_fold_/-_Single Models/Face Identity VGG16bn_fold_{_}'), layers=layers, neurons=neurons)
        for used_unit_type in used_unit_types:
            CKA_human.process_CKA_human(kernel='linear', used_unit_type=used_unit_type)
            #for threshold in [0.5, 1.0, 10.0]:
            #    CKA_human.process_CKA_human(kernel='rbf', threshold=threshold, used_unit_types=used_unit_types)
    
    # --- 3.
# =============================================================================
#     dual_models = CKA_Comparison(
#         N1_root=os.path.join(root_dir, 'Face Identity Baseline'), N1_model='vgg16',
#         N2_root=os.path.join(root_dir, 'Face Identity A2S_Baseline(T64)'), N2_model='spiking_vgg16',
#         used_unit_types=used_unit_types
#         )
#     dual_models()
# =============================================================================
    
    # --- 4.
# =============================================================================
#     ANN_vs_SNN = CKA_Comparison_folds(
#         N1_root=os.path.join(root_dir, 'Face Identity VGG16bn_fold_'), N1_model='vgg16_bn',
#         N2_root=os.path.join(root_dir, 'Face Identity SpikingVGG16bn_LIF_T4_CelebA2622_fold_'), N2_model='spiking_vgg16_bn',
#         used_unit_types=used_unit_types)
#     ANN_vs_SNN()
# =============================================================================
    
    # --- 5.
# =============================================================================
#     for _ in used_unit_types:
#         Model_compare = CKA_Similarity_Comparison(
#             N1_root=os.path.join(root_dir, 'Face Identity VGG16bn_fold_'), 
#             N2_root=os.path.join(root_dir, 'Face Identity SpikingVGG16bn_LIF_T4_CelebA2622_fold_'), 
#             primate_config=f'Human/linear/{_}/50', route='p'
#             )
# =============================================================================
