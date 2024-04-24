#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:47:12 2023

@author: acxyle
    
    ...
    
"""


import os
import warnings
import logging
import numpy as np
import itertools

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from joblib import Parallel, delayed
import scipy

from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind
from statsmodels.stats.multitest import multipletests

import utils_
from utils_ import utils_similarity

from Bio_Cell_Records_Process import Human_Neuron_Records_Process, Monkey_Neuron_Records_Process
from FSA_DRG import FSA_DSM


# ======================================================================================================================
class RSA_Base():
    """
        this class conntains the basic operation of RSA operation
        
            1. Euclidean - Pearson
            2. Euclidean - Spearman
            3. Euclidean - Concordance
            4. Pearson - Pearson
            5. Pearson - Spearman
            6. Pearson - Concordance
            7. Spearman - Pearson
            8. Spearman - Concordance
            9. Mahalanobis - Pearson
            10. Mahalanobis - Spearman
            11. Mahalanobis - Concordance
            12. Concordance - Pearson
            13. Concordance - Spearman
            14. Concordance - Concordance
    """
    
    def __init__(self, **kwargs):
        """
            those attributes should be defined by exterior class
        """
        self.ts
        
        self.dest_primate
        self.layers
        
        self.primate_DM
        self.primate_DM_perm
        
        # --- if use permutation every time, those two can be ignored, 5-fold experiment is suggested
        # --- empirically, the max fluctuation of mean scores between experiments could be ±0.03 with num_perm = 1000
        self.primate_DM_temporal
        self.primate_DM_temporal_perm
        
        
    def calculation_RSA(self, first_corr='pearson', second_corr='spearman', used_unit_type=None, used_id_num=None, FDR_test=True, alpha=0.05, num_perm=1000, FDR_method:str='fdr_bh', primate=None, save_path=None, **kwargs):
        """
            calculation and save the RSA results for monkey, the function will not save RSA_dict without FDR no matter 
            what the flag is
            
            **[notice]** this version removed all abnormal detections for simplification, refer to historical versions if encountered
            
            input:
                first_corr: RSA-I, select from 'euclidean', 'pearson', 'spearman', 'mahalanobis', 'concordance', refer to utisl_similarity.py
                second_corr: RSA-II, select from 'pearson', 'spearman', 'condordance'
                alpha: significant level for FDR based on permutation test
                num_perm: number of permutations
                FDR_method: preferred FDR method, refer to multipletests
                
            return:
                
                   
        """
        
        utils_.make_dir(dest_primate:=os.path.join(self.dest_RSA,  f'{primate}'))
        
        if primate == 'Monkey':
 
            self.dest_primate = os.path.join(dest_primate, f'{first_corr}')
            utils_.make_dir(self.dest_primate)
            
            save_path = os.path.join(self.dest_primate, f'RSA_results_{first_corr}_{second_corr}.pkl')
            
        elif primate == 'Human':
            
            assert used_unit_type != None and used_id_num != None
            
            utils_.make_dir(save_root_first_corr:=os.path.join(dest_primate, f'{first_corr}'))
            utils_.make_dir(save_root_second_corr:=os.path.join(save_root_first_corr, f'{second_corr}'))
            utils_.make_dir(save_root_cell_type:=os.path.join(save_root_second_corr, used_unit_type))   
            self.dest_primate = os.path.join(save_root_cell_type, str(used_id_num))
            utils_.make_dir(self.dest_primate)
            
            save_path = os.path.join(self.dest_primate, f'RSA_results_{first_corr}_{second_corr}_{used_unit_type}_{used_id_num}.pkl')
        
        if os.path.exists(save_path):
            
            RSA_dict = utils_.load(save_path)
            
        else:
            
            # -----
            pl = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(self.calculation_RSA_layer)(layer, first_corr=first_corr, second_corr=second_corr, FDR_test=FDR_test) for layer in tqdm(self.layers, desc='RSA'))
            
            # -----
            if not FDR_test:
                
                pl_k = ['corr_coef', 'corr_coef_temporal']
                
                assert list(pl[0].keys()) == pl_k
                
                extracted_data = [np.array([_[__] for _ in pl]) for __ in pl_k]

                RSA_dict = dict(zip(pl_k, extracted_data))
                
            else:
                
                pl_k = ['corr_coef', 'corr_coef_perm', 'p_perm', 'corr_coef_temporal', 'corr_coef_temporal_perm', 'p_perm_temporal']
            
                assert list(pl[0].keys()) == pl_k
                
                similarity, similarity_perm, similarity_p, similarity_temporal, similarity_temporal_perm, similarity_temporal_p = [np.array([_[__] for _ in pl]) for __ in pl_k]
                
                # --- static
                (sig_FDR, p_FDR, alpha_Sadik, alpha_Bonf) = multipletests(similarity_p, alpha=alpha, method=FDR_method)    # FDR (flase discovery rate) correction
                sig_Bonf = p_FDR<alpha_Bonf
                
                # --- temporal
                time_steps = self.primate_DM_temporal.shape[0]
                
                p_temporal_FDR = np.zeros((len(self.layers), time_steps))     # (num_layers, num_time_steps)
                sig_temporal_FDR =  np.zeros_like(p_temporal_FDR)
                sig_temporal_Bonf = np.zeros_like(p_temporal_FDR)
                
                for _ in range(len(self.layers)):
                    
                    (sig_temporal_FDR[_, :], p_temporal_FDR[_, :], alpha_Sadik_temporal, alpha_Bonf_temporal) = multipletests(similarity_temporal_p[_, :], alpha=alpha, method=FDR_method)      # FDR
                    sig_temporal_Bonf[_, :] = p_temporal_FDR[_, :]<alpha_Bonf_temporal     # Bonf correction
                
                # --- seal results
                RSA_dict = {
                    'similarity': similarity,
                    'similarity_perm': similarity_perm,
                    'similarity_p': similarity_p,
                    
                    'similarity_temporal': similarity_temporal,
                    'similarity_temporal_perm': similarity_temporal_perm,
                    'similarity_temporal_p': similarity_temporal_p,
                    
                    'p_FDR': p_FDR,
                    'sig_FDR': sig_FDR,
                    'sig_Bonf': sig_Bonf,
                    
                    'p_temporal_FDR': p_temporal_FDR,
                    'sig_temporal_FDR': sig_temporal_FDR,
                    'sig_temporal_Bonf': sig_temporal_Bonf,
                    }
                
                utils_.dump(RSA_dict, save_path)
        
        return RSA_dict
    
 
    def calculation_RSA_layer(self, layer, neuron_type='qualified', first_corr='pearson', second_corr='spearman', FDR_test=True, num_perm=1000, **kwargs):    
        """
            1. constant permutation results (default)
            2. random permutation each time, need to manually change the seed to None, otherwise it will be similiar with 
            the first one if the seed was kept
            
            Those two merely have tiny differences
        """
        
        # --- init, NN_DSM_v
        NN_DM = _vectorize_check(self.NN_DM_dict[layer])
        
        if np.isnan(NN_DM).all():
            
            NN_DM = np.full_like(self.primate_DM, np.nan)
        
        assert self.primate_DM.shape == NN_DM.shape
        
        # --- init, corr_func
        corr_func = _corr(second_corr)
        
        # ----- static
        corr_coef = corr_func(self.primate_DM, NN_DM)
        
        # ----- temporal
        time_steps = self.primate_DM_temporal.shape[0]
        
        pl = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(self.calculation_RSA_layer_temporal)(NN_DM, t, corr_func, FDR_test=FDR_test) for t in range(time_steps))
        
        if FDR_test:
            
            # --- static
            # --- type 1, constant permutation results, time saving but may biased
            corr_coef_perm = np.array([corr_func(self.primate_DM_perm[_, :], NN_DM) for _ in range(num_perm)])     # (1000,)
            
            # --- type 2, permute every time, time consuming
            #corr_coef_perm = np.array([corr_func(utils_similarity.DSM_calculation(first_corr, self.FR_id[np.random.permutation(self.FR_id.shape[0]),:], vectorize=True), NN_DM_v) for _ in range(num_perm)])
            
            p_perm = np.mean(corr_coef_perm > corr_coef)     # equal to: np.sum(corr_coef_perm > corr_coef)/num_perm
            
            # --- temporal
            corr_coef_temporal, corr_coef_temporal_perm, p_perm_temporal = map(np.array, zip(*pl))
            
            results = {
                'corr_coef': corr_coef,
                'corr_coef_perm': corr_coef_perm,
                'p_perm': p_perm,
                'corr_coef_temporal': corr_coef_temporal,
                'corr_coef_temporal_perm': corr_coef_temporal_perm,
                'p_perm_temporal': p_perm_temporal
                }
        
        else:
            
            corr_coef_temporal = np.array(pl)
        
            results = {
                'corr_coef': corr_coef,
                'corr_coef_temporal': corr_coef_temporal,
                }
        

        return results
    
    
    def calculation_RSA_layer_temporal(self, NN_DM, t, corr_func, FDR_test=True, num_perm=1000, **kwargs):
        
        assert corr_func is not None
            
        r = corr_func(self.primate_DM_temporal[t,:], NN_DM)
        
        if FDR_test:
            
            r_perm = np.array([corr_func(self.primate_DM_temporal_perm[t, _, :], NN_DM) for _ in range(num_perm)])      # (1000,)
            p = np.mean(r_perm > r)
        
            return r, r_perm, p
        
        else:
            
            return r
    
    
    def plot_RSA(self, fig, ax, RSA_dict, error_control_measure='sig_FDR', stats=True, vlim:list[float]=None, **kwargs):
        """
            this function plot static RSA score and save
            
            input:
                RSA_dict: RSA data
                title: 
                error_control_measure: default 'sig_FDR'. Options: 'sig_Bonf'
                norm_plot: default None. For exterior call with given ylim for fiar comparison
        """
        
        utils_.formatted_print('Plotting static...')
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        plot_RSA(fig, ax, RSA_dict, self.layers, vlim=vlim, **kwargs)
        
        if stats:
            utils_similarity.fake_legend_describe_numpy(RSA_dict['similarity'], ax, RSA_dict[error_control_measure].astype(bool), **kwargs)

            
    def plot_RSA_temporal(self, fig, ax, RSA_dict, error_control_measure='sig_temporal_Bonf', vlim:list[float]=None, stats=True, **kwargs):
        """
            function
            
            input:
                error_control_measure: 'sig_temporal_FDR' default. Options: 'sig_temporal_Bonf'
        """
        
        utils_.formatted_print('Plotting temporal...')
        
        extent = [self.ts.min()-5, self.ts.max()+5, -0.5, RSA_dict['similarity_temporal'].shape[0]-0.5]
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
    
        plot_RSA_temporal(fig, ax, RSA_dict, self.layers, vlim=vlim, extent=extent, **kwargs)
        
        if stats:
            utils_similarity.fake_legend_describe_numpy(RSA_dict['similarity_temporal'], ax, RSA_dict[error_control_measure].astype(bool), **kwargs)
            

# ----------------------------------------------------------------------------------------------------------------------
def merge_RSA_dict_folds(RSA_dict_folds, layers, num_folds, route='p', alpha=0.05, FDR_method='fdr_bh', **kwargs):
    
    # --- static
    similarity_folds = [RSA_dict_folds[fold_idx]['similarity'] for fold_idx in range(num_folds)]
    similarity_mean = np.mean(similarity_folds, axis=0)
    similarity_std = np.std(similarity_folds, axis=0)
    
    similarity_p_folds = np.mean([RSA_dict_folds[fold_idx]['similarity_p'] for fold_idx in range(num_folds)], axis=0)
    (sig_FDR, p_FDR, alpha_Sadik, alpha_Bonf) = multipletests(similarity_p_folds, alpha=alpha, method=FDR_method)    
    
    # --- temporal
    similarity_temporal_folds = np.array([RSA_dict_folds[fold_idx]['similarity_temporal'] for fold_idx in range(num_folds)])
    similarity_temporal_mean = np.mean(similarity_temporal_folds, axis=0)
    similarity_temporal_std = np.std(similarity_temporal_folds, axis=0)
    
    similarity_temporal_p = np.mean([RSA_dict_folds[fold_idx]['similarity_temporal_p'] for fold_idx in range(num_folds)], axis=0)  
    
    if route == 'p':
        
        # --- init
        p_temporal_FDR = np.zeros((len(layers), similarity_temporal_folds.shape[-1]))     # (num_layers, num_time_steps)
        sig_temporal_FDR =  p_temporal_FDR.copy()
        sig_temporal_Bonf = p_temporal_FDR.copy()
        
        for _ in range(len(layers)):
            (sig_temporal_FDR[_, :], p_temporal_FDR[_, :], alpha_Sadik_temporal, alpha_Bonf_temporal) = multipletests(similarity_temporal_p[_, :], alpha=alpha, method=FDR_method)      # FDR
            sig_temporal_Bonf[_, :] = p_temporal_FDR[_, :]<alpha_Bonf_temporal     # Bonf correction
    
    elif route == 'sig':
        
        sig_temporal_FDR = np.mean([scipy.ndimage.gaussian_filter(RSA_dict_folds[fold_idx]['sig_temporal_FDR'], sigma=1) for fold_idx in range(num_folds)], axis=0)
        sig_temporal_Bonf = np.mean([scipy.ndimage.gaussian_filter(RSA_dict_folds[fold_idx]['sig_temporal_Bonf'], sigma=1) for fold_idx in range(num_folds)], axis=0)
        p_temporal_FDR =  np.mean([RSA_dict_folds[fold_idx]['p_temporal_FDR'] for fold_idx in range(num_folds)], axis=0)
    
    # ---
    RSA_dict_folds = {
        'similarity': similarity_mean,
        'similarity_std': similarity_std,
        
        'similarity_perm': np.max([RSA_dict_folds[fold_idx]['similarity_perm'] for fold_idx in range(num_folds)], axis=0),
        'similarity_p': similarity_p_folds,
        
        'similarity_temporal': similarity_temporal_mean,
        'similarity_temporal_std': similarity_temporal_std,
        
        'similarity_temporal_perm': np.max([RSA_dict_folds[fold_idx]['similarity_temporal_perm'] for fold_idx in range(num_folds)], axis=0),
        'similarity_temporal_p': similarity_temporal_p,
        
        'p_FDR': p_FDR,
        'sig_FDR': sig_FDR,
        'sig_Bonf': p_FDR<alpha_Bonf,

        'p_temporal_FDR': p_temporal_FDR,
        'sig_temporal_FDR': sig_temporal_FDR,
        'sig_temporal_Bonf': sig_temporal_Bonf,

        }
    
    return RSA_dict_folds


# ----------------------------------------------------------------------------------------------------------------------
class RSA_Monkey(Monkey_Neuron_Records_Process, FSA_DSM, RSA_Base):
    """
        ...
    """
    
    def __init__(self, seed=6, **kwargs):
        
        # --- init
        Monkey_Neuron_Records_Process.__init__(self, seed=seed)
        FSA_DSM.__init__(self, **kwargs)
        
        self.dest_RSA = os.path.join(self.dest, 'RSA')
        utils_.make_dir(self.dest_RSA)
        utils_.make_dir(os.path.join(self.dest_RSA, 'Monkey'))
        
        
    def __call__(self, first_corr='pearson', second_corr='spearman', FDR_test=True, figsize=(10,6), **kwargs):
        """
            ...
            
            input:
                first_corr: default 'pearson', select from 'euclidean', 'pearson', 'spearman', 'mahalanobis', 'concordance'
                second_corr: default 'spearman', select from 'pearson', 'spearman', 'concordance'
                      
            permutation is only applied to primate data
            
        """
        
        # --- monkey init
        self.FR_id, self.psth_id = self.monkey_neuron_feature_process(**kwargs)
        
        monkey_DM_dict = self.monkey_neuron_DSM_process(first_corr, permutation=FDR_test, **kwargs)     # derived from Monkey_Neuron_Records_Process
        
        self.primate_DM = _vectorize_check(monkey_DM_dict['monkey_DM'])
        self.primate_DM_temporal = np.array([_vectorize_check(_) for _ in monkey_DM_dict['monkey_DM_temporal']])
        
        if FDR_test:
            
            assert set(['monkey_DM_perm', 'monkey_DM_temporal_perm']).issubset(monkey_DM_dict.keys())
            
            self.primate_DM_perm = np.array([_vectorize_check(_) for _ in monkey_DM_dict['monkey_DM_perm']])
            self.primate_DM_temporal_perm = np.array([np.array([_vectorize_check(__) for __ in _]) for _ in monkey_DM_dict['monkey_DM_temporal_perm']])
            
        # --- NN init
        self.NN_DM_dict = {k:v['qualified'] for k,v in self.calculation_DSM(first_corr, vectorize=False, **kwargs).items()}   # layer - cell_type
        ...
 
        # ----- 1. RSA calculation
        RSA_dict = self.calculation_RSA(first_corr, second_corr, primate='Monkey', **kwargs)
        
        # ----- 2. plot
        
        # --- 2.1 static
        fig, ax = plt.subplots(figsize=figsize)
        
        self.plot_RSA(fig, ax, RSA_dict, **kwargs)
        title = f'RSA Score {self.model_structure} {first_corr} {second_corr}'
        ax.set_title(f'{title}')

        fig.tight_layout(pad=1)
        fig.savefig(os.path.join(self.dest_primate, f'{title}.svg'), bbox_inches='tight')
        plt.close()
        
        # --- 2.2 temporal
        fig, ax = plt.subplots(figsize=(np.array(RSA_dict['similarity_temporal'].T.shape)/3.7))
        
        self.plot_RSA_temporal(fig, ax, RSA_dict, **kwargs)
        title=f'RSA Score temporal {self.model_structure} {first_corr} {second_corr}'
        ax.set_title(f'{title}')

        fig.savefig(os.path.join(self.dest_primate, f'{title}.svg'), bbox_inches='tight')
        plt.close()
        
        # --- 3. example correlation
        self.plot_correlation_example(RSA_dict['similarity'], first_corr=first_corr, second_corr=second_corr)
        

    #FIXME --- legacy
    def plot_correlation_example(self, similarity, first_corr='pearson', second_corr='spearman', neuron_type='qualified', attach_psth:bool=False):
        """
            this function plot with fig definition and ax addition
        """
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        # plot correlation for sample layer 
        layer = self.layers[np.argmax(similarity)]     # find the layer with strongest similarity score
        NN_DM_v = _vectorize_check(self.NN_DM_dict[layer])
        
        if not attach_psth:
            
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # plot sample PSTH - they namually set the target time point is 90, same results for below 2 methods
            #bestTimeFR = np.mean(self.meanPSTHID[:, np.where(self.psthTime == 90)[0][0], :], axis=0)
            bestTimeFR = np.mean(self.meanPSTHID[:, self.psthTime>60, :], axis=(0,1))
            
            most_active_cell = np.argmax(bestTimeFR)

            # plot corr example
            r, p, _ = self.plot_corr_2d(self.primate_DM, NN_DM_v, 'blue', ax, 'Spearman')
            ax.set_xlabel('Monkey Pairwise Distance')
            ax.set_ylabel('Network Pairwise Distance')
            ax.set_title(f'r:{r:.3f}, p:{p:.3e}')
        
        else:
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # plot sample PSTH - they namually set the target time point is 90, same results for below 2 methods
            bestTimeFR = np.mean(self.meanPSTHID[:, self.psthTime>60, :], axis=(0,1))
            
            most_active_cell = np.argmax(bestTimeFR)
            
            axes[0].imshow(self.meanPSTHID[:, :, most_active_cell], extent=[self.ts[0], self.ts[-1], 1, 50], aspect='auto')
            axes[0].set_xlabel('Time(ms)')
            axes[0].set_ylabel('ID')
            axes[0].set_title(f'Monkey IT Neuron {most_active_cell}')
            axes[0].tick_params(labelsize=12)
            
            # plot corr example
            r, p, _ = self.plot_corr_2d(self.primate_DM, NN_DM_v, 'b', axes[1], 'Spearman')
            axes[1].set_xlabel('Monkey Pairwise Distance')
            axes[1].set_ylabel('Network Pairwise Distance')
            axes[1].set_title(f'r:{r:.3f}, p:{p:.3e}')
        
        title = f'Monkey - {self.model_structure} {layer} polyfit {first_corr} {second_corr}'
        fig.suptitle(f'{title}', fontsize=20, y=1.)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dest_primate, f'{title}.svg'))

        plt.close()
                      
    
    #FIXME --- legacy
    def plot_corr_2d(self, A, B, color='blue', ax=None, criteria='Pearson'):
        
        if criteria == 'Pearson':  # no tested
            corr_func = pearsonr
        elif criteria == 'Spearman':
            corr_func = spearmanr
        elif criteria == 'Kendalltau':  # no tested
            corr_func = kendalltau
        else:
            raise ValueError('[Coderror] Unknown correlation type')
    
        ind = np.where(~np.isnan(A) & ~np.isnan(B))[0]
    
        if ind.size == 0:
            raise ValueError('[Coderror] All NaN values')
    
        r, p = corr_func(A[ind], B[ind])
    
        title = f'r={r:.5f} p={p:.3e}'
    
        if ax is not None and isinstance(ax, matplotlib.axes.Axes):
            
            ax.plot(A[ind], B[ind], color=color, linestyle='none', marker='.', linewidth=2, markersize=2)
            (k_, p_) = np.polyfit(A[ind], B[ind], 1)     # polynomial fitting, degree=1

            x_ = np.array([np.min(A), np.max(A)])
            y_ = x_*k_ + p_
            
            ax.plot(x_, y_,color='red', linewidth=2)
            ax.axis('tight')
    
        return r, p, title                                  


# ----------------------------------------------------------------------------------------------------------------------
class RSA_Monkey_folds(RSA_Monkey):
    """
        each RSA_dict contains FDR test by default
    """
    
    def __init__(self, num_folds=5, root=None, **kwargs):
        
        super().__init__(root=root, **kwargs)
        
        self.root = root
        self.num_folds = num_folds
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})

        ...
        
    
    def __call__(self, first_corr='pearson', second_corr='spearman', **kwargs):
        
        RSA_dict_folds = self.calculation_RSA_folds(first_corr=first_corr, second_corr=second_corr, **kwargs)
        
        self.plot_RSA_folds(RSA_dict_folds, first_corr=first_corr, second_corr=second_corr, **kwargs)
        
        ...
        
        
    def calculation_RSA_folds(self, first_corr, second_corr, FDR_method='fdr_bh', alpha=0.05, route='sig', **kwargs):
        
        self.dest_primate = os.path.join(self.dest_RSA, f'Monkey/{first_corr}')
        utils_.make_dir(self.dest_primate)
        
        RSA_dict_folds_path = os.path.join(self.dest_primate, f'RSA_results_{first_corr}_{second_corr}_{route}.pkl')
        
        if os.path.exists(RSA_dict_folds_path):
            
            RSA_dict_folds = utils_.load(RSA_dict_folds_path, verbose=False)
        
        else:
            
            RSA_dict_folds = {_ :utils_.load(os.path.join(self.root, f"-_Single Models/{self.root.split('/')[-1]}{_}/Analysis/RSA/Monkey/{first_corr}/RSA_results_{first_corr}_{second_corr}.pkl"), verbose=False) for _ in range(self.num_folds)}
            
            # ---
            RSA_dict_folds = merge_RSA_dict_folds(RSA_dict_folds, self.layers, self.num_folds, route, **kwargs)
            
            # ---
            utils_.dump(RSA_dict_folds, RSA_dict_folds_path)
            
        return RSA_dict_folds
    
    # -----
    def plot_RSA_folds(self, RSA_dict_folds, first_corr, second_corr, route='p', **kwargs):
        
        # --- static
        fig, ax = plt.subplots(figsize=(10,6))
        
        self.plot_RSA(fig, ax, RSA_dict_folds, **kwargs)
        
        title=f'RSA Score {self.model_structure} {first_corr} {second_corr} {route}'
        ax.set_title(f'{title}')

        fig.tight_layout(pad=1)
        fig.savefig(os.path.join(self.dest_primate, f'{title}.svg'), bbox_inches='tight')

        plt.close()
        
        # --- temporal
        fig, ax = plt.subplots(figsize=(np.array(RSA_dict_folds['similarity_temporal'].T.shape)/3.7))
        
        self.plot_RSA_temporal(fig, ax, RSA_dict_folds, **kwargs)
        title=f'RSA Score temporal {self.model_structure} {first_corr} {second_corr} {route}'
        ax.set_title(f'{title}', fontsize=16)
        
        fig.savefig(os.path.join(self.dest_primate, f'{title}.svg'))     
        plt.close()
        
        ...

# ----------------------------------------------------------------------------------------------------------------------
class RSA_Human(Human_Neuron_Records_Process, FSA_DSM, RSA_Base):
    """
        ...
    """
    
    def __init__(self, seed=6, **kwargs):
        
        Human_Neuron_Records_Process.__init__(self, seed=seed)
        FSA_DSM.__init__(self, **kwargs)
        
        self.dest_RSA = os.path.join(self.dest, 'RSA')
        utils_.make_dir(self.dest_RSA)
        
        self.save_root_primate = os.path.join(self.dest_RSA, 'Human')
        utils_.make_dir(self.save_root_primate)
        
        
    def __call__(self, first_corr='pearson', second_corr='spearman', used_unit_type='qualified', used_id_num=50, FDR_test=True, **kwargs):
        """
            Each process consists 3 sections:
                1) generate biological neuron responses according to neuron types - [self.human_neuron_spike_process()]
                2) generate feature maps of artificial units and calculate the similarity - [self.plot_RSA()]
                3) plot according to previous outcomes - [self.human_neuron_RSA_emporal_plot()]
            
            Input:
                cell_type: default 'qualified', select from 'legacy', 'qualified', 'slective', 'strong_selective', 
                'weak_selective', 'non_selective', etc. refer to self.human_neuron_obtain_used_cells()
                used_id_num: default '50', select from '50', '10'
    
        """
        # --- additional parameters
        utils_.formatted_print(f'Processing RSA for Human and NN | {self.model_structure} | {used_unit_type} | {used_id_num} | {first_corr} | {second_corr}')
        
        # --- init
        self.used_id = self.human_corr_select_sub_identities(used_id_num)
        
        NN_DM_dict = self.calculation_DSM(first_corr)
        
        if used_unit_type == 'legacy':
            human_DM_dict = self.human_neuron_DSM_process(first_corr, 'selective', **kwargs)
            self.NN_DM_dict = {_: _vectorize_check(NN_DM_dict[_]['strong_selective'][np.ix_(self.used_id, self.used_id)]) for _ in NN_DM_dict.keys()}
        else:
            human_DM_dict = self.human_neuron_DSM_process(first_corr, used_unit_type, **kwargs)
            self.NN_DM_dict = {_: _vectorize_check(NN_DM_dict[_][used_unit_type][np.ix_(self.used_id, self.used_id)]) if ~np.isnan(NN_DM_dict[_][used_unit_type]).all() else np.nan for _ in NN_DM_dict.keys()}
            
        self.primate_DM = _vectorize_check(human_DM_dict['human_DM'][np.ix_(self.used_id, self.used_id)])
        self.primate_DM_temporal = np.array([_vectorize_check(_[np.ix_(self.used_id, self.used_id)]) for _ in human_DM_dict['human_DM_temporal']])
        
        if FDR_test:
            
            assert set(['human_DM_perm', 'human_DM_temporal_perm']).issubset(human_DM_dict.keys())
            
            self.primate_DM_perm = np.array([_vectorize_check(_[np.ix_(self.used_id, self.used_id)]) for _ in human_DM_dict['human_DM_perm']])
            self.primate_DM_temporal_perm = np.array([np.array([_vectorize_check(__[np.ix_(self.used_id, self.used_id)]) for __ in _]) for _ in human_DM_dict['human_DM_temporal_perm']])
        
        # --- RSA calculation
        RSA_dict = self.calculation_RSA(first_corr=first_corr, second_corr=second_corr, used_unit_type=used_unit_type, used_id_num=used_id_num, primate='Human', **kwargs)
        
        # --- plot
        # --- 2.1 static
        fig, ax = plt.subplots(figsize=(10, 6))
        
        self.plot_RSA(fig, ax, RSA_dict, **kwargs)
        title=f'RSA Score {self.model_structure} {first_corr} {second_corr} {used_unit_type} {used_id_num}'
        ax.set_title(f'{title}')

        fig.tight_layout(pad=1)
        fig.savefig(os.path.join(self.dest_primate, f'{title}.svg'), bbox_inches='tight')
        plt.close()
        
        # --- 2.2 temporal
        fig, ax = plt.subplots(figsize=(np.array(RSA_dict['similarity_temporal'].T.shape)/3.7))
        
        self.plot_RSA_temporal(fig, ax, RSA_dict, **kwargs)
        title=f'RSA Score temporal {self.model_structure} {first_corr} {second_corr} {used_unit_type} {used_id_num}'
        ax.set_title(f'{title}', fontsize=16)
        
        fig.tight_layout(pad=1)
        fig.savefig(os.path.join(self.dest_primate, f'{title}.svg'))     
        plt.close()
        

# ----------------------------------------------------------------------------------------------------------------------
class RSA_Human_folds(RSA_Human):
    
    def __init__(self, num_folds=5, root=None, **kwargs):
        
        super().__init__(root=root, **kwargs)
        
        self.root= root
        self.num_folds = num_folds
        
        self.save_root_primate = os.path.join(self.dest_RSA, 'Human')
        utils_.make_dir(self.save_root_primate)
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        ...
        
        
    def __call__(self, first_corr='pearson', second_corr='spearman', used_unit_type:str=None, used_id_num:int=None, **kwargs):
        
        RSA_dict_folds = self.calculation_RSA_folds(first_corr, second_corr, used_unit_type=used_unit_type, used_id_num=used_id_num, **kwargs)
        
        self.plot_RSA_folds(RSA_dict_folds, first_corr, second_corr, used_unit_type=used_unit_type, used_id_num=used_id_num, **kwargs)
        
        ...
        
        
    def calculation_RSA_folds(self, first_corr, second_corr, used_unit_type:str=None, used_id_num:int=None, FDR_method='fdr_bh', alpha=0.05, route='sig', **kwargs):
        
        # --- additional parameters
        utils_.formatted_print(f'Processing RSA for Human and NN | {self.model_structure} | {used_unit_type} | {used_id_num} | {first_corr} | {second_corr}')
        
        # --- folder init
        utils_.make_dir(save_root_first_corr:=os.path.join(self.save_root_primate, f'{first_corr}'))
        utils_.make_dir(save_root_second_corr:=os.path.join(save_root_first_corr, f'{second_corr}'))
        utils_.make_dir(save_root_cell_type:=os.path.join(save_root_second_corr, used_unit_type))
            
        self.dest_primate = os.path.join(save_root_cell_type, str(used_id_num))
        utils_.make_dir(self.dest_primate)
        
        # ---
        save_path = os.path.join(self.dest_primate, f'RSA_results_{first_corr}_{second_corr}_{used_unit_type}_{used_id_num}_{route}.pkl')
        
        if os.path.exists(save_path):
            
            RSA_dict_folds = utils_.load(save_path, verbose=False)
        
        else:
            
            RSA_dict_folds = {}
            
            for fold_idx in range(self.num_folds):
            
                RSA_dict_folds[fold_idx] = utils_.load(os.path.join(self.root, 
                f"-_Single Models/{self.root.split('/')[-1]}{fold_idx}/Analysis/RSA/Human/{first_corr}/{second_corr}/{used_unit_type}/{used_id_num}/RSA_results_{first_corr}_{second_corr}_{used_unit_type}_{used_id_num}.pkl"), verbose=False)
            
            RSA_dict_folds = merge_RSA_dict_folds(RSA_dict_folds, self.layers, self.num_folds, route, **kwargs)
            
            # ---
            utils_.dump(RSA_dict_folds, save_path)
            
        return RSA_dict_folds

        
    def plot_RSA_folds(self, RSA_dict_folds, first_corr, second_corr, used_unit_type, used_id_num, route, **kwargs):
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        self.plot_RSA(fig, ax, RSA_dict_folds, **kwargs)
        title=f'RSA Score {self.model_structure} {first_corr} {second_corr} {used_unit_type} {used_id_num} {route}'
        ax.set_title(f'{title}')

        fig.tight_layout(pad=1)
        fig.savefig(os.path.join(self.dest_primate, f'{title}.svg'), bbox_inches='tight')
        plt.close()
        
        # --- 2.2 temporal
        fig, ax = plt.subplots(figsize=(np.array(RSA_dict_folds['similarity_temporal'].T.shape)/3.7))
        
        self.plot_RSA_temporal(fig, ax, RSA_dict_folds, **kwargs)
        title=f'RSA Score temporal {self.model_structure} {first_corr} {second_corr} {used_unit_type} {used_id_num} {route}'
        ax.set_title(f'{title}', fontsize=16)
        
        fig.savefig(os.path.join(self.dest_primate, f'{title}.svg'))     
        plt.close()
        

# ----------------------------------------------------------------------------------------------------------------------
def plot_RSA(fig, ax, RSA_dict, layers, error_control_measure='sig_FDR', error_area=True, vlim:list[float]=None, legend=False, color=None, label=None, **kwargs):
    """
        ...
    """
    
    if color is None:
        color = 'blue'

    plot_x = range(len(layers))
    
    # --- 1. plot shaded error bars
    if error_area:
        perm_mean = np.mean(RSA_dict['similarity_perm'], axis=1)  
        perm_std = np.std(RSA_dict['similarity_perm'], axis=1)  
        ax.fill_between(plot_x, perm_mean-perm_std, perm_mean+perm_std, color='lightgray', edgecolor='none', alpha=0.5)
        ax.fill_between(plot_x, perm_mean-2*perm_std, perm_mean+2*perm_std, color='lightgray', edgecolor='none', alpha=0.5)
        ax.fill_between(plot_x, perm_mean-3*perm_std, perm_mean+3*perm_std, color='lightgray', edgecolor='none', alpha=0.5)
        ax.plot(plot_x, perm_mean, color='dimgray')
    
    # --- 2. plot RSA scores with FDR results
    similarity = RSA_dict['similarity']
    
    if 'similarity_std' in RSA_dict.keys():
        ax.fill_between(plot_x, similarity-RSA_dict['similarity_std'], similarity+RSA_dict['similarity_std'], edgecolor=None, facecolor=utils_.lighten_color(utils_.color_to_hex(color), 100), alpha=0.75)

    for idx, _ in enumerate(RSA_dict[error_control_measure], 0):
         if not _:   
             ax.scatter(idx, similarity[idx], facecolors='none', edgecolors=color)
         else:
             ax.scatter(idx, similarity[idx], facecolors=color, edgecolors=color)
             
    ax.plot(similarity, color=utils_.darken_color(utils_.color_to_hex(color)), linestyle='dotted', linewidth=2)

    ax.set_ylabel("Spearman's $\\rho$")
    ax.set_xticks(plot_x)
    ax.set_xticklabels(layers, rotation=90, ha='center')
    ax.set_xlim([0, len(layers)-1])
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    
    handles, labels = ax.get_legend_handles_labels()

    hollow_circle = Line2D([0], [0], marker='o', color=utils_.darken_color(utils_.color_to_hex(color)), markerfacecolor='none', markersize=5, markeredgecolor=color, linestyle='dotted', linewidth=2)
    solid_circle = Line2D([0], [0], marker='o', color=utils_.darken_color(utils_.color_to_hex(color)), markerfacecolor=color, markersize=5, markeredgecolor=color, linestyle='dotted', linewidth=2)

    handles.extend([hollow_circle, solid_circle])
    labels.extend([f"fialed {error_control_measure.split('_')[1]}", f"passed {error_control_measure.split('_')[1]}"])
    
    if legend:
        ax.legend(handles, labels, framealpha=0.5)
    
    similarity_ = similarity[~np.isnan(similarity)]
    if error_area:
        y_radius = np.max(similarity_) - np.min(perm_mean[~np.isnan(perm_mean)])
    else:
        y_radius = np.max(similarity_) - np.min(similarity_)
    
    if not vlim:
        if error_area:
            ylim_bottom = np.min([np.min(similarity_), np.min(perm_mean[~np.isnan(perm_mean)])])
        else:
            ylim_bottom = np.min(similarity_)
        ax.set_ylim([ylim_bottom-0.025*y_radius, np.max(similarity_)+0.05*y_radius])
    else:
        ax.set_ylim(vlim)


def plot_RSA_temporal(fig, ax, RSA_dict, layers, error_control_measure='sig_temporal_Bonf', vlim:list[float]=None, extent:list[float]=None, **kwargs):
    
    def _is_binary(input:np.ndarray):
        input = np.nan_to_num(input, 0)     # assume nan comes from invalid data
        return np.all((input==0)|(input==1))
    
    # the contour() and contourf() are not identical
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
        """
            set the ROI as nan because plt display nothing for nan, so no mask for ROI
        """
        input = scipy.ndimage.gaussian_filter(input, sigma=1)
        input[input>(1-alpha)] = np.nan
        
        ax.imshow(input, aspect='auto',  cmap='gray', extent=extent, alpha=0.5)
        ax.contour(input, levels=[0.25], origin='upper', cmap='jet', extent=extent, linewidths=3)
        
        c_b2 = fig.colorbar(x, cax=fig.add_axes([0.91, 0.125, 0.03, 0.75]))
        c_b2.ax.tick_params(labelsize=16)
        
    # ---
    if vlim:
        x = ax.imshow(RSA_dict['similarity_temporal'], aspect='auto', vmin=vlim[0], vmax=vlim[1], extent=extent)
    else:
        x = ax.imshow(RSA_dict['similarity_temporal'], aspect='auto', extent=extent)

    ax.set_yticks(np.arange(RSA_dict['similarity_temporal'].shape[0]), list(reversed(layers)), fontsize=12)
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)

    # significant correlation (Bonferroni/FDR)
    if error_control_measure == 'sig_temporal_FDR':
        if _is_binary(mask:=RSA_dict[error_control_measure]):
            _p_contour(mask)

        else:
            _mask_contour(mask)
            
        
    elif error_control_measure == 'sig_temporal_Bonf':
        if _is_binary(mask:=RSA_dict[error_control_measure]):
            _p_contour(mask)

        else:
            _mask_contour(mask)
            

# ----------------------------------------------------------------------------------------------------------------------
#FIXME --- seems need to add below abnormal detection? because ns cells/units always generate weird output 
def _vectorize_check(input:np.ndarray):
    
    if np.isnan(input).all() or input.ndim == 1:
        pass
    elif input.ndim == 2:
        input = utils_similarity.RSM_vectorize(input)     # (50,50) -> (1225,)
    else:
        raise ValueError('invalid input')
        
    return input


def _corr(second_corr):
    
    corr_func_map = {
        'spearman': _spearmanr,
        'pearson': _pearson,
        'concordance': _ccc
        }
    
    if second_corr in corr_func_map:
        return corr_func_map[second_corr]
    else:
        raise ValueError('[Coderror] invalid second_corr')


def _spearmanr(x, y):
    """
        x: primate
        y: NN
    """
    
    if np.unique(y).size < 2 or np.any(np.isnan(y)):
        return np.nan
    else:
        return spearmanr(x, y, nan_policy='omit').statistic


def _pearson(x, y):
    return np.corrcoef(x, y)[0, 1]


def _ccc(x, y):
    return utils_similarity._ccc(x, y)


# ----------------------------------------------------------------------------------------------------------------------
class RSA_Monkey_Comparison(RSA_Monkey_folds):
    """
        not a script, manually change the code
    """
    
    def __init__(self, **kwargs):
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        # ---
        self.color_pool = ['blue', 'green', 'red', 'purple', 'orange', 'chocolate']     # manually change the pool
        
        ...
        
        
    def __call__(self, roots_and_models, first_corr, second_corr, route, **kwargs):
        
        # ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        title = []
        handles, labels = ax.get_legend_handles_labels()
        
        for idx, (root, model) in enumerate(roots_and_models):
            
            
            color = self.color_pool[idx]
            
            if 'fold' in root:
                
                super().__init__(root=roots_and_models[idx][0], **kwargs)     # save the comparison results in the fisrt folder
                self.layers, self.neurons, _ = utils_.get_layers_and_units(roots_and_models[idx][1], target_layers='act')
                
                RSA_dict = self.calculation_RSA_folds(first_corr=first_corr, second_corr=second_corr, route=route, **kwargs)

                _label = root.split('/')[-1].split(' ')[-1].replace('_fold_', '').replace('_CelebA2622', '')
                title.append(_label)
                
                self.plot_RSA(fig, ax, RSA_dict, stats=False, color=color)
                
                ...
                
            else:
                
                super().__init__(root=roots_and_models[idx][0], **kwargs)     # save the comparison results in the fisrt folder
                self.layers, self.neurons, _ = utils_.get_layers_and_units(roots_and_models[idx][1], target_layers='act')
                
                RSA_dict = self.calculation_RSA(first_corr=first_corr, second_corr=second_corr, primate='Monkey', route=route, **kwargs)
                
                _label=root.split('/')[-1].split(' ')[-1]
                title.append(_label)
                
                self.plot_RSA(fig, ax, RSA_dict, stats=False, color=color)
                ...
            
            solid_circle = Line2D([0], [0], marker='o', color=utils_.darken_color(utils_.color_to_hex(color)), markerfacecolor=color, markersize=5, markeredgecolor=color, linestyle='dotted', linewidth=2)

            handles.extend([solid_circle])
            labels.extend([f"{_label}"])

        ax.set_title(title:=f'{first_corr} {second_corr} {route} RSA score '+' v.s '.join(title))
        #ax.set_title(title:=f'{first_corr} {second_corr} {route} RSA core ANN v.s SNN')
        
        # --- setting
        ax.legend(handles, labels, framealpha=0.5)
        # ---
        
        fig.tight_layout()
        fig.savefig(os.path.join(roots_and_models[0][0], f'Analysis/RSA/Monkey/Comparison {title}.svg'))
        
        plt.close()


# ----------------------------------------------------------------------------------------------------------------------
class RSA_Human_Comparison(RSA_Human_folds):
    
    def __init__(self, **kwargs):
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        # ---
        self.color_pool = ['blue', 'green', 'red', 'purple', 'orange', 'chocolate']     # manually change the pool
        
        ...
        
        
    def __call__(self, roots_and_models, first_corr, second_corr, used_unit_type='qualified', used_id_num=50, route='p', **kwargs):
        
        # ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        title = []
        handles, labels = ax.get_legend_handles_labels()
        
        for idx, (root, model) in enumerate(roots_and_models):
            
            
            color = self.color_pool[idx]
            
            if 'fold' in root:
                
                super().__init__(root=roots_and_models[idx][0], **kwargs)     # save the comparison results in the fisrt folder
                self.layers, self.neurons, _ = utils_.get_layers_and_units(roots_and_models[idx][1], target_layers='act')
                
                RSA_dict = self.calculation_RSA_folds(first_corr=first_corr, second_corr=second_corr, used_unit_type=used_unit_type, used_id_num=used_id_num, route=route, **kwargs)

                _label = root.split('/')[-1].split(' ')[-1].replace('_fold_', '').replace('_CelebA2622', '')
                title.append(_label)
                
                self.plot_RSA(fig, ax, RSA_dict, stats=False, color=color)
                
                ...
                
            else:
                
                super().__init__(root=roots_and_models[idx][0], **kwargs)     # save the comparison results in the fisrt folder
                self.layers, self.neurons, _ = utils_.get_layers_and_units(roots_and_models[idx][1], target_layers='act')
                
                RSA_dict = self.calculation_RSA(first_corr=first_corr, second_corr=second_corr, used_unit_type=used_unit_type, used_id_num=used_id_num, primate='Human', route=route, **kwargs)
                
                _label=root.split('/')[-1].split(' ')[-1]
                title.append(_label)
                
                self.plot_RSA(fig, ax, RSA_dict, stats=False, color=color)
                ...
            
            solid_circle = Line2D([0], [0], marker='o', color=utils_.darken_color(utils_.color_to_hex(color)), markerfacecolor=color, markersize=5, markeredgecolor=color, linestyle='dotted', linewidth=2)

            handles.extend([solid_circle])
            labels.extend([f"{_label}"])

        ax.set_title(title:=f'{first_corr} {second_corr} {used_unit_type} {used_id_num} {route} RSA score '+' v.s '.join(title))
        #ax.set_title(title:=f'{first_corr} {second_corr} {route} RSA core ANN v.s SNN')
        
        # --- setting
        ax.legend(handles, labels, framealpha=0.5)
        # ---
        
        fig.tight_layout()
        fig.savefig(os.path.join(roots_and_models[0][0], f'Analysis/RSA/Human/Comparison {title}.svg'))
        
        plt.close()
    

# ----------------------------------------------------------------------------------------------------------------------
#FIXME 5-folds model training required
#FIXME the design needs to upgrade for better use rather local debug
#FIXME --- building...
class similarity_scores_comparison_base():
    """
        similarity mean value comparison with ttest
    """
    
    def __init__(self, root, primate, first_corr='pearson', second_corr='spearman', cell_type='qualified', used_id_num=50, **kwargs):
        
        utils_.make_dir(similarity_save_root:=os.path.join(root, 'Face Identity - similarity'))

        self.similarity_save_root = os.path.join(similarity_save_root, f'{primate}')
        utils_.make_dir(self.similarity_save_root)
        
        self.root = root
        self.primate = primate

        self.first_corr = first_corr
        self.second_corr = second_corr
        
        self.cell_type = cell_type
        self.used_id_num = used_id_num
        

    def statistical_calculation(self, rsa_dicts, base_model=None, **kwargs):
        
        # ----- FDR correction
        scores = {k:v['similarity'][v['sig_FDR']].ravel() if v is not np.nan else np.nan for k,v in rsa_dicts.items()}
        
        scores_temporal = {k:v['similarity_temporal'][v['sig_temporal_Bonf'].astype(bool)].ravel() if v is not np.nan else np.nan for k,v in rsa_dicts.items()}
        
        models = list(scores.keys())
        
        if base_model == None:
            base_model = models[0]
        
        models.remove(base_model)
        
        groups = list(itertools.product([base_model], models))
        
        # --- static
        ttest_results = []
        for _ in groups:
            ttest_results.append(ttest_ind(scores[_[0]], scores[_[1]])[1])
        
        # --- temporal
        ttest_temporal_results = []
        for _ in groups:
            ttest_temporal_results.append(ttest_ind(scores_temporal[_[0]], scores_temporal[_[1]])[1])
        
        # -----
        return groups, (scores, ttest_results), (scores_temporal, ttest_temporal_results)
     
    def plot(self, ):
        
        print('6')
        
        
# ----------------------------------------------------------------------------------------------------------------------
class Monkey_similarity_scores_comparison(similarity_scores_comparison_base):
    
    def __init__(self, roots_and_models, **kwargs):
        
        super().__init__(root=root_dir, **kwargs)
        
        self.similarity_save_root_comparison = os.path.join(self.similarity_save_root, f'{self.first_corr}')
        utils_.make_dir(self.similarity_save_root_comparison)
        
        rsa_dicts = {}
        
        for idx, (root, model) in enumerate(roots_and_models):
            
            if 'fold' in root:
                
                file_path = os.path.join(root, f'Analysis/RSA/{self.primate}/{self.first_corr}/RSA_results_{self.first_corr}_{self.second_corr}_p.pkl')
        
            else:
        
                file_path = os.path.join(root, f'Analysis/RSA/{self.primate}/{self.first_corr}/RSA_results_{self.first_corr}_{self.second_corr}.pkl')
        
            rsa_dicts.update({root.split('/')[-1].split(' ')[-1].replace('_fold_', '').replace('SpikingVGG16bn', 'SVGG16bn').replace('CelebA2622', 'C'): utils_.load(file_path)})
        
        # ---
        groups, (scores, ttest_results), (scores_temporal, ttest_temporal_results) = self.statistical_calculation(rsa_dicts)
        
        rsa_models = list(rsa_dicts.keys())
        
        groups = [(rsa_models.index(m1), rsa_models.index(m2)) for m1, m2 in groups]
        
        # --- static
        fig, ax = plt.subplots(figsize=(10,6))
        
        self.plot(fig, ax, rsa_models, scores, groups, ttest_results, title='Static')
        
        plt.tight_layout()
        #fig.savefig(os.path.join(self.similarity_save_root_comparison, f'{title}.svg'))     
        plt.show()
        #plt.close()
        
        # --- temporal
        fig, ax = plt.subplots(figsize=(10,6))
        
        self.plot(fig, ax, rsa_models, scores_temporal, groups, ttest_temporal_results, title='Temporal')
        
        plt.tight_layout()
        #fig.savefig(os.path.join(self.similarity_save_root_comparison, f'{title}.svg'))     
        plt.show()
        #plt.close()
        
    
    def plot(self, fig, ax, rsa_models, scores, groups, ttest_p, title=None, **kwargs):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 14})
        
        means = [np.mean(v) for k, v in scores.items()]
        stds = [np.std(v) for k, v in scores.items()]
        
        for idx, _ in enumerate(means):
            ax.bar(idx, _, width=0.5)
            ax.errorbar(idx, _, yerr=stds[idx], fmt='.', capsize=8, linewidth=2, color='black')
        
        utils_.sigstar(groups, ttest_p, ax)
        
        ax.set_xticks(np.arange(len(rsa_models)), rsa_models, rotation='vertical')
        ax.set_ylabel('Similarity Scores', fontsize=20)
        
        ax.set_title(f'{title}')
        
        
        
        
# ----------------------------------------------------------------------------------------------------------------------
class Human_similarity_scores_comparison(similarity_scores_comparison_base):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        for self.metric in self.metrics:
            
            utils_.make_dir(os.path.join(self.similarity_save_root, f'{self.metric}'))
            
            for self.cell_type in tqdm(self.cell_types, desc=f'{self.metric}'):
                
                utils_.make_dir(os.path.join(self.similarity_save_root, f'{self.metric}', self.cell_type))
                
                if '-_mismatched_comparison' in self.cell_type:
                    
                    self._mismatched_comparison = True
                    
                for self.used_id_num in self.used_id_nums:
                    
                    self.similarity_save_root_comparison = os.path.join(self.similarity_save_root, f'{self.metric}', self.cell_type, str(self.used_id_num))
                    
                    utils_.make_dir(self.similarity_save_root_comparison)
                    
                    self.rsa_dicts = {}
                    
                    for folder in self.folders:
                        
                        if self._mismatched_comparison:
                            
                            file_path = os.path.join(self.root, f'{folder}/Analysis/RSA/{self.primate}/{self.metric}/{self.cell_type}/{self.used_id_num}/RSA_results_{self.metric}_{self.used_id_num}_mismatched_Selective_Human_Strong_Selective_NN.pkl')
                                
                            if not os.path.exists(file_path):
                                self.rsa_dicts.update({
                                    folder: np.nan
                                    })
                                
                            else:
                                self.rsa_dicts.update({
                                    folder: utils_.load(file_path)
                                    })
                            
                        else:
                            
                            file_path = os.path.join(self.root, f'{folder}/Analysis/RSA/{self.primate}/{self.metric}/{self.cell_type}/{self.used_id_num}/RSA_results_{self.metric}_{self.cell_type}_{self.used_id_num}.pkl')
                            
                            if not os.path.exists(file_path):
                                self.rsa_dicts.update({
                                    folder: np.nan
                                    })
                                
                            else:
                                self.rsa_dicts.update({
                                    folder: utils_.load(file_path)
                                    })
                            
                    # ---
                    self.statistical_calculation()
                    
    def _plot(self, k, v, mask):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 14})
        
        fig, ax = plt.subplots(figsize=(10,6))
        
        means = [np.mean(v[_]) for _ in v.keys()]
        stds = [np.std(v[_]) for _ in v.keys()]
        
        for idx, _ in enumerate(means):
            ax.bar(idx+1, _, width=0.5)
            ax.errorbar(idx+1, _, yerr=stds[idx], fmt='.', capsize=8, linewidth=2, color='black')
        
        utils_.sigstar([[3,1], [3,2], [3,4], [3,5], [3,6], [3,7]], self.ttest_ind_p_values, ax)
        
        ax.set_xticks(np.arange(1,8), ['Baseline', 'VGG16', 'VGG16bn', 'S 4 IF C', 'S4 LIF C', 'S 4 LIF v', 'S 16 IF C'], rotation='vertical')
        ax.set_ylabel('Similarity Scores', fontsize=20)
        
        if self._mismatched_comparison:
            cell_type = self.cell_type.split('/')[-1]
        else:
            cell_type = self.cell_type
            
        title = f'Human RSA {k} Scores Comparison (VGG)\n{self.metric} {cell_type} {self.used_id_num} {mask}'
        ax.set_title(f'{title}')
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.similarity_save_root_comparison, f'{title}.svg'))  
        #plt.show()
        plt.close()
    

                 
            
# ======================================================================================================================
        
if __name__ == "__main__":
    
    layers, neurons, shapes = utils_.get_layers_and_units('vgg16', target_layers='act')
    
    root_dir = '/home/acxyle-workstation/Downloads/'

    #for monkey experiments
    #RSA_monkey = RSA_Monkey(root=os.path.join(root_dir, 'Face Identity Baseline'), layers=layers, neurons=neurons)
    #for first_corr in ['pearson']:
    #    for second_corr in ['pearson']:
    #        RSA_monkey(first_corr=first_corr, second_corr=second_corr)
    
# =============================================================================
#     RSA_Monkey_folds = RSA_Monkey_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
#     for first_corr in ['euclidean', 'pearson', 'spearman', 'mahalanobis', 'concordance']:
#         for second_corr in ['pearson', 'spearman', 'concordance']:
#             for route in ['p', 'sig']:
#                 RSA_Monkey_folds(first_corr=first_corr, second_corr=second_corr, route=route)
# =============================================================================
    
    roots_models = [
        (os.path.join(root_dir, 'Face Identity Baseline'), 'vgg16'),
        (os.path.join(root_dir, 'Face Identity VGG16_fold_'), 'vgg16'),
        (os.path.join(root_dir, 'Face Identity VGG16bn_fold_'), 'vgg16_bn'),
        (os.path.join(root_dir, 'Face Identity SpikingVGG16bn_IF_T4_CelebA2622_fold_'), 'spiking_vgg16_bn'),
        (os.path.join(root_dir, 'Face Identity SpikingVGG16bn_IF_T16_CelebA2622_fold_'), 'spiking_vgg16_bn'),
        (os.path.join(root_dir, 'Face Identity SpikingVGG16bn_LIF_T4_CelebA2622_fold_'), 'spiking_vgg16_bn'),
        (os.path.join(root_dir, 'Face Identity SpikingVGG16bn_LIF_T16_CelebA2622_fold_'), 'spiking_vgg16_bn')
        ]

    #RSA_Monkey_Comparison()(roots_models, first_corr='pearson', second_corr='spearman', route='p')
    #RSA_Human_Comparison()(roots_models, first_corr='pearson', second_corr='spearman', route='p')
    
    # for human experiments 
    #RSA_human = RSA_Human(root=os.path.join(root_dir, 'Face Identity Baseline'), layers=layers, neurons=neurons)
    #for firsct_corr in ['euclidean', 'mahalanobis', 'spearman', 'concordance']:
    #    for second_corr in ['pearson', 'spearman', 'concordance']:
    #        for used_unit_type in ['legacy', 'qualified', 'selective', 'non_selective']:
    #            for used_id_num in [50, 10]:
    #                RSA_human.(first_corr=firsct_corr, second_corr=second_corr, used_unit_type=used_unit_type, used_id_num=used_id_num)
    
# =============================================================================
#     RSA_Human_folds = RSA_Human_folds(num_folds=5, root=root, layers=layers, neurons=neurons)
#     for first_corr in ['euclidean', 'pearson', 'spearman', 'mahalanobis']:
#         for second_corr in ['pearson', 'spearman', 'concordance']:
#             for used_unit_type in ['legacy', 'qualified', 'selective', 'non_selective']:
#                 for used_id_num in [50, 10]:
#                     for route in ['p', 'sig']:
#                         RSA_Human_folds(first_corr=first_corr, second_corr=second_corr, used_unit_type=used_unit_type, used_id_num=used_id_num, route=route)
# =============================================================================
    
    
    # ------------------------------------------------------------------------------------------------------------------
    #for model comparison
    
    primate_nn_comparison = Monkey_similarity_scores_comparison(roots_models,
                                                                primate='Monkey',
                                                                first_corr='pearson', 
                                                                second_corr='spearman',
                                                                )
    
# =============================================================================
#     primate_nn_comparison = Human_similarity_scores_comparison(root=root_dir,
#                                                                primate='Human',
#                                                               folders=['Face Identity Baseline', 'Face Identity VGG16', 'Face Identity VGG16bn', 
#                                                                       'Face Identity SpikingVGG16bn_IF_T4_CelebA2622',
#                                                                       'Face Identity SpikingVGG16bn_LIF_T4_CelebA2622',
#                                                                       'Face Identity SpikingVGG16bn_LIF_T4_vggface',
#                                                                       'Face Identity SpikingVGG16bn_LIF_T16_CelebA2622'],  
#                                                               metrics=['euclidean', 'pearson'], 
#                                                               cell_types=['qualified', 'selective', 'non_selective',
#                                                                           'sensitive', 'non_sensitive', 'encode', 'non_encode', 
#                                                                           'all_sensitive_si', 'all_sensitive_mi',
#                                                                           'sensitive_si', 'sensitive_wsi', 'sensitive_mi', 'sensitive_wmi', 'sensitive_non_encode',
#                                                                           'non_sensitive_si', 'non_sensitive_wsi', 'non_sensitive_mi', 'non_sensitive_wmi', 'non_sensitive_non_encode'],
#                                                               used_id_nums=[50, 10],
#                                                               )
#     
#     primate_nn_comparison = Human_similarity_scores_comparison(root=root_dir,
#                                                                primate='Human',
#                                                               folders=['Face Identity Baseline', 'Face Identity VGG16', 'Face Identity VGG16bn', 
#                                                                       'Face Identity SpikingVGG16bn_IF_T4_CelebA2622',
#                                                                       'Face Identity SpikingVGG16bn_LIF_T4_CelebA2622',
#                                                                       'Face Identity SpikingVGG16bn_LIF_T4_vggface',
#                                                                       'Face Identity SpikingVGG16bn_LIF_T16_CelebA2622'],  
#                                                               metrics=['pearson'], 
#                                                               cell_types=[
#                                                                   '-_mismatched_comparison/Human Selective V.S. NN Strong Selective'
#                                                                   ],
#                                                               used_id_nums=[50, 10],
#                                                               )
# =============================================================================
