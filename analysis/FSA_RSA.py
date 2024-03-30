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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from joblib import Parallel, delayed
import scipy

from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind
from statsmodels.stats.multitest import multipletests


import utils_
import utils_similarity

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
        
        self.save_root
        self.layers
        
        self.primate_DM
        self.primate_DM_perm
        
        # --- if use permutation every time, those two can be ignored, 5-fold experiment is suggested
        # --- empirically, the max fluctuation of mean scores between experiments could be ±0.03 with num_perm = 1000
        self.primate_DM_temporal
        self.primate_DM_temporal_perm
        
        
    def calculation_RSA(self, first_corr='pearson', second_corr='spearman', FDR_test=True, alpha=0.05, num_perm=1000, FDR_method:str='fdr_bh', save=True, save_path=None, **kwargs):
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
        if save_path is None:
            
            save_path = os.path.join(self.save_root, f'RSA_results_{first_corr}_{second_corr}.pkl')
        
        if os.path.exists(save_path):
            
            RSA_dict = utils_.load(save_path)
            
        else:
            
            # -----
            pl = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(self.calculation_RSA_layer)(layer, first_corr=first_corr, second_corr=second_corr, FDR_test=FDR_test) for layer in tqdm(self.layers, desc='RSA primate'))
            
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
                
                if save:
                    
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
    
    
    def plot_RSA(self, RSA_dict, title=None, error_control_measure='sig_FDR', error_area=True, legend=False, vlim:list[float]=None, figsize=(10,6), **kwargs):
        """
            this function plot static RSA score and save
            
            input:
                RSA_dict: RSA data
                title: 
                error_control_measure: default 'sig_FDR'. Options: 'sig_Bonf'
                error_area: default True. Plot the mean+std error area caused by permutation area
                norm_plot: default None. For exterior call with given ylim for fiar comparison
        """
        
        print('[Codinfo] Executing static plotting...')
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)

        with warnings.catch_warnings():
            
            warnings.simplefilter(action='ignore')

            fig, ax = plt.subplots(figsize=figsize)
                
            plot_RSA(self.layers, ax, RSA_dict, title=title, vlim=vlim, legend=False)
            utils_similarity.fake_legend_describe_numpy(RSA_dict['similarity'], ax, RSA_dict[error_control_measure].astype(bool))

            plt.tight_layout(pad=1)
            plt.savefig(os.path.join(self.save_root, f'{title}.svg'), bbox_inches='tight')

            plt.close()
    
    
    def plot_RSA_temporal(self, RSA_dict, title=None, error_control_measure='sig_temporal_Bonf', vlim:list[float]=None, **kwargs):
        """
            function
            
            input:
                error_control_measure: 'sig_temporal_FDR' default. Options: 'sig_temporal_Bonf'
        """
        
        print('[Codinfo] Executing temporal plotting')
        
        extent = [self.ts.min()-5, self.ts.max()+5, -0.5, RSA_dict['similarity_temporal'].shape[0]-0.5]
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
    
        logging.getLogger('matplotlib').setLevel(logging.ERROR)    
    
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            fig, ax = plt.subplots(figsize=(np.array(RSA_dict['similarity_temporal'].T.shape)/3.7))
            
            plot_RSA_temporal(self.layers, fig, ax, RSA_dict, title=title, vlim=vlim, extent=extent)
            utils_similarity.fake_legend_describe_numpy(RSA_dict['similarity_temporal'], ax, RSA_dict[error_control_measure].astype(bool))
            
            #plt.tight_layout(pad=1)
            plt.savefig(os.path.join(self.save_root, f'{title}.svg'))     

            plt.close()
    

# ----------------------------------------------------------------------------------------------------------------------
class RSA_Monkey(Monkey_Neuron_Records_Process, FSA_DSM, RSA_Base):
    """
        ...
    """
    
    def __init__(self, seed, **kwargs):
        
        # --- init
        Monkey_Neuron_Records_Process.__init__(seed=seed)
        FSA_DSM.__init__(**kwargs)
        
        self.RSA_root = os.path.join(self.dest, 'RSA')
        utils_.make_dir(self.RSA_root)
        utils_.make_dir(os.path.join(self.RSA_root, 'Monkey'))
        
        
    def process_RSA_monkey(self, first_corr='pearson', second_corr='spearman', FDR_test=True, **kwargs):
        """
            the RSA process is based on self.primate_DM_* and self.NN_DM_*
            
            input:
                first_corr: default 'pearson', select from 'euclidean', 'pearson', 'spearman', 'mahalanobis', 'concordance'
                second_corr: default 'spearman', select from 'pearson', 'spearman', 'concordance'
                      
            permutation is only applied to primate data
            
        """
        
        # --- init
        self.save_root = os.path.join(self.RSA_root, 'Monkey', f'{first_corr}')
        utils_.make_dir(self.save_root)
        
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
        self.NN_DM_dict = {k:v['qualified'] for k,v in self.NN_unit_DSM_process(first_corr, vectorize=False, **kwargs).items()}   # layer - cell_type
        ...
 
        # ----- RSA calculation
        RSA_dict = self.calculation_RSA(first_corr, second_corr, **kwargs)
        
        # ----- plot
        self.plot_RSA(RSA_dict, title=f'RSA Score {self.model_structure} {first_corr} {second_corr}', **kwargs)
        
        self.plot_RSA_temporal(RSA_dict, title=f'RSA Score temporal {self.model_structure} {first_corr} {second_corr}', **kwargs)

        # --- example correlation
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
        plt.savefig(os.path.join(self.save_root, f'{title}.svg'))

        plt.close()
                                                        

# ----------------------------------------------------------------------------------------------------------------------
class RSA_Human(Human_Neuron_Records_Process, FSA_DSM, RSA_Base):
    """
        ...
    """
    
    def __init__(self, seed, **kwargs):
        
        Human_Neuron_Records_Process.__init__(self, seed=seed)
        FSA_DSM.__init__(**kwargs)
        
        self.RSA_root = os.path.join(self.dest, 'RSA')
        utils_.make_dir(self.RSA_root)
        
        self.save_root_primate = os.path.join(self.RSA_root, 'Human')
        utils_.make_dir(self.save_root_primate)
        
        
    def process_RSA_human(self, first_corr='pearson', second_corr='spearman', used_cell_type='qualified', used_id_num=50, FDR_test=True, **kwargs):
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
        utils_._print(f'Processing RSA for Human and NN | {self.model_structure} | {used_cell_type} | {used_id_num} | {first_corr} | {second_corr}')
        
        ...
        
        # --- folder init
        utils_.make_dir(save_root_first_corr:=os.path.join(self.save_root_primate, f'{first_corr}'))
        utils_.make_dir(save_root_second_corr:=os.path.join(save_root_first_corr, f'{second_corr}'))
        utils_.make_dir(save_root_cell_type:=os.path.join(save_root_second_corr, used_cell_type))
            
        self.save_root = os.path.join(save_root_cell_type, str(used_id_num))
        utils_.make_dir(self.save_root)
        
        # --- init
        self.used_id = self.human_corr_select_sub_identities(used_id_num)
        
        NN_DM_dict = self.NN_unit_DSM_process(first_corr)
        
        if used_cell_type == 'legacy':
            human_DM_dict = self.human_neuron_DSM_process(first_corr, 'selective', **kwargs)
            self.NN_DM_dict = {_: _vectorize_check(NN_DM_dict[_]['strong_selective'][np.ix_(self.used_id, self.used_id)]) for _ in NN_DM_dict.keys()}
        else:
            human_DM_dict = self.human_neuron_DSM_process(first_corr, used_cell_type, **kwargs)
            self.NN_DM_dict = {_: _vectorize_check(NN_DM_dict[_][used_cell_type][np.ix_(self.used_id, self.used_id)]) if ~np.isnan(NN_DM_dict[_][used_cell_type]).all() else np.nan for _ in NN_DM_dict.keys()}
            
        self.primate_DM = _vectorize_check(human_DM_dict['human_DM'][np.ix_(self.used_id, self.used_id)])
        self.primate_DM_temporal = np.array([_vectorize_check(_[np.ix_(self.used_id, self.used_id)]) for _ in human_DM_dict['human_DM_temporal']])
        
        if FDR_test:
            
            assert set(['human_DM_perm', 'human_DM_temporal_perm']).issubset(human_DM_dict.keys())
            
            self.primate_DM_perm = np.array([_vectorize_check(_[np.ix_(self.used_id, self.used_id)]) for _ in human_DM_dict['human_DM_perm']])
            self.primate_DM_temporal_perm = np.array([np.array([_vectorize_check(__[np.ix_(self.used_id, self.used_id)]) for __ in _]) for _ in human_DM_dict['human_DM_temporal_perm']])
        
        # --- RSA calculation
        save_path = os.path.join(self.save_root, f'RSA_results_{first_corr}_{second_corr}_{used_cell_type}_{used_id_num}.pkl')
        
        human_NN_corr_dict = self.calculation_RSA(first_corr=first_corr, second_corr=second_corr, save_path=save_path, **kwargs)
        
        # --- plot
        self.plot_RSA(human_NN_corr_dict, title=f'Human-NN RSA Score {self.model_structure} {first_corr} {second_corr} {used_cell_type} {used_id_num}', **kwargs)
        
        self.plot_RSA_temporal(human_NN_corr_dict, title=f'Human-NN RSA Score temporal\n{self.model_structure} {first_corr} {second_corr} {used_cell_type} {used_id_num}', **kwargs)
           


# ======================================================================================================================
#FIXME --- need to upgrade
def plot_RSA(layers, ax, RSA_dict, error_control_measure='sig_FDR', title=None, error_area=True, vlim:list[float]=None, legend=True):
    """
        #TODO 
        add the std error area
    """
    
    plot_x = range(len(layers))
    
    # --- 1. plot shaded error bars
    if error_area:
        perm_mean = np.mean(RSA_dict['similarity_perm'], axis=1)  
        perm_std = np.std(RSA_dict['similarity_perm'], axis=1)  
        ax.fill_between(plot_x, perm_mean-perm_std, perm_mean+perm_std, color='lightgray', edgecolor='none', alpha=0.5)
        ax.fill_between(plot_x, perm_mean-2*perm_std, perm_mean+2*perm_std, color='lightgray', edgecolor='none', alpha=0.5)
        ax.fill_between(plot_x, perm_mean-3*perm_std, perm_mean+3*perm_std, color='lightgray', edgecolor='none', alpha=0.5, label='perm 1~3 std')
        ax.plot(plot_x, perm_mean, color='dimgray', label='perm mean')
    
    # --- 2. plot RSA scores with FDR results
    similarity = RSA_dict['similarity']
    
    if 'similarity_std' in RSA_dict.keys():
        ax.fill_between(plot_x, similarity-RSA_dict['similarity_std'], similarity+RSA_dict['similarity_std'], edgecolor=None, facecolor='skyblue', alpha=0.75)

    for idx, _ in enumerate(RSA_dict[error_control_measure], 0):
         if not _:   
             ax.scatter(idx, similarity[idx], facecolors='none', edgecolors='blue')
         else:
             ax.scatter(idx, similarity[idx], facecolors='blue', edgecolors='blue')
             
    ax.plot(similarity, linestyle='dotted', color='deepskyblue')

    ax.set_ylabel("Spearman's $\\rho$")
    ax.set_xticks(plot_x)
    ax.set_xticklabels(layers, rotation=90, ha='center')
    ax.set_xlim([0, len(layers)-1])
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f'{title}')
    
    handles, labels = ax.get_legend_handles_labels()

    hollow_circle = Line2D([0], [0], marker='o', color='deepskyblue', linestyle='dotted', markerfacecolor='none', markersize=5, markeredgecolor='blue', linewidth=1)
    solid_circle = Line2D([0], [0], marker='o', color='deepskyblue', linestyle='dotted', markerfacecolor='blue', markersize=5, markeredgecolor='blue', linewidth=1)

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


def plot_RSA_temporal(layers, fig, ax, RSA_dict, error_control_measure='sig_temporal_Bonf', title=None, vlim:list[float]=None, extent:list[float]=None):
    
    def _is_binary(input:np.ndarray):
        input = np.nan_to_num(input, 0)     # assume nan comes from invalid data
        return np.all((input==0)|(input==1))
    
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
        """
            set the ROI as nan because plt display nothing for nan, so no mask for ROI
        """
        input = scipy.ndimage.gaussian_filter(input, sigma=1)
        input[input>(1-alpha)] = np.nan
        
        ax.imshow(input, aspect='auto',  cmap='gray', extent=extent, alpha=0.3)
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
    ax.set_title(f'{title}', fontsize=16)
    
    # FIXEME --- need to upgrade to merged model
    # significant correlation (Bonferroni/FDR)
    if error_control_measure == 'sig_temporal_FDR':
        if _is_binary(mask:=RSA_dict[error_control_measure]):
            #ax.imshow(mask, aspect='auto',  cmap='gray', extent=extent, interpolation='none', alpha=0.25)
            _p_contour(mask)

        else:
            _mask_contour(mask)
            
        
    elif error_control_measure == 'sig_temporal_Bonf':
        if _is_binary(mask:=RSA_dict[error_control_measure]):
            #ax.imshow(mask, aspect='auto',  cmap='gray', extent=extent, interpolation='none', alpha=0.25)
            _p_contour(mask)

        else:
            _mask_contour(mask)
            


# ======================================================================================================================
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
        rho = np.nan
    else:
        rho = spearmanr(x, y, nan_policy='omit').statistic
        
    return rho

def _pearson(x, y):
    return np.corrcoef(x, y)[0, 1]

def _ccc(x, y):
    return utils_similarity._ccc(x, y)

    
# ======================================================================================================================
#FIXME 5-folds model training required
#FIXME the design needs to upgrade for better use rather local debug
class similarity_scores_comparison_base():
    
    """
        [warning] the file storage is specialized for local RTX4090 machine, need to adjust for other storage format
    """
    
    def __init__(self, root:str=None, method='RSA', folders:list[str]=None, primate:str=None, 
                 metrics:list[str]=None, cell_types:list[str]=None, used_id_nums:list[str]=None):
        super().__init__()
        
        # ----- init
        utils_.make_dir(os.path.join(root, 'Face Identity - similarity'))
        utils_.make_dir(os.path.join(root, 'Face Identity - similarity', f'{method}'))
        
        self.similarity_save_root = os.path.join(root, 'Face Identity - similarity', f'{method}', f'{primate}')
        utils_.make_dir(self.similarity_save_root)
        
        self.root = root
        self.method = method
        self.folders = folders
        self.primate = primate

        self.metrics = metrics
        self.cell_types = cell_types
        self.used_id_nums = used_id_nums

    def _statistical_calculation(self, mask=None):
        
        # ----- all RSA scores
        if mask is None:
        
            self.rsa_scores_dicts = {_:self.rsa_dicts[_]['similarity'][~np.isnan(self.rsa_dicts[_]['similarity'])].ravel() if self.rsa_dicts[_] is not np.nan else np.nan for _ in self.rsa_dicts.keys()}
            self.rsa_scores_temporal_dicts = {_:self.rsa_dicts[_]['similarity_temporal'][~np.isnan(self.rsa_dicts[_]['similarity_temporal'])].ravel() if self.rsa_dicts[_] is not np.nan else np.nan for _ in self.rsa_dicts.keys()}
        
        # ----- FDR correction
        elif 'FDR' in mask:
            
            self.rsa_scores_dicts = {_:self.rsa_dicts[_]['similarity'][self.rsa_dicts[_]['sig_FDR'].astype(bool)].ravel() if self.rsa_dicts[_] is not np.nan else np.nan for _ in self.rsa_dicts.keys()}
            self.rsa_scores_temporal_dicts = {_:self.rsa_dicts[_]['similarity_temporal'][self.rsa_dicts[_]['sig_temporal_FDR'].astype(bool)].ravel() if self.rsa_dicts[_] is not np.nan else np.nan for _ in self.rsa_dicts.keys()}
        
        # ----- Bonf correction
        elif 'Bonf' in mask:
            
            self.rsa_scores_dicts = {_:self.rsa_dicts[_]['similarity'][self.rsa_dicts[_]['sig_Bonf'].astype(bool)].ravel() if self.rsa_dicts[_] is not np.nan else np.nan for _ in self.rsa_dicts.keys()}
            self.rsa_scores_temporal_dicts = {_:self.rsa_dicts[_]['similarity_temporal'][self.rsa_dicts[_]['sig_temporal_Bonf'].astype(bool)].ravel() if self.rsa_dicts[_] is not np.nan else np.nan for _ in self.rsa_dicts.keys()}
        
        groups = [[2,0], [2,1], [2,3], [2,4], [2,5], [2,6]]
        
        rsa_scores_dicts_merge = {'static': self.rsa_scores_dicts, 'temporal': self.rsa_scores_temporal_dicts}
        
        for rsa_scores_dict_key, rsa_scores_dict in rsa_scores_dicts_merge.items():
            
            rdsk = list(rsa_scores_dict.keys())
            
            self.ttest_ind_results = [ttest_ind(rsa_scores_dict[rdsk[_[0]]], rsa_scores_dict[rdsk[_[1]]]) if (rsa_scores_dict[rdsk[_[0]]] is not np.nan and rsa_scores_dict[rdsk[_[1]]] is not np.nan) else (np.nan, np.nan) for _ in groups]
            self.ttest_ind_statistics, self.ttest_ind_p_values = [_[0] for _ in self.ttest_ind_results], [_[1] for _ in self.ttest_ind_results]
            
            self._plot(rsa_scores_dict_key, rsa_scores_dict, mask)

        
    def _plot(self, ):
        
        print('6')
        
        
class Monkey_similarity_scores_comparison(similarity_scores_comparison_base):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        for self.metric in self.metrics:
            
            self.similarity_save_root_comparison = os.path.join(self.similarity_save_root, f'{self.metric}')
            utils_.make_dir(self.similarity_save_root_comparison)
            
            self.rsa_dicts = {}
            
            for folder in self.folders:
            
                file_path = os.path.join(self.root, f'{folder}/Analysis/RSA/{self.primate}/{self.metric}/RSA_results_{self.metric}.pkl')
                
                self.rsa_dicts.update({
                    folder: utils_.load(file_path)
                    })
            
            # ---
            self._statistical_calculation(mask='sig_FDR')
            self._statistical_calculation(mask='sig_Bonf')
            self._statistical_calculation()
    
    def _plot(self, rsa_scores_dict_key, rsa_scores_dict, mask):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 14})
        
        fig, ax = plt.subplots(figsize=(10,6))
        
        means = [np.mean(rsa_scores_dict[_]) for _ in rsa_scores_dict.keys()]
        stds = [np.std(rsa_scores_dict[_]) for _ in rsa_scores_dict.keys()]
        
        for idx, _ in enumerate(means):
            ax.bar(idx+1, _, width=0.5)
            ax.errorbar(idx+1, _, yerr=stds[idx], fmt='.', capsize=8, linewidth=2, color='black')
        
        utils_.sigstar([[3,1], [3,2], [3,4], [3,5], [3,6], [3,7]], self.ttest_ind_p_values, ax)
        
        ax.set_xticks(np.arange(1,8), ['Baseline', 'VGG16', 'VGG16bn', 'S 4 IF C', 'S4 LIF C', 'S 4 LIF v', 'S 16 IF C'], rotation='vertical')
        ax.set_ylabel('Similarity Scores', fontsize=20)
        
        title = f'Monkey RSA {rsa_scores_dict_key} Scores Comparison (VGG) {self.metric} {mask}'
        ax.set_title(f'{title}')
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.similarity_save_root_comparison, f'{title}.png'))
        fig.savefig(os.path.join(self.similarity_save_root_comparison, f'{title}.pdf'))     
        #plt.show()
        plt.close()
        
        
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
                    self._statistical_calculation(mask='sig_FDR')
                    self._statistical_calculation(mask='sig_Bonf')
                    self._statistical_calculation()
                    
    def _plot(self, rsa_scores_dict_key, rsa_scores_dict, mask):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 14})
        
        fig, ax = plt.subplots(figsize=(10,6))
        
        means = [np.mean(rsa_scores_dict[_]) for _ in rsa_scores_dict.keys()]
        stds = [np.std(rsa_scores_dict[_]) for _ in rsa_scores_dict.keys()]
        
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
            
        title = f'Human RSA {rsa_scores_dict_key} Scores Comparison (VGG)\n{self.metric} {cell_type} {self.used_id_num} {mask}'
        ax.set_title(f'{title}')
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.similarity_save_root_comparison, f'{title}.png'))
        fig.savefig(os.path.join(self.similarity_save_root_comparison, f'{title}.pdf'))     
        #plt.show()
        plt.close()
    

                 
            
# ======================================================================================================================
        
if __name__ == "__main__":
    
    layers, neurons, shapes = utils_.get_layers_and_units('vgg16', target_layers='act')
    
    root_dir = '/home/acxyle-workstation/Downloads/'

    #for monkey experiments
    #RSA_monkey = RSA_Monkey(NN_root=os.path.join(root_dir, 'Face Identity Baseline'), layers=layers, neurons=neurons)
    
    #for first_corr in ['pearson']:
    #    for second_corr in ['pearson']:
    #        RSA_monkey.process_RSA_monkey(first_corr=first_corr, second_corr=second_corr)

    # for human experiments 
    RSA_human = RSA_Human(NN_root=os.path.join(root_dir, 'Face Identity Baseline'), layers=layers, neurons=neurons)
    
    for firsct_corr in ['euclidean', 'mahalanobis', 'spearman', 'concordance']:
        for second_corr in ['pearson', 'spearman', 'concordance']:
            for used_cell_type in ['legacy', 'qualified', 'selective', 'non_selective']:
                for used_id_num in [50, 10]:
                    RSA_human.process_RSA_human(first_corr=firsct_corr, second_corr=second_corr, used_cell_type=used_cell_type, used_id_num=used_id_num)
    
# =============================================================================
#     #for model comparison
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
    
# =============================================================================
#     primate_nn_comparison = Monkey_similarity_scores_comparison(root=root_dir,
#                                                                primate='Monkey',
#                                                               folders=['Face Identity Baseline', 'Face Identity VGG16', 'Face Identity VGG16bn', 
#                                                                       'Face Identity SpikingVGG16bn_IF_T4_CelebA2622',
#                                                                       'Face Identity SpikingVGG16bn_LIF_T4_CelebA2622',
#                                                                       'Face Identity SpikingVGG16bn_LIF_T4_vggface',
#                                                                       'Face Identity SpikingVGG16bn_LIF_T16_CelebA2622'],  
#                                                               metrics=['euclidean','pearson'], 
#                                                               )
# =============================================================================
