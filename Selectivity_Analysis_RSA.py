#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:47:12 2023

@author: acxyle

[notice]
    all function with variable 'inds' and writing style like 'AaaBbbCcc' are not modified yet
    
[action required]
    Nov 10:
        restructure the monkey data and plot
    
"""

import torch

import os
import pickle
import warnings
import logging
import numpy as np
import scipy.io as sio
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from collections import Counter
from joblib import Parallel, delayed
from itertools import chain

import spiking_vgg, spiking_resnet, sew_resnet
from spikingjelly.activation_based import surrogate, neuron, functional

from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import vgg, resnet
import utils_
import utils_similarity

from Bio_Cell_Records_Process import Human_Neuron_Records_Process, Monkey_Neuron_Records_Process
from Selectivity_Analysis_DR_and_SM import Selectiviy_Analysis_SM


class Selectiviy_Analysis_Correlation_Monkey(Monkey_Neuron_Records_Process, Selectiviy_Analysis_SM):
    
    """
        this function inherit from Bio Cell and NN Unit process, aim to produce the RSA results for certain Model     
    
        only RSA between all channels from monkey IT and all units from NN
        
            input: 
                NN_root: path of NN correlation matrix
                
    """
    #FIXME
    def __init__(self, 
                 NN_root,
                 metric,
                 layers=None, neurons=None):
        
        Monkey_Neuron_Records_Process.__init__(self, )
        Selectiviy_Analysis_SM.__init__(self, root=NN_root, layers=layers, neurons=neurons)
        
        self.root = os.path.join(NN_root, 'Features/')     # <- folder for feature maps, which should be generated before analysis
        self.dest = os.path.join(NN_root, 'Analysis/')    # <- folder for analysis results
        utils_.make_dir(self.dest)
        
        if layers == None:
            raise RuntimeError('[Coderror] please assign proper layers')
            
        self.layers = layers
        self.neurons = neurons
        
        self.metric = metric     # euclidean needs to consider scaling
        self.correlation_matrix = utils_.pickle_load(os.path.join(self.dest, f'(Dis)Similarity_Matrix/{self.metric}/{self.metric}.pkl'))
        
        self.RSA_root = os.path.join(self.dest, 'RSA')
        utils_.make_dir(self.RSA_root)
        
        self.save_root = os.path.join(self.RSA_root, 'Monkey')
        utils_.make_dir(self.save_root)
        
        self.save_root = os.path.join(self.RSA_root, 'Monkey', f'{self.metric}')
        utils_.make_dir(self.save_root)
        
        
    #FIXME
    def monkey_neuron_analysis(self):
        
        print('[Codinfo] Excuting monkey neuron analysis...')
        
        #FIXME ----- usful but hard to read
        monkey_DM_dict = self.monkey_neuron_DSM_process()     # inherit from 'Monkey_Neuron_Records_Process'
        
        self.monkey_DM_v = monkey_DM_dict[self.metric]['monkey_DM_v']
        self.monkey_DM_v_perm = monkey_DM_dict[self.metric]['monkey_DM_v_perm']
        self.monkey_DM_v_temporal = monkey_DM_dict[self.metric]['monkey_DM_v_temporal']
        self.monkey_DM_v_perm_temporal = monkey_DM_dict[self.metric]['monkey_DM_v_perm_temporal']
        self.FR_id = monkey_DM_dict[self.metric]['FR_id']
        self.psth_id = monkey_DM_dict[self.metric]['psth_id']
        
        RSA_dict = self.representational_similarity_analysis()
        
        extent = [self.ts.min()-5, self.ts.max()+5, -0.5, RSA_dict['similarity_temporal'].shape[0]-0.5]
        
        # --- all units
        self.plot_static_correlation(self.layers, RSA_dict, title=f'RSA Score {self.model_structure} (all layers) {self.metric}')
        self.plot_temporal_correlation(self.layers, RSA_dict, title=f'RSA Score temporal {self.model_structure} (all layers) {self.metric}', extent=extent)
        
        # --- imaginary neurons
        idx, layer_n, _ = utils_.imaginary_neurons_vgg(self.layers, self.neurons)
        
        RSA_dict_neuron = {_:RSA_dict[_][idx] for _ in RSA_dict.keys()}
        
        self.plot_static_correlation(layer_n, RSA_dict_neuron, title=f'RSA Score {self.model_structure} (neuron) {self.metric}')
        self.plot_temporal_correlation(layer_n, RSA_dict_neuron, title=f'RSA Score temporal {self.model_structure} (neuron) {self.metric}', extent=extent)
        
        # --- example correlation
        self.plot_correlation_example(RSA_dict['similarity'])
        
    def representational_similarity_analysis(self, alpha=0.05, num_perm=1000, FDR_method:str='fdr_bh'):
        """
            calculation and save the RSA results for monkey
        """
        
        save_path = os.path.join(self.save_root, f'RSA_results_{self.metric}.pkl')
        
        if os.path.exists(save_path):
            
            print('[Codinfo] Loading RSA (1) corr scores and (2) permutation p_values...')
            
            RSA_dict = utils_.pickle_load(save_path)
            
        else:
            
            print('[Codinfo] Calculating RSA (1) corr scores and (2) permutation p_values...')
            
            pl = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(self.rsa_computation_layer)(layer) for layer in tqdm(self.layers, desc='RSA monkey'))
            
            similarity, similarity_perm, similarity_p, similarity_temporal, similarity_perm_temporal, similarity_p_perm_temporal = [np.array(_) for _ in list(zip(*pl))]
            
            # --- static
            (sig_FDR, p_FDR, alpha_Sadik, alpha_Bonf) = multipletests(similarity_p, alpha=alpha, method='fdr_bh')    # FDR (flase discovery rate) correction
            sig_Bonf = p_FDR<alpha_Bonf
            
            # --- temporal
            p_temporal_FDR = np.zeros((len(self.layers), self.monkey_DM_v_temporal.shape[0]))     # (num_layers, num_time_steps)
            sig_temporal_FDR =  p_temporal_FDR.copy()
            sig_temporal_Bonf = p_temporal_FDR.copy()
            
            for _ in range(len(self.layers)):
                (sig_temporal_FDR[_, :], p_temporal_FDR[_, :], alpha_Sadik_temporal, alpha_Bonf_temporal) = multipletests(similarity_p_perm_temporal[_, :], alpha=alpha, method=FDR_method)      # FDR
                sig_temporal_Bonf[_, :] = p_temporal_FDR[_, :]<alpha_Bonf_temporal     # Bonf correction
            
            # --- seal results
            RSA_dict = {
                'similarity': similarity,
                'similarity_perm': similarity_perm,
                'similarity_p': similarity_p,
                
                'similarity_temporal': similarity_temporal,
                'similarity_perm_temporal': similarity_perm_temporal,
                'similarity_p_perm_temporal': similarity_p_perm_temporal,
                
                'p_FDR': p_FDR,
                'sig_FDR': sig_FDR,
                'sig_Bonf': sig_Bonf,
                
                'p_temporal_FDR': p_temporal_FDR,
                'sig_temporal_FDR': sig_temporal_FDR,
                'sig_temporal_Bonf': sig_temporal_Bonf,
                }
            
            utils_.pickle_dump(save_path, RSA_dict)
        
        return RSA_dict
    
    #FIXME - for different types of units
    def rsa_computation_layer(self, layer, neuron_type='all', num_perm=1000):    
        
        NN_DM_v = self.correlation_matrix[layer][neuron_type]['vector']     # (1225,)
        
        # ----- static
        corr_coef = spearmanr(self.monkey_DM_v, NN_DM_v, nan_policy='raise').statistic
        
        # --- 1st is a constant permutation results; 2nd is for random permutation each time
        #corr_coef_perm = np.array([spearmanr(self.monkey_DM_v_perm[_, :], NN_DM_v, nan_policy='raise').statistic for _ in range(num_perm)])     # (1000,)
        corr_coef_perm = np.array([spearmanr(utils_similarity.selectivity_analysis_calculation(self.metric, self.FR_id[np.random.permutation(self.FR_id.shape[0]),:])['vector'], NN_DM_v, nan_policy='raise').statistic for _ in range(num_perm)])
        
        p_perm = np.mean(corr_coef_perm > corr_coef)     # equal to: np.sum(corr_coef_perm > corr_coef)/num_perm
    
        # ----- temporal
        time_steps = self.monkey_DM_v_temporal.shape[0]
        
        corr_coef_temporal = np.zeros(time_steps)
        corr_coef_perm_temporal = np.zeros((time_steps, num_perm))
        p_perm_temporal = np.zeros(time_steps)
        
        Parallel(n_jobs=int(os.cpu_count()/2))(delayed(self.rsa_computation_layer_dynamic)(NN_DM_v, t, corr_coef_temporal, corr_coef_perm_temporal, p_perm_temporal) for t in range(time_steps))
            
        results = [corr_coef, corr_coef_perm, p_perm, corr_coef_temporal, corr_coef_perm_temporal, p_perm_temporal]    
        
        return results
    
    def rsa_computation_layer_dynamic(self, NN_DM_v, t, corr_coef_temporal, corr_coef_perm_temporal, p_perm_temporal, num_perm=1000):
        
        r = spearmanr(self.monkey_DM_v_temporal[t,:], NN_DM_v, nan_policy='raise').statistic
        corr_coef_temporal[t] = r
        
        r_perm = np.array([spearmanr(self.monkey_DM_v_perm_temporal[t, _, :], NN_DM_v, nan_policy='raise').statistic for _ in range(num_perm)])      # (1000,)
        corr_coef_perm_temporal[t,:] = r_perm
        
        p_perm_temporal[t] = np.mean(r_perm > r)
    
    #FIXME - make it useful for both Human and Monkey
    def plot_static_correlation(self, layers, RSA_dict, title=None, error_control_measure='sig_FDR', error_area=True, norm_plot:list[float]=None):
        
        """
            this function plot static RSA score and save
            
            input:
                layers: define the x axis
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
            if 'all' in title:
                figsize = (10,6)
            elif 'neuron' in title:
                figsize = (8,5)
                
            fig, ax = plt.subplots(figsize=figsize)
            
            plot_static_correlation(ax, RSA_dict, error_control_measure, title, error_area, norm_plot)
    
            plt.tight_layout(pad=1)
            plt.savefig(os.path.join(self.save_root, f'{title}.png'), bbox_inches='tight')
            plt.savefig(os.path.join(self.save_root, f'{title}.eps'), bbox_inches='tight')   
            #plt.show()
            plt.close()
    
    def plot_temporal_correlation(self, layers, RSA_dict, title=None, error_control_measure='sig_temporal_FDR', vlim:list[float]=None, extent:list[float]=None):
        """
            function
            
            input:
                error_control_measure: 'sig_temporal_FDR' default. Options: 'sig_temporal_Bonf'
        """
        
        print('[Codinfo] Executing temporal plotting')
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
    
        logging.getLogger('matplotlib').setLevel(logging.ERROR)    
    
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            fig, ax = plt.subplots(figsize=(np.array(RSA_dict['similarity_temporal'].T.shape)/5))
            
            plot_temporal_correlation(fig, ax, RSA_dict, error_control_measure, title, vlim, extent)
                    
            plt.tight_layout(pad=1)
            
            plt.savefig(os.path.join(self.save_root, f'{title}.png'))
            plt.savefig(os.path.join(self.save_root, f'{title}.pdf'))     
            #plt.show()
            plt.close()
                
    #FIXME
    def plot_correlation_example(self, similarity, neuron_type='all', attach_psth:bool=False):
        """
            this function plot with fig definition and ax addition
        """
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        # plot correlation for sample layer
        max_idx = np.argmax(similarity)  # find the layer with strongest similarity score
        layer = self.layers[max_idx]
        NN_DM_v = self.correlation_matrix[layer][neuron_type]['vector']
        
        if not attach_psth:
            
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # plot sample PSTH - they namually set the target time point is 90, same results for below 2 methods
            #bestTimeFR = np.mean(self.meanPSTHID[:, np.where(self.psthTime == 90)[0][0], :], axis=0)
            bestTimeFR = np.mean(self.meanPSTHID[:, self.psthTime>60, :], axis=(0,1))
            
            most_active_cell = np.argmax(bestTimeFR)

            # plot corr example
            r, p, _ = self.plot_corr_2d(self.monkey_DM_v, NN_DM_v, 'blue', ax, 'Spearman')
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
            r, p, _ = self.plot_corr_2d(self.monkey_DM_v, NN_DM_v, 'b', axes[1], 'Spearman')
            axes[1].set_xlabel('Monkey Pairwise Distance')
            axes[1].set_ylabel('Network Pairwise Distance')
            axes[1].set_title(f'r:{r:.3f}, p:{p:.3e}')
        
        title = f'Monkey - {self.model_structure} {layer} polyfit {self.metric}'
        fig.suptitle(f'{title}', fontsize=20, y=1.)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root, f'{title}.png'))
        plt.savefig(os.path.join(self.save_root, f'{title}.pdf'))     
        #plt.show()
        plt.close()
                                                        
    #FIXME
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

# ======================================================================================================================
class Selectiviy_Analysis_Correlation_Human(Human_Neuron_Records_Process, Selectiviy_Analysis_SM):
   
    """
        [Purpose] remove the MATLAB results denpendencies in this code
        
        Working...
        
        [Purpose] make human neuron response as an independent work rather embedded into Human_Correlation calculation
        
    """
    def __init__(self,

                 NN_root='/Identity_SpikingVGG16bn_LIF_CelebA9326_Neuron/Correlation/', 
                 data='CelebA',
                 
                 layers=None, neurons=None):
        
        Human_Neuron_Records_Process.__init__(self, )
        Selectiviy_Analysis_SM.__init__(self, root=NN_root, layers=layers, neurons=neurons)
        
        self.root = os.path.join(NN_root, 'Features/')     # <- folder for feature maps, which should be generated before analysis
        self.dest = os.path.join(NN_root, 'Analysis/')    # <- folder for analysis results
        utils_.make_dir(self.dest)
        
        self.layers = layers
        self.neurons = neurons

        self.data_set = data
        
        self.RSA_root = os.path.join(self.dest, 'RSA')
        utils_.make_dir(self.RSA_root)
        
        self.save_root = os.path.join(self.RSA_root, 'Human')
        utils_.make_dir(self.save_root)
        
 
    def human_neuron_analysis(self, metrics=['euclidean', 'pearson'], 
                              cell_types=[
                                          'qualified', 'selective', 'non_selective',
                                          'sensitive', 'non_sensitive', 'encode', 'non_encode', 
                                          'all_sensitive_si', 'all_sensitive_mi',
                                          'sensitive_si', 'sensitive_wsi', 'sensitive_mi', 'sensitive_wmi', 'sensitive_non_encode',
                                          'non_sensitive_si', 'non_sensitive_wsi', 'non_sensitive_mi', 'non_sensitive_wmi', 'non_sensitive_non_encode'
                                          ], 
                              used_id_nums=[50, 10]):
        """
            [task] should make it clear what is bin_size and step_size
            [warning] this is test version now, merged process here, including plot and calculation
        """
        # --- additional parameters
        ...
        
        # --- RSA
        self.human_neuron_RSA_analysis(metrics, cell_types, used_id_nums)
         
    #FIXME use for loop for different type of cells and used_ids
    def human_neuron_RSA_analysis(self, metrics:list[str]=None, cell_types:list[str]=None, used_id_nums:list[int]=None):
        """
            Each process consist 3 sections:
                1) generate biological neuron responses according to neuron types - [self.human_neuron_spike_process()]
                2) generate feature maps of artificial units and calculate the similarity - [self.human_NN_RSA_analysis_sub_id_plot()]
                3) plot according to previous outcomes - [self.human_neuron_RSA_emporal_plot()]
        """
        print(f'[Codinfo] \nUsed metrics: {metrics}\nUsed types: {cell_types}\nUsed ID: {used_id_nums}')
        
        for metric in metrics:
            
            self.save_root_metric = os.path.join(self.save_root, f'{metric}')
            utils_.make_dir(self.save_root_metric)
            
            for used_cell_type in cell_types:
                
                self.save_root_cell_type = os.path.join(self.save_root_metric, used_cell_type)
                utils_.make_dir(self.save_root_cell_type)
                
                for used_id_num in used_id_nums:
                    
                    self.save_root_ids = os.path.join(self.save_root_cell_type, str(used_id_num))
                    utils_.make_dir(self.save_root_ids)
                    
                    # --- from Human_Neuron_Records_Process
                    selected_ids, human_DM_dict = self.human_neuron_DSM_process_sub_id(metric, used_cell_type, used_id_num)
                    
                    if human_DM_dict is None:
                        print(f'[Codinfo] No cell of {used_cell_type}')
                        pass
                    else:
                        self.human_NN_RSA_analysis(metric, human_DM_dict, selected_ids, used_id_num, used_cell_type)
                

    # ------------------------------------------------------------------------------------------------------------------
    def human_NN_RSA_analysis(self, metric, human_DM_dict, selected_ids, used_id_num, used_cell_type):
        """
        
        """

        # --- calculation
        human_NN_corr_dict = self.human_NN_RSA_analysis_sub_id(metric, human_DM_dict, selected_ids, used_id_num, used_cell_type)

        # --- plot - all layers
        
        print(f'[Codinfo] Ploting Single plot for {self.model_structure} {metric} {used_cell_type} {used_id_num}...')
        
        self.human_NN_RSA_analysis_sub_id_plot(
            self.layers,
            human_NN_corr_dict,
            used_cell_type, used_id_num,
            title=f'Human-NN RSA Score {self.model_structure} (all layers) {metric}_{used_cell_type}_{used_id_num}')
        
        self.human_neuron_RSA_temporal_plot(
            self.layers,
            human_NN_corr_dict,
            used_cell_type, used_id_num,
            title=f'Human-NN RSA Score temporal\n{self.model_structure} (all layers) {metric}_{used_cell_type}_{used_id_num}')
    
        # --- plot - neurons
        idx, layers_n, _ = utils_.imaginary_neurons_vgg(self.layers, self.neurons)
        human_NN_corr_dict_neuron = {_:human_NN_corr_dict[_][idx] for _ in human_NN_corr_dict.keys()}
        
        self.human_NN_RSA_analysis_sub_id_plot(
            layers_n,
            human_NN_corr_dict_neuron,
            used_cell_type, used_id_num,
            title=f'Human-NN RSA Score {self.model_structure} (neuron) {metric}_{used_cell_type}_{used_id_num}')
        
        self.human_neuron_RSA_temporal_plot(
            layers_n,
            human_NN_corr_dict_neuron,
            used_cell_type, used_id_num,
            title=f'Human-NN RSA Score temporal\n{self.model_structure} (neuron) {metric}_{used_cell_type}_{used_id_num}')

    #FIXME
    def human_NN_RSA_analysis_sub_id(self, metric, human_DM_dict, selected_ids, used_id_num, used_cell_type, num_perm=1000, alpha=0.05):
        """
            in practice, to avoid the unwanted asymmetric matrix, use 
                'NN_DM[np.triu_indices(used_id_num, 1)]'
            to replace the 
                'squareform(NN_DM)'
            after selected sub ids
        """
        
        save_path = os.path.join(self.save_root_ids, f'RSA_results_{metric}_{used_cell_type}_{used_id_num}.pkl')
        
        if os.path.exists(save_path):
            
            print(f'[Codinfo] Loading Human-NN Similarity of [{metric}] [{used_cell_type}] [{used_id_num}]...')
            
            RSA_dict = utils_.pickle_load(save_path)
        
        else:
            
            print(f'[Codinfo] Calculating Human-NN Similarity of [{metric}] [{used_cell_type}] [{used_id_num}]...')
            
            # ----- from NN Selectivity_Analysis_SM
            metric_dict = self.selectivity_analysis_similarity(metric)
            
            human_DM_v = human_DM_dict['human_DM_v']
            human_DM_v_temporal = human_DM_dict['human_DM_v_temporal']
            
            # ----- init, perm results, method 1 can load data immediately, and method 2 demands 10 mins on 24 (6 cores)
            # threads CPU, the two methods may have tiny differences for each time
            # --- 1. for constant results 
            human_DM_v_perm = human_DM_dict['human_DM_v_perm']
            human_DM_v_perm_temporal = human_DM_dict['human_DM_v_perm_temporal']
            # --- 2. permutation for evry time
            #human_DM_v_perm = np.array([_['vector'] for _ in Parallel(n_jobs=-1)(delayed(utils_similarity.selectivity_analysis_calculation)(metric, self.meanFR_id[np.random.permutation(len(selected_ids))]) for _ in tqdm(range(num_perm), desc='Human corr'))])
            #human_DM_v_perm_temporal = np.zeros((self.meanFR_PSTH_id.shape[0], num_perm, human_DM_v.size))     # (31, 1000, 1225)
            #for t in tqdm(range(self.meanFR_PSTH_id.shape[0]), desc='Human corr temporal'):     # for each time point
            #    human_DM_v_perm_temporal[t, :, :] = np.array([_['vector'] for _ in Parallel(n_jobs=-1)(delayed(utils_similarity.selectivity_analysis_calculation)(metric, self.meanFR_PSTH_id[t, np.random.permutation(len(selected_ids))]) for _ in range(num_perm))])
            
            # ---
            if used_cell_type == 'qualified':
                NN_DM_dict = {_:metric_dict[_]['all']['matrix'] for _ in metric_dict.keys()}
          
            else:
                NN_DM_dict = {_:metric_dict[_][used_cell_type]['matrix'] if (metric_dict[_][used_cell_type] is not None and metric_dict[_][used_cell_type]['matrix'] is not None) else None for _ in metric_dict.keys()}
            
            # --- init
            corr_coef = np.zeros(len(self.layers))     # (num_layers,)
            corr_coef_perm = np.zeros((len(self.layers), human_DM_v_perm.shape[0]))     # (num_layers, num_perm)
            p_perm = np.zeros(len(self.layers))     # (num_layers,)
            
            corr_coef_temporal = np.zeros((len(self.layers), human_DM_v_temporal.shape[0]))     # (num_layers, num_time_steps)
            corr_coef_perm_temporal = np.zeros((len(self.layers), *human_DM_v_perm_temporal.shape[:2]))     # (num_layers, num_time_steps, num_perm)
            p_perm_temporal = np.zeros((len(self.layers), human_DM_v_temporal.shape[0]))     # (num_layers, num_time_steps)
            
            # --- similarity loop over (1) layer and (2) timestep
            tqdm_bar = tqdm(total=len(self.layers), desc=f'Human-NN RSA {metric} {used_cell_type} {used_id_num}')
            for l_idx, layer in enumerate(self.layers, 0):     # for each layer
                
                NN_DM = NN_DM_dict[layer]     # select one layer -> (50,50)
                
                if NN_DM is None:     # abnormal detection 1: NN_DM is None
                    
                    corr_coef[l_idx] = np.nan
                    corr_coef_perm[l_idx, :] = np.full((num_perm, ), np.nan)
                    p_perm[l_idx] = np.nan
                    
                    corr_coef_temporal[l_idx, :] = np.full((human_DM_v_temporal.shape[0], ), np.nan)
                    corr_coef_perm_temporal[l_idx, :, :] = np.full(human_DM_v_perm_temporal.shape[:2], np.nan)
                    p_perm_temporal[l_idx, :] = np.full((human_DM_v_temporal.shape[0], ), np.nan)
                    
                else:
                    # --- select sub ids
                    NN_DM = NN_DM[np.ix_(selected_ids, selected_ids)]     # (50,50) -> (10,10)
                    
                    if metric == 'euclidean':
                        NN_DM = NN_DM/(np.max(NN_DM[~np.isnan(NN_DM)])+1e-3)

                    NN_DM_v = NN_DM[np.triu_indices(used_id_num, 1)]  # (10,10) -> (45)
                    
                    # --- FR (static)
                    rho = spearmanr_(human_DM_v, NN_DM_v)     # abnormal detection 2: NN_DM_v has NaN value or constant
                    corr_coef[l_idx] = rho
                    
                    pl = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(spearmanr_)(human_DM_v_perm[_, :], NN_DM_v) for _ in range(human_DM_v_perm.shape[0]))
                    
                    corr_coef_perm_seg = [_ for _ in pl]
                    corr_coef_perm[l_idx, :] = corr_coef_perm_seg
                    
                    if np.isnan(rho):
                        p_perm[l_idx] = np.nan
                    else:
                        p_perm[l_idx] = np.mean(corr_coef_perm_seg > rho)
                    
                    # --- PSTH (temporal)
                    for t_idx in range(human_DM_v_temporal.shape[0]):
                        
                        rho_temporal = spearmanr_(human_DM_v_temporal[t_idx, :], NN_DM_v)
                        corr_coef_temporal[l_idx, t_idx] = rho_temporal
    
                        pl = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(spearmanr_)(human_DM_v_perm_temporal[t_idx, _, :], NN_DM_v) for _ in range(human_DM_v_perm_temporal.shape[1]))
                        
                        corr_coef_perm_temporal_seg = [_ for _ in pl]
                        corr_coef_perm_temporal[l_idx, t_idx, :] = corr_coef_perm_temporal_seg
                        
                        if np.isnan(rho_temporal):
                            p_perm_temporal[l_idx, t_idx] = np.nan
                        else:
                            p_perm_temporal[l_idx, t_idx] = np.mean(corr_coef_perm_temporal_seg > rho_temporal)
                
                tqdm_bar.update(1)
                
            # --------------------------------------------------------------------------------------------------------------
            # --- static correction (with nan alignment)
            (_, _, alpha_Sadik, alpha_Bonf) = multipletests(p_perm, alpha=alpha, method='fdr_bh')  
            (sig_FDR, p_FDR, _, _) = multipletests(p_perm[~np.isnan(p_perm)], alpha=alpha, method='fdr_bh')    # FDR (flase discovery rate) correction
            
            p_FDR_aligned = np.full_like(p_perm, np.nan)
            sig_FDR_aligned = p_FDR_aligned.copy()
            
            p_FDR_aligned[~np.isnan(p_perm)] = p_FDR
            sig_FDR_aligned[~np.isnan(p_perm)] = sig_FDR
            
            sig_Bonf = p_FDR_aligned<alpha_Bonf
            
            # --- temporal
            p_temporal_FDR = np.zeros((len(self.layers), human_DM_v_temporal.shape[0]))     # (num_layers, num_time_steps)
            sig_temporal_FDR =  p_temporal_FDR.copy()
            sig_temporal_Bonf = p_temporal_FDR.copy()
            
            for l_idx in range(len(self.layers)):
                
                (_, _, alpha_Sadik_temporal, alpha_Bonf_temporal) = multipletests(p_perm_temporal[l_idx, :], alpha=alpha, method='fdr_bh')      # FDR
                
                if np.all(np.isnan(p_perm_temporal[l_idx, :])):
                    p_temporal_FDR_aligned = np.full_like((p_perm_temporal[l_idx, :]), np.nan)
                    sig_temporal_FDR_aligned = p_temporal_FDR_aligned.copy()
                    
                else:
                    (sig_temporal_FDR_seg, p_temporal_FDR_seg, _, _) = multipletests(p_perm_temporal[l_idx, :][~np.isnan(p_perm_temporal[l_idx, :])], alpha=alpha, method='fdr_bh')      # FDR
                
                    p_temporal_FDR_aligned = np.full_like(p_perm_temporal[l_idx, :], np.nan)
                    sig_temporal_FDR_aligned = p_temporal_FDR_aligned.copy()
                    
                    p_temporal_FDR_aligned[~np.isnan(p_perm_temporal[l_idx, :])] = p_temporal_FDR_seg
                    sig_temporal_FDR_aligned[~np.isnan(p_perm_temporal[l_idx, :])] = sig_temporal_FDR_seg
                
                p_temporal_FDR[l_idx, :] = p_temporal_FDR_aligned
                sig_temporal_FDR[l_idx, :] = sig_temporal_FDR_aligned
                
                sig_temporal_Bonf[l_idx, :] = p_temporal_FDR_aligned<alpha_Bonf_temporal     # Bonf correction
                
            # --- seal results
            RSA_dict = {
                'similarity': corr_coef,
                'similarity_perm': corr_coef_perm,
                'similarity_p': p_perm,
                
                'similarity_temporal': corr_coef_temporal,
                'similarity_perm_temporal': corr_coef_perm_temporal,
                'similarity_p_perm_temporal': p_perm_temporal,
                
                'p_FDR': p_FDR_aligned,
                'sig_FDR': sig_FDR_aligned,
                'sig_Bonf': sig_Bonf,
                
                'p_temporal_FDR': p_temporal_FDR,
                'sig_temporal_FDR': sig_temporal_FDR,
                'sig_temporal_Bonf': sig_temporal_Bonf,
                }
            
            # --- save data
            utils_.pickle_dump(save_path, RSA_dict)
        
        return RSA_dict
    
    def human_NN_RSA_analysis_sub_id_plot(self, 
                                     layers,
                                     RSA_dict,
                                     used_cell_type, used_id_num,
                                     title=None, error_control_measure='sig_FDR', error_area=True, norm_plot:list[float]=None
                                     ):
        
        # --------
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)

        with warnings.catch_warnings():
            
            warnings.simplefilter(action='ignore')
            if 'all' in title:
                figsize = (int(np.ceil(len(layers)/4)), 6)
            elif 'neuron' in title:
                figsize = (int(np.ceil(len(layers)/5)), 5)
                
            fig, ax = plt.subplots(figsize=figsize)
            
            plot_static_correlation(ax, layers, RSA_dict, error_control_measure, title, error_area, False, norm_plot)
            utils_similarity.fake_legend_describe_numpy(RSA_dict['similarity'], ax)
    
            plt.tight_layout(pad=1)
            plt.savefig(os.path.join(self.save_root_ids, f'{title}.png'), bbox_inches='tight')
            plt.savefig(os.path.join(self.save_root_ids, f'{title}.pdf'), bbox_inches='tight')   
            #plt.show()
            plt.close()
        # --------
        
        plt.close()
    
    def human_neuron_RSA_temporal_plot(self, 
                                       layers,
                                       RSA_dict,
                                       used_cell_type, used_id_num,
                                       title=None, error_control_measure='sig_FDR', error_area=True, norm_plot:list[float]=None, vlim:list[float]=None
                                       ):
        
        # ------
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        extent = [-250, 1001, -0.5, RSA_dict['similarity_temporal'].shape[0]-0.5]
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)    
    
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            fig, ax = plt.subplots(figsize=(np.array(RSA_dict['similarity_temporal'].T.shape)/5))
            
            plot_temporal_correlation(layers, fig, ax, RSA_dict, error_control_measure, title, vlim, extent)
                    
            plt.tight_layout(pad=1)
            
            plt.savefig(os.path.join(self.save_root_ids, f'{title}.png'))
            plt.savefig(os.path.join(self.save_root_ids, f'{title}.pdf'))     
            #plt.show()
            plt.close()
            
    # ------------------------------------------------------------------------------------------------------------------
    # [notice] test version
    def human_NN_RSA_plot_assemble(self, metrics:list[str]=['euclidean', 'pearson'], 
                                   cell_types:list[str]=['qualified', 'selective', 'non_selective'],
                                   used_id_nums=[50, 10],
                                   ):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        # --- init

        self.human_NN_RSA_plot_static(np.arange(len(self.layers)), self.layers, metrics, cell_types, used_id_nums, postfix='all layers')
        self.human_NN_RSA_plot_temporal(np.arange(len(self.layers)), self.layers, metrics, cell_types, used_id_nums, postfix='all layers')
        
        idx, layers_n, _ = utils_.imaginary_neurons_vgg(self.layers, self.neurons)
        
        self.human_NN_RSA_plot_static(idx, layers_n, metrics, cell_types, used_id_nums, postfix='neuron')
        self.human_NN_RSA_plot_temporal(idx, layers_n, metrics, cell_types, used_id_nums, postfix='neuron')
        
        
              
    def human_NN_RSA_plot_static(self, idx, layers, metrics:list[str]=None, cell_types:list[str]=None, used_id_nums:list[int]=None, 
                                 error_control_measure:str='sig_FDR', postfix=''):
        
        """
        
        """
        c_to_l_idx = layers.index('L5_maxpool')     # for vgg model
        
        for metric in metrics:
            
            metric_path = os.path.join(self.save_root, metric)
        
            # ----- info collection
            hyper_RSA_dict_metric = self._human_NN_RSA_load_similarity_dict(idx, metric, cell_types, used_id_nums, postfix)
               
            assemble_values = np.array([hyper_RSA_dict_metric[type_][ids]['similarity'] for type_ in cell_types for ids in used_id_nums])
            
            hyper_vmax = np.max(assemble_values[~np.isnan(assemble_values)])
            hyper_vmin = np.min(assemble_values[~np.isnan(assemble_values)])
            
            hyper_radius = hyper_vmax - hyper_vmin
                    
            # ----- plot
            fig, axes = plt.subplots(len(used_id_nums), len(cell_types), figsize=(len(cell_types)*7, len(used_id_nums)*5))
            
            row_idx = 0
            col_idx = 0
            
            for cell_type in cell_types:

                for used_id_num in used_id_nums:
                    
                    RSA_dict = hyper_RSA_dict_metric[cell_type][used_id_num]
                    
                    # ---
                    utils_similarity.fake_legend_describe_numpy(RSA_dict['similarity'], axes[row_idx, col_idx])
                    # ---
                    
                    plot_static_correlation(axes[row_idx, col_idx], layers, RSA_dict, norm_vlim=[hyper_vmin-0.1*hyper_radius, hyper_vmax+0.2*hyper_radius])

                    axes[row_idx, col_idx].set_xticks([0, c_to_l_idx, len(layers)-1])
                    axes[row_idx, col_idx].set_xticklabels([0, f'{(c_to_l_idx+1)/len(layers):.1f}', 1], rotation='horizontal')
                    axes[row_idx, col_idx].set_title(f'{cell_type} {used_id_num}')
                    axes[row_idx, col_idx].vlines(c_to_l_idx, hyper_vmin-0.1*hyper_radius, hyper_vmax+0.2*hyper_radius, linestyle='--', colors='gray', alpha=0.75)
                    
                    if col_idx == 0 and row_idx == 0:
                        
                        handles, labels = axes[row_idx, col_idx].get_legend_handles_labels()
                        
                        c_to_l_line = Line2D([0], [0], color='gray', linestyle='--', linewidth=1)
                        hollow_circle = Line2D([0], [0], marker='o', color='deepskyblue', linestyle='dotted', markerfacecolor='none', markersize=5, markeredgecolor='blue', linewidth=1)
                        solid_circle = Line2D([0], [0], marker='o', color='deepskyblue', linestyle='dotted', markerfacecolor='blue', markersize=5, markeredgecolor='blue', linewidth=1)

                        handles.extend([hollow_circle, solid_circle, c_to_l_line])
                        labels.extend([f"fialed {error_control_measure.split('_')[1]}", f"passed {error_control_measure.split('_')[1]}", 'conv_to_linear'])
                        
                        fig.legend(handles, labels, framealpha=0.5, loc='center left', bbox_to_anchor=(-0.1, 0.5))
                        
                    axes[row_idx, col_idx].set_ylabel('')
                    axes[row_idx, col_idx].get_legend().remove()
                    
                    row_idx += 1
                    
                    if row_idx == len(used_id_nums):
                        
                        col_idx += 1
                        row_idx = 0
                
            suptitle = f'Human-NN {self.model_structure} RSA for {metric} {postfix}'
            fig.suptitle(suptitle, fontsize=22)
            
            logging.getLogger('matplotlib').setLevel(logging.ERROR)    
        
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                fig.tight_layout()
                
                fig.savefig(os.path.join(metric_path, f'{suptitle}_{str(len(cell_types))}_types_{postfix}'), bbox_inches='tight')
                plt.close()
              
            
    def human_NN_RSA_plot_temporal(self, idx, layers, metrics:list[str]=None, cell_types:list[str]=None, used_id_nums:list[int]=None, 
                                 error_control_measure:str='sig_FDR', postfix=''):
        
        c_to_l_idx = layers.index('L5_maxpool')     # for vgg model
        
        for metric in metrics:
            
            metric_path = os.path.join(self.save_root, metric)
        
            # ----- info collection
            hyper_RSA_dict_metric = self._human_NN_RSA_load_similarity_dict(idx, metric, cell_types, used_id_nums, postfix)
               
            assemble_values = np.array([hyper_RSA_dict_metric[type_][ids]['similarity_temporal'] for type_ in cell_types for ids in used_id_nums])
            
            hyper_vmax = np.max(assemble_values[~np.isnan(assemble_values)])
            hyper_vmin = np.min(assemble_values[~np.isnan(assemble_values)])
            
            hyper_radius = hyper_vmax - hyper_vmin
                    
            # ----- plot
            fig, axes = plt.subplots(len(used_id_nums), len(cell_types), figsize=(len(cell_types)*7, len(used_id_nums)*5))
            
            row_idx = 0
            col_idx = 0
            
            for cell_type in cell_types:

                for used_id_num in used_id_nums:
                    
                    RSA_dict = hyper_RSA_dict_metric[cell_type][used_id_num]
                    
                    # ---
                    utils_similarity.fake_legend_describe_numpy(RSA_dict['similarity_temporal'], axes[row_idx, col_idx])
                    # ---
                
                    extent = [-250, 1001, -0.5, RSA_dict['similarity_temporal'].shape[0]-0.5]
                    plot_temporal_correlation(layers, fig, axes[row_idx, col_idx], RSA_dict, error_control_measure, postfix, [hyper_vmin-0.1*hyper_radius, hyper_vmax+0.2*hyper_radius], extent, False)
                    
                    axes[row_idx, col_idx].set_yticks([0, len(layers)-1-c_to_l_idx, len(layers)-1])
                    axes[row_idx, col_idx].set_yticklabels([1, f'{(c_to_l_idx+1)/len(layers):.1f}', 0], rotation='horizontal')
                    axes[row_idx, col_idx].set_title(f'{cell_type} {used_id_num}')
                    axes[row_idx, col_idx].hlines(len(layers)-1-c_to_l_idx, -250, 1001, linestyle='--', colors='red', alpha=0.75)

                    axes[row_idx, col_idx].set_ylabel('')
                    axes[row_idx, col_idx].get_legend().remove()
                    
                    row_idx += 1
                    
                    if row_idx == len(used_id_nums):
                        
                        col_idx += 1
                        row_idx = 0
                
            suptitle = f'Human-NN {self.model_structure} RSA temporal for {metric} {postfix}'
            fig.suptitle(suptitle, fontsize=22)
            
            cax = fig.add_axes([1.01, 0.1, 0.01, 0.75])
            norm = plt.Normalize(vmin=hyper_vmin-0.1*hyper_radius, vmax=hyper_vmax+0.2*hyper_radius)
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cax)
            
            logging.getLogger('matplotlib').setLevel(logging.ERROR)    
        
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                fig.tight_layout()
                
                fig.savefig(os.path.join(metric_path, f'{suptitle}_{str(len(cell_types))}_types_{postfix}'), bbox_inches='tight')
                plt.close()
            
    def _human_NN_RSA_load_similarity_dict(self, idx, metric, cell_types, used_id_nums, postfix):
        
        metric_path = os.path.join(self.save_root, metric)
        
        # ----- info collection
        hyper_RSA_dict_metric = {}
        
        for cell_type in cell_types:
        
            type_path = os.path.join(metric_path, cell_type)
            
            hyper_RSA_dict_type = {}
            
            for used_id_num in used_id_nums:
                
                id_path = os.path.join(type_path, str(used_id_num))
                
                data_path = os.path.join(id_path, f'RSA_results_{metric}_{cell_type}_{used_id_num}.pkl')
                
                if not os.path.exists(data_path):
                
                    raise RuntimeError(f'[Coderror] no results detected as [RSA_results_{metric}_{cell_type}_{used_id_num}.pkl]')
                
                else:
                    
                    RSA_dict = utils_.pickle_load(data_path)
                    
                    # ---
                    if postfix == 'neuron':
                        RSA_dict = {_:RSA_dict[_][idx, ...] for _ in RSA_dict.keys()}
                    # ---
                    
                    hyper_RSA_dict_type.update({
                        used_id_num: RSA_dict
                        })
                    
            hyper_RSA_dict_metric.update({
                cell_type: hyper_RSA_dict_type
                })
            
        return hyper_RSA_dict_metric

# ======================================================================================================================
def plot_static_correlation(ax, layers, RSA_dict, error_control_measure='sig_FDR', title=None, error_area=True, legend=True, norm_vlim:list[float]=None):
    
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
    for idx, _ in enumerate(RSA_dict[error_control_measure], 0):
         if not _:   
             ax.scatter(idx, RSA_dict['similarity'][idx], facecolors='none', edgecolors='blue')
         else:
             ax.scatter(idx, RSA_dict['similarity'][idx], facecolors='blue', edgecolors='blue')
    ax.plot(RSA_dict['similarity'], linestyle='dotted', color='deepskyblue')
    
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
    
    y_radius = np.max(RSA_dict['similarity'][~np.isnan(RSA_dict['similarity'])])
    
    if not norm_vlim:
        ax.set_ylim([np.min(RSA_dict['similarity'][~np.isnan(RSA_dict['similarity'])])-0.1*y_radius, 1.2*y_radius])
    else:
        ax.set_ylim(norm_vlim)

def plot_temporal_correlation(layers, fig, ax, RSA_dict, error_control_measure=None, title=None, vlim:list[float]=None, extent:list[float]=None, colorbar=True):
    
    if not vlim:
        cax = ax.imshow(RSA_dict['similarity_temporal'], aspect='auto', extent=extent)
        if colorbar:
            fig.colorbar(cax, ax=ax)
    else:
        cax = ax.imshow(RSA_dict['similarity_temporal'], aspect='auto', vmin=vlim[0], vmax=vlim[1], extent=extent)
        if colorbar:
            fig.colorbar(cax, ax=ax)
    
    ax.set_yticks(np.arange(RSA_dict[error_control_measure].shape[0]), list(reversed(layers)), fontsize=10)
    ax.set_xlabel('Time (ms)')
    ax.tick_params(axis='x', labelsize=12)
    ax.set_title(f'{title}')
    
    # significant correlation (Bonferroni correction)
    ax.imshow(RSA_dict['sig_temporal_Bonf'], aspect='auto',  cmap='gray', extent=extent, interpolation='none', alpha=0.25)


# ======================================================================================================================
def spearmanr_(human_DM_v, NN_DM_v):
    if np.unique(NN_DM_v).size < 2 or np.any(np.isnan(NN_DM_v)):
        rho = np.nan
    else:
        rho = spearmanr(human_DM_v, NN_DM_v, nan_policy='omit').statistic
        
    return rho


# ======================================================================================================================
def across_channel(layers):
    """
        [comment] not in use for the main process, just to test what Dr. Cao has told me
        [update] across ID has been proved not practicle
    """

    root = '/media/acxyle/Data/ChromeDownload/Identity_VGG_Feature_Original/features'
    feature_o = {}
    neuron_recover = [224,224,112,
                      112,112,56,
                      56,56,56,28,
                      28,28,28,14,
                      14,14,14,7,
                      4096,4096,50
                      ]
    
    for idx, layer in enumerate(layers):
        with open(os.path.join(root, layer+'.pkl'), 'rb') as f:
            featuremap = pickle.load(f)
        f.close()
        featuremap = torch.Tensor(featuremap)
        
        neuron = neuron_recover[idx]
        
        if featuremap.shape[1] > neuron:
            channel = featuremap.shape[1]/(neuron**2)
            
            neuron_list = [channel,neuron,neuron]
            neuron_list = [int(i) for i in neuron_list]
            
            feature_o_sub = []
        
            for img in range(featuremap.shape[0]):
                feature_strip = featuremap[img]
                feature_r = feature_strip.view(neuron_list)
                feature_o_sub.append(feature_r)
            
            feature_o_sub = torch.stack(feature_o_sub)    
            feature_o_sub = torch.stack([feature_o_sub[i*10:(i+1)*10] for i in range(50)], dim=0)
            feature_o.update({layer:feature_o_sub})
            
            
        else:
            featuremap = torch.stack([featuremap[i*10:(i+1)*10] for i in range(50)], dim=0)
            feature_o.update({layer:featuremap})
                 
            
# ======================================================================================================================
        
if __name__ == "__main__":
    
    model_name = 'vgg16'
    
    model_ = vgg.__dict__[model_name](num_classes=50)
    layers, neurons, shapes = utils_.generate_vgg_layers_list_ann(model_, model_name)
    
    root_dir = '/home/acxyle-workstation/Downloads/'

    #for monkey experiments
    #test = Selectiviy_Analysis_Correlation_Monkey(
    #    NN_root=os.path.join(root_dir, 'Face Identity Baseline'), metric='pearson',
    #    layers=layers, neurons=neurons)
    
    #test.monkey_neuron_analysis()
    
    # for human experiments 
    test = Selectiviy_Analysis_Correlation_Human(
        NN_root=os.path.join(root_dir, 'Face Identity VGG16'), 
        layers=layers, neurons=neurons)
    
    test.human_neuron_analysis()
    test.human_NN_RSA_plot_assemble()
    
