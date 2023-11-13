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

import spiking_vgg, spiking_resnet, sew_resnet
from spikingjelly.activation_based import surrogate, neuron, functional

from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import vgg, resnet
import utils_

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
        
        # --- all units
        self.plot_static_correlation(self.layers, RSA_dict, title=f'RSA Score {self.model_structure} (all layers) {self.metric}')
        self.plot_temporal_correlation(self.layers, RSA_dict, title=f'RSA Score temporal {self.model_structure} (all layers) {self.metric}')
        
        # --- imaginary neurons
        idx, layer_n, _ = utils_.imaginary_neurons_vgg(self.layers, self.neurons)
        
        RSA_dict_neuron = {_:RSA_dict[_][idx] for _ in RSA_dict.keys()}
        
        self.plot_static_correlation(layer_n, RSA_dict_neuron, title=f'RSA Score {self.model_structure} (neuron) {self.metric}')
        self.plot_temporal_correlation(layer_n, RSA_dict_neuron, title=f'RSA Score temporal {self.model_structure} (neuron) {self.metric}')
        
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
        corr_coef_perm = np.array([spearmanr(selectivity_analysis_calculation(self.metric, self.FR_id[np.random.permutation(self.FR_id.shape[0]),:])['vector'], NN_DM_v, nan_policy='raise').statistic for _ in range(num_perm)])
        
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

            ax.legend(handles, labels, framealpha=0.5)
            
            y_radius = np.max(RSA_dict['similarity'])
            
            if not norm_plot:
                ax.set_ylim([-0.1*y_radius, 1.2*y_radius])
            else:
                ax.set_ylim([norm_plot])
    
            plt.tight_layout(pad=1)
            plt.savefig(os.path.join(self.save_root, f'{title}.png'), bbox_inches='tight')
            plt.savefig(os.path.join(self.save_root, f'{title}.eps'), bbox_inches='tight')   
            #plt.show()
            plt.close()
    
    def plot_temporal_correlation(self, layers, RSA_dict, title=None, error_control_measure='sig_temporal_FDR', vlim:list[float]=None):
        """
            function
            
            input:
                error_control_measure: 'sig_temporal_FDR' default. Options: 'sig_temporal_Bonf'
        """
        
        print('[Codinfo] Executing temporal plotting')
    
        logging.getLogger('matplotlib').setLevel(logging.ERROR)    
    
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            fig, ax = plt.subplots(figsize=(np.array(RSA_dict['similarity_temporal'].T.shape)/5))
            
            extent = [self.ts.min()-5, self.ts.max()+5, -0.5, RSA_dict['similarity_temporal'].shape[0]-0.5]
            
            if not vlim:
                cax = ax.imshow(RSA_dict['similarity_temporal'], aspect='auto', extent=extent)
                fig.colorbar(cax, ax=ax)
            else:
                cax = ax.imshow(RSA_dict['similarity_temporal'], aspect='auto', vmin=vlim[0], vmax=vlim[1], extent=extent)
                fig.colorbar(cax, ax=ax)
            
            ax.set_yticks(np.arange(RSA_dict[error_control_measure].shape[0]), list(reversed(layers)), fontsize=10)
            ax.set_xlabel('Time (ms)')
            ax.tick_params(axis='x', labelsize=12)
            ax.set_title(f'{title}')
            
            # significant correlation (Bonferroni correction)
            ax.imshow(RSA_dict['sig_temporal_Bonf'], aspect='auto',  cmap='gray', extent=extent, interpolation='none', alpha=0.25)
                    
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
                                                        
    def Square2Tri(self, M):
        V = Square2Tri(M)
        return V
    
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




class Selectiviy_Analysis_Correlation_Human():
    # ===== under construction...
    """
        [Purpose] remove the MATLAB results denpendencies in this code
        
        Working...
        
        [Purpose] make human neuron response as an independent work rather embedded into Human_Correlation calculation
        
    """
    def __init__(self,
                 
                 # from Dr. Cao
                 #corr_root = 'FeatureCM/'
                 
                 # local model
                 corr_root='/Identity_SpikingVGG16bn_LIF_CelebA9326_Neuron/Correlation/',     # <- save folder
                 
                 root_process='/home/acxyle-workstation/Downloads/Bio_Neuron_Data/Human/osfstorage-archive-supp/',  # <- contains the processed Bio data (eg. PSTH) calculated from Matlab
                 root_data='/home/acxyle-workstation/Downloads/Bio_Neuron_Data/Human/osfstorage-archive/',  # <- contains the raw Bio data from resources, only used for [human_neuron_get_firing_rate], expand it to PSTH
                 layers=None, neurons=None):
        
        self.corr_root = corr_root
        self.save_root = '/'.join(['', *corr_root.split('/')[1:-2], 'RSA_human/'])
        utils_.make_dir(self.save_root)
        
        self.layers = layers
        self.root_process = root_process
        self.root_data = root_data
        
        # ===== under construction
        # those folders are used to store python generated files
        self.human_neuron_stats = os.path.join(self.save_root, 'human_neuron_stats/')
        utils_.make_dir(self.human_neuron_stats)
        # =====
        
        #FIXME, the documents here were generated by matlab, need to rewrite a python version later
        self.baseDir = os.path.join(self.root_process, "Spike Sorting")     # [notice]
        
        self.dataBaseDir = os.path.join(self.baseDir, 'Sorted Data')
        self.FireDir = os.path.join(self.baseDir, 'FiringRate')
        self.StatsDir = os.path.join(self.baseDir, 'StatsRes/CelebA/unNorm')
        
        self.data_set = 'CelebA'
        
        # [notice] make all the meanings clear and useful
        self.FR_time_range = [750, 1750]
        self.binW = 250
        self.preStim = 500
        self.postStim = 1000
        self.timelim = [0, 1500]
        self.timeTick = [0, 500, 1000, 1500]
        self.timeLabel = [-0.5, 0., 0.5, 1.]    
        
        # [notice] in this test version, the meaenFR document is generated by Matlab
        CelebA_meanFR_Cor_path = os.path.join(self.StatsDir, 'CelebA_meanFR_Cor.mat')     # this file is the samething to 'SortedFR_CelebA.mat'
        self.CelebA_meanFR_Cor = sio.loadmat(CelebA_meanFR_Cor_path)
    
    def human_neuron_analysis(self, used_ID='top50'):
        '''
            [task] should make it clear what is bin_size and step_size
            [warning] this is test version now, merged process here, including plot and calculation
        '''
        # [notice] this file is generated by SU_getFiringRate.m in OSF session_idx_names
        FiringRate_path = os.path.join(self.FireDir, 'FiringRate_CelebA_MTL_countRange_750-1750_Bin250.mat')
        CelebA_Base_Cor_path = os.path.join(self.StatsDir, 'CelebA_Base_Cor.mat')
        Label_path = os.path.join(self.root_process, 'Label.mat')
        
        # in fact, only need a few variables in those .mat session_idx_names
        self.FiringRate = sio.loadmat(FiringRate_path)
        #self.FiringRate.dtype.names = ('countAll','countAllEarly','countAllLate','countBaseline','countEntireTrial','meanOveralFR','PSTH')
        
        self.CelebA_Base_Cor = sio.loadmat(CelebA_Base_Cor_path)
        self.Label = sio.loadmat(Label_path)
        
        self.meanPSTH = sio.loadmat(os.path.join(self.StatsDir, 'meanPSTH250.mat'))['meanPSTH']
        self.neuron_dict = sio.loadmat(os.path.join(self.StatsDir, 'ID neuron Select MeanResponse 2SD_meanFR.mat'))
        
        # 1. raster
        #self.human_neuron_raster_plot()
        
        # 2. RSA
        self.human_neuron_RSA_analysis(used_ID=used_ID)
         
    def human_neuron_RSA_analysis(self, used_ID='top50'):
        """
        Each process consist 3 sections:
            1) generate biological neuron responses according to neuron types - [self.human_neuron_spike_process()]
            2) generate feature maps of artificial units and calculate the similarity - [self.human_neuron_RSA_sub_ID_plot()]
            3) plot according to previous outcomes - [self.human_neuron_RSA_emporal_plot()]
        """
        print(f'[Codinfo] Used ID: {used_ID}')
        
        sorted_ID = self.select_sub_identities(self.neuron_dict, subSelectID = '_10_IDNeuron', used_ID=used_ID)
        
        # 1. all neurons (both 1,577 biological neurons and [from 3 million to 50] artificial units)
        # --- generate biological neuron features based on (1) selected identities and (2) neuron types
        SelMet = 'vKeep'
        DM_IDN, DM_IDN_Perm, DM_IDN_T, DM_IDN_T_Perm = self.human_neuron_spikes_process(sorted_ID, SelMet=SelMet)
        self.human_neuron_RSA_analysis_SelMet(DM_IDN, DM_IDN_Perm, DM_IDN_T, DM_IDN_T_Perm, sorted_ID, used_ID, SelMet=SelMet)
        
        # 2. ID-selectvie neurons (155 bio neurons and [from 1.5 million to 50] artificial units)
        SelMet = 'IDNeuron'
        DM_IDN, DM_IDN_Perm, DM_IDN_T, DM_IDN_T_Perm = self.human_neuron_spikes_process(sorted_ID, SelMet=SelMet)
        self.human_neuron_RSA_analysis_SelMet(DM_IDN, DM_IDN_Perm, DM_IDN_T, DM_IDN_T_Perm, sorted_ID, used_ID, SelMet=SelMet)
        
        # 3. non-ID-selective neurons (1,422 bio neuons and [from 1.5 million to 0] artificial units)
        SelMet = 'nonIDNeuron'
        DM_IDN, DM_IDN_Perm, DM_IDN_T, DM_IDN_T_Perm = self.human_neuron_spikes_process(sorted_ID, SelMet=SelMet)
        self.human_neuron_RSA_analysis_SelMet(DM_IDN, DM_IDN_Perm, DM_IDN_T, DM_IDN_T_Perm, sorted_ID, used_ID, SelMet=SelMet)
        
    def human_neuron_RSA_analysis_SelMet(self, DM_IDN, DM_IDN_Perm, DM_IDN_T, DM_IDN_T_Perm, sorted_ID, used_ID, SelMet):
        print(f'[Codinfo] Loading Correlations of {SelMet} artificial units and calculating similarities...')
        rFNID, rFNID_T, rPermID, pFNID_FDR, pID_T_FDR, sigFN_T, sigFDR_T = self.human_neuron_RSA_sub_ID(DM_IDN, DM_IDN_Perm, DM_IDN_T, DM_IDN_T_Perm, sorted_ID, used_ID, SelMet)
        self.human_neuron_RSA_sub_ID_plot(rFNID, rPermID, pFNID_FDR, SelMet, used_ID)
        self.human_neuron_RSA_temporal_plot(rFNID_T, sigFN_T, sigFDR_T, SelMet, used_ID)
        
    def select_sub_identities(self, neuron_dict, subSelectID = '_10_IDNeuron', used_ID='top10'):
        CodeID = neuron_dict['CodeID'].reshape(-1)
        ID_neuron = neuron_dict['ID_neuron'].reshape(-1)
        
        if subSelectID == '_10_IDNeuron':     # [notice] this 'ID' represents the intersection of ANOVA and mean+2SD
            codeIDAll = []
            for i in range(len(ID_neuron)):
                tmp = CodeID[ID_neuron[i]-1].reshape(-1)
                for j in range(len(tmp)):
                    codeIDAll.append(tmp[j])
        elif subSelectID == '_10_AllNeuron':     # [warning] this 'All' represents all the encoded neuron by mean+2SD
            codeIDAll =[]
            for i in range(len(CodeID)):
                tmp = CodeID[i].reshape(-1)
                for j in range(len(tmp)):
                    codeIDAll.append(tmp[j])
        codeIDAll = np.array(codeIDAll, dtype=object)
        
        # ----- select used_ID
        if 'top' in used_ID:
            sorted_ID = [i[0] for i in self.sub_ID_selection(codeIDAll, int(used_ID[3:]))]     # self.sub_ID_selection() sorts
        elif used_ID == 'selected':
            sorted_ID = [6, 10, 14, 15, 23, 24, 28, 36, 38, 40]
        
        return sorted_ID
    
    def human_neuron_spikes_process(self, sorted_ID, SelMet='IDNeuron', num_perm=1000):

        meanFR = self.CelebA_meanFR_Cor['meanFR']
        
        if SelMet == 'IDNeuron':
            CellToAnalyze = self.neuron_dict['ID_neuron']     
        elif SelMet == 'vKeep':
            CellToAnalyze = self.CelebA_meanFR_Cor['vKeep']
        elif SelMet == 'nonIDNeuron':
            CellToAnalyze = np.setdiff1d(self.CelebA_meanFR_Cor['vKeep'], self.neuron_dict['ID_neuron'])
            
        CellToAnalyze = CellToAnalyze.reshape(-1)-1  
        label = self.Label['label'].reshape(-1)
        
        # calculate similarity matrix of FR across neurons
        # normalize firing rates
        baseline = self.CelebA_Base_Cor['meanFR']
        Data = (meanFR[CellToAnalyze] / np.nanmean(baseline[CellToAnalyze], axis=1).reshape(-1,1)).T
        DataPSTH = (self.meanPSTH[CellToAnalyze,:,:] / np.nanmean(baseline[CellToAnalyze], axis=1).reshape(-1,1,1))
        
        # [notice] this section is required to analyze the difference with above section in monkey
        IDRes = []
        IDPSTH = []
        for idd in range(len(sorted_ID)):
            idd = sorted_ID[idd]
            IDRes.append(np.nanmean(Data[label == idd], axis=0))
            IDPSTH.append(np.nanmean(DataPSTH[:,np.where(label==idd)[0],:], axis=1))
        IDRes = np.array(IDRes)
        IDPSTH = np.array(IDPSTH)
        
        # for static meanFR
        DM_IDN = self.Square2Tri(np.ma.corrcoef(np.ma.masked_invalid(IDRes)))
        DM_IDN_Perm = []
        for _ in range(num_perm):
            N = np.random.permutation(len(sorted_ID))
            DM_IDN_Perm.append(self.Square2Tri(np.ma.corrcoef(np.ma.masked_invalid(IDRes[N]))))
        DM_IDN_Perm = np.array(DM_IDN_Perm)
        
        # for temporal
        DM_IDN_T = []
        DM_IDN_T_Perm = []
        print(f'[Codinfo] Creating temporal dynamics of [{SelMet}] biological neurons...')
        for tt in tqdm(range(IDPSTH.shape[2])):
            DM_IDN_T.append(self.Square2Tri(np.ma.corrcoef(np.ma.masked_invalid(IDPSTH[:,:,tt]))))     
            
            tmpRes = IDPSTH[:, :, tt]
            DM_IDN_T_Perm_seg = []
            for _ in range(num_perm):
                N = np.random.permutation(len(sorted_ID))
                permData = tmpRes[N]
                permRD = np.ma.corrcoef(np.ma.masked_invalid(permData))
                DM_IDN_T_Perm_seg.append(self.Square2Tri(permRD))
            DM_IDN_T_Perm_seg = np.array(DM_IDN_T_Perm_seg)
            DM_IDN_T_Perm.append(DM_IDN_T_Perm_seg)
            
        DM_IDN_T = np.array(DM_IDN_T).T
        DM_IDN_T_Perm = np.array(DM_IDN_T_Perm)
        
        return DM_IDN, DM_IDN_Perm, DM_IDN_T, DM_IDN_T_Perm
        
    def human_neuron_RSA_sub_ID(self, DM_IDN, DM_IDN_Perm, DM_IDN_T, DM_IDN_T_Perm, sorted_ID, used_ID, SelMet, subSelectID='_10_IDNeuron', num_perm=1000):
        
        if SelMet == 'IDNeuron':
            DNNID = sio.loadmat(os.path.join(self.root_process, self.corr_root + 'CorMatrix_avg_ID.mat'))
        elif SelMet == 'vKeep':
            DNNID = sio.loadmat(os.path.join(self.root_process, self.corr_root + 'CorMatrix_avg.mat'))
        elif SelMet == 'nonIDNeuron':
            DNNID = sio.loadmat(os.path.join(self.root_process, self.corr_root + 'CorMatrix_avg_nonID.mat'))
        
        # calculate correlation between bio neuron and artificial unit
        rFNID = []
        rFNID_T = []
        
        pFNID = []
        pFNID_T = []
        
        rPermID = []
        rFNIDPerm_T = []
        
        for ll in tqdm(range(len(self.layers))):
            layer = self.layers[ll]
            rIDF = DNNID[layer]     # select one layer -> (50,50)
            rIDF = rIDF[np.array(sorted_ID)-1]
            rIDF = rIDF[:, np.array(sorted_ID)-1]
            DMIDF = self.Square2Tri(rIDF)  # (50,50) -> (1,225)
            
            # [important] the operation to calculate the correlation between 'bio neuron' and 'artificial unit'
            rho = spearmanr(DM_IDN, DMIDF, nan_policy='omit')[0]
            rFNID.append(rho)
            rPermID_seg = []
            for ii in range(num_perm):
                rPermID_seg.append(spearmanr(DM_IDN_Perm[ii], DMIDF, nan_policy='omit')[0])
                
            rPermID_seg = np.array(rPermID_seg)
            rPermID.append(rPermID_seg)
            
            pFNID.append(np.sum(rPermID_seg > rho) / num_perm)
            
            # for temporal info
            rFNID_T_seg = []
            pFNID_T_seg = []
            rFNIDPerm_T_seg = []
            
            for tt in range(DM_IDN_T.shape[1]):
                rho, _ = spearmanr(DM_IDN_T[:, tt], DMIDF, nan_policy='omit')
                rFNID_T_seg.append(rho)
                rFNIDPerm_seg = np.array([spearmanr(DM_IDN_T_Perm[tt, ii, :], DMIDF, nan_policy='omit')[0] for ii in range(num_perm)])
                rFNIDPerm_T_seg.append(rFNIDPerm_seg)
                pFNID_T_seg.append((rFNIDPerm_seg > rho).mean())
                
            rFNID_T_seg = np.array(rFNID_T_seg)
            rFNID_T.append(rFNID_T_seg)
            
            rFNIDPerm_T_seg = np.array(rFNIDPerm_T_seg)
            rFNIDPerm_T.append(rFNIDPerm_T_seg)
            
            pFNID_T_seg = np.array(pFNID_T_seg)
            pFNID_T.append(pFNID_T_seg)

        rFNID = np.array(rFNID)
        pFNID = np.array(pFNID)
        
        rPermID = np.array(rPermID)
        pFNID_FDR = multipletests(pFNID, alpha=0.05, method='fdr_bh')[1]
        
        rFNID_T = np.array(rFNID_T)
        pFNID_T = np.array(pFNID_T)
        
        rFNIDPerm_T = np.array(rFNIDPerm_T)
        
        sigFN_T = []
        sigFDR_T = []
        pID_T_FDR = []
        
        for ll in range(len(self.layers)):
            pID_T_FDR_seg = multipletests(pFNID_T[ll, :], alpha=0.05, method='fdr_bh')[1]
            pID_T_FDR.append(pID_T_FDR_seg)
            sigFDR_T.append(np.where(pID_T_FDR_seg < 0.05)[0])
            sigFN_T.append(np.where(pFNID_T[ll, :] < (0.05/pFNID_T.shape[1]))[0])
        
        pID_T_FDR = np.array(pID_T_FDR, dtype=object)
        sigFN_T = np.array(sigFN_T, dtype=object)
        sigFDR_T = np.array(sigFDR_T)
        
        # [notice] save data
        with open(self.save_root + f'saved_params_{SelMet}_{used_ID}.pkl', 'wb') as f:
            pickle.dump([rFNID, rFNID_T, rPermID, pFNID_FDR, pID_T_FDR, sigFN_T, sigFDR_T], f, protocol=-1)
        f.close()
        
        return rFNID, rFNID_T, rPermID, pFNID_FDR, pID_T_FDR, sigFN_T, sigFDR_T

    def human_neuron_RSA_sub_ID_plot(self, rFNID, rPermID, pFNID_FDR, SelMet, used_ID):
        
        rPermIDMean = np.mean(rPermID, axis=1)
        rPermIDSD = np.std(rPermID, axis=1)
        
        plt.figure(figsize=(6, 3))
        plt.plot(rFNID, 'k-o', markersize=10, fillstyle='none')
        plt.plot(np.where(pFNID_FDR <= 0.05)[0], rFNID[pFNID_FDR <= 0.05], 'ko', markersize=10, markerfacecolor='k')
        plt.ylabel("Spearman's R")
        plt.xticks(np.arange(len(rFNID)), self.layers, rotation='vertical')
        plot_margin = max(rFNID)-min(rFNID)
        plt.ylim(min(min(rFNID)-0.1*plot_margin, min(rPermIDMean-rPermIDSD)-0.1*plot_margin), max(max(rFNID)+0.1*plot_margin, max(rPermIDMean+rPermIDSD)+0.1*plot_margin))
        plt.tick_params(labelsize=12)
        # ---
        #FIXME
        rFNID = np.nan_to_num(rFNID)
        #pFNID = np.nan_to_num(pFNID)
        # ---
        plt.title(f'neuron: {SelMet}, ID: {used_ID} (max Corr: {np.max(rFNID):.2f})')
        
        # Plot shaded error bars
        plt.plot(range(len(rFNID)), rPermIDMean, color='blue')
        plt.fill_between(range(len(rFNID)), rPermIDMean-rPermIDSD, rPermIDMean+rPermIDSD, color='gray', alpha=0.3)
        
        plt.tight_layout(pad=1)
        plt.savefig(self.save_root+f'RSA_SpikingVGG16bn_neuron_Corr_{SelMet}_{used_ID}.png')
        plt.savefig(self.save_root+f'RSA_SpikingVGG16bn_neuron_Corr_{SelMet}_{used_ID}.eps', format='eps')  
        
        # [notice] .pdf format can avoid 'no transparency' problem
        #plt.switch_backend('pdf')
        #plt.savefig(self.save_root+f'RSA_SpikingVGG16bn_neuron_{SelMet}_{used_ID}.pdf', format='pdf')
        #plt.show()
    
    def human_neuron_RSA_temporal_plot(self, rFNID_T, sigFN_T, sigFDR_T, SelMet, used_ID):
        ts = np.arange(-250, 1001, 250)
        allTs = np.arange(-250, 1001, 50)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(rFNID_T, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.yticks(np.arange(len(self.layers)), self.layers)
        plt.xlabel('Time(ms)')
        plt.ylabel('Layers')
        plt.xticks([list(allTs).index(i) for i in ts], ts)
        for ll in range(len(self.layers)):
            if sigFN_T[ll].size != 0:
                plt.plot(sigFN_T[ll], [ll]*len(sigFN_T[ll]), 'r*')
                plt.plot(sigFDR_T[ll], [ll]*len(sigFDR_T[ll]), 'rd', alpha=0.5, markerfacecolor='None')
        plt.title(f'{SelMet} by ID (max Corr: {np.max(rFNID_T):.2f})')
        plt.tight_layout()
        
        plt.tight_layout(pad=1)
        plt.savefig(self.save_root+f'RSA_SpikingVGG16bn_neuron_temporal_{SelMet}_{used_ID}.png')
        plt.savefig(self.save_root+f'RSA_SpikingVGG16bn_neuron_temporal_{SelMet}_{used_ID}.eps', format='eps')
        #plt.show()
    
    
    def sub_ID_selection(self, input, num):     # [warning] after test of mean+2SD only, it seems not the same value
        '''
        Dr CAO provided: [6, 10, 14, 15, 23, 24, 28, 36, 38, 40]
        Calculated here: [6, 10, 14, 15, 24, 28, 30, 36, 43, 45]
        '''
        freq = Counter(input)     # [notice] looks by default the Counter() can sort?
        freq = sorted(freq.items(), key=lambda x:x[1], reverse=True)
        
        return freq[:num]
    
    def Square2Tri(self, M):
        V = Square2Tri(M)
        return V

    
    
    # ===
    # FIXME
    # [notice] consider merge this with self.human_neuron_get_firing_rate()
    
    
    # [notice] test version
    def plot_merged_(self):
        path = self.save_root
        document = [i for i in os.listdir(path) if '.pkl' in i]
        document = [document[i] for i in [5,1,3,4,0,2]]
        
        name_space = path.split('/')[5].split('_')
        #name_space = '/media/acxyle/Data/ChromeDownload/Identity_SpikingVGG16bn_LIF_CelebA2622_Neuron/RSA_human'
        print_name = '_'.join([name_space[1], name_space[2], 'ATan', name_space[3]])
        
        self.plot_human_merged_static(path, document, print_name)
        self.plot_human_merged_temporal(path, document, print_name)
        
        
    def plot_human_merged_temporal(self, path, document, print_name):
        fig, axes = plt.subplots(2, 3, figsize=((48, 20)))
        c_row, c_col = 0, 0
        for i in range(len(document)):
            
            with open(os.path.join(path, document[i]), 'rb') as f:
                data = pickle.load(f)
            f.close()
            
            [rFNID, rFNID_T, rPermID, pFNID_FDR, pID_T_FDR, sigFN_T, sigFDR_T] = data
            
            [SelMet, used_ID] = document[i].split('.')[0].split('_')[2:]
            
            layers = np.arange(len(rFNID))
            
            im = human_neuron_RSA_temporal_plot(axes[c_row, c_col], layers, rFNID_T, sigFN_T, sigFDR_T, SelMet, used_ID)
            
            c_col += 1
            if c_col == 3:
                c_row += 1
                c_col = 0
                
        cax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(im, cax=cax, extend='both')
        cbar.mappable.set_clim(0, 0.8)
        fig.tight_layout(rect=[0, 0.03, 0.95, 0.95])
        fig.suptitle(f'{print_name}', x=0.5, y=0.97, fontsize=18, ha='center')
        
        plt.savefig(self.save_root+f'RSA_{print_name}_temporal_in_all.png')
        plt.savefig(self.save_root+f'RSA_{print_name}_tenporal_in_all.eps', format='eps')
        
            
    def plot_human_merged_static(self, path, document, print_name):
        # [notice] needs to rewrite for a concise version 
        fig, axes = plt.subplots(2, 3, figsize=((48, 20)))
        
        rolling_ylim_min, rolling_ylim_max = 0, 0 
        for i in range(len(document)):
            with open(os.path.join(path, document[i]), 'rb') as f:
                data = pickle.load(f)
            f.close()
            [rFNID, rFNID_T, rPermID, pFNID_FDR, pID_T_FDR, sigFN_T, sigFDR_T] = data
            
            rPermIDMean = np.mean(rPermID, axis=1)
            rPermIDSD = np.std(rPermID, axis=1)
            plot_margin = max(rFNID)-min(rFNID)
            tmp_min = min(min(rFNID)-0.1*plot_margin, min(rPermIDMean-rPermIDSD)-0.1*plot_margin)
            if tmp_min < rolling_ylim_min:
                rolling_ylim_min = tmp_min
            tmp_max = max(max(rFNID)+0.1*plot_margin, max(rPermIDMean+rPermIDSD)+0.1*plot_margin)
            if rolling_ylim_max < tmp_max:
                rolling_ylim_max = tmp_max
        
        c_row, c_col = 0, 0
        for i in range(len(document)):
            
            with open(os.path.join(path, document[i]), 'rb') as f:
                data = pickle.load(f)
            f.close()
            
            [rFNID, rFNID_T, rPermID, pFNID_FDR, pID_T_FDR, sigFN_T, sigFDR_T] = data
            
            [SelMet, used_ID] = document[i].split('.')[0].split('_')[2:]
            
            layers = np.arange(len(rFNID))
            
            human_neuron_RSA_sub_ID_plot(axes[c_row, c_col], layers, rFNID, rPermID, pFNID_FDR, SelMet, used_ID, rolling_ylim_min, rolling_ylim_max)
            
            c_col += 1
            if c_col == 3:
                c_row += 1
                c_col = 0
                
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f'{print_name}', x=0.5, y=0.97, fontsize=18, ha='center')
        
        plt.savefig(self.save_root+f'RSA_{print_name}_static_in_all.png')
        plt.savefig(self.save_root+f'RSA_{print_name}_static_in_all.eps', format='eps')

# ======================================================================================================================
def selectivity_analysis_calculation(metric: str, feature: np.array):
    """
        based on [metric] to calculate
    """
    
    similarity_dict = {}
    
    if 'euclidean' in metric.lower():
        similarity_value = pdist(feature, 'euclidean')     # (1225,)
        similarity_matrix = squareform(similarity_value)     # (50, 50)
        
        similarity_dict.update({
            'vector': similarity_value,     # for RSA
            'matrix': similarity_matrix,     # for plot
            'contains_nan': False,     # by default, pdist() can receive null input and generate 0 rather NaN as output
            'num_units': feature.shape[1]
            })
    
    elif 'pearson' in metric.lower():
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            if feature.shape[1] == 0:
                similarity_dict = None
            
            else:
                similarity_matrix = np.corrcoef(feature)
                
                if np.any(np.isnan(similarity_matrix)):     # when detecting NaN value, i.e. the values of one class are identical
                    similarity_dict.update({'contains_nan': True})
                    similarity_matrix[np.isnan(similarity_matrix)] = 0
                    
                else:
                    similarity_dict.update({'contains_nan': False})
                    
                DSM = (1 - similarity_matrix)/2     # SM [-1, 1] -> DSM [0, 1]
                similarity_value = Square2Tri(DSM)     # (1225,)
    
                similarity_dict.update({
                    'vector': similarity_value,     # for RSA
                    'matrix': DSM,     # for plot
                    'num_units': feature.shape[1]
                    })
    
    else:
        raise RuntimeError(f'[Coderror] {metric} not supported')
    
    return similarity_dict

# ======================================================================================================================

def across_channel(layers):
    '''
    [comment] not in use for the main process, just to test what Dr. Cao has told me
    [update] across ID has been proved not practicle
    '''
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
                 
def Square2Tri(DSM):
    """
        in python, the squareform() function can convert an array to square or vice versa, 
        but need to make sure the matrix is symmetrical and 0 diagonal values
    """
    # original version
    #M_z = 1 - np.arctanh(DSM)
    #V = np.triu(M_z, k=1).T
    #V = V[V!=0]     # what if the 0 value exists in the upper triangle
    
    DSM_z = np.arctanh(DSM)
    DSM_z = (DSM_z+DSM_z.T)/2
    for _ in range(DSM.shape[0]):
        DSM_z[_,_]=0
    V = squareform(DSM_z)
    # -----
    
    return V
        
def human_neuron_RSA_temporal_plot(ax, layers, rFNID_T, sigFN_T, sigFDR_T, SelMet, used_ID):
    ts = np.arange(-250, 1001, 250)
    allTs = np.arange(-250, 1001, 50)
    
    if 'nonID' in SelMet:
        sig_tmp = np.isnan(rFNID_T)
        sig_tmp = np.array([np.where(_==True) for _ in sig_tmp], dtype=object)
        sigFN_T = np.array([np.delete(sigFN_T[i], sig_tmp[i][0]) for i in range(len(sigFN_T))], dtype=object)
        rFNID_T = np.nan_to_num(rFNID_T)
    
    im = ax.imshow(rFNID_T, aspect='auto', vmax=0.7, cmap='jet')
    ax.set_yticks(np.arange(len(layers)), layers, fontsize=14)
    ax.set_xlabel('Time(ms)', fontsize=14)
    ax.set_ylabel('Layers', fontsize=14)
    ax.set_xticks([list(allTs).index(i) for i in ts], ts, fontsize=14)
    
    for ll in range(len(layers)):
        if sigFN_T[ll].size != 0:
            ax.plot(sigFN_T[ll], [ll]*len(sigFN_T[ll]), 'r*')
            ax.plot(sigFDR_T[ll], [ll]*len(sigFDR_T[ll]), 'rd', alpha=0.5, markerfacecolor='None')
                
    ax.set_title(f'{SelMet}, ID: {used_ID} (max Corr: {np.max(rFNID_T):.2f})', fontsize=14)
    
    return im
     
def human_neuron_RSA_sub_ID_plot(ax, layers, rFNID, rPermID, pFNID_FDR, SelMet, used_ID, rolling_ylim_min, rolling_ylim_max):
    
    rPermIDMean = np.mean(rPermID, axis=1)
    rPermIDSD = np.std(rPermID, axis=1)
    
    ax.plot(rFNID, 'k-o', markersize=10, fillstyle='none')
    ax.plot(np.where(pFNID_FDR <= 0.05)[0], rFNID[pFNID_FDR <= 0.05], 'ko', markersize=10, markerfacecolor='k')
    ax.set_ylabel("Spearman's R", fontsize=14)
    ax.set_xticks(np.arange(len(rFNID)), layers, rotation='vertical', fontsize=14)

    ax.set_ylim(rolling_ylim_min, rolling_ylim_max)
    ax.tick_params(labelsize=14)
    # ---
    rFNID = np.nan_to_num(rFNID)
    #pFNID = np.nan_to_num(pFNID)
    # ---
    ax.set_title(f'neuron: {SelMet}, ID: {used_ID} (max Corr: {np.max(rFNID):.2f})', fontsize=14)
    
    # Plot shaded error bars
    ax.plot(range(len(rFNID)), rPermIDMean, color='blue')
    ax.fill_between(range(len(rFNID)), rPermIDMean-rPermIDSD, rPermIDMean+rPermIDSD, color='gray', alpha=0.3)
    
        
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
        corr_root=os.path.join(root_dir, 'Face Identity VGG16bn/', 'Analysis/0Legacy Results/Correlation/'), 
        layers=layers)
    test.human_neuron_sort_FR()     # current use MATLAB results
    #test.human_neuron_analysis(used_ID='top50')
    #test.human_neuron_analysis(used_ID='top10')
    #test.plot_merged_()
    
