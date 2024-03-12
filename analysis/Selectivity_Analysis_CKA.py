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
from Selectivity_Analysis_DR_and_SM import Selectivity_Analysis_Gram

import sys
sys.path.append('../')
import models_


class Similarity_Analysis_ANN_vs_SNN(Selectivity_Analysis_Gram):
    
    def __init__(self, ANN_root, SNN_root):
        
        Selectivity_Analysis_Gram.__init__(self, root=ANN_root, layers=layers, neurons=neurons)
        
        self.ANN_root = os.path.join(ANN_root, 'Features')
        self.ANN_layers, self.ANN_neurons, self.ANN_shapes = utils_.get_layers_and_units('spiking_vgg16_bn', 'act')
        self.ann_structure = ANN_root.split('/')[-1].split(' ')[-1]
        
        self.SNN_root = os.path.join(SNN_root, 'Features')
        self.SNN_layers, self.SNN_neurons, self.SNN_shapes = utils_.get_layers_and_units('spiking_vgg16_bn', 'act')
        self.snn_structure = SNN_root.split('/')[-1].split(' ')[-1]
        #self.snn_structure = 'A2S_Baseline(T64_all)'
        
        # -----
        self.cka_analysis(kernel='linear')
        
        for threshold in [0.5, 1.0, 2.0, 10.0]:
            self.cka_analysis(kernel='rbf', threshold=threshold)
        
        
    def cka_analysis(self, kernel='linear', **kwargs):
        """
            prepare to add unequal length
            
            [question] similarity shift?
        """
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        assert len(self.ANN_layers)==len(self.SNN_layers)   
        
        # ----- parallel process
        product_list = list(itertools.product(self.ANN_layers, self.SNN_layers))
        
        if kernel == 'linear':
            gram = gram_linear
        elif kernel =='rbf':
            gram = gram_rbf
        
        pl = Parallel(n_jobs=12)(delayed(self.cka_calculation_single)(_[0], _[1], gram, **kwargs) for _ in tqdm(product_list, desc='CKA'))

        cka_results = np.array(pl).reshape(15,15)
        
        if 'threshold' in kwargs:
            title = f"CKA score {kernel} {kwargs['threshold']}" 
        elif kernel == 'linear':
            title = f'CKA score {kernel}'
        else:
            raise ValueError
        
        fig, ax = plt.subplots(figsize=(10,10))
        
        img = ax.imshow(cka_results, origin='lower', cmap='magma')
        ax.set_xlabel(f'{self.snn_structure}')
        ax.set_ylabel(f'{self.ann_structure}')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(img, shrink=0.75, pad=0.04)
        plt.show()

        print('6')
        
    def cka_calculation_single(self, ANN_layer, SNN_layer, gram, **kwargs):
        
        sorted_idx = utils_.lexicographic_order(self.num_classes)     # correct labels
        
        ANN_feature = utils_.load(os.path.join(self.ANN_root, f'{ANN_layer}.pkl'), verbose=False)
        ANN_feature = np.mean(ANN_feature.reshape(50, 10, -1), axis=1)
        ANN_feature = self.restore_order(ANN_feature, sorted_idx)     # (50, num_units)
        
        SNN_feature = utils_.load(os.path.join(self.SNN_root, f'{SNN_layer}.pkl'), verbose=False)
        SNN_feature = np.mean(SNN_feature.reshape(50, 10, -1), axis=1)
        SNN_feature = self.restore_order(SNN_feature, sorted_idx)     # (50, num_units)
        
        return cka(gram(ANN_feature, **kwargs), gram(SNN_feature, **kwargs))


class Selectivity_Analysis_CKA_Base():
    
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
    
    def cka_analysis(self, kernel='linear', FDR_test=True, alpha=0.05, FDR_method='fdr_bh', save=True, **kwargs):
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
            
            print('...')
            
            cka_dict = utils_.load(save_path, verbose=False)
            
        else:
            
            print('...')
            
            #pl = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(self.cka_analysis_layer)(layer, kernel=kernel, FDR_test=FDR_test, **kwargs) for layer in tqdm(self.layers, desc='CKA monkey'))
            
            # --- sequential for debug
            pl = []
            for layer in tqdm(self.layers, desc='CKA monkey'):
                pl.append(self.cka_analysis_layer(layer, kernel=kernel, FDR_test=FDR_test, **kwargs))
                
            if not FDR_test:
                
                pl_k = ['cka_fr', 'cka_psth']
                
                assert list(pl[0].keys()) == pl_k
                
                extracted_data = [np.array([_[__] for _ in pl]) for __ in pl_k]

                cka_dict = dict(zip(pl_k, extracted_data))
            
            else:
                
                pl_k = ['cka_fr', 'cka_fr_perm', 'p_perm', 'cka_psth', 'cka_psth_perm', 'p_temporal_perm']
            
                assert list(pl[0].keys()) == pl_k
                
                cka_score, cka_score_perm, p, cka_score_temporal, cka_score_temporal_perm, p_temporal = [np.array([_[__] for _ in pl]) for __ in pl_k]
                
                # --- static
                (sig_FDR, p_FDR, alpha_Sadik, alpha_Bonf) = multipletests(p, alpha=alpha, method=FDR_method)    # FDR (flase discovery rate) correction
                sig_Bonf = p_FDR<alpha_Bonf
                
                # --- temporal
                time_steps = self.primate_Gram_temporal.shape[0]
                
                p_temporal_FDR = np.zeros((len(self.layers), time_steps))     # (num_layers, num_time_steps)
                sig_temporal_FDR =  np.zeros_like(p_temporal_FDR)
                sig_temporal_Bonf = np.zeros_like(p_temporal_FDR)
                
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
    
    
    def cka_analysis_layer(self, layer, kernel='linear', FDR_test=True, num_perm=1000, **kwargs):    
        """
            [task]
                add the Gram computation for NN, avoid double computation
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
            

    #FIXME - make it useful for both Human and Monkey
    def plot_static_correlation(self, cka_dict, kernel='linear', error_control_measure='sig_FDR', error_area=True, legend=False, vlim:list[float]=None, **kwargs):
        """
            this function plot static CKA score and save
            
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
                title = f"CKA score {self.model_structure} {kernel} {kwargs['threshold']}"
            elif kernel == 'linear':
                title = f'CKA score {self.model_structure} {kernel}'
            else:
                raise ValueError

            fig, ax = plt.subplots(figsize=(10,6))
            
            plot_static_correlation(self.layers, ax, cka_dict, title=title, vlim=vlim, legend=legend)
            utils_similarity.fake_legend_describe_numpy(cka_dict['cka_score'], ax, cka_dict[error_control_measure].astype(bool))

            plt.tight_layout(pad=1)
            plt.savefig(os.path.join(self.save_root, f'{title}.svg'), bbox_inches='tight')   
            plt.close()
            

    def plot_temporal_correlation(self, cka_dict, extent:list[float]=None, error_control_measure='sig_temporal_Bonf', vlim:list[float]=None, kernel='linear', **kwargs):
        """
            function
            
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
            
            plot_temporal_correlation(self.layers, fig, ax, cka_dict, title=title, vlim=vlim, extent=extent)
            utils_similarity.fake_legend_describe_numpy(cka_dict['cka_score_temporal'], ax, cka_dict[error_control_measure].astype(bool))

            plt.savefig(os.path.join(self.save_root, f'{title}.svg'))     

            plt.close()


class Selectivity_Analysis_CKA_Monkey(Monkey_Neuron_Records_Process, Selectivity_Analysis_Gram, Selectivity_Analysis_CKA_Base):
    """
        this function inherit from Bio Cell and NN Unit process, aim to produce the RSA results for certain Model     
    
        only RSA between all channels from monkey IT and all units from NN
        
        [question]:
            CKA is invariant to orthogonal transformation like permutation?
        
            input: 
                NN_root: path of NN correlation matrix
                
    """
    #FIXME
    def __init__(self, 
                 NN_root,
                 layers=None, neurons=None, seed=6):
        
        Monkey_Neuron_Records_Process.__init__(self, seed=seed)
        
        # --- DSM, need to change
        Selectivity_Analysis_Gram.__init__(self, root=NN_root, layers=layers, neurons=neurons)
        
        # FIXME ----- this dual entrance needs to be simplfied and sealed in one function 
        self.root = os.path.join(NN_root, 'Features/')     # <- folder for feature maps, which should be generated before analysis

        # -----       
        self.dest = os.path.join(NN_root, 'Analysis/')    # <- folder for analysis results
        utils_.make_dir(self.dest)
        
        if layers == None:
            raise RuntimeError('[Coderror] please assign proper layers')
            
        self.layers = layers
        self.neurons = neurons
        
        self.CKA_root = os.path.join(self.dest, 'CKA')
        utils_.make_dir(self.CKA_root)
        utils_.make_dir(os.path.join(self.CKA_root, 'Monkey'))
        

    #FIXME --- change this to the entry of CKA
    def monkey_neuron_analysis(self, kernel, FDR_test=True, **kwargs):
        """
            [task] change the metric to model_selection(): (1) linear; (2) rbf
        """
        # --- init
        self.save_root = os.path.join(self.CKA_root, 'Monkey')
        utils_.make_dir(self.save_root)
        
        # --- monkey init
        monkey_Gram_dict = self.monkey_neuron_Gram_process(kernel=kernel, **kwargs)
        
        self.primate_Gram = monkey_Gram_dict['monkey_Gram']
        self.primate_Gram_temporal = monkey_Gram_dict['monkey_Gram_temporal']
        
        if FDR_test:
            
            assert set(['monkey_Gram_perm', 'monkey_Gram_temporal_perm']).issubset(monkey_Gram_dict.keys())
            
            self.primate_Gram_perm = monkey_Gram_dict['monkey_Gram_perm']
            self.primate_Gram_temporal_perm = monkey_Gram_dict['monkey_Gram_temporal_perm']
        
        # --- NN init
        self.Gram_dict = {k:v['qualified'] for k,v in self.NN_unit_Gram_process(kernel=kernel, **kwargs).items()}
        
        # ----- calculation
        cka_dict = self.cka_analysis(kernel=kernel, FDR_test=FDR_test, **kwargs)
        
        # ----- plot
        self.plot_static_correlation(cka_dict, kernel=kernel, **kwargs)
        self.plot_temporal_correlation(cka_dict, kernel=kernel, **kwargs)
        


# ----------------------------------------------------------------------------------------------------------------------
# FIXME --- the entire design needs to be simplified and also simplify the RSA process
class Selectivity_Analysis_Correlation_Human(Human_Neuron_Records_Process, Selectivity_Analysis_Gram, Selectivity_Analysis_CKA_Base):
    """
        ...
    """
    
    def __init__(self,
                 NN_root=None, 
                 layers=None, neurons=None):
        
        # ---
        Human_Neuron_Records_Process.__init__(self, )
        Selectivity_Analysis_Gram.__init__(self, root=NN_root, layers=layers, neurons=neurons)
        
        self.root = os.path.join(NN_root, 'Features/')
        self.dest = os.path.join(NN_root, 'Analysis/')
        utils_.make_dir(self.dest)
        
        self.layers = layers
        self.neurons = neurons

        self.CKA_root = os.path.join(self.dest, 'CKA')
        utils_.make_dir(self.CKA_root)
        
        self.save_root_primate = os.path.join(self.CKA_root, 'Human')
        utils_.make_dir(self.save_root_primate)
        
        
    # FIXME
    def human_neuron_analysis(self, kernel='linear', used_cell_type='qualified', used_id_num=50, FDR_test=True, **kwargs):
        """
            ...
        """
        # --- additional parameters
        utils_._print(f'Used kernel: {kernel} | Used types: {used_cell_type} | Used ID: {used_id_num}')
        ...
        
        utils_.make_dir(save_root_kernel:=os.path.join(self.save_root_primate, f'{kernel}'))
        utils_.make_dir(save_root_cell_type:=os.path.join(save_root_kernel, used_cell_type))

        self.save_root = os.path.join(save_root_cell_type, str(used_id_num))
        utils_.make_dir(self.save_root)
        
        # --- init
        self.used_id = self.human_corr_select_sub_identities(used_id_num)

        NN_Gram_dict = self.NN_unit_Gram_process(kernel=kernel, used_cell_type=used_cell_type, **kwargs)
        
        if used_cell_type == 'legacy':
            human_Gram_dict = self.human_neuron_Gram_process(kernel, 'selective', **kwargs)
            self.Gram_dict = {_: NN_Gram_dict[_]['strong_selective'][np.ix_(self.used_id, self.used_id)] for _ in NN_Gram_dict.keys()}
        else:
            human_Gram_dict = self.human_neuron_Gram_process(kernel, used_cell_type, **kwargs)
            self.Gram_dict = {_: NN_Gram_dict[_][used_cell_type][np.ix_(self.used_id, self.used_id)] for _ in NN_Gram_dict.keys()}
            
        self.primate_Gram = human_Gram_dict['human_Gram'][np.ix_(self.used_id, self.used_id)]
        self.primate_Gram_temporal = np.array([_[np.ix_(self.used_id, self.used_id)] for _ in human_Gram_dict['human_Gram_temporal']])

        if FDR_test:
            
            assert set(['human_Gram_perm', 'human_Gram_temporal_perm']).issubset(human_Gram_dict.keys())
            
            self.primate_Gram_perm = np.array([_[np.ix_(self.used_id, self.used_id)] for _ in human_Gram_dict['human_Gram_perm']])
            self.primate_Gram_temporal_perm = np.array([np.array([__[np.ix_(self.used_id, self.used_id)] for __ in _]) for _ in human_Gram_dict['human_Gram_temporal_perm']])
        
        # ----- calculation

        cka_dict = self.cka_analysis(kernel=kernel, used_cell_type=used_cell_type, used_id_num=used_id_num, **kwargs)
        
        # ----- plot
        self.plot_static_correlation(cka_dict, kernel=kernel, used_cell_type=used_cell_type, used_id_num=used_id_num, **kwargs)

        self.plot_temporal_correlation(cka_dict, kernel=kernel, used_cell_type=used_cell_type, used_id_num=used_id_num, **kwargs)

            
    
    
    
# ----------------------------------------------------------------------------------------------------------------------
def plot_static_correlation(layers, ax, cka_dict, error_control_measure='sig_FDR', title=None, error_area=True, vlim:list[float]=None, legend=False, **kwargs):
    """
        #TODO 
        add the std error area
    """
    
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
        ax.fill_between(plot_x, similarity-cka_dict['cka_score_std'], similarity+cka_dict['cka_score_std'], edgecolor=None, facecolor='skyblue', alpha=0.75)

    for idx, _ in enumerate(cka_dict[error_control_measure], 0):
         if not _:   
             ax.scatter(idx, similarity[idx], facecolors='none', edgecolors='blue')
         else:
             ax.scatter(idx, similarity[idx], facecolors='blue', edgecolors='blue')
             
    ax.plot(similarity, linestyle='dotted', color='deepskyblue')

    ax.set_ylabel("CKA score")
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


def plot_temporal_correlation(layers, fig, ax, cka_dict, error_control_measure='sig_temporal_Bonf', title=None, vlim:list[float]=None, extent:list[float]=None):
      
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

        input = scipy.ndimage.gaussian_filter(input, sigma=1)
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
    
    # FIXEME --- need to upgrade to merged model
    # significant correlation (Bonferroni/FDR)
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
"""
    below functions directly from the CKA colab tutorial
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
  return scaled_hsic / (normalization_x * normalization_y)

# ------


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    layers, neurons, shapes = utils_.get_layers_and_units('spiking_vgg16_bn', target_layers='act')

    root_dir = '/home/acxyle-workstation/Downloads/'
    
    CKA_monkey = Selectivity_Analysis_CKA_Monkey(NN_root=os.path.join(root_dir, 'Face Identity SpikingVGG16bn_IF_T4'), layers=layers, neurons=neurons)
    CKA_monkey.monkey_neuron_analysis(kernel='linear')
    for threshold in [0.5, 1.0, 2.0, 10.0]:
        CKA_monkey.monkey_neuron_analysis(kernel='rbf', threshold=threshold)
    
# =============================================================================
#     CKA_human = Selectivity_Analysis_Correlation_Human(NN_root=os.path.join(root_dir, 'Face Identity SpikingVGG16bn_IF_T4'), layers=layers, neurons=neurons)
#     
#     for used_cell_type in ['qualified', 'selective', 'non_selective', 'legacy']:
#         for used_id_num in [50, 10]:
#             CKA_human.human_neuron_analysis(kernel='linear', used_cell_type=used_cell_type, used_id_num=used_id_num)
#             for threshold in [0.5, 1.0, 2.0, 10.0]:
#                 CKA_human.human_neuron_analysis(kernel='rbf', threshold=threshold, used_cell_type=used_cell_type, used_id_num=used_id_num)
# =============================================================================
    
    #ANN_vs_SNN = Similarity_Analysis_ANN_vs_SNN(
    #ANN_root=os.path.join(root_dir, 'Face Identity SpikingVGG16bn_IF_T4'),
    #SNN_root=os.path.join(root_dir, 'Face Identity SpikingVGG16bn_LIF_T4'))
