#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:47:40 2024

@author: acxyle

"""

import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from joblib import Parallel, delayed
import scipy

from statsmodels.stats.multitest import multipletests
import itertools

import utils_
from utils_ import utils_similarity

from Bio_Cell_Records_Process import Human_Neuron_Records_Process, Monkey_Neuron_Records_Process
from FSA_DRG import FSA_Gram
from FSA_Encode import FSA_Responses

  
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
            raise ValueError(f'[Coderror] Invalid parameters [{kernel}, {kwargs}]')
        
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
    
    #FIXME --- add two-tailed test
    def calculation_CKA_layer(self, layer, kernel='linear', FDR_test=True, num_perm=1000, **kwargs):    
        """
            ...
            one-tailed test
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
            

    def plot_CKA(self, fig, ax, cka_dict, error_control_measure='sig_FDR', legend=False, vlim:list[float]=None, stats=True, **kwargs):
        """
            ...
        """
        
        plot_CKA(self.layers, ax, cka_dict, vlim=vlim, legend=legend, **kwargs)
        
        if stats:
            utils_similarity.fake_legend_describe_numpy(cka_dict['cka_score'], ax, cka_dict[error_control_measure].astype(bool))


    def plot_CKA_temporal(self, fig, ax, cka_dict, extent:list[float]=None, error_control_measure='sig_temporal_Bonf', vlim:list[float]=None, **kwargs):
        """
            ...
        """

        extent = [self.ts.min()-5, self.ts.max()+5, -0.5, cka_dict['cka_score_temporal'].shape[0]-0.5]
 
        plot_CKA_temporal(self.layers, fig, ax, cka_dict, vlim=vlim, extent=extent, **kwargs)
        utils_similarity.fake_legend_describe_numpy(cka_dict['cka_score_temporal'], ax, cka_dict[error_control_measure].astype(bool))


# ----------------------------------------------------------------------------------------------------------------------
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
        
    
    def __call__(self, kernel='linear', **kwargs):
        
        cka_dict = self.calculation_CKA_Monkey(kernel, **kwargs)
        
        self.plot_CKA_Monkey(cka_dict, kernel, **kwargs)
        
        
    def calculation_CKA_Monkey(self, kernel='linear', normalize=True, FDR_test=True, **kwargs):
    
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
        
        return cka_dict


    def plot_CKA_Monkey(self, cka_dict, kernel, **kwargs):
        """
            ...
        """
        
        # --- init
        if kernel == 'rbf' and 'threshold' in kwargs:
            title_static = f"CKA score {self.model_structure} {kernel} {kwargs['threshold']}"
            title_temporal = f"CKA temporal score {self.model_structure} {kernel} {kwargs['threshold']}"
        elif kernel == 'linear':
            title_static = f'CKA score {self.model_structure} {kernel}'
            title_temporal = f'CKA temporal score {self.model_structure} {kernel}'
        else:
            raise ValueError
        
        # 1. static
        fig, ax = plt.subplots(figsize=(10,6))
        
        self.plot_CKA(fig, ax, cka_dict, **kwargs)
        ax.set_title(title_static)
        
        fig.tight_layout(pad=1)
        fig.savefig(os.path.join(self.save_root, f'{title_static}.svg'), bbox_inches='tight')   
        plt.close()
        
        # 2. temporal
        fig, ax = plt.subplots(figsize=(np.array(cka_dict['cka_score_temporal'].T.shape)/3.7))
        
        self.plot_CKA_temporal(fig, ax, cka_dict, **kwargs)
        ax.set_title(title_temporal)
        
        fig.savefig(os.path.join(self.save_root, f'{title_temporal}.svg'), bbox_inches='tight')
        plt.close()
        

# ----------------------------------------------------------------------------------------------------------------------
class CKA_Similarity_Monkey_folds(CKA_Similarity_Monkey):
    """
        this function uses 2 routes to merge the FDR results of all folds. 
        
        Route 'p' uses the mean values of all p values then conduct the FDR test again, the output is boolean
        
        Route 'sig' uses the smoothed mean values of sig results(T/F), the output is float
        
        **Example primate_config:**
            primate_config = 'Monkey'
            primate_config = 'Human/linear/qualified/50'     # 'Human/{kernel}/{used_unit_type}/{used_id_num}'
        
        **Question: what cause inf or nan CKA/RSA values? --- Zeors?
    """
    
    def __init__(self, num_folds=5, root=None, **kwargs):
        
        super().__init__(root=root, **kwargs)
        
        self.root = root
        self.num_folds = num_folds
        
        
    def __call__(self, kernel, **kwargs):
        
        CKA_dict_folds = self.calculation_CKA_Similarity_folds(kernel, **kwargs)
        
        self.plot_CKA_Similarity_folds(CKA_dict_folds, kernel, **kwargs)
        
        
    def calculation_CKA_Similarity_folds(self, kernel, **kwargs):
        
        if kernel == 'rbf' and 'threshold' in kwargs:
            cka_config = f"CKA_results_{kernel}_{kwargs['threshold']}"
            save_path = os.path.join(self.save_root, f"{cka_config}.pkl")
        elif kernel == 'linear':
            cka_config = f"CKA_results_{kernel}"
            save_path = os.path.join(self.save_root, f"{cka_config}.pkl")
        else:
            raise ValueError(f'[Coderror] Invalid parameters [{kernel}, {kwargs}]')
           
        if os.path.exists(save_path):
            
            CKA_dict_folds = utils_.load(save_path, verbose=True)
        
        else:
            
            FSA_config = self.root.split('/')[-1]
            
            CKA_dict_folds = {_ :utils_.load(os.path.join(self.root, f"-_Single Models/{FSA_config}{_}/Analysis/CKA/Monkey/{cka_config}.pkl"), verbose=False) for _ in range(self.num_folds)}
            
            # ---
            CKA_dict_folds = merge_CKA_dict_folds(CKA_dict_folds, self.layers, self.num_folds, route='p', **kwargs)
            
            # ---
            utils_.dump(CKA_dict_folds, save_path, verbose=False)
            
        return CKA_dict_folds
        
        
    def plot_CKA_Similarity_folds(self, CKA_dict_folds, kernel, **kwargs):
        
        # ----- plot
        self.plot_CKA_Monkey(CKA_dict_folds, kernel, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
class CKA_Similarity_Human(Human_Neuron_Records_Process, FSA_Gram, CKA_Similarity_base):
    """
        ...
    """
    
    def __init__(self, **kwargs):
        
        # ---
        Human_Neuron_Records_Process.__init__(self, **kwargs)
        FSA_Gram.__init__(self, **kwargs)
        
        utils_.make_dir(CKA_root:=os.path.join(self.dest, 'CKA'))
        self.save_root_primate = os.path.join(CKA_root, 'Human')
        utils_.make_dir(self.save_root_primate)
        
    
    def __call__(self, kernel='linear', used_unit_type='qualified', used_id_num=50, FDR_test=True, **kwargs):
        
        # --- additional parameters
        utils_.formatted_print(f'Used kernel: {kernel} | Used types: {used_unit_type} | Used ID: {used_id_num}')
        ...
        
        cka_dict = self.calculation_CKA_Human(kernel, used_unit_type, used_id_num, **kwargs)
        
        self.plot_CKA_Human(cka_dict, kernel, used_unit_type, **kwargs)
        
        
    def calculation_CKA_Human(self, kernel, used_unit_type, used_id_num, FDR_test=True, **kwargs):
        
        # --- init
        utils_.make_dir(save_root_cell_type:=os.path.join(self.save_root_primate, used_unit_type))
        
        self.save_root = os.path.join(save_root_cell_type, str(used_id_num))
        utils_.make_dir(self.save_root)
        
        self.used_id = self.human_corr_select_sub_identities(used_id_num)

        NN_Gram_dict = self.calculation_Gram(kernel=kernel, **kwargs)
        
        if used_unit_type == 'legacy':
            human_Gram_dict = self.human_neuron_Gram_process(kernel, 'selective', **kwargs)
            self.Gram_dict = {_: NN_Gram_dict[_]['strong_selective'][np.ix_(self.used_id, self.used_id)] for _ in NN_Gram_dict.keys()}
        else:     # --- nan_to_num for non_selective
            human_Gram_dict = self.human_neuron_Gram_process(kernel, used_unit_type, **kwargs)
            self.Gram_dict = {_: np.nan_to_num(NN_Gram_dict[_][used_unit_type][np.ix_(self.used_id, self.used_id)]) for _ in NN_Gram_dict.keys()}
            
        self.primate_Gram = human_Gram_dict['human_Gram'][np.ix_(self.used_id, self.used_id)]
        self.primate_Gram_temporal = np.array([_[np.ix_(self.used_id, self.used_id)] for _ in human_Gram_dict['human_Gram_temporal']])

        if FDR_test:
            
            assert set(['human_Gram_perm', 'human_Gram_temporal_perm']).issubset(human_Gram_dict.keys())
            
            self.primate_Gram_perm = np.array([_[np.ix_(self.used_id, self.used_id)] for _ in human_Gram_dict['human_Gram_perm']])
            self.primate_Gram_temporal_perm = np.array([np.array([__[np.ix_(self.used_id, self.used_id)] for __ in _]) for _ in human_Gram_dict['human_Gram_temporal_perm']])
        
        # ----- calculation
        cka_dict = self.calculation_CKA_Similarity(kernel=kernel, used_unit_type=used_unit_type, used_id_num=used_id_num, primate='Human', **kwargs)
        
        return cka_dict
        
    
    def plot_CKA_Human(self, cka_dict, kernel, used_unit_type, **kwargs):
        
        # --- init
        if kernel == 'rbf' and 'threshold' in kwargs:
            title_static = f"CKA score {self.model_structure} {used_unit_type} {kernel} {kwargs['threshold']}"
            title_temporal = f"CKA temporal score {self.model_structure} {used_unit_type} {kernel} {kwargs['threshold']}"
        elif kernel == 'linear':
            title_static = f'CKA score {self.model_structure} {used_unit_type} {kernel}'
            title_temporal = f'CKA temporal score {self.model_structure} {used_unit_type} {kernel}'
        else:
            raise ValueError
        
        # 1. static
        fig, ax = plt.subplots(figsize=(10,6))
        
        self.plot_CKA(fig, ax, cka_dict, **kwargs)
        ax.set_title(title_static)
        
        fig.tight_layout(pad=1)
        fig.savefig(os.path.join(self.save_root, f'{title_static}.svg'), bbox_inches='tight')   
        plt.close()
        
        # 2. temporal
        fig, ax = plt.subplots(figsize=(np.array(cka_dict['cka_score_temporal'].T.shape)/3.7))
        
        self.plot_CKA_temporal(fig, ax, cka_dict, **kwargs)
        ax.set_title(title_temporal)
        
        fig.savefig(os.path.join(self.save_root, f'{title_temporal}.svg'), bbox_inches='tight')
        plt.close()

    
class CKA_Similarity_Human_folds(CKA_Similarity_Human):
    
    def __init__(self, num_folds=5, root=None, **kwargs):
        
        super().__init__(root=root, **kwargs)
        
        self.root = root
        self.num_folds = num_folds
        
        
    def __call__(self, kernel, used_unit_type, used_id_num, **kwargs):
        
        utils_.formatted_print(f'Used kernel: {kernel} | Used types: {used_unit_type} | Used ID: {used_id_num}')
        ...

        utils_.make_dir(save_root_cell_type:=os.path.join(self.save_root_primate, used_unit_type))

        self.save_root = os.path.join(save_root_cell_type, str(used_id_num))
        utils_.make_dir(self.save_root)
        
        CKA_dict_folds = self.calculation_CKA_Similarity_folds(kernel, used_unit_type, used_id_num, **kwargs)
        
        self.plot_CKA_Similarity_folds(CKA_dict_folds, kernel, used_unit_type, used_id_num, **kwargs)
        
    
    def calculation_CKA_Similarity_folds(self, kernel, used_unit_type, used_id_num, **kwargs):
        
        if kernel == 'rbf' and 'threshold' in kwargs:
            cka_config = f"CKA_results_{kernel}_{kwargs['threshold']}"
            save_path = os.path.join(self.save_root, f"{cka_config}.pkl")
        elif kernel == 'linear':
            cka_config = f"CKA_results_{kernel}"
            save_path = os.path.join(self.save_root, f"{cka_config}.pkl")
        else:
            raise ValueError(f'[Coderror] Invalid parameters [{kernel}, {kwargs}]')
           
        if os.path.exists(save_path):
            
            CKA_dict_folds = utils_.load(save_path, verbose=True)
        
        else:
            
            FSA_config = self.root.split('/')[-1]
            
            CKA_dict_folds = {_ :utils_.load(os.path.join(self.root, f"-_Single Models/{FSA_config}{_}/Analysis/CKA/Human/{used_unit_type}/{used_id_num}/{cka_config}.pkl"), verbose=False) for _ in range(self.num_folds)}
            
            # ---
            CKA_dict_folds = merge_CKA_dict_folds(CKA_dict_folds, self.layers, self.num_folds, route='p', **kwargs)
            
            # ---
            utils_.dump(CKA_dict_folds, save_path, verbose=False)
            
        return CKA_dict_folds
        
    
    def plot_CKA_Similarity_folds(self, CKA_dict_folds, kernel, used_unit_type, used_id_num, **kwargs):
        
        # ----- plot
        self.plot_CKA_Human(CKA_dict_folds, kernel, used_unit_type, **kwargs)
            
        
# ----------------------------------------------------------------------------------------------------------------------
def merge_CKA_dict_folds(CKA_dict_folds, layers, num_folds, route='p', alpha=0.05, FDR_method='fdr_bh', **kwargs):
    
    # --- static
    cka_score_folds = [CKA_dict_folds[fold_idx]['cka_score'] for fold_idx in range(num_folds)]
    cka_score_mean = np.mean(cka_score_folds, axis=0)
    cka_score_std = np.std(cka_score_folds, axis=0)
    
    cka_score_p_folds = np.mean([CKA_dict_folds[fold_idx]['p'] for fold_idx in range(num_folds)], axis=0)
    (sig_FDR, p_FDR, alpha_Sadik, alpha_Bonf) = multipletests(cka_score_p_folds, alpha=alpha, method=FDR_method)    
    
    # --- temporal
    cka_score_temporal_folds = np.array([CKA_dict_folds[fold_idx]['cka_score_temporal'] for fold_idx in range(num_folds)])
    cka_score_temporal_mean = np.mean(cka_score_temporal_folds, axis=0)
    cka_score_temporal_std = np.std(cka_score_temporal_folds, axis=0)
    
    p_temporal = np.mean([CKA_dict_folds[fold_idx]['p_temporal'] for fold_idx in range(num_folds)], axis=0)  
    
    if route == 'p':
        
        # --- init
        p_temporal_FDR = np.zeros((len(layers), cka_score_temporal_folds.shape[-1]))     # (num_layers, num_time_steps)
        sig_temporal_FDR =  p_temporal_FDR.copy()
        sig_temporal_Bonf = p_temporal_FDR.copy()
        
        for _ in range(len(layers)):
            (sig_temporal_FDR[_, :], p_temporal_FDR[_, :], alpha_Sadik_temporal, alpha_Bonf_temporal) = multipletests(p_temporal[_, :], alpha=alpha, method=FDR_method)      # FDR
            sig_temporal_Bonf[_, :] = p_temporal_FDR[_, :]<alpha_Bonf_temporal     # Bonf correction
    
    elif route == 'sig':
        
        sig_temporal_FDR = np.mean([scipy.ndimage.gaussian_filter(CKA_dict_folds[fold_idx]['sig_temporal_FDR'], sigma=1) for fold_idx in range(num_folds)], axis=0)
        sig_temporal_Bonf = np.mean([scipy.ndimage.gaussian_filter(CKA_dict_folds[fold_idx]['sig_temporal_Bonf'], sigma=1) for fold_idx in range(num_folds)], axis=0)
        p_temporal_FDR =  np.mean([CKA_dict_folds[fold_idx]['p_temporal_FDR'] for fold_idx in range(num_folds)], axis=0)
    
    # ---
    CKA_dict_folds = {
        'cka_score': cka_score_mean,
        'cka_score_std': cka_score_std,
        
        'cka_score_perm': np.max([CKA_dict_folds[fold_idx]['cka_score_perm'] for fold_idx in range(num_folds)], axis=0),
        'p': cka_score_p_folds,
        
        'cka_score_temporal': cka_score_temporal_mean,
        'cka_score_temporal_std': cka_score_temporal_std,
        
        'cka_score_temporal_perm': np.max([CKA_dict_folds[fold_idx]['cka_score_temporal_perm'] for fold_idx in range(num_folds)], axis=0),
        'p_temporal': p_temporal,
        
        'p_FDR': p_FDR,
        'sig_FDR': sig_FDR,
        'sig_Bonf': p_FDR<alpha_Bonf,

        'p_temporal_FDR': p_temporal_FDR,
        'sig_temporal_FDR': sig_temporal_FDR,
        'sig_temporal_Bonf': sig_temporal_Bonf,

        }
    
    return CKA_dict_folds

        
# ----------------------------------------------------------------------------------------------------------------------
class CKA_Similarity_Monkey_Comparison(CKA_Similarity_Monkey_folds, CKA_Similarity_Monkey):
    """
        ...
    """
    
    def __init__(self, roots_and_models, primate_config='Monkey', route='p', **kwargs):
        
        self.roots_and_models = roots_and_models
        self.layers = layers
        self.primate_config = primate_config.replace('/', '_')
        
        if len(self.roots_and_models) == 2:
            self.save_root_ = os.path.join(roots_and_models[0][0], f"Analysis/CKA/Similarity v.s. {roots_and_models[1][0].split(' ')[-1].replace('_fold_', '')}")
        else:
            self.save_root_ = os.path.join(roots_and_models[0][0], 'Analysis/CKA/Similarity v.s. ')
            
        utils_.make_dir(self.save_root_)
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        # ---
        self.color_pool = ['blue', 'green', 'red', 'purple', 'orange', 'chocolate']
        
    
    def __call__(self, kernel, **kwargs):
    
        fig, ax = plt.subplots(figsize=(10, 6))
        
        title = []
        handles, labels = ax.get_legend_handles_labels()
        
        if len(self.roots_and_models) == 2:
            
            cka_dict = {}
        
        for idx, (root, model) in enumerate(self.roots_and_models):

            color = self.color_pool[idx]
            
            if 'fold' in root:
                
                CKA_Similarity_Monkey_folds.__init__(self, root=self.roots_and_models[idx][0], **kwargs)     # save the comparison results in the fisrt folder
                _, self.layers, self.neurons, _ = utils_.get_layers_and_units(self.roots_and_models[idx][1], 'act')
                
                CKA_dict = self.calculation_CKA_Similarity_folds(kernel, **kwargs)
                
                if len(self.roots_and_models) == 2:
                    cka_dict[idx] = CKA_dict

                _label = root.split(' ')[-1].replace('_ATan', '').replace('_C2k_fold_', '')
                title.append(_label)
                
                self.plot_CKA(fig, ax, CKA_dict, color=color, stats=False)
                
                ...
                
            else:
                
                CKA_Similarity_Monkey.__init__(self, root=self.roots_and_models[idx][0], **kwargs)     # save the comparison results in the fisrt folder
                _, self.layers, self.neurons, _ = utils_.get_layers_and_units(self.roots_and_models[idx][1], 'act')
                
                CKA_dict = self.calculation_CKA_Monkey(kernel, **kwargs)
                
                if len(self.roots_and_models) == 2:
                    cka_dict[idx] = CKA_dict
                
                _label=root.split(' ')[-1].replace('_C2k', '')
                title.append(_label)
                
                self.plot_CKA(fig, ax, CKA_dict, color=color, stats=False)
                ...
            
            solid_circle = Line2D([0], [0], marker='o', color=utils_.darken_color(utils_.color_to_hex(color)), markerfacecolor=color, markersize=5, markeredgecolor=color, linestyle='dotted', linewidth=2)

            handles.extend([solid_circle])
            labels.extend([f"{_label}: {np.mean(CKA_dict['cka_score'][CKA_dict['sig_FDR']]):.2f}"])
        
        if kernel == 'rbf' and 'threshold' in kwargs:
            postfix = f"{kernel} {kwargs['threshold']}"
        elif kernel == 'linear':
            postfix = f"{kernel}"

        #ax.set_title(title:=f'{postfix} Monkey CKA Similarity '+' v.s '.join(title))
        ax.set_title(title:=f"{postfix} Monkey CKA Similarity VGG16bn v.s. SVGG (IF)")
        
        # --- setting
        ax.legend(handles, labels, framealpha=0.5)
        # ---
        
        fig.tight_layout()
        
        fig.savefig(os.path.join(self.save_root_, f'Comparison {self.primate_config} {title}.svg'))
        
        plt.close()
        
        # -----
        if len(self.roots_and_models) == 2:
            
            diff_cka_dict = {
                'cka_score_temporal': cka_dict[1]['cka_score_temporal'] - cka_dict[0]['cka_score_temporal'],
                'p_temporal': np.min([cka_dict[1]['p_temporal'], cka_dict[0]['p_temporal']], axis=0),
                'p_temporal_FDR': np.min([cka_dict[1]['p_temporal_FDR'], cka_dict[0]['p_temporal_FDR']], axis=0),
                'sig_temporal_Bonf': cka_dict[1]['sig_temporal_Bonf'] & cka_dict[0]['sig_temporal_Bonf'],
                'sig_temporal_FDR': cka_dict[1]['sig_temporal_FDR'] & cka_dict[0]['sig_temporal_FDR'],
                }
            
            fig, ax = plt.subplots(figsize=(np.array(diff_cka_dict['cka_score_temporal'].T.shape)/3.7))
            
            self.plot_CKA_temporal(fig, ax, diff_cka_dict, **kwargs)
            ax.set_title(title:=f'Temporal {title}')
            
            fig.savefig(os.path.join(self.save_root_, f'Temporal {title}.svg'), bbox_inches='tight')
            
            plt.close()
    
    
        
class CKA_Similarity_Human_Comparison(CKA_Similarity_Human_folds, CKA_Similarity_Human):
    """
        ...
    """
    
    def __init__(self, roots_and_models, primate_config, route='p', **kwargs):
        
        self.roots_and_models = roots_and_models
        self.layers = layers
        self.primate_config = primate_config.replace('/', '_')
        
        if len(self.roots_and_models) == 2:
            self.save_root_ = os.path.join(roots_and_models[0][0], f"Analysis/CKA/Similarity v.s. {roots_and_models[1][0].split(' ')[-1].replace('_fold_', '')}")
        else:
            self.save_root_ = os.path.join(roots_and_models[0][0], 'Analysis/CKA/Similarity v.s. ')
        
        utils_.make_dir(self.save_root_)
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        # ---
        self.color_pool = ['blue', 'green', 'red', 'purple', 'orange', 'chocolate']
        
    
    def __call__(self, kernel, **kwargs):
    
        fig, ax = plt.subplots(figsize=(10, 6))
        
        title = []
        handles, labels = ax.get_legend_handles_labels()
        
        if len(self.roots_and_models) == 2:
            
            cka_dict = {}
        
        for idx, (root, model) in enumerate(self.roots_and_models):
            
            color = self.color_pool[idx]
            
            if 'fold' in root:
                
                CKA_Similarity_Human_folds.__init__(self, root=self.roots_and_models[idx][0], **kwargs)     # save the comparison results in the fisrt folder
                _, self.layers, self.neurons, _ = utils_.get_layers_and_units(self.roots_and_models[idx][1], 'act')
                
                self.save_root = os.path.join(root, f'Analysis/CKA/Human/{kernel}', self.primate_config.split('_')[-2], str(50))
                
                CKA_dict = self.calculation_CKA_Similarity_folds(kernel, used_unit_type=self.primate_config.split('_')[-2], used_id_num=50, **kwargs)
                
                if len(self.roots_and_models) == 2:
                    cka_dict[idx] = CKA_dict
                
                _label = root.split(' ')[-1].replace('_ATan', '').replace('_C2k_fold_', '')
                title.append(_label)
                
                self.plot_CKA(fig, ax, CKA_dict, color=color, stats=False)
                
                ...
                
            else:
                
                CKA_Similarity_Human.__init__(self, root=self.roots_and_models[idx][0], **kwargs)     # save the comparison results in the fisrt folder
                _, self.layers, self.neurons, _ = utils_.get_layers_and_units(self.roots_and_models[idx][1], 'act')
                
                CKA_dict = self.calculation_CKA_Human(kernel, used_unit_type=self.primate_config.split('_')[-2], used_id_num=50, **kwargs)
                
                if len(self.roots_and_models) == 2:
                    cka_dict[idx] = CKA_dict
                
                _label=root.split(' ')[-1].replace('_C2k', '')
                title.append(_label)
                
                self.plot_CKA(fig, ax, CKA_dict, color=color, stats=False)
                ...
            
            solid_circle = Line2D([0], [0], marker='o', color=utils_.darken_color(utils_.color_to_hex(color)), markerfacecolor=color, markersize=5, markeredgecolor=color, linestyle='dotted', linewidth=2)

            handles.extend([solid_circle])
            labels.extend([f"{_label}: {np.mean(CKA_dict['cka_score'][CKA_dict['sig_FDR']]):.2f}"])
        
        if kernel == 'rbf' and 'threshold' in kwargs:
            postfix = f"{kernel} {kwargs['threshold']}"
        elif kernel == 'linear':
            postfix = f"{kernel}"
            
        #FIXME
        ax.set_title(title:=f'{postfix} {self.primate_config} Human CKA Similarity '+' v.s '.join(title))
        #ax.set_title(title:=f"{postfix} Human CKA Similarity VGG16bn v.s. SVGG (LIF)")

        # --- setting
        ax.legend(handles, labels, framealpha=0.5)
        # ---
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_root_, f'Comparison {self.primate_config} {title}.svg'))
        
        plt.close()        
        
        # -----
        if len(self.roots_and_models) == 2:
            
            diff_cka_dict = {
                'cka_score_temporal': cka_dict[1]['cka_score_temporal'] - cka_dict[0]['cka_score_temporal'],
                'p_temporal': np.min([cka_dict[1]['p_temporal'], cka_dict[0]['p_temporal']], axis=0),
                'p_temporal_FDR': np.min([cka_dict[1]['p_temporal_FDR'], cka_dict[0]['p_temporal_FDR']], axis=0),
                'sig_temporal_Bonf': cka_dict[1]['sig_temporal_Bonf'].astype(bool) & cka_dict[0]['sig_temporal_Bonf'].astype(bool),
                'sig_temporal_FDR': cka_dict[1]['sig_temporal_FDR'].astype(bool) & cka_dict[0]['sig_temporal_FDR'].astype(bool),
                }
            
            fig, ax = plt.subplots(figsize=(np.array(diff_cka_dict['cka_score_temporal'].T.shape)/3.7))
            
            self.plot_CKA_temporal(fig, ax, diff_cka_dict, **kwargs)
            ax.set_title(title:=f'Temporal {title}')
            
            fig.savefig(os.path.join(self.save_root_, f'{title}.svg'), bbox_inches='tight')
            
            plt.close()
        

# ----------------------------------------------------------------------------------------------------------------------
def plot_CKA(layers, ax, cka_dict, error_control_measure='sig_FDR', title=None, error_area=True, vlim:list[float]=None, legend=False, color=None, label=None, **kwargs):
    """
        ...
    """
    color = 'blue' if color is None else color

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
        ax.fill_between(plot_x, similarity-cka_dict['cka_score_std'], similarity+cka_dict['cka_score_std'], edgecolor=None, facecolor=utils_.lighten_color(utils_.color_to_hex(color), 100), alpha=0.75)

    for idx, _ in enumerate(cka_dict[error_control_measure], 0):
         if not _:   
             ax.scatter(idx, similarity[idx], facecolors='none', edgecolors=color)
         else:
             ax.scatter(idx, similarity[idx], facecolors=color, edgecolors=color)
             
    ax.plot(similarity, linestyle='dotted', color=utils_.darken_color(utils_.color_to_hex(color)))

    ax.set_ylabel("CKA score")
    ax.set_xticks(plot_x)
    ax.set_xticklabels(layers, rotation=90, ha='center')
    ax.set_xlim([0, len(layers)-1])
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f'{title}')
    
    handles, labels = ax.get_legend_handles_labels()

    hollow_circle = Line2D([0], [0], marker='o', color=utils_.darken_color(utils_.color_to_hex(color)), linestyle='dotted', markerfacecolor='none', markersize=5, markeredgecolor=color, linewidth=1)
    solid_circle = Line2D([0], [0], marker='o', color=utils_.darken_color(utils_.color_to_hex(color)), linestyle='dotted', markerfacecolor=color, markersize=5, markeredgecolor=color, linewidth=1)

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


def plot_CKA_temporal(layers, fig, ax, cka_dict, error_control_measure='sig_temporal_Bonf', title=None, vlim:list[float]=None, extent:list[float]=None, **kwargs):
      
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
#FIXME
class CKA_base():
    
    def __init__(self, used_unit_types, **kwargs):
        
        self.used_unit_types = used_unit_types
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        ...
        
        
    def __call__(self, **kwargs):
        
        # --- 1.
        save_path = os.path.join(self.CKA_root, f'CKA {self.N1_structure} v.s. {self.N2_structure}.pkl')
        
        if os.path.exists(save_path):
            
            cka_dict = utils_.load(save_path)
            
        else:
            
            cka_dict = self.calculation_CKA(**kwargs)
        
            utils_.dump(cka_dict, save_path)
        
        # --- 2.
        for k, v in cka_dict.items():
            
            self.plot_CKA(v, k, **kwargs)
            
        # --- 3.
        fig, ax = plt.subplots()
        
        self.plot_diag_CKA(fig, ax, cka_dict, **kwargs)
        
        ax.legend()
        #ax.set_xticks(np.arange(len(self.N1_layers)))
        #ax.set_xticklabels(self.N1_layers, rotation='vertical')
        ax.set_ylim(0,1.1)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        ax.set_title(f'CKA diag {self.N1_structure} v.s. {self.N2_structure}')
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.CKA_root, f'CKA diag {self.N1_structure} v.s. {self.N2_structure}.svg'), bbox_inches='tight')
        
        plt.close()
        
    
    def plot_diag_CKA(self, fig, ax, cka_dict, used_unit_type=None, color=None, label=None, **kwargs):
        
        if used_unit_type is None:
            
            for k, v in cka_dict.items():
                
                ax.plot(np.diag(v), label=k)
        
        else:
            
            ax.plot(np.diag(cka_dict[used_unit_type]), color=color, label=label)
        
        
    def calculation_CKA(self, **kwargs):
        
        product_list = list(itertools.product(self.N1_layers, self.N2_layers))
        
        return {_type: np.array([cka(self.N1_G_dict[_[0]][_type], self.N2_G_dict[_[1]][_type]) for _ in product_list]).reshape(len(self.N1_layers), len(self.N2_layers)) for _type in tqdm(self.used_unit_types)}
        
    
    def plot_CKA(self, cka_results, _type, intensity=False, layer_ticks=True, **kwargs):
        
        if not intensity:
            
            fig, ax = plt.subplots()
           
            # === 1
            ax = plt.gcf().add_axes([0.5, 0.5, 0.5, 0.5])
            img = ax.imshow(cka_results, origin='lower', cmap='magma', aspect='auto')     # vmin=0.2, vmax=1
            
            ax.set_title(_type)
            ax.set_xlabel(f'{self.N2_structure}')
            ax.set_ylabel(f'{self.N1_structure}')
            ax.set_xticks([])
            ax.set_yticks([])
            
            fig.savefig(os.path.join(self.CKA_root, f'{_type}.svg'), bbox_inches='tight')
            plt.close()
            
        else:
            
            log_Gram_dict_N1 = utils_.load(os.path.join(self.N1_root, 'Analysis/Gram/Figures/log_Gram_dict.pkl'), verbose=False)
            log_Gram_dict_N2 = utils_.load(os.path.join(self.N2_root, 'Analysis/Gram/Figures/log_Gram_dict.pkl'), verbose=False)
            
            log_Gram_N1 = log_Gram_dict_N1[_type]
            log_Gram_N2 = log_Gram_dict_N2[_type]
            
            Intensity_dict_N1 = utils_.load(os.path.join(self.N1_root, 'Analysis/Encode/Responses/Intensity/Intensity.pkl'), verbose=False)
            units_pct_N1 = utils_.load(os.path.join(self.N1_root, 'Analysis/Encode/Responses/Intensity/units_pct.pkl'), verbose=False)
            
            Intensity_dict_N2 = utils_.load(os.path.join(self.N2_root, 'Analysis/Encode/Responses/Intensity/Intensity.pkl'), verbose=False)
            units_pct_N2 = utils_.load(os.path.join(self.N2_root, 'Analysis/Encode/Responses/Intensity/units_pct.pkl'), verbose=False)
            
            fig = plt.figure(figsize=(10, 10))
           
            ax_1 = plt.gcf().add_axes([0.5, 0.5, 0.5, 0.5])
            img = ax_1.imshow(cka_results, origin='lower', cmap='magma', aspect='auto', vmin=0., vmax=1)     # vmin=0., vmax=1
            
            ax_1.set_xticks([])
            ax_1.set_yticks([])
            
            ax_2 = plt.gcf().add_axes([0., 0.5, 0.24, 0.5])
            FSA_Responses.plot_Feature_Intensity_single(ax_2, self.N1_layers, Intensity_dict_N1, _type, units_pct_N1, direction='vertical')
            
            if layer_ticks:
                ax_2.set_yticks(np.arange(len(self.N1_layers)))
                ax_2.set_yticklabels(self.N1_layers)
            else:
                ax_2.set_yticks([])
                
            ax_2.set_ylabel(f'{self.N1_structure}', fontsize=18)
            
            ax_3 = plt.gcf().add_axes([0.25, 0.5, 0.24, 0.5])
            self.plot_log_Gram_intensity_single(fig, ax_3, self.N1_layers, _type, log_Gram_N1, direction='vertical', text=False)
            
            ax_4 = plt.gcf().add_axes([0.5, 0.25, 0.5, 0.24])
            self.plot_log_Gram_intensity_single(fig, ax_4, self.N2_layers, _type, log_Gram_N2, direction='horizontal', text=False)
            
            ax_5 = plt.gcf().add_axes([0.5, 0., 0.5, 0.24])
            FSA_Responses.plot_Feature_Intensity_single(ax_5, self.N2_layers, Intensity_dict_N2, _type, units_pct_N2, direction='horizontal')
            
            if layer_ticks:
                ax_5.set_xticks(np.arange(len(self.N2_layers)))
                ax_5.set_xticklabels(self.N2_layers, rotation='vertical')
            else:
                ax_5.set_xticks([])
                
            ax_5.set_xlabel(f'{self.N2_structure}', fontsize=18)
            
            c_ax1 = fig.add_axes([1.05, 0.1, 0.03, 0.8])
            c_b1 = fig.colorbar(img, cax=c_ax1)
            c_b1.ax.tick_params(labelsize=16)
            
            # ---
            legend_lines = [
                Line2D([0], [0], color='blue', linestyle='-', linewidth=3, label='log_Gram_value'),
                Line2D([0], [0], marker='o', markersize=8, color='red', linewidth=2, label='zero_pct'),
                Line2D([0], [0], marker='d', markersize=8, markeredgecolor='coral', color='coral', linestyle='--', linewidth=2, label='units_pct'),
                Line2D([0], [0], linewidth=2, label='units_value')
            ]
            
            fig.legend(handles=legend_lines, loc='lower left', bbox_to_anchor=(0.125, 0.2))
            
            fig.suptitle(f'{_type}', y=1.075, fontsize=24)
            #fig.tight_layout()

            fig.savefig(os.path.join(self.CKA_root, f'{_type} with intensity.svg'), bbox_inches='tight')
            plt.close()
            


# ----------------------------------------------------------------------------------------------------------------------
#FIXME --- the process is complicated
class CKA_Comparison(FSA_Gram, CKA_base):
    
    def __init__(self, N1_root, N1_model, N2_root, N2_model, used_unit_types=['qualified', 'selective', 'non_selective'], **kwargs):
        
        CKA_base.__init__(self, used_unit_types=used_unit_types, **kwargs)
        
        self.N1_root = N1_root
        self.N2_root = N2_root
        
        def _load_folds(nn_root, _model, _type='act',  norm=True, **kwargs):
            
            nn_grams = utils_.load(os.path.join(nn_root, f'Analysis/Gram/Gram_linear_norm_{norm}.pkl'))
            
            _, nn_layers, nn_neurons, _ = utils_.get_layers_and_units(_model, _type)
            
            nn_grams_dict = {layer: {_: nn_grams[layer][_] for _ in used_unit_types} for layer in nn_layers}
            
            nn_structure = nn_root.split('/')[-1].split(' ')[-1].replace('ATan_', '').replace('_C2k_fold_0', '')
            
            return nn_layers, nn_neurons, nn_grams_dict, nn_structure
        
        self.N1_layers, _, self.N1_G_dict, self.N1_structure = _load_folds(N1_root, N1_model)
        self.N2_layers, _, self.N2_G_dict, self.N2_structure = _load_folds(N2_root, N2_model)
        
        self.CKA_root = os.path.join(N1_root, f'Analysis/CKA/v.s. {self.N2_structure}')
        utils_.make_dir(self.CKA_root)
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
        
        self.CKA_root = os.path.join(N1_root, f'CKA/v.s. {self.N2_structure}')
        utils_.make_dir(self.CKA_root)
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
# local debug
if __name__ == '__main__':
    
    FSA_root = '/home/acxyle-workstation/Downloads/FSA'
    FSA_dir = 'Resnet/Resnet'
    FSA_config = 'Resnet152_C2k_fold_'
    FSA_model =  'resnet152'
    
    _, layers, neurons, shapes = utils_.get_layers_and_units(FSA_model, target_type='act')

    used_unit_types = ['qualified', 'selective', 'non_selective', 'legacy']
    

    # ----- CKA_similarity
    # --- 1.
    for _ in [0]:
        
        root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}/-_Single Models/FSA {FSA_config}{_}')
        #root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
        
        #CKA_monkey = CKA_Similarity_Monkey(root=root, layers=layers, neurons=neurons)
        #CKA_monkey(kernel='linear', normalize=True)
        #for threshold in [1.0, 10.0]:
        #    CKA_monkey(kernel='rbf', threshold=threshold, normalize=True)
        
        CKA_human = CKA_Similarity_Human(root=root, layers=layers, neurons=neurons)
        for used_unit_type in used_unit_types:
            CKA_human(kernel='linear', used_unit_type=used_unit_type)
            #for threshold in [1.0, 10.0]:
            #    CKA_human(kernel='rbf', threshold=threshold, used_unit_type=used_unit_type)
    
# =============================================================================
#     root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')
# 
#     CKA_Similarity_Monkey_folds(num_folds=5, root=root, layers=layers, neurons=neurons)(kernel='linear')
#     #for threshold in [1.0, 10.0]:
#     #    CKA_Similarity_Monkey_folds(num_folds=5, root=root, layers=layers, neurons=neurons)(kernel='rbf', threshold=threshold)
# 
#     for used_unit_type in used_unit_types:
#         CKA_Similarity_Human_folds(num_folds=5, root=root, layers=layers, neurons=neurons)(kernel='linear', used_unit_type=used_unit_type, used_id_num=50)
#         #for threshold in [1.0, 10.0]:
#         #    CKA_Similarity_Human_folds(num_folds=5, root=root, layers=layers, neurons=neurons)(kernel='rbf', threshold=threshold, used_unit_type=used_unit_type, used_id_num=50)
#     
# =============================================================================
# =============================================================================
#     roots_models = [
#         (os.path.join(FSA_root, 'VGG/VGG/FSA VGG16bn_C2k_fold_'), 'vgg16_bn'),
#         (os.path.join(FSA_root, 'VGG/SpikingVGG/FSA SpikingVGG16bn_IF_ATan_T4_C2k_fold_'), 'spiking_vgg16_bn'),
#         #(os.path.join(FSA_root, 'VGG/SpikingVGG/FSA SpikingVGG16bn_IF_ATan_T8_C2k_fold_'), 'spiking_vgg16_bn'),
#         #(os.path.join(FSA_root, 'VGG/SpikingVGG/FSA SpikingVGG16bn_IF_ATan_T16_C2k_fold_'), 'spiking_vgg16_bn'),
#         ]
# 
#     # --- CKA Similarity Comparison
#     #CKA_Similarity_Monkey_Comparison(roots_models, primate_config='Monkey')(kernel='linear')
#     #for threshold in [1.0, 10.0]:
#     #    CKA_Similarity_Monkey_Comparison(roots_models, primate_config='Monkey')(kernel='rbf', threshold=threshold)
#     
#     for _ in used_unit_types:
#         CKA_Similarity_Human_Comparison(roots_models, primate_config=f'Human/linear/{_}/50')(kernel='linear')
# =============================================================================
    
    # ----- CKA
# =============================================================================
#     used_unit_types = ['qualified', 'selective', 'strong_selective', 'weak_selective', 'non_selective']
#     
#     # --- 3.
#     CKA_Comparison(
#         #N1_root=os.path.join(FSA_root, 'VGG/VGG/FSA Baseline'), N1_model='vgg16',
#         #N2_root=os.path.join(FSA_root, 'VGG/VGG/FSA Baseline'), N2_model='vgg16',
#         N1_root=os.path.join(FSA_root, 'Resnet/Resnet/FSA Resnet152_C2k_fold_/-_Single Models/FSA Resnet152_C2k_fold_0'), N1_model='resnet152',
#         N2_root=os.path.join(FSA_root, 'Resnet/SEWResnet/FSA SEWResnet152_IF_ATan_T4_C2k_fold_/-_Single Models/FSA SEWResnet152_IF_ATan_T4_C2k_fold_0'), N2_model='sew_resnet152',
#         used_unit_types=used_unit_types
#         )(intensity=True, layer_ticks=False)
#     
# =============================================================================

    #FIXME
# =============================================================================
#     # --- 4.
#     ANN_vs_SNN = CKA_Comparison_folds(
#         N1_root=os.path.join(root_dir, 'Face Identity VGG16bn_fold_'), N1_model='vgg16_bn',
#         N2_root=os.path.join(root_dir, 'Face Identity SpikingVGG16bn_LIF_T16_CelebA2622_fold_'), N2_model='spiking_vgg16_bn',
#         used_unit_types=used_unit_types)
#     ANN_vs_SNN()
# =============================================================================
    
   
