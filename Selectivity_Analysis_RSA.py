#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:47:12 2023

@author: acxyle

[notice]
    all function with variable 'inds' and writing style like 'AaaBbbCcc' are not modified yet
    
[action required]
    simplify/optimize this code  - Jul 17, 2023
    1. save the constructed Bio data for re-use, avoid recalculation when calls
    
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


class Selectiviy_Analysis_Correlation_Monkey():

    def __init__(self, 
                 corr_root = '/media/acxyle/Data/ChromeDownload/Identity_SEWResnet50_LIF_CelebA2622_Neuron/Correlation/',
                 bio_neuron_root='/home/acxyle-workstation/Downloads/Bio_Neuron_Data/Monkey/',     # from Dr. Cao
                 layers=None, neurons=None):
        """
            [notice] only RSA between all channels from monkey IT and all units from NN
            corr_root: NN correlation matrix
            bio_neuron_root: Monkey data
        """
        
        if layers == None:
            raise RuntimeError('[Coderror] please assign proper layers')
            
        self.layers = layers
        self.neurons = neurons
        self.ts = np.arange(-50,201,10)     # target time steps
            
        self.correlation_matrix = sio.loadmat(os.path.join(corr_root,'correlation_matrix_id_all.mat'))

        self.save_root = '/'.join(['', *corr_root.split('/')[1:-2], 'RSA_monkey/'])
        utils_.make_dir(self.save_root)

        self.label = sio.loadmat(os.path.join(bio_neuron_root, 'Label.mat'))['label'].reshape(-1)
        
        monkey_neuron_data_path = os.path.join(bio_neuron_root, 'IT_FR_CA_Range70-180.mat')     # processed monkey neural data
        monkey_neuron_data = sio.loadmat(monkey_neuron_data_path)
        #print(sio.whosmat(monkey_neuron_data_path))     #  [('label', (500, 1), 'double')]
        
        monkey_dict_keys = [i for i in monkey_neuron_data.keys() if '__' not in i]
        monkey_dict = {_:monkey_neuron_data[_] for _ in monkey_dict_keys}  # rebuild the dict to store monkey IT MUA data
        
        self.FR = monkey_dict['FR']     # (3, 27911, 53)     # [comment] no method from FR to PSTH, I susspect (1) clean from 27911 to 24500 (2) operation to the first dim [question] what the first dim?
        # np.sum(np.isnan(FR)) --> 0
        
        self.meanPSTH = monkey_dict['meanPSTH']     # (500,49,53), [disordered img idx, time steps, channels], normalized value
        self.meanPSTHID = monkey_dict['meanPSTHID']     # (50,49,53), [id idx, time steps, channels], normalized value
        
        # -----
        self.meanFR = monkey_dict['meanFR']     # (53,500)
        # np.sum(np.isnan(meanFR)) --> 0
        self.meanBase = monkey_dict['meanBase']     # (53,500)
        self.meanGray = monkey_dict['meanGray'].reshape(-1)     # (53)
        #self.meanVis = monkey_dict['meanVis']     # (53,500)
        # -----
        
        self.psthTime = monkey_dict['psthTime'].reshape(-1)     # (49,)
        
    def monkey_neuron_analysis(self):
        
        print('[Codinfo] Excuting monkey neuron analysis...')
        
        #self.plot_sample_response()
        
        self.monkey_neuron_spikes_process()     # <- every time those 4 values are the same, so can be saved
        
        rID, rID_Perm, rID_T, pID_FDR, sig_T = self.representational_similarity_analysis()
        
        idx, layer_n, _ = utils_.imaginary_neurons_vgg(self.layers, self.neurons)
        rID_n = rID[idx]
        rID_Perm_n = rID_Perm[idx,:]
        pID_FDR_n = pID_FDR[idx]
        rID_T_n = rID_T[idx]
        sig_T_n = [sig_T[_] for _ in idx]

        self.plot_static_correlation(self.layers, rID, rID_Perm, pID_FDR, 0.05, 'all')
        self.plot_static_correlation(layer_n, rID_n, rID_Perm_n, pID_FDR_n, 0.05, 'neuron')
         
        self.plot_temporal_correlation(self.layers, rID_T, sig_T, 'all')
        self.plot_temporal_correlation(layer_n, rID_T_n, sig_T_n, 'neuron')
       
        #self.plot_correlation_example(uDMN, rFNID)
        
    def monkey_neuron_spikes_process(self, time_bin=10, nPerm=1000):
        """
            this function returns the correlation matrix and triangle from monkey neural responses.
            
            - Input
                psthTime: 49 time steps for PSTH from -100 ms to 380 ms
                meanPSTH: [500, 49, 53], [img, time steps, channels]
                label: label for 500 imgs
                
            - Output
                uDMN: condense form of tranformed DSM
                uDMNPerm: condense form of transformed DSM with extra dimension of permutation
                uDMN_T: condense form of transformed DSM with temporal dimension
                uDMN_TPerm: condense form of transformed DSM with temporal dimension and permutation dimension
        """
        
        print('[Codinfo] Calculating monkey neuron stats...')
        
        file_path = os.path.join(self.save_root, 'monkey_spikes_corr.pkl')
        
        if os.path.exists(file_path):
            
            results = utils_.pickle_load(file_path)
            
            self.uDMN = results['uDMN']
            self.uDMNPerm = results['uDMNPerm']
            self.uDMN_T = results['uDMN_T']
            self.uDMN_TPerm = results['uDMN_TPerm']
            
            self.IDPSTH = results['sIDPSTH']
            self.IDFR = results['sIDFR']
            
        else:
            # -----
            if time_bin == 10:
                usePSTH = self.meanPSTH[:, np.where((-50<=self.psthTime) & (self.psthTime<=200))[0], :]
            else:
                usePSTH = np.zeros((self.meanPSTH.shape[0], len(self.ts), self.meanPSTH.shape[2]))     # (500,26,53) (ID,time,neuron)
                for idx, tt in enumerate(self.ts): 
                    usePSTH[:, idx, :] = np.mean(self.meanPSTH[:, np.where(((tt-time_bin/2)<=self.psthTime) & (self.psthTime<=(tt+time_bin/2)))[0], :], axis=1)
            
            usePSTHID = np.array([np.mean(usePSTH[np.where(self.label==_)[0],:,:], axis=0) for _ in  range(1, 1+len(np.unique(self.label)))])
            
            # [notice] meanGray != np.mean(meanBase, axis=1)
            self.IDFR = np.array([np.mean(self.meanFR[:,np.where(self.label==_)[0]], axis=1)/self.meanGray for _ in range(1,51)])
            
            self.IDPSTH = np.array([np.array([usePSTHID[i,j,:]/np.mean(self.meanBase,axis=1) for j in range(usePSTH.shape[1])]) for i in range(50)])
            #self.IDPSTH = np.array([np.array([usePSTHID[i,j,:]/self.meanGray for j in range(usePSTH.shape[1])]) for i in range(50)])
            
            # for static meanFR
            self.uDMN = self.Square2Tri((1 - np.corrcoef(self.IDFR))/2)  # -> (1225,)
            self.uDMNPerm = np.array([self.Square2Tri((1-np.corrcoef(self.IDFR[np.random.permutation(self.IDFR.shape[0]),:]))/2) for _ in range(nPerm)])      # (1000,1225)
    
            # for temporal PSTH
            self.uDMN_T = np.array([self.Square2Tri((1-np.corrcoef(self.IDPSTH[:,_,:]))/2) for _ in range(self.IDPSTH.shape[1])])
            self.uDMN_TPerm = np.array([np.array([self.Square2Tri((1-np.corrcoef(self.IDPSTH[np.random.permutation(self.IDFR.shape[0]), t,:]))/2) for _ in range(nPerm)]) for t in range(self.IDPSTH.shape[1])])

            results = {
                'uDMN': self.uDMN,
                'uDMNPerm': self.uDMNPerm,
                'uDMN_T': self.uDMN_T,
                'uDMN_TPerm': self.uDMN_TPerm,
                'sIDFR': self.IDFR,
                'sIDPSTH': self.IDPSTH
                }
            
            utils_.pickle_dump(file_path, results)
        
    def rsa_computation_layer(self, layer, nPerm=1000):    

        DMIDF = self.Square2Tri((1-self.correlation_matrix[layer])/2)     # (1225,)
        
        # ----- static
        r_seg = spearmanr(self.uDMN, DMIDF, nan_policy='raise').statistic
        r_perm_seg = np.array([spearmanr(self.uDMNPerm[_,:], DMIDF, nan_policy='raise').statistic for _ in range(nPerm)])     # (1000,)
        p_seg = np.mean(r_perm_seg > r_seg)
    
        # ----- temporal
        time_steps = self.IDPSTH.shape[1]
        r_T_seg = np.zeros(time_steps)
        r_T_perm_seg = np.zeros((time_steps, nPerm))
        p_T_seg = np.zeros(time_steps)
        
        Parallel(n_jobs=int(os.cpu_count()/2))(delayed(self.rsa_computation_layer_dynamic)(DMIDF, t, r_T_seg, r_T_perm_seg, p_T_seg) for t in range(time_steps))
            
        results = [r_seg, r_perm_seg, p_seg, r_T_seg, r_T_perm_seg, p_T_seg]    
        
        return results
    
    def rsa_computation_layer_dynamic(self, DMIDF, t, r_T_seg, r_T_perm_seg, p_T_seg, nPerm=1000):
        
        r = spearmanr(self.uDMN_T[t,:], DMIDF, nan_policy='raise').statistic
        r_T_seg[t] = r
        
        r_perm = np.array([spearmanr(self.uDMN_TPerm[t, i_, :], DMIDF, nan_policy='raise').statistic for i_ in range(nPerm)])      # (1000,)
        r_T_perm_seg[t,:] = r_perm
        
        p_T_seg[t] = np.mean(r_perm > r)
        
    def representational_similarity_analysis(self, alpha=0.05, nPerm=1000):

        print('[Codinfo] Calculating RSA (1) corr scores and (2) permutation p_values...')
        
        save_path = os.path.join(self.save_root, 'RSA_results.pkl')
        
        if os.path.exists(save_path):
            [rID, rID_Perm, pID, rID_T, rID_T_Perm, pID_T, pID_FDR, sig_T] = utils_.pickle_load(save_path)
            
        else:
            
            results = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(self.rsa_computation_layer)(layer) for layer in tqdm(self.layers, desc='RSA monkey'))
            rID, rID_Perm, pID, rID_T, rID_T_Perm, pID_T = [np.array(_) for _ in list(zip(*results))]
            
            pID_FDR = multipletests(pID, alpha=alpha, method='fdr_bh')[1]     # FDR (flase discovery rate) correction
            
            nLayers = len(self.layers)
            
            pID_T_FDR = np.zeros((nLayers, self.IDPSTH.shape[1]))
            
            sig_T_FDR = [[] for _ in range(nLayers)]
            sig_T = [[] for _ in range(nLayers)]
            
            for ll in range(nLayers):
                pID_T_FDR[ll, :] = multipletests(pID_T[ll, :], alpha=alpha, method='fdr_bh')[1]      # FDR
                sig_T_FDR[ll] = np.flatnonzero(pID_T_FDR[ll, :] < alpha)
                
                sig_T[ll] = np.flatnonzero(pID_T[ll, :] < (alpha/self.IDPSTH.shape[1]))     # Bonferroni correction
            
            utils_.pickle_dump(save_path, [rID, rID_Perm, pID, rID_T, rID_T_Perm, pID_T, pID_FDR, sig_T])
        
        return rID, rID_Perm, rID_T, pID_FDR, sig_T
    
    #FIXME
    def plot_sample_response(self):
        '''
            looks no significant differences? but perhaps this section has other usages?
        '''
        # normed
        norm_factor = np.nanmean(self.meanBase, axis=1)     # (1,53)
        self.meanPSTHIDNorm = [self.meanPSTHID[:,t,:]/norm_factor for t in range(self.meanPSTHID.shape[1])]
        self.meanPSTHIDNorm = np.array(self.meanPSTHIDNorm)
        self.meanPSTHIDNorm = self.meanPSTHIDNorm.transpose((1,0,2))
        self.meanPSTHIDNorm = np.nanmean(self.meanPSTHIDNorm,axis=2)

        plt.figure(figsize=((20,10)))
        plt.imshow(self.meanPSTHIDNorm)
        plt.plot(np.full((51,), np.where(self.psthTime==-50)[0][0]),np.arange(51)-0.5,color='red',linewidth=3)
        plt.plot(np.full((51,), np.where(self.psthTime==200)[0][0]),np.arange(51)-0.5,color='red',linewidth=3)
        loc = np.arange(np.where(self.psthTime==-50)[0][0], np.where(self.psthTime==200)[0][0])
        loc = np.append(loc, max(loc)+1)
        plt.plot(loc, np.full(len(loc),-0.3), color='red', linewidth=3)
        plt.plot(loc, np.full(len(loc),49.3), color='red', linewidth=3)
        plt.colorbar()
        loc = np.where(self.psthTime%50==0)[0]
        plt.xticks(loc, list(self.psthTime[loc]), fontsize=14)
        plt.xlabel('Times(ms)', fontsize=20)
        plt.ylabel('ID',fontsize=20)
        #plt.show()
        plt.close()
        
        # un normed
        meanIDPSTHunNorm = np.nanmean(self.meanPSTHID[:,np.where(self.psthTime==-50)[0][0]:np.where(self.psthTime==200)[0][0],:],axis=2)
        plt.figure(figsize=((20,10)))
        plt.imshow(meanIDPSTHunNorm)
        plt.colorbar()
        loc = np.where((self.psthTime%50==0) & (-50 <= self.psthTime) & (self.psthTime < 200))[0]
        plt.xticks(loc-5, self.psthTime[loc])
        plt.xlabel('Times(ms)', fontsize=20)
        plt.ylabel('ID',fontsize=20)
        #plt.show()
        plt.close()
    
    def plot_static_correlation(self, layers, rID, rID_Perm, pID_FDR, alpha=0.05, title=None):
        
        print('[Codinfo] Executing static plotting...')
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')

            rID_Perm_Mean = np.mean(rID_Perm, axis=1)  # -> (num_layers,)
            rID_Perm_SD = np.std(rID_Perm, axis=1)  # -> (num_layers,)

            fig, ax = plt.subplots(figsize=(len(rID)/5*np.array([1, 0.75])))
            ax.plot(rID, 'ko-', markersize=5, linewidth=1)     # hollow circle
    
            sig_idx = np.where(pID_FDR<=alpha)[0]
            sig_rID = rID[sig_idx]
            
            ax.plot(sig_idx, sig_rID, 'ko', markersize=5, markerfacecolor='k')     # solid circle
            ax.set_ylabel("Spearman's $\\rho$")
            ax.set_xticks(range(len(rID)))
            ax.set_xticklabels(layers, rotation=90, ha='center')
            ax.set_xlim([0, len(rID)-1])
            ax.set_ylim([-0.1,1.2*np.max(rID)])
            ax.yaxis.grid(True, linestyle='--', alpha=0.5)
            ax.set_title(f'Static dynamics of correlation [{title}]')
    
            # Plot shaded error bars
            ax.plot(range(len(rID)), rID_Perm_Mean, color='blue')
            ax.fill_between(range(len(rID)), rID_Perm_Mean-rID_Perm_SD, rID_Perm_Mean+rID_Perm_SD, color='gray', alpha=0.3)
            
            plt.tight_layout(pad=1)
            plt.savefig(self.save_root+f'RSA_static_corr_{title}.eps', format='eps')   
            plt.savefig(self.save_root+f'RSA_static_corr_{title}.png', bbox_inches='tight')
            #plt.show()
            plt.close()
    
    def plot_temporal_correlation(self, layers, rID_T, sig_T, title=None):     # variable: sig_T can be 'sig_T' or 'sig_T_FDR' (former is Bonferroni)
        
        print('[Codinfo] Executing temporal plotting')
    
        logging.getLogger('matplotlib').setLevel(logging.ERROR)    
    
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            fig, ax = plt.subplots(figsize=(np.array(rID_T.T.shape)/5))
            cax = ax.imshow(rID_T, aspect='auto', extent=[self.ts.min()-5, self.ts.max()+5, -0.5, rID_T.shape[0]-0.5])
            cbar = fig.colorbar(cax, ax=ax)
            ax.set_yticks(np.arange(rID_T.shape[0]), list(reversed(layers)), fontsize=10)
            ax.set_xlabel('Time (ms)')
            ax.tick_params(axis='x', labelsize=12)
            ax.set_title(f'Temporal dynamics of correlation [{title}]')
    
            for _ in range(rID_T.shape[0]):     # significant correlation (Bonferroni correction)
                if np.any(sig_T[_]):
                    ax.plot(self.ts[sig_T[_]], [_]*len(sig_T[_]), 'r*')
             
            plt.tight_layout(pad=1)
        
            plt.savefig(self.save_root+f'RSA_dynamic_corr_{title}.eps', format='eps')     
            plt.savefig(self.save_root+f'RSA_dynamic_corr_{title}.png', bbox_inches='tight')
            #plt.show()
            plt.close()
                
    #FIXME
    def plot_correlation_example(self, uDMN, rID):
        # plot correlation for sample layer
        max_idx, max_r = max(enumerate(rID), key=lambda x: x[1])  # find the layer with strongest correlation
        layer = self.layers[max_idx]   
        DMIDF = self.Square2Tri((1-self.correlation_matrix[layer])/2)
        
        fig = plt.figure(figsize=(10,5))
        
        # plot sample PSTH
        sT = np.where(self.psthTime == 90)[0][0]
        bestTimeFR = np.mean(self.IDPSTH[:, sT, :], axis=0)
        bestNeuron = np.argsort(bestTimeFR)[::-1]
        
        iCell = bestNeuron[0]
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(self.IDPSTH[:, :, iCell], extent=[self.ts[0], self.ts[-1], 1, 50], aspect='auto')
        ax1.set_xlabel('Time(ms)')
        ax1.set_ylabel('Identity Index')
        ax1.set_title(f'IT Neuron {iCell}')
        ax1.tick_params(labelsize=12)
        
        # plot corr example
        ax2 = fig.add_subplot(1, 2, 2)
        r,p,_ = self.plotCorr(uDMN, DMIDF, 'b', ax2, 'Spearman')
        ax2.set_title(f'{layer}\nr:{r:.3f}, p:{p:.3e}')
        
        plt.tight_layout()
        #plt.show()
        plt.close()
                                                        
    def Square2Tri(self, M):
        V = Square2Tri(M)
        return V
    
    #FIXME
    # [comment] this fuction seems not very necessary because python packages can do this better than matlab
    def plotCorr(self, A, B, c='blue', isPlot=None, corrType='Pearson'):
        if corrType == 'Pearson':  # no tested
            corr_func = pearsonr
        elif corrType == 'Spearman':
            corr_func = spearmanr
        elif corrType == 'Kendalltau':  # no tested
            corr_func = kendalltau
        else:
            raise ValueError('Unknown correlation type')
    
        ind = np.where(~np.isnan(A) & ~np.isnan(B))[0]
    
        if ind.size == 0:
            r, p = np.nan, np.nan
            titleMsg = 'All NaN!'
            return r, p, titleMsg
    
        r, p = corr_func(A[ind], B[ind])
    
        titleMsg = f'r={r:.5f} p={p:.3e}'
    
        if isPlot is not None and isinstance(isPlot, matplotlib.axes.Axes):
            isPlot.plot(A[ind], B[ind], c=c, linestyle='none', marker='.', linewidth=2, markersize=2)
            P = np.polyfit(A[ind], B[ind], 1)     # polynomial fitting, degree=1

            xx = np.array([np.min(A), np.max(A)])
            
            yy = xx*P[0] + P[1]
            isPlot.plot(xx, yy,c='red', linewidth=2)
            isPlot.axis('tight')
    
        return r, p, titleMsg




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
    
    def human_neuron_spikes_process(self, sorted_ID, SelMet='IDNeuron', nPerm=1000):

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
        for _ in range(nPerm):
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
            for _ in range(nPerm):
                N = np.random.permutation(len(sorted_ID))
                permData = tmpRes[N]
                permRD = np.ma.corrcoef(np.ma.masked_invalid(permData))
                DM_IDN_T_Perm_seg.append(self.Square2Tri(permRD))
            DM_IDN_T_Perm_seg = np.array(DM_IDN_T_Perm_seg)
            DM_IDN_T_Perm.append(DM_IDN_T_Perm_seg)
            
        DM_IDN_T = np.array(DM_IDN_T).T
        DM_IDN_T_Perm = np.array(DM_IDN_T_Perm)
        
        return DM_IDN, DM_IDN_Perm, DM_IDN_T, DM_IDN_T_Perm
        
    def human_neuron_RSA_sub_ID(self, DM_IDN, DM_IDN_Perm, DM_IDN_T, DM_IDN_T_Perm, sorted_ID, used_ID, SelMet, subSelectID='_10_IDNeuron', nPerm=1000):
        
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
            for ii in range(nPerm):
                rPermID_seg.append(spearmanr(DM_IDN_Perm[ii], DMIDF, nan_policy='omit')[0])
                
            rPermID_seg = np.array(rPermID_seg)
            rPermID.append(rPermID_seg)
            
            pFNID.append(np.sum(rPermID_seg > rho) / nPerm)
            
            # for temporal info
            rFNID_T_seg = []
            pFNID_T_seg = []
            rFNIDPerm_T_seg = []
            
            for tt in range(DM_IDN_T.shape[1]):
                rho, _ = spearmanr(DM_IDN_T[:, tt], DMIDF, nan_policy='omit')
                rFNID_T_seg.append(rho)
                rFNIDPerm_seg = np.array([spearmanr(DM_IDN_T_Perm[tt, ii, :], DMIDF, nan_policy='omit')[0] for ii in range(nPerm)])
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
    
    model_ = vgg.__dict__['vgg16_bn'](num_classes=50)
    layers, neurons, shapes = utils_.generate_vgg_layers_list_ann(model_, 'vgg16_bn')
    
    root_dir = '/home/acxyle-workstation/Downloads/'

    #for monkey experiments
    #test = Selectiviy_Analysis_Correlation_Monkey(
    #    corr_root=os.path.join(root_dir, 'Identity_VGG16bn_ReLU_CelebA2622_Neuron/', 'Correlation/'), 
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
    