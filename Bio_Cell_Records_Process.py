#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 13:35:41 2023

@author: acxyle-workstation

    [Oct 19, 2023] Most of the confusing strcutures regarding number/index come from MATLAB design. One can change that to
    python dict for the entire code if necessary as I managed part of it in raster plot
    
"""



import os
import pandas as pd

import scipy.stats as stats
import warnings
import logging
import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from joblib import Parallel, delayed
from matplotlib import gridspec
from scipy.stats import gaussian_kde, norm, skew, lognorm, kstest
from scipy.spatial.distance import pdist, squareform

from scipy.integrate import quad

import utils_
import utils_similarity

#FIXME - move the DM process from RSA to here
# =============================================================================
class Human_Neuron_Records_Process():
    """
        function: 
        
        1) convert raw records to response map;
    
        2) plot raster plot;
        
        3) analysis detailed neuron features  
        
    """
    
    def __init__(self,
                 root='/home/acxyle-workstation/Downloads/Bio_Neuron_Data/Human/',  
                 ):

        self.root_process = os.path.join(root, 'osfstorage-archive-supp/')     # <- contains the processed Bio data (eg. PSTH) calculated from Matlab
        self.root_data = os.path.join(root, 'osfstorage-archive/')      # <- contains the raw Bio data from resources, only used for [human_neuron_get_firing_rate], expand it to PSTH
        
        self.human_neuron_stats = os.path.join(root, 'human_neuron_stats/')
        utils_.make_dir(self.human_neuron_stats)
        
        # -----
        self.baseDir = os.path.join(self.root_process, "Spike Sorting")     # [notice]
        self.dataBaseDir = os.path.join(self.baseDir, 'Sorted Data')
        self.FireDir = os.path.join(self.baseDir, 'FiringRate')
        self.StatsDir = os.path.join(self.baseDir, 'StatsRes/CelebA/unNorm')
        # -----
        
        # ↓ raw timestamps were stored by .mat format from OSF database, data is sorted on 1-based order of MATLAB
        self.Spikes = sio.loadmat(os.path.join(self.root_data, 'SingleNeuron/Data/Spikes.mat'))  
        # -----
        
        # -----
        self.FaceImageIndex = np.array(pd.read_csv(os.path.join(self.root_data, 'Stimuli/FaceImageIndex.csv')))[:, 0]
        # -----
        
        # [notice] seems only used in plot_raster
        self.FR_time_range = [750, 1750]
        self.binW = 250
        self.preStim = 500
        self.postStim = 1000
        self.timelim = [0, 2000]

        
    # ===== module 1, obtain response map
    # ----- 1. calculate FR from raw records
    # FIXME
    def human_neuron_sort_FR(self, reject_rate:float=0.15, data_type:str='default', data_set:str='CelebA'):
        """
            currently the results is not identical with the MATLAB version, need to fix, and upgrade
            -----
            this function calculates the sorted mean firing rates (entire trial: 0ms - 2000ms) and qualified cells
            
            this function use reject_rate (default: 0.15) and experiment performance to filter out unwanted cells. After 
            that, for each neuron, retrieve session_idx first, then based on it's back_id, the imgs used for one-back trial, to 
            remove repeated neuron responses. Third, place all responses into a (cell_num, img_num) matrix. Finally, make a 
            correction due to experiment errors.
            
            [note] the output meanFR is NOT disordered while the original MATLAB code provides disordered meanFR.
            
            for session with code != 500 means the images displayed in the experiments are not standard.
            for session 8, with 711 'code', 67 'back_id', 644 non_back images, in which 500 standard images and 144 
            repeated images (each repeated 1 time), they took the average value of repeated experiment
        
            input: 
                
            - reject_rate (default=0.15), suggests the threshold to determine the qualified cells
            - data_type (default='default'). 'base': firing rate from 250ms to 500ms; 'trial' firing rate from 0ms to 
            2000ms; 'defualt': firing rate from 750ms to 1750ms
            - deta_set (default='CelebA'), currently only 'CelebA'
            
            output:
            
            - meanFR: response map with shape (num_cells, num_imgs), sorted based on ID
            - cells: 1,577 qualified cell idces from all 2082 cells
            
            involved variables:
            
            - FR_stats: preliminary firing rates from original experimental records, simply count spikes in given time period
            - beh_stats: original experimental settings and logs, indicates the relationships between images and records
            - adjust_idx [AdjustInd]: corrected indeces of displayed images in experiments due to image errors
            - displayed_image_sequence [Code]: the displayed image sequence of each session, used to build relationship 
            between image and neuron records
            ...
            
        """
        
        meanFR_path = os.path.join(self.human_neuron_stats, f'meanFR_{data_type}.pkl')
        
        if os.path.exists(meanFR_path):
            
            meanFR_dict = utils_.pickle_load(meanFR_path)
            
        else:
            
            print('[Codinfo] Calculating sorted meanFR...')
            
            if data_set == 'CelebA':
                
                # ----- obtain firing rate (FR) and peri-stimulus histogram (PSTH)
                FR_stats = self.human_neuron_get_firing_rate()
                
                if data_type == 'base':
                    FR_list = [FR_stats[_]['spike_count_250_500'] for _ in range(len(FR_stats))]
                elif data_type == 'trial':
                    FR_list = [FR_stats[_]['spike_count_0_2000'] for _ in range(len(FR_stats))]
                elif data_type == 'default':
                    FR_list = [FR_stats[_]['spike_count'] for _ in range(len(FR_stats))]
                else:
                    raise ValueError(f'[Codinfo] data_type [{data_type}] is invalid')
                    
                FR_PSTH_list = [FR_stats[_]['PSTH_250'] for _ in range(len(FR_stats))]
        
                # ----- obtain behavior data
                beh_stats = self.human_neuron_get_beh()
                behavior = beh_stats['beh']
                
                # --- 1.1 1st condition, firing rate
                cell_reject = np.array([]).astype(int)
                for cell_idx in range(len(FR_stats)):  
                    if np.nanmean(FR_stats[cell_idx]['spike_count_0_2000']) < reject_rate:  
                        cell_reject = np.append(cell_reject, cell_idx)
                
                # --- 1.2 2nd condition, manually "exclude sessions 12(only has 117 trials) and 18(only 1 neuron kept and patients did not pay attention)"
                neuron_session_idces = self.Spikes['vCell'].reshape(-1)-1     # 1-based (MATLAB) -> 0-based (Python)
                exclude_cell = np.array([index for index, value in enumerate(neuron_session_idces) if value == 11 or value == 17])
                cell_reject = np.union1d(cell_reject, exclude_cell)
         
                qualified_cells = np.setdiff1d(np.arange(len(FR_stats)), cell_reject)     # qualified neurons with 2 conditions
                
                # ----- build [im_code] img_idces and img_idces_new dict
                # --- 1.2.1 old (wrong) img_idces and id_img dict
                CelebA_img_idces = sio.loadmat(os.path.join(self.root_data, 'SingleNeuron/Code/CelebA_Image_Code.mat'))
                img_idces = CelebA_img_idces['im_code'].reshape(-1)     # [idx as right number: ID]
                adjust_idx = CelebA_img_idces['AdjustInd'].reshape(-1).astype(np.int16) - 1   # [idx as right number: wrong number]
                
                self.img_idces_dict = {_:img_idces[_] for _ in range(len(img_idces))}     # {right number: ID}
                self.adjust_idx_dict = {adjust_idx[_].astype(int): _ for _ in range(len(adjust_idx)) if adjust_idx[_] > -1}     # {wrong number: right number}
                
                # --- 1.2.2 new img_idces and id_img dict
                CelebA_img_idx_new = sio.loadmat(os.path.join(self.root_data, 'SingleNeuron/Code/CelebA_Image_Code_new.mat'))
                img_idces_new = CelebA_img_idx_new['im_code'].reshape(-1)     # img_idx linked ID, 50 IDs in total
                
                id_img_idces = np.array([_.reshape(-1)-1 for _ in CelebA_img_idx_new['id_code'].reshape(-1) ])     # each ID contains 10 imgs
                
                self.img_idces_new_dict = {_:img_idces_new[_] for _ in range(len(img_idces_new))}     # {img number: ID}
                self.id_img_idces_dict = {_:id_img_idces[_] for _ in range(len(id_img_idces))}     # {ID: 10 img numbers}     0-based
                
                # ----- start meanFR calculation [notice] this process is for all 2082 cells, including the filtered cells
                meanFR = np.full((len(FR_stats), 500), np.nan)     # empty response map waited to receive values
                meanFR_PSTH = np.full((len(FR_stats), 500, 31), np.nan)
                
                for cell_idx in tqdm(range(len(FR_stats))):     # for each neuron
                    
                    # --- init
                    FR = FR_list[cell_idx]
                    FR_PSTH = FR_PSTH_list[cell_idx]
                
                    # --- 2.1 retrive session_idx
                    session_idx = neuron_session_idces[cell_idx]     # get session idx
                
                    displayed_image_sequence = behavior[session_idx]['code'].copy()     # delete back_id in img_list, 1-based
                    back_id = behavior[session_idx]['back_id'].copy()     # 
                    
                    # --- 2.2 remove unwanted records
                    displayed_image_sequence = np.delete(displayed_image_sequence, back_id) - 1     # img idx, 0-based

                    FR = np.delete(FR, back_id)   # remove back_id in neuron response
                    FR_PSTH = np.delete(FR_PSTH, back_id, axis=0)
                    
                    # --- manually set the responses of error image as np.nan
                    if session_idx < 10:
                        error_image_idces = []
                        for _ in [51, 52, 53]:
                            error_image_idces.append(np.where(img_idces==_)[0].item())     # [77, 97, 122], 0-based
                            
                        for _ in error_image_idces:
                            error_position = [_ for _ in np.where(displayed_image_sequence==_)[0]]
                            for __ in error_position:
                                FR[__] = np.nan
                                FR_PSTH[__, :] = np.full(FR_PSTH.shape[1], np.nan)
                    
                    # --- 2.3 sort displayed_image_sequence and corresponding responses
                    sort_idx = np.argsort(displayed_image_sequence)     # the index of 'Code' in ascending order
                    sorted_image_idces = displayed_image_sequence[sort_idx]     
                    FR = FR[sort_idx]     # sorted FR based on the index of 'Code', make the responses align with the image idces
                    FR_PSTH = FR_PSTH[sort_idx, :]
                    
                    # --- 3 correction and store into FR matrix
                    if session_idx < 10:     # for the first 10 sessions
                        
                        adjustFR = np.full(500, np.nan)  
                        adjustFR_PSTH = np.full((500, 31), np.nan)
                        
                        if len(displayed_image_sequence) != 500:     # 3: (498); 7 (645); 8: (499)
    
                            new_FR = np.full(500, np.nan)  
                            new_FR_PSTH = np.full((500, 31), np.nan)
                            
                            for _ in range(500):     # for each img_idx, wrong_label
                                used_imgs = np.where(sorted_image_idces == _)[0]     # for most of the cases, 1 record for 1 img
                                if len(used_imgs) != 0:
                                    new_FR[_] = np.mean(FR[used_imgs])   
                                    new_FR_PSTH[_, :] = np.mean(FR_PSTH[used_imgs, :], axis=0)
                                    
                            for _ in range(500):    
                                if adjust_idx[_] >= 0:    
                                    adjustFR[_] = new_FR[adjust_idx[_]]     # put the firing rate of wrong labeled img to the right position
                                    adjustFR_PSTH[_, :] = new_FR_PSTH[adjust_idx[_], :]
                        else:
                            
                            for _ in range(500):
                                if adjust_idx[_] >= 0:
                                    adjustFR[_] = FR[adjust_idx[_]]
                                    adjustFR_PSTH[_, :] = FR_PSTH[adjust_idx[_], :]
                        
                        meanFR[cell_idx, :] = adjustFR  # Note: Python uses 0-based indexing
                        meanFR_PSTH[cell_idx, :, :] = adjustFR_PSTH
                        
                    else:     # for the rest 30 sessions
                        
                        adjustFR = np.full(500, np.nan)
                        adjustFR_PSTH = np.full((500, 31), np.nan)
                    
                        if len(displayed_image_sequence) != 500:     # 11: (161); 13: (452)

                            for _ in range(500):
                                used_imgs = np.where(sorted_image_idces == _)[0]
                                if len(used_imgs) != 0:
                                    adjustFR[_] = np.mean(FR[used_imgs])
                                    adjustFR_PSTH[_, :] = np.mean(FR_PSTH[used_imgs, :], axis=0)
                                    
                            meanFR[cell_idx, :] = adjustFR
                            meanFR_PSTH[cell_idx, :, :] = adjustFR_PSTH
                            
                        else:
                            meanFR[cell_idx, :] = FR
                            meanFR_PSTH[cell_idx, :, :] = FR_PSTH
                               
                # --- plot the raw response map, follow the order of cell and img along x and y axis
                # --- [notice] please refer [meanFR figs] to check the figs
                #plt.rcParams.update({"font.family": "Times New Roman"})
                #fig, ax = plt.subplots(figsize=(20,10))
                #meanFR_ = meanFR.copy()
                #c = ax.imshow(meanFR_.T, aspect='auto', cmap='turbo')
                #fig.colorbar(c)
                #ax.set_ylabel('img idx', fontsize=18)
                #ax.set_xlabel('cell idx', fontsize=18)
                #ax.set_title('raw response map', fontsize=20)
                #title = 'raw response map (with nan values)'
                #plt.savefig(f'{title}.png', bbox_inches='tight')
                #plt.savefig(f'{title}.pdf', format='pdf', bbox_inches='tight')
                
                # --- 4. repair because of image errors
                print('[Codinfo] Start images repair...')
                # --- 4.1. face 121 is a mis-identified photo of identity 6, replace the FR to average FR of other 9 faces
                meanFR[:, 120] = np.nanmean(meanFR[:, self.id_img_idces_dict[5][:-1]], 1)
                meanFR_PSTH[:, 120, :] = np.nanmean(meanFR_PSTH[:, self.id_img_idces_dict[5][:-1], :], 1)
                
                # --- 4.2. replace the problemd face with mean of that ID for the problemed ID (only for the first 381 neurons),in
                # which another 2 faces(2 trials) were mis-identified, results are similar when replaced with Nan
                defective_ids = [17, 39]
                for _ in defective_ids:
                     defective_imgs = self.id_img_idces_dict[_][-1]
                     meanFR[:381, defective_imgs] = np.nanmean(meanFR[:381, self.id_img_idces_dict[_][:-1]], 1)
                     meanFR_PSTH[:381, defective_imgs, :] = np.nanmean(meanFR_PSTH[:381, self.id_img_idces_dict[_][:-1], :], 1)
                
                # --- 4.3 update
                # [notice] consider to remove session 12 because here's a lot of nan values for session 12 (cell 415 to 432) (1-based)
                # [notice] until now the feature map is still based on img_idx, i.e. the adjacent 10 imgs does not belong
                # to one class, below section sorts feature map based on ID. The finally output has similiar order as NN feature map
                
                # ----- [notice] for the MATLAB version feturemap, ignore below opration
                # --- 5. sort the feature map based on ID
                print('[Codinfo] Start sorting based on ID...')
                for cell_idx in tqdm(range(meanFR.shape[0]), desc='Sorting'):
                    
                    FR = meanFR[cell_idx, :].copy()
                    FR_PSTH = meanFR_PSTH[cell_idx, :, :].copy()
                    
                    FR_ID = []
                    FR_PSTH_ID = []
                    
                    for ID in range(50):
                        sorted_FR = [FR[_] for _ in sorted(self.id_img_idces_dict[ID])]
                        FR_ID.append(sorted_FR)
                        
                        sorted_FR_PSTH = [FR_PSTH[_, :] for _ in sorted(self.id_img_idces_dict[ID])]
                        FR_PSTH_ID.append(sorted_FR_PSTH)
                    
                    FR_ID = np.array(FR_ID).reshape(-1)
                    meanFR[cell_idx, :] = FR_ID
                    
                    FR_PSTH_ID = np.array(FR_PSTH_ID).reshape(-1, 31)
                    meanFR_PSTH[cell_idx, :, :] = FR_PSTH_ID
                    
                #-------------------------------------------------------------------------------------------------------
                meanFR_dict = {
                    'meanFR': meanFR,
                    'meanFR_PSTH': meanFR_PSTH,
                    'qualified_cells': qualified_cells,
                    }
                
                utils_.pickle_dump(meanFR_path, meanFR_dict)
                    
            else:
                
                raise RuntimeError('[Coderror] data_set not a_licable')
                
                # [notice] legacy from MATLAB code, not in use for current task
                #FR = np.array([])
                #for _ in range(1, imgNum+1):
                #    defective_imgs = np.where(sorted_image_idces ==_)
                #    if not defective_imgs:
                #        FR = np.concatenate((FR, FR[defective_imgs]))
                #meanFR[cell_idx-1,:] = FR;

        return meanFR_dict
        
    def human_neuron_get_firing_rate(self, time_window=250, time_step=50):  
        """
            the saved FR_stats['FR'] is identical with FR.m of source MATLAB code
            -----
            'Spikes.mat': b'MATLAB 5.0 MAT-file, Platform: MACI64, Created on: Thu Dec 30 12:11:01 2021'
            
            contains combined neurons from all session_idces. For each neuron:
    
            'timestampsOfCellAll' - timestamps (in μs)
            'vCell' - session_idx ID
            'vCh' - channel ID
            'vClusterID' - cluster ID
            'areaCell' - recording brain area
            
            note that these variables were all matched.
            
            There are three variables that describe spike sorting quality. 
            
            'IsoDist' - the isolation distance value for each cluster (i.e., a single neuron). 
            
            'statsSNR' contains 6 columns: 
                1 - session_idx ID, 
                2 - channel ID, 
                3 - cluster ID, 
                4 - the average signal-to-noise ratio (SNR), 
                5 - inter-spike intervals (ISI) that are below 3 ms,
                6 - peak SNR. 
            
            'statsProjectAll' contains 5 columns: 
                1 - session_idx ID, 
                2 - channel ID, 
                3 and 4 - IDs for a cluster pair, 
                5 - pairwise distance between two clusters.
            
            -----
            the name of saved variables does not include the parameters configuration
        """
        
        file_path = os.path.join(self.human_neuron_stats, 'FR_stats.pkl')
        
        if os.path.exists(file_path):
            FR_stats = utils_.pickle_load(file_path)
            
        else:
            
            print('[Codinfo] Calculating original firing rates...')
            
            time_stamps_all_cells = [_.reshape(-1) for _ in self.Spikes['timestampsOfCellAll'].reshape(-1)]
            
            sessions = self.get_session_idces()     # <- stores time periods of each trial (each session_idx)
            session_idx_dir = os.path.join(self.root_data, 'Events Files')     # <- store timestamps for each responses, self.root_data: osfstorage-archive
            
            # consider add a code like self.get_periods() to make the code more clear and easy to read
            all_periods = []     # provides the time range of one single trial
            for session_idx in tqdm(range(len(sessions)), desc='Load session_idces'):
                # ↓ 'periods' contains 3 columns: trial indices, timestamps to 500 ms before stumuli onset, timestamps to 1500 ms after stimuli onset
                periods = sio.loadmat(os.path.join(session_idx_dir, sessions[session_idx]+'.mat'))['periods']     
                all_periods.append(periods)
            
            PSTH_start = 500 - time_window     # 500 ms is image onset
            PSTH_end = 2000 - time_window     # 2000 ms is the end of one trial
            
            num_frames = int((PSTH_end - PSTH_start)/time_step + 1)     # <- number of time bins
            neuron_session_idces = self.Spikes['vCell'].reshape(-1)     # <- session_idx ID
            
            # [notice] 'threads' are 10 times faster than 'loky', see https://joblib.readthedocs.io/en/latest/parallel.html
            FR_stats = {}
            
            # -----
            pl = Parallel(n_jobs=os.cpu_count(), prefer="threads")(delayed(calculate_firing_rate)(
                    time_stamps_all_cells[i], neuron_session_idces[i], all_periods, time_window, num_frames, PSTH_start, time_step) 
                    for i in tqdm(range(len(time_stamps_all_cells)), desc='Cell stats calculation'))
            # -----
            
            for i in tqdm(range(len(time_stamps_all_cells)), desc='Cell stats dict'):
                FR_stats[i] = pl[i]
                
            utils_.pickle_dump(file_path, FR_stats)
        
        return FR_stats
        
    def human_neuron_FR_stats_plot(self, init:float=0.15):
        
        FR_stats = self.human_neuron_get_firing_rate()
        
        feature = [FR_stats[_]['spike_count_0_2000'] for _ in range(len(FR_stats))]
        feature = np.array([np.mean(_[~np.isnan(_)]) for _ in feature])
        
        scale_factor = (np.max(feature)-np.min(feature))
        
        init_th = init/scale_factor
        feature = feature/scale_factor
        
        fig = self.neuron_FR_stats_plot(feature=feature, init_th=init_th, scale_factor=scale_factor)
        
        plt.tight_layout()
        
        fig.savefig(os.path.join(self.human_neuron_stats, 'neuron PDF.png'))
        fig.savefig(os.path.join(self.human_neuron_stats, 'neuron PDF.eps'))
    
    @staticmethod
    def neuron_FR_stats_plot(model_structure:str='human MTL', target:str='cell', feature:np.array=None, init_th=None, scale_factor:float=1.):
        """
            this function plots the log gaussian hist and PDF of human neuron data
        """
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 18})
        
        if feature.ndim == 1:
            feature = feature
        elif feature.ndim ==2 and feature.shape[0]==500:     # assume the shape is (num_samples, num_features)
            feature = np.mean(feature, axis=0)
        else:
            raise RuntimeError(f'[Coderror] invalid feature shape {feature.shape}')
        
        if np.any(feature<0):
            
            kde = gaussian_kde(feature)     # kde estimation
            
            x = np.linspace(np.min(feature), np.max(feature)*1.1, 1000)
            y_kde = kde(x)
            
            feature_mean = np.mean(feature)
            feature_std = np.std(feature)
            y_norm = stats.norm.pdf(x, feature_mean, feature_std)
    
            pct = []
            for _ in range(1,4):
                pct1 = quad(kde, -np.inf, feature_mean-_*feature_std)[0]*100
                pct2 = quad(kde, feature_mean+_*feature_std, np.inf)[0]*100
                pct.append([pct1, pct2])
            
            fig, ax = plt.subplots(1, 2, figsize=(20,10))
            
            (hist_pct, hist_x, _) = ax[0].hist(feature, bins=100, density=True)
            ylim_max = np.ceil(np.max(hist_pct)/10)*10
            
            ax[0].plot(x, y_kde, linestyle='--', linewidth=2, color='orange', label='gaussian_kde')
            ax[0].plot(x, y_norm, linestyle='--', linewidth=2, color='red', label='gaussian')
            
            ax[0].set_ylim([0, ylim_max])
            ax[0].set_title('hist of original data', fontsize=24)
            
            ylim_max_auto = ax[0].get_ylim()[1]
            
            for _ in range(3):
                
                ax[0].vlines(feature_mean-(_+1)*feature_std, 0, ylim_max_auto, linestyle='dotted', color='gold', label=f'p < mean-{(_+1)}std: {pct[_][0]:.2f}%')
                ax[0].vlines(feature_mean+(_+1)*feature_std, 0, ylim_max_auto, linestyle='dotted', color='purple', label=f'p > mean+{(_+1)}std: {pct[_][1]:.2f}%')
            
            ax[0].legend(framealpha=0.5)
            
            feature_log = np.log10(feature[feature>0])
            
            kde = gaussian_kde(feature_log) 
            
            x = np.linspace(np.min(feature_log), (np.max(feature_log)+2)*1.2, 1000)
            y_kde = kde(x)
            
            feature_log_mean = np.mean(feature_log)
            feature_log_std = np.std(feature_log)
            y_norm = stats.norm.pdf(x, feature_log_mean, feature_log_std)
            
            x_radius = max(feature_log_mean-np.min(feature_log), np.max(feature_log)-feature_log_mean)
            
            ax[1].hist(feature_log, bins=100, density=True)
            ax[1].set_xlim([feature_log_mean-x_radius, feature_log_mean+x_radius])
            
            ax[1].plot(x, y_kde, linestyle='--', linewidth=2, color='orange', label='gaussian_kde')
            ax[1].plot(x, y_norm, linestyle='--', linewidth=2, color='red', label='gaussian')
            
            ax[1].set_title(f'log10 hist and gaussian kde excluse 0 ({(feature.size-feature_log.size)/feature.size*100:.2f}%)', fontsize=24)
            
            ax[1].legend(framealpha=0.5)
            
            fig.suptitle(f'{model_structure} {target} PDF', y=1, fontsize=28)
            
        else:

            fig, ax = plt.subplots(1, 2, figsize=(20,10))
            
            (hist_pct, hist_x, _) = ax[0].hist(feature, bins=100, density=True)
            ylim_max = np.ceil(np.max(hist_pct)/10)*10
            
            ax[0].set_ylim([0, ylim_max])
            
            y_array = np.arange(0, ylim_max+1)
            y_array = np.array([_ for _ in y_array if _%10==0])
            
            ax[0].set_title('hist of original data', fontsize=24)
     
            # ---
            feature_log = np.log10(feature[feature>0])
            
            kde = gaussian_kde(feature_log)
            
            x = np.linspace(np.min(feature_log), (np.max(feature_log)+1)*1.2, 1000)     # extend the x_lim and truncated later
            y_kde = kde(x)
            
            feature_log_mean = np.mean(feature_log)
            feature_log_std = np.std(feature_log)
            y_norm = stats.norm.pdf(x, feature_log_mean, feature_log_std)
    
            pct = []
            for _ in range(1,4):
                pct1 = quad(kde, -np.inf, feature_log_mean-_*feature_log_std)[0]*100
                pct2 = quad(kde, feature_log_mean+_*feature_log_std, np.inf)[0]*100
                pct.append([pct1, pct2])   
     
            ax[1].hist(feature_log, bins=100, density=True)
            ax[1].plot(x, y_kde, linestyle='--', color='orange', linewidth=2,  label='gaussian_kde')
            ax[1].plot(x, y_norm, linestyle='--', color='red', linewidth=2,  label='gaussian')
            ax[1].set_title(f'log10 hist and gaussian kde excluse 0 ({(feature.size-feature_log.size)/feature.size*100:.2f}%)', fontsize=24)
            
            ylim_max_auto = ax[1].get_ylim()[1]
            
            ax[1].vlines(np.mean(feature_log), 0, ylim_max_auto, color='red', label='mean')
            
            for _ in range(3):
                
                ax[1].vlines(feature_log_mean-(_+1)*feature_log_std, 0, ylim_max_auto, linestyle='dotted', color='gold', label=f'p < mean-{(_+1)}std ({pct[_][0]:.2f}%)')
                ax[1].vlines(feature_log_mean+(_+1)*feature_log_std, 0, ylim_max_auto, linestyle='dotted', color='purple', label=f'p > mean+{(_+1)}std ({pct[_][1]:.2f}%)')
    
            x_radius = max(feature_log_mean-np.min(feature_log), np.max(feature_log)-feature_log_mean)
            
            ax[1].set_xlim([feature_log_mean-x_radius, feature_log_mean+x_radius])
            ax[1].set_ylim([0, ylim_max_auto])
            
            ax[1].legend(framealpha=0.5)
            
            if init_th is not None:
                
                pct_init = quad(kde, -np.inf, np.log10(init_th))[0]*100
                ax[0].vlines(init_th, 0, ylim_max, linestyle='--', color='red', alpha=0.5, label=f'manual value of {init_th*scale_factor:.2f} ({pct_init:.2f}%)')
                ax[0].legend(framealpha=0.5)
                ax[1].vlines(np.log10(init_th), 0, ylim_max_auto, linestyle='dotted', color='red', label=f'manual value of {init_th*scale_factor:.2f} ({pct_init:.2f}%)')
                ax[1].fill_between(x, y_kde, where= (x < np.log10(init_th)), color='gray', alpha=0.5)
            
            fig.suptitle(f'{model_structure} {target} PDF', y=1, fontsize=28)
        
        return fig
    
    def human_neuron_get_beh(self, ):
        """
            Behavioral data for each patient is stored in the ‘behaviorData/p*WV/CelebA/Sess*/’ directory (* indicates the 
            session number). 
    
            For most sessions, patients completed 550 trials without interruption. Starting from patient 16WV, 555 trials 
            were completed as we added 5 more one-back trials. In some sessions, a few trials were removed manually due to 
            interruptions. 
            
            Note that P9WV Sess2 was restarted due to a short interruption to the experiment; and the behavioral data 
            should be concatenated. Specifically, in the first part (before interruption) we recorded 161 trials and in 
            the second part (after experiment restart) we recorded the entire 500 trials. 
            
            We pooled the data for further analysis to keep as many trials as possible. 
            
            Behavior data of each session was stored in the .mat file. 
            
            - ‘iT’ - timing parameters for each trial. 
            - ‘code’ - stimulus presentation order. 
            - ‘back_id’ - indices for one-back trials (img labels, not idx of .code). 
            - ‘vResp’ - if the patient responded (i.e., pressed the space bar) in the one-back trial, the value is 1 for this trial in the variable ; otherwise, the value is 0. 
            - ‘RT’ - response time.
        """
        
        print('[Codinfo] Calculating neuron behavior stats...')
        
        file_path = os.path.join(self.human_neuron_stats, 'beh_stats.pkl')
        
        if os.path.exists(file_path):
            beh_stats = utils_.pickle_load(file_path)
            
        else:
        
            beh_dict = []
            Acc = []
            
            sessions = self.get_session_idces()     # <- stores time periods of each trial
            
            for session in tqdm(sessions, desc='Processing session behavior data'):     # for each session
                
                participant_idx = session.split('_')[0]  # 'pXXX'
                
                if 'Sess' in session:
                    session_idx = session.split('_')[-1]
                else:
                    session_idx = 'Sess' + session[session.find('S')+1]
                    
                behavior_data_dict = self.GetBehavior3(participant_idx, session_idx)
                beh_dict.append(behavior_data_dict)  
                
                Acc.append(np.sum(behavior_data_dict['vResp'])/len(behavior_data_dict['back_id']))     # [notice] looks like some records have significantly low accuracy
                
            beh_stats = {
                'beh': beh_dict,
                'Acc': Acc
                }
            
            utils_.pickle_dump(file_path, beh_stats)
            
        return beh_stats
            
    def GetBehavior3(self, participant_idx, session_idx, data_set='CelebA'):     
        """
            this function wraps the self.behavior_data_dict_correlation() which loads the original behavior data and transforms to python dict
                
            input: 
                participant_idx: idx of 12 participants
                session_idx: idx of sessions of each participant
                
            output:
                behavior_data_dict: behavior_data_dict with needed variables of each session
        """
        
        behavior_directory = os.path.join(self.root_data, 'behaviorData/Archive/')
        path = os.path.join(behavior_directory, participant_idx, data_set, session_idx)
        
        behavior_data_dict = self.behavior_data_dict_correction(path)

        if len(behavior_data_dict['iT']['TRIAL_START']) < len(behavior_data_dict['code']):     # no such case in current experiments
            raise RuntimeError(f'Error Detected: {path}')    
            
            # ----- legacy from MATLAB code
            #behavior_data_dict['code'] = behavior_data_dict['code'][:len_iT]
            #behavior_data_dict['back_id'] = behavior_data_dict['back_id'][np.where(behavior_data_dict['back_id']<=(len_iT+1))[0]]
            #behavior_data_dict['vResp'] = behavior_data_dict['vResp'][:len_iT]
            #behavior_data_dict['RT'] = behavior_data_dict['RT'][:len_iT]
            #behavior_data_dict['vCorr'] = behavior_data_dict['vCorr'][:len_iT]

        return behavior_data_dict
    
    def behavior_data_dict_correction(self, path:str):
        """
            input: path of the behavior record of one session
            
            output: python dict of needed variables of one session
        """
        # --- 1. obtain all records for 1 session, usually, one session only has one record
        # [p9WV Sess2] is the only one has 2 records due to interruption, for current task
        inputfiles = [f for f in os.listdir(path) if f.endswith('.mat')]
        
        # --- 2. list the needed variables from all 97 variables 
        variables_list = ['vResp', 'vCorr', 'RT', 'code']
        behavior_data_dict = {}
        # -----
        
        if 'p9WV' in path and 'Sess2' in path:     # concatenate 2 records, [Exception 1, P9WV_Sess2]
            
            if inputfiles != ['CelebA_p9WV_26Oct2019163716.mat', 'CelebA_p9WV_26Oct2019164627.mat']:     # integrity check
                raise RuntimeError('[Coderror] P9WV_Sess2 records incorrect')
            
            for filename in inputfiles:
                
                data_dict = self.behavior_data_dict_format_transformation(os.path.join(path, filename))
                
                if filename == 'CelebA_p9WV_26Oct2019163716.mat':     # cut-off
                    
                    record_length = len(data_dict['resp_log'])     # 161
                    back_id_length = np.where(data_dict['back_id']<record_length)[0].size     # 16
                    
                    data_1 = {variable: data_dict[variable][:record_length] for variable in variables_list}
                    data_1['iT'] = {_: data_dict['iT'][_][:record_length] for _ in data_dict['iT'].keys()}
                    data_1['back_id'] = data_dict['back_id'][:back_id_length]
                    
                    behavior_data_dict.update(**data_1)
                    
                elif filename == 'CelebA_p9WV_26Oct2019164627.mat':     # change back_id
                
                    data_2 = {variable: data_dict[variable] for variable in variables_list}
                    data_2['iT'] = {_: data_dict['iT'][_] for _ in data_dict['iT'].keys()}
                    data_2['back_id'] = data_dict['back_id']
                    
                    data_2['back_id'] = data_dict['back_id'] + record_length
                
            # --- concatenate
            for key in [_ for _ in list(behavior_data_dict.keys()) if 'iT' not in _]:
                behavior_data_dict[key] = np.concatenate((behavior_data_dict[key], data_2[key]))
                
            for key in behavior_data_dict['iT'].keys():
                behavior_data_dict['iT'][key] = np.concatenate((behavior_data_dict['iT'][key], data_2['iT'][key]))
                
        else:
            
            filename = inputfiles[0]
            data_dict = self.behavior_data_dict_format_transformation(os.path.join(path, filename))

            # --- processes from matlab source code
            if filename == 'CelebA_p10WV_02Feb2020152353.mat':     # cut-off, [Exception 2, P10WV_Sess3]
                record_length = len(data_dict['resp_log'])     # 182
                back_id_length = np.where(data_dict['back_id']<record_length)[0].size     # 18
                
                behavior_data_dict = {variable: data_dict[variable][:179] for variable in variables_list}
                behavior_data_dict['iT'] = {_: data_dict['iT'][_][:179] for _ in data_dict['iT'].keys()}
                behavior_data_dict['back_id'] = data_dict['back_id'][:18]
                
            elif filename == 'CelebA_p11WV_25Feb2020095049.mat':     # cut-off, [Exception 3, P11WV_Sess1]
                behavior_data_dict = {variable: data_dict[variable][54:] for variable in variables_list}
                behavior_data_dict['iT'] = {_: data_dict['iT'][_][54:] for _ in data_dict['iT'].keys()}
                behavior_data_dict['back_id'] = data_dict['back_id'][6:] - 54
            
            # --- processes local added
            elif filename == 'CelebA_p7WV_20Sep2019101736.mat':     # cut-off and shift, [Exception 4, P7WV_Sess2]
                behavior_data_dict = {variable: data_dict[variable][2:] for variable in variables_list}
                behavior_data_dict['iT'] = {_: data_dict['iT'][_][2:] for _ in data_dict['iT'].keys()}
                behavior_data_dict['back_id'] = data_dict['back_id'] - 2
                
            elif filename == 'CelebA_p9WV_27Oct2019113332.mat':     # cut-off and shift, [Exception 5, P9WV_Sess3]
                behavior_data_dict = {variable: data_dict[variable][1:] for variable in variables_list}
                behavior_data_dict['iT'] = {_: data_dict['iT'][_][1:] for _ in data_dict['iT'].keys()}
                behavior_data_dict['back_id'] = data_dict['back_id'] - 1
            
            # --- for the rest 35 normal cases
            else:
                behavior_data_dict = {variable: data_dict[variable] for variable in variables_list}
                behavior_data_dict['iT'] = {_: data_dict['iT'][_] for _ in data_dict['iT'].keys()}
                behavior_data_dict['back_id'] = data_dict['back_id']

        # ↓ can not find such file and looks like the function is using TTL to refine the exact timing.
        # adjustBehaviorAccordingToTTL()  
        
        behavior_data_dict['isEyeTrack'] = data_dict['isEyeTrack'].item()
        behavior_data_dict['stimWindowSize'] = data_dict['stimWindowSize']
        behavior_data_dict['windowRect'] = data_dict['windowRect']
        behavior_data_dict['T'] = data_dict['T']
        
        return behavior_data_dict
    
    def behavior_data_dict_format_transformation(self, path):
        """
            load and convert experimental data from MATLAB structure to Python dict. In fact, the files contain detailed 
            experiment records, including mechine conditions, time stamps, ...
        """
        
        data = sio.loadmat(path)
        data = {_: data[_] for _ in data.keys() if '__' not in _}
        
        # basic types: int, float, str
        data_basic = {_:data[_].reshape(-1) for _ in data.keys() if 'int' in str(data[_].dtype.type) or 'float' in str(data[_].dtype.type) or 'str' in str(data[_].dtype.type)}
        for _ in data_basic.keys():
            if 'str' in str(data_basic[_].dtype.type):
               data_basic[_] = str(data_basic[_].item())
               
        data_basic['back_id'] = data_basic['back_id'] - 1     # 1-based (MATLAB) -> 0-based (python)
        
        # structures (matlab): void, obj
        data_structure = {_:data[_] for _ in data.keys() if _ not in data_basic.keys()}
        
        data_structure_void = {_: data_structure[_].reshape(-1) for _ in data_structure.keys() if 'void' in str(data_structure[_].dtype.type)}
        for __ in data_structure_void.keys():
            if len(data_structure_void[__]) == 1:
                data_structure_void[__] = {_: data_structure_void[__][_][0].reshape(-1).item() if data_structure_void[__][_][0].size == 1 else data_structure_void[__][_][0].reshape(-1) for _ in data_structure_void[__].dtype.names}
            else:
                data_structure_void[__] = {_: np.array([np.nan if ___.size == 0 else ___.item() for ___ in data_structure_void[__][_]]) for _ in data_structure_void[__].dtype.names}
                
        data_structure_obj = {_: data_structure[_] for _ in data_structure.keys() if 'obj' in str(data_structure[_].dtype.type)}
        data_structure_obj = {__: [_.item() for _ in data_structure_obj[__].reshape(-1)] for __ in data_structure_obj.keys()}
        
        # merge as one
        data_dict = data_basic.copy()
        data_dict.update(**data_structure_void)
        data_dict.update(**data_structure_obj)
        
        return data_dict
    
    def get_session_idces(self):
        """
            Those are raw neural responses collected by Dr. Cao.
            The CelebA/FBI/NavFace indicates the used datasets, refer to: https://www.biorxiv.org/content/10.1101/2020.09.01.278283v2.abstract
            Those are manually selected files
        """
        sessions = [
                'p6WV_CelebA_Sess1',
                'p6WV_CelebA_Sess2',
                'p7WV_CelebA_Sess1',
                'p7WV_CelebA_Sess2',
                'p7WV_CelebA_Sess3',
                'p7WV_CelebA_Sess4',
                'p9WV_CelebA_Sess1',
                'p9WV_CelebA_Sess2',
                'p9WV_CelebA_Sess3',
                'p9WV_CelebA_Sess4',
                'p10WV_CelebA_S2_FBI_S2',
                'p10WV_CelebA_Sess3',
                'p10WV_Loc2_S1_CelebA_S1_FBI_S1',
                'p11WV_CelebA_S1_FBI_S1_Loc2_S1',
                'p11WV_CelebA_S2_FBI_S2_Loc2_S2',
                'p11WV_CelebA_S3_FBI_S3_Loc2_S3',
                'p11WV_CelebA_S4_FBI_S4_Loc2_S4',
                'p11WV_CelebA_Sess5',
                'p13WV_CelebA_Sess1',
                'p14WV_CelebA_S1_FBI_S1',
                'p14WV_CelebA_S2_FBI_S2',
                'p14WV_CelebA_S3_FBI_S3',
                'p14WV_CelebA_S4_FBI_S4',
                'p15WV_CelebA_S1_FBI_S1',
                'p15WV_CelebA_S2_FBI_S2',
                'p16WV_CelebA_S1',
                'p16WV_CelebA_S2_NavFace_S1',
                'p16WV_CelebA_S3_NavFace_S3',
                'p16WV_CelebA_S4_NavObj_S2',
                'p16WV_CelebA_S5_FBI_S1_NavFace_S4',
                'p16WV_CelebA_S6_NavFace_S5',
                'p18WV_CelebA_S1_FBI_S1',
                'p18WV_CelebA_S2_NavFace_S1',
                'p18WV_CelebA_S3_NavFace_S2',
                'p18WV_CelebA_S4',
                'p19WV_CelebA_S1_NavFace_S1',
                'p19WV_CelebA_S2',
                'p20WV_CelebA_S1_NavFace_S1',
                'p20WV_CelebA_S2_NavFace_S2',
                'p20WV_CelebA_S3_FBI_S1']
        
        return sessions
    
    def get_session_attr(self):
        """
            this function wraps one single session information with the stats belongs the corresponding patient
        """
        
        sessions = self.get_session_idces()
        
        sessions_attr = []
        
        for idx, session in enumerate(sessions):
            
            patient = session.split('_')[0]
            
            sessions_tmp = [_ for _ in sessions if patient in _]

            patient_sessions_num = len(sessions_tmp)
            patient_session_idx = sessions_tmp.index(session)
            
            session_attr_single = [
                idx,
                session,
                patient,
                patient_sessions_num,
                patient_session_idx
                ]
            
            sessions_attr.append(session_attr_single)
        
        return sessions_attr
    
    # ===== module 2. obtain identity-selective cells
    def humane_identity_cell_selection(self, ):
        """
            this function select identity_sensitive cells, identity_encode cells and identity_selective cells
            
            the MATLAB code based on: https://uk.mathworks.com/matlabcentral/fileexchange/22088-repeated-measures-anova?s_tid=mwa_osa_a
            
                "This program was originally released when MATLAB had no support for repeated measures ANOVA. However, 
                since a few releases ago, MATLAB statistics toolbox has added this functionality (see the fitrm function). 
                Thus this program is now deprecated and is not recommended anymore. The issue is that it only support a 
                very small subclass of the problems that fitrm can solve. Also, it might not have been tested as extensively 
                as fitrm so it is possible that it does not produce correct results in all cases.
                I keep the program as it is here but it will not be maintained any more."
            
            hence, the ANOVA function used in this code is f_oneway() from stats, the results are significantly different 
            with the results from anova_rm() of MATLAB. The results from f_oneway() are averagely higher than anova_rm()
            
            The selection for ANOVA:
                
            - [in use] f_oneway(): merge all data and analysis in one time
            - smf(): mixed effect model
            
            [notice] the s_si comes from MATLAB and python are the same
        """
        
        save_path = os.path.join(self.human_neuron_stats, 'cell_types.pkl')
        
        if os.path.exists(save_path):
            
            cell_stats = utils_.pickle_load(save_path)
            
        else:
        
            # -----
            meanFR_dict = self.human_neuron_sort_FR(data_type='default')
            
            meanFR = meanFR_dict['meanFR']
            qualified_cells = meanFR_dict['qualified_cells']
            
            # -----
            neuron_session_idces = self.Spikes['vCell'].reshape(-1) - 1     # 0-based to 1-based
            sessions = self.get_session_idces()
            
            sessions_attr = self.get_session_attr()
            
            cell_attr = {}
            for _ in range(2082):
                
                session_idx = neuron_session_idces[_]
                
                all_neurons_of_this_session = np.where(neuron_session_idces==session_idx)[0]
                
                sessions_of_this_patient = [idx for idx,_ in enumerate(sessions) if sessions_attr[session_idx][2] in _]
                all_neurons_of_this_patient = [__ for _ in sessions_of_this_patient for __ in np.where(neuron_session_idces==_)[0]]
                
                cell_attr.update({_:{
                'session_idx': session_idx,
                'session': sessions[session_idx],
                'patient': sessions_attr[session_idx][2],
                'patient_sessions_num': sessions_attr[session_idx][3],
                'patient_session_idx': sessions_attr[session_idx][4],
                'all_neurons_of_this_session': all_neurons_of_this_session,
                'all_neurons_of_this_patient': all_neurons_of_this_patient
                }})
                
            cell_attr = {_: cell_attr[_] for _ in qualified_cells}
            
            # -----
            meanFR[np.isnan(meanFR)] = 0     # <- convert NaN values to 0
            
            meanFR_ = []
            p_list = []
            encode_id = {}
           
            for cell_idx in tqdm(range(meanFR.shape[0]), desc='cell ANOVA'):     # for each cell
                
                meanFR_single_cell = [meanFR[cell_idx, self.id_img_idces_dict[_]] for _ in range(50)]    # list, (50, 10)
                meanFR_.append(np.array(meanFR_single_cell))
                
                # ----- 1. one way ANOVA
                p = stats.f_oneway(*meanFR_single_cell)[1]    
                p_list.append(p)
                
                # ----- 2. mean+2SD
                # [notice] use: | si | wsi | mi | wmi | n |
                meanFR_single_cell = np.array(meanFR_single_cell)
                th = np.mean(meanFR_single_cell) + 2*np.std(meanFR_single_cell)
                ref = np.mean(meanFR_single_cell) + 2*np.std(np.mean(meanFR_single_cell, axis=1))
                
                encode = np.array([_ for _ in range(50) if np.mean(meanFR_single_cell[_]) > th])
                weak_encode = np.array([_ for _ in range(50) if np.mean(meanFR_single_cell[_]) > ref])
                weak_encode = np.setdiff1d(weak_encode, encode)
                
                encode_id[cell_idx] = {
                    'encode': encode,
                    'weak_encode': weak_encode,
                                       }
            
            p_list = np.array(p_list)
            
            # ----- 1. sensitive test
            s = np.where(p_list<0.05)[0]     # <- consider how to handle the unqualified cells
            
            s = np.intersect1d(s, qualified_cells)
            ns = np.setdiff1d(qualified_cells, s)
            
            # ----- 2. encode test
            encode_id = {_: encode_id[_] for _ in qualified_cells}
            
            si = np.array([_ for _ in encode_id.keys() if len(encode_id[_]['encode']) == 1])     # 10
            wsi = np.array([_ for _ in encode_id.keys() if len(encode_id[_]['weak_encode']) == 1 and len(encode_id[_]['encode']) == 0])     
            wsi = np.setdiff1d(wsi, si)     # 499
        
            mi = np.array([_ for _ in encode_id.keys() if len(encode_id[_]['encode']) > 1])     # 3
            wmi = np.array([_ for _ in encode_id.keys() if len(encode_id[_]['weak_encode']) > 1 and len(encode_id[_]['encode']) == 0])     
            wmi = np.setdiff1d(wmi, mi)     # 949
    
            non_encode = np.array([_ for _ in encode_id.keys() if len(encode_id[_]['weak_encode']) == 0 and len(encode_id[_]['encode']) == 0])     # 116
            
            # ----- 3. advanced types
            s_si = np.intersect1d(s, si)     # 10
            s_wsi = np.intersect1d(s, wsi)     # 47
            s_mi = np.intersect1d(s, mi)     # 3
            s_wmi = np.intersect1d(s, wmi)     # 95
            s_non_encode = np.intersect1d(s, non_encode)     # 7
            
            ns_si = np.intersect1d(ns, si)     # 0
            ns_wsi = np.intersect1d(ns, wsi)     # 452
            ns_mi = np.intersect1d(ns, mi)     # 0
            ns_wmi = np.intersect1d(ns, wmi)     # 854
            ns_non_encode = np.intersect1d(ns, non_encode)     # 109
            
            cell_types_dict = {
                'sensitive_si': s_si,
                'sensitive_wsi': s_wsi,
                'sensitive_mi': s_mi,
                'sensitive_wmi': s_wmi,
                'sensitive_non_encode': s_non_encode,
                'non_sensitive_si': ns_si,
                'non_sensitive_wsi': ns_wsi,
                'non_sensitive_mi': ns_mi,
                'non_sensitive_wmi': ns_wmi,
                'non_sensitive_non_encode': ns_non_encode
                }
            
            cell_stats = {
                'cell_attr': cell_attr,
                'encode_id': encode_id,
                'cell_types_dict': cell_types_dict
                }
            
            utils_.pickle_dump(save_path, cell_stats)
        
        # ----- plot
        #for cell_idx in np.intersect1d(si, qualified_cells):
        #    meanFR_id = meanFR_[cell_idx]
        #    fig, ax = plt.subplots(figsize=(10,10))
        #    ax.scatter(np.arange(50), np.mean(meanFR_id, axis=1))
        #    ax.hlines(np.mean(meanFR_id)+2*np.std(meanFR_id), 0, 49, color='red', label='th')
        #    ax.hlines(np.mean(meanFR_id)+2*np.std(np.mean(meanFR_id, axis=1)), 0, 49, color='teal', label='ref')
        #    ax.set_title(f'{cell_idx}')
        #    ax.legend()
        #    plt.show()
        
        return cell_stats
        
    def human_neuron_stacked_encode_map(self, ):
        
        """
            [task] this function should provide the 
        """
           
        # ---
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 14})
        
        colorpool_jet = plt.get_cmap('jet', 50)
        colors = [colorpool_jet(i) for i in range(50)]
        
        # --- load cell_types
        cell_stats = self.humane_identity_cell_selection()
        idx_dict = cell_stats['cell_types_dict']
        
        # --- load feature
        meanFR_dict = self.human_neuron_sort_FR(data_type='default')
        meanFR = meanFR_dict['meanFR']
        
        feature = meanFR.T
        feature[np.isnan(feature)] = 0
        
        # --- 
        y_lim_min = np.min(feature)
        y_lim_max = np.max(feature)
        
        fig, ax = plt.subplots(figsize=(26, 10))
        gs_main = gridspec.GridSpec(2, 5, figure=fig)
        
        tqdm_bar = tqdm(total=10, desc='Human Neurons')
        
        i_ = 0
        for i in range(2):
            for j in range(5):
                # Define a sub-grid within the current cell of the main grid
                gs_sub = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[4, 1], subplot_spec=gs_main[i, j])

                ax_left = fig.add_subplot(gs_sub[0])
                ax_right = fig.add_subplot(gs_sub[1])
                
                if i_ != 0:
                    ax_left.set_xticks([])
                    ax_left.set_yticks([])

                ax_right.set_xticks([])
                ax_right.set_yticks([])
                
                if idx_dict[list(idx_dict.keys())[i_]].size == 0:
                    pct = len(idx_dict[list(idx_dict.keys())[i_]])/feature.shape[1]*100
                    ax_left.set_title(list(idx_dict.keys())[i_] + f'[{pct:.2f}% | {len(idx_dict[list(idx_dict.keys())[i_]])}/{feature.shape[1]}]')
                    ax_right.set_title('th')
                    i_ += 1
                else:
                    feature_test = feature[:, idx_dict[list(idx_dict.keys())[i_]]]     # (500, num_units)
                    feature_test_mean = feature_test.reshape(50, 10, -1)     # (50, 10, num_units)
                    
                    # -----
                    x = np.array([[[_] for _ in range(50)]*feature_test_mean.shape[2]]).reshape(-1)
                    y = np.mean(feature_test_mean, axis=1).T.reshape(-1)     # (50, num_classes)
                    
                    c = np.array(colors)
                    c = np.tile(c, [feature_test_mean.shape[2], 1])
                    #c = np.repeat(c, 10, axis=0)     # <- for each img
                    
                    # -----
                    ax_left.scatter(x, y, color=c, alpha=0.7, marker='.', s=10)     # use small size to replace adjustable alpha
                    # -----
                    
                    pct = len(idx_dict[list(idx_dict.keys())[i_]])/feature.shape[1]*100
                    ax_left.set_title(list(idx_dict.keys())[i_] + f' [{pct:.2f}% | {len(idx_dict[list(idx_dict.keys())[i_]])}/{feature.shape[1]}]')
                    # -----
                    
                    feature_test_mean = np.mean(feature_test_mean, axis=1)     # (50, num_units)
                    # ----- stats: mean for each ID
                    values = feature_test_mean.reshape(-1)    # (50*num_units)
                    if np.std(values) == 0:
                        pass
                    else:
                        kde_mean = gaussian_kde(values)
                        x_vals_mean = np.linspace(np.min(values), np.max(values)*1.1, 1000)
                        y_vals_mean = kde_mean(x_vals_mean)
                        ax_right.plot(y_vals_mean, x_vals_mean, color='blue')
                        
                    # ----- stats: threshold (mean+2std of all 500 values)
                    values = np.mean(feature_test, axis=0) + 2*np.std(feature_test, axis=0)     # (units,)
                    if np.std(values) == 0:
                        pass
                    else:
                        kde = gaussian_kde(values)
                        x_vals = np.linspace(np.min(values), np.max(values)*1.1, 1000)
                        y_vals = kde(x_vals)
                        ax_right.plot(y_vals, x_vals, color='red')
                    
                        y_vals_max = np.max(y_vals)
                        x_vals_max = x_vals[np.where(y_vals==y_vals_max)[0].item()]
                        
                        ax_left.hlines(x_vals_max, 0, 50, colors='red', alpha=0.75, linestyle='--')
                        ax_right.hlines(x_vals_max, np.min(y_vals), np.max(y_vals), colors='red', alpha=0.75, linestyle='--')
                    
                    # ----- stats: ref (mean+2std of all 50 mean values)
                    values = np.mean(feature_test_mean, axis=0) + 2*np.std(feature_test_mean, axis=0)     # (units,)
                    
                    if np.std(values) == 0:
                        pass
                    else:
                        kde = gaussian_kde(values)
                        x_vals = np.linspace(np.min(values), np.max(values)*1.1, 1000)
                        y_vals = kde(x_vals)
                        ax_right.plot(y_vals, x_vals, color='teal')
                    
                        y_vals_max = np.max(y_vals)
                        x_vals_max = x_vals[np.where(y_vals==y_vals_max)[0].item()]
                        
                        ax_left.hlines(x_vals_max, 0, 50, colors='teal', alpha=0.75, linestyle='--')
                        ax_right.hlines(x_vals_max, np.min(y_vals), np.max(y_vals), colors='teal', alpha=0.75, linestyle='--')
                    
                    ax_left.set_ylim([y_lim_min, y_lim_max/2])
                    ax_right.set_ylim([y_lim_min, y_lim_max/2])
                    ax_right.set_title('th')
                    
                    i_ += 1
                tqdm_bar.update(1)
               
        ax.axis('off')
        ax.plot([],[],color='blue',label='mean')
        ax.plot([],[],color='teal',label='ref')
        ax.plot([],[],color='red',label='threshold')
        
        fig.suptitle('Human MTL Neuron Responses for Human Faces', y=0.97, fontsize=20)
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, bbox_transform=plt.gcf().transFigure)
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.human_neuron_stats, 'Human MTL Neuron Responses for Human Faces.png'), bbox_inches='tight')
        fig.savefig(os.path.join(self.human_neuron_stats, 'Human MTL Neuron Responses for Human Faces.eps'), bbox_inches='tight', format='eps')
        fig.savefig(os.path.join(self.human_neuron_stats, 'Human MTL Neuron Responses for Human Faces.pdf'), bbox_inches='tight', format='pdf', transparent=True)
        plt.close()

    # ===== module 3. raster plot
    def human_neuron_raster_plot(self,  type_to_plot:list=None, plot_cell_num:int=10):
        """
            this function calculates the dependencies to plot and depict cell responses in different ways,
            
            input: 
                type_to_plot: plot what types of cell (default: all)
                plot_cell_num: plot how many cells for each type, or plot all cells if the number of qualified cells is less than plot_cell_num  (default: 10)
            
            
        """
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        # -----
        cell_plot_path = os.path.join(self.human_neuron_stats, 'cell plot')
        utils_.make_dir(cell_plot_path)
        
        # --- set color
        colorpool_jet = plt.get_cmap('jet', 50)
        colors = [colorpool_jet(i) for i in range(50)]
        
        # ----- Firing Rates and cells information
        FR_stats = self.human_neuron_get_firing_rate()
        cell_stats = self.humane_identity_cell_selection()

        # --- preperation
        neuron_session_idces = self.Spikes['vCell'].reshape(-1) - 1    # <- session_idx ID, 0-based
        time_stamps_all_cells = [_.reshape(-1) for _ in self.Spikes['timestampsOfCellAll'].reshape(-1)]
        
        sessions = self.get_session_idces()     # <- stores time periods of each trial (each session_idx)
        session_idx_dir = os.path.join(self.root_data, 'Events Files')     # <- store timestamps for each responses, self.root_data: osfstorage-archive
        
        all_periods = []     # provides the time range of one single trial with label
        for session_idx in tqdm(range(len(sessions)), desc='Load sessions'):
            # ↓ 'periods' contains 3 columns: trial indices, timestamps to 500 ms before stumuli onset, timestamps to 1500 ms after stimuli onset
            periods = sio.loadmat(os.path.join(session_idx_dir, sessions[session_idx]+'.mat'))['periods']     
            all_periods.append(periods)
    
        beh_stats = self.human_neuron_get_beh()
        behavior = beh_stats['beh']
        
        # ----- build [im_code] img_idces and img_idces_new dict
        CelebA_img_idces = sio.loadmat(os.path.join(self.root_data, 'SingleNeuron/Code/CelebA_Image_Code.mat'))
        img_idces = CelebA_img_idces['im_code'].reshape(-1)     # img_idx linked ID, 53 IDs in total, 3 unwanted
        adjust_idx = CelebA_img_idces['AdjustInd'].reshape(-1).astype(np.int16) -1   # [notice] what the adjust_idx is?
        
        self.img_idces_dict = {_:img_idces[_] for _ in range(len(img_idces))}     # {img number: ID}
        self.adjust_idx_dict = {adjust_idx[_].astype(int): _ for _ in range(len(adjust_idx)) if adjust_idx[_] > -1}     # {wrong number: right number}
        
        # ---
        CelebA_img_idx_new = sio.loadmat(os.path.join(self.root_data, 'SingleNeuron/Code/CelebA_Image_Code_new.mat'))
        img_idces_new = CelebA_img_idx_new['im_code'].reshape(-1)     # img_idx linked ID, 50 IDs in total
        
        id_img_idces = CelebA_img_idx_new['id_code'].reshape(-1)     # each ID contains 10 imgs
        id_img_idces = np.array([_.reshape(-1)-1 for _ in id_img_idces])
        
        self.img_idces_new_dict = {_:img_idces_new[_] for _ in range(len(img_idces_new))}     # {img number: ID}
        self.id_img_idces_dict = {_:id_img_idces[_] for _ in range(len(id_img_idces))}     # {ID: 10 img numbers}     0-based
        
        # ----- start plot for different types of cells
        if type_to_plot == None:
            type_to_plot = list(cell_stats['cell_types_dict'].keys())    
        
        for cell_type in type_to_plot:  # for each neuron
        
            cell_type_plot_path = os.path.join(cell_plot_path, f'{cell_type}')
            utils_.make_dir(cell_type_plot_path)
        
            if len(cell_stats['cell_types_dict'][cell_type]) > plot_cell_num:
                cell_to_plot = np.random.choice(cell_stats['cell_types_dict'][cell_type], plot_cell_num)
            
            elif len(cell_stats['cell_types_dict'][cell_type]) != 0:
                cell_to_plot = cell_stats['cell_types_dict'][cell_type]
                
            else:
                cell_to_plot = []
                
            for cell_idx in tqdm(cell_to_plot, desc=f'{cell_type}'):

                self.session_idx = neuron_session_idces[cell_idx]  # session_idx
                
                # ----- im_code: key is the img label, value is the corresponding ID label
                if self.session_idx < 10:
                    im_code = self.img_idces_dict
                else:
                    im_code = self.img_idces_new_dict
            
                time_stamps = time_stamps_all_cells[cell_idx]
                
                periods = all_periods[self.session_idx]     # e.g (550, 3)
                
                displayed_image_sequence = behavior[self.session_idx]['code']     # the displayed img list
                back_id = behavior[self.session_idx]['back_id']     # local version, 0-based
                
                # -----
                FR_tmp = FR_stats[cell_idx]['spike_count'].copy()     # FR for this cell
                FR_PSTH = FR_stats[cell_idx]['PSTH_250'].copy()
                
                # -----
                displayed_image_sequence = np.delete(displayed_image_sequence, back_id) - 1     # 0-based
                periods = periods[np.delete(np.arange(periods.shape[0]), back_id), :]     # e.g. (500, 3) the first column is the idx in original displayed_image_sequence
                
                FR_tmp = np.delete(FR_tmp, back_id)
                FR_PSTH = np.delete(FR_PSTH, back_id, axis=0)
                
                # -----
                displayed_ID_sequence = np.array([im_code[_] for _ in displayed_image_sequence])      # id of each image
                # -----
                
                # [notice] the mean firing rate here is different with meanFR which is based on the entire trial, here is based on 750-1750
                # [notice] this excludes information for every img
                FR_ID = []
                PSTH_ID = []
                for ID in range(1, 51):
                    FR_ID.append(FR_tmp[np.where(displayed_ID_sequence==ID)[0]])
                    PSTH_ID.append(FR_PSTH[np.where(displayed_ID_sequence==ID)[0], :])
                    
                # ----- stats
                FR_ID_all = np.array([__ for _ in FR_ID for __ in _])
                FR_ID_all_mean = np.mean(FR_ID_all)
                FR_ID_std = np.std(FR_ID_all)
                
                FR_ID_mean = np.array([np.mean(_) for _ in FR_ID])
                FR_ID_ref = np.std(FR_ID_mean)
                
                # ----- return jagged periods_and_infos
                periods_and_infos = []
                
                for ID in range(1, 51):    # for each ID
                    
                    positions = np.where(displayed_ID_sequence == ID)[0]
                    
                    img_labels = displayed_image_sequence[positions]
                    ID = np.array([im_code[_] for _ in img_labels])
                    
                    if np.unique(ID) != ID:
                        raise RuntimeError(f'[Codinfo] ID check failed for [{ID}]')
                    
                    period = periods[positions, :]     # order in experiment | trial start time | trial end time
                    period = np.hstack((period, np.vstack((img_labels, ID)).T))     # | order in experiment | trial start time | trial end time | img label | ID | session |
                    
                    periods_and_infos.append(period[period[:,0].argsort()])     # legacy design from MATLAB code, not necessary
    
                # ----- this should return neat spikes_to_plot of (500,)
                spikes_to_plot = self.getTimestampsOfBubbles(time_stamps, periods_and_infos)
            
                # ----- plot
                fig = plt.figure(figsize=(30, 20))
    
                # --- subplot_1
                spikeheight = 2
                spikewidth = 2
                
                axes_0 = plt.gcf().add_axes([0.05, 0.45, 0.55, 0.475])
                
                self.plotSpikeRasterMain(axes_0, spikes=spikes_to_plot, colors=colors, spikeheight=spikeheight, spikewidth=spikewidth)
                
                axes_0.vlines(500, -1, 501, linestyle='-', alpha=0.75, color='gray', label='image on')
                axes_0.vlines(1500, -1, 501, linestyle='--', alpha=0.75, color='gray', label='image off')
                
                axes_0.vlines(750, -1, 501, linestyle='-', alpha=0.75, color='tomato', label='count on')
                axes_0.vlines(1750, -1, 501, linestyle='--', alpha=0.75, color='tomato', label='count off')
                
                axes_0.set_xlim([0, 2000])
                axes_0.set_ylim([- spikeheight/2, 500 + spikeheight/2])
                
                axes_0.tick_params(axis='both', labelsize=20, width=2, length=5, labelcolor='black', labelbottom=True)
                axes_0.set_xlabel('Time (ms)', fontsize=24)
                axes_0.set_ylabel('Image (10 imgs per ID)', fontsize=24)

                axes_0.set_title('Raster Plot', fontsize=28)
                axes_0.legend(fontsize=24, loc='upper right')
                
                # --- subplot_2
                axes_1 = plt.gcf().add_axes([0.65, 0.45, 0.3, 0.475])
                
                axes_1.barh(np.arange(50), FR_ID_mean, color=colors)
                
                sem_list = [np.std(FR_ID[_])/np.sqrt(len(FR_ID[_])) for _ in range(len(FR_ID))]
                
                for idx, _ in enumerate(FR_ID):
                    
                    axes_1.scatter(FR_ID_mean[idx] + sem_list[idx], idx, color='black', marker='d')
                    axes_1.scatter(FR_ID_mean[idx] - sem_list[idx], idx, color='black', marker='d')
    
                    axes_1.hlines(idx, FR_ID_mean[idx] - sem_list[idx], FR_ID_mean[idx] + sem_list[idx], linestyle='--', color='black')
                
                axes_1.vlines(FR_ID_all_mean+2*FR_ID_std, -1, 50, linestyle='--', alpha=0.75, color='red', label='threshold')
                axes_1.vlines(FR_ID_all_mean+2*FR_ID_ref, -1, 50, linestyle='--', alpha=0.75, color='teal', label='ref')
                
                axes_1.tick_params(axis='both', labelsize=20, width=2, length=5, labelcolor='black', labelbottom=True)
                axes_1.set_xlabel('Firing Rate (Hz)', fontsize=24)
                axes_1.set_ylabel('ID', fontsize=24)
                #axes_1.set_yticks([])
                
                axes_1.set_xlim([0, np.max([np.max(FR_ID_mean) + np.max(sem_list), FR_ID_all_mean+2*FR_ID_std])*1.2])
                axes_1.set_ylim([-0.5, 49.5])
    
                axes_1.legend(fontsize=24)
                axes_1.set_title('Mean Firing Rate [750ms - 1750ms] (with SE)', fontsize=28)
                
                # --- subplots_3
                PSTH_ID_ = np.array([np.mean(_, axis=0) for _ in PSTH_ID])
                
                axes_2 = plt.gcf().add_axes([0.05, 0.05, 0.55, 0.325])
                
                fig_psth = axes_2.imshow(PSTH_ID_, origin='lower', aspect='auto', cmap='turbo')
                axes_2.set_xticks([0,5,10,15,20,25,30])
                axes_2.set_xticklabels([250, 500, 750, 1000, 1250, 1500, 1750], fontsize=20)
                axes_2.tick_params(axis='both', labelsize=20, width=2, length=5, labelcolor='black', labelbottom=True)
                axes_2.set_xlabel('Time (ms) [250ms - 2000ms]', fontsize=24)
                axes_2.set_ylabel('ID', fontsize=24)
                #axes_2.set_yticks([])
                axes_2.set_title('PSTH [window: 50ms, step: 250ms]', fontsize=28)
                
                cbar_ax = fig.add_axes([0.6125, 0.05, 0.01, 0.325])  # [left, bottom, width, height]
                cbar = fig.colorbar(fig_psth, cax=cbar_ax)
                cbar.ax.tick_params(labelsize=24)
                
                # --- subplot_4
                # ----- the displayed statistics exclude back_id
                all_neurons_of_this_session = cell_stats['cell_attr'][cell_idx]['all_neurons_of_this_session']
                all_neurons_of_this_patient = cell_stats['cell_attr'][cell_idx]['all_neurons_of_this_patient']
                
                FR_all = [__ for _ in range(2082) for __ in FR_stats[_]['spike_count']]
                FR_session = [__ for _ in all_neurons_of_this_session for __ in FR_stats[_]['spike_count']]
                FR_patient = [__ for _ in all_neurons_of_this_patient for __ in FR_stats[_]['spike_count']]
                
                data = [
                    [int(np.max(FR_tmp)), int(np.max(FR_session)), int(np.max(FR_patient)), int(np.max(FR_all))],
                    [int(np.min(FR_tmp)), int(np.min(FR_session)), int(np.min(FR_patient)), int(np.min(FR_all))],
                    [int(np.median(FR_tmp)), int(np.median(FR_session)), int(np.median(FR_patient)), int(np.median(FR_all))],
                    [np.mean(FR_tmp).round(2), np.mean(FR_session).round(2), np.mean(FR_patient).round(2), np.mean(FR_all).round(2)],
                    [np.std(FR_tmp).round(2), np.std(FR_session).round(2), np.std(FR_patient).round(2), np.std(FR_all).round(2)]
                    ]
                
                # ----- 
                
                col_labels = ['cell', f'session\n({len(all_neurons_of_this_session)})', f'patient\n({len(all_neurons_of_this_patient)})', 'all cells\n(2082)']
                row_labels = ['max', 'min', 'median', 'mean', 'std']
                
                ccolors = plt.cm.BuPu(np.linspace(0, 0.5, len(col_labels)))
                rcolors = plt.cm.BuPu(np.linspace(0, 0.5, len(row_labels)))
                
                axes_3 = plt.gcf().add_axes([0.7, 0.05, 0.25, 0.325])
                axes_3.axis('tight')
                axes_3.axis('off')
                table = axes_3.table(cellText=data, colLabels=col_labels, rowLabels=row_labels, 
                        cellLoc='center', rowLoc='center', colColours=ccolors, rowColours=rcolors, loc='center',
                        colWidths=[0.25]*4, 
                        bbox=[0, 0, 1, 1])
                
                table.auto_set_font_size(False)
                table.set_fontsize(26)
                
                axes_3.set_title('Statistic (Firing Rate)', fontsize=28)
                
                # --- title
                cell_attr = cell_stats['cell_attr'][cell_idx]
                fig.suptitle(f"Cell No. {cell_idx+1} | Session No. {cell_attr['session_idx']+1} | Session: {cell_attr['session']} | ({cell_attr['patient_session_idx']+1}/{cell_attr['patient_sessions_num']})", fontsize=32)
                
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore')
                    
                    plt.tight_layout()
                    #plt.show()
                    
                    plt.savefig(cell_type_plot_path + f'/{cell_idx}.png', bbox_inches='tight')
                    plt.savefig(cell_type_plot_path + f'/{cell_idx}.eps', format='eps', bbox_inches='tight')
                    plt.savefig(cell_type_plot_path + f'/{cell_idx}.pdf', format='pdf', bbox_inches='tight')
                    plt.close()

    def plotSpikeRasterMain(self, ax, spikes, colors, spikeheight=1, spikewidth=1, start_time=0, end_time=2000):
        
        """
            [Oct 19, 2023]
            some of the hyper paramaters are designed only for this experiments, needed to be changed in future use
        """

        if len(spikes) != 0:
            for idx, _ in enumerate(spikes):     # for each trial
                if _[3] is not np.nan and _[3].size != 0:

                    n = _[3].size
                    
                    # ----- plot spikes
                    ax.plot(
                        np.vstack((_[3], _[3])), np.vstack(((idx - spikeheight/2)*np.ones(n), (idx + spikeheight/2)*np.ones(n))), 
                        linewidth=spikewidth, color=colors[_[0]-1]
                        )
  
    def getTimestampsOfBubbles(self, timestampsOfCell, periods_and_infos):
        """
            prepare bubbles trials for plotting of raster (with color info)
            periods_and_infos: list of periods (each a list of trials)     [ order in experiment | trial start time | trial end time | img label | ID ]
            
            urut/nov09
            
            timestampsOfCell: disordered img sequence
        """

        spikes_to_plot = []

        for inds in periods_and_infos:     # for each group
          
            trialTimestamps = self.getRelativeTimestamps(timestampsOfCell, inds)     # timestamps for one ID,  | ID | img_idx | img_num | timestamps |
            
            # ----- intergrity check
            if self.session_idx < 10: 
                img_idces = np.array([self.adjust_idx_dict[_] for _ in sorted([_[1] for _ in trialTimestamps])])     # img_idces correction
            else:
                img_idces = np.array(sorted([_[1] for _ in trialTimestamps]))
                
            # --- integrity check. ID must be one
            ID = np.unique([trialTimestamps[_][0] for _ in range(len(trialTimestamps))]).item()

            entire_imgs = self.id_img_idces_dict[ID-1]
            
            if img_idces.shape != entire_imgs.shape:
                missed_imgs = np.setdiff1d(entire_imgs, img_idces)
            
                for _ in missed_imgs:
                    trialTimestamps.append([ID, _, np.nan, np.nan])
                
            # --- sort based on label, this order is also the order in each folder
            trialTimestamps = sorted(trialTimestamps, key=lambda x:x[1])
            
            # ----- add into all
            for _ in trialTimestamps:
                spikes_to_plot.append(_)
        
        return spikes_to_plot
        
    def getRelativeTimestamps(self, timestampsOfCell, periods):
        """
            reference timestamps to beginning of the trial
            periods: each row is one trial. 3 columns: trial nr, from, to.
            returns a cell array; each item contains the timestamps of one trial. the number of trials is equal to the number of rows in periods.
            
            urut/dec07
            
            modified: acxyle/Oct17,2023
        """
        
        trialTimestamps = []
        
        for i in range(periods.shape[0]):     # for each trial
            
            inds = np.where((periods[i, 1] <= timestampsOfCell) & (timestampsOfCell <= periods[i, 2]))[0]
            
            # | ID | img_idx | img_num | timestamps |
            trialTimestamps.append([periods[i, 4].astype(int), periods[i, 3].astype(int), periods.shape[0], (timestampsOfCell[inds] - periods[i, 1])/1000])
            
        return trialTimestamps

    # ===== module 4. corr
    #FIXME, this for loop has problem, this will only take the results of the final iteration
    def human_neuron_DSM_process_sub_id(self, metrics:list[str]=None, used_cell_types:list[str]=None, used_id_nums:list[int]=None):
        """
            
        """
        for metric in metrics:
            
            for used_cell_type in used_cell_types:
                
                for used_id_num in used_id_nums:
                    
                    selected_ids, human_DM_dict = self.human_neuron_DSM_process(metric, used_cell_type, used_id_num)
        
        return selected_ids, human_DM_dict
    
    def human_neuron_DSM_process(self, metric, used_cell_type, used_id_num, num_perm=1000):
        """
            this function calculates the human pairwise distance matrices based on selected_ids and Used_cell_type, 
            
            input:
                selected_ids: default all 50 ids
                used_cell_type: default qualified 1,577 cells
        """
        # ---
        selected_ids = self.human_corr_select_sub_identities(used_id_num)
        
        # ---
        save_root = os.path.join(self.human_neuron_stats, 'corr')
        utils_.make_dir(save_root)
        
        save_root = os.path.join(save_root, metric)
        utils_.make_dir(save_root)

        # ---
        used_cells = self.human_neuron_obtain_used_cells(used_cell_type)
            
        if used_cells.size == 0:
            
            return None, None
    
        else:
            
            save_path = os.path.join(save_root, f'Human_DM_dict_{used_cell_type}_{len(selected_ids)}.pkl')
            
            # --- init
            self.meanFR = self.meanFR_dict['meanFR']     # (2082, 500)
            self.meanFR_PSTH = self.meanFR_dict['meanFR_PSTH']
        
            # --- normalize firing rates
            FR_baseline = np.nanmean(self.meanFR_baseline_dict['meanFR'][used_cells, :], axis=1, keepdims=True)
            
            self.normalized_meanFR = self.meanFR[used_cells, :]/FR_baseline     # (num_cells, num_imgs)
            self.normalized_meanFR_PSTH = self.meanFR_PSTH[used_cells, :, :]/FR_baseline.reshape(-1, 1, 1)     # (num_cells, num_imgs, num_time_steps)
            
            # --- 
            self.meanFR_id = np.nanmean(self.normalized_meanFR.reshape(-1, 50, 10), axis=2)[:, selected_ids].T     # (num_selected_ids, num_cells)
            self.meanFR_PSTH_id = np.nanmean(self.normalized_meanFR_PSTH.reshape(-1, 50, 10, 31), axis=2)[:, selected_ids, :].T     # (num_time_steps, num_selected_ids, num_cells)
            
            if os.path.exists(save_path):
                
                print(f'[Codinfo] Loading Human_neuron_DM for {metric} {used_cell_type} {len(selected_ids)}...')
                
                human_DM_dict = utils_.pickle_load(save_path)
                
            else:
                
                print(f'[Codinfo] Calculating Human_neuron_DM for {metric} {used_cell_type} {len(selected_ids)}...')
                
                # --- for static meanFR
                human_DM_v = utils_similarity.selectivity_analysis_calculation(metric, self.meanFR_id)['vector']     # (num_selected_ids*(num_selected_ids-1)/2, )
                human_DM_v_perm = np.array([_['vector'] for _ in Parallel(n_jobs=-1)(delayed(utils_similarity.selectivity_analysis_calculation)(metric, self.meanFR_id[np.random.permutation(len(selected_ids))]) for _ in tqdm(range(num_perm), desc='Human corr'))])
                
                # --- for temporal
                human_DM_v_temporal = np.full((self.meanFR_PSTH_id.shape[0], human_DM_v.size), np.nan)     # (31, 1225)
                human_DM_v_perm_temporal = np.full((self.meanFR_PSTH_id.shape[0], num_perm, human_DM_v.size), np.nan)     # (31, 1000, 1225)
                
                for t in tqdm(range(self.meanFR_PSTH_id.shape[0]), desc='Human corr temporal'):     # for each time point
    
                    human_DM_v_temporal[t, :] = utils_similarity.selectivity_analysis_calculation(metric, self.meanFR_PSTH_id[t])['vector']
                    human_DM_v_perm_temporal[t, :, :] = np.array([_['vector'] for _ in Parallel(n_jobs=-1)(delayed(utils_similarity.selectivity_analysis_calculation)(metric, self.meanFR_PSTH_id[t, np.random.permutation(len(selected_ids))]) for _ in range(num_perm))])
                    
                # --- seal data
                human_DM_dict = {
                    'human_DM_v': human_DM_v, 
                    'human_DM_v_perm': human_DM_v_perm,
                    'human_DM_v_temporal': human_DM_v_temporal,
                    'human_DM_v_perm_temporal': human_DM_v_perm_temporal,
                    'selected_ids': selected_ids
                    }
                
                utils_.pickle_dump(save_path, human_DM_dict)
            
            return selected_ids, human_DM_dict
    
    # FIXME - the MATLAB souce code is actually using 48 ids for all 50 ids, let's try the difference first
    def human_corr_select_sub_identities(self, used_id_num:int=None, cell_type='selective_cells'):
        
        # --- init
        self.meanFR_dict = self.human_neuron_sort_FR(data_type='default')
        self.meanFR_baseline_dict = self.human_neuron_sort_FR(data_type='base')
        
        self.cell_stats = self.humane_identity_cell_selection()
        self.FR_stats = self.human_neuron_get_firing_rate()
        
        encode_dict = self.cell_stats['encode_id']     # encode_dict
        cell_types_dict = self.cell_stats['cell_types_dict']     # ID cells
        
        # --- rebuild wanted cells
        all_selective_cells = {_:cell_types_dict[_] for _ in cell_types_dict.keys() if 'non_sensitive' not in _ and 'non_encode' not in _}
        all_selective_cells = [__ for _ in all_selective_cells.values() for __ in _]
        
        # --- calculate
        if cell_type == 'selective_cells':     # for ID encoded by ID cells
            encoded_id_pool = np.concatenate(np.array([[__ for _ in encode_dict[cell_idx].values() for __ in _] for cell_idx in all_selective_cells], dtype=object))
        elif cell_type == 'encode_cells':     # for ID encoded by all Encode cells
            encoded_id_pool =np.concatenate(np.array([[__ for _ in encode_dict[cell_idx].values() for __ in _] for cell_idx in encode_dict.keys()], dtype=object))
 
        # ----- select used_id_num
        if used_id_num is None:
            selected_ids = [6, 10, 14, 15, 23, 24, 28, 36, 38, 40]
        elif used_id_num is not None and used_id_num > 48:     # [notice] 'selective' human cells only encode 48 ids
            selected_ids = list(np.arange(50))
        elif used_id_num is not None:
            selected_ids = list(self.human_corr_used_ids_selection(encoded_id_pool, used_id_num).keys())[:used_id_num]
        else:
            raise RuntimeError('[Coderror] invalid used_id_num')
        
        return sorted(selected_ids)
    
    def human_corr_used_ids_selection(self, encoded_id_pool, used_id_num):
        """
            Dr CAO provided: [6, 10, 14, 15, 23, 24, 28, 36, 38, 40]
            Calculated here: [6, 10, 14, 15, 24, 28, 30, 36, 43, 45]
            
            return:
                dict
        """
            
        freq = dict(Counter(encoded_id_pool))
        freq={int(k):v for k,v in sorted(freq.items(), key=lambda x:x[1], reverse=True)}
        
        return {k:v for idx, (k,v) in enumerate(freq.items(), 0) if idx < used_id_num}
    
    def human_neuron_obtain_used_cells(self, used_cell_type):
        
        # --- 1.
        if used_cell_type == 'qualified':
            used_cells = np.array(list(self.cell_stats['encode_id'].keys()))
            
        # --- 2-5
        elif used_cell_type == 'selective':     # 'selective' = 's_si' + 's_wsi' + 's_mi' + 's_wmi'
            used_cells = np.concatenate(np.array([
                                        self.cell_stats['cell_types_dict']['sensitive_si'],
                                        self.cell_stats['cell_types_dict']['sensitive_wsi'],
                                        self.cell_stats['cell_types_dict']['sensitive_mi'],
                                        self.cell_stats['cell_types_dict']['sensitive_wmi'], ], dtype=object))
            
        elif used_cell_type == 'strong_selective':     # 'strong_selective' = 's_si' + 's_mi'
            used_cells = np.concatenate(np.array([
                                        self.cell_stats['cell_types_dict']['sensitive_si'],
                                        self.cell_stats['cell_types_dict']['sensitive_mi'],], dtype=object))
            
        elif used_cell_type == 'weak_selective':     # 'weak_selective' =  's_wsi' + 's_wmi'
            used_cells = np.concatenate(np.array([
                                        self.cell_stats['cell_types_dict']['sensitive_wsi'],
                                        self.cell_stats['cell_types_dict']['sensitive_wmi'], ], dtype=object))
            
        elif used_cell_type == 'non_selective':     # 'non_sensitive' = 's_non_encode' + 'ns_si' + 'ns_wsi' + 'ns_mi' + 'ns_wmi' + 'ns_non_encode'
            used_cells = np.concatenate(np.array([
                                        self.cell_stats['cell_types_dict']['sensitive_non_encode'],
                                        self.cell_stats['cell_types_dict']['non_sensitive_si'],
                                        self.cell_stats['cell_types_dict']['non_sensitive_wsi'],
                                        self.cell_stats['cell_types_dict']['non_sensitive_mi'],
                                        self.cell_stats['cell_types_dict']['non_sensitive_wmi'],
                                        self.cell_stats['cell_types_dict']['non_sensitive_non_encode'], ], dtype=object))
        
        # --- 6-15
        elif used_cell_type == 'sensitive_si':
            used_cells = self.cell_stats['cell_types_dict']['sensitive_si']
            
        elif used_cell_type == 'sensitive_wsi':
            used_cells = self.cell_stats['cell_types_dict']['sensitive_wsi']
            
        elif used_cell_type == 'sensitive_mi':
            used_cells = self.cell_stats['cell_types_dict']['sensitive_mi']
            
        elif used_cell_type == 'sensitive_wmi':
            used_cells = self.cell_stats['cell_types_dict']['sensitive_wmi']
            
        elif used_cell_type == 'sensitive_non_encode':
            used_cells = self.cell_stats['cell_types_dict']['sensitive_non_encode']
            
            
        elif used_cell_type == 'non_sensitive_si':
            used_cells = self.cell_stats['cell_types_dict']['non_sensitive_si']
            
        elif used_cell_type == 'non_sensitive_wsi':
            used_cells = self.cell_stats['cell_types_dict']['non_sensitive_wsi']
            
        elif used_cell_type == 'non_sensitive_mi':
            used_cells = self.cell_stats['cell_types_dict']['non_sensitive_mi']
            
        elif used_cell_type == 'non_sensitive_wmi':
            used_cells = self.cell_stats['cell_types_dict']['non_sensitive_wmi']
        
        elif used_cell_type == 'non_sensitive_non_encode':
            used_cells = self.cell_stats['cell_types_dict']['non_sensitive_non_encode']
            
        # --- 16-17
        elif used_cell_type == 'sensitive':
            used_cells = np.concatenate(np.array([
                                        self.cell_stats['cell_types_dict']['sensitive_si'],
                                        self.cell_stats['cell_types_dict']['sensitive_wsi'],
                                        self.cell_stats['cell_types_dict']['sensitive_mi'],
                                        self.cell_stats['cell_types_dict']['sensitive_wmi'], 
                                        self.cell_stats['cell_types_dict']['sensitive_non_encode'], ], dtype=object))
            
        elif used_cell_type == 'non_sensitive':
            used_cells = np.concatenate(np.array([
                                        self.cell_stats['cell_types_dict']['non_sensitive_si'],
                                        self.cell_stats['cell_types_dict']['non_sensitive_wsi'],
                                        self.cell_stats['cell_types_dict']['non_sensitive_mi'],
                                        self.cell_stats['cell_types_dict']['non_sensitive_wmi'],
                                        self.cell_stats['cell_types_dict']['non_sensitive_non_encode'], ], dtype=object))
            
        # --- 18-19
        elif used_cell_type == 'encode':
            used_cells = np.concatenate(np.array([
                                        self.cell_stats['cell_types_dict']['sensitive_si'],
                                        self.cell_stats['cell_types_dict']['sensitive_wsi'],
                                        self.cell_stats['cell_types_dict']['sensitive_mi'],
                                        self.cell_stats['cell_types_dict']['sensitive_wmi'], 
                                        
                                        self.cell_stats['cell_types_dict']['non_sensitive_si'],
                                        self.cell_stats['cell_types_dict']['non_sensitive_wsi'],
                                        self.cell_stats['cell_types_dict']['non_sensitive_mi'],
                                        self.cell_stats['cell_types_dict']['non_sensitive_wmi'], ], dtype=object))
            
        elif used_cell_type == 'non_encode':
            used_cells = np.concatenate(np.array([
                                        self.cell_stats['cell_types_dict']['sensitive_non_encode'],
                                        self.cell_stats['cell_types_dict']['non_sensitive_non_encode'], ], dtype=object))
            
        # --- 20-21
        elif used_cell_type == 'all_sensitive_si':
            used_cells = np.concatenate(np.array([
                                        self.cell_stats['cell_types_dict']['sensitive_si'],
                                        self.cell_stats['cell_types_dict']['sensitive_wsi'], ], dtype=object))
            
        elif used_cell_type == 'all_sensitive_mi':
            used_cells = np.concatenate(np.array([
                                        self.cell_stats['cell_types_dict']['sensitive_mi'],
                                        self.cell_stats['cell_types_dict']['sensitive_wmi'], ], dtype=object))
        
        else:
            
            raise RuntimeError(f'[Coderror] invalid used_cell_type [{used_cell_type}]')
        
        return used_cells
    
    @staticmethod
    def _compare_beh_and_behm(behavior):
        """
            compare the beh_matlab and beh_python
            [Oct 5, 2023] beh_python is identical with beh_matlab
        """
        # --- convert beh_matlab to python friendly structure
        beh_m = sio.loadmat('/home/acxyle-workstation/Downloads/Bio_Neuron_Data/Human/osfstorage-archive/SingleNeuron/FiringRate/Original Data/SortedFR_CelebA.mat')['beh'].reshape(-1)
        beh_m = {_: beh_m[_] for _ in beh_m.dtype.names}
        beh_m = [{__: beh_m[__][_].reshape(-1) for __ in behavior[0].keys()} for _ in range(40)]
        
        for i in range(40):
            beh_m[i]['iT'] = {_: np.array([np.nan if ___.size == 0 else ___.item() for ___ in beh_m[i]['iT'][_]]) for _ in beh_m[i]['iT'].dtype.names}
            
        for i in range(40):
            beh_m[i]['T'] = {_: beh_m[i]['T'][_][0].reshape(-1).item() if beh_m[i]['T'][_][0].size == 1 else ['T'][_][0].reshape(-1) for _ in beh_m[i]['T'].dtype.names}
        
        for i in range(40):
            beh_m[i]['back_id'] = beh_m[i]['back_id'] - 1
        
        # --- compare each segment
        for i in range(40):
            for key in behavior[i].keys():
                if isinstance(behavior[i][key], np.ndarray):
                    if not np.all(behavior[i][key][~np.isnan(behavior[i][key])] == beh_m[i][key][~np.isnan(beh_m[i][key])]):
                        print(i, key, np.all(behavior[i][key][~np.isnan(behavior[i][key])] == beh_m[i][key][~np.isnan(beh_m[i][key])]))
                if isinstance(behavior[i][key], dict):
                    if 'iT' in key:
                        for key_ in behavior[i][key].keys():
                            if not np.all( behavior[i][key][key_][~np.isnan(behavior[i][key][key_])] == beh_m[i][key][key_][~np.isnan(beh_m[i][key][key_])] ):
                                print(i, key, key_, np.all( behavior[i][key][key_][~np.isnan(behavior[i][key][key_])] == beh_m[i][key][key_][~np.isnan(beh_m[i][key][key_])] ))
                    elif 'T' in key:
                        for key_ in behavior[i][key].keys():
                            if not behavior[i][key][key_]==beh_m[i][key][key_]:
                                print(i, key, key_, behavior[i][key][key_]==beh_m[i][key][key_])
                                
    
# ======================================================================================================================
# ----- 

              
#FIXME - use one code merge the basis of Monkey and Human Process
# ======================================================================================================================
class Monkey_Neuron_Records_Process():
    
    """
        considering the monkey data is a well sealed one, this dunction only (1) select the information used for this 
        work; (2) change the save format from .mat to .pkl 
    """
    
    def __init__(self, 
                 bio_root='/home/acxyle-workstation/Downloads/Bio_Neuron_Data/Monkey/',  
                 seed=6
                 ):      
        #super().__init__()
        np.random.seed(seed)

        self.bio_root = bio_root
        
        self.ts = np.arange(-50,201,10)     # target time steps, manually selected from original [-100, 380] 
            
        self.label = sio.loadmat(os.path.join(bio_root, 'Original Data/Label.mat'))['label'].reshape(-1)
        
        monkey_neuron_data_path = os.path.join(bio_root, 'Original Data/IT_FR_CA_Range70-180.mat')     # processed monkey neural data
        #print(sio.whosmat(monkey_neuron_data_path))
        
        monkey_neuron_data = sio.loadmat(monkey_neuron_data_path)

        monkey_dict_keys = [i for i in monkey_neuron_data.keys() if '__' not in i]
        monkey_dict = {_:monkey_neuron_data[_] for _ in monkey_dict_keys}     # rebuild the dict to store monkey IT MUA data
        
        self.FR = monkey_dict['FR']     # [warning] no approach from 'FR' to 'meanFR'  
        self.FR_countAll = self.FR['countAll'][0][0]     
        self.FR_countBase = self.FR['countBase'][0][0]
        self.FR_countVis = self.FR['countVis'][0][0]
        
        # -----
        self.meanFR = monkey_dict['meanFR']     # (53, 500)
        self.meanBase = monkey_dict['meanBase']     # (53, 500)
        self.meanGray = monkey_dict['meanGray'].reshape(-1)     # (53, )
        self.meanVis = monkey_dict['meanVis']     # (53, 500)
        # -----
        
        self.psthTime = monkey_dict['psthTime'].reshape(-1)     # (49,)
        
        # [notice] meanPSTHID is not identical with values calculated from meanPSTH
        self.meanPSTH = monkey_dict['meanPSTH']     # (500, 49, 53), [disordered img idx, time steps, channels], normalized value
        self.meanPSTHID = monkey_dict['meanPSTHID']     # (50, 49, 53), [id idx, time steps, channels], normalized value
        

    def Monkey_restructure(self, ):
        
        data_path = os.path.join(self.bio_root, 'data.pkl')
        
        if os.path.exists(data_path):
            
            data = utils_.pickle_load(data_path)
            
        else:
            
            data = {
                'FR_count_all': self.FR_countAll,
                'FR_count_base': self.FR_countBase,
                'FR_count_vis': self.FR_countVis,
                'meanFR': self.meanFR,
                'meanBase': self.meanBase,
                'meanGray': self.meanGray,
                'meanVis': self.meanVis,
                'psthTime': self.psthTime,
                'meanPSTH': self.meanPSTH,
                'meanPSTHID': self.meanPSTHID
                }
            
            utils_.pickle_dump(data_path, data)
            
        return data
        
    def Monkey_plot_sample_response(self):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 18})
        
        # --- normalized
        
        norm_factor = np.nanmean(self.meanBase, axis=1)     # (1,53)
        
        meanPSTHIDNorm = np.array([self.meanPSTHID[:,t,:]/norm_factor for t in range(self.meanPSTHID.shape[1])])
        meanPSTHIDNorm = np.mean(meanPSTHIDNorm, axis=2).T      # (ID, )
        
        fig, ax = plt.subplots(figsize=(10,10))
        title = 'Monkey normalized channel-average PSTH'
        plot_PSTH(fig, ax, meanPSTHIDNorm, title, self.psthTime, -50, 200)
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.bio_root, title+'.png'))
        fig.savefig(os.path.join(self.bio_root, title+'.eps'))
    
    #FIXME - add the entrence for other types of distance
    #FIXME, this for loop has problem, this will only take the results of the final iteration
    def monkey_neuron_DSM_process(self, metrics:list[str]=['pearson'], time_bin=10, num_perm=1000):
        """
            this function returns the correlation matrix and triangle from monkey neural responses.
            
            input
                psthTime: 49 time steps for PSTH from -100 ms to 380 ms
                meanPSTH: [500, 49, 53], [img, time steps, channels]
                label: label for 500 imgs
                
            return
                monkey_DM_v: condense form of tranformed DSM
                monkey_DM_v_perm: condense form of transformed DSM with extra dimension of permutation
                monkey_DM_v_temporal: condense form of transformed DSM with temporal dimension
                monkey_DM_v_perm_temporal: condense form of transformed DSM with temporal dimension and permutation dimension
        """
        
        print('[Codinfo] Calculating monkey neuron stats...')
        
        utils_.make_dir(os.path.join(self.bio_root, 'corr'))
        
        for metric in metrics:
            
            utils_.make_dir(os.path.join(self.bio_root, 'corr', f'{metric}'))
        
            file_path = os.path.join(self.bio_root, 'corr', f'{metric}', 'Monkey_DM_dict_qualified_50.pkl')
            
            if os.path.exists(file_path):
                
                results = utils_.pickle_load(file_path)
                
            else:
                # -----
                if time_bin == 10:
                    used_psth = self.meanPSTH[:, [np.where(self.psthTime==_)[0][0] for _ in self.ts], :]
                    
                else:
                    used_psth = np.zeros((self.meanPSTH.shape[0], len(self.ts), self.meanPSTH.shape[2]))     # (500, 26, 53) (img, time, unit)
                    for idx, tt in enumerate(self.ts): 
                        used_psth[:, idx, :] = np.mean(self.meanPSTH[:, np.where(((tt-time_bin/2)<=self.psthTime) & (self.psthTime<=(tt+time_bin/2)))[0], :], axis=1)
                
                used_psth_id = np.array([np.mean(used_psth[np.where(self.label==_)[0], :, :], axis=0) for _ in  range(1, 51)])     
                used_psth_id = np.transpose(used_psth_id, (1,0,2))     # (time, ID, unit)
                
                # [notice] meanGray != np.mean(meanBase, axis=1)
                self.FR_id = np.array([np.mean(self.meanFR[:, np.where(self.label==_)[0]], axis=1)/self.meanGray for _ in range(1, 51)])
                
                scaling_factor = np.mean(self.meanBase,axis=1)
                #sacling_factor = self.meanGray
                
                self.psth_id = np.array([np.array([used_psth_id[i, j, :]/scaling_factor for j in range(50)]) for i in range(used_psth.shape[1])])     # (26, 50, 53)
                
                # --- 
                # for static meanFR
                self.monkey_DM_v = utils_similarity.selectivity_analysis_calculation(metric, self.FR_id)['vector']
                self.monkey_DM_v_perm = np.array([utils_similarity.selectivity_analysis_calculation(metric, self.FR_id[np.random.permutation(self.FR_id.shape[0]),:])['vector'] for _ in range(num_perm)])
        
                # for temporal PSTH
                self.monkey_DM_v_temporal = np.array([utils_similarity.selectivity_analysis_calculation(metric, self.psth_id[_, :, :])['vector'] for _ in range(self.psth_id.shape[0])])
                self.monkey_DM_v_perm_temporal = np.array([np.array([utils_similarity.selectivity_analysis_calculation(metric, self.psth_id[t, np.random.permutation(self.FR_id.shape[0]), :])['vector'] for _ in range(num_perm)]) for t in range(self.psth_id.shape[0])])
                
                # --- seal data
                results = {
                    'monkey_DM_v': self.monkey_DM_v,     # (1225,)
                    'monkey_DM_v_perm': self.monkey_DM_v_perm,      # (1000, 1225)
                    'monkey_DM_v_temporal': self.monkey_DM_v_temporal,     # (26, 1225)
                    'monkey_DM_v_perm_temporal': self.monkey_DM_v_perm_temporal,     # (26, 1000, 1225)
                    'FR_id': self.FR_id,
                    'psth_id': self.psth_id
                    }
                
                utils_.pickle_dump(file_path, results)
            
        return results
        
    def monkey_neuron_DSM_plot(self,):
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 16})
        
        results = self.monkey_neuron_DSM_process()
        
        for metric in results.keys():
            fig, ax = plt.subplots(figsize=(10,10))
            
            title = f'Monkey Distance Matrix | {metric}'
            
            img = ax.imshow(squareform(results[metric]['monkey_DM_v']), aspect='auto', origin='lower')
            
            ax.set_title(title, fontsize=24)
            
            fig.colorbar(img)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.bio_root, f'{title}.png'))
        
 
# ======================================================================================================================    
def plot_PSTH(fig, ax, PSTH, title=None, time_point=None, time_start=None, time_end=None):
    """
        
    """
    img = ax.imshow(PSTH, aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(img)
    
    if time_point is not None:
        
        ax.vlines(np.where(time_point==time_start)[0][0], 0, PSTH.shape[0], linestyle='--', color='red', linewidth=3)
        ax.vlines(np.where(time_point==time_end)[0][0], 0, PSTH.shape[0], linestyle='--', color='red', linewidth=3)
        
        loc = np.arange(np.where(time_point==time_start)[0][0], np.where(time_point==time_end)[0][0])
        loc = np.append(loc, max(loc)+1)
        
        loc = np.where(time_point%50==0)[0]
        ax.set_xticks(loc, time_point[loc], fontsize=14)
        
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('ID',fontsize=20)
    
    ax.set_ylim([0, PSTH.shape[0]-1])
    ax.set_title(title, fontsize=24)
    
    #plt.close()
          
# ======================================================================================================================
def calculate_firing_rate(time_stamps, session_idx, all_periods, time_window, num_frames, PSTH_start, time_step=50):
    """
        - time_stamps: timing of spikes
        - session_idx: the experiment is which one among all 40 sessions
    
        one record: |period_start --- (timestamps of spikes) --- period_end|
        
        Count how many spikes inside the period
        
        [Update Oct 1] tested, this python code is 100% equal with MATLAB code, even considered the different shape of each experiments
                        next, data clean process
        
    """
    
    # ----- 1. initialization
    FR_stats = {}
    periods = all_periods[session_idx-1]     # <- time periods of each trial stored in session_idces
    
    # ----- 2. calculate firing rate (FR)     [original comment] use [250,500] as baseline for most analysis except RQ's criteria
    FR_stats.update({'spike_count': get_normalized_spike_count(time_stamps, periods)})     # [question] why not [750, 1250] since many neuron after 1250 the responses turns weak
    FR_stats.update({'spike_count_250_500': get_normalized_spike_count(time_stamps, periods, [250, 500])})     # [question] why not [0, 500]  
    FR_stats.update({'spike_count_0_2000': get_normalized_spike_count(time_stamps, periods, [0, 2000])})
    
    # ----- 3. calculate peri-stimulus histogram (PSTH)
    if not time_window <= 500:
        raise RuntimeError('[Coderror] time window must be no greater than 500ms in current design')
    else:
        PSTH = np.zeros((len(periods), num_frames))
        for _ in range(num_frames):     # for each frame
            PSTH[:, _] = get_normalized_spike_count(time_stamps, periods, [PSTH_start + _*time_step + 1, _*time_step + 500])     # identical with MATLAB code
        FR_stats.update({f'PSTH_{time_window}':PSTH})
    
    return FR_stats
         
def get_normalized_spike_count(time_stamps, periods, count_period=None):
    """
        returns spike count, as Hz (normalized to counting period) for fixed counting period
    """
    if count_period == None:
        count_period = (750, 1750)
    
    count = extract_period_counts(time_stamps, periods, count_period[0], count_period[1])
    count = count/((count_period[1]-count_period[0])/1000)  #convert to frequency
    
    return count
    
def extract_period_counts(time_stamps, periods, from_, to_):
    """
        Directly converted from matlab code, removed the second condition,
        
        one trial: |0ms---250ms---500ms---750ms(from_)---1000ms---1250ms---1500ms---1750ms(to_)---2000ms|
        
        The 'count_baseline' is a legacy from urut's code (https://www.urut.ch/new/serendipity/) , kept but not in use in outside functions
    """
    len_periods = len(periods)
    from_ = from_*1000     # convert millisecond to microsecond to fit the timescale of the neural equipment
    to_ = to_*1000
    
    count=np.zeros(len_periods)
    #count_baseline=np.zeros(len_periods)     # ideally, this baseline represents the silent brain status

    for _ in range(len_periods):     # [notice] seems can collect temporal information here
        count[_] = np.where((periods[_,1]+from_ < time_stamps) & (time_stamps <= periods[_,1]+to_))[0].size
        #count_baseline[_] = np.where((periods[_,1] < time_stamps) & (time_stamps <= periods[_,1]+from_))[0].size
        
        return count
    
if __name__ == "__main__":

    human_record_process = Human_Neuron_Records_Process()
    
    human_record_process.human_neuron_sort_FR(data_type='base')
    #human_record_process.humane_identity_cell_selection()
    #human_record_process.human_neuron_raster_plot()
    #human_record_process.human_neuron_stacked_encode_map()
    #human_record_process.human_neuron_FR_stats_plot()
    
    human_record_process.human_neuron_DSM_process_sub_id(metrics=['pearson'], used_cell_types=['qualified', 'selective', 'non_selective'], used_id_nums=[10])

    #monkey_record_process = Monkey_Neuron_Records_Process()
    
    #monkey_record_process.Monkey_restructure()
    #monkey_record_process.Monkey_plot_sample_response()
    #monkey_record_process.monkey_neuron_DSM_process(metrics=['euclidean', 'pearson'])
    #monkey_record_process.monkey_neuron_DSM_plot()
