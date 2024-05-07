#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 13:35:41 2023

@author: Runnan Cao

    refer to: https://osf.io/824s7/
    
@modified: acxyle

    ...
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

from scipy.integrate import quad, IntegrationWarning
from sklearn.manifold import TSNE

import utils_
from utils_ import _bio_cells, utils_similarity

import Selectivity_Analysis_Feature


# =============================================================================
class Human_Neuron_Records_Process():
    """
        function: 
            1) convert raw records to response map;
            2) raster plot;
            3) analyze cell features  
        
    """
    
    def __init__(self, bio_root='/home/acxyle-workstation/Downloads/Bio Neuron Data/Human/', seed=6, **kwargs):
        
        np.random.seed(seed)

        self.root_process = os.path.join(bio_root, 'osfstorage-archive-supp/')     # <- contains the processed Bio data (eg. PSTH) calculated from Matlab
        self.root_data = os.path.join(bio_root, 'osfstorage-archive/')      # <- contains the raw Bio data from resources, only used for [human_cell_FiringRate], expand it to PSTH
        
        self.human_neuron_stats = os.path.join(bio_root, 'human_neuron_stats/')
        utils_.make_dir(self.human_neuron_stats)
        
        # -----
        self.baseDir = os.path.join(self.root_process, "Spike Sorting")     # [notice]
        self.dataBaseDir = os.path.join(self.baseDir, 'Sorted Data')
        self.FireDir = os.path.join(self.baseDir, 'FiringRate')
        self.StatsDir = os.path.join(self.baseDir, 'StatsRes/CelebA/unNorm')

        # raw timestamps were stored by .mat format from OSF database, data is sorted on 1-based order of MATLAB
        self.Spikes = sio.loadmat(os.path.join(self.root_data, 'SingleNeuron/Data/Spikes.mat'))  

        # ---
        self.FaceImageIndex = np.array(pd.read_csv(os.path.join(self.root_data, 'Stimuli/FaceImageIndex.csv')))[:, 0]
        
        self.FR_time_range = [750, 1750]
        self.binW = 250
        self.preStim = 500
        self.postStim = 1000
        self.timelim = [0, 2000]
        
        self.ts = np.arange(-250, 1001, 50)
        
        # ---
        CelebA_img_idx_new = sio.loadmat(os.path.join(self.root_data, 'SingleNeuron/Code/CelebA_Image_Code_new.mat'))
        img_idces_new = CelebA_img_idx_new['im_code'].reshape(-1)     # img_idx linked ID, 50 IDs in total
        
        id_img_idces = np.array([_.reshape(-1)-1 for _ in CelebA_img_idx_new['id_code'].reshape(-1)])     # each ID contains 10 imgs
        
        self.img_idces_new_dict = {_:img_idces_new[_] for _ in range(len(img_idces_new))}     # {img number: ID}
        self.id_img_idces_dict = {_:id_img_idces[_] for _ in range(len(id_img_idces))}     # {ID: 10 img numbers}     0-based
        
 
    # ===== module 1, obtain response map
    def human_cell_FeatureMap(self, used_id_num:int=50, used_cell_type:str='all', normalization_method='ap'):
        """
            normalization_method (depends on downstream task):
                1. None
                2. ap: after_stimuli/pre_stimuli. this can enhanced discriminability but may lead to abnormal value, **this is
                the default method in previous work and manually set the abnormal values to zero**.
                3. ae: after_stimuli/entire_trial. this keeps value range but may reduced data discriminability
                
            return:
                meanFR_id: (50, num_cells)
                meanFR_psth: (num_time_steps, 50, num_cells)
 
        """
        
        feature_path = os.path.join(self.human_neuron_stats, f'feature_{used_cell_type}_{normalization_method}.pkl')
        
        if os.path.exists(feature_path):
            
            feature_dict = utils_.load(feature_path)
            
            meanFR_id = feature_dict['FR_id']
            meanFR_PSTH_id = feature_dict['psth_id']
            
        else:
            
            used_ids = self.human_corr_select_sub_identities(used_id_num)
            used_cells = self.human_neuron_obtain_used_cells()[used_cell_type]
            
            # --- init
            meanFR_dict = self.human_cell_SortFR(data_type='default')
            
            meanFR = meanFR_dict['meanFR']     # (2082, 500), with nan
            meanFR_PSTH = meanFR_dict['meanFR_PSTH']     # (2080, 500, time_steps), with nan
        
            # --- normalize firing rates
            if normalization_method is None:
                
                meanFR_id = np.nanmean(meanFR.reshape(-1, 50, 10), axis=2)[np.ix_(used_cells, used_ids)].T     # (num_selected_ids, num_cells)
                meanFR_PSTH_id = np.nanmean(meanFR_PSTH.reshape(-1, 50, 10, 31), axis=2)[np.ix_(used_cells, used_ids)].T     # (num_time_steps, num_selected_ids, num_cells)
                
            elif normalization_method == 'ap':
                
                meanFR_baseline_dict = self.human_cell_SortFR(data_type='base')
                
                FR_baseline = np.nanmean(meanFR_baseline_dict['meanFR'], axis=1)    # (num_cells, 1)
                
                def _further_process(_meanFR, _meanFR_PSTH, fp_type='mutual_truncate'):
                    # this operation causes inf value by devide 0, like one cell has no spike before stimuli but fires after
                    # that. uncomment the prefered process or manually modify even from calculation of firing rates
                    
                    if fp_type is None:
                        processed_meanFR = _meanFR/np.expand_dims(FR_baseline, axis=1)     # (num_cells, num_imgs)
                        processed_meanFR_PSTH = _meanFR_PSTH/np.expand_dims(FR_baseline, axis=(1, 2))     # (num_cells, num_imgs, num_time_steps)
                    
                    elif 'mutual_truncate' in fp_type:
                        
                        epsilon = 1e-5
                        
                        processed_meanFR = _meanFR/np.expand_dims(FR_baseline+epsilon, axis=1)     # (num_cells, num_imgs)
                        processed_meanFR[processed_meanFR>1623] = 1623
                        
                        processed_meanFR_PSTH = _meanFR_PSTH/np.expand_dims(FR_baseline+epsilon, axis=(1, 2))     # (num_cells, num_imgs, num_time_steps)
                        processed_meanFR_PSTH[processed_meanFR_PSTH>7020] = 7020
                        
                    elif 'sdandardization' in fp_type or 'normalization' in fp_type:
                        
                        _meanFR = np.nan_to_num(_meanFR)
                        _meanFR_PSTH = np.nan_to_mean(_meanFR_PSTH)
                        
                        processed_meanFR = _meanFR/np.expand_dims(FR_baseline+epsilon, axis=1)     # (num_cells, num_imgs)
                        processed_meanFR_PSTH = _meanFR_PSTH/np.expand_dims(FR_baseline+epsilon, axis=(1, 2))     # (num_cells, num_imgs, num_time_steps)
                    
                        if 'sdandardization' in fp_type:
                        
                            processed_meanFR = (processed_meanFR-np.mean(processed_meanFR)) / np.sqrt(np.var(processed_meanFR)+epsilon)
                            processed_meanFR_PSTH = (processed_meanFR_PSTH-np.mean(processed_meanFR_PSTH)) / np.sqrt(np.var(processed_meanFR_PSTH)+epsilon)
                        
                        elif 'normalization' in fp_type:
                            processed_meanFR = (processed_meanFR-np.min(processed_meanFR)) / (np.max(processed_meanFR) - np.min(processed_meanFR))
                            processed_meanFR_PSTH = (processed_meanFR_PSTH-np.min(processed_meanFR_PSTH)) / (np.max(processed_meanFR_PSTH) - np.min(processed_meanFR_PSTH))
                            
                    return processed_meanFR, processed_meanFR_PSTH
                
                processed_meanFR, processed_meanFR_PSTH = _further_process(meanFR, meanFR_PSTH)
                        
                meanFR_id = np.nanmean(processed_meanFR.reshape(-1, 50, 10), axis=2)[np.ix_(used_cells, used_ids)].T     # (num_selected_ids, num_cells)
                meanFR_PSTH_id = np.nanmean(processed_meanFR_PSTH.reshape(-1, 50, 10, 31), axis=2)[np.ix_(used_cells, used_ids)].T     # (num_time_steps, num_selected_ids, num_cells)
                
            elif normalization_method == 'ae':
                
                meanFR_baseline_dict = self.human_cell_SortFR(data_type='trial')
                
                FR_trial = np.nanmean(meanFR_baseline_dict['meanFR'], axis=1)    # (num_cells, 1)
                FR_trial = np.array([_ if _!=0 else np.nanmean(FR_trial) for _ in FR_trial])
                
                trial_meanFR = meanFR/np.expand_dims(FR_trial, axis=1)     # (num_cells, num_imgs)
                trial_meanFR_PSTH = meanFR_PSTH/np.expand_dims(FR_trial, axis=(1, 2))     # (num_cells, num_imgs, num_time_steps)
                
                # --- -> (50, num_cells)
                meanFR_id = np.nanmean(trial_meanFR.reshape(-1, 50, 10), axis=2)[np.ix_(used_cells, used_ids)].T     # (num_selected_ids, num_cells)
                meanFR_PSTH_id = np.nanmean(trial_meanFR_PSTH.reshape(-1, 50, 10, 31), axis=2)[np.ix_(used_cells, used_ids)].T     # (num_time_steps, num_selected_ids, num_cells)
                
            # -----
            feature_dict = {
                'FR_id': meanFR_id,
                'psth_id': meanFR_PSTH_id
                }
            
            utils_.dump(feature_dict, feature_path)
        
        return meanFR_id, meanFR_PSTH_id
    
    
    # ----- 1. calculate FR from raw records
    def human_cell_SortFR(self, reject_rate:float=0.15, data_type:str='default'):
        """
            currently the results is not identical with the MATLAB version, need to fix, and upgrade
            -----
            this function calculates the sorted mean firing rates (entire trial: 0ms - 2000ms) and qualified cells
            
            this function use reject_rate (default: 0.15) and experiment performance to filter unwanted cells. After 
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
            - cells: 1,577 qualified cell idces from all 2,082 cells
            
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
            
            meanFR_dict = utils_.load(meanFR_path)
            
        else:
            
            print('[Codinfo] Calculating sorted meanFR...')
            
            # ----- obtain firing rate (FR) and peri-stimulus histogram (PSTH)
            FR_stats = self.human_cell_FiringRate()
            
            if data_type == 'base':
                FR_list = [FR_stats[_]['spike_count_250_500'] for _ in range(len(FR_stats))]
            elif data_type == 'trial':
                FR_list = [FR_stats[_]['spike_count_0_2000'] for _ in range(len(FR_stats))]
            elif data_type == 'default':
                FR_list = [FR_stats[_]['spike_count'] for _ in range(len(FR_stats))]
            else:
                raise ValueError(f'[Codinfo] invalid data_type [{data_type}]')
                
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
            self.CelebA_img_idces = sio.loadmat(os.path.join(self.root_data, 'SingleNeuron/Code/CelebA_Image_Code.mat'))
            img_idces = self.CelebA_img_idces['im_code'].reshape(-1)     # [idx as right number: ID]
            adjust_idx = self.CelebA_img_idces['AdjustInd'].reshape(-1).astype(np.int16) - 1   # [idx as right number: wrong number]
            
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
            # to one class, below section sorts feature map based on ID. The final output has similiar order as NN feature map
            
            # --- 5. sort the feature map based on ID
            for cell_idx in tqdm(range(meanFR.shape[0]), 'Sorting', total=meanFR.shape[0]):
                
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
                
            # -----
            meanFR_dict = {
                'meanFR': meanFR,
                'meanFR_PSTH': meanFR_PSTH,
                'qualified_cells': qualified_cells,
                }
            
            utils_.dump(meanFR_dict, meanFR_path)
                    
        return meanFR_dict
        
    
    def human_cell_FiringRate(self, time_window=250, time_step=50):  
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

            time_stamps_all_cells = [_.reshape(-1) for _ in self.Spikes['timestampsOfCellAll'].reshape(-1)]
            
            sessions = _bio_cells.get_session_idces()     # <- stores time periods of each trial (each session_idx)
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
                
            utils_.dump(FR_stats, file_path)
        
        return FR_stats
        
    
    def human_plot_FR_PDF(self, init:float=0.15):
        """
            1. bio data > 0, so use feature/scaling_factor rather than (feature-min(feature))/scaling_factor
            2. this function does not remove the responses of repeated images
        """
        
        FR_stats = self.human_cell_FiringRate()
        
        feature = [FR_stats[_]['spike_count_0_2000'] for _ in range(len(FR_stats))]     
        feature = np.array([np.mean(_[~np.isnan(_)]) for _ in feature])     # (2082,)
        
        scaling_factor = (np.max(feature)-np.min(feature))
        
        init_th = init/scaling_factor
        feature = feature/scaling_factor     # (0, 1)
        
        fig = self.plot_FR_PDF(feature=feature, init_threshold=init_th, scaling_factor=scaling_factor)
        plt.tight_layout()
        fig.savefig(os.path.join(self.human_neuron_stats, 'neuron PDF.svg'))
    
    
    # FIXME --- need to simplify
    @staticmethod
    def plot_FR_PDF(model_structure:str='human MTL', target:str='cell', feature:np.array=None, init_threshold=None, scaling_factor:float=1., layer='', unit_type='', **kwargs) -> None:
        """
            this function plots the log gaussian hist and PDF of human cell
        """
        
        warnings.filterwarnings('ignore', category=IntegrationWarning)     # [caution]
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 18})
        
        if feature.ndim == 1:
            feature = feature
        elif feature.ndim ==2 and feature.shape[0]==500:     # assume the shape is (num_samples, num_features)
            feature = np.mean(feature, axis=0)
        else:
            raise RuntimeError(f'[Coderror] invalid feature shape {feature.shape}')
        
        # -----
        def _hist(feature):
            
            kde = gaussian_kde(feature)
            feature_mean = np.mean(feature)
            feature_std = np.std(feature)
            feature_radius = max(np.max(feature) - feature_mean, feature_mean - np.min(feature))
            x = np.linspace(feature_mean - feature_radius, feature_mean + feature_radius, 1000)
            y_kde = kde(x)
            
            return kde, x, y_kde, feature_mean, feature_std, feature_radius
        
        def _plot_hist(ax, feature, title, plot_fitted_PDF=False, plot_legend=False, init_threshold=None):
            
            kde, x, y_kde, feature_mean, feature_std, feature_radius = _hist(feature)
            
            # ---
            (hist_pct, hist_x, _) = ax.hist(feature, bins=100, density=True)
            ylim_max = max(np.ceil(np.max(hist_pct) / 10) * 10, 1.5 * np.max(hist_pct)) if np.max(hist_pct) > 5 else np.max(hist_pct) * 1.5
            ax.set_ylim([0, ylim_max])
            ax.set_title(title, fontsize=24)
            ylim_max_auto = ax.get_ylim()[1]
            
            # ---
            if plot_fitted_PDF:
                ax.vlines(np.mean(feature), 0, ylim_max_auto, color='red', label='mean')
                y_norm = stats.norm.pdf(x, feature_mean, feature_std)
                ax.plot(x, y_kde, linestyle='--', linewidth=2, color='orange', label='gaussian_kde')
                ax.plot(x, y_norm, linestyle='--', linewidth=2, color='red', label='gaussian_fit')
            
            # ---
            if plot_legend:
                for i in range(3):
                    pct_below = quad(kde, -np.inf, feature_mean - (i + 1) * feature_std)[0] * 100
                    pct_above = quad(kde, feature_mean + (i + 1) * feature_std, np.inf)[0] * 100
                    ax.vlines(feature_mean - (i + 1) * feature_std, 0, ylim_max_auto, linestyle='dotted', color='gold', label=f'p < mean-{i+1}std: {pct_below:.2f}%')
                    ax.vlines(feature_mean + (i + 1) * feature_std, 0, ylim_max_auto, linestyle='dotted', color='purple', label=f'p > mean+{i+1}std: {pct_above:.2f}%')
                ax.legend(framealpha=0.5)
                ax.set_xlim([feature_mean - feature_radius, feature_mean + feature_radius])
        
            # ---
            if init_threshold:
                pct_init = quad(kde, -np.inf, np.log10(init_threshold))[0] * 100
                ax.vlines(np.log10(init_threshold), 0, ylim_max_auto, linestyle='dotted', color='red', label=f'manual value of {init_threshold:.2f} ({pct_init:.2f}%)')
                ax.fill_between(x, y_kde, where=(x < np.log10(init_threshold)), color='gray', alpha=0.5)
  
            return kde, x, y_kde, feature_mean, feature_std, feature_radius
        
        # -----
        suptitle = f"{model_structure} {layer} {unit_type} {target} PDF"
        
        if np.any(feature<0):     # -> details for original data, simple for log data
            
            # --- original
            fig, ax = plt.subplots(1, 2, figsize=(20,10))
            
            _plot_hist(ax[0], feature, 'hist of original data', True, True, init_threshold=init_threshold)
            
            # --- log
            feature_log = np.log10(feature[feature>0])
            
            _plot_hist(ax[1], feature_log, f'log10 hist and gaussian kde exclude <=0 ({(feature.size-feature_log.size)/feature.size*100:.2f}%)', True, init_threshold=init_threshold)
            
            fig.suptitle(suptitle, y=0.975, fontsize=28)
            
        else:   # -> simple for original data, details for log data
    
            fig, ax = plt.subplots(1, 2, figsize=(20,10))
            
            _plot_hist(ax[0], feature, 'hist of original data', True, init_threshold=init_threshold)
            
            # --- log
            feature_log = np.log10(feature[feature>0])
            
            _plot_hist(ax[1], feature_log, f'log10 hist and gaussian kde excluse <=0 ({(feature.size-feature_log.size)/feature.size*100:.2f}%)', True, True, init_threshold=init_threshold)
  
            fig.suptitle(suptitle, y=0.975, fontsize=28)
            
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
        
        file_path = os.path.join(self.human_neuron_stats, 'beh_stats.pkl')
        
        if os.path.exists(file_path):
            beh_stats = utils_.pickle_load(file_path)
            
        else:
        
            beh_dict = []
            Acc = []
            
            sessions = _bio_cells.get_session_idces()     # <- stores time periods of each trial
            
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

        return behavior_data_dict
    
    
    def behavior_data_dict_correction(self, path:str):
        """
            input: 
                path of the behavior record of one session
            
            output: 
                python dict of needed variables of one session
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
    

    # -----
    def human_plot_pie_chart(self, ):
        """
            ...
        """

        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 14})
        
        cell_stats = self.human_identity_cell_selection()

        bio_pct = cell_stats['cell_types_dict']
        
        tmp = [bio_pct[_] for _ in bio_pct.keys() if 'non' in _]
        tmp = [__ for _ in tmp for __ in _]
        
        bio_pct_new = {}

        for _ in bio_pct.keys():
            if 'non' not in _:
                bio_pct_new.update({_: bio_pct[_]})
        
        bio_pct_new.update({'non_encode': np.array(tmp)})
        
        values = [len(bio_pct_new[_]) for _ in bio_pct_new.keys()]
        labels = [f's_si ({values[0]})', f's_wsi ({values[1]})', f's_mi ({values[2]})', f's_wmi ({values[3]})', f'n_e ({values[4]})']
        colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']
        explode = np.array([0.5, 0.1, 0.5, 0.1, 0.])
        
        title = f'{np.sum(values)} Cells'

        fig, ax = plt.subplots(figsize=(10,6))
        utils_.plot_pie_chart(fig, ax, values, labels, title, colors, explode)
        fig.savefig(os.path.join(self.human_neuron_stats, 'bio_pct_pie_chart.svg'), transparent=True)
        plt.close()
        
        
    # ===== module 2. obtain identity-selective cells
    def human_identity_cell_selection(self, ):
        """
            this function aims to capture *identity_sensitive cells, identity_encode cells and identity_selective cells*
            
            the original ANOVA(MATLAB) code is based on: https://uk.mathworks.com/matlabcentral/fileexchange/22088-repeated-measures-anova?s_tid=mwa_osa_a
            
                "This program was originally released when MATLAB had no support for Repeated Measures ANOVA. However, 
                since a few releases ago, MATLAB statistics toolbox has added this functionality (see the fitrm function). 
                Thus this program is now deprecated and is not recommended anymore. The issue is that it only support a 
                very small subclass of the problems that fitrm can solve. Also, it might not have been tested as extensively 
                as fitrm so it is possible that it does not produce correct results in all cases."
            
            instead, the ANOVA function used in this Python code is f_oneway(). The statistic values are **significantly** 
            different from the results from anova_rm() of MATLAB, but the selected cells are highly consistent. Use any 
            stats package or manually write ANOVA code depends on purpose, like:
                
                - f_oneway(): merge all data and analysis in one time (default)
                - smf(): mixed effect model
                - ...
            
        """

        save_path = os.path.join(self.human_neuron_stats, 'cell_stats.pkl')
        
        if os.path.exists(save_path):
            
            cell_stats = utils_.load(save_path, verbose=False)
            
        else:
            
            def _get_session_attr():

                sessions = _bio_cells.get_session_idces()
                
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
            
            # -----
            meanFR_dict = self.human_cell_SortFR(data_type='default')
            
            meanFR = meanFR_dict['meanFR']
            qualified_cells = meanFR_dict['qualified_cells']
            
            # -----
            neuron_session_idces = self.Spikes['vCell'].reshape(-1) - 1     # 0-based to 1-based
            sessions = _bio_cells.get_session_idces()
            
            sessions_attr = _get_session_attr()
            
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
            
            p_list = []
            encode_id = {}
           
            for cell_idx in tqdm(range(meanFR.shape[0]), desc='cell ANOVA'):     # for each cell
                
                meanFR_single_cell = list(meanFR[cell_idx].reshape(50, 10))    # list, (50, 10)

                # ----- 1. one way ANOVA
                p = stats.f_oneway(*meanFR_single_cell)[1]    
                p_list.append(p)
                
                # ----- 2. mean+2SD,  use: | si | wsi | mi | wmi | n |
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
                's_si': s_si,
                's_wsi': s_wsi,
                's_mi': s_mi,
                's_wmi': s_wmi,
                's_non_encode': s_non_encode,
                'ns_si': ns_si,
                'ns_wsi': ns_wsi,
                'ns_mi': ns_mi,
                'ns_wmi': ns_wmi,
                'ns_non_encode': ns_non_encode
                }
            
            cell_stats = {
                'cell_attr': cell_attr,
                'encode_id': encode_id,
                'cell_types_dict': cell_types_dict
                }
            
            utils_.dump(cell_stats, save_path)
        
        return cell_stats
        
    
    #FIXME --- refer to the fig of NN
    def human_cell_stacked_responses(self, num_types=5):
        """
            ...
        """
           
        # ---
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 14})

        # --- load cell_stats
        cell_stats = self.human_identity_cell_selection()
        idx_dict = cell_stats['cell_types_dict']
        
        # --- load feature
        meanFR_dict = self.human_cell_SortFR(data_type='default')
        meanFR = meanFR_dict['meanFR']

        feature = meanFR.T     
        feature[np.isnan(feature)] = 0
        
        # ---
        if num_types == 5:
           
            idx_dict = {
                's_si': idx_dict['sensitive_si'],
                's_wsi': idx_dict['sensitive_wsi'],
                
                's_mi': idx_dict['sensitive_mi'],
                's_wmi': idx_dict['sensitive_wmi'],
                
                'n_e': np.concatenate((idx_dict['sensitive_non_encode'], 
                        idx_dict['non_sensitive_si'], 
                        idx_dict['non_sensitive_wsi'], 
                        idx_dict['non_sensitive_mi'], 
                        idx_dict['non_sensitive_wmi'], 
                        idx_dict['non_sensitive_non_encode'])).astype(np.int64)
                }
        
            # init the canvas
            fig, ax = plt.subplots(figsize=(26, 6))
            gs_main = gridspec.GridSpec(1, 5, figure=fig)
    
            plot_single(fig, gs_main, 5, idx_dict, feature, 50, 10)
        
        
        elif num_types == 10:

            idx_dict = {
                's_si': idx_dict['sensitive_si'],
                's_wsi': idx_dict['sensitive_wsi'],
                
                's_mi': idx_dict['sensitive_mi'],
                's_wmi': idx_dict['sensitive_wmi'],
                
                's_non_encode': idx_dict['sensitive_non_encode'],
                
                'ns_si': idx_dict['non_sensitive_si'],
                'ns_wsi': idx_dict['non_sensitive_wsi'],
                
                'ns_mi': idx_dict['non_sensitive_mi'],
                'ns_wmi': idx_dict['non_sensitive_wmi'],
                
                'ns_non_encode': idx_dict['non_sensitive_non_encode']
                }
        
            # init the canvas
            fig, ax = plt.subplots(figsize=(26,10))
            gs_main = gridspec.GridSpec(2, 5, figure=fig)
    
            plot_single(fig, gs_main, 5, idx_dict, feature, 50, 10)
        
        # -----
        ax.axis('off')
        ax.plot([],[],color='blue', linestyle='--', label='mean')
        ax.plot([],[],color='teal', linestyle='--', label='ref')
        ax.plot([],[],color='red', linestyle='--', label='threshold')
        
        fig.suptitle('Human MTL Neuron Responses for Human Faces', y=0.97, fontsize=20)
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, bbox_transform=plt.gcf().transFigure)
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.human_neuron_stats, f'Human MTL Neuron Responses {num_types} types.svg'), bbox_inches='tight')
        plt.close()


    # ===== module 3. raster plot
    def human_neuron_raster_plot(self, type_to_plot:list=None, plot_cell_num:int=10):
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
        FR_stats = self.human_cell_FiringRate()
        cell_stats = self.human_identity_cell_selection()

        # --- preperation
        neuron_session_idces = self.Spikes['vCell'].reshape(-1) - 1    # <- session_idx ID, 0-based
        time_stamps_all_cells = [_.reshape(-1) for _ in self.Spikes['timestampsOfCellAll'].reshape(-1)]
        
        sessions = _bio_cells.get_session_idces()     # <- stores time periods of each trial (each session_idx)
        session_idx_dir = os.path.join(self.root_data, 'Events Files')     # <- store timestamps for each responses, self.root_data: osfstorage-archive
        
        all_periods = []     # provides the time range of one single trial with label
        for session_idx in tqdm(range(len(sessions)), desc='Load sessions'):
            # ↓ 'periods' contains 3 columns: trial indices, timestamps to 500 ms before stumuli onset, timestamps to 1500 ms after stimuli onset
            periods = sio.loadmat(os.path.join(session_idx_dir, sessions[session_idx]+'.mat'))['periods']     
            all_periods.append(periods)
    
        beh_stats = self.human_neuron_get_beh()
        behavior = beh_stats['beh']
        
        # ----- build [im_code] img_idces and img_idces_new dict
        self.CelebA_img_idces = sio.loadmat(os.path.join(self.root_data, 'SingleNeuron/Code/CelebA_Image_Code.mat'))
        img_idces = self.CelebA_img_idces['im_code'].reshape(-1)     # img_idx linked ID, 53 IDs in total, 3 unwanted
        adjust_idx = self.CelebA_img_idces['AdjustInd'].reshape(-1).astype(np.int16) -1   # [notice] what the adjust_idx is?
        
        img_idces_dict = {_:img_idces[_] for _ in range(len(img_idces))}     # {img number: ID}
        adjust_idx_dict = {adjust_idx[_].astype(int): _ for _ in range(len(adjust_idx)) if adjust_idx[_] > -1}     # {wrong number: right number}
        
        # ---
        CelebA_img_idx_new = sio.loadmat(os.path.join(self.root_data, 'SingleNeuron/Code/CelebA_Image_Code_new.mat'))
        img_idces_new = CelebA_img_idx_new['im_code'].reshape(-1)     # img_idx linked ID, 50 IDs in total
        
        id_img_idces = CelebA_img_idx_new['id_code'].reshape(-1)     # each ID contains 10 imgs
        id_img_idces = np.array([_.reshape(-1)-1 for _ in id_img_idces])
        
        img_idces_new_dict = {_:img_idces_new[_] for _ in range(len(img_idces_new))}     # {img number: ID}
        id_img_idces_dict = {_:id_img_idces[_] for _ in range(len(id_img_idces))}     # {ID: 10 img numbers}     0-based
        
        # ----- start plot for different types of cells
        if type_to_plot == None:
            type_to_plot = list(cell_stats['cell_types_dict'].keys())    
        
        # ---
        for cell_type in type_to_plot:  # for each type
        
            cell_type_plot_path = os.path.join(cell_plot_path, f'{cell_type}')
            utils_.make_dir(cell_type_plot_path)
        
            if len(cell_stats['cell_types_dict'][cell_type]) > plot_cell_num:
                cell_to_plot = np.random.choice(cell_stats['cell_types_dict'][cell_type], plot_cell_num)
            
            elif len(cell_stats['cell_types_dict'][cell_type]) != 0:
                cell_to_plot = cell_stats['cell_types_dict'][cell_type]
                
            else:
                cell_to_plot = []
                
            for cell_idx in tqdm(cell_to_plot, desc=f'{cell_type}'):     # for each cell

                session_idx = neuron_session_idces[cell_idx]  # session_idx
                
                # ----- im_code: key is the img label, value is the corresponding ID label
                if session_idx < 10:
                    im_code = img_idces_dict
                else:
                    im_code = img_idces_new_dict
            
                time_stamps = time_stamps_all_cells[cell_idx]
                
                periods = all_periods[session_idx]     # e.g (550, 3)
                
                displayed_image_sequence = behavior[session_idx]['code']     # the displayed img list
                back_id = behavior[session_idx]['back_id']     # local version, 0-based
                
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
                    
                # ----- return jagged periods_and_infos
                periods_and_infos = []
                
                for ID in range(1, 51):    # for each ID
                    
                    positions = np.where(displayed_ID_sequence == ID)[0]
                    
                    img_labels = displayed_image_sequence[positions]
                    _ID = np.array([im_code[_] for _ in img_labels])
                    
                    if np.unique(_ID) != ID:
                        raise RuntimeError(f"ID check failed for '{ID}'")
                    
                    period = periods[positions, :]     # order in experiment | trial start time | trial end time
                    period = np.hstack((period, np.vstack((img_labels, _ID)).T))     # | order in experiment | trial start time | trial end time | img label | ID | session |
                    
                    periods_and_infos.append(period[period[:,0].argsort()])     # legacy design from MATLAB code, not necessary
    
                # ----- this should return neat spikes_to_plot of (500,)
                spikes_to_plot = _bio_cells.getTimestampsOfBubbles(time_stamps, periods_and_infos, session_idx, adjust_idx_dict, id_img_idces_dict)
            
                # ----- plot
                fig = plt.figure(figsize=(30, 20))
    
                # --- subplot_1
                axes_0 = plt.gcf().add_axes([0.05, 0.45, 0.55, 0.475])
                _bio_cells._plot_spike_raster(axes_0, spikes_to_plot, colors)

                # --- subplot_2
                axes_1 = plt.gcf().add_axes([0.65, 0.45, 0.3, 0.475])
                _bio_cells._plot_bar_chart(axes_1, FR_ID, colors)
               
                # --- subplots_3
                axes_2 = plt.gcf().add_axes([0.05, 0.05, 0.55, 0.325])
                _bio_cells._plot_psth(fig, axes_2, PSTH_ID)
   
                # --- subplot_4
                # ----- the displayed statistics exclude back_id
                axes_3 = plt.gcf().add_axes([0.7, 0.05, 0.25, 0.325])
                _bio_cells._plot_table(axes_3, cell_stats, cell_idx, FR_stats, FR_tmp)
                
                # --- title
                cell_attr = cell_stats['cell_attr'][cell_idx]
                fig.suptitle(f"Cell No. {cell_idx+1} | Session No. {cell_attr['session_idx']+1} | Session: {cell_attr['session']} | ({cell_attr['patient_session_idx']+1}/{cell_attr['patient_sessions_num']})", fontsize=32)
                
                with warnings.catch_warnings():
                    warnings.simplefilter(action='ignore')
                    
                    plt.tight_layout()

                    plt.savefig(cell_type_plot_path + f'/{cell_idx}.svg', bbox_inches='tight')
                    plt.close()
                    
    
    # ===== module 4. corr
    def human_neuron_DSM_process(self, metric='pearson', used_cell_type='qualified', used_id_num=50, permutation=True, num_perm=1000, vectorize=False, **kwargs):
        """
            this function calculates the human pairwise distance matrices based on used_ids and used_cell_type, 
            
            input:
                - used_ids: default all 50 ids
                - used_cell_type: default qualified 1,577 cells
                
            this permutation process inherit from original Matlab code, permute Bio data to check whether the NN data 
            aligns with the pattern of Bio data, and vice versa (not very sure :P). 
            
            this process depends on downstream tasks and research purposes.
        """
        # --- 
        save_root = os.path.join(self.human_neuron_stats, 'corr')
        utils_.make_dir(save_root)
        
        save_root = os.path.join(save_root, metric)
        utils_.make_dir(save_root)

        # --- init
        used_ids = self.human_corr_select_sub_identities(used_id_num)
        used_cells = self.human_neuron_obtain_used_cells()[used_cell_type]
            
        if used_cells.size == 0:
            
            return None
    
        else:
            
            save_path = os.path.join(save_root, f'Human_DM_dict_{used_cell_type}_{used_id_num}.pkl')

            if os.path.exists(save_path):
                
                human_DM_dict = utils_.load(save_path)
                
            else:
                
                # --- 
                FR_id, psth_id = self.human_cell_FeatureMap()
                
                FR_id = FR_id[np.ix_(used_ids, used_cells)]
                psth_id = np.array([_[np.ix_(used_ids, used_cells)] for _ in psth_id])
                
                # ---
                num_samples = FR_id.shape[0]
                num_steps = psth_id.shape[0]
                
                human_DM_dict = {
                    'human_DM': utils_similarity.DSM_calculation(metric, FR_id, vectorize=vectorize), 
                    'human_DM_temporal': np.array([utils_similarity.DSM_calculation(metric, psth_id[_, :, :], vectorize=vectorize) for _ in range(num_steps)]),
                    }
                
                if permutation:
                
                    human_DM_dict.update({
                        'human_DM_perm': np.array([utils_similarity.DSM_calculation(metric, FR_id[np.random.permutation(num_samples)], vectorize=vectorize) for _ in range(num_perm)]),
                        'human_DM_temporal_perm': np.array([np.array([utils_similarity.DSM_calculation(metric, psth_id[t, np.random.permutation(num_samples), :], vectorize=vectorize) for _ in range(num_perm)]) for t in range(num_steps)])
                        })
                
                utils_.dump(human_DM_dict, save_path)
            
            return human_DM_dict
    
    
    def human_corr_select_sub_identities(self, used_id_num:int=None, cell_type='selective_cells'):
        """
            Dr CAO provided: [5, 9, 13, 14, 22, 23, 27, 35, 37, 39]
            Calculated here: [5, 9, 13, 14, 23, 27, 29, 35, 42, 44]
            
            return:
                dict
        """
        # --- init
        self.cell_stats = self.human_identity_cell_selection()
        #self.FR_stats = self.human_cell_FiringRate()
        
        encode_dict = self.cell_stats['encode_id']     # encode_dict
        cell_types_dict = self.cell_stats['cell_types_dict']     # ID cells
        
        # --- rebuild wanted cells
        all_selective_cells = {_:cell_types_dict[_] for _ in cell_types_dict.keys() if 'ns' not in _ and 'non_encode' not in _}
        all_selective_cells = [__ for _ in all_selective_cells.values() for __ in _]     # (155,)
        
        # --- calculate
        if cell_type == 'selective_cells':     # for ID encoded by ID cells
            encoded_id_pool = np.concatenate(np.array([[__ for _ in encode_dict[cell_idx].values() for __ in _] for cell_idx in all_selective_cells], dtype=object))
        elif cell_type == 'encode_cells':     # for ID encoded by all Encode cells
            encoded_id_pool = np.concatenate(np.array([[__ for _ in encode_dict[cell_idx].values() for __ in _] for cell_idx in encode_dict.keys()], dtype=object))
 
        # ----- select used_id_num
        if used_id_num is None:
            used_ids = [6, 10, 14, 15, 23, 24, 28, 36, 38, 40]
        elif used_id_num is not None and used_id_num > 48:     # [notice] 'selective' human cells only encode 48 ids
            used_ids = list(np.arange(50))
        elif used_id_num is not None:
            used_ids = list(used_ids_selection(encoded_id_pool, used_id_num).keys())[:used_id_num]
        else:
            raise RuntimeError('[Coderror] invalid used_id_num')
        
        return sorted(used_ids)
    

    def human_neuron_obtain_used_cells(self, ):
        """
            for bio cells, 'qualified' != 'all'
        """

        if not hasattr(self, 'cell_stats'):
            self.cell_stats = self.human_identity_cell_selection()
        
        units_type_dict = self.cell_stats['cell_types_dict']
        
        units_type_dict['qualified'] = np.array([_ for _ in list(self.cell_stats['encode_id'].keys())])
        units_type_dict['all'] = np.arange(2082)
        
        upgraded_cell_types_dict = {
                'selective': ['s_si', 's_wsi', 's_mi', 's_wmi'],
                'non_selective': ['s_non_encode', 'ns_si', 'ns_wsi', 'ns_mi', 'ns_wmi', 'ns_non_encode'],
                'strong_selective': ['s_si', 's_mi'],
                'weak_selective': ['s_wsi', 's_wmi'],
                
                'sensitive': ['s_si', 's_wsi', 's_mi', 's_wmi', 's_non_encode'],
                'non_sensitive': ['ns_si', 'ns_wsi', 'ns_mi', 'ns_wmi', 'ns_non_encode'],
                'all_sensitive_si': ['s_si', 's_wsi'],
                'all_sensitive_mi': ['s_mi', 's_wmi']
                                    }
        
        for k, v in upgraded_cell_types_dict.items():
            units_type_dict[k] = np.concatenate([self.cell_stats['cell_types_dict'][_] for _ in v])
        
        return units_type_dict
    
    
    # ------------------------------------------------------------------------------------------------------------------
    def human_neuron_Gram_process(self, kernel='linear', used_cell_type='qualified', used_id_num=50, permutation=True, num_perm=1000, save=True, **kwargs):
        """
            this permutation process inherit from original Matlab code, permute Bio data to check whether the NN data 
            aligns with the pattern of Bio data, and vice versa (not very sure :P). 
            
            this process depends on downstream tasks and research purposes.
        """
        
        utils_.make_dir(os.path.join(self.human_neuron_stats, 'gram'))
        utils_.make_dir(os.path.join(self.human_neuron_stats, 'gram', f'{kernel}'))
        
        if kernel == 'rbf' and 'threshold' in kwargs:
            file_path = os.path.join(self.human_neuron_stats, 'gram', f'{kernel}', f"CKA_results_{kernel}_{kwargs['threshold']}_{used_cell_type}_{used_id_num}.pkl")
        elif kernel == 'linear':
            file_path = os.path.join(self.human_neuron_stats, 'gram', f'{kernel}', f"CKA_results_{kernel}_{used_cell_type}_{used_id_num}.pkl")
        else:
            raise ValueError(f'[Coderror] Invalid kernel [{kernel}]')

        if os.path.exists(file_path):
            
            results = utils_.load(file_path)
            
        else:
            
            used_ids = self.human_corr_select_sub_identities(used_id_num)
            used_cells = self.human_neuron_obtain_used_cells()[used_cell_type]
            
            FR_id, psth_id = self.human_cell_FeatureMap()
            
            FR_id = FR_id[np.ix_(used_ids, used_cells)]
            psth_id = np.array([_[np.ix_(used_ids, used_cells)] for _ in psth_id])
            
            if kernel == 'linear':
                gram = utils_similarity.gram_linear
            elif kernel =='rbf':
                gram = utils_similarity.gram_rbf
            
            num_samples = FR_id.shape[0]
            num_steps = psth_id.shape[0]
            
            results = {
                'human_Gram': gram(FR_id, **kwargs), 
                'human_Gram_temporal': np.array([gram(psth_id[_, :, :], **kwargs) for _ in tqdm(range(num_steps), desc='human_Gram_temporal')]),
                }
            
            if permutation:
            
                results.update({
                    'human_Gram_perm': np.array([gram(FR_id[np.random.permutation(num_samples)], **kwargs) for _ in tqdm(range(num_perm), desc='human_Gram_perm')]),
                    'human_Gram_temporal_perm': np.array([np.array([gram(psth_id[t, np.random.permutation(num_samples), :], **kwargs) for _ in range(num_perm)]) for t in tqdm(range(num_steps), desc='human_Gram_temporal_perm')])
                    })
            
            if save:
                
                utils_.dump(results, file_path)
            
        return results
                                
    # ------------------------------------------------------------------------------------------------------------------
    def human_DR(self, NN_folder, layer='neuron_2'):
        
        coor_name = ''.join([NN_folder.split(' ')[-1], ' ', layer])
        tsne_dict = utils_.load(f'/home/acxyle-workstation/Downloads/{NN_folder}/Analysis/Dimension_Reduction/TSNE/tsne_all.pkl')
        
        # --- default: all
        tsne = tsne_dict[layer]['tsne_dict']['all']
        
        self.human_DR_single(coor_name, tsne)
        

    # FIXME --- need to simplify
    def human_DR_single(self, coor_name:str=None, tsne:np.array=None):
        """
            this function generates the coordinates based on the 
        """
        
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        id_labels = np.arange(1, 51)
        img_labels = np.array([np.array([_]*10) for _ in id_labels]).reshape(-1)
        
        meanFR_dict = self.human_cell_SortFR()
        cell_stats = self.human_identity_cell_selection()
        
        meanFR = meanFR_dict['meanFR']
        qualified_cells = meanFR_dict['qualified_cells']
        
        self.DR_save_folder = os.path.join(self.human_neuron_stats, 'DR results')
        utils_.make_dir(self.DR_save_folder)
        
        self.tsne_folder = os.path.join(self.DR_save_folder, 'TSNE')
        utils_.make_dir(self.tsne_folder)
        
        # -----
        if coor_name is None and tsne is None:
            
            meanFR_ = np.nan_to_num(meanFR[qualified_cells, :])
            
            perplexity = np.min([np.sqrt(len(qualified_cells)), 50*10-1])
            
            # --- local coordinates
            tsne = TSNE(perplexity=perplexity).fit_transform(meanFR_.T)
            
            coor_name = 'human_coor'

        else:
            
            assert isinstance(tsne, np.ndarray) and coor_name is not None
            
        self.tsne_save_folder = os.path.join(self.tsne_folder, coor_name)
        utils_.make_dir(self.tsne_save_folder)
            
        # ----- p_values
        # --- init
        DR_sub_type = 'all'
        layer = 'neuron_2'
        sq = 0.035
        
        feature = np.nan_to_num(self.human_cell_SortFR()['meanFR']).T
        
        file_path = os.path.join(self.tsne_save_folder, f'{layer}_{DR_sub_type}_sq{sq}.pkl')
        
        if os.path.exists(file_path):
            
            results = utils_.pickle_load_tqdm(file_path)
            
        else:
            
            kernel_size, kernel_sigma = Selectivity_Analysis_Feature.Selectivity_Analysis_Feature.get_kernel_size(tsne)
            gaussian_kernel = Selectivity_Analysis_Feature.Selectivity_Analysis_Feature.gausskernel(kernel_size, kernel_sigma)
            
            # --- calculate p values
            p = Parallel(n_jobs=int(os.cpu_count()/2))(delayed(Selectivity_Analysis_Feature.calculate_density_perm_p)(tsne, feature[:, i], num_perm=1000, kernel=gaussian_kernel) for i in tqdm(np.arange(feature.shape[1]), desc=f'{DR_sub_type}'))
            
            # --- wrap results and save
            results = {
                       'layer': layer, 
                       'tsne': tsne,     
                       'DR_sub_type': DR_sub_type,
                       'p': p, 
                       'sigma_scaling_factor': sq, 
                       'kernel_size': kernel_size, 
                       'kernel_sigma': kernel_sigma, 
                       'kernel': gaussian_kernel,
                       }
            
            utils_.pickle_dump(file_path, results)
            
        # ----- feature regions
        # --- init
        p_values = results['p']
        gaussian_kernel = results['kernel']
        tsne = results['tsne']
        maskFactor = 0.1
        cluster_size_scaling_factor=0.025
        alpha=0.01

        # ---
        file_path = os.path.join(self.tsne_save_folder, f'{layer}_{DR_sub_type}_unit_stats.pkl')
        
        if os.path.exists(file_path):
            
            results = utils_.pickle_load_tqdm(file_path)
            
        else:
        
            density_map, convolved_density_map = Selectivity_Analysis_Feature.calculate_convolved_density_map(tsne, None, gaussian_kernel)
            
            # --- remove corners and edges with too sparse dots
            mask = convolved_density_map >= (maskFactor*np.mean(convolved_density_map))
 
            # --- init
            reversed_sort_dict = {value: [key for key, vals in cell_stats['cell_types_dict'].items() if value in vals][0] for key_list in cell_stats['cell_types_dict'].values() for value in key_list}
            reversed_sort_dict = {_: reversed_sort_dict[_] for _ in sorted(reversed_sort_dict.keys())}
            
            cluster_size_threshold = mask.size*cluster_size_scaling_factor
            
            # --- Sequential, for test
            pl = {}
            
            units = list(reversed_sort_dict.keys())
            for unit in tqdm(units, desc='Sequential region selection'):
                
                results = Selectivity_Analysis_Feature.feature_region_selection_single_unit(

                                                          tsne, 
                                                          
                                                          {unit: reversed_sort_dict[unit]},
                                                          p_values[unit], 
                                                          cell_stats['encode_id'][unit],      # ---
                                                               
                                                          mask, 
                                                          cluster_size_threshold, 
                                                          
                                                          img_labels,
                                                          )
                                            
                pl[unit] = results
        
            feature_selective_stats = {_: pl[_]['feature_selective_unit'] for _ in units if pl[_] is not None and len(pl[_]['feature_selective_unit']) != 0}
            tmp_pool = [___ for __ in [feature_selective_stats[_] for _ in feature_selective_stats.keys()] for ___ in __]
            tmp_pool_new = [_.split('encode_')[-1] for _ in tmp_pool]
            
            feature_component_stats = {_:pl[_]['feature_component_dict'] for _ in units if pl[_] is not None and len(pl[_]['feature_component_dict']) != 0}
    
            # feature_unit_sorting
            feature_units = np.array(list(feature_component_stats.keys()))
            feature_selective_units = np.array(list(feature_selective_stats.keys()))
            feature_non_selective_units = np.setdiff1d(feature_units, feature_selective_units)
            
            # ---
            feature_strong_sensitive_mi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'strong_encode_sensitive_mi' in feature_selective_stats[_]])
            feature_weak_sensitive_mi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'weak_encode_sensitive_mi' in feature_selective_stats[_] and 'strong_encode_sensitive_mi' not in feature_selective_stats[_]])
            feature_merged_sensitive_mi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'merged_encode_sensitive_mi' in feature_selective_stats[_] and 'strong_encode_sensitive_mi' not in feature_selective_stats[_] and 'weak_encode_sensitive_mi' not in feature_selective_stats[_]])
            
            feature_strong_sensitive_wmi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'strong_encode_sensitive_wmi' in feature_selective_stats[_]])
            feature_weak_sensitive_wmi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'weak_encode_sensitive_wmi' in feature_selective_stats[_] and 'strong_encode_sensitive_wmi' not in feature_selective_stats[_]])
            feature_merged_sensitive_wmi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'merged_encode_sensitive_wmi' in feature_selective_stats[_] and 'strong_encode_sensitive_wmi' not in feature_selective_stats[_] and 'weak_encode_sensitive_wmi' not in feature_selective_stats[_]])
            
            # ---
            feature_strong_sensitive_si_idx = np.array([_ for _ in feature_selective_stats.keys() if 'strong_encode_sensitive_si' in feature_selective_stats[_]])
            feature_weak_sensitive_si_idx = np.array([_ for _ in feature_selective_stats.keys() if 'weak_encode_sensitive_si' in feature_selective_stats[_] and 'strong_encode_sensitive_si' not in feature_selective_stats[_]])
            feature_merged_sensitive_si_idx = np.array([_ for _ in feature_selective_stats.keys() if 'merged_encode_sensitive_si' in feature_selective_stats[_] and 'strong_encode_sensitive_si' not in feature_selective_stats[_] and 'weak_encode_sensitive_si' not in feature_selective_stats[_]])
            
            feature_strong_sensitive_wsi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'strong_encode_sensitive_wsi' in feature_selective_stats[_]])
            feature_weak_sensitive_wsi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'weak_encode_sensitive_wsi' in feature_selective_stats[_] and 'strong_encode_sensitive_wsi' not in feature_selective_stats[_]])
            feature_merged_sensitive_wsi_idx = np.array([_ for _ in feature_selective_stats.keys() if 'merged_encode_sensitive_wsi' in feature_selective_stats[_] and 'strong_encode_sensitive_wsi' not in feature_selective_stats[_] and 'weak_encode_sensitive_wsi' not in feature_selective_stats[_]])
            
            feature_unit_sorting_dict = {
                'feature_non_selective_units': feature_non_selective_units,
                
                'feature_strong_sensitive_mi_idx': feature_strong_sensitive_mi_idx,
                'feature_weak_sensitive_mi_idx': feature_weak_sensitive_mi_idx,
                'feature_merged_sensitive_mi_idx': feature_merged_sensitive_mi_idx,
                
                'feature_strong_sensitive_wmi_idx': feature_strong_sensitive_wmi_idx,
                'feature_weak_sensitive_wmi_idx': feature_weak_sensitive_wmi_idx,
                'feature_merged_sensitive_wmi_idx': feature_merged_sensitive_wmi_idx,
                
                'feature_strong_sensitive_si_idx': feature_strong_sensitive_si_idx,
                'feature_weak_sensitive_si_idx': feature_weak_sensitive_si_idx,
                'feature_merged_sensitive_si_idx': feature_merged_sensitive_si_idx,
                
                'feature_strong_sensitive_wsi_idx': feature_strong_sensitive_wsi_idx,
                'feature_weak_sensitive_wsi_idx': feature_weak_sensitive_wsi_idx,
                'feature_merged_sensitive_wsi_idx': feature_merged_sensitive_wsi_idx
                }
            
            # -----
            results = {
                'original_results': pl,
                
                'preliminary_p_masks': {_:pl[_]['preliminary_p_mask'] for _ in units if pl[_] is not None},
                'qualified_p_masks': {_:pl[_]['qualified_p_mask'] for _ in units if pl[_] is not None},
                
                'feature_selective_stats': feature_selective_stats,
                
                'feature_component_stats': feature_component_stats,
                
                'feature_unit_sorting_dict': feature_unit_sorting_dict
                }
            
            utils_.pickle_dump(file_path, results)
            
        # -----
        pl = results['original_results']
        feature_unit_sorting_dict = results['feature_unit_sorting_dict']
        feature_component_stats = results['feature_component_stats']
        
        self.single_unit_folder = os.path.join(self.tsne_save_folder, 'Single Unit Plot')
        utils_.make_dir(self.single_unit_folder)
        
        colors = [plt.get_cmap('jet', 50)(i) for i in range(50)]
        
        for plot_type in feature_unit_sorting_dict.keys():     # foe each type
        
            plot_types_idces = feature_unit_sorting_dict[plot_type]
            
            if len(plot_types_idces) != 0:
        
                for unit in plot_types_idces:
                
                    Selectivity_Analysis_Feature.Selectivity_Analysis_Feature.plot_region_based_coding_single_unit(unit, tsne, feature, feature_component_stats[unit], 
                                                     layer='neuron_1', img_labels=img_labels, colors=colors, num_classes=50, plot_type_folder=self.single_unit_folder)
        
        # --------------------------------------------------------------------------------------------------------------
        self.sample_folder = os.path.join(self.tsne_save_folder, 'sample figs')
        utils_.make_dir(self.sample_folder)
        
        for type_ in cell_stats['cell_types_dict'].keys():
            
            if cell_stats['cell_types_dict'][type_].size > 0:
                
                # ---
                #cell = np.random.choice(cell_stats['cell_types_dict'][type_])
                
                # ---
                if type_ == 'sensitive_si':
                    cell = 127
                elif type_ == 'sensitive_wsi':
                    cell = 1121
                elif type_ == 'sensitive_mi':
                    cell = 56
                elif type_ == 'sensitive_wmi':
                    cell = 1287
                elif type_ == 'sensitive_non_encode':
                    cell = 998
                elif type_ == 'non_sensitive_wsi':
                    cell = 163
                elif type_ == 'non_sensitive_wmi':
                    cell = 1618
                elif type_ == 'non_sensitive_non_encode':
                    cell = 1525
            
                FR = meanFR[cell, :]
                
                # -----
                encoded_ids = np.append(cell_stats['encode_id'][cell]['encode'], cell_stats['encode_id'][cell]['weak_encode']).astype(int)
                
                fig, ax = plt.subplots(figsize=(10,10))
                
                DR_scatter(ax, tsne, img_labels, FR, encoded_ids)
                
                ax.set_title(f'Human Cells DR(TSNE) | Coordinates from: {coor_name} | Unit: {cell} | Type: {type_}')

                fig.tight_layout()
                fig.savefig(os.path.join(self.sample_folder, f'{type_} {cell}.png'))
                plt.close()

        print('6')
                
# ======================================================================================================================
def used_ids_selection(encoded_id_pool, used_id_num):
      
    freq = dict(Counter(encoded_id_pool))
    freq={int(k):v for k,v in sorted(freq.items(), key=lambda x:x[1], reverse=True)}
    
    return {k:v for idx, (k,v) in enumerate(freq.items(), 0) if idx < used_id_num}


def plot_single(fig, gs_main, num_types, idx_dict, feature, num_classes, num_samples, layer=None, percentile=99.):
    """
        [notice] no auto-adjust for figure size, the proper figsize must be manually appointed
    """
    colorpool_jet = plt.get_cmap('jet', 50)
    colors = [colorpool_jet(i) for i in range(50)]
    
    tqdm_bar = tqdm(total=num_types, desc=f'{layer}')
    
    y_lim_min = np.min(feature)
    
    #y_lim_max = np.max(feature)     # this will extremely extend the radius by outliers
    y_lim_max = np.percentile(feature, percentile)

    y_lin_range = y_lim_max - y_lim_max
    
    num_cols = gs_main.ncols
    num_rows = gs_main.nrows
    
    for i in range(num_rows):
        for j in range(num_cols):
            
            unit_type = list(idx_dict.keys())[i*num_cols+j]
            
            gs_sub = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[4, 1], subplot_spec=gs_main[i, j])

            ax_left = fig.add_subplot(gs_sub[0])
            ax_right = fig.add_subplot(gs_sub[1])
            
            if (i+j) != 0:
                ax_left.set_xticks([])
                ax_left.set_yticks([])

            if idx_dict[unit_type].size == 0:
                ax_left.set_title(unit_type + ' [0.00%]')
                ax_right.set_title('th')

            else:
                feature_test = feature[:, idx_dict[unit_type]]     # (500, num_units)

                feature_test_mean = np.mean(feature_test.reshape(num_classes, num_samples, -1), axis=1)     # (50, num_units)
                
                num_units = len(idx_dict[unit_type])
                
                # -----
                x = np.tile(np.arange(num_classes), num_units)     # (0,1,...,49,0,1,...)
                y = feature_test_mean.T.reshape(-1)     # every 50 ids for unit by unit
                
                c = np.tile(np.array(colors), [num_units, 1])

                # -----
                ax_left.scatter(x, y, color=c, alpha=0.75, marker='.', s=1)     # use small size to replace adjustable alpha
                # -----
                
                #pct = num_units/feature.shape[1]*100
                pct = num_units/1577*100
                
                ax_left.set_title(unit_type + f' [{pct:.2f}%]')
                # -----
                
                # ----- stats: mean firing rate for each id
                values = feature_test_mean.reshape(-1)    # (50*num_units)
                
                plot_single_subsubplot(ax_left, ax_right, values, color='blue')
                
                # ----- stats: threshold (mean+2std of all 500 values)
                values = np.mean(feature_test, axis=0) + 2*np.std(feature_test, axis=0)     # (num_units,)

                plot_single_subsubplot(ax_left, ax_right, values, linestyle='dotted', color='red')
                
                # ----- stats: ref (mean+2std of all 50 mean values)
                values = np.mean(feature_test_mean, axis=0) + 2*np.std(feature_test_mean, axis=0)     # (num_units,)
                
                plot_single_subsubplot(ax_left, ax_right, values, linestyle='dotted', color='teal')
                
                # -----
                scaling_factor = 0.1
                
                ax_left.set_ylim([y_lim_min - y_lin_range*scaling_factor, y_lim_max + y_lin_range*scaling_factor])
                ax_right.set_ylim([y_lim_min - y_lin_range*scaling_factor, y_lim_max + y_lin_range*scaling_factor])
                ax_right.set_title('PDF')
                ax_right.set_yticks([])
                
            tqdm_bar.update(1)

def plot_single_subsubplot(ax_left, ax_right, values, linestyle=None, color=None, scaling_factor=0.1):
    
    if np.std(values) == 0:
        pass
    else:
        kde = gaussian_kde(values)
        
        min_values = np.min(values)
        max_values = np.max(values)
        
        values_range = max_values - min_values
        
        x_vals = np.linspace(min_values - scaling_factor*values_range, max_values + scaling_factor*values_range, 101)
        y_vals = kde(x_vals)
        ax_right.plot(y_vals, x_vals, linestyle=linestyle, color=color)
    
        y_vals_max = np.max(y_vals)
        
        if len(y_peak:=np.where(y_vals==y_vals_max)[0]) == 1:
            x_vals_max = x_vals[y_peak.item()]
        else:
            x_vals_max = x_vals[y_peak[0]]
        
        ax_left.hlines(x_vals_max, 0, 50, colors=color, alpha=0.75, linestyle='--')
        ax_right.hlines(x_vals_max, np.min(y_vals), np.max(y_vals), colors=color, alpha=0.75, linestyle='--')


def DR_scatter(ax, tsne, img_labels, weights, encoded_ids):
    
    x = tsne[:, 0] - np.min(tsne[:, 0])
    y = tsne[:, 1] - np.min(tsne[:, 1])
    
    colors = [plt.get_cmap('jet', 50)(i) for i in range(50)]
    
    if np.sum(weights) == 0 or np.sum(weights!=0) ==1:
        for gg in range(1,51):  # this can be changed to different types of id
            current_scatter = ax.scatter(x[img_labels == gg], y[img_labels == gg], s=5, color=colors[gg-1], alpha=0.5)
            
    else:
        
        size_weight = weights / max(weights)     # [notice] can not divide by 0 if all values are 0
        sizes = np.ones(500) * 15 * (1 + 20 * size_weight)
        
        for gg in range(1,51):  # this can be changed to different types of id
            current_scatter = ax.scatter(x[img_labels == gg], y[img_labels == gg], s=sizes[img_labels==gg], color=colors[gg-1], alpha=0.5)
        
        # -----
        if len(encoded_ids) > 0:
        
            handles_featured = []
            labels = []
            
            for gg in encoded_ids:
                current_scatter = ax.scatter(x[img_labels == gg], y[img_labels == gg], s=sizes[img_labels == gg], color=colors[gg-1], alpha=0.7)
                handles_featured.append(current_scatter)
                labels.append(f'{gg}')
            
            ax.add_artist(ax.legend(handles=handles_featured, labels=labels, framealpha=0.5))


# ======================================================================================================================
def calculate_firing_rate(time_stamps, session_idx, all_periods, time_window, num_frames, PSTH_start, time_step=50):
    """
        - time_stamps: timing of spikes
        - session_idx: the experiment is which one among all 40 sessions
    
        one record: |period_start --- (timestamps of spikes) --- period_end|
        
        Count how many spikes inside the period
        
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


# ======================================================================================================================
class Monkey_Neuron_Records_Process():
    """
        Unlike human cell data, no data process here due to the Monkey data is a well processed dataset
 
        ...
    """
    
    def __init__(self, bio_root='/home/acxyle-workstation/Downloads/Bio Neuron Data/Monkey/', seed=6, **kwargs):
        """
            this function determines the time range of interest [-50, 200] from the original time range [-100, 380], which 
            directly inherit from original Matlab code
        """
        np.random.seed(seed)
        
        self.bio_root = bio_root
        self.ts = np.arange(-50, 201, 10)
        
        # ---
        self.Monkey_data_reconstruction()
        

    def Monkey_data_reconstruction(self, ):
        """
            this function converts the original .mat file to a python dict
            
            The **img** FR and PSTH are not under the natural order but indices of displayed imgs.
            
            ...
        """
        
        data_path = os.path.join(self.bio_root, 'data.pkl')
        
        monkey_neuron_data_path = os.path.join(self.bio_root, 'Original Data/IT_FR_CA_Range70-180.mat')     # processed monkey neural data
        monkey_neuron_data = sio.loadmat(monkey_neuron_data_path)

        monkey_dict_keys = [i for i in monkey_neuron_data.keys() if '__' not in i]
        monkey_dict = {_:monkey_neuron_data[_] for _ in monkey_dict_keys}     # rebuild the dict to store monkey IT MUA data
        
        # -----
        self.FR = monkey_dict['FR']
        
        # ---
        self.FR_countAll = self.FR['countAll'][0][0]     
        self.FR_countBase = self.FR['countBase'][0][0]
        self.FR_countVis = self.FR['countVis'][0][0]
        
        # ---
        self.meanFR = monkey_dict['meanFR']     # (53, 500)
        self.meanBase = monkey_dict['meanBase']     # (53, 500)
        self.meanGray = monkey_dict['meanGray'].reshape(-1)     # (53, )
        self.meanVis = monkey_dict['meanVis']     # (53, 500)
        
        # ---
        self.psthTime = monkey_dict['psthTime'].reshape(-1)     # (49,)
        self.meanPSTH = monkey_dict['meanPSTH']     # (500, 49, 53), [disordered img idx, time steps, channels], normalized
        self.meanPSTHID = monkey_dict['meanPSTHID']     # (50, 49, 53), [id idx, time steps, channels], normalized
        
        # -----
        if os.path.exists(data_path):
            data = utils_.load(data_path)
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
            
            utils_.dump(data, data_path)
            
        return data
    
    
    def monkey_neuron_feature_process(self, time_bin=10):
        """
            return:
                feature dict of 'FR_id' and 'psth_id' of natural order
        """
        
        file_path = os.path.join(self.bio_root, 'features.pkl')
        
        if os.path.exists(file_path):
            
            feature_dict = utils_.load(file_path)
            
            FR_id = feature_dict['FR_id']
            psth_id = feature_dict['psth_id']
        
        else:
        
            label = sio.loadmat(os.path.join(self.bio_root, 'Original Data/Label.mat'))['label'].reshape(-1)
            
            # ----- FR
            sacling_factor = self.meanGray
            FR_id = np.array([np.mean(self.meanFR[:, np.where(label==_)[0]], axis=1)/sacling_factor for _ in range(1, 51)])     # (50, 53)
            
            # ----- PSTH
            if time_bin == 10:
                used_psth = self.meanPSTH[:, [np.where(self.psthTime==_)[0][0] for _ in self.ts], :]
                
            else:
                used_psth = np.zeros((self.meanPSTH.shape[0], len(self.ts), self.meanPSTH.shape[2]))     # (500, 26, 53) (img, time, unit)
                for idx, tt in enumerate(self.ts): 
                    used_psth[:, idx, :] = np.mean(self.meanPSTH[:, np.where(((tt-time_bin/2)<=self.psthTime) & (self.psthTime<=(tt+time_bin/2)))[0], :], axis=1)
            
            # ---
            used_psth_id = np.array([np.mean(used_psth[np.where(label==_)[0], :, :], axis=0) for _ in  range(1, 51)])     
            used_psth_id = np.transpose(used_psth_id, (1,0,2))     # (time, ID, unit)
            
            scaling_factor = np.mean(self.meanBase, axis=1)
            psth_id = np.array([np.array([used_psth_id[i, j, :]/scaling_factor for j in range(50)]) for i in range(used_psth.shape[1])])     # (26, 50, 53)
            
            # ---
            feature_dict = {
                'FR_id': FR_id,
                'psth_id': psth_id
                }
            
            utils_.dump(feature_dict, file_path)
            
        return FR_id, psth_id
        
    
    def monkey_neuron_DSM_process(self, metric='pearson', permutation=True, num_perm=1000, save=True, vectorize=False, **kwargs):
        """
            this function returns the correlation matrix and triangle from monkey neural responses.
            
            input
                - psthTime: 49 time steps for PSTH from -100 ms to 380 ms
                - meanPSTH: [500, 49, 53] (img idx, time steps, cell channels)
                - label: label for 500 imgs
                
            return
                - monkey_DM: tranformed DSM
                - monkey_DM_perm: transformed DSM with extra dimension of permutation
                - monkey_DM_temporal: transformed DSM with temporal dimension
                - monkey_DM_temporal_perm: transformed DSM with temporal dimension and permutation dimension
                
            the firing rates has been normalized by scaling_factors, FR uses meanGray and PSTH uses meanBase by default,
            this selection inherits from the original Matlab code. meanGray is not identical with np.mean(meanBase, axis=1)
            
            the permutation before DSM_calculation is equal to the shuflle of the calculated DSM
            
            this permutation process inherit from original Matlab code, permute Bio data to check whether the NN data 
            aligns with the pattern of Bio data, and vice versa (not very sure :P). 
            
            this process depends on downstream tasks and research purposes.
        """

        utils_.make_dir(os.path.join(self.bio_root, 'corr'))
        utils_.make_dir(os.path.join(self.bio_root, 'corr', f'{metric}'))
        file_path = os.path.join(self.bio_root, 'corr', f'{metric}', 'Monkey_DM_dict_qualified_50.pkl')
        
        if os.path.exists(file_path):
            
            results = utils_.load(file_path)
            
        else:

            FR_id, psth_id = self.monkey_neuron_feature_process()
            
            num_samples = FR_id.shape[0]
            num_steps = psth_id.shape[0]
            
            results = {
                'monkey_DM': utils_similarity.DSM_calculation(metric, FR_id, vectorize=vectorize), 
                'monkey_DM_temporal': np.array([utils_similarity.DSM_calculation(metric, psth_id[_, :, :], vectorize=vectorize) for _ in range(num_steps)]),
                }
            
            if permutation:
            
                results.update({
                    'monkey_DM_perm': np.array([utils_similarity.DSM_calculation(metric, FR_id[np.random.permutation(num_samples)], vectorize=vectorize) for _ in range(num_perm)]),
                    'monkey_DM_temporal_perm': np.array([np.array([utils_similarity.DSM_calculation(metric, psth_id[t, np.random.permutation(num_samples), :], vectorize=vectorize) for _ in range(num_perm)]) for t in range(num_steps)])
                    })
            
            if save:
                
                utils_.dump(results, file_path)
            
        return results
        
    
    def monkey_neuron_DSM_plot(self,):
        
        results = self.monkey_neuron_DSM_process()
        
        for metric in results.keys():
            fig, ax = plt.subplots(figsize=(10,10))
            
            title = f'Monkey Distance Matrix | {metric}'
            
            # use squareform() for vector
            img = ax.imshow(results[metric]['monkey_DM'], aspect='auto', origin='lower')
            
            ax.set_title(title, fontsize=24)
            
            fig.colorbar(img)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.bio_root, f'{title}.svg'))
        
    
    def monkey_neuron_Gram_process(self, kernel='linear', permutation=True, num_perm=1000, save=True, **kwargs):
        """
            this permutation process inherit from original Matlab code, permute Bio data to check whether the NN data 
            aligns with the pattern of Bio data, and vice versa (not very sure :P). 
            
            this process depends on downstream tasks and research purposes.
        """
        
        utils_.make_dir(os.path.join(self.bio_root, 'gram'))
        utils_.make_dir(os.path.join(self.bio_root, 'gram', f'{kernel}'))
        
        if kernel == 'rbf' and 'threshold' in kwargs:
            file_path = os.path.join(self.bio_root, 'gram', f'{kernel}', f"CKA_results_{kernel}_{kwargs['threshold']}.pkl")
        elif kernel == 'linear':
            file_path = os.path.join(self.bio_root, 'gram', f'{kernel}', f"CKA_results_{kernel}.pkl")
        else:
            raise ValueError(f'[Coderror] Invalid kernel [{kernel}]')
            
        if os.path.exists(file_path):
            
            results = utils_.load(file_path)
            
        else:

            FR_id, psth_id = self.monkey_neuron_feature_process()
            
            if kernel == 'linear':
                gram = utils_similarity.gram_linear
            elif kernel =='rbf':
                gram = utils_similarity.gram_rbf
            
            num_samples = FR_id.shape[0]
            num_steps = psth_id.shape[0]
            
            results = {
                'monkey_Gram': gram(FR_id, **kwargs), 
                'monkey_Gram_temporal': np.array([gram(psth_id[_, :, :], **kwargs) for _ in range(num_steps)]),
                }
            
            if permutation:
            
                results.update({
                    'monkey_Gram_perm': np.array([gram(FR_id[np.random.permutation(num_samples)], **kwargs) for _ in range(num_perm)]),
                    'monkey_Gram_temporal_perm': np.array([np.array([gram(psth_id[t, np.random.permutation(num_samples), :], **kwargs) for _ in range(num_perm)]) for t in range(num_steps)])
                    })
            
            if save:
                
                utils_.dump(results, file_path)
            
        return results
    
    
    def Monkey_plot_sample_response(self, average=True):
        """
            this function plot the sample responses of monkey IT cells
            
            Parameter:
                average: if True, plot the average responses across channels, otherwise plot the channel with strongest
                reponses, i.e. channel 51 (0-based)
        """
        
        # --- normalized
        scaling_factor = np.mean(self.meanBase, axis=1)     # (53,)

        meanPSTHIDNorm = self.meanPSTHID/scaling_factor     # [id idx, time steps, channels]
        
        if average:
            meanPSTHIDNorm = np.mean(meanPSTHIDNorm, axis=2)     # [time, id]
            title = 'Monkey normalized channel-average PSTH'
        else:
            target_channel = np.argmax(np.mean(meanPSTHIDNorm, axis=(0, 1)))
            meanPSTHIDNorm = meanPSTHIDNorm[..., target_channel]
            title = f'Monkey normalized channel {target_channel} PSTH'
        
        # -----
        plt.rcParams.update({"font.family": "Times New Roman"})
        plt.rcParams.update({'font.size': 18})
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        plot_PSTH(fig, ax, meanPSTHIDNorm, title, self.psthTime, -50, 200)
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.bio_root, f'{title}.svg'))
        plt.close()


# ======================================================================================================================    
def plot_PSTH(fig, ax, PSTH, title=None, time_point=None, time_start=None, time_end=None):
    """
        ...
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


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- 1. human analysis
    human_record_process = Human_Neuron_Records_Process()
    
    #human_record_process.human_cell_FeatureMap()
    #human_record_process.human_cell_SortFR(data_type='default')
    #human_record_process.human_identity_cell_selection()
    #human_record_process.human_neuron_raster_plot()
    #human_record_process.human_cell_stacked_responses()
    #human_record_process.human_plot_FR_PDF()
    #human_record_process.human_neuron_DSM_process()
    
    for used_cell_type in ['qualified', 'selective', 'non_selective', 'strong_selective', 'weak_selective']:
        human_record_process.human_neuron_Gram_process(used_cell_type=used_cell_type)
        for threshold in [1.0, 10.0]:
            human_record_process.human_neuron_Gram_process(kernel='rbf', threshold=threshold, used_cell_type=used_cell_type)
    
    #human_record_process.human_plot_pie_chart()
    
    # --- 2. monkey analysis
    #monkey_record_process = Monkey_Neuron_Records_Process()
    
    #monkey_record_process.Monkey_data_reconstruction()
    #monkey_record_process.Monkey_plot_sample_response()
    #monkey_record_process.monkey_neuron_feature_process()
    #monkey_record_process.monkey_neuron_DSM_process(metrics=['euclidean', 'pearson'])
    #monkey_record_process.monkey_neuron_DSM_plot()
    #monkey_record_process.monkey_neuron_Gram_process()
    
    # ----- local debug...
    # --- 3. Primate responses + NN coordinates
    #for folder in ['FSA Baseline', 'FSA VGG16']:
    #    human_record_process.human_DR(folder)
