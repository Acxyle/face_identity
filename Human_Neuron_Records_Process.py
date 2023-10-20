#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 13:35:41 2023

@author: acxyle-workstation

    [Oct 19, 2023] Most of the confusing strcutures regarding number/index come from MATLAB design. One can change that to
    python dict for the entire code if necessary as I managed part of it in raster plot
    
"""

import torch

import os
import pandas as pd
import pickle
import scipy.stats as stats
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

#FIXME
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
    def human_neuron_sort_FR(self, reject_rate=0.15, data_type='default', data_set='CelebA'):
        """
            this function calculates the sorted mean firing rates and qualified cells
        
            for session with code != 500 means the images displayed in the experiments are not standard.
            for session 8, with 711 'code', 67 'back_id', 644 non_back images, in which 500 standard images and 144 
            repeated images (each repeated 1 time), they took the average value of repeated experiment
        
            input: 
                
            - reject_rate (default=0.15), suggests the threshold to determine the qualified cells
            - data_type (default='default'), 'Base': firing rate from 250ms to 500ms; 'defualt': firing rate from 750ms to 1750ms
            - deta_set (default='CelebA'), currently only 'CelebA'
            
            output:
            
            - meanFR: response map with shape (num_cells, num_imgs)
            - cells: 1,577 qualified cell idces from all 2082 cells
            
            involved variables:
            
            - FR_stats: preliminary firing rates from original experimental records, simply count spikes in given time period
            - beh_stats: original experimental settings and logs, indicates the relationships between images and records
            - adjust_idx [AdjustInd]: corrected indeces of displayed images in experiments due to image errors
            - image_sequence [Code]: the displayed image sequence of each session, used to build relationship between image and neuron records
            ...
            
        """
        
        print('[Codinfo] Calculating sorted meanFR...')
        
        meanFR_path = os.path.join(self.human_neuron_stats, 'meanFR.pkl')
        
        if os.path.exists(meanFR_path):
            
            meanFR_dict = utils_.pickle_load(meanFR_path)
            
        else:
            
            if data_set == 'CelebA':
                
                # ----- 1. obtain firing rate (FR) and peri-stimulus histogram (PSTH)
                FR_stats = self.human_neuron_get_firing_rate()
        
                # ----- 2. obtain behavior data
                beh_stats = self.human_neuron_get_beh()
                behavior = beh_stats['beh']
                
                # ----- 3. determine qualified neurons
                neuron_session_idces = self.Spikes['vCell'].reshape(-1)-1     # 1-based (MATLAB) -> 0-based (Python)
                
                # --- 3.1 1st condition, firing rate
                cell_reject = np.array([]).astype(int)
                for cell_idx in range(len(FR_stats)):  
                    if np.nanmean(FR_stats[cell_idx]['spike_count_0_2000']) < reject_rate:  
                        cell_reject = np.append(cell_reject, cell_idx)
                
                # --- 3.2 2nd, manually "exclude sessions 12(only has 117 trials) and 18(only 1 neuron kept and patients did not pay attention)"
                exclude_cell = np.array([index for index, value in enumerate(neuron_session_idces) if value == 11 or value == 17])
                cell_reject = np.union1d(cell_reject, exclude_cell)
         
                qualified_cells = np.setdiff1d(np.arange(len(FR_stats)), cell_reject)     # qualified neurons with 2 conditions
            
                # ----- 4. load Code idx [adjust_idx]
                # [warning] assume the 'im_code' in CelebA_img_idces is deprecated and replaced by 'im_code' from CelebA_img_idces_new
                CelebA_img_idces = sio.loadmat(os.path.join(self.root_data, 'SingleNeuron/Code/CelebA_Image_Code.mat'))
                adjust_idx = CelebA_img_idces['AdjustInd'].reshape(-1).astype(np.int16) -1   
                
                CelebA_img_idx_new = sio.loadmat(os.path.join(self.root_data, 'SingleNeuron/Code/CelebA_Image_Code_new.mat'))
                img_idces = CelebA_img_idx_new['im_code'].reshape(-1) 
                id_img_idces = CelebA_img_idx_new['id_code'].reshape(-1)     # each id contains 10 imgs
                id_img_idces = np.array([_.reshape(-1)-1 for _ in id_img_idces])
                
                img_idces_dict = {
                    'adjust_idx': adjust_idx,
                    'img_idces': img_idces,
                    'id_img_idces': id_img_idces
                    }
                
                # ----- 5. start meanFR calculation
                meanFR = np.full((len(FR_stats), 500), np.nan)     # empty response map waited to receive values
                
                for cell_idx in tqdm(range(len(FR_stats))):     # for each neuron
                    
                    session_idx = neuron_session_idces[cell_idx]     # get session idx
                
                    image_sequence = behavior[session_idx]['code'].copy()     # delete back_id in img_list
                    back_id = behavior[session_idx]['back_id'].copy()
        
                    image_sequence = np.delete(image_sequence, back_id)     # img idx
                    
                    if data_type == 'Base':
                        FR = FR_stats[cell_idx]['spike_count_250_500'].copy()
                    elif data_type == 'default':
                        FR = FR_stats[cell_idx]['spike_count'].copy()
                    
                    FR = np.delete(FR, back_id)   # remove back_id in neuron response
                    
                    sort_idx = np.argsort(image_sequence)     # the index of 'Code' in ascending order
                    sorted_image_idces = image_sequence[sort_idx]     # 1-based
                    FR = FR[sort_idx]     # sorted FR based on the index of 'Code', make the responses align with the image idces
                    
                    # ----- correction
                    if session_idx < 10:     # for the first 10 sessions
                        
                        adjustFR = np.zeros(500)
                        
                        if len(image_sequence) != 500:     # 3: (498), 8: (499)
    
                            new_FR = np.full(500, np.nan)  
                            
                            for _ in range(500):     # for each img
                                used_imgs = np.where(sorted_image_idces == (_+1))[0]     # for most of the cases, 1 record for 1 img
                                if len(used_imgs) != 0:
                                    new_FR[_] = np.mean(FR[used_imgs])   # take average 
                                    
                            for _ in range(500):     # for each img
                                if adjust_idx[_] >= 0:     # leave the unwanted imgs
                                    adjustFR[_] = new_FR[adjust_idx[_]]
                        else:
                            
                            for _ in range(500):
                                if adjust_idx[_] >= 0:
                                    adjustFR[_] = FR[adjust_idx[_]]
                        
                        meanFR[cell_idx, :] = adjustFR  # Note: Python uses 0-based indexing
                        
                    else:     # for the rest 30 sessions
                    
                        if len(image_sequence) != 500:     # 11: (161), 13: (452)
                            new_FR = np.full(500, np.nan)
                            
                            for _ in range(500):
                                used_imgs = np.where(sorted_image_idces == (_+1))[0]
                                if len(used_imgs) != 0:
                                    new_FR[_] = np.mean(FR[used_imgs])
                                    
                            meanFR[cell_idx, :] = new_FR
                            
                        else:
                            meanFR[cell_idx, :] = FR
                               
                # ----- 6. repair because of image errors
                # --- 6.1. face 121 is a mis-identified photo of identity 6, replace the FR to average FR of other 9 faces
                meanFR[:, 120] = np.nanmean(meanFR[:, id_img_idces[5][:-1]], 1)
                
                # --- 6.2. replace the problemd face with mean of that ID for the problemed ID (only for the first 381 neurons),in
                # which another 2 faces(2 trials) were mis-identified, results are similar when replaced with Nan
                defective_ids = [17, 39]
                for _ in defective_ids:
                    useful_idces = id_img_idces[_][-1]
                    meanFR[:381, useful_idces] = np.nanmean(meanFR[:381, id_img_idces[_][:-1]], 1)
                    
                meanFR_dict = {
                    'meanFR': meanFR,
                    'qualified_cells': qualified_cells,
                    'img_idces_dict': img_idces_dict
                    }
                
                utils_.pickle_dump(meanFR_path, meanFR_dict)
                    
            else:
                
                raise RuntimeError('[Coderror] data_set not a_licable')
                
                # [notice] legacy from MATLAB code, not in use for current task
                #FR = np.array([])
                #for _ in range(1, imgNum+1):
                #    useful_idces = np.where(sorted_image_idces ==_)
                #    if not useful_idces:
                #        FR = np.concatenate((FR, FR[useful_idces]))
                #meanFR[cell_idx-1,:] = FR;

        return meanFR_dict
        
    
    def human_neuron_get_firing_rate(self, time_window=250, time_step=50):  
        """
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
        """
        
        print('[Codinfo] Calculating original firing rates...')
        
        file_path = os.path.join(self.human_neuron_stats, 'FR_stats.pkl')
        
        if os.path.exists(file_path):
            FR_stats = utils_.pickle_load(file_path)
            
        else:

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
                
            # ----- concatenate
            for key in [_ for _ in list(behavior_data_dict.keys()) if 'iT' not in _]:
                behavior_data_dict[key] = np.concatenate((behavior_data_dict[key], data_2[key]))
                
            for key in behavior_data_dict['iT'].keys():
                behavior_data_dict['iT'][key] = np.concatenate((behavior_data_dict['iT'][key], data_2['iT'][key]))
                
        else:
            
            filename = inputfiles[0]
            data_dict = self.behavior_data_dict_format_transformation(os.path.join(path, filename))

            # ----- processes from matlab source code
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
            # -----
            
            # ----- processes local added
            elif filename == 'CelebA_p7WV_20Sep2019101736.mat':     # cut-off and shift, [Exception 4, P7WV_Sess2]
                behavior_data_dict = {variable: data_dict[variable][2:] for variable in variables_list}
                behavior_data_dict['iT'] = {_: data_dict['iT'][_][2:] for _ in data_dict['iT'].keys()}
                behavior_data_dict['back_id'] = data_dict['back_id'] - 2
                
            elif filename == 'CelebA_p9WV_27Oct2019113332.mat':     # cut-off and shift, [Exception 5, P9WV_Sess3]
                behavior_data_dict = {variable: data_dict[variable][1:] for variable in variables_list}
                behavior_data_dict['iT'] = {_: data_dict['iT'][_][1:] for _ in data_dict['iT'].keys()}
                behavior_data_dict['back_id'] = data_dict['back_id'] - 1
            # -----
            
            # ----- for the rest 35 normal cases
            else:
                behavior_data_dict = {variable: data_dict[variable] for variable in variables_list}
                behavior_data_dict['iT'] = {_: data_dict['iT'][_] for _ in data_dict['iT'].keys()}
                behavior_data_dict['back_id'] = data_dict['back_id']
            # -----

        # ↓ can not find such file and looks like the function is using TTL to refine the exact timing.
        # adjustBehaviorAccordingToTTL()  
        
        # ----- 
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
            meanFR_dict = self.human_neuron_sort_FR()
            
            meanFR = meanFR_dict['meanFR']
            qualified_cells = meanFR_dict['qualified_cells']
            img_idces = meanFR_dict['img_idces_dict']['img_idces']
            id_img_idces = meanFR_dict['img_idces_dict']['id_img_idces']
            
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
                
                meanFR_single_cell = [meanFR[cell_idx, id_img_idces[_]] for _ in range(50)]
                meanFR_.append(np.array(meanFR_single_cell))
                
                # ----- 1. one way ANOVA
                p = stats.f_oneway(*meanFR_single_cell)[1]    
                p_list.append(p)
                
                # ----- 2. mean+2SD
                # [notice] use: | si | wsi | mi | wmi | n |
                #FIXME
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
            
            # ----- FIXME
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
        
        return cell_stats
        
    # ----- plot
# =============================================================================
#     for cell_idx in np.intersect1d(si, qualified_cells):
#         meanFR_id = meanFR_[cell_idx]
#         
#         fig, ax = plt.subplots(figsize=(10,10))
#         ax.scatter(np.arange(50), np.mean(meanFR_id, axis=1))
#         ax.hlines(np.mean(meanFR_id)+2*np.std(meanFR_id), 0, 49, color='red', label='th')
#         ax.hlines(np.mean(meanFR_id)+2*np.std(np.mean(meanFR_id, axis=1)), 0, 49, color='teal', label='ref')
#         ax.set_title(f'{cell_idx}')
#         ax.legend()
#         plt.show()
# =============================================================================

    # ===== module 3. raster plot
    def human_neuron_raster_plot(self, data_set='CelebA', CellToPlot:list=None):
        """
            [task] find out the used variables
                
            - data_set -- taskInstruction
        """

        plt.rcParams.update({"font.family": "Times New Roman"})
        
        # -----
        cell_plot_path = os.path.join(self.human_neuron_stats, 'cell plot')
        utils_.make_dir(cell_plot_path)
        
        # [comment] seems can be simplified
        colorpool_jet = plt.get_cmap('jet', 50)
        colors = [colorpool_jet(i) for i in range(50)]
        
        # ----- [notice] see what to use
        FR_stats = self.human_neuron_get_firing_rate()
        
        # -----
        meanFR_dict = self.human_neuron_sort_FR()
        meanFR = meanFR_dict['meanFR']
        
        # -----
        cell_stats = self.humane_identity_cell_selection()
        
        if CellToPlot == None:
            CellToPlot = list(cell_stats['cell_types_dict'].keys())
        
        # -----
        label = sio.loadmat(os.path.join(self.root_process, 'Label.mat'))['label'].reshape(-1)
        # -----
        
        # -----
        neuron_session_idces = self.Spikes['vCell'].reshape(-1) - 1    # <- session_idx ID, 0-based
        time_stamps_all_cells = [_.reshape(-1) for _ in self.Spikes['timestampsOfCellAll'].reshape(-1)]
        
        sessions = self.get_session_idces()     # <- stores time periods of each trial (each session_idx)
        session_idx_dir = os.path.join(self.root_data, 'Events Files')     # <- store timestamps for each responses, self.root_data: osfstorage-archive
        
        # consider add a code like self.get_periods() to make the code more clear and easy to read
        all_periods = []     # provides the time range of one single trial with label
        for session_idx in tqdm(range(len(sessions)), desc='Load sessions'):
            # ↓ 'periods' contains 3 columns: trial indices, timestamps to 500 ms before stumuli onset, timestamps to 1500 ms after stimuli onset
            periods = sio.loadmat(os.path.join(session_idx_dir, sessions[session_idx]+'.mat'))['periods']     
            all_periods.append(periods)
        
        # ---
        beh_stats = self.human_neuron_get_beh()
        behavior = beh_stats['beh']
        
        beforeOnset = 500     # [notice]
        
        # -----
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
        
        # ----- [notice] im_code is in dict format now
        for cell_type in CellToPlot:  # for each neuron
        
            cell_type_plot_path = os.path.join(cell_plot_path, f'{cell_type}')
            utils_.make_dir(cell_type_plot_path)
        
            if len(cell_stats['cell_types_dict'][cell_type]) > 9:
                cell_to_plot = np.random.choice(cell_stats['cell_types_dict'][cell_type] ,2)
            
            elif len(cell_stats['cell_types_dict'][cell_type]) != 0:
                cell_to_plot = cell_stats['cell_types_dict'][cell_type]
                
            else:
                print(f'[Codinfo] no cells of [{cell_type}]')
                pass
                
            for cell_idx in tqdm(cell_to_plot, desc=f'{cell_type}'):

                self.session_idx = neuron_session_idces[cell_idx]  # session_idx
                
                if self.session_idx < 10:
                    im_code = self.img_idces_dict
                else:
                    im_code = self.img_idces_new_dict
            
                time_stamps = time_stamps_all_cells[cell_idx]
                
                periods = all_periods[self.session_idx]     # e.g (550, 3)
                
                image_sequence = behavior[self.session_idx]['code']     # the displayed img list
                
                # -----
                FR_tmp = FR_stats[cell_idx]['spike_count']     # FR for this cell
                FR_PSTH = FR_stats[cell_idx]['PSTH_250']
                
                # -----
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
                back_id = behavior[self.session_idx]['back_id']     # local version, 0-based
                
                # -----
                image_sequence = np.delete(image_sequence, back_id) - 1     # 0-based
                periods = periods[np.delete(np.arange(periods.shape[0]), back_id), :]     # e.g. (500, 3) the first column is the idx in original image_sequence
                
                FR_tmp = np.delete(FR_tmp, back_id)
                FR_PSTH = np.delete(FR_PSTH, back_id, axis=0)
                
                # -----
                vimg_ID = np.array([im_code[_] for _ in image_sequence])      # id of each image
                # -----
                
                
                # calculating mean firing rate
                FR_ID = []
                PSTH_ID = []
                for idd in range(1, 51):
                    FR_ID.append(FR_tmp[np.where(vimg_ID==idd)[0]])
                    PSTH_ID.append(FR_PSTH[np.where(vimg_ID==idd)[0], :])
                    
                # ----- stats
                FR_ID_all = np.array([__ for _ in FR_ID for __ in _])
                FR_ID_all_mean = np.mean(FR_ID_all)
                FR_ID_std = np.std(FR_ID_all)
                
                FR_ID_mean = np.array([np.mean(_) for _ in FR_ID])
                FR_ID_ref = np.std(FR_ID_mean)
                
                # ----- return jagged indsOfGrps
                indsOfGrps = []
                
                for idd in range(1, 51):    # for each ID
                    
                    positions = np.where(vimg_ID == idd)[0]
                    
                    img_labels = image_sequence[positions]
                    ID = np.array([im_code[_] for _ in img_labels])
                    
                    if np.unique(ID) != idd:
                        raise RuntimeError(f'[Codinfo] ID check failed for [{idd}]')
                    
                    period = periods[positions, :]     # order in experiment | trial start time | trial end time
                    
                    # | order in experiment | trial start time | trial end time | img label | ID | session |
                    period = np.hstack((period, np.vstack((img_labels, ID)).T))     
                    
                    indsOfGrps.append(period[period[:,0].argsort()])     # is this necessary?
    
                # ----- this should return neat spikesToPlot
                spikesToPlot = self.getTimestampsOfBubbles(time_stamps, indsOfGrps)     # [notice] the current spikesToPlot has unite form of (500,)
            
                # ----- subplot 1
                # [warning] this section has warnings, later to fix that
                fig = plt.figure(figsize=(29.7,21.0))
    
                # ---
                spikeheight = 2
                spikewidth = 2
                
                axes_0 = plt.gcf().add_axes([0.05, 0.45, 0.55, 0.475])
                
                self.plotSpikeRasterMain(axes_0, spikes=spikesToPlot, colors=colors, spikeheight=spikeheight, spikewidth=spikewidth)
                
                axes_0.vlines(500, -1, 501, linestyle='-', alpha=0.75, color='gray', label='image on')
                axes_0.vlines(1500, -1, 501, linestyle='--', alpha=0.75, color='gray', label='image off')
                
                axes_0.set_xlim([0, 2000])
                axes_0.set_ylim([- spikeheight/2, 500 + spikeheight/2])
                
                axes_0.tick_params(axis='both', labelsize=20, width=2, length=5, labelcolor='black', labelbottom=True)
                axes_0.set_xlabel('Time (ms)', fontsize=24)
                axes_0.set_ylabel('Image (10 imgs per ID)', fontsize=24)
                #axes_0.set_yticks([])
                axes_0.set_title('Raster Plot', fontsize=28)
                axes_0.legend(fontsize=24)
                
                # ---
                axes_1 = plt.gcf().add_axes([0.65, 0.45, 0.3, 0.475])
                
                axes_1.barh(np.arange(50), FR_ID_mean, color=colors)
                
                for idx, _ in enumerate(FR_ID):
                    
                    sem = np.std(FR_ID[idx])/np.sqrt(len(FR_ID[idx]))
                    
                    axes_1.scatter(FR_ID_mean[idx] + sem, idx, color='black', marker='d')
                    axes_1.scatter(FR_ID_mean[idx] - sem, idx, color='black', marker='d')
    
                    axes_1.hlines(idx, FR_ID_mean[idx] - sem, FR_ID_mean[idx] + sem, linestyle='--', color='black')
                
                axes_1.vlines(FR_ID_all_mean+2*FR_ID_std, -1, 50, linestyle='--', alpha=0.75, color='red', label='threshold')
                axes_1.vlines(FR_ID_all_mean+2*FR_ID_ref, -1, 50, linestyle='--', alpha=0.75, color='teal', label='ref')
                
                axes_1.tick_params(axis='both', labelsize=20, width=2, length=5, labelcolor='black', labelbottom=True)
                axes_1.set_xlabel('Firing Rate (Hz)', fontsize=24)
                axes_1.set_ylabel('ID', fontsize=24)
                #axes_1.set_yticks([])
                
                axes_1.set_xlim([0, np.max([np.max(FR_ID_mean) + sem, FR_ID_std])*1.2])
                axes_1.set_ylim([-0.5, 49.5])
    
                axes_1.legend(fontsize=24)
                axes_1.set_title('Mean Firing Rate', fontsize=28)
                
                # ---
                PSTH_ID_ = np.array([np.mean(_, axis=0) for _ in PSTH_ID])
                
                axes_2 = plt.gcf().add_axes([0.05, 0.05, 0.55, 0.325])
                
                fig_psth = axes_2.imshow(PSTH_ID_, origin='lower', aspect='auto', cmap='turbo')
                axes_2.set_xticks([0,5,10,15,20,25,30])
                axes_2.set_xticklabels([250, 500, 750, 1000, 1250, 1500, 1750], fontsize=20)
                axes_2.tick_params(axis='both', labelsize=20, width=2, length=5, labelcolor='black', labelbottom=True)
                axes_2.set_xlabel('Time (ms) | cover from 250ms to 2000ms', fontsize=24)
                axes_2.set_ylabel('ID', fontsize=24)
                #axes_2.set_yticks([])
                axes_2.set_title('PSTH (window: 50ms, step: 250ms)', fontsize=28)
                
                cbar_ax = fig.add_axes([0.6125, 0.05, 0.01, 0.325])  # [left, bottom, width, height]
                cbar = fig.colorbar(fig_psth, cax=cbar_ax)
                cbar.ax.tick_params(labelsize=24)
                
                # --- table for information
                col_labels = ['cell', 'session', 'patient', 'all cells']
                row_labels = ['max', 'min', 'mean', 'std', 'median']
                
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
                
                axes_3.set_title('Stats (Firing Rate)', fontsize=28)
                
                # ---
                cell_attr = cell_stats['cell_attr'][cell_idx]
                fig.suptitle(f"Cell No. {cell_idx} | Session No. {cell_attr['session_idx']} | Session: {cell_attr['session']} | ({cell_attr['patient_session_idx']+1}/{cell_attr['patient_sessions_num']})", fontsize=32)
                
                plt.tight_layout()
                #plt.show()
                
                plt.savefig(cell_type_plot_path + f'/{cell_idx}.png', bbox_inches='tight')
                plt.savefig(cell_type_plot_path + f'/{cell_idx}.eps', format='eps', bbox_inches='tight')
                plt.close()

    def plotSpikeRasterMain(self, ax, spikes, colors, spikeheight=1, spikewidth=1, start_time=0, end_time=2000):
        
        """
            [Oct 19, 2023]
            some of the hyper paramaters are based on this experiments, needed to be changed in future use
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
                    # -----
  
    def getTimestampsOfBubbles(self, timestampsOfCell, indsOfGrps):
        """
            prepare bubbles trials for plotting of raster (with color info)
            indsOfGrps: list of periods (each a list of trials)     [ order in experiment | trial start time | trial end time | img label | ID ]
            
            urut/nov09
            
            timestampsOfCell: disordered img sequence
        """

        spikesToPlot = []

        for inds in indsOfGrps:     # for each group
          
            trialTimestamps = self.getRelativeTimestamps(timestampsOfCell, inds)     # obtain the timestamps for one ID
            
            # ----- intergrity check
            if self.session_idx < 10: 
                img_idces = np.array([self.adjust_idx_dict[_] for _ in sorted([_[1] for _ in trialTimestamps])])
            else:
                img_idces = np.array(sorted([_[1] for _ in trialTimestamps]))
                
            # --- ID must be one
            ID = np.unique([trialTimestamps[_][0] for _ in range(len(trialTimestamps))]).item()

            entire_imgs = self.id_img_idces_dict[ID-1]
            
            if img_idces.shape != entire_imgs.shape:
                missed_imgs = np.setdiff1d(entire_imgs, img_idces)
            
                for _ in missed_imgs:
                    trialTimestamps.append([ID, _, np.nan, np.nan])
                
            # --- sort based on label
            trialTimestamps = sorted(trialTimestamps, key=lambda x:x[1])
            
            # ----- add into all
            for _ in trialTimestamps:
                spikesToPlot.append(_)
        
        return spikesToPlot
        
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
            
            # | ID | img_idx | img_num | relative_timestamps |
            trialTimestamps.append([periods[i, 4].astype(int), periods[i, 3].astype(int), periods.shape[0], (timestampsOfCell[inds] - periods[i, 1])/1000])
            
        return trialTimestamps

    
# ======================================================================================================================
    
    def color_cube(self, num_colors):
        values = np.linspace(0, 1, num_colors)
        colors_arr = np.zeros((num_colors, 3))
        colors_arr[:, 0] = (np.sin(2 * np.pi * values)+1)/2  # R
        colors_arr[:, 1] = (np.sin(2 * np.pi * values + (4 * np.pi / 3))+1)/2  # G
        colors_arr[:, 2] = (np.sin(2 * np.pi * values + (2 * np.pi / 3))+1)/2  # B
        
        cmap = colors.ListedColormap(colors_arr)

        return colors_arr, cmap
    
# ======================================================================================================================

# set the input is the folder



    
# ======================================================================================================================
# ----- 
def compare_beh_and_behm(behavior):
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

    test = Human_Neuron_Records_Process()
    
    #test.human_neuron_sort_FR()
    
    #test.humane_identity_cell_selection()
    
    #FIXME
    test.human_neuron_raster_plot()
    
