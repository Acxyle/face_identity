#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 13:35:41 2023

@author: acxyle-workstation
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

# =============================================================================
#FIXME
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

        self.root_process = os.path.join(root, 'osfstorage-archive-su_/')     # <- contains the processed Bio data (eg. PSTH) calculated from Matlab
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
        
        # [notice] seems only used in plot_raster
        self.FR_time_range = [750, 1750]
        self.binW = 250
        self.preStim = 500
        self.postStim = 1000
        self.timelim = [0, 1500]
        self.timeTick = [0, 500, 1000, 1500]
        self.timeLabel = [-0.5, 0., 0.5, 1.]    
        
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
            
            meanFR = meanFR_dict['meanFR']
            qualified_cells = meanFR_dict['qualified_cells']
            
        else:
        
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
            CelebA_image_idces = sio.loadmat(os.path.join(self.root_data, 'SingleNeuron/Code/CelebA_Image_Code.mat'))
            adjust_idx = CelebA_image_idces['AdjustInd'].reshape(-1).astype(np.int16) -1   
            #im_code = CelebA_image_idces['im_code'].reshape(-1)     # [question] 53 identities, and this 'im_code' in 'CelebA_Image_Code.mat' and 'CelebA_Image_Code_new.mat' are not the same
            
            # ----- 5. start meanFR calculation
            if data_set == 'CelebA':
                
                meanFR = np.full((len(FR_stats), 500), np.nan)     # empty response map waited to receive values
                
                for cell_idx in tqdm(range(len(FR_stats))):     # for each neuron
                    
                    session_idx = neuron_session_idces[cell_idx]     # get session idx
                
                    image_sequence = behavior[session_idx]['code'].copy()     # delete back_id in img_list
                    back_id = behavior[session_idx]['back_id'].copy()
        
                    image_sequence = np.delete(image_sequence, back_id)     # image sequence without back_id
                    
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
                        
                        if len(image_sequence) != 500:
    
                            new_FR = np.full(500, np.nan)  
                            
                            for _ in range(500):     # for each image
                                used_imgs = np.where(sorted_image_idces == (_+1))[0]     # for most of the cases, 1 record for 1 img
                                if len(used_imgs) != 0:
                                    new_FR[_] = np.mean(FR[used_imgs])   # take average 
                                    
                            for _ in range(500):
                                if adjust_idx[_] >= 0:
                                    adjustFR[_] = new_FR[adjust_idx[_]]
                        else:
                            
                            for _ in range(500):
                                if adjust_idx[_] >= 0:
                                    adjustFR[_] = FR[adjust_idx[_]]
                        
                        meanFR[cell_idx, :] = adjustFR  # Note: Python uses 0-based indexing
                        
                    else:
                        if len(image_sequence) != 500:
                            new_FR = np.full(500, np.nan)
                            
                            for _ in range(500):
                                used_imgs = np.where(sorted_image_idces == (_+1))[0]
                                if len(used_imgs) != 0:
                                    new_FR[_] = np.mean(FR[used_imgs])
                                    
                            meanFR[cell_idx, :] = new_FR
                            
                        else:
                            meanFR[cell_idx, :] = FR
                           
                            
                # ----- 6. repair because of image errors
                CelebA_sort_idx_new = sio.loadmat(os.path.join(self.root_data, 'SingleNeuron/Code/CelebA_Image_Code_new.mat'))
                
                id_img_idces = CelebA_sort_idx_new['id_code'].reshape(-1)     # each id contains 10 imgs
                id_img_idces = np.array([_.reshape(-1)-1 for _ in id_img_idces])
                
                # --- 1. face 121 is a mis-identified photo of identity 6, replace the FR to average FR of other 9 faces
                meanFR[:, 120] = np.nanmean(meanFR[:, id_img_idces[5][:-1]], 1)
                
                # --- 2. replace the problemd face with mean of that ID for the problemed ID (only for the first 381 neurons),in
                # which another 2 faces(2 trials) were mis-identified, results are similar when replaced with Nan
                defective_ids = [17, 39]
                for _ in defective_ids:
                    useful_idces = id_img_idces[_][-1]
                    meanFR[:381, useful_idces] = np.nanmean(meanFR[:381, id_img_idces[_][:-1]], 1)
                    
                meanFR_dict = {
                    'meanFR': meanFR,
                    'qualified_cells': qualified_cells
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

        return meanFR, qualified_cells
        
    
    def human_neuron_get_firing_rate(self, time_window=250, time_step=50):  
        """
            'Spikes.mat': b'MATLAB 5.0 MAT-file, Platform: MACI64, Created on: Thu Dec 30 12:11:01 2021'
            
            contains combined neurons from all session_idces. For each neuron:
    
            'timestampsOfCellAll' - timestamps (in μs)
            'vCell' - session_idxion ID
            'vCh' - channel ID
            'vClusterID' - cluster ID
            'areaCell' - recording brain area
            
            note that these variables were all matched.
            
            There are three variables that describe spike sorting quality. 
            
            'IsoDist' - the isolation distance value for each cluster (i.e., a single neuron). 
            
            'statsSNR' contains 6 columns: 
                1 - session_idxion ID, 
                2 - channel ID, 
                3 - cluster ID, 
                4 - the average signal-to-noise ratio (SNR), 
                5 - inter-spike intervals (ISI) that are below 3 ms,
                6 - peak SNR. 
            
            'statsProjectAll' contains 5 columns: 
                1 - session_idxion ID, 
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
            
            sessions = self.get_session_idces()     # <- stores time periods of each trial (each session_idxion)
            session_idxion_dir = os.path.join(self.root_data, 'Events Files')     # <- store timestamps for each responses, self.root_data: osfstorage-archive
            
            # consider add a code like self.get_periods() to make the code more clear and easy to read
            all_periods = []     # provides the time range of one single trial
            for session_idxion in tqdm(range(len(sessions)), desc='Load session_idces'):
                # ↓ 'periods' contains 3 columns: trial indices, timestamps to 500 ms before stumuli onset, timestamps to 1500 ms after stimuli onset
                periods = sio.loadmat(os.path.join(session_idxion_dir, sessions[session_idxion]+'.mat'))['periods']     
                all_periods.a_end(periods)
            
            PSTH_start = 500 - time_window     # 500 ms is image onset
            PSTH_end = 2000 - time_window     # 2000 ms is the end of one trial
            
            num_frames = int((PSTH_end - PSTH_start)/time_step + 1)     # <- number of time bins
            neuron_session_idces = self.Spikes['vCell'].reshape(-1)     # <- session_idxion ID
            
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
                beh_dict.a_end(behavior_data_dict)  
                
                Acc.a_end(np.sum(behavior_data_dict['vResp'])/len(behavior_data_dict['back_id']))     # [notice] looks like some records have significantly low accuracy
                
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
        sessions =[
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
    
    # ==================================================================================================================
    
    def human_neuron_raster_plot(self, data_set='CelebA'):
        """
            [task] find out the used variables
            [task] get the proper data structure for python use
            [task] make sure what is session_idxion ,trial, ...
        """
        # [comment] seems can be simplified
        colorpol, _ = self.color_cube(60)
        colors = []
        for i in range(50):
            colors.a_end(colorpol[i,:])
        colors = np.array(colors)
        
        '''
            [notice] load FR directly
        '''
        vCell = self.FiringRate['vCell'].reshape(-1)  # (1,2082)
        timestampsOfCellAll = self.FiringRate['timestampsOfCellAll'].reshape(-1)  # -> time_stamps_all_cells
        periodsAll = self.FiringRate['periodsAll'].reshape(-1)  # (1,40) with variable sub_arrays
        
        behavior = self.CelebA_meanFR_Cor['beh'].reshape(-1)
        #beh_keys = beh.dtype.names  #['iT', 'vResp', 'vCorr', 'RT', 'code', 'vTruth', 'back_id', 'isEyeTrack', 'stimWindowSize', 'T', 'windowRect']
        beforeOnset = self.CelebA_meanFR_Cor['beforeOnset'].reshape(-1)  # 
        session_idces = self.CelebA_meanFR_Cor['session_idces'].reshape(-1)
        meanFR = self.CelebA_meanFR_Cor['meanFR']
        
        label = self.Label['label'].reshape(-1)
        
        useSpikes = []
        # -----
        # [comment] manual set
        CellToPlot = [197,14,78]
        
        for ii in range(len(CellToPlot)):  # for each neuron
            cell_idx = CellToPlot[ii]  # 197
            print('cell_idx =', cell_idx)
            cell_idx = cell_idx-1
            session_idx = vCell[cell_idx]  # session_idxion
            session_idx = session_idx-1
            if session_idx < 11-1:
                im_code = sio.loadmat(os.path.join(self.root_process, data_set+'_Image_Code.mat'))['im_code'][0]
                im_code[78-1] = 51  # [notice] according to the document, this is the fixation of incorrect label
                im_code[98-1] = 52  # perhaps can correct it later if have time
            else:
                im_code = sio.loadmat(os.path.join(self.root_process, data_set+'_Image_Code_new.mat'))  # [noice] this thing contains variabl id_img_idces
                im_code = im_code['im_code'][0]
                id_img_idces = im_code['id_code'][0]
        
            timestampsOfCell = timestampsOfCellAll[cell_idx]
            periods = periodsAll[session_idx]
            
            image_sequence = behavior[session_idx]['code'][0]
            vimg_ID = im_code[image_sequence-1]
            
            indsOfGrps = []
            
            for idd in range(1, 51):
                tmp_ind = np.where(vimg_ID == idd)[0]
                tmp_ind = np.setdiff1d(tmp_ind, behavior[session_idx]['back_id'][0]-1)
                #tmp_ind = [x for x in tmp_ind if x not in behavior[session_idx]['back_id'][0]]

                tmp_per = periods[tmp_ind, :]
                idx = np.argsort(tmp_per[:, 0])[::-1]  # [::-1] means descending order
                indsOfGrps.a_end(tmp_per[idx, :])
            indsOfGrps = np.array(indsOfGrps, dtype=object)
        
            # [outside function] getTimestampsOfBubbles()
            spikesToPlot, colortill, nrTrialsTot = self.getTimestampsOfBubbles(timestampsOfCell, indsOfGrps)
            useSpikes.a_end(spikesToPlot)
        
            plt.subplots(figsize=(10,10))

            # subplot 1
            # [warning] this section has warnings, later to fix that
            ax1 = plt.subplot(1, 2, 1)
            
            # [outside function] plotSpikeRasterMain()
            hs = self.plotSpikeRasterMain(spikes=spikesToPlot, colorTill=colortill, colors=colors, range_=np.arange(1, nrTrialsTot+1), spikeheight=2, spikewidth=2)
            lw = 2
            plt.plot([beforeOnset, beforeOnset], [1, nrTrialsTot], '-', linewidth=lw, color=[0.7, 0.7, 0.7])
            if session_idces[session_idx]['data_set'][0] == 'Loc_Face':
                plt.plot([beforeOnset + self.preStim, beforeOnset + self.preStim], [1, nrTrialsTot], '-', linewidth=lw, color=[0.7, 0.7, 0.7])
            else:
                plt.plot([beforeOnset + self.postStim, beforeOnset + self.postStim], [1, nrTrialsTot], '-', linewidth=lw, color=[0.7, 0.7, 0.7])
            
            ax1.set_ylim([0, nrTrialsTot])
            ax1.set_xlim(self.timelim)
            ax1.set_xticks(self.timeTick)
            ax1.set_xticklabels(self.timeLabel)
            ax1.set_ylabel('Trial Number (sort by ID)')
            ax1.tick_params(axis='both', labelsize=16, width=2, length=5, labelcolor='black', labelbottom=True)

            
            # calculating mean firing rate
            id_img_idces = sio.loadmat(os.path.join(self.root_process, session_idces[session_idx]['data_set'][0] + '_Image_Code_new.mat'))['id_img_idces'][0]
            FR_ID = []
            meanFR_ID = np.zeros(50)
            stdFR_ID = np.zeros(50)
            
            for idd in range(50):
                FR_ID.a_end(meanFR[cell_idx, id_img_idces[idd][0]-1])
                meanFR_ID[idd] = np.nanmean(FR_ID[idd])
                stdFR_ID[idd] = np.nanstd(FR_ID[idd]) / np.sqrt(FR_ID[idd].size - 1)
            
            # box plot
            
            tmp = []
            for i in range(1,51):
                tmp.a_end(meanFR[cell_idx,np.where(label==i)[0]])
            tmp = np.array(tmp)
            
            ax2 = plt.subplot(1, 2, 2)
            ax2.boxplot(tmp.T, widths=0.5, vert=False, patch_artist=True, boxprops=dict(facecolor=(0.75, 0.75, 0.75)))
            ax2.set_xlabel('Firing Rate')
            ax2.tick_params(axis='both', labelsize=12, width=2, length=5, labelcolor='black', labelbottom=True)
            ax2.set_yticklabels('')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            plt.show()
            ax1.remove()
            ax2.remove()
        
        useSpikes = np.array(useSpikes, dtype=object)
    
    
    def plotSpikeRasterMain(self, spikes, range_=None,linesflag=0,endtime=None,xlabelstr='Time [ms]',ylabelstr='',fs=[],colors='',colorTill=0,spikeheight=0.7,spikewidth=0.5):
        # main function for plotting rasters.
        #
        # SPIKERASTER - Spike raster of multiple neurons
        #    SPIKERASTER(spikes, options), plot spike trains given in variable
        #    SPIKES, which is in format [neuron time] or in a sparse matrix
        #    with time down columns and neuron number across rows.  If
        #    there are no spikes (SPIKES is empty) then a plot is created,
        #    with the dimensions specified by the 'Range' and 'EndTime'
        #    variables.
        #    
        #    Optional arguments: 
        # 
        #    'Range', RANGE  Plot only neurons specified in the vector
        #                    RANGE.  If neurons are specified that have no
        #                    spikes a line will still be made for them in
        #                    the raster.
        #    'EndTime', ET   Plot up until ET
        #    'Fs', fs        Set the sampling frequncy to FS.  This scales
        #                    the TIME by 1/FS and is especially useful for
        #                    sparse matrix spiketrains.
        #    'Lines'         Make a line for each spike to sit on
        #    'Xlabel'        Set the x-axis label.  The default is 'time'.
        #    'Ylabel'        Set the y-axis label.  The default is no label.
        #
        #    'spikeheight'   height of spike (line)
        #
        #    'colortill': array of numbers that indicate the color scheme for a
        #    particular trial. First entry: till when should the first color be
        #    used (including this trial). second and further entries: trials
        #    smaller this number (but bigger or equal the previous) have the next color. 
        #    for example,to switch colors every two trials, colortill=[2 5 7 9 ...]
        #    (this odd scheme is for compatibility reasons with legacy code).
        #    coloring starts at trial nr 1 (bottom of plot).
        #
        #    'colors': list of color codes. if more colors are needed then
        #    available, this code cycles through the available once in sequential
        #    order.
        #
        #    'spikewidth' -> with of line of spike
        #
        #    returns: array of handles of lines. only the handle of the first spike
        #    in each trial (line) is returned.
        #
        #    modified extensively by: ueli rutishauser <urut@caltech.edu>
        #    Original Author:     David Sterratt <David.C.Sterratt@ed.ac.uk>
        
        # ---------------------------------------------------------------------
        # [acxyle] looks like below section can be modified by python grammar
        handles = []

        # defaults
        colorMode = False         
                    
        # ---------------------------------------------------------------------
    
        # Check to see if the input is a sparse matrix with time down rows and neurons across columns
        # [acxyle] check whether a input matrix is sparse or not. A sparse matrix contains a large number of zero elements, looks in python only types like scipy.sparse.* will triger below judgement
        if issparse(spikes):  
            t, n = np.where(spikes)
            spikes = np.column_stack((n, t))
    
        # make sure that there is at least 1 spike for each neuron, add one before 0 to make sure. this is important!! otherwise lines are ski_ed in the plot.
        if spikes.size > 0:
            if range_ is None:  # if it hasn't been set externally alread
                range_ = np.arange(spikes[:, 0].min(), spikes[:, 0].max() + 1)  # Neurons to plot
            for i in range_:
                spikes = np.vstack((spikes, [i, -10000]))
        else:
            range_ = np.array([1])
            endtime = 1
    
        # see if coloring mode is on
        if len(colors) > 0:
            colorMode = True
    
        # Divide by the sampling frequency, if set
        if spikes.size > 0 and fs != []:
            spikes[:, 1] = spikes[:, 1] / fs
    
        # If endtime hasn't been specified in the arguments, set it to the time of the last spike of the neurons we want to look at (that is those specified by range_).
        if endtime is None:
            endtime = np.max(spikes[np.isin(spikes[:, 0], range_), 1])
    
        # ----- plot -----
        # Prepare the axes
        h = plt.gca()
        
        # Save existing properties
        if "lines.linestyle" in plt.rcParams:
            oldls = plt.rcParams["lines.linestyle"]
        else:
            oldls = ["-"]
        if "axes.prop_cycle" in plt.rcParams:
            oldco = plt.rcParams["axes.prop_cycle"]
        else:
            oldco = plt.rcParams["axes.color_cycle"]
        
        # Full, Black lines
        h.set_prop_cycle(color=[(0, 0, 0)])
        plt.rcParams["lines.linestyle"] = "solid"
        
        # Do the plotting one neuron at a time
        if spikes.size != 0:
            for n in range(len(range_)):
                s = spikes[(spikes[:, 0] == range_[n]) & (spikes[:, 1] <= endtime), 1]
                lineHandle = plt.plot(np.vstack([s, s]), np.vstack([(n - spikeheight / 2) * np.ones(s.size), (n + spikeheight / 2) * np.ones(s.size)]), linewidth=spikewidth)
                handles.a_end(lineHandle[0])
        
                # if flags are set, change color of the spike
                if colorMode:
                    if range_[n] <= colorTill[0]:
                        lineHandle[0].set_color(colors[0])
                    else:
                        ind = np.where(colorTill > range_[n])[0]
                        if len(ind) == 0:
                            ind = len(colors)
                        lineHandle[0].set_color(colors[(ind[0] - 1) % len(colors)])
        
        # Make the plot the right length but only when we're not adding to
        # a plot
        if h.get_autoscale_on():
            if endtime > 0:
                h.set_xlim([0, endtime])
            h.set_ylim([0.5, len(range_) + 0.5])
        
        # Add lines for the spikes to sit on if required
        if linesflag:
            xline = h.get_xlim()
            for n in range(len(range_)):
                plt.plot(xline, [n + 1, n + 1])
        
        plt.xlabel(xlabelstr)
        plt.ylabel(ylabelstr)
        
        # Restore existing properties
        plt.rcParams["lines.linestyle"] = oldls
        plt.rcParams["axes.prop_cycle"] = oldco

    def getTimestampsOfBubbles(self, timestampsOfCell, indsOfGrps):
        # prepare bubbles trials for plotting of raster (with color info)
        # indsOfGrps: cell array of list of periods (each a list of trials)
        # urut/nov09
        
        # [acxyle] all 3 ourpur will be used in plotRasterMain()
        
        spikesToPlot = []
        trialNr = 0
        colortill = []
    
        for k in range(len(indsOfGrps)):  # [acxyle] for each ID
            # [acxyle] call below function
            # output: relative timestamp in one array | input: timestamps, img list (10 imgs) of one identity
            spikesOfCat = self.getRelativeTimestamps(timestampsOfCell, indsOfGrps[k])
            if k == 0:
                colortill.a_end(len(spikesOfCat))
            else:
                if k == 1:
                    offset = 1
                else:
                    offset = 0
                colortill.a_end(colortill[k-1] + len(spikesOfCat) + offset)
    
            for kk in range(len(spikesOfCat)):  # [acxyle] for each trial
                trialNr += 1
                trialSpikes = np.column_stack((np.repeat(trialNr, len(spikesOfCat[kk])), spikesOfCat[kk]))
                spikesToPlot.a_end(trialSpikes)
                
        colortill = np.array(colortill)
        spikesToPlot = np.vstack(spikesToPlot)
        nrTrialsTot = trialNr
        
        return spikesToPlot, colortill, nrTrialsTot
        
    def getRelativeTimestamps(self, timestampsOfCell, periods):
        # reference timestamps to beginning of the trial
        # periods: each row is one trial. 3 columns: trial nr, from, to.
        # returns a cell array; each item contains the timestamps of one trial. the number of trials is equal to the number of rows in periods.
        # urut/dec07
        
        # [acxyle] output:timestamps of one trial | input: timestamps, time points of trial duration(stimuli timestamps)
        
        # [acxyle] call below function
        # [acxyle] only return the cell array of qualified timestamps and abandon the reset 2 attributes
        trialsTimestamps,_,_ = self.getTimestampsOfTrials(timestampsOfCell, periods[:,1:3]) 

        for i in range(len(trialsTimestamps)):
            # [acxyle] remove offset and convert to ms, convert the abs time to relative time
            trialsTimestamps[i] = (trialsTimestamps[i] - periods[i,1])/1000
        
        return trialsTimestamps

    def getTimestampsOfTrials(self, timestampsOfCell, stimuliTimestamps):
        # returns timestamps of trials in a cell array
        # stimuliTimestamps: first column is begin timestamp, second column is end timestamp of trial, returns a cell array of trials
        # urut/may04
        
        trials=[]
        indsAll=[]
        indsOrigPerTrial=[]
        
        for i in range(stimuliTimestamps.shape[0]):
            # return the idxes of qualified timestamps
            inds = np.where(np.logical_and(stimuliTimestamps[i,0] <= timestampsOfCell, timestampsOfCell <= stimuliTimestamps[i,1]))[0]
            
            trials.a_end(timestampsOfCell[inds])  # cell array of qualified timestamps
            
            # [warning] this looks like an error
            # [update] although it works, needed to be verified later for more details
            indsAll = np.concatenate([indsAll, inds])  # matrix of idxes
            
            indsOrigPerTrial.a_end(inds)  # cell array of idxes
        
        return trials, indsAll, indsOrigPerTrial
    
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
    test.human_neuron_raster_plot()
    