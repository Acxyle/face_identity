#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:50:22 2024

@author: acxyle-workstation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

import sys
sys.path.append('../')
import utils_

from FSA_SVM import FSA_SVM

type_1 = ['qualified', 'selective', 'strong_selective', 'weak_selective', 'non_selective']
type_2 = ['sensitive', 'non_sensitive']

type_3 = ['qualified', 'sensitive', 'non_sensitive', 'selective', 'strong_selective', 'weak_selective', 'non_selective']

FSA_root = '/home/acxyle-workstation/Downloads/FSA'
FSA_dir = 'VGG/SpikingVGG'
FSA_config = 'SpikingVGG16bn_IF_ATan_T4_C2k_fold_'
FSA_model = 'spiking_vgg16_bn'
fold_idx = 4

root = f'{FSA_root}/{FSA_dir}/FSA {FSA_config}/-_Single Models/FSA {FSA_config}{fold_idx}/Analysis'

SVM_dict_1 = utils_.load(os.path.join(root, f'Encode/SVM/SVM {type_1}.pkl'))['acc']
SVM_dict_2 = utils_.load(os.path.join(root, f'SVM/SVM {type_2}.pkl'))

SVM_dict = {**SVM_dict_1, **SVM_dict_2}

utils_.dump(SVM_dict, os.path.join(root, f'SVM/SVM {type_3}.pkl'))