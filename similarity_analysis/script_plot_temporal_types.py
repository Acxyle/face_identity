#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:11:41 2024

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

FSA_root = '/home/acxyle-workstation/Downloads/FSA'
FSA_dir = 'VGG/VGG'
FSA_config = 'Baseline'
FSA_model = 'vgg16'

root = os.path.join(FSA_root, FSA_dir, f'FSA {FSA_config}')

used_unit_types = ['qualified', 'selective', 'strong_selective', 'weak_selective', 'non_selective']