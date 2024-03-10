#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:12:39 2023

@author: acxyle
"""
import torch
import torchvision
import torchvision.transforms as transforms

import pickle
import gzip
import joblib

import skimage
import os
import sys
import numpy as np
import random
import datetime
#import cv2

from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model.tv_ref_classify import utils
from spikingjelly.activation_based import functional

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append('../')
import models_

from spikingjelly import visualizing
from spikingjelly.activation_based import tensor_cache

import subprocess


#TODO ----- need to find a solution to save small pieces and merge them into one single feature map
def spikes_to_frs(feature, select_ratio=0, **kwargs):
    """
        in this function, the feature is compressed by sipikingjelly module, it demands 60GB+ for the first act layer 
        if the A2S time step is greatre than 64
        please read the tutorial of spikingjelly for more information of compression of spikes
        if zlib is used, the compressed file may greater than the original one when the data is diverse
    """
    assert feature[0][0].dtype==torch.uint8, "the data must be compressed 'bool spikes'."

    target_T_idx = int(feature[0][2][1]*select_ratio)

    feature = [np.mean(tensor_cache.bool_spike_to_float(*_).type(torch.uint8).numpy()[-target_T_idx:, :], axis=0) for _ in tqdm(feature, desc='bool_spikes -> frs')]
    feature = np.array(feature).astype(np.float32)

    return feature

def bool_spikes_to_spikes(feature, data_type='ndarray'):
    assert feature[0][0].dtype==torch.uint8, "the data must be compressed 'bool spikes'."
    
    if data_type == 'tensor':
        feature = [tensor_cache.bool_spike_to_float(*_).type(torch.uint8) for _ in tqdm(feature, desc='bool_spikes - > spikes')]
        feature = torch.stack(feature)
        
    elif data_type == 'ndarray':
        
        feature = [tensor_cache.bool_spike_to_float(*_).type(torch.uint8).numpy() for _ in tqdm(feature, desc='bool_spikes - > spikes')]
        feature = np.array(feature)
    
    else:
        raise ValueError
        
    return feature

def _file_system_type(path):
    result = subprocess.check_output(['df', '-T', path], encoding='utf-8')
    lines = result.splitlines()
    return lines[1].split()[1]


def _is_binary(input):
    return np.all((input==0)|(input==1))


def _print(content, message_type='[Codinfo]', symbol='-', padding=2, border_length=None):
    """

    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row_data = [current_time, message_type, content]
    formatted_row = "|".join("{:<{}}".format(' '*padding + item, len(item)+padding*2) for item in row_data)
    
    if border_length:
        border = symbol * (len(formatted_row) + 2) 
    else:
        border = symbol * 20

    print(formatted_row)


# -----
def lighten_color(hex_color, increase_value=80):
    """
        Lightens a given hex color by a specified value.
    """
    increase_value = min(255, increase_value)

    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:], 16)

    r, g, b = min(r + increase_value, 255), min(g + increase_value, 255), min(b + increase_value, 255)

    return f"#{r:02x}{g:02x}{b:02x}"


# ----- pie chart
def plot_pie_chart(fig, ax, values, labels, title=None, colors=None, explode=None):
    """
    
    """
    plt.rcParams.update({"font.family": "Times New Roman"})
    
    if explode is None:
        explode = np.zeros(len(values))
    
    ax.pie(values, colors = colors, labels=labels, autopct='%1.1f%%', pctdistance=0.85, explode=explode)
    
    centre_circle = plt.Circle((0,0),0.70, fc='white')

    fig.gca().add_artist(centre_circle)    

    ax.axis('equal')  
    
    for i, label in enumerate(labels):
        if values[i] < 10:  # Adjust threshold for small slices
            x, y = ax.patches[i].theta1, ax.patches[i].r
            ax.annotate(label + " - " + str(values[i]) + "%", xy=(x, y), xytext=(x+1.2, y+0.5),
                         arrowprops=dict(facecolor='black', shrink=0.05), fontsize=20)
    
    ax.set_title(f'{title}', x=0.45, y=0.45)
    
# ------ sigstar functions
def sigstar(groups, stats, ax, nosort=False):
    """
        local python rewrite for matlab code [sigstar]: https://github.com/raacampbell/sigstar/tree/master
    """
    if stats is None or len(stats) == 0:
        stats = [0.05] * len(groups)
        
    xlocs = np.array(groups)
    
    """
        Optionally sort sig bars from shortest to longest so we plot the shorter ones first
        in the loop below. Usually this will result in the neatest plot. If we waned to 
        optimise the order the sig bars are plotted to produce the neatest plot, then this 
        is where we'd do it. Not really worth the effort, though, as few plots are complicated
        enough to need this and the user can define the order very easily at the command line. 
    """
    if not nosort:
        sort_inds = np.argsort(xlocs[:, 1] - xlocs[:, 0])
        xlocs = xlocs[sort_inds]
        stats = np.array(stats)[sort_inds]
    
    """
        Add the sig bar lines and asterisks 
    """
    H = []
    for i in range(len(groups)):
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_offset = y_range * 0.05
        
        thisY = findMinY(xlocs[i], ax) + y_offset
        
        h = makeSignificanceBar(xlocs[i], thisY, stats[i], ax)
        H.append(h)
        
    """
        Now we can add the little downward ticks on the ends of each line. We are
        being extra cautious and leaving this it to the end just in case the y limits
        of the graph have changed as we add the highlights. The ticks are set as a
        proportion of the y axis range and we want them all to be the same the same
        for all bars.
    """
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    yd = y_range * 0.01
    
    for h in H:
        line = h['line']
        y = line.get_ydata()
        y[0] -= yd
        y[3] -= yd
        line.set_ydata(y)

    return H

def makeSignificanceBar(x, y, p, ax):
    """
    makeSignificanceBar produces the bar and defines how many asterisks we get for a 
    given p-value ★ *
    """
    if p<=1E-3:
        stars='***'
    elif p<=1E-2:
        stars='**'
    elif p<=0.05:
        stars='*'
    elif np.isnan(p):
        stars='N/A'
    else:
        stars='n.s.'
        
    x = np.tile(x, (2,1))
    y = np.tile(y, (4,1))
    
    # [notice] need to find the correct way to retrieve this line
    line, = ax.plot(x.T.ravel(), y, 'C0', linewidth=1.5)
    line.set_label('sigstar_bar')
    
    if not np.isnan(p):
        offset=0.015
    else:
        offset=0.02
    
    starY=np.mean(y)+myRange(ax.get_ylim())*offset
    
    text = ax.text(np.mean(x), starY, stars, horizontalalignment='center', backgroundcolor='none', fontsize=14)
    text.set_label('sigstar_stars')
    
    Y = ax.get_ylim()
    if Y[1]<starY:
        ax.set_ylim([Y[0], starY+myRange(Y)*0.05])
            
    H = {'line': line, 'text': text}
    
    return H
    
def myRange(x):
    return np.max(x)-np.min(x)

def findMinY(x, ax):
    """
    The significance bar needs to be plotted a reasonable distance above all the data points
    found over a particular range of X values. So we need to find these data and calculat the 
    the minimum y value needed to clear all the plotted data present over this given range of 
    x values. 
   
    This version of the function is a fix from Evan Remington
    """
    
    old_xlim = ax.get_xlim()
    old_ylim = ax.get_ylim()
    
    ax.autoscale_view(tight=True)
    
    x[0] += 0.1
    x[1] += 0.1
    
    ax.set_xlim(x)
    
    yLim = ax.get_ylim()
    Y = max(yLim)
    
    #ax.relim()
    #ax.autoscale_view()
    #ax.set_aspect('auto')
    
    ax.set_xlim(old_xlim)
    ax.set_ylim(old_ylim)
    
    return Y     # [question] should return ax

# -----
def generate_threshold(input, alpha=1, delta=2):
    """
        return np.mean(input)+delta*np.std(input), delta(default)=2
    """
    return alpha*np.mean(input)+delta*np.std(input)
    
# -----
def lexicographic_order(num_classes):
    id_order = np.arange(1, num_classes+1).astype(str)
    id_order_idx = np.argsort(id_order)
    id_order_lexical = id_order[id_order_idx].astype(int)
    
    return id_order_lexical

# ----------------------------------------------------------------------------------------------------------------------
def pickle_dump(file, file_path, cmd='wb'):

    assert cmd in ['w', 'wb', 'w+', 'wb+']
    
    if os.path.exists(file_path):
        print(f'\n-----\n[Codwarning] file {file_path} already exists, overwriting...\n-----')
        
    with open(file_path, cmd) as f:
        pickle.dump(file, f, protocol=-1)
        

def pickle_dump_tqdm(file, file_path, cmd='wb'):
    
    assert cmd in ['w', 'wb', 'w+', 'wb+']
    
    if os.path.exists(file_path):
        print(f'\n-----\n[Codwarning] file {file_path} already exists, overwriting...\n-----')

    with tqdm_file_object(file_path, cmd) as f:
        pickle.dump(file, f, protocol=-1)

# FIXME -----
def dump(file, file_path, cmd='wb', dump_type='pickle', verbose=True):
    
    assert cmd in ['w', 'wb', 'w+', 'wb+']
    
    if os.path.exists(file_path):
        _print(f'file {file_path} already exists, overwriting...', '[Codwarning]')
        
    if dump_type == 'gzip':
        assert os.path.splitext(file_path)[-1] in ['.gz', '.tar.gz', '.xz', '.tar.xz']
        with gzip.open(file_path, cmd) as f:
            pickle.dump(file, f, protocol=-1)
        
    elif dump_type == 'pickle':
        assert os.path.splitext(file_path)[-1] in ['.pkl', '.pickle']
        with tqdm_file_object(file_path, cmd, verbose) as f:
            pickle.dump(file, f, protocol=-1)
        
    elif dump_type == 'joblib':
        joblib.dump(file, file_path, compress=0, protocol=-1)     # see joblib.dump() for more details

    else:
        #raise ValueError(f"Invalid dump_type: {dump_type}. Choose from 'pickle', 'gzip', 'joblib', 'json', 'bson', 'msgpack'.")
        raise ValueError(f"Invalid dump_type: {dump_type}. Choose from 'pickle', 'gzip', 'joblib'.")

# ----------------------------------------------------------------------------------------------------------------------
def pickle_load(file_path, cmd='rb'):
    
    assert cmd in ['r', 'rb', 'r+', 'rb+']
    
    if not os.path.exists(file_path):
        raise ValueError(f'[Coderror] can not find the file {file_path}')
    
    else:
        with open(file_path, cmd) as f:
            loaded_file = pickle.load(f)
        return loaded_file

        
def pickle_load_tqdm(file_path, cmd='rb'):
    
    assert cmd in ['r', 'rb', 'r+', 'rb+']
    
    if not os.path.exists(file_path):
        raise ValueError(f'[Coderror] can not find the file {file_path}')
    
    else:
        with tqdm_file_object(file_path, cmd) as f:
            loaded_file = pickle.load(f)
        return loaded_file
        
# -----
def load_feature(file_path, **kwargs):
    
    feature = load(file_path, **kwargs)
    
    if isinstance(feature, list):
        feature = spikes_to_frs(feature, **kwargs)
    
    return feature

# FIXME -----
def load(file_path, cmd='rb', load_type='pickle', verbose=True, **kwargs):
    
    assert cmd in ['r', 'rb', 'r+', 'rb+']
    
    if not os.path.exists(file_path):
        raise ValueError(f'[Coderror] can not find the file {file_path}')
        
    if load_type == 'gzip':
        assert os.path.splitext(file_path)[-1] in ['.gz', '.tar.gz', '.xz', '.tar.xz']
        with gzip.open(file_path, cmd) as f:
            loaded_file = pickle.load(f)
        
    elif load_type == 'pickle':
        assert os.path.splitext(file_path)[-1] in ['.pkl', '.pickle']
        with tqdm_file_object(file_path, cmd, verbose) as f:
            loaded_file = pickle.load(f)
        
    elif load_type == 'joblib':
        assert os.path.splitext(file_path)[-1] in ['.pkl', '.pickle', '.joblib']
        loaded_file = joblib.load(file_path)    

    else:
        #raise ValueError(f"Invalid load_type: {load_type}. Choose from 'pickle', 'gzip', 'joblib', 'json', 'bson', 'msgpack'.")
        raise ValueError(f"Invalid load_type: {load_type}. Choose from 'pickle', 'gzip', 'joblib'.")
    
    return loaded_file


# FIXME --- test version, create a visible progress bar for loading and saving
# FIXME --- add time stamp?
class tqdm_file_object:
    def __init__(self, file_path, mode='rb', verbose=True):
        self.file = open(file_path, mode)
        self.mode = mode
        self.verbose = verbose

        # init progress bar
        if 'r' in mode:
            self.length = os.fstat(self.file.fileno()).st_size
            if self.verbose:
                self.tqdm = tqdm(total=self.length, unit='B', unit_scale=True, desc=f'Loading {file_path}')
        elif 'w' in mode or 'a' in mode:
            if self.verbose:
                self.tqdm = tqdm(unit='B', unit_scale=True, desc=f'Saving {file_path}')

    def read(self, size=-1):
        if 'r' not in self.mode:
            raise NotImplementedError("read() not implemented on file opened in write mode")
        data = self.file.read(size)
        if self.verbose:
            self.tqdm.update(len(data))
        return data
    
    def readline(self, size=-1):
        data = self.file.readline(size)
        if self.verbose:
            self.tqdm.update(len(data))
        return data

    def write(self, data):
        if 'w' not in self.mode and 'a' not in self.mode:
            raise NotImplementedError("write() not implemented on file opened in read mode")
        
        if isinstance(data, pickle.PickleBuffer):     # PickleBuffer, for writing
            data = bytes(data)
        
        self.file.write(data)
        if self.verbose:
            self.tqdm.update(len(data))
            self.tqdm.refresh()  # Refresh the progress bar display
        os.fsync(self.file.fileno())  # Ensure data is written to disk

    def close(self):
        if self.verbose:
            self.tqdm.close()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


        
# ----------------------------------------------------------------------------------------------------------------------
def get_layers_and_units(model_name:str, target_layers:str='full', **kwargs):
    """
        ...
    """
    model_name = model_name.lower()
    
    # ---
    if 'vgg' in model_name:
        
        layers, neurons, shapes = generate_vgg_layers(model_name, **kwargs)
    
    elif 'resnet' in model_name:
        
        layers, neurons, shapes = generate_resnet_layers(model_name, **kwargs)
        
    # ---
    if target_layers == 'full':
    
        return layers, neurons, shapes
    
    elif target_layers == 'feature':
        
        return key_layers_selection(model_name, layers, neurons, shapes, act_only=False, **kwargs)
        
    elif target_layers == 'act':
        
        return key_layers_selection(model_name, layers, neurons, shapes, act_only=True, **kwargs)
        
    else:
        
        raise ValueError(f"model '{model_name}' is not supported")
        
        
# ----------------------------------------------------------------------------------------------------------------------
def key_layers_selection(model_name, layers:list[str], neurons:list[int]=None, shapes:list[int]=None, act_only:bool=False, 
                         return_idx:bool=False, **kwargs):
    """
        ...
    """
    # ----- init
    if 'spiking' in model_name:
        act = 'sn'
    else:
        act = 'an'
    
    # ----- determine layers
    if act_only:
        layers_ = [i for i in layers if f'{act}' in i]
    else:
        if 'vgg' in model_name.lower() or 'baseline' in model_name.lower():
            layers_ = [i for i in layers if f'{act}' in i or 'Pool' in i or 'fc_3' in i]
        elif 'resnet' in model_name.lower():
            layers_ = [i for i in layers if f'{act}' in i or 'Pool' in i or 'fc' in i]
        else:
            raise ValueError(f'[Coderror] Invalid model_name [{model_name}]')
        
    idx = [layers.index(i) for i in layers_]
    layers = layers_
    
    if neurons is not None:
        neurons = [neurons[i] for i in idx]
    
    if shapes is not None:
        shapes = [shapes[i] for i in idx]
    
    # --- return
    if not return_idx:
        return layers, neurons, shapes
    else:
        return idx, layers, neurons, shapes
    

# ----------------------------------------------------------------------------------------------------------------------
def generate_vgg_layers(model_name, T=4, num_classes=50, feature_shape=(3,224,224), **kwargs):
    """
        1. done
        2. need to write a contentional version of layer names
    """
    
    _print(f'Generating layers name for {model_name}...')
    
    # assume the SNN vgg has identical feature map shape with ANN vgg
    if 'spiking' in model_name or 'sew' in model_name:
        model_name = '_'.join(model_name.split('_')[1:])
        act = 'sn'
    else:
        act = 'an'

    # --- structural init
    model_dict = {'vgg5':'O', 'vgg11': 'A', 'vgg13': 'B', 'vgg16': 'D', 'vgg19': 'E'}
    
    cfgs = {
        'O': [64, 'M', 128, 128, 'M'],
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    # --- 0. init
    cba = ['Conv', 'BN', f'{act}']
    ca = ['Conv', f'{act}']
        
    layers = [] 
    
    _l = 1
    _b = 1
    
    # --- 1. features
    for v in cfgs[model_dict[model_name.split('_')[0]]]:     # for each block
        
        if v == 'M':
            layers.append(f'L{_l}_MaxPool')
            _b = 1
            _l += 1
            
        else:
            if 'bn' in model_name:
                layers += [f'L{_l}_B{_b}_{_}' for _ in cba]
            else:
                layers += [f'L{_l}_B{_b}_{_}' for _ in ca]
                
            _b += 1
    
    layers.append('AvgPool')
    
    # --- 2. classifier
    if '5' in model_name:
        layers += ['fc_1', f'{act}_1', 'dp_1', 'fc_2']
        
    else:
        layers += ['fc_1', f'{act}_1', 'dp_1', 'fc_2', f'{act}_2', 'dp_2', 'fc_3']     # add neuron_3 if manually addede the results after act for fc_3
    
    # ----- dummy img
    x = torch.randn(1, *feature_shape)
    
    # --- dummny model
    model = models_.vgg.__dict__[model_name](num_classes=num_classes)
    
    # --- dummy forward
    features = model(x)
    functional.reset_net(model)
    
    units = []
    shapes = []
    for idx, _ in enumerate(features):
        units.append(_.mean(0).detach().cpu().numel())
        shapes.append(_.mean(0).squeeze(0).detach().cpu().numpy().shape)
            
    return layers, units, shapes


# ----------------------------------------------------------------------------------------------------------------------
def generate_resnet_layers(model_name, T=4, num_classes=50, feature_shape=(3,224,224), **kwargs):     # return layers and neurons
    """
        ...
    """
    
    model_dict = {'resnet18': [2, 2, 2, 2], 
                  'resnet34': [3, 4, 6 ,3], 
                  'resnet50': [3, 4, 6, 3], 
                  'resnet101': [3, 4, 23, 3], 
                  'resnet152': [3, 8, 36, 3],
        
                  'spiking_resnet18': [2, 2, 2, 2], 
                  'spiking_resnet34': [3, 4, 6 ,3], 
                  'spiking_resnet50': [3, 4, 6, 3], 
                  'spiking_resnet101': [3, 4, 23, 3], 
                  'spiking_resnet152': [3, 8, 36, 3],
                  
                  'sew_resnet18': [2, 2, 2, 2], 
                  'sew_resnet34': [3, 4, 6 ,3], 
                  'sew_resnet50': [3, 4, 6, 3], 
                  'sew_resnet101': [3, 4, 23, 3], 
                  'sew_resnet152': [3, 8, 36, 3],}
    
    if 'spiking' in model_name or 'sew' in model_name:
        act = 'sn'
    else:
        act = 'an'
    
    layers = ['Conv_0', 'BN_0', f'{act}_0', 'MaxPool_0']
    
    if 'sew' in model_name:
        single_bottleneck_features = ['Conv_1', 'BN_1', f'{act}_1', 'Conv_2', 'BN_2', f'{act}_2', 'Conv_3', 'BN_3', f'{act}_3', 'Residual']
        single_basicblock_features = ['Conv_1', 'BN_1', f'{act}_1', 'Conv_2', 'BN_2', f'{act}_2', 'Residual']
    else:
        single_bottleneck_features = ['Conv_1', 'BN_1', f'{act}_1', 'Conv_2', 'BN_2', f'{act}_2', 'Conv_3', 'BN_3', 'Residual', f'{act}_3']
        single_basicblock_features = ['Conv_1', 'BN_1', f'{act}_1', 'Conv_2', 'BN_2', 'Residual', f'{act}_2']
     
    # --- dummy img
    if 'spiking' in model_name or 'sew' in model_name:
        x = torch.randn(T, 1, *feature_shape)     # dummy img
    else:
        x = torch.randn(1, *feature_shape)
    
    # --- dummy model
    if model_name in models_.spiking_resnet.__all__:
        model = models_.spiking_resnet.__dict__[model_name](spiking_neuron=neuron.IFNode, num_classes=num_classes)
        functional.set_step_mode(model, step_mode='m')
    
    elif model_name in models_.sew_resnet.__all__:
        model = models_.sew_resnet.__dict__[model_name](spiking_neuron=neuron.IFNode, num_classes=num_classes, detach_reset=True, cnf='ADD')
        functional.set_step_mode(model, step_mode='m')
    
    else:
        model = models_.resnet.__dict__[model_name](num_classes=num_classes)   # no this resnet
    
    # --- dummy forward
    if '18' in model_name or '34' in model_name:
        target_blocks = single_basicblock_features
    else:
        target_blocks = single_bottleneck_features
    
    for idx, i in enumerate(model_dict[model_name]):   # each layer
        for j in range(i):          # each block
            layers += [f'L{idx+1}_B{j+1}_{_}' for _ in target_blocks]
            
    layers += ['AvgPool', 'fc']     # manually change to fit different model
    
    features = model(x)
    
    assert isinstance(features, list), 'features is not list'
    
    units = []
    shapes = []
    for idx, layer in enumerate(features):
        units.append(layer.mean(0).numel())
        shapes.append(layer.mean(0).squeeze(0).detach().cpu().numpy().shape)
    
    functional.reset_net(model)
    
    return layers, units, shapes


# ----------------------------------------------------------------------------------------------------------------------
# -----
def rename():
    
    root = '/home/acxyle-workstation/Downloads/Face Identity Resnet50_CelebA2622_fold_0/Features'
    
    files = os.listdir(root)
    
    for file in files:
        
        if 'conv01' in file:
            
            new_name = '_'.join(file.split('_')[:-1]) + '_conv_1.pkl'
            
            os.rename(os.path.join(root, file), os.path.join(root, new_name))
        
        elif 'bn01' in file:
            
            new_name = '_'.join(file.split('_')[:-1]) + '_bn_1.pkl'
            
            os.rename(os.path.join(root, file), os.path.join(root, new_name))
        
        elif 'act01' in file:
            
            new_name = '_'.join(file.split('_')[:-1]) + '_neuron_1.pkl'
            
            os.rename(os.path.join(root, file), os.path.join(root, new_name))
        
        elif 'conv02' in file:
            
            new_name = '_'.join(file.split('_')[:-1]) + '_conv_2.pkl'
            
            os.rename(os.path.join(root, file), os.path.join(root, new_name))
        
        elif 'bn02' in file:
            
            new_name = '_'.join(file.split('_')[:-1]) + '_bn_2.pkl'
            
            os.rename(os.path.join(root, file), os.path.join(root, new_name))
        
        elif 'act02' in file:
            
            new_name = '_'.join(file.split('_')[:-1]) + '_neuron_2.pkl'
            
            os.rename(os.path.join(root, file), os.path.join(root, new_name))
            
        elif 'conv03' in file:
            
            new_name = '_'.join(file.split('_')[:-1]) + '_conv_3.pkl'
            
            os.rename(os.path.join(root, file), os.path.join(root, new_name))
        
        elif 'bn03' in file:
            
            new_name = '_'.join(file.split('_')[:-1]) + '_bn_3.pkl'
            
            os.rename(os.path.join(root, file), os.path.join(root, new_name))
            
        elif 'act03' in file:
            
            new_name = '_'.join(file.split('_')[:-1]) + '_neuron_3.pkl'
            
            os.rename(os.path.join(root, file), os.path.join(root, new_name))


def params_affine_from_spikingjelly04(params, verbose=True):
    print('[Codinfo] Executing params_affine_from_spikingjelly04...')
    if verbose:
        print('[Codinfo] experiment details:', params['args'])
        print('[Codinfo] params.keys():', params.keys())
        print('[Codinfo] best_val_acc1: {:.3f}, best_val_acc5: {:.3f}'.format(params['max_test_acc1'], params['test_acc5_at_max_test_acc1']))
        
    # 04版本中 spiking_resnet: conv[i].module.0 表示某个 block 内的第 i 个 conv, conv[i].module.1 则表示对应的 bn, downsample 模块没有 module
    # 这个现象来自于 SeqToANNContainer 的封装，使用了 Sequential 后会添加 module
    params_replace = {}     
    for layer in params['model']:
        temp = layer
        if 'conv1.module.0' in layer:
            temp = layer.split('conv1.module.0')
            temp.insert(1, 'conv1')
        if 'conv1.module.1' in layer:
            temp = layer.split('conv1.module.1')
            temp.insert(1, 'bn1')
        if 'conv2.module.0' in layer:
            temp = layer.split('conv2.module.0')
            temp.insert(1, 'conv2')
        if 'conv2.module.1' in layer:
            temp = layer.split('conv2.module.1')
            temp.insert(1, 'bn2')
        if 'conv3.module.0' in layer:
            temp = layer.split('conv3.module.0')
            temp.insert(1, 'conv3')
        if 'conv3.module.1' in layer:
            temp = layer.split('conv3.module.1')
            temp.insert(1, 'bn3')
        if 'downsample.0.module' in layer:     # for sew_resnet
            temp = layer.split('downsample.0.module')
            temp.insert(1, 'downsample')
        if 'downsample.module' in layer:     # for spiking_resnet
            temp = layer.split('downsample.module')
            temp.insert(1, 'downsample')
        temp = ''.join(temp)
        params_replace.update({temp:params['model'][layer]})

    print('[Codinfo] Params affine done')
    
    return params_replace
    

# [warning] not in sue for current code
def get_picture(pic_name):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32)
    transform = transforms.ToTensor()
    return transform(img)

def make_dir(path):
  if os.path.exists(path) is False:
    os.makedirs(path)
    
    
def cal_acc1_acc5(output, target):
    # define how to calculate acc1 and acc5
    acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
    return acc1, acc5
    

# FIXME --- add k-fold entrance
def SVM_classification(matrix, label, test_size=0.2, random_state=42):
    """
        1. default kernel='rbf', non-linear. 
        2. train_test_split() splits train data and test data randomly, not like n_fold experiments
        
        "It should be noticed the performance of svm.SVC() can be highly sensitive 
        to the choice of kernel, used what kernel depends on the data and the 
        research problem, sounds like can use GridSearchCV from sklearn.model_selection
        to perform a grid search for best combination for a model, but perhaps time
        consuming and computation intensive."
    """
    
    matrix_train, matrix_test, label_train, label_test = train_test_split(matrix, label, test_size=test_size, random_state=random_state)

    clf = svm.SVC()     # .SVC() .LinearSVC() .NuSVC() ... 
    
    if matrix_train.shape[1] == 0:
      acc = 0.
    else:
      clf.fit(matrix_train, label_train)
      
      predicted = clf.predict(matrix_test)

      acc = accuracy_score(label_test, predicted)*100
      
    return acc


def makeLabels(num_samples, num_classes):  # generate a label list
    label = []
    for i in range(num_classes):
        label += [i + 1] * num_samples
    return label


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def describe_model(layers, neurons, shapes):

    layers_info = [list(pair) for pair in zip(list(np.arange(len(layers)+1)), layers, neurons, shapes)]
    max_widths = [max(len(str(row[i])) for row in layers_info)+2 for i in range(len(layers_info[0]))]
    
    print("|".join("{:<{}}".format(header, max_widths[i]) for i, header in enumerate(["No.", "layer", "neurons", "shapes"])))
    print("-" * sum(max_widths))
    
    for row in layers_info:
        print("|".join("{:<{}}".format(str(item), max_widths[i]) for i, item in enumerate(row)))
        
    print("-" * sum(max_widths))


# ----------------------------------------------------------------------------------------------------------------------



# FIXME --- Vanilla SNN, 意义不明
def generate_resnet_layers_list_snnrat(model, model_name):     # return layers and neurons
    
    layers = ['cb_0', 'neuron_0']
    single_bottleneck_features = ['conv01', 'bn01', 'neuron01', 'conv02', 'bn02', 'neuron02', 'conv03', 'bn03', 'add_residual', 'neuron03']
    single_basicblock_features = ['conv01', 'bn01', 'neuron01', 'conv02', 'bn02', 'add_residual', 'neuron02']
    x = torch.randn(1, 3, 32, 32)
    
    if '50' in model_name:
        blocks = [3, 4, 6, 3]
        layers_ = layers
        for idx, i in enumerate(blocks):   # each layer
            for j in range(i):          # each block
                temp = []
                for each_layer in single_bottleneck_features:
                    temp.append('L'+str(idx+1)+'_B'+str(j+1)+'_'+each_layer)
                layers_ += temp
        layers_ += ['avgpool', 'fc']
        
        features = model(x)
        neurons_ = []
        for idx, layer in enumerate(features):
            neurons_.append(layer.mean(0).numel())
    
    if '18' in model_name:
        blocks = [2, 2, 2, 2]
        layers_ = layers
        for idx, i in enumerate(blocks):   # each layer
            for j in range(i):          # each block
                temp = []
                for each_layer in single_basicblock_features:
                    temp.append('L'+str(idx+1)+'_B'+str(j+1)+'_'+each_layer)
                layers_ += temp
        layers_ += ['avgpool', 'fc']
        
        features = model(x)
        neurons_ = []
        for idx, layer in enumerate(features):
            neurons_.append(layer.mean(0).numel())
        
    return layers_, neurons_

    
