#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:10:50 2024

@author: acxyle-workstation

    hook based feature extraction

"""

import os
import sys
import random

import time
import datetime

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import time
import numpy as np
import pickle
import argparse

from spikingjelly.activation_based import surrogate, neuron, functional, layer
from spikingjelly.activation_based.model.tv_ref_classify import presets, utils

from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import autoaugment, transforms


try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url
    
sys.path.append('../')
import models_
from analysis import utils_


# ======================================================================================================================
root =  "/home/acxyle-workstation/Downloads/FSA"


# ----------------------------------------------------------------------------------------------------------------------
def get_args_parser(_neuron, _surrogate, T=4, fold_idx=0):
    
    parser = argparse.ArgumentParser(description="Featuremap Extractor")

    parser.add_argument("--device", type=str, default='cuda:0')

    parser.add_argument("--model", type=str, default='vgg16_bn', help='Model name only')     # [notice] this will triger different branches
    
    # --- SNN model config
    parser.add_argument("--step_mode", type=str, default='m')
    parser.add_argument('--neuron', type=str, default=f'{_neuron}')
    parser.add_argument('--surrogate', type=str, default=f'{_surrogate}')
    parser.add_argument("--T", type=int, default=T)

    parser.add_argument("--data_path", type=str, default='/home/acxyle-workstation/Dataset/CelebA50') # collect data from celebA-50

    #FIXME - make it auto generate
    parser.add_argument("--FSA_dir", type=str, default='VGG/VGG/FSA VGG16bn_C2k_fold_')
    #parser.add_argument("--params_dir", type=str, default='logs_ft_C50_VGG16bn_C2k_fold_0')
    parser.add_argument("--num_folds", type=int, default=5)
    

    parser.add_argument("--num_classes", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=10)
    
    parser.add_argument("--batch_size", type=int, default=1)

    return parser.parse_args()


# ----------------------------------------------------------------------------------------------------------------------
# building ...
#DONE --- identical results with below operation
class CNNStaticExtraction():
    
    def __init__(self, args, **kwargs):
        
        self.model = eval(f"torchvision.models.{args.model}(num_classes=args.num_classes)")
        self.model.to(args.device)
        
        if 'fold' in args.FSA_dir:
            
            FSA_folder = args.FSA_dir.split('/')[-1]
            
            for fold_idx in range(args.num_folds):
                
                FSA_dir = os.path.join(root, args.FSA_dir, f'-_Single Models/{FSA_folder}{fold_idx}')
                params_path = os.path.join(FSA_dir, f"logs_ft_C50_{FSA_folder.split(' ')[-1]}{fold_idx}/checkpoint_max_test_acc1.pth")
                
                # assme the model is trained from spikingjelly script, otherwise remove <['model']>
                params = torch.load(params_path, map_location=args.device)['model']
                self.model.load_state_dict(params)
                
                # -----
                self.layer_extraction(args)
                # -----
                
                
    # same data transform with spikingjelly script, different preprocessing may leads to different outcome

    def hook_fn(self, module, inputs, outputs):
        
        self.feature_map.append(outputs.detach().cpu().reshape(args.batch_size, -1))
    

    def layer_extraction(self, args):
        
        set_deterministic()
        
        self.model.eval()
        
        with torch.inference_mode():
            
            self.feature_map = []
            
            for _name, _layer in self.model.named_modules():
                if isinstance(_layer, torch.nn.ReLU):
                    _layer.register_forward_hook(self.hook_fn)
            
            # -----
            transform = transforms.Compose(
                    [
                    transforms.Resize(232, interpolation=InterpolationMode('bilinear')),
                    transforms.CenterCrop(224),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )
            
            dataset = torchvision.datasets.ImageFolder(args.data_path, transform=transform)
            
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=False, worker_init_fn=seed_worker)
            
            # ---
            features = []
            
            for idx, (imgs, labels) in tqdm(enumerate(dataloader)):
                
                imgs = imgs.to(args.device, non_blocking=True)
                
                _batch_size = len(imgs)
                
                out = self.model(imgs)
                
                features.append(self.feature_map)
                
                self.feature_map = []
                
            features = [torch.vstack([features[idx][_] for idx in range(len(features))]).numpy() for _ in range(15)]
            
            print('6')
                

  
# ----------------------------------------------------------------------------------------------------------------------
def neuron_selection(args):     # [warning] most of the neuron name should be xxNode           
    neuron_ = neuron.__dict__[f'{args.neuron}Node']
    return neuron_


def surrogate_selection(args):
    surrogate_function_ = surrogate.__dict__[args.surrogate]()
    return surrogate_function_


def set_deterministic(_seed_=2020):
    random.seed(_seed_)
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG (Random Number Generator) for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = Trueadd_help=True
    torch.backends.cudnn.benchmark = False
    
    """
        "A handful of CUDA operations are nondeterministic if the CUDA version is 10.2 or greater, unless the environment
        variable CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8 is set. See the CUDA documentation for
        more details: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility"
    """
    
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    
    worker_seed = torch.initial_seed() % int(np.power(2, 32))

    np.random.seed(worker_seed)
    random.seed(worker_seed)
        

# ======================================================================================================================
if __name__ =="__main__":
    
    
    args = get_args_parser('IF', 'ATan', T=6, fold_idx=0)
    
    CNNStaticExtraction(args)
    
