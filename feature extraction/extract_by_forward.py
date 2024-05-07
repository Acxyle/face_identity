#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 00:12:41 2023
    
    Function:
        (1) model validation
        (2) generate featuremap
    
    This script extracts the feature maps by modifying the forward process of Pytorch. Auto-generate layer names.
    
    pros: automatically scan all layers and assign names
    cons: can not manually select layers, like hook based method


"""

import os
import sys
import random

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

import numpy as np

import argparse

from spikingjelly.activation_based import surrogate, neuron, functional, layer
from spikingjelly.activation_based.model.tv_ref_classify import utils     # inherit the validation

from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import autoaugment, transforms

# ---
sys.path.append('../')
import models_
from analysis import utils_


# ======================================================================================================================
def set_deterministic(_seed_=2020):
    """
        "A handful of CUDA operations are nondeterministic if the CUDA version is 10.2 or greater, unless the environment
        variable 'CUBLAS_WORKSPACE_CONFIG=:4096:8' or 'CUBLAS_WORKSPACE_CONFIG=:16:8' is set. See the CUDA documentation 
        for more details: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility"
    """
    
    random.seed(_seed_)
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG (Random Number Generator) for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'      # ←
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    
    worker_seed = torch.initial_seed() % int(np.power(2, 32))     # this needs to be changed for Win OS due to int limit

    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def get_args_parser():
    
    parser = argparse.ArgumentParser(description="Featuremap Extractor")
    
    # --- common config
    parser.add_argument("--data_path", type=str, default='/home/acxyle-workstation/Dataset/CelebA50') # collect data from celebA-50

    parser.add_argument("--FSA_dir", type=str, default='Resnet/SEWResnet')
    parser.add_argument("--FSA_config", type=str, default='SEWResnet152_IF_ATan_T4_C2k_fold_')
    parser.add_argument("--FSA_params_dir", type=str, default=None)
    parser.add_argument("--fold_idx", type=int, default=4)

    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--workers", default=12, help="data loading workers")
    parser.add_argument("--num_classes", type=int, default=2622)
    parser.add_argument("--batch_size", type=int, default=1)     # different batch_size leads to bias and must be divisible by 500
    
    sub_parser = parser.add_subparsers(dest="command", help='Sub-command help')
    
    # --- ANN config
    parser_ANN = sub_parser.add_parser('ANN', help='Train the model')
    parser_ANN.add_argument("--model", type=str, default='resnet50', help='Model name only')
    
    # --- SNN config
    parser_SNN = sub_parser.add_parser('SNN', help='Test the model')
    parser_SNN.add_argument("--model", type=str, default='sew_resnet152', help='Model name only')
    parser_SNN.add_argument("--step_mode", type=str, default='m')
    parser_SNN.add_argument('--neuron', type=str, default='IF')
    parser_SNN.add_argument('--surrogate', type=str, default='ATan')
    parser_SNN.add_argument("--T", type=int, default=4)

    return parser.parse_args()


# ----------------------------------------------------------------------------------------------------------------------
class feature_extractor_base():
    
    def __init__(self, args, **kwargs):
        
        set_deterministic()
        
        # 1. 
        self.model = self.create_model_and_load_weight(args)
        self.model.to(args.device)
        self.model.eval()
        
        self.save_root = os.path.join(self.FSA_folder, 'Features')
        utils_.make_dir(self.save_root)
        
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
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False, worker_init_fn=seed_worker)
        
        # ---
        self.criterion = nn.CrossEntropyLoss()     # for validation
        
        
    def __len__(self, ):
        ...
        
    def __getitem__(self, ):
        ...
        
    def __repr__(self, ):
        ...
        
    def create_model_and_load_weight(self, args):  # -> return model with loaded weight
        
        # ----- 1. create model
        model = self.create_model(args)

        # ----- 2. load parameters
        model = self.load_weight(model, args)
        
        # ----- 3. replace the final layer from finetune, if applicable
        model = self.replace_classification_head(model, args)
        
        return model
    
    
    def create_model(self, args, **kwargs):
        """
            ...
        """
        
        if args.command == 'ANN':
        
            if args.model in models_.vgg.__all__: 
                model_base =  models_.vgg
            elif args.model in models_.resnet.__all__: 
                model_base =  models_.resnet
            
            model = model_base.__dict__[args.model](num_classes=args.num_classes)
        
        elif args.command == 'SNN':
            
            common_opts = {
                'spiking_neuron': neuron.__dict__[f'{args.neuron}Node'],
                'surrogate_function': surrogate.__dict__[args.surrogate](),
                'num_classes': args.num_classes,
                'detach_reset': True
                }
            
            if 'spiking_resnet' in args.model:
                model_base = models_.spiking_resnet
                common_opts['zero_init_residual'] = True
            elif 'sew_resnet' in args.model:
                model_base = models_.sew_resnet
                common_opts['cnf'] = 'ADD'
            elif 'spiking_vgg' in args.model:
                model_base = models_.spiking_vgg
            
            model = model_base.__dict__[args.model](**common_opts)
            functional.set_step_mode(model, step_mode=args.step_mode)
        
        utils_.formatted_print('Model Created')
        
        return model


    def load_weight(self, model, args, **kwargs):
        
        config = f'{args.FSA_config}{args.fold_idx}'
        
        self.FSA_folder = os.path.join(FSA_root, f'{args.FSA_dir}/FSA {args.FSA_config}/-_Single Models/FSA {config}')
        
        if args.FSA_params_dir is None:
            pth_path = os.path.join(self.FSA_folder, f'logs_{config}/checkpoint_max_test_acc1.pth')     # manually modify
        else:
            pth_path = os.path.join(self.FSA_folder, f'{args.FSA_params_dir}{args.fold_idx}/checkpoint_max_test_acc1.pth')

        model.load_state_dict(torch.load(pth_path, map_location=torch.device(args.device))['model'])     # for spikingjelly model
           
        utils_.formatted_print('Params loaded')
        
        return model
        

    def replace_classification_head(self, model, args, **kwargs):
        
        if args.command == 'ANN':
            base_module = nn.Linear
        elif args.command == 'SNN':
            base_module = layer.Linear  
        else:
            raise ValueError
        
        if 'vgg' in args.model:
            model.classifier[-1] = base_module(1024, args.num_classes) if '5' in args.model else base_module(4096, args.num_classes)     # rewrite this layer will enable the requires_grad()
        elif 'resnet' in args.model:
            model.fc = base_module(512, args.num_classes) if ('18' in args.model or '34' in args.model) else base_module(2048, args.num_classes)

        utils_.formatted_print(f"Classification head replaced to '{args.num_classes}'")
        
        return model
    
    
    def extractor(self, model_name, target_type='act', device='cuda', **kwargs):
        
        # ---
        l_idx, layers, neurons, shapes = utils_.get_layers_and_units(model_name, target_type)
        utils_.describe_model(layers, neurons, shapes, l_idx)
        
        with torch.inference_mode(): 
            
            feature_maps = []
             
            for idx, (image, target) in tqdm(enumerate(self.dataloader), 'Extracting Feature'):     # for each image
                
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                feature = self.process_single_img(image, **kwargs)
                output = feature[-1]
                
                b_loss = self.criterion(output, target)
                b_acc1, _ = utils_.cal_acc1_acc5(output, target)
                
                feature = [feature[_].data.detach().flatten(start_dim=1).cpu() for _ in l_idx]
                
                feature_maps.append(feature)

                #print(' | layer: ', layer_info[0], layer_info[1], ' | feature_strip:', feature_strip.shape[0], 
                #      ' | label: {}, prediction: {}'.format(target.data.detach().cpu().numpy()[0], np.argmax(output.squeeze(0).detach().cpu().numpy())),  
                #      ' | image count: ', idx)    
            
            feature_maps = [torch.stack([feature_maps[i_idx][l_idx] for i_idx in range(idx+1)], dim=0).reshape(500, -1).numpy() for l_idx in range(len(layers))]
            
            utils_.formatted_print('Feature map generated')
            
            for idx, _layer in tqdm(enumerate(layers), 'Saving Feature', total=len(layers)):
                
                feature_map = feature_maps[idx]
                
                assert feature_map.shape[1] == neurons[idx], 'Feature size check failed'

                utils_.dump(feature_map, os.path.join(self.save_root, f'{_layer}.pkl'), verbose=False)
                
            del feature_maps, feature_map



# ----------------------------------------------------------------------------------------------------------------------
class feature_extractor_ANN(feature_extractor_base):
    
    def __init__(self, args, **kwargs):
        
        super().__init__(args, **kwargs)
    
    
    def __call__(self, args, **kwargs):
        
        self.extractor(args.model, device=args.device, **kwargs)
    
    
    def process_single_img(self, image, **kwargs):
          
        return self.model(image)
    
    
class feature_extractor_SNN(feature_extractor_base):
    
    def __init__(self, args, **kwargs):
        
        super().__init__(args, **kwargs)
    
    
    def __call__(self, args, **kwargs):
        
        self.extractor(args.model, T=args.T, device=args.device, **kwargs)
    
    
    def process_single_img(self, image, T=4, **kwargs):
        """
            by default, the spike train is compressed as a single firing rate
        """
        
        image = image.unsqueeze(0).repeat(T, 1, 1, 1, 1)    
        feature = self.model(image)
        functional.reset_net(self.model)     # crucial
        
        return [torch.mean(_, axis=0) for _ in feature]
    

if __name__ =="__main__":
    
    FSA_root =  "/home/acxyle-workstation/Downloads/FSA"
    
    args = get_args_parser()
    
    print(args)
    
    # -----
    if args.command == 'ANN':
        feature_extractor = feature_extractor_ANN(args)
    elif args.command == 'SNN':
        feature_extractor = feature_extractor_SNN(args)
    
    feature_extractor(args)
                
        
