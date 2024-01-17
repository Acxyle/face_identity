#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 00:12:41 2023

generate featuremap of SpikingVGG

[Notice] This Version is valid for --- Lab Desktop --- with default settings

#TODO

    upgrade, make a basic class to store all operation

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

from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model.tv_ref_classify import presets, utils
from torchvision.transforms.functional import InterpolationMode

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url
    
sys.path.append('../')
import models_
from analysis import utils_


root = "/home/acxyle-workstation/Downloads/"


def get_args_parser(fold_idx):
    
    parser = argparse.ArgumentParser(description="Spikingjelly Resnet Featuremap Extractor", add_help=True)

    parser.add_argument("--return_T_dim", dest = "return_T_dim", help="return the output value before merging into firing rate", action="store_true")

    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--model", type=str, default='resnet50')     # [notice] this will triger different branches

    parser.add_argument('--neuron', type=str, default='IF')
    parser.add_argument("--step_mode", type=str, default='m')
    parser.add_argument("--T", type=int, default=4)

    parser.add_argument("--data_path", type=str, default='/home/acxyle-workstation/Dataset/CelebA50/') # collect data from celebA-50

    #FIXME - make it auto generate
    parser.add_argument("--output_dir", type=str, default=f'Face Identity Resnet50_CelebA2622_fold_{fold_idx}')
    parser.add_argument("--weight_path", type=str, default=f"logs_ft_CelebA50_Resnet50_CelebA2622_fold_{fold_idx}")

    parser.add_argument("--mode", type=str, default='feature')
    parser.add_argument("--num_classes", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)

    return parser.parse_args()


def set_deterministic(_seed_: int=2020, disable_torch_deterministic=False):
    random.seed(_seed_)
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG (Random Number Generator) for all devices (both CPU and CUDA)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if disable_torch_deterministic:
        pass
    else:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    
    worker_seed = torch.initial_seed() % int(np.power(2, 32))

    np.random.seed(worker_seed)
    random.seed(worker_seed)
        

def _print(content, message_type='[Codinfo]', symbol='-', padding=2):
    """

    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row_data = [current_time, message_type, content]
    formatted_row = "|".join("{:<{}}".format(' '*padding + item, len(item)+padding*2) for item in row_data)
    border = symbol * (len(formatted_row) + 2)  

    print(border)
    print(formatted_row)

    
# ----------------------------------------------------------------------------------------------------------------------
def create_model_and_load_weight(args):  # -> return model with loaded weight

    # for spikingjelly models
    if args.neuron == 'IF':
        neuron_ = neuron.IFNode
    elif args.neuron == 'LIF':
        neuron_ = neuron.LIFNode

    if 'spiking_resnet' in args.model:
        model = models_.spiking_resnet.__dict__[args.model](spiking_neuron=neuron_, num_classes=args.num_classes, surrogate_function=surrogate.ATan(), detach_reset=True, mode=args.mode)
    elif 'sew_resnet' in args.model:
        model = models_.sew_resnet.__dict__[args.model](spiking_neuron=neuron_, num_classes=args.num_classes, surrogate_function=surrogate.ATan(), detach_reset=True, cnf='ADD', mode=args.mode)
    elif 'spiking_vgg' in args.model:
        model = models_.spiking_vgg.__dict__[args.model](spiking_neuron=neuron_, num_classes=args.num_classes, surrogate_function=surrogate.ATan(), detach_reset=True)
    
    # for ANN
    if args.model in models_.vgg.__all__: 
        model = models_.vgg.__dict__[args.model](num_classes=args.num_classes)
    if args.model in models_.resnet.__all__: 
        model = models_.resnet.__dict__[args.model](num_classes=args.num_classes)

    functional.set_step_mode(model, step_mode=args.step_mode)
    #print(model)
    _print('Model created')

    pth_path = os.path.join(root, args.output_dir, args.weight_path, 'checkpoint_max_test_acc1.pth')
    params = torch.load(pth_path, map_location=torch.device(args.device))

    #count = 0
    #for name, param in model.named_parameters():
    #    print(name, '  |  ', [i for i in list(params['model'].keys()) if 'running' not in i and 'tracked' not in i][count])
    #    count +=1
    
    # for spikingjelly model
    if 'spiking' in args.model.lower() or 'sew' in args.model.lower() and 'resnet' in args.model.lower():
        params_replace = utils_.params_affine_from_spikingjelly04(params, verbose=False)     # [warning] open verbose only for validate spikingjelly0.4 .pth
        model.load_state_dict(params_replace)
    elif 'spiking' in args.model.lower() and 'vgg' in args.model.lower(): 
        model.load_state_dict(params['model'])
    
    # for ANN
    if 'resnet' in args.model or 'vgg' in args.model:
        model.load_state_dict(params['model'])     # for spikingjelly-trained model
        #model.load_state_dict(params)     # for baseline model
    
    _print('Params loaded')
    
    return model


def generate_layers_and_neurons(model, args):  # -> return layers_list and neurons_list
    
    _print('Generating layers_list and neurons_list')

    if 'resnet' in args.model and 'spiking' in args.model or 'sew' in args.model:
        layers, neurons, shapes = utils_.generate_resnet_layers_list(model, args.model)  # [notice] the second model is model name
        return layers, neurons, shapes
    
    if 'vgg' in args.model and 'spiking' in args.model:     # [warning] this is required to be modified
        layers, neurons, shapes = utils_.generate_vgg_layers(model, args.model)
        return layers, neurons, shapes
    
    # [notice] need to rewrite this part for a better entrance
    if 'resnet' in args.model:
        layers, neurons, shapes = utils_.generate_resnet_layers_list_ann(model, args.model)
        return layers, neurons, shapes
    
    if 'vgg' in args.model:
        layers, neurons, shapes = utils_.generate_vgg_layers_list_ann(model, args.model)
        return layers, neurons, shapes
        
    _print('layers_list and neurons_list obtained')


def process_one_image(T, model, criterion, image, device, target):
    
    image = image.to(device, non_blocking=True)
    image = image.unsqueeze(0).repeat(T, 1, 1, 1, 1)    
    target = target.to(device, non_blocking=True)
    
    feature_list = model(image)
    
    output=feature_list[-1].mean(0)
       
    loss = criterion(output, target)
    acc1, acc5 = utils_.cal_acc1_acc5(output, target)
    
    return feature_list, output, loss, acc1, acc5


def spikingjelly_evaluate(model, layers, layers_, neurons, args, verbose=False):
    """
        if try to save all the feature map in one iteration, the expected RAM is 314.58 GB, hence use the multi-iteration, 
        only save one layer in one forward process
    """
    
    # ----- init
    set_deterministic()
    
    save_root = os.path.join(root, args.output_dir, 'Features')
    utils_.make_dir(save_root)
    
    batch_size = args.batch_size
    valdir = args.data_path
    
    preprocessing = presets.ClassificationPresetEval(crop_size=224, resize_size=232, interpolation=InterpolationMode('bilinear'))
    
    dataset_test = torchvision.datasets.ImageFolder(valdir, preprocessing)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=12, pin_memory=False, worker_init_fn=seed_worker)
    
    model.to(args.device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    _print('Please set verbose=True if running check is necessary, generating feature map...')

    with torch.inference_mode(): 
        with torch.no_grad():
            for idx, layer_ in enumerate(layers_):     # for each layer
                
                idx = layers.index(layer_)
                
                feature_matrix = []     # (num_layers, C, W, H)
                
                if verbose:
                    print('idx: ', idx, 'layer: ', layer_)
                    
                count = 0
                
                metric_logger = utils.MetricLogger(delimiter="  ")
                header = "Test: {}".format(layer_)

                num_processed_samples = 0
                start_time = time.time()
                
                for image, target in tqdm(metric_logger.log_every(data_loader_test, -1, header), desc=f'{layer_}'):     # for each batch (1 image per batch)
                        
                    feature_list, output, loss, acc1, acc5 = process_one_image(args.T, model, criterion, image, args.device, target)
                    functional.reset_net(model)
            
                    batch_size = target.shape[0]
                        
                    metric_logger.update(loss=loss.item())
                    metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                    metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
                    num_processed_samples += batch_size

                    feature_strip = feature_list[idx].mean(0).squeeze(0).flatten().detach()
                    
                    if feature_strip.shape[0] != neurons[idx]:
                        raise RuntimeError('[Coderror] generated feature_strip ({}) is not exactly the same with neurons ({})'.format(feature_strip.shape[0], neurons[idx]))
    
                    feature_matrix.append(feature_strip)
                    
                    if verbose:
                        print(' | layer: ', idx, layer_, ' | feature_strip:', feature_strip.shape[0], 
                              ' | label: {}, prediction: {}'.format(target.data.detach().numpy()[0], np.argmax(output.squeeze(0).detach().cpu().numpy())),  
                              ' | image count: ', count, ' | feature_matrix: ({}, {})'.format(len(feature_matrix), feature_strip.shape[0]))
                    count += 1
                                    
                num_processed_samples = utils.reduce_across_processes(num_processed_samples)
                metric_logger.synchronize_between_processes()
                test_loss, test_acc1, test_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
                if verbose:
                    print(f'Test: test_acc1={test_acc1:.3f}, test_acc5={test_acc5:.3f}, test_loss={test_loss:.6f}, samples/s={num_processed_samples / (time.time() - start_time):.3f}')
                    
                feature_matrix = torch.stack(feature_matrix, dim=0).detach().cpu().numpy()
                 
                utils_.dump(feature_matrix, os.path.join(save_root, layer_+'.pkl'))
    
    
def ANN_evaluate(model, layers, layers_, neurons, args, verbose=True):
    """
        This function generates the feature map of ANN. This function use 
    """
    set_deterministic()
    
    save_root = os.path.join(root, args.output_dir, 'Features')
    utils_.make_dir(save_root)
    
    batch_size = args.batch_size
    valdir = args.data_path

    preprocessing = presets.ClassificationPresetEval(crop_size=224, resize_size=232, interpolation=InterpolationMode('bilinear'))     
    dataset_test = torchvision.datasets.ImageFolder(valdir, preprocessing)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    data_loader_test = torch.utils.data.DataLoader(
                                                dataset_test, 
                                                batch_size=batch_size, 
                                                sampler=test_sampler, 
                                                num_workers=1, 
                                                pin_memory= False,
                                                worker_init_fn=seed_worker
                                                )
    
    model.to(args.device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    _print('Please set verbose=True if running check is necessary, generating feature map...')
    
    with torch.inference_mode(): 
        with torch.no_grad():
            for idx, layer_ in enumerate(layers_):     # for each layer

                feature_matrix = []
                
                idx = layers.index(layer_)
                
                if verbose:
                    print('idx: ', idx, 'layer: ', layer_)
                    
                count = 0
                
                metric_logger = utils.MetricLogger(delimiter="  ")
                header = "Test: {}".format(layer_)

                num_processed_samples = 0
                start_time = time.time()
                
                for image, target in tqdm(metric_logger.log_every(data_loader_test, -1, header), desc=f'{layer_}'):     # for each image
                        
                    image = image.to(args.device, non_blocking=True)   
                    target = target.to(args.device, non_blocking=True)
                    
                    feature_list = model(image)

                    output=feature_list[-1]

                    loss = criterion(output, target)
                    acc1, acc5 = utils_.cal_acc1_acc5(output, target)
            
                    batch_size = target.shape[0]
                        
                    metric_logger.update(loss=loss.item())
                    metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                    metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
                    num_processed_samples += batch_size

                    feature_strip = feature_list[idx].squeeze(0).flatten()
                    
                    if feature_strip.shape[0] != neurons[idx]:
                        raise RuntimeError('[Coderror] generated feature_strip ({}) is not exactly the same with neurons ({})'.format(feature_strip.shape[0], neurons[idx]))
    
                    feature_matrix.append(feature_strip)
                    
                    if verbose:
                        print(' | layer: ', idx, layer_, ' | feature_strip:', feature_strip.shape[0], 
                              ' | label: {}, prediction: {}'.format(target.data.detach().cpu().numpy()[0], np.argmax(output.squeeze(0).detach().cpu().numpy())),  
                              ' | image count: ', count, ' | feature_matrix: ({}, {})'.format(len(feature_matrix), feature_strip.shape[0]))
                    count += 1
                                    
                num_processed_samples = utils.reduce_across_processes(num_processed_samples)
                metric_logger.synchronize_between_processes()
                test_loss, test_acc1, test_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
                
                if verbose:
                    print(f'Test: test_acc1={test_acc1:.3f}, test_acc5={test_acc5:.3f}, test_loss={test_loss:.6f}, samples/s={num_processed_samples / (time.time() - start_time):.3f}')
                
                feature_matrix = torch.stack(feature_matrix, dim=0)
                feature_matrix = feature_matrix.detach().cpu().numpy()
         
                utils_.dump(feature_matrix, os.path.join(save_root, layer_+'.pkl'), verbose=False)

if __name__ =="__main__":
    
    for fold_idx in [3,4]:
    
        args = get_args_parser(fold_idx)
        
        print(args)
        
        # 1. create model and load weights
        model = create_model_and_load_weight(args)
        
        # 2. obtain layers_list and neurons_list
        layers, neurons, _ = generate_layers_and_neurons(model, args)     # [notice] neurons may be removed in future because of low generalization and additional workload
        
        # 3. feed model and layers&neurons into the evaluator
        if 'spiking' in args.model.lower():
            spikingjelly_evaluate(model, layers, layers, neurons, args, verbose=True)
        else:
            ANN_evaluate(model, layers, layers, neurons, args, verbose=False)


    

