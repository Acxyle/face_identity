#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 00:12:41 2023

generate featuremap of SpikingVGG

[Notice] This Version is valid for --- Lab Desktop --- with default settings

#TODO
"""

import os
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import time
import numpy as np
import pickle
import argparse

from spikingjelly.activation_based import surrogate, neuron, functional
import spiking_resnet, sew_resnet, spiking_vgg
from spikingjelly.activation_based.model.tv_ref_classify import presets, utils
from torchvision.transforms.functional import InterpolationMode

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url
    
import utils_
import vgg, resnet
    
parser = argparse.ArgumentParser(description="Spikingjelly Resnet Featuremap Extractor", add_help=True)

parser.add_argument("--spikingjelly_evaluate", dest="spikingjelly_evaluate", help="execute model evaluate on CelebA50", action="store_true")
parser.add_argument("--obtain_feature_strip", dest="obtain_feature_strip", help="generate feature map for CelebA50 Dataset", action="store_true")
parser.add_argument("--return_T_dim", dest = "return_T_dim", help="return the output value before merging into firing rate", action="store_true")

parser.add_argument("--device", type=str, default='cpu')

parser.add_argument("--model", type=str, default='spiking_vgg16_bn')
parser.add_argument('--neuron', type=str, default='LIF')

parser.add_argument("--step_mode", type=str, default='m')
parser.add_argument("--T", type=int, default=4)

parser.add_argument("--data_path", type=str, default='/home/acxyle/Downloads/celeb50/') # for finetuned
#parser.add_argument("--data_path", type=str, default='/home/acxyle/Downloads/MNIST-10/') 

parser.add_argument("--weight_path", type=str, default="VGG/SNN/logs-ft-CelebA50-from-spikingvgg16bn-LIF_CelebA2622")
#parser.add_argument("--data_path", type=str, default='/media/acxyle/Data/ChromeDownload/CelebA/CelebA_2262/val') # for validation
parser.add_argument("--output_dir", type=str, default='/media/acxyle/Data/ChromeDownload/Identity_SpikingVGG16bn_LIF_CelebA2622_Results/')

parser.add_argument("--mode", type=str, default='feature')
parser.add_argument("--num_classes", type=int, default=50)
parser.add_argument("--num_samples", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=1)

args = parser.parse_args()
model_root = "/home/acxyle/Git/spikingjelly-master/spikingjelly/activation_based/model/"

def create_model_and_load_weight(args):  # -> return model with loaded weight

    # for spikingjelly models
    if args.neuron == 'IF':
        neuron_ = neuron.IFNode
    elif args.neuron == 'LIF':
        neuron_ = neuron.LIFNode

    if 'spiking_resnet' in args.model:
        model = spiking_resnet.__dict__[args.model](spiking_neuron=neuron_, num_classes=args.num_classes, surrogate_function=surrogate.ATan(), detach_reset=True, mode=args.mode)
    elif 'sew_resnet' in args.model:
        model = sew_resnet.__dict__[args.model](spiking_neuron=neuron_, num_classes=args.num_classes, surrogate_function=surrogate.ATan(), detach_reset=True, cnf='ADD', mode=args.mode)
    elif 'spiking_vgg' in args.model:
        model = spiking_vgg.__dict__[args.model](spiking_neuron=neuron_, num_classes=args.num_classes, surrogate_function=surrogate.ATan(), detach_reset=True)
    
    # for ANN
    if args.model in vgg.__all__: 
        model = vgg.__dict__[args.model](num_classes=args.num_classes)
    if args.model in resnet.__all__: 
        model = resnet.__dict__[args.model](num_classes=args.num_classes)

    functional.set_step_mode(model, step_mode=args.step_mode)
    #print(model)
    print('\n[Codinfo] Model created.')
    weight_path = args.weight_path
    pth_path = os.path.join(model_root, weight_path, 'checkpoint_max_test_acc1.pth')
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
    
    print('[Codinfo] Params loaded')
    
    return model

def generate_layers_and_neurons(model, args):  # -> return layers_list and neurons_list
    
    print('[Codinfo] Generating layers_list and neurons_list')

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
        
    print('[Codinfo] layers_list and neurons_list obtained')


def process_one_image(T, model, criterion, image, device, target):
    
    image = image.to(device, non_blocking=True)
    image = image.unsqueeze(0).repeat(T, 1, 1, 1, 1)    
    target = target.to(device, non_blocking=True)
    
    feature_list = model(image)
    
    output=feature_list[-1].mean(0)
       
    loss = criterion(output, target)
    acc1, acc5 = utils_.cal_acc1_acc5(output, target)
    
    return feature_list, output, loss, acc1, acc5

def spikingjelly_evaluate(model, layers, layers_, neurons, args, verbose=True):

    utils_.make_dir(args.output_dir)
    batch_size = args.batch_size
    valdir = args.data_path
    
    # [warning] write like this because may change default 'bilinear' to 'bicubic', etc.
    # [warning] center crop can be used for clean dataset which centered target object like face, but may not good for others
    preprocessing = presets.ClassificationPresetEval(crop_size=224, resize_size=232, interpolation=InterpolationMode('bilinear'))     
    
    dataset_test = torchvision.datasets.ImageFolder(valdir, preprocessing)
    #print(dataset_test.class_to_idx)
    #dataset_test.class_to_idx = {str(i):i for i in range(50)}  # [warning] no impact if TEST ONLY
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    data_loader_test = torch.utils.data.DataLoader(
                                                dataset_test, 
                                                batch_size=batch_size, 
                                                sampler=test_sampler, 
                                                num_workers=8, 
                                                pin_memory= False,
                                                worker_init_fn=utils_.seed_worker
                                                )
    
    model.to(args.device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print('[Codelp] Please set verbose=True if running check is necessary')
    print('[Codinfo] Start generating feature map...')
    
    with torch.inference_mode(): 
        with torch.no_grad():
            for idx, layer_ in tqdm(enumerate(layers_)):     # for each layer
                
                idx = layers.index(layer_)
                
                #feature_matrix = np.zeros(shape=(num_classes*num_samples, neurons[idx]))
                feature_matrix = []
                
                if verbose:
                    print('idx: ', idx, 'layer: ', layer_)
                    
                count = 0
                
                metric_logger = utils.MetricLogger(delimiter="  ")
                header = "Test: {}".format(layer_)

                num_processed_samples = 0
                start_time = time.time()
                
                for image, target in metric_logger.log_every(data_loader_test, -1, header):     # for each image
                        
                    feature_list, output, loss, acc1, acc5 = process_one_image(args.T, model, criterion, image, args.device, target)
                    functional.reset_net(model)
            
                    batch_size = target.shape[0]
                        
                    metric_logger.update(loss=loss.item())
                    metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                    metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
                    num_processed_samples += batch_size

                    feature_strip = feature_list[idx].mean(0).squeeze(0).flatten()
                    
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
                    
                feature_matrix = torch.stack(feature_matrix, dim=0)
                feature_matrix = feature_matrix.detach().cpu().numpy()
                
                if verbose:
                    print('feature_matrix: ', feature_matrix.shape)
                    print('saving feature map of layer [{}]'.format(layer_))
                with open(args.output_dir + '/' + layer_ + '.pkl',  'wb') as f:
                    pickle.dump(feature_matrix, f, protocol=-1)
    
def ANN_evaluate(model, layers, layers_, neurons, args, verbose=True):

    utils_.make_dir(args.output_dir)
    batch_size = args.batch_size
    valdir = args.data_path

    preprocessing = presets.ClassificationPresetEval(crop_size=224, resize_size=232, interpolation=InterpolationMode('bilinear'))     
    dataset_test = torchvision.datasets.ImageFolder(valdir, preprocessing)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    data_loader_test = torch.utils.data.DataLoader(
                                                dataset_test, 
                                                batch_size=batch_size, 
                                                sampler=test_sampler, 
                                                num_workers=8, 
                                                pin_memory= False,
                                                worker_init_fn=utils_.seed_worker
                                                )
    
    model.to(args.device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print('[Codelp] Please set verbose=True if running check is necessary')
    print('[Codinfo] Start generating feature map...')
    
    with torch.inference_mode(): 
        with torch.no_grad():
            for idx, layer_ in tqdm(enumerate(layers_)):     # for each layer

                feature_matrix = []
                
                idx = layers.index(layer_)
                
                if verbose:
                    print('idx: ', idx, 'layer: ', layer_)
                    
                count = 0
                
                metric_logger = utils.MetricLogger(delimiter="  ")
                header = "Test: {}".format(layer_)

                num_processed_samples = 0
                start_time = time.time()
                
                for image, target in metric_logger.log_every(data_loader_test, -1, header):     # for each image
                        
                    image = image.to(args.device, non_blocking=True)   
                    target = target.to(args.device, non_blocking=True)
                    
                    feature_list = model(image)
                    
                    #FIXME
                    output=feature_list[-1]
                    #output=feature_list
                       
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
                              ' | label: {}, prediction: {}'.format(target.data.detach().numpy()[0], np.argmax(output.squeeze(0).detach().cpu().numpy())),  
                              ' | image count: ', count, ' | feature_matrix: ({}, {})'.format(len(feature_matrix), feature_strip.shape[0]))
                    count += 1
                                    
                num_processed_samples = utils.reduce_across_processes(num_processed_samples)
                metric_logger.synchronize_between_processes()
                test_loss, test_acc1, test_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
                if verbose:
                    print(f'Test: test_acc1={test_acc1:.3f}, test_acc5={test_acc5:.3f}, test_loss={test_loss:.6f}, samples/s={num_processed_samples / (time.time() - start_time):.3f}')
                feature_matrix = torch.stack(feature_matrix, dim=0)
                feature_matrix = feature_matrix.detach().cpu().numpy()
                
                if verbose:
                    print('feature_matrix: ', feature_matrix.shape)
                    print('saving feature map of layer [{}]'.format(layer_))
                with open(args.output_dir + '/' + layer_ + '.pkl',  'wb') as f:
                    pickle.dump(feature_matrix, f, protocol=-1)

if __name__ =="__main__":
    
    print(args)
    
    # 1. create model and load weights
    model = create_model_and_load_weight(args)
    
    # 2. obtain layers_list and neurons_list
    layers, neurons, _ = generate_layers_and_neurons(model, args)     # [notice] neurons may be removed in future because of low generalization and additional workload
    
    layers_ = [i for i in layers if 'neuron' in i or 'fc' in i or 'pool' in i]
    
    # 3. feed model and layers&neurons into the evaluator
    spikingjelly_evaluate(model, layers, layers_, neurons, args, verbose=False)
    #ANN_evaluate(model, layers, layers_, neurons, args, verbose=True)


    

