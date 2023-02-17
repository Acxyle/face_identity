#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 00:12:41 2023

generate featuremap of SpikingVGG

[Notice] This Version is valid for --- Lab Desktop --- with default settings
"""

import torch
import torch.nn as nn
import torchvision
import os
from tqdm import tqdm
import time
import numpy as np
import pickle
from PIL import Image

from spikingjelly.activation_based.model.tv_ref_classify import presets, utils
from torchvision.transforms.functional import InterpolationMode
from spikingjelly.activation_based import functional, neuron, surrogate

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url
    
import spikingVGG_model
import spiking_featuremap_utils
import argparse
    
parser = argparse.ArgumentParser(description="Spikingjelly VGG Featuremap Extractor", add_help=True)

parser.add_argument("--spikingjelly_evaluate", dest="spikingjelly_evaluate", help="execute model evaluate on CelebA50", action="store_true")
parser.add_argument("--obtain_feature_map", dest="obtain_feature_map", help="generate feature map for CelebA50 Dataset", action="store_true")
parser.add_argument("--return_T_dim", dest = "return_T_dim", help="return the output value before merging into firing rate", action="store_true")

parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--model", type=str, default='spiking_vgg16_bn')
parser.add_argument("--step_mode", type=str, default='m')
parser.add_argument("--T", type=int, default=4)
parser.add_argument("--weight_path", type=str, default='/home/acxyle/Git/spikingjelly-master/spikingjelly/activation_based/model/logs-ft-CelebA50-from-vgg16bn_CelebA9326/pt/checkpoint_max_test_acc1.pth')
parser.add_argument("--data_path", type=str, default='/home/acxyle/Downloads/celeb50/')
parser.add_argument("--output_dir", type=str, default='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Results/')
parser.add_argument("--mode", type=str, default='classification')

args = parser.parse_args()


def spikingjelly_evaluate(model, data_path, device='cuda:0', log_suffix="", T=args.T, batch_size=4):
    
    valdir = data_path
    
    preprocessing = presets.ClassificationPresetEval(crop_size=224, resize_size=232, interpolation=InterpolationMode('bilinear'))
    
    dataset_test = torchvision.datasets.ImageFolder(valdir, preprocessing)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=8, pin_memory= False,
        worker_init_fn=spiking_featuremap_utils.seed_worker)
    
    model.to(device)
    model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    start_time = time.time()
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    with torch.inference_mode():
        for image, target in tqdm(metric_logger.log_every(data_loader_test, -1, header)):
            
            image = image.to(device, non_blocking=True)
            image = image.unsqueeze(0).repeat(T, 1, 1, 1, 1)
            target = target.to(device, non_blocking=True)
            output = model(image).mean(0)
            
            loss = criterion(output, target)

            acc1, acc5 = spiking_featuremap_utils.cal_acc1_acc5(output, target)
            batch_size = target.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
            functional.reset_net(model)

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    metric_logger.synchronize_between_processes()
    test_loss, test_acc1, test_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f'Test: test_acc1={test_acc1:.3f}, test_acc5={test_acc5:.3f}, test_loss={test_loss:.6f}, samples/s={num_processed_samples / (time.time() - start_time):.3f}')
    
    return test_loss, test_acc1, test_acc5


def obtain_feature_map(model, data_root, dest, device='cuda:0', T=args.T):
    """
    generate the .pkl file for the feature map for each layer    
    """
    spiking_featuremap_utils.make_dir(dest)
    
    data_list = [data_root+'/'+folder for folder in os.listdir(data_root)]
    target_list = spiking_featuremap_utils.layer_list_vgg16bn
    neurons= spiking_featuremap_utils.neuron_list_vgg16bn

    model.to(device)
    model.eval()
    
    preprocessing = presets.ClassificationPresetEval(crop_size=224, resize_size=232, interpolation=InterpolationMode('bilinear'))   # 这个preprocessing
    
    for idx, layer_ in enumerate(target_list):   # each layer
    
        feature_matrix = np.zeros(shape=(50*10, neurons[idx]))
        print('idx: ', idx, 'layer: ', layer_, 'feature_matrix: ', feature_matrix.shape)
        count = 0
        
        for path in tqdm(sorted(data_list)):    # each ID
           pic_dir = sorted([os.path.join(path, f) for f in os.listdir(path)])  
           
           for pic in pic_dir: # each fig
               pic = Image.open(pic)
               pic = preprocessing(pic)
               with torch.inference_mode():
                  image = pic.to(device, non_blocking=True)
                  image = image.unsqueeze(0).repeat(T, 1, 1, 1, 1)
                  feat_list = model(image) 
               feature_map = feat_list[idx].mean(0).squeeze(0).flatten().detach().cpu().numpy() # [return firing rate]
               feature_matrix[count] = feature_map
               count += 1
               
        with open(dest + '/' + layer_ + '.pkl',  'wb') as f:
            pickle.dump(feature_matrix, f, protocol=-1)  # or try protocol=-1?

        
        
if __name__ =="__main__":
    print(args)
    if args.model == 'spiking_vgg16_bn':
        model = spikingVGG_model.SpikingVGG16bn_f(mode=args.mode, spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), num_classes=50, detach_reset=True)
    else:
        raise RuntimeError('\n [Codwaring] Please connect to other SpikingVGG model in main code')
    functional.set_step_mode(model, step_mode=args.step_mode)
    #print(model)
    print('\n[Codinfo] Model created.')
    
    pth_path = args.weight_path
    params = torch.load(pth_path, map_location=torch.device(args.device))
    params_replace = spiking_featuremap_utils.params_affine_vgg16bn(params, verbose=False)
    model.load_state_dict(params_replace)
    
    data_path = args.data_path
    
    if args.spikingjelly_evaluate:
        spikingjelly_evaluate(model, data_path)
    
    if args.obtain_feature_map:
        dest = args.output_dir
        obtain_feature_map(model, data_path, dest, args.device)

