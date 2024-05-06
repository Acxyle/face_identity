#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 22:39:36 2022

@author: fangwei123456

@Modified: acxyle

"""

import os

import torch
import torchvision

from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import spiking_resnet, sew_resnet, spiking_vgg

import train_classify

import tdBN_spiking_resnet


# ----------------------------------------------------------------------------------------------------------------------
def get_args_parser(add_help=True):
    
    # --- basic config for NN training
    parser = train_classify.Trainer.training_parser()
    
    # --- env config
    parser.add_argument("--data_dir", type=str, default='/home/acxyle-workstation/Dataset')
    parser.add_argument("--dataset", type=str, default='CelebA_fold_0')
    parser.add_argument("--num_classes", type=int, default=2622)
   
    sub_parser = parser.add_subparsers(dest="command", help='Sub-command help')
    
    # --- ANN config
    parser_ANN = sub_parser.add_parser('ANN', help='triger the training for ANN')
    parser_ANN.add_argument("-m", "--model", type=str, default='vgg16')
    
    # --- SNN config
    parser_SNN = sub_parser.add_parser('SNN', help='triger the training for SNN')
    parser_SNN.add_argument("-m", "--model", type=str, default='tdbn_spiking_resnet50')
    parser_SNN.add_argument("--step_mode", type=str, default='m')
    parser_SNN.add_argument('--neuron', type=str, default='IF')
    parser_SNN.add_argument('--surrogate', type=str, default='ATan')
    parser_SNN.add_argument("--T", type=int, default=4)
    
    args = parser.parse_args()
    args.data_path = os.path.join(args.data_dir, args.dataset)
    
    if args.command == 'ANN':
        args.output_dir = os.path.join(f"./logs_{args.model}_{args.dataset}")
    elif args.command == 'SNN':
        args.output_dir = os.path.join(f"./logs_{args.model}_{args.neuron}_{args.surrogate}_T{args.T}_{args.dataset}")
    else:
        raise ValueError
    
    return args


# ----------------------------------------------------------------------------------------------------------------------
class SpikingjellyTrainer_ANN(train_classify.Trainer):
    
    def preprocess_train_sample(self, args, x: torch.Tensor):
        return x

    def preprocess_test_sample(self, args, x: torch.Tensor):
        return x

    def process_model_output(self, args, y: torch.Tensor):
        return y

 
    def get_tb_logdir_name(self, args):
        return super().get_tb_logdir_name(args)


    def load_model(self, args, num_classes):
        
        # --- ANN
        if args.model in torchvision.models.resnet.__all__:
            return torchvision.models.__dict__[args.model](num_classes=num_classes)
        
        elif args.model in torchvision.models.vgg.__all__:
            return torchvision.models.__dict__[args.model](num_classes=num_classes)
        
        else:
            raise ValueError(f"args.model should be one of {torchvision.models.vgg.__all__} {torchvision.models.resnet.__all__}")



# ----------------------------------------------------------------------------------------------------------------------
class SpikingjellyTrainer_SNN(train_classify.Trainer):
    
    def preprocess_train_sample(self, args, x: torch.Tensor):
        return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

    def preprocess_test_sample(self, args, x: torch.Tensor):
        return x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

    def process_model_output(self, args, y: torch.Tensor):
        return y.mean(0)  # return firing rate

    def get_tb_logdir_name(self, args):
        return super().get_tb_logdir_name(args) + f'_T{args.T}'


    def load_model(self, args, num_classes):
        
        _neuron = neuron.__dict__[f'{args.neuron}Node']
        _surrogate = surrogate.__dict__[args.surrogate]()
        
        # --- SNN
        if args.model in spiking_resnet.__all__:
            model = spiking_resnet.__dict__[args.model](pretrained=args.pretrained, 
                                                        num_classes=num_classes,
                                                        spiking_neuron=_neuron, 
                                                        surrogate_function=_surrogate, 
                                                        detach_reset=False, 
                                                        zero_init_residual=True)
            functional.set_step_mode(model, step_mode='m')
            return model
        
        elif args.model in tdBN_spiking_resnet.__all__:
            model = tdBN_spiking_resnet.__dict__[args.model](
                                                        alpha=1.,
                                                        v_threshold=1.,
                                                        num_classes=num_classes,
                                                        spiking_neuron=_neuron, 
                                                        surrogate_function=_surrogate, 
                                                        detach_reset=False, 
                                                        zero_init_residual=True)
            functional.set_step_mode(model, step_mode='m')
            return model
        
        elif args.model in sew_resnet.__all__:
            model = sew_resnet.__dict__[args.model](pretrained=args.pretrained, 
                                                    num_classes=num_classes,
                                                    spiking_neuron=_neuron,
                                                    surrogate_function=_surrogate, 
                                                    detach_reset=True, 
                                                    cnf='ADD')
            functional.set_step_mode(model, step_mode='m')
            return model
        
        elif args.model in spiking_vgg.__all__:
            model = spiking_vgg.__dict__[args.model](pretrained=args.pretrained, 
                                                     num_classes=num_classes,
                                                     spiking_neuron=_neuron,
                                                     surrogate_function=_surrogate, 
                                                     detach_reset=True)
            functional.set_step_mode(model, step_mode='m')
            return model
        
        else:
            raise ValueError(f"args.model should be one of {spiking_vgg.__all__} {spiking_resnet.__all__} \
                             {tdBN_spiking_resnet.__all__}, {sew_resnet.__all__}")


if __name__ == "__main__":
    
    args = get_args_parser()
    
    # -----
    if args.command == 'ANN':
        trainer = SpikingjellyTrainer_ANN()
    elif args.command == 'SNN':
        trainer = SpikingjellyTrainer_SNN()

    trainer.main(args)
                    



