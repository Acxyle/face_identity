#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:12:39 2023

@author: acxyle
"""
import torch
import torchvision.transforms as transforms

import skimage
import os
import numpy as np
import random
#import cv2

from spikingjelly.activation_based.model.tv_ref_classify import utils
from spikingjelly.activation_based import functional

from sklearn import svm
from sklearn.model_selection import train_test_split

def generate_vgg_layers(model, model_name, T=4):     # FIXME add T and other useful attributes
    
    model_dict = {'vgg5':'O', 'vgg11': 'A', 'vgg13': 'B', 'vgg16': 'D', 'vgg19': 'E'}
    cfgs = {
        'O': [64, 'M', 128, 128, 'M'],
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    cfg = model_dict[model_name.split('_')[1]]
    cfg = cfgs[cfg]
    
    layers = []     # [target] generate required layer name list
    cba = ['conv', 'bn', 'neuron'] 
    ca = ['conv', 'neuron']
    
    _l = 1
    _b = 1
    for v in cfg:
        
        if v == 'M':
            temp = ['L'+str(_l)+'_maxpool']
            layers += temp
            _b = 0
            _l += 1
        else:
            if 'bn' in model_name:
                temp = ['L'+str(_l)+'_B'+str(_b)+'_'+_i for _i in cba]
                layers += temp      # the basic structure of layer with BN

            else:
                temp = ['L'+str(_l)+'_B'+str(_b)+'_'+_i for _i in ca]
                layers += temp
        _b += 1
    
    layers.append('AdptiveAvgP')
    
    if '5' in model_name:
        classifier_A = ['fc_1', 'neuron_1', 'dropout_1', 'fc_2']
        for classifier_layer in classifier_A:
            layers.append(classifier_layer)
    else:
        classifier_B = ['fc_1', 'neuron_1', 'dropout_1', 'fc_2', 'neuron_2', 'dropout_2', 'fc_3']
        for classifier_layer in classifier_B:
            layers.append(classifier_layer)
    
    x = torch.randn(T, 1, 3, 224, 224)
    
    features = model(x)
    functional.reset_net(model)
    
    neurons = []
    shapes = []
    for idx, layer in enumerate(features):
        neurons.append(layer.mean(0).numel())
        shapes.append(layer.mean(0).squeeze(0).detach().cpu().numpy().shape)
            
    return layers, neurons, shapes

def generate_resnet_layers_list(model, model_name, T=4):     # return layers and neurons
    
    model_dict = {'spiking_resnet18': [2, 2, 2, 2], 
                  'spiking_resnet34': [3, 4, 6 ,3], 
                  'spiking_resnet50': [3, 4, 6, 3], 
                  'spiking_resnet101': [3, 4, 23, 3], 
                  'spiking_resnet152': [3, 8, 36, 3],
                  
                  'sew_resnet18': [2, 2, 2, 2], 
                  'sew_resnet34': [3, 4, 6 ,3], 
                  'sew_resnet50': [3, 4, 6, 3], 
                  'sew_resnet101': [3, 4, 23, 3], 
                  'sew_resnet152': [3, 8, 36, 3],}
    
    layers = ['conv_0', 'bn_0', 'neuron_0', 'maxpool_0']
    single_bottleneck_features = ['conv01', 'bn01', 'neuron01', 'conv02', 'bn02', 'neuron02', 'conv03', 'bn03', 'add_residual', 'neuron03']
    single_basicblock_features = ['conv01', 'bn01', 'neuron01', 'conv02', 'bn02', 'add_residual', 'neuron02']
    x = torch.randn(T, 1, 3, 224, 224)     # [notice] ransom sample used to collect the layer info
    
    if '18' in model_name or '34' in model_name:
        blocks = model_dict[model_name]
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
        shapes_ = []
        for idx, layer in enumerate(features):
            neurons_.append(layer.mean(0).numel())
            shapes_.append(layer.mean(0).squeeze(0).detach().cpu().numpy().shape)
    
    else:
        blocks = model_dict[model_name]
        layers_ = layers
        for idx, i in enumerate(blocks):   # each layer
            for j in range(i):          # each block
                temp = []
                for each_layer in single_bottleneck_features:
                    temp.append('L'+str(idx+1)+'_B'+str(j+1)+'_'+each_layer)
                layers_ += temp
        layers_ += ['avgpool', 'fc']
        
        features = model(x)     # [notice] 如果不需要生成 neurons，只需要生成 layers，则无需使用 model 进行一次 inference
        neurons_ = []
        shapes_ = []
        for idx, layer in enumerate(features):
            neurons_.append(layer.mean(0).numel())
            shapes_.append(layer.mean(0).squeeze(0).detach().cpu().numpy().shape)
    
    #[notice] this step is important for spikingjelly SNN and any other architectures without neuron reset
    functional.reset_net(model)
    
    return layers_, neurons_, shapes_

def generate_resnet_layers_list_ann(model, model_name):     # return layers and neurons
    
    model_dict = {'resnet18': [2, 2, 2, 2], 
                  'resnet34': [3, 4, 6 ,3], 
                  'resnet50': [3, 4, 6, 3], 
                  'resnet101': [3, 4, 23, 3], 
                  'resnet152': [3, 8, 36, 3],}
    
    layers = ['conv_0', 'bn_0', 'act_0', 'maxpool_0']
    single_bottleneck_features = ['conv01', 'bn01', 'act01', 'conv02', 'bn02', 'act02', 'conv03', 'bn03', 'add_residual', 'act03']
    single_basicblock_features = ['conv01', 'bn01', 'act01', 'conv02', 'bn02', 'add_residual', 'act02']
    x = torch.randn(1, 3, 224, 224)
    
    if '18' in model_name or '34' in model_name:
        blocks = model_dict[model_name]
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
        shapes_ = []
        for idx, layer in enumerate(features):
            neurons_.append(layer.numel())
            shapes_.append(layer.squeeze(0).detach().cpu().numpy().shape)
    
    else:
        blocks = model_dict[model_name]
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
        shapes_ = []
        for idx, layer in enumerate(features):
            neurons_.append(layer.numel())
            shapes_.append(layer.squeeze(0).detach().cpu().numpy().shape)
    
    return layers_, neurons_, shapes_

def generate_vgg_layers_list_ann(model, model_name):
    
    model_dict = {'vgg5':'O', 'vgg11': 'A', 'vgg13': 'B', 'vgg16': 'D', 'vgg19': 'E'}
    cfgs = {
        'O': [64, 'M', 128, 128, 'M'],
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    cfg = model_dict[model_name.split('_')[0]]
    cfg = cfgs[cfg]
    
    layers = []     # [target] generate required layer name list
    cba = ['conv', 'bn', 'neuron'] 
    ca = ['conv', 'neuron']
    
    _l = 1
    _b = 1
    for v in cfg:
        
        if v == 'M':
            temp = ['L'+str(_l)+'_maxpool']
            layers += temp
            _b = 0
            _l += 1
        else:
            if 'bn' in model_name:
                temp = ['L'+str(_l)+'_B'+str(_b)+'_'+_i for _i in cba]
                layers += temp      # the basic structure of layer with BN

            else:
                temp = ['L'+str(_l)+'_B'+str(_b)+'_'+_i for _i in ca]
                layers += temp
        _b += 1
    
    layers.append('AdptiveAvgP')
    
    if '5' in model_name:
        classifier_A = ['fc_1', 'neuron_1', 'dropout_1', 'fc_2']
        for classifier_layer in classifier_A:
            layers.append(classifier_layer)
    else:
        classifier_B = ['fc_1', 'neuron_1', 'dropout_1', 'fc_2', 'neuron_2', 'dropout_2', 'fc_3']
        for classifier_layer in classifier_B:
            layers.append(classifier_layer)
            
    #layers.append('sigmoid')
    
    x = torch.randn(1, 3, 224, 224)
    
    features = model(x)
    
    neurons = []
    shapes = []
    for idx, layer in enumerate(features):
        neurons.append(layer.mean(0).numel())
        shapes.append(layer.mean(0).squeeze(0).detach().cpu().numpy().shape)
            
    return layers, neurons, shapes

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
    
def SVM_classification(matrix, label):     # [notice] use default setting of SVM
    # [notice] test_size should be the same with finetune but not 100% necessary, random_state is a personal preference
    matrix_train, matrix_test, label_train, label_test = train_test_split(matrix, label, test_size=0.2, random_state=6)    
    clf = svm.SVC()     # .SVC() .LinearSVC() .NuSVC() ... 
    if matrix_train.shape[1] == 0:
      acc = 0.
    else:
      clf.fit(matrix_train, label_train)
      predicted = clf.predict(matrix_test)
      correct = 0
      samples = len(label_test)
      for i in range(samples):
          if predicted[i] == label_test[i]:
              correct += 1
      acc = correct / samples
    return acc

def makeLabels(sample_num, class_num):  # generate a label list
    label = []
    for i in range(class_num):
        label += [i + 1] * sample_num
    return label

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# [warning] waiting to write
# 不一样的地方就是 Vanilla SNN 不需要提供五维的输入，这导致这段代码看起来意义不明
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

    