#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:08:25 2023

@author: acxyle
"""

import torch
import torchvision.transforms as transforms

import numpy as np
import random
import os
import skimage

from sklearn import svm
from sklearn.model_selection import train_test_split

from spikingjelly.activation_based.model.tv_ref_classify import utils

def make_dir(path):
  if os.path.exists(path) is False:
    os.makedirs(path)
    
def get_picture(pic_name):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32)
    transform = transforms.ToTensor()
    return transform(img)
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def cal_acc1_acc5(output, target):
    # define how to calculate acc1 and acc5
    acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
    return acc1, acc5

def SVM_classification(matrix, label):  # reusable function for SVM
    matrix_train, matrix_test, label_train, label_test = train_test_split(matrix, label, test_size=0.33, random_state=42)
    clf = svm.SVC()
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

#%% VGG16bn
layer_list_vgg16bn = [
        'conv_1_1','bn_1_1', 'neuron_1_1', 'conv_1_2', 'bn_1_2', 'neuron_1_2',
        'conv_2_1','bn_2_1', 'neuron_2_1', 'conv_2_2', 'bn_2_2', 'neuron_2_2',             
        'conv_3_1','bn_3_1', 'neuron_3_1', 'conv_3_2','bn_3_2', 'neuron_3_2', 'conv_3_3','bn_3_3', 'neuron_3_3', 
        'conv_4_1','bn_4_1', 'neuron_4_1', 'conv_4_2','bn_4_2', 'neuron_4_2', 'conv_4_3','bn_4_3', 'neuron_4_3', 
        'conv_5_1','bn_5_1', 'neuron_5_1', 'conv_5_2','bn_5_2', 'neuron_5_2', 'conv_5_3','bn_5_3', 'neuron_5_3', 
        'avgpool',
        'fc_6', 'neuron_6',
        'fc_7', 'neuron_7',
        'fc_8'
]

neuron_list_vgg16bn = [
        64*224*224, 64*224*224,64*224*224, 64*224*224,64*224*224,64*224*224,
        128*112*112, 128*112*112,128*112*112, 128*112*112,128*112*112,128*112*112,
        256*56*56,256*56*56, 256*56*56,256*56*56,256*56*56, 256*56*56,256*56*56,256*56*56,256*56*56,
        512*28*28,512*28*28, 512*28*28,512*28*28,512*28*28, 512*28*28,512*28*28,512*28*28,512*28*28, 
        512*14*14,512*14*14, 512*14*14,512*14*14,512*14*14, 512*14*14,512*14*14,512*14*14,512*14*14,
        25088,
        4096,4096,
        4096,4096, 
        50
]

def params_affine_vgg16bn(params, verbose=False):
    print('\n[Codinfo] redirecting params...')
    tmp = [  
        "conv_1_1.weight", "conv_1_1.bias", 
       "bn_1_1.weight", "bn_1_1.bias", "bn_1_1.running_mean", "bn_1_1.running_var", "bn_1_1.num_batches_tracked",
       "conv_1_2.weight", "conv_1_2.bias", 
       "bn_1_2.weight", "bn_1_2.bias", "bn_1_2.running_mean", "bn_1_2.running_var", "bn_1_2.num_batches_tracked",
       
       "conv_2_1.weight", "conv_2_1.bias", 
       "bn_2_1.weight", "bn_2_1.bias", "bn_2_1.running_mean", "bn_2_1.running_var", "bn_2_1.num_batches_tracked",
       "conv_2_2.weight", "conv_2_2.bias", 
       "bn_2_2.weight", "bn_2_2.bias", "bn_2_2.running_mean", "bn_2_2.running_var", "bn_2_2.num_batches_tracked",
       
       "conv_3_1.weight", "conv_3_1.bias", 
       "bn_3_1.weight", "bn_3_1.bias", "bn_3_1.running_mean", "bn_3_1.running_var", "bn_3_1.num_batches_tracked",
       "conv_3_2.weight", "conv_3_2.bias", 
       "bn_3_2.weight", "bn_3_2.bias", "bn_3_2.running_mean", "bn_3_2.running_var", "bn_3_2.num_batches_tracked",
       "conv_3_3.weight", "conv_3_3.bias", 
       "bn_3_3.weight", "bn_3_3.bias", "bn_3_3.running_mean", "bn_3_3.running_var", "bn_3_3.num_batches_tracked",
       
       "conv_4_1.weight", "conv_4_1.bias", 
       "bn_4_1.weight", "bn_4_1.bias", "bn_4_1.running_mean", "bn_4_1.running_var", "bn_4_1.num_batches_tracked",
       "conv_4_2.weight", "conv_4_2.bias", 
       "bn_4_2.weight", "bn_4_2.bias", "bn_4_2.running_mean", "bn_4_2.running_var", "bn_4_2.num_batches_tracked",
       "conv_4_3.weight", "conv_4_3.bias", 
       "bn_4_3.weight", "bn_4_3.bias", "bn_4_3.running_mean", "bn_4_3.running_var", "bn_4_3.num_batches_tracked",
       
       "conv_5_1.weight", "conv_5_1.bias", 
       "bn_5_1.weight", "bn_5_1.bias", "bn_5_1.running_mean", "bn_5_1.running_var", "bn_5_1.num_batches_tracked",
       "conv_5_2.weight", "conv_5_2.bias", 
       "bn_5_2.weight", "bn_5_2.bias", "bn_5_2.running_mean", "bn_5_2.running_var", "bn_5_2.num_batches_tracked",
       "conv_5_3.weight", "conv_5_3.bias", 
       "bn_5_3.weight", "bn_5_3.bias", "bn_5_3.running_mean", "bn_5_3.running_var", "bn_5_3.num_batches_tracked",
       
       "fc_6.weight", "fc_6.bias", "fc_7.weight", "fc_7.bias", "fc_8.weight", "fc_8.bias"]

    params_replace = {}
    i = 0
    for p in params['model']:
      params_replace[tmp[i]] = params['model'][p]
      if verbose:
          print('moving [%s]'%(p), 'to [%s]'%(tmp[i]))
      i += 1
    
    for i in params_replace:
      for j in params['model']:
        if torch.equal(params_replace[i].type_as(params['model'][j]), params['model'][j]):
          if verbose:
            print('%s in [params_replace] == %s in [params]'%(i, j))
          break
    
    print('\n[Codinfo] Parameters names modified.')
    
    return params_replace

#%% VGG19bn
layer_list_vgg19bn = [
        'conv_1_1','bn_1_1', 'neuron_1_1', 'conv_1_2', 'bn_1_2', 'neuron_1_2',
        'conv_2_1','bn_2_1', 'neuron_2_1', 'conv_2_2', 'bn_2_2', 'neuron_2_2',             
        'conv_3_1','bn_3_1', 'neuron_3_1', 'conv_3_2','bn_3_2', 'neuron_3_2', 'conv_3_3','bn_3_3', 'neuron_3_3', 'conv_3_4','bn_3_4', 'neuron_3_4', 
        'conv_4_1','bn_4_1', 'neuron_4_1', 'conv_4_2','bn_4_2', 'neuron_4_2', 'conv_4_3','bn_4_3', 'neuron_4_3', 'conv_4_4','bn_4_4', 'neuron_4_4', 
        'conv_5_1','bn_5_1', 'neuron_5_1', 'conv_5_2','bn_5_2', 'neuron_5_2', 'conv_5_3','bn_5_3', 'neuron_5_3', 'conv_5_4','bn_5_4', 'neuron_5_4', 
        'avgpool',
        'fc_6', 'neuron_6',
        'fc_7', 'neuron_7',
        'fc_8'
]

neuron_list_vgg19bn = [
        64*224*224, 64*224*224,64*224*224, 64*224*224,64*224*224,64*224*224,
        128*112*112, 128*112*112,128*112*112, 128*112*112,128*112*112,128*112*112,
        256*56*56,256*56*56, 256*56*56,256*56*56,256*56*56, 256*56*56,256*56*56,256*56*56,256*56*56, 256*56*56,256*56*56,256*56*56,
        512*28*28,512*28*28, 512*28*28,512*28*28,512*28*28, 512*28*28,512*28*28,512*28*28,512*28*28, 512*28*28,512*28*28,512*28*28,
        512*14*14,512*14*14, 512*14*14,512*14*14,512*14*14, 512*14*14,512*14*14,512*14*14,512*14*14, 512*14*14,512*14*14,512*14*14,
        25088,
        4096,4096,
        4096,4096, 
        50
]

def params_affine_vgg19bn(params, verbose=False):
    print('\n[Codinfo] redirecting params...')
    tmp = [  
        "conv_1_1.weight", "conv_1_1.bias", 
       "bn_1_1.weight", "bn_1_1.bias", "bn_1_1.running_mean", "bn_1_1.running_var", "bn_1_1.num_batches_tracked",
       "conv_1_2.weight", "conv_1_2.bias", 
       "bn_1_2.weight", "bn_1_2.bias", "bn_1_2.running_mean", "bn_1_2.running_var", "bn_1_2.num_batches_tracked",
       
       "conv_2_1.weight", "conv_2_1.bias", 
       "bn_2_1.weight", "bn_2_1.bias", "bn_2_1.running_mean", "bn_2_1.running_var", "bn_2_1.num_batches_tracked",
       "conv_2_2.weight", "conv_2_2.bias", 
       "bn_2_2.weight", "bn_2_2.bias", "bn_2_2.running_mean", "bn_2_2.running_var", "bn_2_2.num_batches_tracked",
       
       "conv_3_1.weight", "conv_3_1.bias", 
       "bn_3_1.weight", "bn_3_1.bias", "bn_3_1.running_mean", "bn_3_1.running_var", "bn_3_1.num_batches_tracked",
       "conv_3_2.weight", "conv_3_2.bias", 
       "bn_3_2.weight", "bn_3_2.bias", "bn_3_2.running_mean", "bn_3_2.running_var", "bn_3_2.num_batches_tracked",
       "conv_3_3.weight", "conv_3_3.bias", 
       "bn_3_3.weight", "bn_3_3.bias", "bn_3_3.running_mean", "bn_3_3.running_var", "bn_3_3.num_batches_tracked",
       "conv_3_4.weight", "conv_3_4.bias", 
       "bn_3_4.weight", "bn_3_4.bias", "bn_3_4.running_mean", "bn_3_4.running_var", "bn_3_4.num_batches_tracked",
       
       "conv_4_1.weight", "conv_4_1.bias", 
       "bn_4_1.weight", "bn_4_1.bias", "bn_4_1.running_mean", "bn_4_1.running_var", "bn_4_1.num_batches_tracked",
       "conv_4_2.weight", "conv_4_2.bias", 
       "bn_4_2.weight", "bn_4_2.bias", "bn_4_2.running_mean", "bn_4_2.running_var", "bn_4_2.num_batches_tracked",
       "conv_4_3.weight", "conv_4_3.bias", 
       "bn_4_3.weight", "bn_4_3.bias", "bn_4_3.running_mean", "bn_4_3.running_var", "bn_4_3.num_batches_tracked",
       "conv_4_4.weight", "conv_4_4.bias", 
       "bn_4_4.weight", "bn_4_4.bias", "bn_4_4.running_mean", "bn_4_4.running_var", "bn_4_4.num_batches_tracked",
       
       "conv_5_1.weight", "conv_5_1.bias", 
       "bn_5_1.weight", "bn_5_1.bias", "bn_5_1.running_mean", "bn_5_1.running_var", "bn_5_1.num_batches_tracked",
       "conv_5_2.weight", "conv_5_2.bias", 
       "bn_5_2.weight", "bn_5_2.bias", "bn_5_2.running_mean", "bn_5_2.running_var", "bn_5_2.num_batches_tracked",
       "conv_5_3.weight", "conv_5_3.bias", 
       "bn_5_3.weight", "bn_5_3.bias", "bn_5_3.running_mean", "bn_5_3.running_var", "bn_5_3.num_batches_tracked",
       "conv_5_4.weight", "conv_5_4.bias", 
       "bn_5_4.weight", "bn_5_4.bias", "bn_5_4.running_mean", "bn_5_4.running_var", "bn_5_4.num_batches_tracked",
       
       "fc_6.weight", "fc_6.bias", "fc_7.weight", "fc_7.bias", "fc_8.weight", "fc_8.bias"]

    params_replace = {}
    i = 0
    for p in params['model']:
      params_replace[tmp[i]] = params['model'][p]
      if verbose:
          print('moving [%s]'%(p), 'to [%s]'%(tmp[i]))
      i += 1
    
    for i in params_replace:
      for j in params['model']:
        if torch.equal(params_replace[i].type_as(params['model'][j]), params['model'][j]):
          if verbose:
            print('%s in [params_replace] == %s in [params]'%(i, j))
          break
    
    print('\n[Codinfo] Parameters names modified.')
    
    return params_replace