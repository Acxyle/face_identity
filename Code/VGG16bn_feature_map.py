#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 16:27:54 2023

@author: acxyle

    generate all featuremaps of VGG
"""

import torch

import os
import numpy as np
from tqdm import tqdm
import pickle

import models
import params
import utils

device = 'cuda:0'
model_pth = "/home/acxyle/下载/-_Acxyle's Work/VGG/SNN-RAT_training_type/Results/CelebA50-finetuned-vgg16bn/vgg16bn_celebA50.pth"
param = torch.load(model_pth, map_location=torch.device(device))

params_replace = params.params_affine(param, verbose=False)

model = models.VGG16_bn(num_classes=50, mode='feature_map')
model.load_state_dict(params_replace)
model.to(device)

#utils.intermediate_output_show(model, device='cpu', test_sample="/home/acxyle/Downloads/017903.jpg", dest="/home/acxyle/下载/-_Acxyle's Work/Face_Identity/VGG_Identity_Selective/intermediate_output")

data_root = '/home/acxyle/Downloads/celeb50'
dest = "/media/acxyle/Data/ChromeDownload/Identity_VGG_Results"
utils.make_dir(dest)

data_list = [data_root+'/'+folder for folder in os.listdir(data_root)]
target_list = [
    'conv_1_1','bn_1_1', 'ReLU_1_1', 'conv_1_2', 'bn_1_2', 'ReLU_1_2',
               'conv_2_1','bn_2_1', 'ReLU_2_1', 'conv_2_2', 'bn_2_2', 'ReLU_2_2',
               'conv_3_1','bn_3_1', 'ReLU_3_1', 'conv_3_2','bn_3_2', 'ReLU_3_2', 'conv_3_3','bn_3_3', 'ReLU_3_3', 
               'conv_4_1','bn_4_1', 'ReLU_4_1', 'conv_4_2','bn_4_2', 'ReLU_4_2', 'conv_4_3','bn_4_3', 'ReLU_4_3', 
               'conv_5_1','bn_5_1', 'ReLU_5_1', 'conv_5_2','bn_5_2', 'ReLU_5_2', 'conv_5_3','bn_5_3', 'ReLU_5_3', 
               'avgpool',
               'fc6', 'ReLU_6',
               'fc7', 'ReLU_7',
               'fc8']

neurons = [
    64*224*224,64*224*224,64*224*224,64*224*224,64*224*224,64*224*224,
           128*112*112,128*112*112,128*112*112,128*112*112,128*112*112,128*112*112,
           256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,
           512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,
           512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,
           25088,
           4096, 4096,
           4096, 4096,
           50]

model.eval()

# 目标：results 文件夹下，每个 layer 生成一个对应的 .csv 文件

for idx, layer in enumerate(target_list):   # 对于期望的特定层的神经元输出
    
  feature_matrix = np.zeros(shape=(50*10, neurons[idx]))
  print('idx: ', idx, 'layer: ', layer, 'feature_matrix: ', feature_matrix.shape)
  count = 0

  for path in tqdm(sorted(data_list)):    # 对于celeb50中的每个ID文件夹 [0:50]
    #print('path: ', path)
    pic_dir = sorted([os.path.join(path, f) for f in os.listdir(path)])  # pic_dir[] 每一张图片
    
    for pic in pic_dir: # 对于每张图片 [0:10]

      img = utils.get_picture(pic)  
      img = img.unsqueeze(0)
      img = img.to(device)

      feat_list = model(img)  # 输出为包含所有目标层 feature_map 的 list

      feature_map = feat_list[idx].squeeze(0).flatten().detach().cpu().numpy()
      feature_matrix[count] = feature_map
      count += 1

  with open(dest + '/' + layer + '.pkl',  'wb') as f:
    pickle.dump(feature_matrix, f, protocol=4)