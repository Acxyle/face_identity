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

from sklearn import svm
from sklearn.model_selection import train_test_split

layer_sequence = [
                   'conv_1_1','bn_1_1', 'ReLU_1_1', 'conv_1_2', 'bn_1_2', 'ReLU_1_2',
                   'conv_2_1','bn_2_1', 'ReLU_2_1', 'conv_2_2', 'bn_2_2', 'ReLU_2_2',
                   'conv_3_1','bn_3_1', 'ReLU_3_1', 'conv_3_2','bn_3_2', 'ReLU_3_2', 'conv_3_3','bn_3_3', 'ReLU_3_3', 
                   'conv_4_1','bn_4_1', 'ReLU_4_1', 'conv_4_2','bn_4_2', 'ReLU_4_2', 'conv_4_3','bn_4_3', 'ReLU_4_3', 
                   'conv_5_1','bn_5_1', 'ReLU_5_1', 'conv_5_2','bn_5_2', 'ReLU_5_2', 'conv_5_3','bn_5_3', 'ReLU_5_3', 
                   'avgpool',
                   'fc_6', 'ReLU_6',
                   'fc_7', 'ReLU_7',
                   'fc_8'
]

dim_list = [64*224*224,64*224*224,64*224*224,64*224*224,64*224*224,64*224*224,
       128*112*112,128*112*112,128*112*112,128*112*112,128*112*112,128*112*112,
       256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,256*56*56,
       512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,512*28*28,
       512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,512*14*14,
       25088,
       4096, 4096,
       4096, 4096,
       50]

def get_picture(pic_name):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32)
    transform = transforms.ToTensor()
    return transform(img)


def make_dir(path):
  if os.path.exists(path) is False:
    os.makedirs(path)
    
    
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

#def plot_selective_neuron_ratio():
    

# =============================================================================
# # 由于 opencv 和其他模块的冲突，使用时需要调用不同的环境并注释cv2的调用，并不方便，此问题等待后续解决
# def intermediate_output_show(model, device, test_sample, dest):
#     """
#     generate intermediateoutput of each layer
#     """
#     make_dir(dest)
#     
#     target_list = ['conv_1_1','bn_1_1', 'ReLU_1_1', 'conv_1_2', 'bn_1_2', 'ReLU_1_2',
#                    'conv_2_1','bn_2_1', 'ReLU_2_1', 'conv_2_2', 'bn_2_2', 'ReLU_2_2',
#                    'conv_3_1','bn_3_1', 'ReLU_3_1', 'conv_3_2','bn_3_2', 'ReLU_3_2', 'conv_3_3','bn_3_3', 'ReLU_3_3', 
#                    'conv_4_1','bn_4_1', 'ReLU_4_1', 'conv_4_2','bn_4_2', 'ReLU_4_2', 'conv_4_3','bn_4_3', 'ReLU_4_3', 
#                    'conv_5_1','bn_5_1', 'ReLU_5_1', 'conv_5_2','bn_5_2', 'ReLU_5_2', 'conv_5_3','bn_5_3', 'ReLU_5_3']
#     
#     th_size = 256
#     
#     pic = get_picture(test_sample)
#     pic = pic.unsqueeze(0)
#     
#     with torch.inference_mode():
#         image = pic.to(device, non_blocking=True)
#         feat_list = model(image)    # 输出为包含所有目标层 feature_map 的 list
#     
#     for idx, layer_ in enumerate(target_list):   # 对于期望的特定层的神经元输出
#         print('idx: ', idx, 'layer: ', layer_)
#         
#         #feature_map = feat_list[idx].mean(0).squeeze(0).flatten().detach().cpu().numpy()
#         feature_map = feat_list[idx].mean(0).squeeze(0).detach().cpu().numpy()
#         channel = random.randint(0, feature_map.shape[0]-1)
#         feature_map = feature_map[channel,:,:]
#         feature_img = np.asarray(feature_map * 255, dtype=np.uint8)
#         feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
#         
#         
#         if feature_img.shape[0] <= th_size:
#             tmp_file = os.path.join(dest, layer_ + '-C_' + str(channel) + '.png')
#             tmp_img = feature_img.copy()
#             tmp_img = cv2.resize(tmp_img, (th_size,th_size), interpolation =  cv2.INTER_NEAREST)
#             cv2.imwrite(tmp_file, tmp_img)
# =============================================================================
