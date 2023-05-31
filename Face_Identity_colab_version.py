#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:54:41 2023

@author: Jinge Wang
@modified by: acxyle

    [notice] this document is sa the same as the colab file but for local experiment and especially to plot the wantted fig
    
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage.data
import skimage.io
import skimage.transform
import torchfile

import torchvision

import os
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

import torch
import os
import numpy as np
from tqdm import tqdm
import pickle

class VGG_16(nn.Module):
  def __init__(self, mode='classification'):
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.mode = mode
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)

  def load_weights(self, path='/content/drive/MyDrive/Colab Notebooks/colab_datasets/Face_Identity_pth/VGG_FACE.t7'):
    """ Function to load luatorch pretrained
    Args:
      path: path for the luatorch pretrained
    """
    model = torchfile.load(path)
    counter = 1
    block = 1
    for i, layer in enumerate(model.modules):
      if layer.weight is not None:
        if block <= 5:
          self_layer = getattr(self, "conv_%d_%d" % (block, counter))
          # print(self_layer)
          counter += 1
          if counter > self.block_size[block - 1]:
            counter = 1
            block += 1
          self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
          # print(self_layer.weight.data.shape)
          self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
        else:
          self_layer = getattr(self, "fc%d" % (block))
          block += 1
          self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
          # print(self_layer.weight.data.shape)  # [out, in, h, w]
          self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                
  def forward(self, x):

        x01 = F.relu(self.conv_1_1(x))
        x02 = F.relu(self.conv_1_2(x01))
        x03 = F.max_pool2d(x02, 2, 2)
        x04 = F.relu(self.conv_2_1(x03))
        x05 = F.relu(self.conv_2_2(x04))
        x06 = F.max_pool2d(x05, 2, 2)
        x07 = F.relu(self.conv_3_1(x06))
        x08 = F.relu(self.conv_3_2(x07))
        x09 = F.relu(self.conv_3_3(x08))
        x10 = F.max_pool2d(x09, 2, 2)
        x11 = F.relu(self.conv_4_1(x10))
        x12 = F.relu(self.conv_4_2(x11))
        x13 = F.relu(self.conv_4_3(x12))
        x14 = F.max_pool2d(x13, 2, 2)
        x15 = F.relu(self.conv_5_1(x14))
        x16 = F.relu(self.conv_5_2(x15))
        x17 = F.relu(self.conv_5_3(x16))
        x18 = F.max_pool2d(x17, 2, 2)
        x19 = x18.view(x18.size(0), -1)
        x20 = F.relu(self.fc6(x19))
        x21 = F.dropout(x20, 0.5, self.training)
        x22 = F.relu(self.fc7(x21))
        x23 = F.dropout(x22, 0.5, self.training)
        x24 = F.softmax(self.fc8(x23), dim=-1)

        feat = [
            x01, x02, x03,
            x04, x05, x06,
            x07, x08, x09, x10,
            x11, x12, x13, x14,
            x15, x16, x17, x18,
            x20, x22, x24
        ]
        if self.mode == 'classification':
          return x24
        elif self.mode == 'feature':
          return feat
        else:
          raise RuntimeError('--- Please type the right mode')
         
            
def get_picture(pic_name):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32)
    transform = transforms.ToTensor()
    return transform(img)

def load_split_train_test(root, valid_size):

    data_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root, transform=data_transform)
    test_data = datasets.ImageFolder(root, transform=data_transform)

    num_train = len(train_data)         # num of training set, actually ATM this is the total num of img under root
    indices = list(range(num_train))    # creat indices for randomize

    split = int(np.floor(valid_size * num_train))  # pick the break point based on the ratio given by validtion size
    np.random.shuffle(indices)          # randomize the data indices

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)  # randomize again
    test_sampler = SubsetRandomSampler(test_idx)

    # ===================load data========================
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=4)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=1)
    return train_loader, test_loader

device = 'cpu'
model_pth = "/content/drive/MyDrive/Colab Notebooks/colab_datasets/Face_Identity_pth/celebA50_pth/9.pth"
model = torch.load(model_pth, map_location=torch.device(device))

model.mode = 'feature'  # √
model.eval()

data_root = '/content/drive/MyDrive/Colab Notebooks/colab_datasets/celeb50/'
dest = "/content/drive/MyDrive/Colab Notebooks/colab_datasets/Face_Identity_Results/feature_maps/"

data_list = [data_root+'/'+folder for folder in os.listdir(data_root)]

target_list = [
        'Conv_1_1', 
        'Conv_1_2', 'Pool_1',
        'Conv_2_1', 'Conv_2_2', 'Pool_2',
        'Conv_3_1', 'Conv_3_2', 'Conv_3_3', 'Pool_3',
        'Conv_4_1', 'Conv_4_2', 'Conv_4_3', 'Pool_4',
        'Conv_5_1', 'Conv_5_2', 'Conv_5_3', 'Pool_5',
        'FC_6', 'FC_7', 'FC_8'
        ]

neurons = [
        64*224*224,
        64*224*224,64*112*112,
        128*112*112,128*112*112,128*56*56,
        256*56*56,256*56*56,256*56*56,256*28*28,
        512*28*28,512*28*28,512*28*28,512*14*14,
        512*14*14,512*14*14,512*14*14,512*7*7,
        4096,4096,50
        ]

for idx, layer in enumerate(target_list):   # 对于期望的特定层的神经元输出
    
  feature_matrix = np.zeros(shape=(50*10, neurons[idx]))
  print('idx: ', idx, 'layer: ', layer, 'feature_matrix: ', feature_matrix.shape)
  count = 0

  feature_matrix = []

  for path in tqdm(sorted(data_list)):    # 对于celeb50中的每个ID文件夹 [0:50]
    pic_dir = sorted([os.path.join(path, f) for f in os.listdir(path)])  # pic_dir[] 每一张图片
    
    for pic in pic_dir: # 对于每张图片 [0:10]

      img = get_picture(pic)  
      img = img.unsqueeze(0)
      img = img.to(device)

      feat_list = model(img)  # 输出为包含所有目标层 feature_map 的 list

      feature_map = feat_list[idx]
      feature_matrix.append(feature_map)

      feature_map = feat_list[idx].squeeze(0).flatten().detach().cpu().numpy()
      feature_matrix[count] = feature_map
      count += 1

  with open(dest + '/' + layer + '.pkl',  'wb') as f:
    pickle.dump(feature_matrix, f, protocol=4)
