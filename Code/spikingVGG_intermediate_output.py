#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:42:32 2023

@author: acxyle

    由于 opencv 和其他模块的冲突，使用时需要调用不同的环境 
    [action required] optimize this
    
    [Notice] 此模块使用了 opencv 的伪灰度图/热力图 JET，也使用了 spikingjelly，所以需要
    将此程序移动到 spikingjelly 文件夹下并启用 安装了 spikingjelly 和 opencv 的环境来执行
"""
import torch

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from spikingjelly.activation_based import functional, neuron, surrogate

import spikingVGG_model
import spiking_featuremap_utils

target_list = [
'conv_1_1','bn_1_1', 'neuron_1_1', 'conv_1_2', 'bn_1_2', 'neuron_1_2',
'conv_2_1','bn_2_1', 'neuron_2_1', 'conv_2_2', 'bn_2_2', 'neuron_2_2',             
'conv_3_1','bn_3_1', 'neuron_3_1', 'conv_3_2','bn_3_2', 'neuron_3_2', 'conv_3_3','bn_3_3', 'neuron_3_3', 
'conv_4_1','bn_4_1', 'neuron_4_1', 'conv_4_2','bn_4_2', 'neuron_4_2', 'conv_4_3','bn_4_3', 'neuron_4_3', 
'conv_5_1','bn_5_1', 'neuron_5_1', 'conv_5_2','bn_5_2', 'neuron_5_2', 'conv_5_3','bn_5_3', 'neuron_5_3'
]

def independent_intermediate_output_show(model, device, test_sample, dest):
    """
    generate intermediateoutput of each layer
    """
    model.to(device)
    model.eval()
    
    spiking_featuremap_utils.make_dir(dest)
    
    print(target_list)
    
    th_size = 256
    
    pic = spiking_featuremap_utils.get_picture(test_sample)
    pic = pic.unsqueeze(0)
    
    with torch.inference_mode():
        image = pic.to(device, non_blocking=True)
        image = image.unsqueeze(0).repeat(4, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        feat_list = model(image)    # 输出为包含所有目标层 feature_map 的 list
    
    for idx, layer_ in enumerate(target_list):   # 对于期望的特定层的神经元输出
        print('idx: ', idx, 'layer: ', layer_)
        
        #feature_map = feat_list[idx].mean(0).squeeze(0).flatten().detach().cpu().numpy()
        feature_map = feat_list[idx].mean(0).squeeze(0).detach().cpu().numpy()
        channel = random.randint(0, feature_map.shape[0]-1)
        feature_map = feature_map[channel,:,:]
        feature_img = np.asarray(feature_map * 255, dtype=np.uint8)
        feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
        
        
        if feature_img.shape[0] <= th_size:
            tmp_file = os.path.join(dest, layer_ + '-C_' + str(channel) + '.png')
            tmp_img = feature_img.copy()
            tmp_img = cv2.resize(tmp_img, (th_size,th_size), interpolation =  cv2.INTER_NEAREST)
            cv2.imwrite(tmp_file, tmp_img)

def merged_intermediate_output():
    root = "/home/acxyle/下载/-_Acxyle's Work/Face_Identity/SpikingVGG_Identity_Selective/[Important] Figures - CelebA9326/2.feature extractor/intermediate_output/"
    origin_image = '017903.jpg'
    
    image_list = [os.path.join(root, i) for i in os.listdir(root)]
    
    fig = plt.figure(figsize=(8,32))
    plt.subplots_adjust(left=0.01, right=0.99,
                       bottom=0.02, top=0.985,
                       wspace=0.01, hspace=0.15)
    
    # =============================================================================
    #         if img_name == origin_image:
    #             oimg = mpimg.imread(img)
    #             oimg = skimage.transform.resize(oimg, (224,224))
    # =============================================================================
    
    for idx, target in enumerate(target_list):
        print(idx, target)
        
        for img in image_list:
            img_name = img.split('/')[-1]
            img_l = img_name.split('-')[0]
            print(img_name, img_l)
            
            if img_l == target:
                img = mpimg.imread(img)
                plt.subplot(13, 3, idx+1)
                plt.title(img_name, y=-0.12)
                plt.imshow(img)
                plt.xticks([]), plt.yticks([])
    
    left, bottom, width, height = 0.7,0.67,0.25,0.25
    plt.savefig(os.path.join(root,'./test.jpg'), dpi = 300)
    plt.savefig(os.path.join(root,'./test.jpg'), dpi = 300)
            

if __name__ == "__main__":
# =============================================================================
#     model = spikingVGG_model.SpikingVGG16bn_f(mode='feature', spiking_neuron=neuron.IFNode, surrogate_function=surrogate.ATan(), num_classes=50, detach_reset=True)
#     functional.set_step_mode(model, step_mode='m')
#     
#     device = 'cpu'
#     test_sample = "/home/acxyle/下载/-_Acxyle's Work/Face_Identity/SpikingVGG_Identity_Selective/[Important] Figures - CelebA9326/2.feature extractor/017903.jpg"
#     dest = "/home/acxyle/下载/-_Acxyle's Work/Face_Identity/SpikingVGG_Identity_Selective/[Important] Figures - CelebA9326/2.feature extractor/intermediate_output"
#     independent_intermediate_output_show(model, device, test_sample, dest)
# =============================================================================
    
    merged_intermediate_output()