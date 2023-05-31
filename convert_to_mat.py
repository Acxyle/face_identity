#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:43:13 2023

@author: acxyle
"""

import os
import pickle
import numpy as np
import scipy.io as sio
import hdf5storage
from tqdm import tqdm

def convert_features():
    root = r"/media/acxyle/Data/ChromeDownload/Identity_VGG_Feature_Original/features"
    dest = r"/media/acxyle/Data/ChromeDownload/Identity_VGG_Feature_Original/features_mat/"
    files = os.listdir(root)
    files.remove('SIMI_cnt.pkl')
    
    for file in tqdm(files):
        with open(os.path.join(root, file), 'rb') as pkl:
            f = pickle.load(pkl)
        pkl.close()
        file = dest + file.split('.')[0]
        hdf5storage.savemat(f'{file}.mat', {'feature':f}, do_compression=True, format='7.3')
        del f
    
# -----
dim_list = [
        64*224*224,64*224*224,64*112*112,
        128*112*112,128*112*112,128*56*56,
        256*56*56,256*56*56,256*56*56,256*28*28,
        512*28*28,512*28*28,512*28*28,512*14*14,
        512*14*14,512*14*14,512*14*14,512*7*7,
        4096,4096,50]

def convert_encode():
    root = r"/media/acxyle/Data/ChromeDownload/Identity_VGG_Feature_Original/features/encode_class_dict_ver2.pkl"
    dest = r"/media/acxyle/Data/ChromeDownload/Identity_VGG_Feature_Original/encode_mat/"
    with open(root, 'rb') as pkl:
        f = pickle.load(pkl)
    pkl.close()
    
    layers = list(f.keys())
    
    for layer in tqdm(layers):
        encode_class = f[layer]
        
        neurons = list(encode_class.keys())
        tmp = []
        for neuron in neurons:
            tmp.append(encode_class[neuron])
        
        tmp = np.array(tmp, dtype=object)
        
        sio.savemat(dest+f'{layer}_encode.mat', {'encodeID':tmp})
    
    
    

if __name__ == "__main__":
    convert_encode()
