
import pickle
import os
import torch
import numpy as np

pklpath = '/content/drive/MyDrive/Colab Notebooks/colab_datasets/Face_Identity_Neuron/SigNeuron'

pkl_list = [os.path.join(pklpath, f) for f in os.listdir(pklpath) if f.split('.')[-1]=='pkl']
print(pkl_list)

encode_class_dict = {}    # 此 key-value 可以对所有层建立 dict
num_per_class = 10

for pklfile in pkl_list:    # 对于一层
  with open(pklfile, 'rb') as pkl:
    selective_feature = pickle.load(pkl)
    selective_feature = torch.tensor(selective_feature)
    selective_feature = selective_feature.flatten(0,1)
    sig_neuron = selective_feature.numpy()
    print(sig_neuron.shape)
    row, col = sig_neuron.shape
    encode_class = []
    for i in range(col):  # loop in neurons 对于每一个神经元
      neuron = sig_neuron[:, i]
      global_mean = np.mean(neuron)
      global_std = np.std(neuron)
      threshold = global_mean + 2 * global_std
      d = [neuron[i * num_per_class: i * num_per_class + num_per_class] for i in range(int(row / num_per_class))]
      d = np.array(d) 
      local_mean = np.mean(d, axis=1)   # local_mean 的 array
      for idx, mean in enumerate(local_mean):   # 如果满足条件，则记录下这个神经元特别敏感的ID | 这里好像有问题，没有建立神经元和ID的关系
        if mean > threshold:
          encode_class.append(idx + 1)  # 这里保存下来的是该层[不知道是哪个]神经元敏感的一个ID
    print(len(encode_class))  
    print(len(list(set(encode_class))))
  encode_class_dict.update({pklfile.split('/')[-1].split('-')[0]: encode_class})
print(encode_class_dict)
