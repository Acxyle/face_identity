"""
Section 2: Selectivity
    ANOVA计算 selectivity
    此部分建立的是 layer 和 ID 的关系
"""

from tqdm import tqdm
import scipy.stats as stats
import pickle
import os
import numpy as np


alpha = 0.01

root = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Results/'
dest = '/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/'
layer_list = [(root+f) for f in sorted(os.listdir(root)) if f.split('.')[-1]=='pkl']
#print(layer_list)

# ANOVA for each neuron

for layer in layer_list:
  neuron_idx = []
  pl = []
  with open(layer, 'rb') as pkl:
      feature = pickle.load(pkl)
  layer_name = layer.split('/')[-1].split('.')[0]
  print('\n', layer_name, feature.shape)

  for i in tqdm(range(len(feature[1]))):
    neuron = feature[:, i]
    d = [neuron[i * 10: i * 10 + 10] for i in range(50)]
    p = stats.f_oneway(
              d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9],
              d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19],
              d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29],
              d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39],
              d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49])[1]  # [0] for F-value, [1] for p-value
    pl.append(p)

  pl = np.array(pl)   # C*W*H () 个p值， 每个p值由500个vector进行ANOVA而得
  if not os.path.exists(dest + layer_name + '-pvalue.csv'):
    np.savetxt(dest + layer_name + '-pvalue.csv', pl)

  sig_neuron_idx = [idx for idx, p in enumerate(pl) if p < alpha]
  print(layer, 'has', len(sig_neuron_idx), 'significant neurons')
  neuron_idx.append(sig_neuron_idx)

  neuron_idx = np.array(neuron_idx)
  if not os.path.exists(dest + layer_name + '-neuronIdx.csv'):
    np.savetxt(dest + layer_name + '-neuronIdx.csv', neuron_idx, delimiter=',')

print('Count finished!')