import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from tqdm import tqdm

root = "/home/acxyle/下载/-_Acxyle's Work/Face_Identity"
dest = "/home/acxyle/下载/-_Acxyle's Work/Face_Identity"

full_matrix = np.loadtxt(dest+'/'+'full_matirix.csv', delimiter=',')
print(full_matrix.shape)

# Select significantly responding neurons by Mann-Whitney U test
print('Neuron selection start...')

alpha = 0.01
sig_ind_total = []
for i in tqdm(range(50), desc='Comparing'): # 每个人
  A = full_matrix[i * 10: i * 10 + 10, :]
  B = np.vstack((full_matrix[: i * 10, :], full_matrix[i * 10 + 10:, :]))
  sig_ind = []
  for j in range(len(full_matrix[1])):  # for each neurob in all neuron of [C,W,H]
    # print(str(j)+'th neuron...')
    Ai = A[:, j]
    temp_p = []
    for k in range(49):
       # print(k)
       Bjk = B[k * 10: k * 10 + 10, j]
       try:
        stat, p = mannwhitneyu(Ai, Bjk, alternative='greater')
       except(Exception):
        p = 1  # 2 samples are identical
       temp_p.append(p)
    if max(temp_p) < alpha:
      # print(str(j)+'th neuron is significantly respond to '+str(i)+'th class')
      sig_ind.append(j)
  sig_ind = np.array(sig_ind)
  sig_ind_total.append(sig_ind)
sig_ind_total = np.array(sig_ind_total)

with open(dest+'/sig_ind_total.csv', 'wb') as f:
  for row in sig_ind_total:
    np.savetxt(f, [row], fmt='%d', delimiter=',')
    
print('Neuron selection finished!')