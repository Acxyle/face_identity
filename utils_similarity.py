#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 23:31:53 2023

@author: acxyle-workstation
"""

import os
import pandas as pd

import scipy.stats as stats
import warnings
import logging
import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from collections import Counter
from joblib import Parallel, delayed
from matplotlib import gridspec
from scipy.stats import gaussian_kde, norm, skew, lognorm, kstest
from scipy.spatial.distance import pdist, squareform

from scipy.integrate import quad


def selectivity_analysis_calculation(metric: str, feature: np.array):
    """
        based on [metric] to calculate
        [notice] for 'euclidean' the scaling is not in consider
    """
    
    similarity_dict = {}
    
    if 'euclidean' in metric.lower():
        
        if feature.size == 0:
            similarity_dict = None
            
        else:
            
            if np.any(np.isnan(feature)):
                
                mask = np.full((feature.shape[1],), True)
                for _ in range(feature.shape[1]):
                    if np.any(np.isnan(feature[:, _])):
                        mask[_] = False
                        
                if np.all(np.isnan(mask)):
                    similarity_dict = None
                    
                else:
                    feature = feature[:, mask]
                    similarity_value = pdist(feature, 'euclidean')     # (1225,)
                    similarity_dict.update({
                        'vector': similarity_value,     # for RSA
                        'matrix': squareform(similarity_value),     # (50, 50)
                        'contains_nan': True,     
                        'num_units': feature.shape[1]
                        })
            
            else:
                
                similarity_value = pdist(feature, 'euclidean')     # (1225,)
                similarity_dict.update({
                    'vector': similarity_value,     # for RSA
                    'matrix': squareform(similarity_value),     # (50, 50)
                    'contains_nan': False,     
                    'num_units': feature.shape[1]
                    })
    
    elif 'pearson' in metric.lower():
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            if feature.shape[1] == 0:
                similarity_dict = None
            
            else:
                similarity_matrix = np.corrcoef(feature)
                
                if np.any(np.isnan(similarity_matrix)):     # when detecting NaN value, i.e. the values of one class are identical
                    similarity_dict.update({'contains_nan': True})
                    similarity_matrix = np.ma.corrcoef(np.ma.masked_invalid(feature)).data
                    
                else:
                    similarity_dict.update({'contains_nan': False})
                    
                DSM = (1 - similarity_matrix)/2     # SM [-1, 1] -> DSM [0, 1]
                DSM_z, similarity_value = Square2Tri(DSM)     # (1225,)
    
                similarity_dict.update({
                    'vector': similarity_value,     # for RSA
                    'matrix': DSM_z,     # for plot
                    'num_units': feature.shape[1]
                    })
    
    else:
        raise ValueError(f'[Coderror] {metric} not supported')
    
    return similarity_dict


def Square2Tri(DSM):
    """
        in python, the squareform() function can convert an array to square or vice versa, 
        but need to make sure the matrix is symmetrical and 0 diagonal values
    """
    # original version
    #M_z = 1 - np.arctanh(DSM)
    #V = np.triu(M_z, k=1).T
    #V = V[V!=0]     # what if the 0 value exists in the upper triangle
    
    if np.max(DSM) > 1:
        raise ValueError(f'[Codinfo] Value range is (-1, 1) but detected {np.max(DSM)}')
    
    DSM_z = np.arctanh(DSM)
    DSM_z = (DSM_z+DSM_z.T)/2
    for _ in range(DSM.shape[0]):
        DSM_z[_,_]=0
    V = squareform(DSM_z)
    # -----
    
    return DSM_z, V


def describe_numpy(input:np.array=None):
    
    if input is None or input[~np.isnan(input)].size == 0:
        
        print('[Codinfo] [describe_numpy] received input is None or full of nan values, returned None')
        
        return None
    
    array_stats = {
        "count_valid": input[~np.isnan(input)].size,
        "count_all": input.size,
        "mean": np.mean(input[~np.isnan(input)]),
        "std": np.std(input[~np.isnan(input)]),
        "min": np.min(input[~np.isnan(input)]),
        "25%": np.percentile(input[~np.isnan(input)], 25),
        "50% (median)": np.median(input[~np.isnan(input)]),
        "75%": np.percentile(input[~np.isnan(input)], 75),
        "max": np.max(input[~np.isnan(input)])
    }
    
    return array_stats

def fake_legend_describe_numpy(similarity, ax):
    
    similarity_stats_dict = describe_numpy(similarity)
    
    fake_legend_stats_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='orange', markersize=5, markeredgecolor='orange') for _ in range(8)]
    fake_legend_stats_labels = [
        f"count: {similarity_stats_dict['count_valid']}/{similarity_stats_dict['count_all']}",
        f"mean: {similarity_stats_dict['mean']:.3f}",
        f"std: {similarity_stats_dict['std']:.3f}",
        f"min: {similarity_stats_dict['min']:.3f}",
        f"25%: {similarity_stats_dict['25%']:.3f}",
        f"50%: {similarity_stats_dict['50% (median)']:.3f}",
        f"75%: {similarity_stats_dict['75%']:.3f}",
        f"max: {similarity_stats_dict['max']:.3f}"
        ]
    
    fake_legend = ax.legend(fake_legend_stats_handles, fake_legend_stats_labels, framealpha=0.25, ncol=2)
    ax.add_artist(fake_legend)
