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
            
            if np.any(np.isnan(feature)):     # (500, num_units)
                
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
                if np.any(np.isnan(feature)) or np.any([len(np.unique(feature[_, :]))==1 for _ in range(feature.shape[0])]):
                    similarity_dict.update({'contains_nan': True})
                    similarity_matrix = np.ma.corrcoef(np.ma.masked_invalid(feature)).data
                
                else:
                    similarity_dict.update({'contains_nan': False})
                    similarity_matrix = np.corrcoef(feature)
                    
                similarity_matrix_z, similarity_value = matrix_to_vector(similarity_matrix)     # (1225,)
    
                similarity_dict.update({
                    'vector': similarity_value,     # for RSA
                    'matrix': similarity_matrix_z,     # for plot
                    'num_units': feature.shape[1]
                    })
    
    else:
        raise ValueError(f'[Coderror] {metric} not supported')
    
    return similarity_dict


def matrix_to_vector(matrix):

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # --- original version
    matrix_z = 1 - np.arctanh(matrix)     # similarity [-1, 1] -> distance [0,2]
    vector = matrix_z[np.triu_indices(matrix.shape[0], 1)]     

    return matrix_z, vector


def describe_numpy(input:np.array=None):
    """
        this function uses a dict format to print the information of a numpy array like Pandas describe
    """
    
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

def fake_legend_describe_numpy(input, ax, mask=None):
    

    similarity_interest = input[mask]
    
    similarity_stats_dict = describe_numpy(similarity_interest)
    
    if similarity_stats_dict is None:
        
        fake_legend_stats_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='orange', markersize=5, markeredgecolor='orange') for _ in range(8)]
        
        fake_legend_stats_labels = [
            f"count: {similarity_interest.size}/{input.size}",
            "mean: - ",
            "std: - ",
            "min: - ",
            "25%: - ",
            "50%: - ",
            "75%: - ",
            "max: - "
            ]

        fake_legend = ax.legend(fake_legend_stats_handles, fake_legend_stats_labels, framealpha=0.25, ncol=2, handlelength=0, borderpad=0, labelspacing=0)
        ax.add_artist(fake_legend)
        
    else:
    
        fake_legend_stats_handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='orange', markersize=5, markeredgecolor='orange') for _ in range(8)]
        
        fake_legend_stats_labels = [
            f"count: {similarity_interest.size}/{input.size}",
            f"mean: {similarity_stats_dict['mean']:.3f}",
            f"std: {similarity_stats_dict['std']:.3f}",
            f"min: {similarity_stats_dict['min']:.3f}",
            f"25%: {similarity_stats_dict['25%']:.3f}",
            f"50%: {similarity_stats_dict['50% (median)']:.3f}",
            f"75%: {similarity_stats_dict['75%']:.3f}",
            f"max: {similarity_stats_dict['max']:.3f}"
            ]
        
        fake_legend = ax.legend(fake_legend_stats_handles, fake_legend_stats_labels, framealpha=0.25, ncol=2, handlelength=0, borderpad=0, labelspacing=0)
        ax.add_artist(fake_legend)
    
    return fake_legend