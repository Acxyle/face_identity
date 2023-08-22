#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:44:28 2023

@author: acxyle

    Task: (1) make the code clear and precise; (2) make the code computation and plot separate
    
    Status: under construction
    
    Update: basic function achieved, need to upgrade: (1) code clean; (2) parallel calculation

"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import logging

from tqdm import tqdm

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import label, generate_binary_structure  # the name [label] looks contradictory with many kinds of labels
from scipy.stats import ttest_ind
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist
from itertools import combinations

from scipy.io import loadmat, savemat

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from spikingjelly.activation_based import surrogate, neuron, functional

import utils_

class Selectiviy_Analysis_Feature():
    def __init__(self, root='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Results/',
                 dest='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/',
                 num_samples=10, num_classes=50, data_name='', layers=None, units=None, taskInstruction=None):
        
        if layers == None or units == None:
            raise RuntimeError('[Codwarning] invalid layers and units')
        
        # --- overall variables
        self.layers = layers
        self.units = units
        
        self.root = root
        self.dest = dest
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        self.Save_folder = os.path.join(self.dest, 'Feature')
        utils_.make_dir(self.Save_folder)
        
        # --- local variables
        self.taskInstruction = taskInstruction
        
        if self.taskInstruction == 'ImageNet':
            self.nSD = 1.8
            self.sq = 0.021
            self.maskFactor = 0.5
        elif self.taskInstruction == 'CoCo':
            self.nSD = 1.5
            self.maskFactor = 0.3
        elif self.taskInstruction == 'CelebA':
            self.nSD = 4
            self.sq = 0.035
            self.maskFactor = 0.1
            
    #FIXME
    # test version for one layer
    def data_preparation(self, layer, SM_selective_idx):
         
        feature = utils_.pickle_load(os.path.join(self.root, layer+'.pkl'))
        tSNE = loadmat(os.path.join(self.dest, f'TSNE/Results/tSNE_{layer}_all.mat'))[f'{layer}_all']
        
        self.population_folder = os.path.join(self.layer_folder, 'Population_Level')
        utils_.make_dir(self.population_folder)
        
        # lexicographic order
        #encode_id = loadmat(os.path.join(self.dest, f'encode_mat/{layer}_encode.mat'))['encodeID'].reshape(-1)
        
        #FIXME change the method later, now use it for temporal
        # the encode_id should be re-build from sensitive_unit_idx and SIMI_dict
        
        # --- test
        sensitive_unit_idx = np.loadtxt(os.path.join(self.dest, f'{layer}-neuronIdx.csv'), delimiter=',')
        sensitive_unit_idx = list(map(int, sensitive_unit_idx)) 
        
        encode_id = [[]]*feature.shape[1]
        for i in list(SM_selective_idx.keys()):
            encode_id[sensitive_unit_idx[i]] = SM_selective_idx[i]
        encode_id = [np.array(_, dtype=object) for _ in encode_id]    
        
        si_idx = np.array([i for i in range(len(encode_id)) if encode_id[i].size == 1])
        mi_idx = np.array([i for i in range(len(encode_id)) if encode_id[i].size > 1])
        # ---
        
        #  --- label correction
        id_label = self.lexicographic_order()
        img_label = np.array([[id_label[_]]*10 for _ in range(50)]).reshape(-1)
        for _ in range(len(encode_id)):
            correct_id = id_label[encode_id[_].reshape(-1).astype(int)-1]
            encode_id[_] = correct_id
        # ---
            
        #isFDR = 0      # [notice] wait to explore how this works later
        
        return feature, tSNE, encode_id, si_idx, mi_idx, id_label, img_label


    def feature_analysis(self):
        
        encode_class_dict = utils_.pickle_load(os.path.join(self.dest, 'Frequency/ID_neuron_encode_class_dict.pkl'))
        
        for idx, layer in enumerate(self.layers): 
            
            print(f'[Codinfo] Executing feature analysis of layer [{layer}]')
            
            num_units = self.units[idx]
            
            self.layer_folder = os.path.join(self.Save_folder, f'{layer}')
            utils_.make_dir(self.layer_folder)
            
            SM =  encode_class_dict[layer]
            SM_selective_idx = {**SM[2]['SI_idx'], **SM[3]['MI_idx']}
            SM_selective_idx = dict(sorted(SM_selective_idx.items()))
            
            feature, tSNE, encode_id, si_idx, mi_idx, id_label, img_label = self.data_preparation(layer, SM_selective_idx)
            self.feature_analysis_single_layer(tSNE, feature, np.arange(num_units), encode_id, si_idx, mi_idx, img_label, layer)
        
        print('[Codinfo] Completed')
        
    def feature_analysis_single_layer(self, tSNE, feature, unit_to_analyze, encode_id, si_idx, mi_idx, img_label, layer):    
        
        # first time consuming calculation
        p_values, kernel_size, kernel_sigma = self.generate_p_values(tSNE, feature, unit_to_analyze, layer)
        
        gaussian_kernel = self.gausskernel(kernel_size, kernel_sigma)
        
        x = tSNE[:, 0] - np.min(tSNE[:, 0])
        y = tSNE[:, 1] - np.min(tSNE[:, 1])
        
        fmi_idx, feature_idx, featured_img, featured_id, qualified_p_masks, preliminary_p_masks, list_pixels = self.feature_region_selection(p_values, tSNE, x, y, unit_to_analyze, encode_id, mi_idx, img_label, gaussian_kernel)

        #bionorP = 1 - binom.cdf(len(feature_idx), len(unit_to_analyze), 0.05)
        
        # ====== single_feature_unit_feature_coding analysis
        #FIXME, the units selection should be improved, implecate the intersection rule to si_unit either
        self.feature_coding_plot(si_idx, mi_idx, feature_idx, fmi_idx, feature, tSNE, x, y, img_label, qualified_p_masks, layer)
        # ======
        
        # ====== figures of population level
        # [notice] this step used nothing from self.feature_region_selection()
        # [idea] maybe those 2 functions can be merged as one?
        # [idea] the make all calculation can receive any input rather than special type of units/units
        
        unit_types_dict = self.unit_types(si_idx, mi_idx, fmi_idx, feature_idx, feature)
        
        # -----
        target_cluster_types = ['max', 'all']
        # [notice] for mi_nonf unit, that is possible to have no cluster, ALL si_nonf and non_id_nonf
        target_unit_types = ['fmi_idx', 'mi_nonf_idx', 'si_idx']
        empty_feature_map = np.zeros(qualified_p_masks[0].shape)
        
        feature_cluster_sizes, overlapped_pixel_map = self.population_feature_size(target_cluster_types[0], target_unit_types, unit_types_dict, si_idx, mi_idx, fmi_idx, list_pixels, empty_feature_map)
        
        # --- 1. size
        #self.plot_sizes_boxplot(feature_cluster_sizes, None, layer)
        
        # --- 2. distance
        groups = [fmi_idx, np.setdiff1d(mi_idx, fmi_idx)]
        
        distance_stats = self.calculate_distance(tSNE, encode_id, img_label, unit_to_analyze)  # this function can receive any units as the input, and one units generate one value
        groups_stats, group_stats = self.calculate_distance_ttest(groups, distance_stats['median'])  # groups_stats: group pairs; group_stats: single group stats
        
        # [notice] use one function to contain all plots for distance
        #self.plot_distance_figures(groups, distance_stats, group_stats,  layer)
        
        # --- 3. overlapped area
        self.plot_overlapped_receptive_field(target_unit_types, tSNE, x, y, feature_cluster_sizes, overlapped_pixel_map, layer)
        
    def plot_distance_figures(self, groups, distance_stats, stats, layer):
        
        save_folder = os.path.join(self.population_folder, 'Distance')
        utils_.make_dir(save_folder)
        
        fig, ax = plt.subplots(figsize=(10,10))
        self.plot_distance_bar(groups, stats, ax, save_folder, layer)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10,10))
        self.plot_distance_box(groups, distance_stats['median'], ax, save_folder, layer)
        plt.close()
    
    def plot_overlapped_receptive_field(self, target_unit_types, tSNE, x, y, feature_cluster_sizes, overlapped_pixel_map, layer=None):
        
        save_folder = os.path.join(self.population_folder, 'Overlapped_receptive_field')
        utils_.make_dir(save_folder)
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))       

        vmin = np.min([np.min(overlapped_pixel_map[_]) for _ in range(len(target_unit_types))])
        vmax = np.max([np.max(overlapped_pixel_map[_]) for _ in range(len(target_unit_types))])
        
        for labeled_p in range(len(target_unit_types)):  # for each unit
            cax = ax[labeled_p].imshow(overlapped_pixel_map[labeled_p], extent=[min(x), max(x), min(y), max(y)], aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
            ax[labeled_p].set_title(f"{np.sum(overlapped_pixel_map[labeled_p]>0)/overlapped_pixel_map[labeled_p].size*100:.2f}%")
            ax[labeled_p].set_xlabel(target_unit_types[labeled_p])
            ax[labeled_p].tick_params(axis='both', which='major', labelsize=12)
        
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(cax, cax=cbar_ax)
        fig.suptitle(f'{layer} overlapped receptive field')
        
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            
            fig.tight_layout(rect=[0, 0, 0.9, 1])
            plt.savefig(save_folder + f'/{layer}.png', bbox_inches='tight')
            plt.savefig(save_folder + f'/{layer}.eps', format='eps', bbox_inches='tight')
            plt.close()
            
    def plot_sizes_boxplot(self, feature_cluster_sizes:dict, interested_type=None, layer=None) -> None:
        """
            the input is a list in which each element contains numbers of area sizes of each unit of certain unit type,
            if the interested_type==None, then it will plot all combinations, be careful when too many types.
        """
        
        save_folder = os.path.join(self.population_folder, 'Size')
        utils_.make_dir(save_folder)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        flierprops = dict(marker='.', markerfacecolor='red', markersize=1, linestyle='none', markeredgecolor='red')
        ax.boxplot(list(feature_cluster_sizes.values()), notch=True, flierprops=flierprops)
        ax.set_xticklabels(list(feature_cluster_sizes.keys()))
        ax.set_xlabel(f'{layer}')
        ax.set_ylabel('Percentage of Feature Space(%)')
        ax.tick_params(axis='both', which='major', labelsize=12)
        #plt.grid(axis='y')
        
        group_idcs = np.arange(len(feature_cluster_sizes.keys()))
        groups = list(combinations(group_idcs, 2))
        
        if interested_type != None:
            if len(interested_type) == 1:
                groups = [np.array(_) for _ in groups if list(feature_cluster_sizes.keys()).index(interested_type) in _]
            else:
                raise RuntimeError('[Coderror] Do not support multiple [interested_type] in current code')
        
        p_value_list = []
        
        for idx, _ in enumerate(groups):
            ttest_stat, ttest_p = ttest_ind(list(feature_cluster_sizes.values())[_[0]], list(feature_cluster_sizes.values())[_[1]])
            p_value_list.append(ttest_p)
            groups[idx] = np.array(_) + 1
            
        utils_.sigstar(groups, p_value_list, ax)
        
        plt.savefig(save_folder + f'/{layer}.png', bbox_inches='tight')
        plt.savefig(save_folder + f'/{layer}.eps', format='eps', bbox_inches='tight')
        plt.close()
    
    def plot_distance_bar(self, groups, group_stats, ax, save_folder, layer=None):
        
        stats_mean = group_stats['mean']
        stats_std = group_stats['std']
        
        # Plotting bars
        b1 = ax.bar(1, stats_mean[0], color=[217/255, 83/255, 25/255], width=0.5)
        b2 = ax.bar(2, stats_mean[1], color=[0, 114/255, 189/255], width=0.5)
        
        # Adding error bars
        ax.errorbar(1, stats_mean[0], yerr=stats_std[0], fmt='.', capsize=8, linewidth=2, color='black')
        ax.errorbar(2, stats_mean[1], yerr=stats_std[1], fmt='.', capsize=8, linewidth=2, color='black')
        
        # Formatting plot
        ax.legend([b1, b2], ['Feature MI', 'non Feature MI'], frameon=False, loc='upper left')
        ax.set_xticks([1, 2], ['FeatureMI', 'Non-feature MI'])
        ax.set_ylabel('Normalized Distance')
        ax.set_title(f'{layer}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        name = 'fmi_vs_nonfmi'
        plt.savefig(save_folder + f'/{layer}_{name}_barplot.png', bbox_inches='tight')
        plt.savefig(save_folder + f'/{layer}_{name}_barplot.eps', format='eps', bbox_inches='tight')
        #plt.show()


    def plot_distance_box(self, groups, distances, ax, save_folder, layer=None):
        
        group_values = [distances[group] for group in groups]
        
        flierprops = dict(marker='.', markerfacecolor='red', markersize=1, linestyle='none', markeredgecolor='red')
        ax.boxplot(group_values, notch=True, flierprops=flierprops)
        
        # Formatting plot
        ax.set_xticks([1, 2], ['Feature MI', 'Non-feature MI'])
        ax.set_xlabel(f'{layer}')
        ax.set_ylabel('Normalized Distance')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        _, ttest_p = ttest_ind(group_values[0], group_values[1])
        utils_.sigstar([[1,2]], [ttest_p], ax)
        
        name = 'fmi_vs_nonfmi'
        plt.savefig(save_folder + f'/{layer}_{name}_boxplot.png', bbox_inches='tight')
        plt.savefig(save_folder + f'/{layer}_{name}_boxplot.eps', format='eps', bbox_inches='tight')
        #plt.show()
        
    #FIXME
    # [notice] this function should like the sigstar to receive any legth of input
    # [notice] one option is build a relationship between groups and distance_stats
    def calculate_distance_ttest(self, groups, distance_stats):
        """
            this function now receive any length of types of units and any types of input values, return a list contains all ttest_stats and ttest_pvalues
            
            groups: list of unit idx of different types of units
            distance_stats: list of values of different types of units
        """
        
        groups_stats = {}
        group_stats = {}
        
        ttest_stats_list = []
        ttest_pvalue_list = []
        
        group_pairs = list(combinations(np.arange(len(groups)), 2))
        
        for _ in group_pairs:
 
            group_a = distance_stats[groups[_[0]]]
            group_b = distance_stats[groups[_[1]]]

            ttest_stats, ttest_p = ttest_ind(group_a, group_b)
            
            ttest_stats_list.append(ttest_stats)
            ttest_pvalue_list.append(ttest_p)
    
        stats_mean = [np.mean(distance_stats[group]) for group in groups]
        stats_std = [np.std(distance_stats[group]) for group in groups]
        stats_sem = [np.std(distance_stats[group])/np.sqrt(len(group)-1) for group in groups]
        
        groups_stats['pairs'] = group_pairs
        groups_stats['stats'] = np.array(ttest_stats_list)
        groups_stats['pvalue'] = np.array(ttest_pvalue_list)
        
        group_stats['mean'] = stats_mean
        group_stats['std'] = stats_std
        group_stats['sem'] = stats_sem

        return groups_stats, group_stats
    
    def calculate_distance(self, tSNE, encode_id, img_label, target_units):      # try to make a figure to illustrate this?
        """
            这里最奇怪的问题，为什么不用 featured 而用 encoded
        """
        max_distance = pdist([[np.min(tSNE[:, 0]), np.min(tSNE[:, 1])], [np.max(tSNE[:, 0]), np.max(tSNE[:, 1])]], 'euclidean')     # normalization factor

        list_distance = []
        distance_stats = {}
        
        for unit in target_units:
            encoded_id = encode_id[unit]     # obtain the emcoded ID   [question] why not featured ID?
            
            if encoded_id.size != 0:     # check if this is a valid encoded unit/unit_typeron
                encoded_img = np.hstack([np.where(img_label==encoded_id[i])[0] for i in range(len(encoded_id))])
                #encoded_img = [idx for idx, val in enumerate(img_label) if val in encoded_id]   # obtain the idx of images
                
                point_wise_pairs = list(combinations(encoded_img, 2))
                
                tmp_distance = []
                for i in point_wise_pairs:
                    distance = pdist([tSNE[i[0]], tSNE[i[1]]], 'euclidean')
                    tmp_distance.append(distance)
                
                tmp_distance = np.array(tmp_distance)
                list_distance.append(tmp_distance)
                
            else:
                list_distance.append(np.array([0]))
        
        distance_stats['distances'] = np.array(list_distance, dtype='object')
        distance_stats['mean'] = np.array([np.mean(i)/max_distance for i in list_distance]).reshape(-1)
        distance_stats['std'] = np.array([np.std(i)/max_distance for i in list_distance]).reshape(-1)
        distance_stats['median'] = np.array([np.median(i)/max_distance for i in list_distance]).reshape(-1)

        return distance_stats
    
    def unit_types(self, si_idx, mi_idx, fmi_idx, feature_idx, feature):
        
        non_id_selective_idx = np.setdiff1d(np.arange(feature.shape[1]), np.union1d(mi_idx, si_idx))
        
        mi_nonf_idx = np.setdiff1d(mi_idx, fmi_idx)
        
        si_feature_idx = np.intersect1d(si_idx, feature_idx)
        si_non_feature_idx = np.setdiff1d(si_idx, feature_idx)
        
        non_id_feature_idx = np.intersect1d(feature_idx, non_id_selective_idx)
        non_id_non_feature_idx = np.setdiff1d(np.arange(feature.shape[1]), np.unique(np.concatenate([si_idx, mi_idx, feature_idx])))
        non_id_non_feature_idx = [idx for idx in non_id_non_feature_idx if not np.sum(feature[:, idx]) == 0 and np.sum(np.mean(feature[:,idx].reshape(-1,10), axis=1)!=0)>1]
        
        unit_types_dict = {
            'si_idx': si_idx,
            'mi_idx': mi_idx,
            'non_id_selective_idx': non_id_selective_idx,
            'fmi_idx': fmi_idx,
            'mi_nonf_idx': mi_nonf_idx,
            'si_feature_idx': si_feature_idx,
            'si_non_feature_idx': si_non_feature_idx,
            'non_id_feature_idx': non_id_feature_idx,
            'non_id_non_feature_idx': np.array(non_id_non_feature_idx)
            }
        
        return unit_types_dict
        
    def feature_coding_plot(self, si_idx, mi_idx, feature_idx, fmi_idx, feature, tSNE, x, y, img_label, qualified_p_masks, layer):
        
        print('[Codinfo] Executing feature unit plotting...')
        
        unit_types_dict = self.unit_types(si_idx, mi_idx, fmi_idx, feature_idx, feature)
        
        # colors
        colorppol = plt.get_cmap('tab20c', 60)
        colors = [colorppol(ii) for ii in range(50)]
       
        # Another kernel size and kernel sigma?
        #ff1 = 0.2
        #Ksd1 = 4
        #KS1 = [round((max(y1) - min(y1)) * ff1), round((max(x1) - min(x1)) * ff1)]

        #density_map, convolved_density_map = self.calculate_density_map(tSNE, x, y, FR=None, kernel=None)
        
        # Select cells to plot based on plot_type
        # === test
        
        k = 10
        plot_types_dict = {
            'fmi_idx': unit_types_dict['fmi_idx'][::k],
            'mi_nonf_idx': unit_types_dict['mi_nonf_idx'][::k],
            'si_feature_idx': unit_types_dict['si_feature_idx'][::k],
            'si_non_feature_idx': unit_types_dict['si_non_feature_idx'][::k],
            'non_id_feature': unit_types_dict['non_id_feature_idx'][::k],
            'non_id_non_feature': unit_types_dict['non_id_non_feature_idx'][::k],
        }
        
        plot_type_keys = list(plot_types_dict.keys())
        
        for plot_type in plot_type_keys:
            
            save_folder = os.path.join(self.layer_folder, plot_type)
            utils_.make_dir(save_folder)
            
            unit_to_plot = plot_types_dict.get(plot_type)
            
            #FIXME
            #self.plot_scatter(unit_to_plot, feature_idx, feature, x2, y2, colors, img_label, qualified_p_masks)
            
            self.plot_region_based_coding(plot_type, unit_to_plot, feature_idx, feature, colors, x, y, img_label, qualified_p_masks, save_folder, layer)
        
    def plot_region_based_coding(self, plot_type, unit_to_plot, feature_idx, feature, colors, x, y, img_label, qualified_p_masks, save_folder, layer=None):
        
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        for unit in tqdm(unit_to_plot, desc=f'{plot_type}'):
            
            #sigInd = np.where(feature_idx == unit)[0]  
            
            FR = feature[:, unit].astype(float)
            fig = plt.figure(figsize=(18, 9))
            #plt.annotate(f'FC_6 Unit: {unit}', (0.5, 0.98), xycoords='axes fraction', ha='center', fontsize=14, bbox=dict(boxstyle="square", ec="none", fc="white"))
            
            # ===== 1
            ax1_pos = [0.05, 0.1, 0.2, 0.8]
            ax_1 = plt.gcf().add_axes(ax1_pos)
            self.plot_distance_boxplot(ax_1, FR, img_label, colors)
            
            # ===== 2
            ax2_pos = [0.3, 0.1, 0.4, 0.8] 
            ax_2 = plt.gcf().add_axes(ax2_pos)
            self.plot_scatter_with_contour(ax_2, FR, x, y, img_label, colors, unit, feature_idx, qualified_p_masks[unit])
            
            # ===== 3
            ax3_upper_pos = [0.75, 0.55, 0.175, 0.35]
            ax3_lower_pos = [0.75, 0.1, 0.175, 0.35]
            ax_3_upper = plt.gcf().add_axes(ax3_upper_pos)
            ax_3_lower = plt.gcf().add_axes(ax3_lower_pos)
            pdfxy = self.kde_2d_v3(x, y, weights=FR)
            pdfPerm = self.kde_2d_perm(x, y, weights=FR)
            vmin, vmax = self.plot_kde(ax_3_upper, ax_3_lower, pdfxy, pdfPerm)     # [question] maerge the value range from all units?
            
            cmap = plt.get_cmap("viridis")
            norm_ = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_)
            sm.set_array([])  # Just a dummy array
            cbar_ax = fig.add_axes([0.95, 0.1, 0.0125, 0.8])
            fig.colorbar(sm, cax=cbar_ax)
            #cbar = plt.colorbar(cax1, ax=axes, orientation='vertical', fraction=0.02, pad=0.06)
            
            fig.suptitle(f'{layer} Unit: {unit}', y=0.95, fontsize=16)
            
            plt.savefig(save_folder + f'/{save_folder.split("/")[-1]}_{layer}_{unit}.png', bbox_inches='tight')
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                plt.savefig(save_folder + f'/{save_folder.split("/")[-1]}_{layer}_{unit}.eps', format='eps', bbox_inches='tight')
                plt.close()
            
    def plot_distance_boxplot(self, ax, FR, img_label, colors):
        
        bp = ax.boxplot([FR[img_label == i] for i in range(1,51)], vert=False, patch_artist=True, sym='+')
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set(edgecolor='none')

        threshold = np.mean(FR) + 2*np.std(FR)
        ax.vlines(threshold, 0, 52, colors='red', linestyles='-', linewidth=1.0, alpha=0.75)
        mean_list = np.mean(np.array([FR[img_label == i] for i in range(1,51)]), axis=1)

        encoded_idx = np.where(mean_list > threshold)[0]
        non_encoded_idx = np.setdiff1d(np.arange(50), encoded_idx)

        ax.scatter(mean_list[encoded_idx], encoded_idx+1, color='red', linewidth=0, alpha=0.5, label=r'$\overline{x}>V_{th}$', zorder=2)
        ax.scatter(mean_list[non_encoded_idx], non_encoded_idx+1, color='blue', linewidth=0, alpha=0.5, label=r'$\overline{x}<V_{th}$', zorder=2)

        for idx, _ in enumerate(mean_list):
            if idx in encoded_idx:
                ax.hlines(idx+1, 0, _, colors='orange', linestyles='--', linewidth=2.0, alpha=0.5)
            else:
                ax.hlines(idx+1, 0, _, colors='blue', linestyles='--', linewidth=2.0, alpha=0.5)

        ax.set_yticks(range(1, 51))
        ax.set_ylim([0,51])
        ax.set_yticklabels([str(i) for i in range(1, 51)])
        ax.set_xlabel('Response')

    def plot_scatter_with_contour(self, ax, FR, x, y, img_label, colors, unit, feature_idx, sig_pixel):
        
        if np.sum(FR) != 0:
            size_weight = FR / max(FR)     # [notice] can not divide by 0 if all values are 0
            sizes = np.ones(500) * 15 * (1 + 20 * size_weight)
            
            handles_not_featured = []
            handles_featured = []
            labels = []
            
            featured_img_idx = []
            for _ in range(len(x)):     # for each point
                if sig_pixel[round(y[_]-1), round(x[_]-1)] == 1:
                    featured_img_idx.append(_)
            featured_id_idx = np.unique(img_label[featured_img_idx])
            not_featured_id_idx = np.setdiff1d(np.arange(50)+1, featured_id_idx)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                for gg in not_featured_id_idx:  # this can be changed to different types of id
                    current_scatter = ax.scatter(x[img_label == gg], y[img_label == gg], s=sizes[img_label == gg], color=colors[gg-1], alpha=0.5)
                    handles_not_featured.append(current_scatter)
                    
                for gg in featured_id_idx:
                    label_text = f'{gg}'
                    current_scatter = ax.scatter(x[img_label == gg], y[img_label == gg], s=sizes[img_label == gg], color=colors[gg-1], alpha=0.7)
                    handles_featured.append(current_scatter)
                    labels.append(label_text)
                
                if unit in feature_idx: 
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        contours = ax.contour(sig_pixel, [], colors='c')
                
                ax.legend(handles=handles_featured, labels=labels)
                
        else:
            for gg in range(1,51):  # this can be changed to different types of id
                current_scatter = ax.scatter(x[img_label == gg]-1, y[img_label == gg]-1, s=5, color=colors[gg-1], alpha=0.5)
        
        ax.set_xlabel('Feature Dimension 1')
        ax.set_ylabel('Feature Dimension 2')

    def plot_kde(self, ax_up, ax_down, pdfxy, pdfPerm):
        
        vmin = np.min([np.min(pdfxy), np.min(pdfPerm)])
        vmax = np.max([np.max(pdfxy), np.max(pdfPerm)])
        
        ax_up.imshow(pdfxy, origin='lower', vmin=vmin, vmax=vmax)
        ax_up.set_xticks([])
        ax_up.set_yticks([])
        ax_up.set_xlabel('Feature Dimension 1')
        ax_up.set_ylabel('Feature Dimension 2')
        ax_up.set_title('observed density map')
        
        ax_down.imshow(pdfPerm, origin='lower', vmin=vmin, vmax=vmax)
        ax_down.set_xticks([])
        ax_down.set_yticks([])
        ax_down.set_xlabel('Feature Dimension 1')
        ax_down.set_ylabel('Feature Dimension 2')
        ax_down.set_title('mean permuted density map')
        
        return vmin, vmax

    def kde_2d_perm(self, x, y, bw=None, weights=None, isplot=False):
 
        pdfPerm = []
    
        for ii in range(1000):
            sData = weights[np.random.permutation(len(weights))]
            pdf_xy = self.kde_2d_v3(x, y, bw=bw, weights=sData)
            pdfPerm.append(pdf_xy)

        pdfPerm = np.mean(np.array(pdfPerm), axis=0)

        return pdfPerm
          
    def kde_2d_v3(self, x, y, bw=None, weights=None, isplot=False, plot_scale=100):
        pdfx = self.ksdensity(x, bw, weights)
        pdfy = self.ksdensity(y, bw, weights)
        
        pdfx = pdfx(np.linspace(min(x), max(x), plot_scale))
        pdfy = pdfy(np.linspace(min(y), max(y), plot_scale))

        pdfx, pdfy = np.meshgrid(pdfx, pdfy)
        pdfxy = pdfx * pdfy
        
        return pdfxy
        
    def ksdensity(self, data, bw=None, weights=None):
        if weights is not None and np.sum(weights) != 0 and len(np.where(weights!=0)[0])!=1:
            ksdensity = gaussian_kde(data, bw_method=bw, weights=weights)
        else:
            ksdensity = gaussian_kde(data, bw_method=bw, weights=None)
        return ksdensity
    
    def plot_scatter(self, ax, unit_to_plot, feature_idx, feature, x, y, colors, img_label, sig_pixel_clean):
        
        for celltoplot_idx in unit_to_plot:
            iCell = celltoplot_idx
        
            # Find if the cell is in feature_idx
            #sigInd = np.where(feature_idx == iCell)[0]
        
            FR = feature[:, iCell].astype(float)
        
            size_weight = FR / max(FR)
            sizes = np.ones(500) * 15 * (1 + 20 * size_weight)
            handles = []
        
            for gg in range(1, 51):  # Python's range starts at 0, so adjust alabeled_pordingly
                current_scatter = ax.scatter(x[img_label == gg], y[img_label == gg], s=sizes[img_label == gg], color=colors[gg-1], alpha=0.7)
                handles.append(current_scatter)
            
            if iCell in feature_idx:  # or if len(sigInd) > 0:
                contours = ax.contour(sig_pixel_clean[iCell], [1], colors='c')
                for collection in contours.collections:
                    collection.set_linewidth(3)
                    
    # -----
    
    def population_feature_size(self, target_cluster_type, target_unit_types, unit_types_dict, si_idx, mi_idx, fmi_idx, list_pixels, mask):
        """
            this function calculates (1) the average cluster size and (2) overlapped size for one type of units
        """
        
        map_size = mask.size
    
        #meanSize = []
        #stdSize = []
        feature_cluster_sizes = {}
        overlapped_pixel_map = []
        #feature_cluster_sizes = []
    
        for unit_type in target_unit_types:     # for each type of units
            unit_idx = unit_types_dict[unit_type]
    
            unit_cluster_size = []
            overlapped_pixels = np.zeros(mask.shape)
            
            for unit in tqdm(unit_idx, desc=f'Feature Size of[{unit_type}]'):     # for each unit
                self.single_unit_cluster_size_calculation(target_cluster_type, unit, unit_cluster_size, list_pixels[unit], map_size, overlapped_pixels)
                
            feature_cluster_sizes[unit_type] = np.array(unit_cluster_size)   
            
            #meanSize.append(np.mean(unit_cluster_size))
            #stdSize.append(np.std(unit_cluster_size))
            
            #FIXME
            overlapped_pixel_map.append(overlapped_pixels)
            #feature_cluster_sizes.append(np.sum(overlapped_pixels >= 1) / map_size)      # [question] and only count the covered area? no more conditions?
    
        return feature_cluster_sizes, overlapped_pixel_map
             
    def single_unit_cluster_size_calculation(self, target_cluster_type, unit, unit_cluster_size, unit_clusters, map_size, overlapped_pixels):
        
        if len(unit_clusters) != 0:      # prevent unit  without useful cluster
            if target_cluster_type == 'max':     # only count the biggest cluster
                unit_cluster_size.append(unit_clusters[0][1] / map_size * 100)     # [unit_idx][biggest_cluster_idx][cluster_size]
                for pix in unit_clusters[0][2].T:     
                    overlapped_pixels[pix[0],  pix[1]] += 1
            
            elif target_cluster_type == 'all':     # count all qualified clusters
                for cluster in unit_clusters:
                    unit_cluster_size.append(cluster[1] / map_size * 100)
                    for pix in cluster[2].T:
                        overlapped_pixels[pix[0],  pix[1]] += 1
        else:
            unit_cluster_size.append(0)
    
    def feature_region_selection(self, p, tSNE, x, y, target_units, encode_id, mi_idx, img_label, kernel, cluster_size_scaling_factor=0.025, alpha=0.01):
        
        density_map, convolved_density_map = self.calculate_density_map(tSNE, x, y, None, kernel)
        
        mask = convolved_density_map >= (self.maskFactor * np.mean(convolved_density_map))
        #mask = convolved_density_map >= 0.05
        
        cluster_size_threshold = mask.size * cluster_size_scaling_factor

        # ----- results storage
        preliminary_p_masks = np.zeros((len(target_units), mask.shape[0], mask.shape[1]))     # pixels satisfy 2 masks
        qualified_p_masks = np.zeros((len(target_units), mask.shape[0], mask.shape[1]))     # pixels satisfy conditions of feature unit
        
        featured_img = [[]]*len(target_units)    # featured imgs of given unit
        featured_id = [[]]*len(target_units)     # featured id of given unit
        
        feature_idx = np.zeros(len(target_units))     # index of feature unit
        fmi_idx = np.zeros(len(target_units))     # index of feature-multi-selective unit
        
        list_pixels = [[]]*len(target_units) 
        # -----
        
        # ===
        #FIXME
        # --- 1. Thread/ProcessPool     - why very slow? 竞性条件?序列等待?
        #executor = ProcessPoolExecutor(max_workers=os.cpu_count())
        #job_pool = []
        #for unit in tqdm(target_units, desc='Submit'):     
        #    job = executor.submit(self.feature_region_selection_single_unit, p, x, y, unit, img_label, encode_id, mi_idx, alpha, mask, cluster_size_threshold, preliminary_p_masks, qualified_p_masks, featured_img, featured_id, feature_idx, fmi_idx, list_pixels)
        #    job_pool.append(job)
        #for unit in tqdm(target_units, desc='Collect'):
        #    job_pool[unit].result()
        #executor.shutdown()
        
        # --- 2. Sequential
        for unit in tqdm(target_units, desc='region selection'):
            self.feature_region_selection_single_unit(p, x, y, unit, img_label, encode_id, mi_idx, alpha, mask, cluster_size_threshold, preliminary_p_masks, qualified_p_masks, featured_img, featured_id, feature_idx, fmi_idx, list_pixels)
        
        # ===
        feature_idx = np.array(np.where(feature_idx == 1)).reshape(-1)     # index of feature unit
        fmi_idx = np.array(np.where(fmi_idx == 1)).reshape(-1)
        
        return np.unique(np.array(fmi_idx)), np.unique(np.array(feature_idx)), np.array(featured_img, dtype=object), np.array(featured_id, dtype=object), qualified_p_masks, preliminary_p_masks, list_pixels

    def feature_region_selection_single_unit(self, p, x, y, unit, img_label, encode_id, mi_idx, alpha, mask, cluster_size_threshold, preliminary_p_masks, qualified_p_masks, featured_img, featured_id, feature_idx, fmi_idx, list_pixels):
        
        p_regions = (p[unit]<alpha)*mask
        preliminary_p_masks[unit] = p_regions

        labeled_p, num_component = label(p_regions, structure=generate_binary_structure(2,2)) 
        
        if num_component > 0:     # if the unit has connected component
           
            sig_id = []     # ID passed 2 conditions
            sig_img = []    

            for i in range(1, num_component+1):     # for each component
                
                component = (labeled_p == i)
            
                if np.sum(component) < cluster_size_threshold:     # condition 1
                    p_regions[component] = 0
                    labeled_p[component] = 0
                
                else:
                    
                    tmp_sig_img = [_ for _ in range(len(x)) if labeled_p[int(y[_]-1), int(x[_]-1)] == i]
                    tmp_sig_id = np.unique(img_label[tmp_sig_img])

                    if len(tmp_sig_id) < 2 or len(tmp_sig_img) < 5:     # condition 2
                        p_regions[component] = 0
                        labeled_p[component] = 0
                    else:
                        feature_idx[unit] = 1     # need to use np.unique() to remove replicated idx
                        
                        sig_id.append(tmp_sig_id.reshape(-1))
                        sig_img.append(np.array(tmp_sig_img).reshape(-1))
                        
                        if len(set(tmp_sig_id) & set(list(encode_id[unit].reshape(-1)))) > 1 and (unit in mi_idx):     # need to use np.unique() 
                            fmi_idx[unit] = 1
            
            pixNum = [[j, np.sum(labeled_p == j), np.array(np.where(labeled_p==j))] for j in range(1, num_component+1) if np.sum(labeled_p == j) != 0]
            pixNum = list(sorted(pixNum,  key=lambda x:x[1], reverse=True))
            
            list_pixels[unit] = pixNum

            qualified_p_masks[unit] = p_regions
            featured_img[unit] = np.array(sig_img, dtype=object)
            featured_id[unit] = np.array(sig_id, dtype=object)

        else:
            # when use parallel operation, this branch is not in use
            pass
            
    
    def lexicographic_order(self):
        id_order = np.arange(1,1+self.num_classes).astype(str)
        id_order_idx = np.argsort(id_order)
        id_order_lexical = id_order[id_order_idx].astype(int)
        
        return id_order_lexical

    
    def generate_p_values(self, tSNE, feature, unit_idx, layer=None):
        """
            this function is the parallel executor of calculate_perm_density_p_value() to obtain p values for all units
        """
        
        print(f'[Codinfo] Executing p_value generation of layer [{layer}]')
    
        file_path = os.path.join(self.layer_folder, f'{layer}_sq{self.sq}.pkl')
        
        if os.path.exists(file_path):
            results = utils_.pickle_load(file_path)
            p = results['p']
            kernel_size = results['kernel_size']
            kernel_sigma = results['kernel_sigma']
           
        else:
            
            kernel_size, kernel_sigma = self.get_kernel_size(tSNE)
            gaussian_kernel =  self.gausskernel(kernel_size, kernel_sigma)
            
            x = tSNE[:, 0] - np.min(tSNE[:, 0])
            y = tSNE[:, 1] - np.min(tSNE[:, 1])
            empty_map = np.ones((int(np.ceil(np.max(y)+1)), int(np.ceil(np.max(x)+1))))
            
            # ----- parallel computing -----
            p = [[]]*len(unit_idx)
            executor = ProcessPoolExecutor(max_workers=os.cpu_count())
            job_pool = []
            for i in tqdm(range(len(unit_idx)), desc=f'[{layer}] submit'):
                job = executor.submit(self.calculate_p_values_parallel, i, tSNE, x, y, feature, empty_map, 1000, gaussian_kernel)
                job_pool.append(job)
            for i in tqdm(range(len(unit_idx)), desc=f'[{layer}] collect'):
                p[i] = job_pool[i].result()
            executor.shutdown()
            # -----
            
            # ----- sequential computing
            #p = []
            #for i in tqdm(range(len(unit_idx)), desc='Sequential'):
            #    FR = feature[:,i]
            #    p_tmp, *_ = self.calculate_perm_density_p_value(tSNE, x, y, FR, empty_map, 1000, gaussian_kernel)
            #    p.append(p_tmp)
            # -----
                
            results = {'p': p, 'sq': self.sq, 'kernel_size': kernel_size, 'kernel_sigma': kernel_sigma, 'tSNE': tSNE, 'layer': layer}
            utils_.pickle_dump(file_path, results)
            
        return p, kernel_size, kernel_sigma
        
    def calculate_p_values_parallel(self, i, tSNE, x, y, feature, empty_map, perm=None, kernel=None):
        FR = feature[:, i]
        p_tmp, *_ = self.calculate_perm_density_p_value(tSNE, x, y, FR, empty_map, perm, kernel)
        return p_tmp
        
    
    def create_empty_density_map(self, tSNE, x, y):

        imgW = int(np.ceil(np.max(x)))+1
        imgH = int(np.ceil(np.max(y)))+1
    
        density_map = np.zeros((imgH, imgW))
        
        return density_map
    
    def calculate_2d_density_map(self, tSNE, x, y, FR=None):
        """
            this function returns the 2d density map based on calculation of weighted points on the grid, 
            [note] if the values of FR are all 0, it will generate a blank density map
        """
        if FR is None:
            FR = np.ones(500)
        
        density_map = self.create_empty_density_map(tSNE, x, y)
    
        for i in range(len(x)):
            density_map[round(y[i])-1, round(x[i])-1] += FR[i]

        return density_map
        
    def calculate_perm_density_p_value(self, tSNE, x, y, FR=None, empty_map=None, num_perm=1000, kernel=None):
        """
            this function is an advanced wrap of calculate_density_map(), added permutation test to generate p_value
        """
        if np.sum(FR) != 0 and np.sum(FR!=0) > 1:     # without consideration of units with all 0 values
            density_map, convolved_density_map = self.calculate_density_map(tSNE, x, y, FR, kernel)
        
            permutation_stats = np.zeros(density_map.shape)
        
            perm_density_maps = []
            perm_convolved_density_maps = []
        
            for ii in range(num_perm):
                N = np.random.permutation(len(FR))
                permuted_FR = FR[N]
                
                perm_density_map, perm_convolved_density_map = self.calculate_density_map(tSNE, x, y, permuted_FR, kernel)
                permutation_stats += perm_convolved_density_map > convolved_density_map
                
                perm_density_maps.append(perm_density_map)
                perm_convolved_density_maps.append(perm_convolved_density_map)
        
            p = permutation_stats / num_perm
        
        else:
            p = empty_map
            perm_density_maps = []
            perm_convolved_density_maps = []
        
        return p, np.array(perm_density_maps), np.array(perm_convolved_density_maps)
    
    #FIXME
    def calculate_perm_density_map_stats(self, perm_density_maps, perm_convolved_density_maps):
        meanum_permZ = np.mean(perm_density_maps, axis=0)
        meanum_permZC = np.mean(perm_convolved_density_maps, axis=0)
    
    def calculate_density_map(self, tSNE, x, y, FR=None, kernel=None):
        """
            this function is a simple wrap of calculate_2d_density_map(), added the smoothed (convolved) density map
        
        """
        density_map = self.calculate_2d_density_map(tSNE, x, y, FR)
        convolved_density_map = convolve(density_map, kernel, mode='constant')

        return density_map, convolved_density_map
        
    def get_kernel_size(self, tSNE):
        """
            this function calculate the kernel_size and sigma for the following calculation on p values
            
            In fact, the function here is not the same one described in the paper. 
            (1) The method here calculate kernel_sigma and kernel_size based on 'the number of 
            connected components', and use a scaling factor: self.sq=0.035 to 
            decide the value of kernel_sigma(sigma, which decides the speed of decrease)
            -> kernel_sigma(sigma) = sq*num_component, KS = 2*radius(3*sigma)+1
            (2) The method described in paper is based on 'the size of feature map',
            and use another scaling factor: ff1=0.2(in self.feature_coding_plot()) to
            decide the kernel size -> KS = ff1*[ylim,xlim]
            
            x: first dimension of tSNE tSNE;
            y: second dimension of tSNE tSNE
            self.sq: an empirical scale factor to decide the sigma of the gaussian filter. default to be 0.035, (close to sd = 4, used in previous expriments)
        """
        x = tSNE[:, 0] - np.min(tSNE[:, 0])
        y = tSNE[:, 1] - np.min(tSNE[:, 1])
        
        density_map = self.calculate_2d_density_map(tSNE, x, y)
                
        labeled, num_component = label(density_map, structure=generate_binary_structure(2,2))  
        kernel_sigma = num_component * self.sq  # determine the sigma
        
        # decide the kernel size
        #kernel_size_y = int(2 * 3 * (np.floor(kernel_sigma)) + 1)  
        kernel_size_y = int(np.floor(2 * 3 * kernel_sigma + 1))
        kernel_size_x = int(np.floor(kernel_size_y * density_map.shape[1] / density_map.shape[0]))
        
        kernel_size = [kernel_size_y, kernel_size_x]
    
        return kernel_size, kernel_sigma
    
    #FIXME
    # [notice] now a severe problem is the function calculate every time in each density calculation, remove it from those function
    def gausskernel(self, R, S):
        """
            Creates a discretized N-dimensional Gaussian kernel.
            R: kernel size (pixels in one side)
            S: standard deviation
            [note] in current use, the R is a 2-D vector and S is a scalar
        """
    
        # Check Inputs
        R = np.asarray(R)
        S = np.asarray(S)
        D = R.size
        D2 = S.size
    
        if ((D > 1 and R.ndim != 1) or (D2 > 1 and S.ndim != 1)):
            raise ValueError('Matrix arguments are not supported.')
    
        if ((D > 1 and D2 > 1) and (D != D2)):
            raise ValueError('R and S must have same number of elements (unless one is scalar).')
    
        # Force bins/sigmas 
        if (D2 > D):  
            D = D2  
            R = R * np.ones(D) 
    
        # To be same length
        if (D > D2): 
            S = S * np.ones(D)  
    
        # And force row vectors
        R = R.flatten()
        S = S.flatten()
    
        # Make the Kernel
        kernel = None
        
        RR = 2*R+1
        for k in range(D):
           
            grid = np.arange(-R[k], R[k] + 1)
            gauss = np.exp(-grid**2 / (2 * S[k]**2))
            gauss = gauss / np.sum(gauss)  # normalization
    
            # [note] this function is only valid for current 2D kernel, higher dimension needs more experiments
            if (k == 0):
                kernel = gauss
            else:
                Dpast = [1] * k
                expand = np.tile(gauss.reshape(1,-1), [*RR[0:k], 1])
                kernel = np.tile(kernel.reshape(-1,1), [*Dpast, RR[k]]) * expand
    
        return kernel





if __name__ == '__main__':
    
    #layers = ['Pool_5', 'Conv_5_3']
    #FIXME
    # make sure the function  of units here, as a check,  not only an input
    #units = [25088, 100352]
    #root_dir = '/media/acxyle/Data/ChromeDownload/'
    #selectivity_feature_analyzer = Selectiviy_Analysis_Feature(
    #            root=os.path.join(root_dir, 'Identity_VGG_Feature_Results/'), 
    #            dest=os.path.join(root_dir, 'Identity_VGG_Feature_Original/'), 
    #            layers=layers, units=units, taskInstruction='CelebA')
    #selectivity_feature_analyzer.feature_analysis()
    
    #neuron_ = neuron.LIFNode
    #spiking_model = spiking_vgg.__dict__['spiking_vgg16_bn'](spiking_neuron=neuron_, num_classes=50, surrogate_function=surrogate.ATan(), detach_reset=True)
    #functional.set_step_mode(spiking_model, step_mode='m')
    #layers, neurons, shapes = utils_.generate_vgg_layers(spiking_model, 'spiking_vgg16_bn')
    
    layers = ['L4_B1_neuron01', 'L4_B2_neuron02', 'avgpool', 'fc']
    neurons  = [25088, 25088, 512, 50]

    root_dir = '/media/acxyle/Data/ChromeDownload/'

    selectivity_feature_analyzer = Selectiviy_Analysis_Feature(
                root=os.path.join(root_dir, 'Identity_SpikingResnet18_LIF_CelebA2622_Results/'), 
                dest=os.path.join(root_dir, 'Identity_SpikingResnet18_LIF_CelebA2622_Neuron/'), 
                layers=layers, units=neurons, taskInstruction='CelebA')
    
    selectivity_feature_analyzer.feature_analysis()
