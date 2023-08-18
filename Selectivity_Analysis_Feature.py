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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import utils_

class Selectiviy_Analysis_Feature():
    def __init__(self, root='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Results/',
                 dest='/media/acxyle/Data/ChromeDownload/Identity_Spikingjelly_VGG_Neuron/',
                 num_samples=10, num_classes=50, data_name='', layers=None, neurons=None, taskInstruction=None):
        
        if layers == None or neurons == None:
            raise RuntimeError('[Codwarning] invalid layers and neurons')
        
        # --- overall variables
        self.layers = layers
        self.neurons = neurons
        
        self.root = root
        self.dest = dest
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        # --- local variables
        self.taskInstruction = taskInstruction
        
        if self.taskInstruction == 'ImageNet':
            self.nSD = 1.8
            self.sq = 0.021
            self.maskFactor = 2
        elif self.taskInstruction == 'CoCo':
            self.nSD = 1.5
            self.maskFactor = 3
        elif self.taskInstruction == 'CelebA':
            self.nSD = 4
            self.sq = 0.035
            self.maskFactor = 2
        
            
    # test version for one layer
    def load_useful_data(self):
        
        feature = utils_.pickle_load(os.path.join(self.root, layers[0]+'.pkl'))
        #encode_class_dict = utils_.pickle_load(os.path.join(self.dest, 'SIMI/SIMI_cnt.pkl'))
        
        # lexicographic order
        encode_id = loadmat(os.path.join(self.dest, 'encode_mat/FC_6_encode.mat'))['encodeID'].reshape(-1)
        
        si_idx = np.array([i for i in range(len(encode_id)) if encode_id[i].shape[1] == 1])
        mi_idx = np.array([i for i in range(len(encode_id)) if encode_id[i].shape[1] > 1])
        
        tSNE = loadmat(os.path.join(self.dest, 'TSNE/tSNE_FC_6_all.mat'))['FC_6_all']
        
        unit_to_analyze = np.arange(4096)
        
        self.StatsDir = os.path.join(self.dest, 'Feature')
        utils_.make_dir(self.StatsDir)
        
        # =====
        # [notice] p_values varies slightly because permutations here
        # -----
        # [notice] in below code, this should use parallel operation because this is very slow
        p_values, KS, Ksd = self.generate_p_value(tSNE, feature, unit_to_analyze)

        # -----
        
        # [notice] if the p_value generated locally, then the KS and Ksd also need to be obtain from local functions
        #p_file = loadmat('/home/acxyle/Downloads/osfstorage-archive-supp/Res/DensityStats/CelebA/VGG16/FC_6_Sq035.mat')  # <- this is the key of this function
        #p_values = p_file['p'].reshape(-1) 
        #KS = p_file['KS'].reshape(-1).astype(int)
        #Ksd = p_file['Ksd'].reshape(-1)[0]
        # -----
        # =====
        
        #isFDR = 0      # [notice] wait to explore how this works later
        
        # --- order correction
        id_label = self.lexicographic_order()
        img_label = np.array( [[id_label[_]]*10 for _ in range(50)] ).reshape(-1)
        
        for _ in range(len(encode_id)):
            correct_id = id_label[encode_id[_].reshape(-1).astype(int)-1]
            encode_id[_] = correct_id
        # ---
    
        feature_idx, sigP_clean, fmi_idx, InCludeFace, InCludeID, InCludePix, maskLevel, clusterSize = self.region_sel(p_values, tSNE, unit_to_analyze, encode_id, mi_idx, img_label, KS, Ksd)
        
        #bionorP = 1 - binom.cdf(len(feature_idx), len(unit_to_analyze), 0.05)
        
        # ======
        #self.single_neuron_plot(si_idx, mi_idx, feature_idx, fmi_idx, feature, tSNE, img_label, sigP_clean)
        # ======
        
        # =====================================================================
        # [notice] figures of population level
        self.population_folder = os.path.join(self.StatsDir, 'Population_Level')
        utils_.make_dir(self.population_folder)
        
        # [notice] this step used nothing from self.region_sel()
        # [notice] this calculate the distance
        # [idea] maybe those 2 functions can be merged as one?
        # [idea] the make all calculation can receive any input rather than pecial type of units/neurons
        

        tls = ['fmi_idx', 'nonfMIInd', 'si_idx']
        
        InCludePixNum, InCludePix, mask, valVox = self.matlab_conversion(tSNE, p_values, feature, KS=KS, Ksd=Ksd)
        meanSize, stdSize, EncodeSize, RP, groupSize = self.calculate_sizes(tls, si_idx, mi_idx, fmi_idx, InCludePixNum, InCludePix, valVox, mask)
        
        # --- 1. size
        #self.plot_sizes_boxplot(EncodeSize)
        
        # --- 2. distance
        groups = [fmi_idx, np.setdiff1d(mi_idx, fmi_idx)]
        
        tDis_stats = self.calculate_distance(tSNE, encode_id, img_label, np.arange(4096))  # this function can receive any units as the input, and one units generate one value
        groups_stats, group_stats = self.calculate_distance_ttest(groups, tDis_stats['median'])  # groups_stats: group pairs; group_stats: single group stats
        
        # [notice] use one function to contain all plots for distance
        self.plot_distance_figures(groups, tDis_stats, group_stats)
        
        # --- 3. overlapped area
        #self.plot_overlapped_receptive_field(tls, tSNE, RP, groupSize, Ksd)
        
        print('[Codinfo] Completed')
        
    def plot_distance_figures(self, groups, tDis_stats, stats):
        
        save_folder = os.path.join(self.population_folder, 'Distance')
        utils_.make_dir(save_folder)
        
        fig, ax = plt.subplots(figsize=(10,10))
        self.plot_distance_bar(groups, stats, ax, save_folder)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10,10))
        self.plot_distance_box(groups, tDis_stats['median'], ax, save_folder)
        plt.close()
    
    def plot_overlapped_receptive_field(self, tls, tSNE, RP, groupSize, Ksd):
        
        save_folder = os.path.join(self.population_folder, 'Overlapped_receptive_field')
        utils_.make_dir(save_folder)
        
        layer = 'FC_6'
        
        fig, axarr = plt.subplots(1, 3, figsize=(15, 4))
        plotInd = 0
    
        x, y = tSNE[:,0], tSNE[:, 1]
        ff = 0.2
        KS = [round((max(y)-min(y))*ff), round((max(x)-min(x))*ff)]     # this KS is [8,8] but another method generates KS as [7,7]
        
        _, _, x2, y2 = self.Cal_Density(tSNE, np.ones(tSNE.shape[0]), KS, Ksd)     # weights all 1
        
        vmin = float('inf')
        vmax = float('-inf')
    
        # First, determine the global color limits across all plots
        for cc in range(3):
            vmin = min(vmin, np.min(RP[cc]))
            vmax = max(vmax, np.max(RP[cc]))
        
        for cc in range(3):  # for each unit
            ax = axarr[plotInd]
            cax = ax.imshow(RP[cc], extent=[min(x2), max(x2), min(y2), max(y2)], aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
            ax.set_title(f"{round(groupSize[cc]*100, 2)}%")
            ax.set_xlabel(tls[cc])
            ax.tick_params(axis='both', which='major', labelsize=12)
            plotInd += 1
        
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(cax, cax=cbar_ax)
    
        fig.tight_layout(rect=[0, 0, 0.9, 1])
        
        plt.savefig(save_folder + f'/{layer}.png', bbox_inches='tight')
        plt.savefig(save_folder + f'/{layer}.eps', format='eps', bbox_inches='tight')
        
    def plot_sizes_boxplot(self, EncodeSize):
        
        save_folder = os.path.join(self.population_folder, 'Size')
        utils_.make_dir(save_folder)
        
        layer = 'FC_6'
        
        fig, ax = plt.subplots(figsize=(8, 5))
        flierprops = dict(marker='.', markerfacecolor='red', markersize=1, linestyle='none', markeredgecolor='red')
        ax.boxplot(EncodeSize, notch=True, flierprops=flierprops)
        ax.set_xticklabels(['Feature MI', 'Non-feature MI', 'SI'])
        ax.set_xlabel(f'{layer}')
        ax.set_ylabel('Percentage of Feature Space(%)')
        ax.tick_params(axis='both', which='major', labelsize=12)
        #plt.grid(axis='y')
        
        groups = [[0,1],[0,2]]
        p_value_list = []
        
        for idx, _ in enumerate(groups):
            ttest_stat, ttest_p = ttest_ind(EncodeSize[0], EncodeSize[1])
            p_value_list.append(ttest_p)
            groups[idx] = np.array(_) + 1
            
        utils_.sigstar(groups, p_value_list, ax)
        
        plt.savefig(save_folder + f'/{layer}.png', bbox_inches='tight')
        plt.savefig(save_folder + f'/{layer}.eps', format='eps', bbox_inches='tight')
    
    def calculate_sizes(self, tls, si_idx, mi_idx, fmi_idx, InCludePixNum, InCludePix, valVox, mask):
        
        nonfMIInd = list(set(mi_idx) - set(fmi_idx))
    
        meanSize = []
        stdSize = []
        EncodeSize = []
        RP = []
        groupSize = []
    
        for neu in tls:
            if neu == 'fmi_idx':
                useInd = fmi_idx
            elif neu == 'nonfMIInd':
                useInd = nonfMIInd
            elif  neu  == 'si_idx':
                useInd = si_idx
    
            tmpRP = np.zeros(mask.shape)
    
            tmpEncodeSize = []
            for iCell in useInd:
                
                tmpEncodeSize.append(InCludePixNum[iCell] / valVox * 100)
                
                for pix in InCludePix[iCell].T:
                    tmpRP[pix[0],  pix[1]] += 1
    
            meanSize.append(np.mean(tmpEncodeSize))
            stdSize.append(np.std(tmpEncodeSize) / len(useInd))
            EncodeSize.append(tmpEncodeSize)
            RP.append(tmpRP)
            groupSize.append(np.sum(tmpRP >= 1) / valVox)
    
        return meanSize, stdSize, EncodeSize, RP, groupSize

    
    # [notice] looks like this code is something similiar with below codes
    # [notice] the function is region_sel()
    def matlab_conversion(self, tSNE, p_values, feature, KS=None, Ksd=None):
        Z1, ZC1, x2, y2 = self.Cal_Density(tSNE, np.ones(len(tSNE)), KS, Ksd)
        
        maskLevel = 0.05
        mask = ZC1 >= maskLevel
        
        valVox = mask.shape[0] * mask.shape[1]
        sigP_mask = []
        InCludePixNum = []
        InCludePix = []
        
        for iCell in range(len(p_values)):
            wData = feature[:, iCell].astype(float)
            
            # Assuming p[iCell] is a numpy array or similar
            tmpP = p_values[iCell]
            sigP = tmpP <= 0.01
            tmp_sigP_mask = sigP*mask
            sigP_mask.append(tmp_sigP_mask)
            
            # Connected components
            s = generate_binary_structure(2,2)
            labeled_array, nComp = label(tmp_sigP_mask, structure=s) 
            
            #FIXME
            # need to remove the impact of empty (mask) region
            if nComp > 0:
                # List of sizes of each component
                pixNum = [[i, np.sum(labeled_array == i)] for i in range(1, nComp+1)]
                maxInd = sorted(pixNum,  key=lambda x:x[1], reverse=True)[0][0]
                
                tmpP1 = np.copy(tmp_sigP_mask)
                SigPixel = np.array(np.where(labeled_array == maxInd))
                
                non_sig_cluster = np.arange(1,nComp+1)
                non_sig_cluster = non_sig_cluster[non_sig_cluster!=maxInd]
                
                for i in non_sig_cluster:
                    tmpP1[labeled_array == i] = 0
                
                InCludePixNum.append(SigPixel.shape[1])
                InCludePix.append(SigPixel)
                
            else:
                InCludePixNum.append(0)
                InCludePix.append(np.array([]))
            
        return InCludePixNum, InCludePix, mask, valVox
        
    def plot_distance_bar(self, groups, group_stats, ax, save_folder):
        
        stats_mean = group_stats['mean']
        stats_std = group_stats['std']
        
        layer = 'FC_6'
        
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


    def plot_distance_box(self, groups, distances, ax, save_folder):
        
        layer = 'FC_6'
        
        group_values = [distances[group] for group in groups]
        
        flierprops = dict(marker='.', markerfacecolor='red', markersize=1, linestyle='none', markeredgecolor='red')
        ax.boxplot(group_values, notch=True, flierprops=flierprops)
        
        # Formatting plot
        ax.set_xticks([1, 2], ['Feature MI', 'Non-feature MI'])
        ax.set_xlabel('FC_6')
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
    # [notice] one option is build a relationship between groups and tDis_stats
    def calculate_distance_ttest(self, groups, tDis_stats):
        """
            this function now receive any length of types of units and any types of input values, return a list contains all ttest_stats and ttest_pvalues
            
            groups: list of unit idx of different types of units
            tDis_stats: list of values of different types of units
        """
        
        groups_stats = {}
        group_stats = {}
        
        ttest_stats_list = []
        ttest_pvalue_list = []
        
        group_pairs = list(combinations(np.arange(len(groups)), 2))
        
        for _ in group_pairs:
 
            group_a = tDis_stats[groups[_[0]]]
            group_b = tDis_stats[groups[_[1]]]

            ttest_stats, ttest_p = ttest_ind(group_a, group_b)
            
            ttest_stats_list.append(ttest_stats)
            ttest_pvalue_list.append(ttest_p)
    
        stats_mean = [np.mean(tDis_stats[group]) for group in groups]
        stats_std = [np.std(tDis_stats[group]) for group in groups]
        stats_sem = [np.std(tDis_stats[group])/np.sqrt(len(group)-1) for group in groups]
        
        groups_stats['pairs'] = group_pairs
        groups_stats['stats'] = np.array(ttest_stats_list)
        groups_stats['pvalue'] = np.array(ttest_pvalue_list)
        
        group_stats['mean'] = stats_mean
        group_stats['std'] = stats_std
        group_stats['sem'] = stats_sem

        return groups_stats, group_stats
    
    def calculate_distance(self, tSNE, encode_id, img_label, target_unit):      # try to make a figure to illustrate this?
        
        max_distance = pdist([[np.min(tSNE[:, 0]), np.min(tSNE[:, 1])], [np.max(tSNE[:, 0]), np.max(tSNE[:, 1])]], 'euclidean')     # normalization factor

        tDis = []
        tDis_stats = {}
        
        for _ in target_unit:
            tmpID = encode_id[_]     # obtain the emcoded ID   [question] why not featured ID?
            
            if tmpID.size != 0:     # check if this is a valid encoded unit/neuron
                tmpFace = np.hstack([np.where(img_label==tmpID[i])[0] for i in range(len(tmpID))])
                #tmpFace = [idx for idx, val in enumerate(img_label) if val in tmpID]   # obtain the idx of images
                
                tmpPair = list(combinations(tmpFace, 2))
                
                tmpDis = []
                for i in tmpPair:
                    distance = pdist([tSNE[i[0]], tSNE[i[1]]], 'euclidean')
                    tmpDis.append(distance)
                
                tmpDis = np.array(tmpDis)
                tDis.append(tmpDis)
                
            else:
                tDis.append(np.array([0]))
        
        tDis_stats['distances'] = np.array(tDis, dtype='object')
        tDis_stats['mean'] = np.array([np.mean(i)/max_distance for i in tDis]).reshape(-1)
        tDis_stats['std'] = np.array([np.std(i)/max_distance for i in tDis]).reshape(-1)
        tDis_stats['median'] = np.array([np.median(i)/max_distance for i in tDis]).reshape(-1)

        return tDis_stats
        
    def single_neuron_plot(self, si_idx, mi_idx, feature_idx, fmi_idx, feature, tSNE, img_label, sigP_clean):

        nonID_Ind = np.setdiff1d(np.arange(4096), np.union1d(mi_idx, si_idx))
        nonfmi_idx = np.setdiff1d(mi_idx, fmi_idx)
        nonID_FeatureInd = np.intersect1d(feature_idx, nonID_Ind)
        nonID_nonFeatureInd = np.setdiff1d(np.arange(4096), np.unique(np.concatenate([si_idx, mi_idx, feature_idx])))
        SI_FeatureInd = np.intersect1d(si_idx, feature_idx)
        SI_NonFeature = np.setdiff1d(si_idx, feature_idx)
        
        nonID_nonFeatureInd1 = [idx for idx in nonID_nonFeatureInd if not np.sum(feature[:, idx]) == 0]
        
        # colors
        colorppol = plt.get_cmap('tab20c', 60)
        colors = [colorppol(ii) for ii in range(50)]
       
        # Another kernel size and kernel std?
        x1 = tSNE[:, 0]
        y1 = tSNE[:, 1]
        ff1 = 0.2
        Ksd1 = 4
        KS1 = [round((max(y1) - min(y1)) * ff1), round((max(x1) - min(x1)) * ff1)]

        _, _, x2, y2 = self.Cal_Density(tSNE, np.ones(len(tSNE)), KS1, Ksd1)
        
        # Select cells to plot based on plotType
        plot_mapping = {
            'fMI': fmi_idx[::10],
            'nonfmi_idx': nonfmi_idx[:300:10],
            'nonID_feature': nonID_FeatureInd[:300:10],
            'nonID_nonFeature': nonID_nonFeatureInd1[:100:10],
            'SI_FeatureInd': SI_FeatureInd[:100:10],
            'SI_NonFeature': SI_NonFeature[:100:10]
        }
        
        plot_keys = list(plot_mapping.keys())
        for plotType in plot_keys:
            save_folder = os.path.join(self.StatsDir, plotType)
            utils_.make_dir(save_folder)
            CellToPlot = plot_mapping.get(plotType)
            #self.plot_scatter(CellToPlot, feature_idx, feature, x2, y2, colors, img_label, sigP_clean)
            self.plot_region_based_coding(CellToPlot, feature_idx, feature, colors, x2, y2, img_label, sigP_clean, save_folder)
        
    def plot_region_based_coding(self, CellToPlot, feature_idx, feature, colors, x, y, img_label, sigP_clean, save_folder):
        
        layer = 'FC_6'
        
        for iCell in tqdm(CellToPlot):
    
            #sigInd = np.where(feature_idx == iCell)[0]  
            
            wData = feature[:, iCell].astype(float)
            fig = plt.figure(figsize=(18, 9))
            #plt.annotate(f'FC_6 Unit: {iCell}', (0.5, 0.98), xycoords='axes fraction', ha='center', fontsize=14, bbox=dict(boxstyle="square", ec="none", fc="white"))
            
            # ===== 1
            ax1_pos = [0.05, 0.1, 0.2, 0.8]
            ax_1 = plt.gcf().add_axes(ax1_pos)
            self.plot_distance_boxplot(ax_1, wData, img_label, colors)
            
            # ===== 2
            ax2_pos = [0.3, 0.1, 0.4, 0.8] 
            ax_2 = plt.gcf().add_axes(ax2_pos)
            self.plot_scatter_with_contour(ax_2, wData, x, y, img_label, colors, iCell, feature_idx, sigP_clean[iCell])
            
            # ===== 3
            ax3_upper_pos = [0.75, 0.55, 0.175, 0.35]
            ax3_lower_pos = [0.75, 0.1, 0.175, 0.35]
            ax_3_upper = plt.gcf().add_axes(ax3_upper_pos)
            ax_3_lower = plt.gcf().add_axes(ax3_lower_pos)
            pdfxy = self.kde_2d_v3(x, y, weights=wData)
            pdfPerm = self.kde_2d_perm(x, y, weights=wData)
            vmin, vmax = self.plot_kde(ax_3_upper, ax_3_lower, pdfxy, pdfPerm)     # [question] maerge the value range from all units?
            
            cmap = plt.get_cmap("viridis")
            norm_ = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_)
            sm.set_array([])  # Just a dummy array
            cbar_ax = fig.add_axes([0.95, 0.1, 0.0125, 0.8])
            fig.colorbar(sm, cax=cbar_ax)
            #cbar = plt.colorbar(cax1, ax=axes, orientation='vertical', fraction=0.02, pad=0.06)
            
            fig.suptitle(f'{layer} Unit: {iCell}', y=0.95, fontsize=16)
            
            plt.savefig(save_folder + f'/{save_folder.split("/")[-1]}_{layer}_{iCell}.png', bbox_inches='tight')
            plt.savefig(save_folder + f'/{save_folder.split("/")[-1]}_{layer}_{iCell}.eps', format='eps', bbox_inches='tight')
            plt.close()
            
    def plot_distance_boxplot(self, ax, wData, img_label, colors):
        
        bp = ax.boxplot([wData[img_label == i] for i in range(1,51)], vert=False, patch_artist=True, sym='+')
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set(edgecolor='none')

        threshold = np.mean(wData) + 2*np.std(wData)
        ax.vlines(threshold, 0, 52, colors='red', linestyles='-', linewidth=1.0, alpha=0.75)
        mean_list = np.mean(np.array([wData[img_label == i] for i in range(1,51)]), axis=1)

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

    def plot_scatter_with_contour(self, ax, wData, x, y, img_label, colors, iCell, feature_idx, sigP):
        
        if np.sum(wData) != 0:
            size_weight = wData / max(wData)     # [notice] can not divide by 0 if all values are 0
            sizes = np.ones(500) * 15 * (1 + 20 * size_weight)
            
            handles_not_featured = []
            handles_featured = []
            labels = []
            
            encoded_img_idx = []
            for _ in range(len(x)):
                if sigP[int(y[_]-1), int(x[_]-1)] == 1:
                    encoded_img_idx.append(_)
            encoded_id_idx = np.unique(img_label[encoded_img_idx])
            not_encoded_id_idx = np.setdiff1d(np.arange(50)+1, encoded_id_idx)
            
            for gg in not_encoded_id_idx:  # this can be changed to different types of id
                current_scatter = ax.scatter(x[img_label == gg]-1, y[img_label == gg]-1, s=sizes[img_label == gg], color=colors[gg-1], alpha=0.5)
                handles_not_featured.append(current_scatter)
                
            for gg in encoded_id_idx:
                label_text = f'{gg}'
                current_scatter = ax.scatter(x[img_label == gg]-1, y[img_label == gg]-1, s=sizes[img_label == gg], color=colors[gg-1], alpha=0.7)
                handles_featured.append(current_scatter)
                labels.append(label_text)
            
            if iCell in feature_idx: 
                contours = ax.contour(sigP, [1], colors='c')
            
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
    
    def plot_scatter(self, ax, CellToPlot, feature_idx, feature, x, y, colors, img_label, sigP_clean):
        
        for celltoplot_idx in CellToPlot:
            iCell = celltoplot_idx
        
            # Find if the cell is in feature_idx
            #sigInd = np.where(feature_idx == iCell)[0]
        
            wData = feature[:, iCell].astype(float)
        
            size_weight = wData / max(wData)
            sizes = np.ones(500) * 15 * (1 + 20 * size_weight)
            handles = []
        
            for gg in range(1, 51):  # Python's range starts at 0, so adjust accordingly
                current_scatter = ax.scatter(x[img_label == gg], y[img_label == gg], s=sizes[img_label == gg], color=colors[gg-1], alpha=0.7)
                handles.append(current_scatter)
            
            if iCell in feature_idx:  # or if len(sigInd) > 0:
                contours = ax.contour(sigP_clean[iCell], [1], colors='c')
                for collection in contours.collections:
                    collection.set_linewidth(3)
                    
    def region_sel(self, p, mappedX, CellToAnalyze, encode_id, mi_idx, img_label, KS=[20,20], Ksd=2, clusThre=0.025, alpha=0.01):
        
        Z1, ZC1, x, y = self.Cal_Density(mappedX, None, KS, Ksd)
        
        #maskLevel = np.median(ZC1) / maskFactor
        maskLevel = np.median(ZC1, axis=0) / self.maskFactor
        
        mask = ZC1 >= maskLevel
        
        valVox = mask.shape[0] * mask.shape[1]
        clusterSize = valVox * clusThre

        sigP_clean = []
        InCludeFace = []
        InCludeID = []
        InCludePix = []
        
        feature_idx = []
        sigP_mask = []
        fmi_idx = []
        
        for icc in tqdm(CellToAnalyze):
            
            tmpP = p[icc]
            sigP = tmpP <= alpha
            sigP_mask.append(sigP*mask)
            tmpP = sigP*mask
            
            s = generate_binary_structure(2,2)
            cc, nComp = label(tmpP, structure=s) 
            
            if nComp > 0:
               
                SigID_all = []     # ID passed condition 1 - cluster size
                SigID = []     # ID passed 2 conditions
                SigFace = []    
                SigPixel = []
    
                for ii in range(nComp):     # for each component
                    
                    if np.sum(cc == ii+1) < clusterSize:
                        tmpP[cc == ii+1] = 0
                        continue
                    
                    # -------------------------------------------------------------------------
                    tmpSigFace = [i for i in range(len(x)) if cc[int(y[i]-1), int(x[i]-1)] == ii+1]
                    #tmpSigFace = [i for i in range(len(x)) if cc[int(y[i]), int(x[i])] == ii+1]
                    # -------------------------------------------------------------------------
                    
                    tmpSigID = np.unique(img_label[tmpSigFace])
                    
                    SigID_all.append(tmpSigID)  
    
                    if len(tmpSigID) < 2 or len(tmpSigFace) < 5:
                        tmpP[cc == ii+1] = 0
                    else:
                        if icc not in feature_idx:
                            feature_idx.append(icc)
                        
                        SigID.append(tmpSigID)
                        SigFace.append(tmpSigFace)
                        SigPixel.append(cc[cc == ii+1])
                        
                        if len(set(tmpSigID) & set(list(encode_id[icc].reshape(-1)))) > 1 and icc in mi_idx and icc not in fmi_idx:
                            fmi_idx.append(icc)
                
                sigP_clean.append(tmpP)
                InCludeFace.append(SigFace)
                InCludeID.append(SigID)
                InCludePix.append(SigPixel)
            
            else:
                sigP_clean.append(np.zeros(tmpP.shape))
                InCludeFace.append(SigFace)
                InCludeID.append(SigID)
                InCludePix.append(SigPixel)
        
        return feature_idx, sigP_clean, np.array(fmi_idx), InCludeFace, InCludeID, InCludePix, maskLevel, clusterSize

    def lexicographic_order(self):
        id_order = np.arange(1,1+self.num_classes).astype(str)
        id_order_idx = np.argsort(id_order)
        id_order_lexical = id_order[id_order_idx].astype(int)
        
        return id_order_lexical

    def Cal_Perm_Density(self, mappedX, wData=None, nPerm=1000, kSize=[20,20], kSD=2):
        
        if wData is None:
            wData = np.ones(500)
    
        Z, ZC, x, y = self.Cal_Density(mappedX, wData, kSize, kSD)
    
        FalsePos = np.zeros(Z.shape)
    
        permZ = []
        permZC = []
    
        for ii in range(nPerm):
            N = np.random.permutation(len(wData))
            Data = wData[N]
            
            perm_Z, perm_ZC, _, _ = self.Cal_Density(mappedX, Data, kSize, kSD)
            
            permZ.append(perm_Z)
            permZC.append(perm_ZC)
            
            FalsePos += perm_ZC > ZC
    
        p = FalsePos / nPerm
    
        meanPermZ = np.mean(permZ, axis=0)
        meanPermZC = np.mean(permZC, axis=0)
    
        return Z, ZC, meanPermZ, meanPermZC, p, permZ, permZC, x, y

    #FIXME
    # [notice] obvisouly, this is basically the same upper part with self.get_kernel_size(), just with different weight
    def Cal_Density(self, mappedX, FR=None, kSize=[20,20], kSD=2):
        
        if FR is None:
            FR = np.ones(500)
    
        minX = np.min(mappedX[:, 0])
        x = mappedX[:, 0] - minX + 1
        minY = np.min(mappedX[:, 1])
        y = mappedX[:, 1] - minY + 1
    
        imgW = np.ceil(np.max(x)).astype(int)
        imgH = np.ceil(np.max(y)).astype(int)
    
        Z = np.zeros((imgH, imgW))
    
        for i in range(len(x)):
            if not np.isnan(y[i]) and not np.isnan(x[i]):
                Z[round(y[i])-1, round(x[i])-1] += FR[i]
    
        kernel = self.gausskernel(kSize, kSD)
        
        ZC = convolve(Z, kernel, mode='constant')

        return Z, ZC, x, y
    
    # =========================================================================
    def generate_p_value(self, tSNE, feature, unit_idx):
 
        layer = 'FC_6'
        
        file_path = os.path.join(self.StatsDir, f'{layer}_sq{self.sq}.pkl')
        
        if os.path.exists(file_path):
            results = utils_.pickle_load(file_path)
            p = results['p']
            KS = results['KS']
            Ksd = results['Ksd']
           
        else:
            
            KS, Ksd = self.get_kernel_size(tSNE[:, 0], tSNE[:, 1])
            
            # ----- parallel computing -----
            p = [0]*len(unit_idx)
            executor = ProcessPoolExecutor(max_workers=os.cpu_count())
            job_pool = []
            for i in tqdm(range(len(unit_idx)), desc='Submit'):
                job = executor.submit(self.calculate_p_values_parallel, feature, i, tSNE, KS, Ksd)
                job_pool.append(job)
            for i in tqdm(range(len(unit_idx)), desc='Collect'):
                p[i] = job_pool[i].result()
            executor.shutdown()
            # -----
            
            # ----- sequential computing
            #p = []
            #for i in tqdm(range(len(unit_idx)), desc='Sequential'):
            #    _, _, _, _, p_tmp, _, _, _, _ = self.Cal_Perm_Density(tSNE, feature[:, i], 1000, KS, Ksd)
            #    p.append(p_tmp)
            # -----
                
            results = {'p': p, 'sq': self.sq, 'KS': KS, 'Ksd': Ksd, 'tSNE': tSNE, 'layer': layer}
            utils_.pickle_dump(file_path, results)
            
        return p, KS, Ksd
        
    def calculate_p_values_parallel(self, feature, i, tSNE, KS, Ksd):
        _, _, _, _, p_tmp, _, _, _, _ = self.Cal_Perm_Density(tSNE, feature[:, i], 1000, KS, Ksd)
        return p_tmp
        
    # =========================================================================
        
    def get_kernel_size(self, x, y):
        """
            this function calculate the kernel_size and sigma for the following calculation on p values
            
            In fact, the function here is not the same one described in the paper. 
            (1) The method here calculate Ksd and Ks based on 'the number of 
            connected components', and use a scaling factor: self.sq=0.035 to 
            decide the value of Ksd(sigma, which decides the speed of decrease)
            -> Ksd(sigma) = sq*nComp, KS = 2*radius(3*sigma)+1
            (2) The method described in paper is based on 'the size of feature map',
            and use another scaling factor: ff1=0.2(in self.single_neuron_plot()) to
            decide the kernel size -> KS = ff1*[ylim,xlim]
            
            x: first dimension of tSNE mappedX;
            y: second dimension of tSNE mappedX
            self.sq: an empirical scale factor to decide the sigma of the gaussian filter. default to be 0.035, (close to sd = 4, used in previous expriments)
        """
        x = x - np.min(x)
        y = y - np.min(y)
    
        imgW = int(np.ceil(np.max(x)))+1
        imgH = int(np.ceil(np.max(y)))+1
    
        Z = np.zeros((imgH, imgW))  # empty grid
        
        for i in range(len(x)):
            Z[int(np.round(x[i])), int(np.round(y[i]))] += 1
                
            # [notice] for bio experiments
            #if not np.isnan(y) and not np.isnan(x):
            #    Z[x-1, y-1] += 1
            
        labeled, nComp = label(Z, structure=generate_binary_structure(2,2))  
        Ksd = nComp * self.sq  # determine the sigma
        
        # decide the kernel size
        # Ksy = int(2 * 3 * Ksd) + 1
        Ksy = int(2 * 3 * (np.floor(Ksd)) + 1)  
        Ksx = int(np.floor(Ksy * Z.shape[0] / Z.shape[1]))
        
        Ks = [Ksy, Ksx]
    
        return Ks, Ksd
    
    #FIXME
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
        
        for k in range(D):
            # Make the appropriate 1-D Gaussian
            grid = np.arange(-R[k], R[k] + 1)
            gauss = np.exp(-grid**2 / (2 * S[k]**2))
            gauss = gauss / np.sum(gauss)  # normalization
    
            # Then expand it against kernel-so-far
            if (k == 0):
                kernel = gauss
            else:
                Dpast = np.ones(k, dtype=int)
                expand = np.reshape(gauss, [*Dpast, -1])
                kernel = np.squeeze(np.outer(kernel, expand).reshape(*expand.shape, -1))
    
        return kernel





if __name__ == '__main__':
    
    layers = ['FC_6']
    neurons = [4096]

    root_dir = '/media/acxyle/Data/ChromeDownload/'

    selectivity_feature_analyzer = Selectiviy_Analysis_Feature(
                root=os.path.join(root_dir, 'Identity_VGG_Feature_Results/'), 
                dest=os.path.join(root_dir, 'Identity_VGG_Feature_Original/'), 
                layers=layers, neurons=neurons, taskInstruction='CelebA')
    
    selectivity_feature_analyzer.load_useful_data()
