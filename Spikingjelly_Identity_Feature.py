#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:44:28 2023

@author: acxyle

    Task: (1) make the code clear and precise; (2) make the code computation and plot separate

"""
import os

import scipy.stats

import pickle

from scipy.stats import norm

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import convolve

from scipy.stats import norm
from tqdm import tqdm

import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import label, generate_binary_structure  # the name [label] looks contradictory with many kinds of labels
from skimage.measure import regionprops
from scipy.stats import binom
from scipy.stats import gaussian_kde

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
        
        encode_class_dict = utils_.pickle_load(os.path.join(self.dest, 'SIMI/SIMI_cnt.pkl'))
        
        # lexicographic order
        encode_id = loadmat(os.path.join(self.dest, 'encode_mat/FC_6_encode.mat'))['encodeID'].reshape(-1)
        
        si_idx = np.array([i for i in range(len(encode_id)) if encode_id[i].shape[1] == 1])
        mi_idx = np.array([i for i in range(len(encode_id)) if encode_id[i].shape[1] > 1])
        
        tSNE = loadmat(os.path.join(self.dest, 'TSNE/tSNE_FC_6_all.mat'))['FC_6_all']
        
        unit_to_analyze = np.arange(4096)
        StatsDir = os.path.join(self.dest, 'Feature')
        utils_.make_dir(StatsDir)
        #self.generate_p_value(tSNE, self.sq, StatsDir, unit_to_analyze, feature)
        p_file = loadmat('/home/acxyle/Downloads/osfstorage-archive-supp/Res/DensityStats/CelebA/VGG16/FC_6_Sq035.mat')
        p_values = p_file['p'].reshape(-1) 
        KS = p_file['KS'].reshape(-1).astype(int)
        Ksd = p_file['Ksd'].reshape(-1)[0]
        
        #Ks, Ksd = self.getKernelsize(tSNE[:,0], tSNE[:,1])
        #_, _, _, _, p, _, _, _ = self.Cal_Perm_Density(tSNE, wData=feature[:,0], kSize=Ks, kSD=Ksd)
        
        clusThre = 0.025
        alpha = 0.01
        isFDR = 0
        
        id_label = self.lexicographic_order()
        img_label = np.array( [[id_label[_]]*10 for _ in range(50)] ).reshape(-1)
        
        for _ in range(len(encode_id)):
            correct_id = id_label[encode_id[_].reshape(-1).astype(int)-1]
            encode_id[_] = correct_id
                
    
        feature_idx, sigP_clean, fmi_idx, InCludeFace, InCludeID, InCludePix, maskLevel, clusterSize = self.region_sel(p_values, tSNE, unit_to_analyze, encode_id, mi_idx, img_label, clusThre, alpha, self.maskFactor, KS, Ksd)
        
        bionorP = 1 - binom.cdf(len(feature_idx), len(unit_to_analyze), 0.05)
        
        self.single_neuron_plot(si_idx, mi_idx, feature_idx, fmi_idx, feature, tSNE, img_label, sigP_clean)
        
        print('6')
        
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
        
        # Directories and plot types
        plotType = 'fMI'
       
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
        
        CellToPlot = plot_mapping.get(plotType)
        
        #self.plot_scatter(CellToPlot, feature_idx, feature, x2, y2, colors, img_label, sigP_clean)
        self.plot_region_based_coding(CellToPlot, feature_idx, feature, colors, x2, y2, img_label, sigP_clean)
        
    # the plot of an entire figure
    def plot_region_based_coding(self, CellToPlot, feature_idx, feature, colors, x, y, img_label, sigP_clean):
        
        layer = 'FC_6'
        
        for iCell in CellToPlot:
    
            sigInd = np.where(feature_idx == iCell)[0]  
            
            wData = feature[:, iCell].astype(float)
            fig = plt.figure(figsize=(18, 8))
            #plt.annotate(f'FC_6 Unit: {iCell}', (0.5, 0.98), xycoords='axes fraction', ha='center', fontsize=14, bbox=dict(boxstyle="square", ec="none", fc="white"))
            
            # ===== 1
            ax1_pos = [0.05, 0.1, 0.2, 0.8]
            ax_1 = plt.gcf().add_axes(ax1_pos)
            self.plot_boxplot(ax_1, wData,  img_label, colors)
            
            # ===== 2
            ax2_pos = [0.3, 0.1, 0.4, 0.8] 
            ax_2 = plt.gcf().add_axes(ax2_pos)
            self.plot_scatter_with_contour(ax_2, wData, x, y, img_label, colors, iCell, feature_idx, sigP_clean)
            
            # ===== 3
# =============================================================================
#             ax3_upper_pos = [0.75, 0.55, 0.15, 0.3]
#             ax3_lower_pos = [0.75, 0.1, 0.15, 0.3]
#             ax_3_upper = plt.gcf().add_axes(ax3_upper_pos)
#             ax_3_lower = plt.gcf().add_axes(ax3_lower_pos)
#             pdfxy = self.kde_2d_v3(x, y, weights=wData)
#             pdfPerm = self.kde_2d_perm(x, y, weights=wData)
#             vmin, vmax = self.plot_kde(ax_3_upper, ax_3_lower, pdfxy, pdfPerm)     # [question] maerge the value range from all units?
#             
#             cmap = plt.get_cmap("viridis")
#             norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#             sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#             sm.set_array([])  # Just a dummy array
#             cbar_ax = fig.add_axes([0.9, 0.1, 0.0125, 0.75])
#             fig.colorbar(sm, cax=cbar_ax)
#             #cbar = plt.colorbar(cax1, ax=axes, orientation='vertical', fraction=0.02, pad=0.06)
# =============================================================================
            
            fig.suptitle(f'{layer} Unit: {iCell}', y=0.95, fontsize=16)
            
            print('6')
            
    def plot_boxplot(self, ax, wData, img_label, colors):
        
        bp = ax.boxplot([wData[img_label == i] for i in range(1,51)], vert=False, patch_artist=True, sym='+')
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
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
        
    def plot_scatter_with_contour(self, ax, wData, x, y, img_label, colors, iCell, feature_idx, sigP_clean):
        sizeweigt = wData / max(wData)
        sizes = np.ones(500) * 15 * (1 + 20 * sizeweigt)
        handles = []
    
        for gg in range(1, 51):  
            current_scatter = ax.scatter(x[img_label == gg], y[img_label == gg], s=sizes[img_label == gg], color=colors[gg-1], alpha=0.7)
            handles.append(current_scatter)
        
        if iCell in feature_idx: 
            contours = ax.contour(sigP_clean[iCell], [1], colors='c')
        
        ax.set_xlabel('Feature Dimension 1')
        ax.set_ylabel('Feature Dimension 2')
    
    # --- the plot of kde
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

    # --- the computation of kde_2d_perm
    def kde_2d_perm(self, x, y, bw=None, weights=None, isplot=False):
 
        pdfPerm = []
    
        for ii in range(1000):
            sData = weights[np.random.permutation(len(weights))]
            pdf_xy = self.kde_2d_v3(x, y, bw=bw, weights=sData)
            pdfPerm.append(pdf_xy)

        pdfPerm = np.mean(np.array(pdfPerm), axis=0)

        return pdfPerm
          
    # --- the  computation of ked_2d
    def kde_2d_v3(self, x, y, bw=None, weights=None, isplot=False, plot_scale=100):
        pdfx = self.ksdensity(x, bw, weights)
        pdfy = self.ksdensity(y, bw, weights)
        
        pdfx = pdfx(np.linspace(min(x), max(x), plot_scale))
        pdfy = pdfy(np.linspace(min(y), max(y), plot_scale))

        pdfx, pdfy = np.meshgrid(pdfx, pdfy)
        pdfxy = pdfx * pdfy
        
        return pdfxy
        
    def ksdensity(self, data, bw=None, weights=None):
        ksdensity = gaussian_kde(data, bw_method=bw, weights=weights)
        return ksdensity
 
    
    def plot_scatter(self, ax, CellToPlot, feature_idx, feature, x, y, colors, img_label, sigP_clean):
        
        for celltoplot_idx in CellToPlot:
            iCell = celltoplot_idx
        
            # Find if the cell is in feature_idx
            #sigInd = np.where(feature_idx == iCell)[0]
        
            wData = feature[:, iCell].astype(float)
        
            sizeweigt = wData / max(wData)
            sizes = np.ones(500) * 15 * (1 + 20 * sizeweigt)
            handles = []
        
            for gg in range(1, 51):  # Python's range starts at 0, so adjust accordingly
                current_scatter = ax.scatter(x[img_label == gg], y[img_label == gg], s=sizes[img_label == gg], color=colors[gg-1], alpha=0.7)
                handles.append(current_scatter)
            
            if iCell in feature_idx:  # or if len(sigInd) > 0:
                contours = ax.contour(sigP_clean[iCell], [1], colors='c')
                for collection in contours.collections:
                    collection.set_linewidth(3)
                    
    # ---
    
    def region_sel(self, p, mappedX, CellToAnalyze, encode_id, mi_idx, img_label, clusThre, alpha, maskFactor, KS, Ksd):
        
        Z1, ZC1, x, y = self.Cal_Density(mappedX, None, KS, Ksd)
        
        #maskLevel = np.median(ZC1) / maskFactor
        maskLevel = np.median(ZC1, axis=0) / maskFactor
        
        mask = ZC1 >= maskLevel
        
        valVox = np.sum(mask)
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
        
        return feature_idx, sigP_clean, fmi_idx, InCludeFace, InCludeID, InCludePix, maskLevel, clusterSize
                
        

    
    def lexicographic_order(self):
        id_order = np.arange(1,1+self.num_classes).astype(str)
        id_order_idx = np.argsort(id_order)
        id_order_lexical = id_order[id_order_idx].astype(int)
        
        return id_order_lexical

    # Assuming `Cal_Density` and `gausskernel` are defined elsewhere in your code
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
    
        sigP = p < 0.001
    
        return Z, ZC, meanPermZ, meanPermZC, p, permZ, permZC, x, y

    #FIXME
    # [notice] obvisouly, this is basically the same upper part with self.getKernelSize(), just with different weight
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

    # MATLAB 5分钟 可以做完的事情， python 需要 7小时， 多线程/进程操作 is necesary
    # 测试使用 MATLAB 生成的 p文件 进行 
    def generate_p_value(self, tSNE, sq, StatsDir, unit_idx, feature):
 
        layer = 'FC_6'
        
        mappedX = tSNE
        
        x1 = mappedX[:, 0]
        y1 = mappedX[:, 1]
        
        KS, Ksd = self.getKernelsize(x1, y1, sq)
 
        file_path = os.path.join(StatsDir, f'{layer}_sq{sq}.pkl')
        
        if os.path.exists(file_path):
            results = utils_.pickle_load(file_path)
            p = results['p']
           
        else:
            CellToAnalyze = np.arange(feature.shape[1])
            p = []
            
            for ii in tqdm(range(len(CellToAnalyze))):
                iCell = CellToAnalyze[ii]
        
                wData = feature[:, iCell]
                wData[np.isnan(wData)] = 0
        
                _, _, _, _, p_tmp, _, _, _, _ = self.Cal_Perm_Density(mappedX, wData, 1000, KS, Ksd)
                p.append(p_tmp)
            
            p = np.array(p)
        
            results = {'p': p, 'sq': sq, 'KS': KS, 'Ksd': Ksd, 'mappedX': mappedX, 'layer': layer}
            
            utils_.pickle_dump(file_path, results)

        
    # ----- 
    def getKernelsize(self, x1, x2, sq=0.035):
        """
        x1: first dimension of tSNE mappedX;
        x2: second dimension of tSNE mappedX
        sq: an empirical scale factor to decide the sigma of the gaussian filter. default to be 0.035, (close to sd = 4, used in previous expriments)
        """
        x1 = x1 - np.min(x1) + 1  
        x2 = x2 - np.min(x2) + 1  
    
        imgW = np.ceil(np.max(x1))
        imgH = np.ceil(np.max(x2))  
    
        Z = np.zeros((int(imgH), int(imgW)))  # the raw distribution 
        
        # cal probability for each point
        # [notice] here +1 because without consideration of weights, just times
        for i in range(len(x1)):
            x = int(np.round(x1[i]))
            y = int(np.round(x2[i]))
            if not np.isnan(y) and not np.isnan(x) and y <= imgH and x <= imgW:
                Z[x-1, y-1] += 1
    
        s = generate_binary_structure(2,2)
        labeled, nComp = label(Z, structure=s)  # determine the size of the 
    
        Ksd = nComp * sq  # determine the sigma
        
        # [question] why calculate like this?
        Ksy = int(2 * 3 * (np.floor(Ksd)) + 1)  # determine kernel size using sigma
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