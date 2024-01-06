#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:35:00 2023

@author: acxyle

#TODO
    write a script to do feature analysis
    
    [Induction]
    1. use spiking_model.py to seperate layers;
    2. use spiking_intermediate_output.py to visualize and validate the features in diferent levels;
    3. use spiking_featuremap.py to extract and save the feature.pkl
    4. use Selectivity_Analyzer to execute analysis
    
#TODO
    [Jan 3, 2023] add the k-folds comparisons
"""

import os
import time
import psutil
import argparse
import numpy as np
import matplotlib.pyplot as plt


import Selectivity_Analysis_ANOVA
import Selectivity_Analysis_Encode
import Selectivity_Analysis_DR_and_SM
import Selectivity_Analysis_RSA
import Selectivity_Analysis_Feature

import vgg, resnet
import utils_

# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Selectivity Analyzer Ver 4.0", add_help=True)

parser.add_argument("--num_classes", type=int, default=50, help="[Codelp] set the number of classes")
parser.add_argument("--num_samples", type=int, default=10, help="[Codelp] set the sample number of each class")
parser.add_argument("--alpha", type=float, default=0.01, help='[Codelp] assign the alpha value for ANOVA')

parser.add_argument("--root_dir", type=str, default="/home/acxyle-workstation/Downloads", help="[Codelp] root directory for features and neurons")

parser.add_argument("--model", type=str, default='vgg16_bn')     # trigger


# -----
args = parser.parse_args()

# -----
def get_layers_and_neurons(model):
    
    if '16_bn' in model.lower():
    
        model_name = 'vgg16_bn'
        
        model_ = vgg.__dict__[model_name](num_classes=50)
        
    elif '16' in model.lower():
        
        model_name = 'vgg16'
        
        model_ = vgg.__dict__[model_name](num_classes=50)
        
    else:
        
        raise ValueError(f"[Codwarning] model '{model}' not supported")
    
    return utils_.generate_vgg_layers_list_ann(model_, model_name)
    

def describe_model(layers, neurons, shapes):

    layers_info = [list(pair) for pair in zip(list(np.arange(len(layers)+1)), layers, neurons, shapes)]
    max_widths = [max(len(str(row[i])) for row in layers_info)+2 for i in range(len(layers_info[0]))]
    print("|".join("{:<{}}".format(header, max_widths[i]) for i, header in enumerate(["No.", "layer", "neurons", "shapes"])))
    print("-" * sum(max_widths))
    for row in layers_info:
        print("|".join("{:<{}}".format(str(item), max_widths[i]) for i, item in enumerate(row)))
    print("-" * sum(max_widths))


# ----------------------------------------------------------------------------------------------------------------------
# FIXME - under construction...
class Multi_Model_Analysis:
    
    def __init__(self, 
                 feature_root_general='Face Identity VGG16_fold_',
                 num_fold=5):
        
        self.feature_root_general = feature_root_general
        self.num_fold = num_fold
        self.model_structure = feature_root_general.split('_')[0].split(' ')[-1]
        
        self.model_root = os.path.join(args.root_dir, self.feature_root_general)
        utils_.make_dir(self.model_root)
        
        if 'spiking' not in args.model and 'vgg' in args.model:
        
            model_ = vgg.__dict__[args.model](num_classes=50)
            self.layers, self.neurons, _ = utils_.generate_vgg_layers_list_ann(model_, args.model)
        
    
    def plot_Encode_pct_multi_models(self, ):
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        Encode_save_root = os.path.join(self.model_root, 'Encode')
        utils_.make_dir(Encode_save_root)
        
        def _Encode_curve_pct_collect():

            curve_dict_folds_path = os.path.join(Encode_save_root, 'Encode_folds_dict.pkl')                
            
            if os.path.exists(curve_dict_folds_path):
                
                curve_dict_folds = utils_.load(curve_dict_folds_path, verbose=False)
            
            else:
                
                curve_dict_folds = {}
        
                for fold_idx in np.arange(self.num_fold):
                    
                    root = os.path.join(self.model_root+str(fold_idx))
                    
                    Sort_dict = utils_.load(os.path.join(root, 'Analysis', 'Encode', 'Sort_dict.pkl'), verbose=True)
                    
                    Encode_types_pct = Selectivity_Analysis_Encode.Encode_feaquency_analyzer.obtain_Encode_types_pct(self.layers, self.neurons, Sort_dict)
                    
                    curve_dict_folds[fold_idx] = Selectivity_Analysis_Encode.Encode_feaquency_analyzer.obtain_Encode_types_curve_dict(Encode_types_pct)
                    
                utils_.dump(curve_dict_folds, curve_dict_folds_path)
            
            return curve_dict_folds
        
        curve_dict_folds = _Encode_curve_pct_collect()
        
        #TODO --- rebuild the curve dict and simplify
        types = list(curve_dict_folds[0].keys())
        
        curve_folds = {}
        
        for type_ in types:
             
            values_folds_array = np.array([curve_dict_folds[i][type_]['values'] for i in range(self.num_fold)])
            
            folds_mean = np.mean(values_folds_array, axis=0)
            folds_std = np.std(values_folds_array, axis=0)  
            
            color = curve_dict_folds[0][type_]['color']
            if color == 'black':
                color = '#555555'
            
            curve_folds[type_] = {'color': color,
                                  'linestyle': curve_dict_folds[0][type_]['linestyle'],
                                  'linewidth': curve_dict_folds[0][type_]['linewidth'],
                                  'label': curve_dict_folds[0][type_]['label'],
                                  'values': folds_mean,
                                  'std': folds_std}
            
        # -----
        fig, ax = plt.subplots(figsize=(10,6))
        Selectivity_Analysis_Encode.Encode_feaquency_analyzer.encode_layer_percent_plot(Encode_save_root, fig, ax, self.layers, curve_folds, None)
        
        title = 'Encode_pct_5_types'
        ax.set_title(f'{self.model_structure} {title}')
        fig.savefig(os.path.join(Encode_save_root, title+'.png'), bbox_inches='tight')
        fig.savefig(os.path.join(Encode_save_root, title+'.eps'), bbox_inches='tight')     
        plt.close()
        
        # -----
        act_idx, act_layers, act_neurons = utils_.activation_function_vgg(self.layers, self.neurons)
        
        for type_ in types:
            
            curve_folds[type_]['values'] = curve_folds[type_]['values'][act_idx]
            curve_folds[type_]['std'] = curve_folds[type_]['std'][act_idx]
        
        fig, ax = plt.subplots(figsize=(10,6))
        Selectivity_Analysis_Encode.Encode_feaquency_analyzer.encode_layer_percent_plot(Encode_save_root, fig, ax, act_layers, curve_folds, None)
        
        title = 'Encode_pct_act_5_types'
        ax.set_title(f'{self.model_structure} {title}')
        fig.savefig(os.path.join(Encode_save_root, title+'.png'), bbox_inches='tight')
        fig.savefig(os.path.join(Encode_save_root, title+'.eps'), bbox_inches='tight')     
        plt.close()
        
        print('g')
        
            
    def plot_ANOVA_pct_multi_models(self, ):
        
        plt.rcParams.update({'font.size': 18})    
        plt.rcParams.update({"font.family": "Times New Roman"})
        
        ANOVA_save_root = os.path.join(self.model_root, 'ANOVA')
        utils_.make_dir(ANOVA_save_root)
        
        def _ANOVA_pct_collect(save_data:bool=True):

            ANOVA_folds = {}
    
            for fold_idx in np.arange(5):
                
                root = os.path.join(self.model_root+str(fold_idx))
                
                ANOVA_folds[fold_idx] = utils_.load(os.path.join(root, 'Analysis', 'ANOVA', 'ratio.pkl'), verbose=False)
                
            ANOVA_folds_array = np.array([np.array(_) for _ in list(ANOVA_folds.values())])     # (num_folds, num_layers)
            
            if save_data:
                
                utils_.dump(ANOVA_folds_array, os.path.join(ANOVA_save_root, 'ANOVA_folds_array.pkl'))
            
            return ANOVA_folds_array
        
        def _ANOVA_pct_plot(ax, layers, ANOVA_folds_array, title):
            
            folds_mean = np.mean(ANOVA_folds_array, axis=0)
            folds_std = np.std(ANOVA_folds_array, axis=0)  
            
            ax.fill_between(np.arange(len(layers)), folds_mean-folds_std, folds_mean+folds_std, edgecolor=None, facecolor='skyblue', alpha=0.75)
            ax.plot(np.arange(len(layers)), folds_mean, color='blue', linewidth=0.5)
            ax.set_xticks(np.arange(len(layers)), layers, rotation='vertical')
            ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
            ax.set_title(f'{self.model_structure} {title}')
            
            plt.tight_layout()
            #fig.savefig(os.path.join(ANOVA_save_root, f'{title}_folds.png'))
            fig.savefig(os.path.join(ANOVA_save_root, f'{title}_folds.eps'))
            plt.close()
        
        # ---
        ANOVA_folds_array = _ANOVA_pct_collect()
        
        # ---
        fig, ax = plt.subplots(figsize=(18,10))
        _ANOVA_pct_plot(ax, self.layers, ANOVA_folds_array, 'ANOVA_pct')

        # ---
        act_idx, act_layers, act_neurons = utils_.activation_function_vgg(self.layers, self.neurons)
        ANOVA_folds_array = ANOVA_folds_array[:, act_idx]
        
        fig, ax = plt.subplots(figsize=(10,10))
        _ANOVA_pct_plot(ax, act_layers, ANOVA_folds_array, 'ANOVA_pct_act')


def single_model_analysis(args, feature_folder):

    start_time = time.time()
    
    # --- init
    feature_root = os.path.join(args.root_dir, feature_folder)
    layers, neurons, shapes = get_layers_and_neurons(args.model)
    
    # ---
    print(f'[Codinfo] Listing model [{feature_folder}] layers and neuron numbers')

    describe_model(layers, neurons, shapes)
    
    # ----- 1. ANOVA
# =============================================================================
#     ANOVA_analyzer = Selectivity_Analysis_ANOVA.ANOVA_analyzer(
#                                                                feature_root, 
#                                                                alpha=args.alpha, num_classes=args.num_classes, num_samples=args.num_samples, 
#                                                                layers=layers, neurons=neurons)
#     
#     ANOVA_analyzer.calculation_ANOVA()
#     ANOVA_analyzer.plot_ANOVA_pct()
#     
#     del ANOVA_analyzer     # release memory space
# =============================================================================
    
    # ----- 2. Encode
    Encode_analyzer = Selectivity_Analysis_Encode.Encode_feaquency_analyzer(
                                                                            feature_root, 
                                                                            layers=layers, neurons=neurons)
    
    Encode_analyzer.calculation_Encode()
    
    Encode_analyzer.plot_Encode_pct(num_types=23)
    Encode_analyzer.plot_Encode_pct(num_types=5)
    
    Encode_analyzer.plot_Encode_freq()
    # ---
    Encode_analyzer.plot_stacked_responses(num_types=5)
    Encode_analyzer.plot_stacked_responses(num_types=10)

    Encode_analyzer.plot_sample_responses()
    
    # ---
    Encode_analyzer.SVM_plot()
    
    # ---
    Encode_analyzer.NN_unit_FR_stats_plot()
    Encode_analyzer.NN_plot_pie_chart()
    
    del Encode_analyzer
    
    # ----- 3. DR SM
    DR_analyzer = Selectivity_Analysis_DR_and_SM.Selectiviy_Analysis_DR(
                                                                        feature_root, 
                                                                        layers=layers, neurons=neurons)

    DR_analyzer.selectivity_analysis_tsne()
    
    del DR_analyzer
    
    # ---
    SM_analyzer = Selectivity_Analysis_DR_and_SM.Selectiviy_Analysis_SM(
                                                                        feature_root, 
                                                                        layers=layers, neurons=neurons)
    
    SM_analyzer.selectivity_analysis_similarity_metrics(
                                                        metrics=['euclidean', 'pearson']
                                                        )
    
    del SM_analyzer
    
    # ----- 4. RSA
    RSA_monkey = Selectivity_Analysis_RSA.Selectiviy_Analysis_Correlation_Monkey(NN_root=feature_root, layers=layers, neurons=neurons)
    RSA_monkey.monkey_neuron_analysis(metrics=['euclidean', 'pearson'])
    
    del RSA_monkey

    RSA_human = Selectivity_Analysis_RSA.Selectiviy_Analysis_Correlation_Human(NN_root=feature_root, layers=layers, neurons=neurons)
    RSA_human.human_neuron_analysis()
    
    del RSA_human
    
    # ----- 5. Feature
    selectivity_feature_analyzer = Selectivity_Analysis_Feature.Selectivity_Analysis_Feature(
                feature_root, 
                layers=['L5_B3_neuron', 'neuron_1', 'neuron_2'], 
                neurons=[100352, 4096, 4096]
                )
    
    selectivity_feature_analyzer.feature_analysis('TSNE')
    
    del selectivity_feature_analyzer
    
    # --- 
    print(f"[Codinfo] All results are saved in {os.path.join(feature_root, 'Analysis')}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    print('[Codinfo] Elapsed Time: {}:{:0>2}:{:0>2} '.format(int(elapsed/3600), int((elapsed%3600)/60), int((elapsed%3600)%60)))
    print('[Codinfo] Experiment Done.')    

#FIXME
def Main_Analyzer(args):

    print('[Codinfo] Starting Selective Analysis Experiment...')
    print(args)
    
    for fold_idx in [2]:
        single_model_analysis(args, f'Face Identity SpikingVGG16bn_IF_T4_CelebA2622_fold_{fold_idx}')
        #single_model_analysis(args, f'Face Identity VGG16_fold_{fold_idx}')
        
        
if __name__ == "__main__":
    
    Main_Analyzer(args)
    
    #multi_model_analysis = Multi_Model_Analysis(feature_root_general='Face Identity VGG16_fold_')
    #multi_model_analysis.plot_Encode_pct_multi_models()
