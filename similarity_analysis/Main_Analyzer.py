#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:35:00 2023

@author: acxyle

    ...
    
"""

import os
import time
import argparse

from spikingjelly import visualizing

import FSA_ANOVA, FSA_Encode, FSA_Responses, FSA_DRG, FSA_RSA, FSA_CKA, FSA_SVM
#import Selectivity_Analysis_Feature

import sys
sys.path.append('../')
import utils_


# ======================================================================================================================
def get_args_parser(fold_idx=0):
    parser = argparse.ArgumentParser(description="FSA Ver 5.1", add_help=True)
    
    parser.add_argument("--num_classes", type=int, default=50, help="[Codelp] set the number of classes")
    parser.add_argument("--num_samples", type=int, default=10, help="[Codelp] set the sample number of each class")
    parser.add_argument("--alpha", type=float, default=0.01, help='[Codelp] assign the alpha value for ANOVA')
    
    parser.add_argument("--FSA_root", type=str, default="/home/acxyle-workstation/Downloads/FSA", help="[Codelp] root directory for features and neurons")
    
    parser.add_argument("--FSA_dir", type=str, default='VGG/SpikingVGG')
    parser.add_argument("--FSA_config", type=str, default='SpikingVGG16bn_IF_ATan_T8_C2k_fold_')
    parser.add_argument("--fold_idx", type=int, default=f'{fold_idx}')

    parser.add_argument("--model", type=str, default='spiking_vgg16_bn')     
    
    return parser.parse_args()
    

# ----------------------------------------------------------------------------------------------------------------------
# FIXME - under construction...
class Multi_Model_Analysis():
    
    def __init__(self, args, num_folds=5, **kwargs):

        self.num_folds = num_folds
        self.root = os.path.join(args.FSA_root, args.FSA_dir, f'FSA {args.FSA_config}')
        
        self.model_structure = args.FSA_config.replace('C2k_fold_', '')
        
        _, self.layers, self.neurons, self.shapes = utils_.get_layers_and_units(args.model, 'act')
        
        self.used_unit_types = ['qualified', 'a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne', 'non_anova']
        
        self.used_id_nums = [50, 10]
        
        self._configs = {
            'num_folds': self.num_folds,
            'root': self.root,
            'layers': self.layers,
            'neurons': self.neurons
            }
        
        # --- 
        self.Encode_folds()
        self.RSA_folds()
        self.CKA_folds()
        self.SVM_folds()
        
        
    def ANOVA_folds(self, **kwargs):
    
        FSA_ANOVA.FSA_ANOVA_folds(**self._configs)()
        
        
    def Encode_folds(self, **kwargs):
        
        FSA_Encode_folds = FSA_Encode.FSA_Encode_folds(**self._configs)
        FSA_Encode_folds(**kwargs)
        
        
    def RSA_folds(self, **kwargs):

        FSA_RSA.RSA_Monkey_folds(**self._configs)()
        
        RSA_Human_folds = FSA_RSA.RSA_Human_folds(**self._configs)
        for used_unit_type in self.used_unit_types:
            for used_id_num in self.used_id_nums:
                RSA_Human_folds(used_unit_type=used_unit_type, used_id_num=used_id_num)
        
        for used_id_num in self.used_id_nums:
            RSA_Human_folds.process_all_used_unit_results(used_id_num=used_id_num, used_unit_types=self.used_unit_types)
        
    
    def CKA_folds(self, **kwargs):
        
        FSA_CKA.CKA_Similarity_Monkey_folds(**self._configs)()
        
        CKA_Human_folds = FSA_CKA.CKA_Similarity_Human_folds(**self._configs)
        for used_unit_type in self.used_unit_types:
            for used_id_num in self.used_id_nums:
                CKA_Human_folds(used_unit_type=used_unit_type, used_id_num=used_id_num)
        
        for used_id_num in self.used_id_nums:
            CKA_Human_folds.process_all_used_unit_results(used_id_num=used_id_num, used_unit_types=self.used_unit_types)
            
    
    def SVM_folds(self, **kwargs):
        
        used_unit_types = [
                           'a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne',
                           'a_s', 'a_m',
                           'qualified', 
                           'non_anova', 
                           'selective', 'high_selective', 'low_selective', 'non_selective'
                           ]
        
        FSA_SVM.FSA_SVM_folds(**self._configs)(used_unit_types=used_unit_types)

# ----------------------------------------------------------------------------------------------------------------------
def single_model_analysis(args):
    """
        used_id_num=10 is proved meaningless for CKA
    """
    start_time = time.time()
    
    # --- init
    config = f'{args.FSA_config}{args.fold_idx}'

    used_unit_types = [
                       'a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne',
                       'a_s', 'a_m',
                       'qualified', 
                       'non_anova', 
                       'selective', 'high_selective', 'low_selective', 'non_selective'
                       ]

    
    if 'fold' in args.FSA_config:
        FSA_folder = os.path.join(args.FSA_root, args.FSA_dir, f'FSA {args.FSA_config}/-_Single Models/FSA {config}')
    else:
        FSA_folder = os.path.join(args.FSA_root, args.FSA_dir, f'FSA {args.FSA_config}')
    
    _, layers, neurons, shapes = utils_.get_layers_and_units(args.model, 'act')

    # ----- start single model processing
    utils_.formatted_print(f'Listing model [{FSA_folder}] layers and neuron numbers')
    utils_.describe_model(layers, neurons, shapes)
    
    # ----- 1. ANOVA
    FSA_ANOVA.FSA_ANOVA(root=FSA_folder, layers=layers, neurons=neurons, alpha=args.alpha, num_classes=args.num_classes, num_samples=args.num_samples)() 

    # ----- 2. Encode
    Encode_analyzer = FSA_Encode.FSA_Encode(root=FSA_folder, layers=layers, neurons=neurons)

    Encode_analyzer.calculation_Encode()
    Encode_analyzer.plot_Encode_pct_bar_chart()
    Encode_analyzer.plot_Encode_freq()
    
    del Encode_analyzer

    Responses_analyzer = FSA_Responses.FSA_Responses(root=FSA_folder, layers=layers, neurons=neurons)

    Responses_analyzer.plot_unit_responses()
    Responses_analyzer.plot_stacked_responses(used_unit_types)
    Responses_analyzer.plot_responses_PDF()
    Responses_analyzer.plot_pct_pie_chart()
    Responses_analyzer.plot_Feature_Intensity()
    
    del Responses_analyzer
    
    SVM_analyzer = FSA_SVM.FSA_SVM(root=FSA_folder, layers=layers, neurons=neurons)
    SVM_analyzer.process_SVM(used_unit_types=used_unit_types)
    del SVM_analyzer

    # ----- 3. DR, DSM, Gram
    DSM_analyzer = FSA_DRG.FSA_DSM(root=FSA_folder, layers=layers, neurons=neurons)
    DSM_analyzer.calculation_DSM()
    del DSM_analyzer
    
    Gram_analyzer = FSA_DRG.FSA_Gram(root=FSA_folder, layers=layers, neurons=neurons)
    Gram_analyzer.calculation_Gram()
    Gram_analyzer.plot_Gram_intensity()
    del Gram_analyzer

    # ----- 4. RSA
    used_unit_types_ = ['qualified', 'a_hs', 'a_ls', 'a_hm', 'a_lm', 'a_ne', 'non_anova']
    
    FSA_RSA.RSA_Monkey(root=FSA_folder, layers=layers, neurons=neurons)() 
            
    # ---
    RSA_human = FSA_RSA.RSA_Human(root=FSA_folder, layers=layers, neurons=neurons)
    for used_unit_type in used_unit_types:
        for used_id_num in [50, 10]:
            RSA_human(used_unit_type=used_unit_type, used_id_num=used_id_num)

    for used_id_num in [50, 10]:
        RSA_human.process_all_used_unit_results(used_id_num=used_id_num, used_unit_types=used_unit_types_)
    
    del RSA_human
    
    # ----- 5. CKA
    FSA_CKA.CKA_Similarity_Monkey(root=FSA_folder, layers=layers, neurons=neurons)()

    CKA_human = FSA_CKA.CKA_Similarity_Human(root=FSA_folder, layers=layers, neurons=neurons)
    for used_unit_type in used_unit_types:
        for used_id_num in [50, 10]:
            CKA_human(used_unit_type=used_unit_type, used_id_num=used_id_num)
            
    for used_id_num in [50, 10]:
        CKA_human.process_all_used_unit_results(used_id_num=used_id_num, used_unit_types=used_unit_types_)
    
    del CKA_human
    
    # ----- 6. Feature --- deprecated
    #selectivity_feature_analyzer = Selectivity_Analysis_Feature.Selectivity_Analysis_Feature(FSA_folder, 'TSNE')
    #selectivity_feature_analyzer.feature_analysis()
    #del selectivity_feature_analyzer
    
    # --- 
    end_time = time.time()
    elapsed = end_time - start_time
    
    utils_.formatted_print(f"All results are saved in {os.path.join(FSA_folder, 'Analysis')}")
    utils_.formatted_print('Elapsed Time: {}:{:0>2}:{:0>2} '.format(int(elapsed/3600), int((elapsed%3600)/60), int((elapsed%3600)%60)))
    utils_.formatted_print('Experiment Done.')    



#FIXME
if __name__ == "__main__":
    
    utils_.formatted_print('Face Selectivity Analysis Experiment...')
    
# =============================================================================
#     for fold_idx in range(5):
#         args = get_args_parser(fold_idx)
#         print(args)
#         single_model_analysis(args)
# =============================================================================
    
    args = get_args_parser()
    Multi_Model_Analysis(args)
    
