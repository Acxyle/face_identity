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

import FSA_ANOVA, FSA_Encode, FSA_DRG, FSA_RSA, FSA_CKA
#import Selectivity_Analysis_Feature

import utils_


# ======================================================================================================================
def get_args_parser():
    parser = argparse.ArgumentParser(description="FSA Ver 5.1", add_help=True)
    
    parser.add_argument("--num_classes", type=int, default=50, help="[Codelp] set the number of classes")
    parser.add_argument("--num_samples", type=int, default=10, help="[Codelp] set the sample number of each class")
    parser.add_argument("--alpha", type=float, default=0.01, help='[Codelp] assign the alpha value for ANOVA')
    
    parser.add_argument("--FSA_root", type=str, default="/home/acxyle-workstation/Downloads/FSA", help="[Codelp] root directory for features and neurons")
    
    parser.add_argument("--FSA_dir", type=str, default='Resnet/SEWResnet')
    parser.add_argument("--FSA_config", type=str, default='SEWResnet152_IF_ATan_T4_C2k_fold_')
    parser.add_argument("--fold_idx", type=int, default=1)

    parser.add_argument("--model", type=str, default='sew_resnet152')     
    
    return parser.parse_args()
    

# ----------------------------------------------------------------------------------------------------------------------
# FIXME - under construction...
class Multi_Model_Analysis():
    
    def __init__(self, args, num_folds=5, **kwargs):

        self.num_folds = num_folds
        self.root = os.path.join(args.FSA_root, args.FSA_dir, f'FSA {args.FSA_config}')
        
        self.model_structure = args.FSA_config.replace('C2k_fold_', '')
        
        _, self.layers, self.neurons, self.shapes = utils_.get_layers_and_units(args.model, 'act')
        
        self.used_unit_types = ['selective', 'strong_selective', 'weak_selective', 'non_selective']
        
        self._configs = {
            'num_folds': self.num_folds,
            'root': self.root,
            'layers': self.layers,
            'neurons': self.neurons
            }
        
    def ANOVA_folds(self, **kwargs):
    
        FSA_ANOVA.FSA_ANOVA_folds(**self._configs)()
        
        
    def Encode_folds(self, **kwargs):
        
        FSA_Encode_folds = FSA_Encode.FSA_Encode_folds(**self._configs)
        FSA_Encode_folds(**kwargs)
        FSA_Encode_folds(used_unit_types=self.used_unit_types, **kwargs)
        
    
    def RSA_folds(self, **kwargs):
        
        #first_corrs = ['euclidean', 'pearson', 'spearman', 'mahalanobis', 'concordance']
        first_corrs = ['euclidean', 'pearson', 'spearman', 'mahalanobis']
        second_corrs = ['pearson', 'spearman', 'concordance']
        
        #RSA_unit_types = self.used_unit_types
        RSA_unit_types = ['qualified', 'selective', 'non_selective', 'legacy']
        
        RSA_Monkey_folds = FSA_RSA.RSA_Monkey_folds(**self._configs)
        for first_corr in first_corrs:
            for second_corr in second_corrs:
                RSA_Monkey_folds(first_corr=first_corr, second_corr=second_corr, route='p')
        
        RSA_Human_folds = FSA_RSA.RSA_Human_folds(**self._configs)
        for first_corr in first_corrs:
            for second_corr in second_corrs:
                for used_unit_type in RSA_unit_types:
                    for used_id_num in [50, 10]:
                        RSA_Human_folds(first_corr=first_corr, second_corr=second_corr, used_unit_type=used_unit_type, used_id_num=used_id_num, route='p')
        
    
    def CKA_folds(self, **kwargs):
        
        CKA_unit_types = ['qualified', 'selective', 'non_selective', 'legacy']
        
        # --- Monkey
        FSA_CKA.CKA_Similarity_Monkey_folds(**self._configs)(kernel='linear')
        #for threshold in [1.0, 10.0]:
        #    CKA_Similarity_Monkey_folds(**self._configs)(kernel='rbf', threshold=threshold)
        
        # --- Human
        for used_unit_type in CKA_unit_types:
            FSA_CKA.CKA_Similarity_Human_folds(**self._configs)(kernel='linear', used_unit_type=used_unit_type, used_id_num=50)
            #for threshold in [1.0, 10.0]:
            #    CKA_Similarity_Human_folds(**self._configs)(kernel='rbf', threshold=threshold, used_unit_type=used_unit_type, used_id_num=50)
        

# ----------------------------------------------------------------------------------------------------------------------
def single_model_analysis(args, thresholds=[1.0, 10.0]):
    """
        used_id_num=10 is proved meaningless for CKA
    """
    start_time = time.time()
    
    # --- init
    config = f'{args.FSA_config}{args.fold_idx}'
    
    if 'fold' in args.FSA_config:
        FSA_folder = os.path.join(args.FSA_root, args.FSA_dir, f'FSA {args.FSA_config}/-_Single Models/FSA {config}')
    else:
        FSA_folder = os.path.join(args.FSA_root, args.FSA_dir, f'FSA {args.FSA_config}')
    
    _, layers, neurons, shapes = utils_.get_layers_and_units(args.model, 'act')

    # ---
    utils_.formatted_print(f'Listing model [{FSA_folder}] layers and neuron numbers')
    utils_.describe_model(layers, neurons, shapes)
    
    # ----- 1. ANOVA
    FSA_ANOVA.FSA_ANOVA(root=FSA_folder, layers=layers, neurons=neurons, alpha=args.alpha, num_classes=args.num_classes, num_samples=args.num_samples)() 

    # ----- 2. Encode
    Encode_analyzer = FSA_Encode.FSA_Encode(root=FSA_folder, layers=layers, neurons=neurons)
    Encode_analyzer.calculation_Encode()
    Encode_analyzer.plot_Encode_pct_single()
    Encode_analyzer.plot_Encode_pct_comprehensive()
    Encode_analyzer.plot_Encode_freq()
    del Encode_analyzer

    Responses_analyzer = FSA_Encode.FSA_Responses(root=FSA_folder, layers=layers, neurons=neurons)
    Responses_analyzer.plot_unit_responses()
    Responses_analyzer.plot_stacked_responses(num_types=5)
    Responses_analyzer.plot_responses_PDF()
    Responses_analyzer.plot_pct_pie_chart()
    Responses_analyzer.plot_Feature_Intensity(used_unit_types=['qualified', 'selective', 'strong_selective', 'weak_selective', 'non_selective'])
    del Responses_analyzer
    
    #SVM_analyzer = FSA_Encode.FSA_SVM(root=FSA_folder, layers=layers, neurons=neurons)
    #SVM_analyzer.process_SVM()
    #del SVM_analyzer

    # ----- 3. DR, DSM, Gram
    #DR_analyzer = FSA_DRG.FSA_DR(root=FSA_folder, layers=layers, neurons=neurons)
    #DR_analyzer.DR_TSNE()
    #del DR_analyzer
    
    #DSM_analyzer = FSA_DRG.FSA_DSM(root=FSA_folder, layers=layers, neurons=neurons)
    #DSM_analyzer.process_DSM(metric='pearson')
    #del DSM_analyzer
    
    Gram_analyzer = FSA_DRG.FSA_Gram(root=FSA_folder, layers=layers, neurons=neurons)
    Gram_analyzer.calculation_Gram(kernel='linear', normalize=True)
    Gram_analyzer.plot_Gram_intensity(kernel='linear')
    #for threshold in thresholds:
    #    Gram_analyzer.calculation_Gram(kernel='rbf', threshold=threshold)
    del Gram_analyzer

    # ----- 4. RSA
    #for monkey experiments
    RSA_monkey = FSA_RSA.RSA_Monkey(root=FSA_folder, layers=layers, neurons=neurons)
    
    for first_corr in ['euclidean', 'pearson', 'spearman', 'mahalanobis', 'concordance']:
        for second_corr in ['pearson', 'spearman', 'concordance']:
            RSA_monkey(first_corr=first_corr, second_corr=second_corr)
            
    del RSA_monkey
            
    # for human experiments 
    RSA_human = FSA_RSA.RSA_Human(root=FSA_folder, layers=layers, neurons=neurons)
    
    for firsct_corr in ['euclidean', 'pearson', 'spearman', 'mahalanobis']:
        for second_corr in ['pearson', 'spearman', 'concordance']:
            for used_unit_type in ['legacy', 'qualified', 'selective', 'non_selective']:
                for used_id_num in [50, 10]:
                    RSA_human(first_corr=firsct_corr, second_corr=second_corr, used_unit_type=used_unit_type, used_id_num=used_id_num)
    
    del RSA_human
    
    # ----- 5. CKA
    CKA_monkey = FSA_CKA.CKA_Similarity_Monkey(root=FSA_folder, layers=layers, neurons=neurons)
    CKA_monkey(kernel='linear', normalize=True)
    #for threshold in thresholds:
    #    CKA_monkey(kernel='rbf', threshold=threshold, normalize=True)
    
    CKA_human = FSA_CKA.CKA_Similarity_Human(root=FSA_folder, layers=layers, neurons=neurons)
    for used_unit_type in ['legacy', 'qualified', 'selective', 'non_selective']:
        CKA_human(kernel='linear', used_unit_type=used_unit_type, normalize=True)
        #for threshold in thresholds:
        #    CKA_human(kernel='rbf', threshold=threshold, used_unit_type=used_unit_type, normalize=True)
    
    
    # ----- 6. Feature
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
    
    utils_.formatted_print('Face Identity Selective Analysis Experiment...')
    
    args = get_args_parser()
    
    single_model_analysis(args)
    
    #multi_model_analysis = Multi_Model_Analysis(args)
    #multi_model_analysis.CKA_folds()
    
