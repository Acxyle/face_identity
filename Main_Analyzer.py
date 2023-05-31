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
"""

import os
import time
import argparse
import numpy as np

import spiking_vgg, spiking_resnet, sew_resnet
from spikingjelly.activation_based import surrogate, neuron, functional

import Selective_Neuron_ANOVA
import Selectivity_Analysis_Encode
import Selectivity_Analysis_SIMI
import Selectivity_Analysis_Add
import Selectivity_Analysis_Correlation

import SNNRAT_Resnet  # [notice] Vanilla SNN from SNNRAT
import vgg, resnet
import utils_


parser = argparse.ArgumentParser(description="Selectivity Analyzer Ver 2.1", add_help=True)
parser.add_argument("--num_classes", type=int, default=50, help="{Codelp] set the number of classes")
parser.add_argument("--num_samples", type=int, default=10, help="[Codelp] set the sample number of each class")
parser.add_argument("--alpha", type=float, default=0.01, help='[Codelp] assign the alpha value for ANOVA')
parser.add_argument("--root_dir", type=str, default="/media/acxyle/Data/ChromeDownload/", help="[Codelp] root directory for features and neurons")
parser.add_argument("--feature_folder" ,type=str, default='Identity_SEWResnet50_IF_CelebA2622_Results/', help="[Codelp] folder for features")
parser.add_argument("--model", type=str, default='sew_resnet50')
# [warning] 此 model 并不执行 train 等操作，只是一个名称（或者val一轮）以获取 layers 和 neurons，如果只获取 layers，则不会进行任何计算
parser.add_argument("--neuron", type=str, default='IF')
# [warning] 因此这个 neuron 也没有任何意义，目前的考虑是可以在后续将 featuremap 放进此代码后生效
parser.add_argument("--include_ANOVA", type=bool, default=True)
# [warning] 此处是为了测试此入口与其他函数的连接，后续debug中可将此部分置 default=False 

args = parser.parse_args()

output_folder = '_'.join([*args.feature_folder.split('_')[:-1], 'Neuron/'])

#FIXME
# waiting to rewrite
def get_layers_and_neurons(args):
    
    # [notice] wait to add more details
    if args.neuron == 'IF':
        neuron_ = neuron.IFNode
    elif args.neuron == 'LIF':
        neuron_ = neuron.LIFNode
    
    model_name = args.model
    if model_name == None:
        raise RuntimeError('[Codwarning] please type the correct model')
        
    # [notice] below is for spikingjelly SNN models
    elif 'spiking_vgg' in model_name.lower():
        spiking_model = spiking_vgg.__dict__[args.model](spiking_neuron=neuron_, num_classes=args.num_classes, surrogate_function=surrogate.ATan(), detach_reset=True)
        functional.set_step_mode(spiking_model, step_mode='m')
        layers, neurons, shapes = utils_.generate_vgg_layers(spiking_model, model_name)
    elif 'spiking_resnet' in model_name.lower():
        spiking_model = spiking_resnet.__dict__[args.model](spiking_neuron=neuron_, num_classes=args.num_classes, surrogate_function=surrogate.ATan(), detach_reset=True, mode='feature')
        functional.set_step_mode(spiking_model, step_mode='m') 
        layers, neurons, shapes = utils_.generate_resnet_layers_list(spiking_model, model_name.lower())
    elif 'sew_resnet' in model_name.lower():
        spiking_model = sew_resnet.__dict__[args.model](spiking_neuron=neuron_, num_classes=args.num_classes, surrogate_function=surrogate.ATan(), detach_reset=True, cnf='ADD', mode='feature')
        functional.set_step_mode(spiking_model, step_mode='m') 
        layers, neurons, shapes = utils_.generate_resnet_layers_list(spiking_model, model_name.lower())
    
    # below for vanilla SNN
    elif 'vgg5' in model_name.lower() and 'snnrat' in model_name.lower():
        layers = utils_.layer_list_vgg5_snnrat
        neurons = utils_.neuron_list_vgg5_snnrat
    elif 'vgg16' in model_name.lower() and 'snnrat' in model_name.lower():
        layers = utils_.layer_list_vgg16_snnrat
        neurons = utils_.neuron_list_vgg16_snnrat
    elif 'resnet18' in model_name.lower() and 'snnrat' in model_name.lower():
        model = SNNRAT_Resnet.Resnet(T=4)
        model.set_simulation_time(mode='bptt', input_decay=False, tau=1.0)
        layers, neurons = utils_.generate_resnet_layers_list_snnrat(model, model_name.lower())
        
    # for ANN
    elif 'resnet' in model_name.lower():
        model_ = resnet.__dict__[args.model](num_classes=args.num_classes)
        layers, neurons, shapes = utils_.generate_resnet_layers_list_ann(model_, args.model)
    # [notice] looks an ANN version VGG is required 
    elif 'vgg' in model_name.lower():
        model_ = vgg.__dict__[args.model](num_classes=args.num_classes)
        layers, neurons, shapes = utils_.generate_vgg_layers_list_ann(model_, args.model)
        
    if layers == None or neurons == None:
        raise RuntimeError('[Coderror] No layers or neurons obained')
        
    return layers, neurons, shapes     # [warning] only spikingjelly now
    
    
#FIXME
#在下版本中对每张重要图片添加标注
def Main_Analyzer(args):
    start_time = time.time()
    
    print('[Codinfo] Starting Selective Analysis Experiment...')
    print(args)

    num_classes = args.num_classes
    num_samples = args.num_samples
    
    feature_root = os.path.join(args.root_dir, args.feature_folder)
    idx_root = os.path.join(args.root_dir, output_folder)     # [default] this folder should be generated by ANOVA_analyzer
    layers, neurons, shapes = get_layers_and_neurons(args)
    
    #FIXME
    layers_ = [i for i in layers if 'neuron' in i or 'fc' in i or 'pool' in i]
    index_ = [layers.index(i) for i in layers_]
    neurons_ = [neurons[i] for i in index_]
    shapes_ = [shapes[i] for i in index_]
    
    layers = layers_
    neurons = neurons_
    shapes = shapes_
    
    print('[Codinfo] Listing model layers and neuron numbers')

    layers_info = [list(pair) for pair in zip(list(np.arange(len(layers)+1)), layers, neurons, shapes)]
    max_widths = [max(len(str(row[i])) for row in layers_info)+2 for i in range(len(layers_info[0]))]
    print("|".join("{:<{}}".format(header, max_widths[i]) for i, header in enumerate(["No.", "layer", "neurons", "shapes"])))
    print("-" * sum(max_widths))
    for row in layers_info:
        print("|".join("{:<{}}".format(str(item), max_widths[i]) for i, item in enumerate(row)))
    print("-" * sum(max_widths))
    
    # ----- Start Analyze
    # --- 1.
    if args.include_ANOVA == True:
        ANOVA_analyzer = Selective_Neuron_ANOVA.ANOVA_analyzer(feature_root, idx_root, alpha=args.alpha, num_classes=num_classes, num_samples=num_samples, layers=layers, neurons=neurons)
        ANOVA_analyzer.selectivity_neuron_ANOVA()
        ANOVA_analyzer.selectivity_neuron_ANOVA_plot()
        
    # --- 2.
    selectivity_analyzer = Selectivity_Analysis_Encode.Encode_feaquency_analyzer(feature_root, idx_root, num_classes=num_classes, num_samples=num_samples, layers=layers, neurons=neurons, 
                                                                                 mode=None
                                                                                 #mode='reload_encode_dict'
                                                                                 )
    
    selectivity_analyzer.obtain_encode_class_dict(single_neuron_test=True)
    selectivity_analyzer.draw_encode_frequency()
    selectivity_analyzer.draw_encode_frequency_for_each_layer()
    selectivity_analyzer.draw_merged_encode_frequency_for_each_layer()
    
    # --- 3.
    SIMI_Analyzer = Selectivity_Analysis_SIMI.Selectivity_Analysis_SIMI(feature_root, idx_root, num_classes=num_classes, num_samples=num_samples, layers=layers, neurons=neurons, status=False)
    
    SIMI_Analyzer.SIMI_SVM()
    SIMI_Analyzer.bins_percent_of_SIMI()
    SIMI_Analyzer.single_neuron_boxplot()
    
    # --- 4.
    selectivity_additional_analyzer = Selectivity_Analysis_Add.Selectiviy_Analysis_Additional(feature_root, idx_root, num_samples=num_samples, num_classes=num_classes, layers=layers, neurons=neurons, status=False,
                                                                                              #data_name = ''
                                                                                              data_name = 'CelebA'
                                                                                              )
    
    selectivity_additional_analyzer.selectivity_analysis_Tsne()
    selectivity_additional_analyzer.selectivity_analysis_distance()
    selectivity_additional_analyzer.selectivity_analysis_Correlation()
    
    # --- 5.  -> [notice] models comparison refer to Selectivity_Analysis_Correlation_plot.py
    corr_root = os.path.join(feature_root, 'Correlation/')
    selectivity_correlation_monkey_analyzer = Selectivity_Analysis_Correlation.Selectiviy_Analysis_Correlation_Monkey(corr_root=corr_root, layers=layers)
    selectivity_correlation_monkey_analyzer.monkey_neuron_analysis()
    
    selectivity_correlation_human_analyzer = Selectivity_Analysis_Correlation.Selectiviy_Analysis_Correlation_Human(corr_root=corr_root, layers=layers)
    #selectivity_correlation_human_analyzer.human_neuron_get_firing_rate()
    selectivity_correlation_human_analyzer.human_neuron_analysis(used_ID='top50')
    selectivity_correlation_human_analyzer.human_neuron_analysis(used_ID='top10')
    
    # --- 6. feature coding (optional)
    # [notice] working...
    
    
    # --- 
    print('[Codinfo] All data and figures saved in {}'.format(output_folder))
    
    end_time = time.time()
    elapsed = end_time - start_time
    print('[Codinfo] Elapsed Time: {}:{:0>2}:{:0>2} '.format(int(elapsed/3600), int((elapsed%3600)/60), int((elapsed%3600)%60)))
    print('[Codinfo] Experiment Done.')
        
if __name__ == "__main__":
    Main_Analyzer(args)
