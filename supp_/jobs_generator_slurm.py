#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:57:17 2023

@author: acxyle
"""

import os
import datetime

neuron_types = ['IF', 'LIF', 'ParametricLIF', 'QIF', 'EIF', 'Izhikevich']
activation_types = ['PiecewiseQuadratic', 'PiecewiseExp', 'Sigmoid', 'SoftSign', 'ATan', 
                    'NonzeroSignLogAbs', 'Erf', 'PiecewiseLeakyReLU', 'SquarewaveFourierSeries', 'S2NN', 
                    'QPseudoSpike', 'LeakyKReLU', 'FakeNumericalGradient', 'LogTailedReLU']

T = '4'
model_name = 'spiking_resnet18'
num_classes = '2622'

now_time = datetime.datetime.now()
formated_time = now_time.strftime("%Y-%m-%d_%H-%M-%S")

folder_path = f"slurm_jobs_{formated_time}/"
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

for i, neuron in enumerate(neuron_types):
    for j, activation in enumerate(activation_types):
        print(i, j, neuron, activation)
        if os.path.exists(folder_path+f"b{str(i+2)}-neuron_{neuron}-activation_{activation}.sh"):
            raise RuntimeWarning(folder_path+f"[Codwarning] file b{str(i+2)}-neuron_{neuron}-activation_{activation}.sh already exists, skipped")
            pass
        # Open the .sh file in write mode
        with open(folder_path+f"b{str(i+2)}-neuron_{neuron}-activation_{activation}.sh", "w") as f:
            # Write some text to the file
            f.write("#! /bin/bash\n")
            f.write("#SBATCH --account=bdlds21\n")
            f.write("#SBATCH --time=36:0:0\n")
            f.write("#SBATCH --partition=gpu\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --gres=gpu:1\n")
            
            # change the job name
            f.write(f"#SBATCH --job-name=b{str(i+2)}-neuron_{neuron}-activation_{activation}\n")
            
            # command
            f.write("#path\n")
            f.write("cd /nobackup/projects/bdlds21/acxyle/Git/spikingjelly-master/spikingjelly/activation_based/model/\n")
            f.write("#conda env\n")
            f.write("source /nobackup/projects/bdlds21/acxyle/miniconda/bin/activate\n")
            f.write("conda activate spikingjelly\n")
            
            f.write("#commands\n")
            f.write(f"python spikingjelly-trainer.py --T {T} --num_classes {num_classes} --neuron {neuron} --surrogate {activation} --data_path /nobackup/projects/bdlds21/acxyle/data/CelebA2622/ --model {model_name} --output_dir ./logs_{model_name}_{neuron}_{activation}_CelebA2622\n")
            
            f.write("#additionals\n")
            f.write("echo '-----'\n")
            f.write("echo 'end of job'\n")
            
        f.close()
        
