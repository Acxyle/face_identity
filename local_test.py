"""
    Author: FANG Wei
    
    Study time: Jan 20, 2024
    
    [notice] can not use the modified forward() of container.py in pytorch when execute the conversion process
"""

import torch
import torchvision
from tqdm import tqdm
import spikingjelly.activation_based.ann2snn as ann2snn
from spikingjelly.activation_based import monitor, neuron, functional, layer
from spikingjelly.activation_based import tensor_cache

import numpy as np
import os
import pickle
from tqdm import tqdm

from torchvision.models import vgg16

def tensor_memory(x: torch.Tensor):
    return x.element_size() * x.numel()

def ann_val(net, device, data_loader, T=None):
    net.eval().to(device)
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(data_loader)):
            
            img = img.to(device)

            out = net(img)
           
            correct += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            total += out.shape[0]
            
        acc = correct / total
        print('ANN Val Acc: %.3f' % (acc))
        
    return acc

def snn_val(net, device, data_loader, T):
    
    net.eval().to(device)
    correct = 0.0
    total = 0.0
    
    #feature_dict = {}
    vol_dict = {}
    
    for m in net.modules():
        if isinstance(m, neuron.IFNode):
            m.store_v_seq = True
    
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            
            #spike_seq_monitor = monitor.OutputMonitor(net, neuron.IFNode)     # for big model, RAM consuming
            v_seq_monitor = monitor.AttributeMonitor('v', pre_forward=False, net=net, instance=neuron.IFNode)
    
            for t in range(T):
                if t == 0:
                    out = net(img)
                else:
                    out += net(img)     # why this operation? plus?
                    
            # --- reset all module with memory
            for m in net.modules():
                if hasattr(m, 'reset'):
                    m.reset()
                    
            correct += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            total += out.shape[0]
            
            #records = [_.detach().squeeze().cpu() for _ in spike_seq_monitor.records]
            #records = [ tensor_cache.float_spike_to_bool(torch.stack([records[_*15+l] for _ in range(T)]).reshape(T, -1)) for l in range(15) ]
            
            #feature_dict[batch] = records
            
            #spike_seq_monitor.remove_hooks()
            #del spike_seq_monitor
            
            records = [_.detach().squeeze().cpu() for _ in v_seq_monitor.records]
            records = [ tensor_cache.float_spike_to_bool(torch.stack([records[_*15+l] for _ in range(T)]).reshape(T, -1)) for l in range(15) ]
            
            vol_dict[batch] = records
            
            v_seq_monitor.remove_hooks()
            del v_seq_monitor
            
        acc = correct / total
        print('\nSNN Val Acc: %.3f' % (acc))
        
# =============================================================================
#     print('Saving spike_train...')
#     
#     for layer_idx in tqdm(range(15), desc='layers'):
#         spike_train_layer = [feature_dict[_][layer_idx] for _ in range(500)]
#         with open(f'Features(spike)/{layer_idx}.pkl', 'wb') as f:
#             pickle.dump(spike_train_layer, f, protocol=-1)
# =============================================================================
        
    return acc

def main(T=32):
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda'
    dataset_dir = '/home/acxyle-workstation/Dataset/CelebA50'
    batch_size = 1

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.PILToTensor(),
        torchvision.transforms.ConvertImageDtype(torch.float),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    model = torchvision.models.vgg16(num_classes=50)
    model.load_state_dict(torch.load('/home/acxyle-workstation/Downloads/Face Identity Baseline/pth/params.pth'))
    
    # -----
    train_data_dataset = torchvision.datasets.ImageFolder(
        root=dataset_dir,
        transform=transform)
    
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
    
    test_data_dataset = torchvision.datasets.ImageFolder(
        root=dataset_dir,
        transform=transform)
    
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False)
    # -----

    print('ANN accuracy:')
    ann_val(model, device, test_data_loader)
    
    print('Converting...')
    model_converter = ann2snn.Converter(mode='Max', dataloader=train_data_loader, fuse_flag=False)
    snn_model = model_converter(model)
    
    print('SNN accuracy:')
    snn_acc = snn_val(snn_model, device, test_data_loader, T=T)
    
    return snn_acc


if __name__ == '__main__':
    main(T=64)
    
    print('6')

