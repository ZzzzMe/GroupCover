import argparse
import os.path as osp
import os
import sys



from datetime import datetime
import json
from collections import defaultdict as dd
from pdb import set_trace as st
import time
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils.data import Dataset, DataLoader, Subset

import sys
import  copy
import random
import knockoff.config as cfg
from knockoff import datasets
import knockoff.models.zoo as zoo
import numpy as np
    # ----------- Set up dataset

import argparse





def get_conv_layers(model):
    layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    return layers



def modify_conv_layers(model, modify_ratio=0.1):
    modified_model = copy.deepcopy(model).cuda()
    modified_indices = {}
    shuffle_indices = {}

    for layer_name, layer in modified_model.named_modules():
        if isinstance(layer, nn.Conv2d):
            print(f'modifing layer {layer_name}')
            with torch.no_grad():
                out_channels, in_channels, kernel_height, kernel_width = layer.weight.shape

                
                total_kernels = in_channels
                num_kernels_to_modify = int(total_kernels * modify_ratio)  
                indices = torch.randperm(total_kernels)[:num_kernels_to_modify].cuda()

                modified_indices[f'{layer_name}'] = indices.cpu().numpy()
                    
                random_factors = (torch.rand(num_kernels_to_modify).cuda() * 5.0) + 1

                for i,idx in enumerate(indices):
                    layer.weight.data[:, idx, :, :] *= random_factors[i]

                perm = torch.randperm(total_kernels).cuda()
                shuffle_indices[f'{layer_name}'] = perm.cpu().numpy()
                layer.weight.data = layer.weight.data[:,perm, :, :]

    return modified_model, modified_indices, shuffle_indices


def main():
    
    parser = argparse.ArgumentParser(description='Train a model')   
    parser.add_argument('dataset', type=str, default='CIFAR10', help='Dataset to use')
    parser.add_argument('model_arch', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--out_path', type=str, default='output', help='Output path')
    parser.add_argument('--pretrained', type=str, default='imagenet_for_cifar', help='Use pretrained network')
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum')
    parser.add_argument('--log_interval', type=int, default=50, help='How many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, help='Path to latest checkpoint')
    parser.add_argument('--lr_step', type=int, default=60, help='Step sizes for LR')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='LR Decay Rate')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of worker threads to load data')
    parser.add_argument('--weighted_loss', action='store_true', help='Use a weighted loss')
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels')
    parser.add_argument('--optimizer_choice', type=str, default='sgdm', choices=['sgd', 'sgdm', 'adam', 'adagrad'], help='Optimizer choice')    
    parser.add_argument('--sigma', type=float, default=0.2, help='Scale param')    

    args = parser.parse_args()  

    params = vars(args) 


    dataset_name = params['dataset']
    valid_datasets = datasets.__dict__.keys()
    # if dataset_name not in valid_datasets:
    #     raise   ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    trainset = dataset(train=True, transform=train_transform)
    testset = dataset(train=False, transform=test_transform)
    num_classes = len(trainset.classes)
    params['num_classes'] = num_classes
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=10, pin_memory=True)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=10, pin_memory=True)
    else:
        test_loader = None  


    model_name = params['model_arch']
    pretrained = params['pretrained']
    model_arch = params['model_arch']
    epoch = params['epochs']
    # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)b
    # pre_model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
    # print(pre_model)  






    pre_model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes).cuda()
    checkpoint_path = f'models/victim/{dataset_name}-{model_name}/checkpoint.pth.tar'
    model = zoo.get_net(model_arch, modelfamily, pretrained=pretrained, num_classes=num_classes).cuda()
    checkpoint = torch.load(checkpoint_path)
    cp_epoch = checkpoint['epoch']
    best_test_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(cp_epoch, best_test_acc))   




    modified_model,modified_indices,shuffle_indices = modify_conv_layers(model,modify_ratio =  params['sigma'])
    # model_replaced = replace_with_most_similar(pre_model, modified_model)
    device = torch.device('cuda')
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            nclasses = outputs.size(1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()   

    acc = 100. * correct / total    

    print('Secret shadownet model: [Test]  Epoch: {}\tAcc: {:.1f}% ({}/{})'.format(epoch, acc,
                                                                             correct, total))
    
    device = torch.device('cuda')
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = modified_model(inputs)
            nclasses = outputs.size(1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()   

    acc = 100. * correct / total    

    print('shadownet model: [Test]  Epoch: {}\tAcc: {:.1f}% ({}/{})'.format(epoch, acc,
                                                                         correct, total))

    state = {
                    'epoch': epoch,
                    'arch': model.__class__,
                    'state_dict': modified_model.state_dict(),
                    'best_acc': acc,
                    'created_on': str(datetime.now()),
                    'modified_indices':modified_indices,
                    'shuffle_indices':shuffle_indices
                }
    torch.save(state, params['out_path'])   
if __name__ == '__main__':
    main()
