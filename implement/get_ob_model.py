import argparse
import os.path as osp
import os
import sys

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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



def cluster_vectors(vectors, cluster_size=4):
    index_pairs = np.array([np.array([i]) for i in range(len(vectors))])
    iter = 1
    while(iter<cluster_size):
        iter*=2
        cos_sim_matrix = cosine_similarity(vectors)
        sum_cos_sim_dis = np.mean(cos_sim_matrix, axis=0)
        sorted_indices = np.argsort(sum_cos_sim_dis)[::-1]
        np.fill_diagonal(cos_sim_matrix, np.inf)  
        pairs = []
        index_pair = []
        repeat_index = []
        for i in sorted_indices:
            if i in repeat_index:
                continue
            j = np.argmin(cos_sim_matrix[i])
            repeat_index.append(j)
            cos_sim_matrix[i, :] = np.inf
            cos_sim_matrix[:, i] = np.inf
            cos_sim_matrix[j, :] = np.inf
            cos_sim_matrix[:, j] = np.inf
            index_pair.append(np.concatenate((index_pairs[i], index_pairs[j])))
            pairs.append(np.mean([vectors[i], vectors[j]],axis=0))
        vectors = pairs
        index_pairs = index_pair 
        # print(index_pair) 
    return index_pairs

def modify_conv_layers(original_model, cluster_size=4):
    modified_model = copy.deepcopy(original_model)
    device = torch.device('cuda')
    restore_params = []
    
    for layer_name, layer in modified_model.named_modules():
        if isinstance(layer, nn.Conv2d):
            print(layer_name)

            with torch.no_grad():
                out_channels, in_channels, kernel_height, kernel_width = layer.weight.shape
                weights = torch.zeros((out_channels,in_channels * kernel_height * kernel_width))

                for i in range(out_channels):
                    mod_weight = layer.weight[i,:, :, :].view(in_channels, -1).flatten()
                    weights[i] = mod_weight
                # print("{:.10e}".format(layer.weight.data[0,1,0,0].item()))

                cluster_index = cluster_vectors(weights.detach().numpy(), cluster_size=cluster_size)
                random_coeff_list = [[] for _ in range(out_channels)]
                inv_A_list = []
                for idlist in cluster_index:
                    new_kernels = []
                    for i in idlist:
                        random_coeffs = np.random.randint(1, 100, size=cluster_size)
                        random_coeff_list[i] = random_coeffs

                        new_kernel = sum(coeff * layer.weight[idlist[j], : , :, :] for j, coeff in enumerate(random_coeffs))
                        new_kernels.append(new_kernel)

                    for index, idx in enumerate(idlist):
                        layer.weight.data[idx, :, :, :] = new_kernels[index]
                for idlist in cluster_index:
                    A = []
                    for i in idlist:
                        A.append(np.array(random_coeff_list[i]))
                    A = np.array(A, dtype=np.float64)
                    inv_A = np.linalg.inv(A)
                    inv_A_list.append(torch.tensor(inv_A,dtype=torch.float32))  
                
                perm = torch.randperm(in_channels)
                layer.weight.data = layer.weight.data[ :,perm, :, :]

                restore_params.append(
                    {
                        'shuffle_indices':perm,
                        'cluster_index':cluster_index,
                        'inv_A':inv_A_list
                    }
                )

    return modified_model, restore_params


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

    pre_model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes).cuda()
    checkpoint_path = f'models/victim/{dataset_name}-{model_name}/checkpoint.pth.tar'
    model = zoo.get_net(model_arch, modelfamily, pretrained=pretrained, num_classes=num_classes).cuda()
    checkpoint = torch.load(checkpoint_path)
    cp_epoch = checkpoint['epoch']
    best_test_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(cp_epoch, best_test_acc))   




    modified_model,restore_params = modify_conv_layers(model)
    print('-'*10,"need save params to restore the correct output y in TEE.")
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

    print('our obfuscated model: [Test]  Epoch: {}\tAcc: {:.1f}% ({}/{})'.format(epoch, acc,
                                                                         correct, total))

    state = {
                    'epoch': epoch,
                    'arch': model.__class__,
                    'state_dict': modified_model.state_dict(),
                    'best_acc': acc,
                    'created_on': str(datetime.now()),
                    "restore_params":restore_params
                }
    torch.save(state, params['out_path'])   
if __name__ == '__main__':
    main()
