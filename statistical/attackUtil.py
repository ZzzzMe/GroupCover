import argparse
import os.path as osp
import os
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

def get_conv_layers(model):
    layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    return layers
def compute_statistics(weight):
    mean = weight.mean().item()
    std = weight.std().item()
    return mean, std


def replace_with_most_similar(model, modified_model):
    model = model.cuda()
    modified_model = modified_model.cuda()

    conv_layers1 = get_conv_layers(model)
    conv_layers2 = get_conv_layers(modified_model)

    for conv1, conv2 in zip(conv_layers1, conv_layers2):
        with torch.no_grad(): 

            for i, w1_tensor in enumerate(conv1.weight):
                max_similarity = -1
                max_similar_tensor = None

                for j, w2_tensor in enumerate(conv2.weight):
                    w1_flat = w1_tensor.view(-1)
                    w2_flat = w2_tensor.view(-1)
                    similarity = torch.dot(w1_flat, w2_flat) / (torch.norm(w1_flat) * torch.norm(w2_flat))

                    if similarity > max_similarity:
                        max_similarity = similarity
                        max_similar_tensor = w2_tensor

                if max_similar_tensor is not None:
                    conv1.weight.data[i] = max_similar_tensor.clone()
                    # print(i,conv2.index(max_similar_tensor))

    return model





def modify_conv_layers(model, modify_ratio=0.5):
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


