#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import json
import os
import os.path as osp
import pickle
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader

import knockoff.config as cfg
import knockoff.utils.model as model_utils
from knockoff import datasets
import knockoff.models.zoo as zoo

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils.data import Dataset, DataLoader, Subset

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform


class TransferSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def samples_to_transferset(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    if isinstance(sample_x, str):
        return TransferSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))
def find_max_similarity(pre_model, modified_model):
    pre_model = pre_model.cuda()
    modified_model = modified_model.cuda()


    similarity_results = {}

    for layer_name, pre_layer in pre_model.named_modules():
        if isinstance(pre_layer, nn.Conv2d):
            mod_layer = dict(modified_model.named_modules())[layer_name]
            assert pre_layer.weight.shape == mod_layer.weight.shape

            out_channels, in_channels, kernel_height, kernel_width = pre_layer.weight.shape
            max_sim_index = []
            tmp_co_num = 0
            pre_weights = torch.zeros((in_channels, out_channels * kernel_height * kernel_width)).cuda()
            mod_weights = torch.zeros((in_channels, out_channels * kernel_height * kernel_width)).cuda()

            for i in range(in_channels):
                max_sim_values = -1
                max_index = -1
                pre_weight = pre_layer.weight[:, i, :, :].view(out_channels, -1).flatten()
                pre_weights[i] = pre_weight
                mod_weight = mod_layer.weight[:, i, :, :].view(out_channels, -1).flatten()
                mod_weights[i] = mod_weight
            for i in range(in_channels):
                mod_weight = mod_weights[i].unsqueeze(0)
                cos_sim = torch.nn.functional.cosine_similarity(pre_weights, mod_weight, dim=1)
                max_index = torch.argmax(cos_sim).item()
                max_sim_index.append(max_index)

            similarity_results[layer_name] = max_sim_index
            # print(layer_name, tmp_co_num / in_channels, similarity_results[layer_name])

    return similarity_results

def restore_scaling(pre_model, modified_model, scale_factor=3.5, re_factor=0.01):
    pre_model = pre_model.cuda()
    modified_model = modified_model.cuda()

    for layer_name, pre_layer in pre_model.named_modules():
        if isinstance(pre_layer, nn.Conv2d):
            mod_layer = dict(modified_model.named_modules())[layer_name]
            assert pre_layer.weight.shape == mod_layer.weight.shape

            out_channels, in_channels, kernel_height, kernel_width = pre_layer.weight.shape
            ra = pre_layer.weight[:, :, :, :].mean()/mod_layer.weight[:, :, :, :].mean()
            for i in range(in_channels):
                pre_mean = pre_layer.weight[:, i, :, :].mean()
                mod_mean = mod_layer.weight[:, i, :, :].mean()

                if abs(mod_mean) > scale_factor * abs(pre_mean):
                    with torch.no_grad():
                        mod_layer.weight.data[:, i, :, :] *= re_factor
    
    return modified_model
def log_training_results(file_path, epoch, loss, accuracy):
    with open(file_path, 'a') as f:
        f.write(f"{epoch}\t{loss}\t{accuracy}\n")

def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')

    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('--budgets', metavar='B', type=str,
                        help='Comma separated values of budgets. Knockoffs will be trained for each budget.')
    # Optional arguments
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    parser.add_argument('--clip', type=bool, default=None, metavar='N',
                        help='use grad clip')
    # Attacker's defense
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm', choices=('sgd', 'sgdm', 'adam', 'adagrad'))

    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model_dir = params['model_dir']
    epoch = params['epochs']
    # ----------- Set up transferset
    transferset_path = osp.join(model_dir, 'transferset.pickle')
    with open(transferset_path, 'rb') as rf:
        transferset_samples = pickle.load(rf)
    num_classes = transferset_samples[0][1].size(0)
    print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples), num_classes))

    # ----------- Clean up transfer (if necessary)
    if params['argmaxed']:
        new_transferset_samples = []
        print('=> Using argmax labels (instead of posterior probabilities)')
        for i in range(len(transferset_samples)):
            x_i, y_i = transferset_samples[i]
            argmax_k = y_i.argmax()
            y_i_1hot = torch.zeros_like(y_i)
            y_i_1hot[argmax_k] = 1.
            new_transferset_samples.append((x_i, y_i_1hot))
        transferset_samples = new_transferset_samples

    # ----------- Set up testset
    dataset_name = params['testdataset']
    valid_datasets = datasets.__dict__.keys()
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transform)
    if len(testset.classes) != num_classes:
        raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=10, pin_memory=True)
    else:
        test_loader = None
    # ----------- Set up model & statistical attack
    model_name = params['model_arch']
    pretrained = params['pretrained']
    # # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    pre_model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
    modified_model = zoo.get_net(model_name, modelfamily,pretrained=pretrained ,num_classes=num_classes)
    our_path = (f'models/ourscheme/{dataset_name}-{model_name}.pth')
    modified_dict = torch.load(our_path)
    print('-'*10,modified_dict['arch'],modified_dict['best_acc'])

    modified_model.load_state_dict(modified_dict['state_dict'])
    model = modified_model.to(device)

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
    print('our model: [Test]  Epoch: {}\tAcc: {:.1f}% ({}/{})'.format(epoch, acc,
                                                                             correct, total))


    # de_shuffle = find_max_similarity(pre_model, modified_model)
    # deshuffled_model = restore_shuffling(modified_model,de_shuffle)
    # total = 0
    # correct = 0
    # with torch.no_grad():
    #     for batch_idx, (inputs, targets) in enumerate(test_loader):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = deshuffled_model(inputs)
    #         nclasses = outputs.size(1)
    #         _, predicted = outputs.max(1)
    #         total += targets.size(0)
    #         correct += predicted.eq(targets).sum().item()
    # acc = 100. * correct / total
    # print('de-shuffle model without budget: [Test]  Epoch: {}\tAcc: {:.1f}% ({}/{})'.format(epoch, acc,
    #                                                                          correct, total))

    model = restore_scaling(pre_model,model)
    # total = 0
    # correct = 0
    # device = torch.device('cuda')
    # model.to(device)
    # with torch.no_grad():
    #     for batch_idx, (inputs, targets) in enumerate(test_loader):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = model(inputs)
    #         nclasses = outputs.size(1)
    #         _, predicted = outputs.max(1)
    #         total += targets.size(0)
    #         correct += predicted.eq(targets).sum().item()

    # acc = 100. * correct / total
    # print('after de-shuffle and de-scaled model without budget: [Test]  Epoch: {}\tAcc: {:.1f}% ({}/{})'.format(epoch, acc, correct, total))
                                   



    # ----------- Train
    budgets = [int(b) for b in params['budgets'].split(',')]
    
    for b in budgets:
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        transferset = samples_to_transferset(transferset_samples, budget=b, transform=transform)
        print()
        print('=> Training at budget = {}'.format(len(transferset)))

        optimizer = get_optimizer(model.parameters(), params['optimizer_choice'], **params)
        print(params)

        checkpoint_suffix = '.{}'.format(b)
        criterion_train = model_utils.soft_cross_entropy
        outputs_path = f'models/ourscheme/log-{model_name}-{dataset_name}'
        model_utils.train_model(model, transferset, outputs_path,testset=testset, criterion_train=criterion_train,
                                checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer ,**params)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(model_dir, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)



if __name__ == '__main__':
    main()
