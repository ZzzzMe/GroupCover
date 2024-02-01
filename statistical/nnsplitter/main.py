import os.path as osp
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from utils import *
from train import Trainer
from knockoff import datasets
import knockoff.models.zoo as zoo
from controller_rnn import *
from utils import *
from torch.utils.data import Dataset, DataLoader, Subset


class DictToArgs:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def find_max_and_min_weights(model, percentile=0.001):
    all_weights = []
    for param in model.parameters():
        if param.requires_grad:
            all_weights.extend(param.data.view(-1).cpu().numpy())
    all_weights = torch.tensor(all_weights)
    sorted_weights, _ = torch.sort(all_weights)
    n = len(sorted_weights)
    lower_idx = int(n * (percentile / 100))
    upper_idx = int(n * (1 - (percentile / 100)))

    smallest_weight = sorted_weights[lower_idx - 1] if lower_idx > 0 else sorted_weights[0]
    largest_weight = sorted_weights[upper_idx] if upper_idx < n else sorted_weights[-1]
    return smallest_weight.item(), largest_weight.item()
def find_approx_most_populated_range(net, eps):
    all_weights = []
    for param in net.parameters():
        if param.requires_grad:
            all_weights.extend(param.data.view(-1).cpu().numpy())

    all_weights = np.array(all_weights)
    min_weight = all_weights.min()
    max_weight = all_weights.max()
    bins = np.arange(min_weight, max_weight + eps, eps)
    hist, bin_edges = np.histogram(all_weights, bins=bins)

    max_count_idx = np.argmax(hist)
    most_populated_bin = bin_edges[max_count_idx]

    return most_populated_bin, hist[max_count_idx]
def main():

    parser = argparse.ArgumentParser('nn splitter model')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('--batch_size_rl', type=int, default=5)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=25)
    parser.add_argument('--max_val_iter', type=int, default=2)
    parser.add_argument('--filter_flag', type=int, default=1) # 0 - conv filter; 1 - all filters; 2 - conv para; 3- all para
    # parser.add_argument('--net', type=int, default=1) # 0 - vgg11_bn; 1 - resnet18; 2 - mobilenetv2
    # parser.add_argument('--lmd', type=float, default=0)
    parser.add_argument('--lr_rl', type=float, default=0.01)
    parser.add_argument('--num_epoch_rl', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_epoch_cnn', type=int, default=100)
    parser.add_argument('--eps', type=float, default=8e-5)
    parser.add_argument('--min_w', type=float, default=-0.14)
    parser.add_argument('--max_w', type=float, default=0.24)
    parser.add_argument('--b', type=float, default=0.0012)
    parser.add_argument('--PATH', type=str, default='models/nnsplitter/result')
    parser.add_argument('--pretrained', type=str, default='imagenet_for_cifar')
    args = parser.parse_args()
    params = vars(args)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dataset_name = params['testdataset']
    dataset = datasets.__dict__[dataset_name]
    args.PATH = os.path.join(args.PATH)
    print(args.PATH)

    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    trainset = dataset(train=True, transform=train_transform)
    testset = dataset(train=False, transform=test_transform)
    num_classes = len(trainset.classes)

    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=10, pin_memory=True)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=10, pin_memory=True)

    model_name = params['model_arch']
    # # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    checkpoint_path = f'models/victim/{dataset_name}-{model_name}/checkpoint.pth.tar'
    net = zoo.get_net(model_name, modelfamily, pretrained=params['pretrained'], num_classes=num_classes).cuda()
    checkpoint = torch.load(checkpoint_path)
    cp_epoch = checkpoint['epoch']
    best_test_acc = checkpoint['best_acc']
    state_dict = checkpoint['state_dict']
    # if "alexnet" in model_name or "vgg16_bn" in model_name:
    #     new_state_dict = {}
    #     for key in state_dict:
    #         new_key = key
    #         print(key)
    #         if key == 'last_linear.weight':
    #             new_key = 'classifier.weight'
    #         elif key == 'last_linear.bias':
    #             new_key = 'classifier.bias'
    #     new_state_dict[new_key] = state_dict[key]
    #     net.load_state_dict(new_state_dict)
    # else:
    #     net.load_state_dict(state_dict)
    net.load_state_dict(state_dict)
    print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(cp_epoch, best_test_acc))   


    layer_list = get_conv_layer_list(net)
    if (params['max_w'] == 0.24 and params['min_w']==-0.14):
        args.min_w,args.max_w = find_max_and_min_weights(net)
    device = torch.device('cuda')
    print(args)
    layer_dict = {
        'vgg11_bn':[64, 128, 256, 256, 512, 512, 512, 512],
        'resnet18':[64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512],
        'vgg16':[64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
        'vgg16_bn':[64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
        'resnet50':[64, 256, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048],
        'alexnet':[64, 192, 384, 256, 256]
    }


    lmd=[0.0]
    b = args.b
    if(b==0.0012):
        b,count = find_approx_most_populated_range(net,args.eps)
        print(count)
    B_list = [b]
    for i in range(len(lmd)):
        print('\n=============== Training with lambda = %s '%(lmd[i]))   
        if args.filter_flag < 2:
            control = Controller_rnn(device, layer_list, B_list)
            control.to(device)
            loss_list, record = control.policy_gradient(args, lmd[i], train_loader, test_loader,net)
            print('loss = %s\n' % loss_list)
            print('acc = %s\n' % record[0])
            print('change = %s\n' % record[-1])

if __name__ == '__main__':
    main()
