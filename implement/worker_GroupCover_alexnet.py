import os
import numpy as np
import torch
import torch.distributed.rpc as rpc
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import time  
import pickle
from knockoff import datasets
import knockoff.utils.transforms as transform_utils
import knockoff.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
import knockoff.models.zoo as zoo


params = {
    'dataset': 'CIFAR10',
    'model_arch':'alexnet',
    'out_path': 'output',
    'pretrained':'imagenet_for_cifar',
    'device_id': 0,
    'batch_size': 128
}



def worker1_process_layer(x, layer_fullname, model):
    layer = dict(model.named_modules())[layer_fullname]
    if torch.cuda.is_available():
        device = torch.device("cuda") 
        layer.to(device) 
        x = x.to(device) 
    else:
        device = torch.device("cpu")     
    with torch.no_grad():
        x = layer(x)
    x = x.to(torch.device("cpu"))
    return x

def worker0_alex_infer_in_tee(model,x,restore_params):
    # rpc_times = [] 
    with torch.no_grad():
        # start_time = time.time()
        for fullname, layer in model.named_modules():
            if not fullname or isinstance(layer, nn.Sequential):
                continue
            if fullname.startswith("classifier"):
                x = x.view(x.size(0), -1)
                # print("Applying flatten")
            if isinstance(layer, (nn.Conv2d,nn.Linear)):
                # print(f'Sending {fullname} to worker1')
                # start_rpc_time = time.time()  
                x += restore_params[fullname]['r']
                x = rpc.rpc_sync("worker1", worker1_process_layer, args=(x, fullname, model))
                original_shape = x.shape
                C = original_shape[1]
                # N,C,H,W = original_shape
                x -= restore_params[fullname]['Wr']
                # x_reshaped = x.view(N, C, H*W)
                groups = restore_params[fullname]['group']
                cof_mat = restore_params[fullname]['cof']
                x = x.view(C, -1)
                cof_tensor = torch.stack([torch.tensor(cof_mat[i]) for i in range(C)]) 
                group_indices = torch.tensor(groups)  
                for i in range(C):
                    group_idx = group_indices[i]  
                    cof = cof_tensor[i]  
                    x[i, :] = torch.sum(x[group_idx, :] * cof[:, None], dim=0)  
                x = x.view(*original_shape)  #
                # end_rpc_time = time.time() 
                # rpc_duration = end_rpc_time - start_rpc_time
                # rpc_times.append(rpc_duration)  
                # print(f'RPC for {fullname} took {rpc_duration:.6f} seconds')
            else:
                # print(f'Executing {fullname} on worker0')
                x = layer(x)
        # total_time = time.time() - start_time
        # print(f"Total processing time: {total_time:.2f} seconds")
    # print(f'Average RPC time: {sum(rpc_times) / len(rpc_times):.6f} seconds')
    return x


def main():
    dataset_name = params['dataset']
    dataset = datasets.__dict__[dataset_name]
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    testset = dataset(train=False, transform=test_transform)
    params['num_classes'] = 10
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, num_workers=10, pin_memory=True)
    else:
        test_loader = None


    model_name = params['model_arch']
    pretrained = params['pretrained']

    checkpoint_path = f'models/victim/{dataset_name}-{model_name}/checkpoint.pth.tar'
    model_arch = params['model_arch']

    model = zoo.get_net(model_arch, modelfamily, pretrained=pretrained, num_classes=params['num_classes'])
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    best_test_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))


    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    
    with open('restore_params.pkl', 'rb') as f:
                restore_params = pickle.load(f)
    if rank == 0:
        
        no_RPC = True
        st = time.time()
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if inputs.shape[0]<params['batch_size']:
                break
            res = worker0_alex_infer_in_tee(model,inputs,restore_params)
        print('throughput '+'-'*10,(10000)/(time.time()-st))
    rpc.shutdown()

if __name__ == "__main__":
    main()
