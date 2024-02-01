'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import time
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import copy
from pdb import set_trace as st
import numpy as np
import torch.nn.functional as F


__all__ = ['alexnet']

class mod_conv(nn.Conv2d):
    def __init__(self, *args, restore_params=None,name = None, **kwargs):
        super(mod_conv, self).__init__(*args, **kwargs)
        self.restore_params = restore_params
        self.name = name


#  # """ this is  the  simulation of darknight """
#     def forward(self, x):
#         x = x[:, self.restore_params['shuffle_indices'], :, :]
#         y = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
#         y = y.to('cpu')

#         channels = y.size(1)
#         random_matrix = torch.rand(channels, channels)

#         original_shape = y.shape

#         y_flat = y.view(-1, channels)
#         result = torch.matmul(y_flat, random_matrix)
#         y = result.view(original_shape)

#         random_tensor = torch.ones_like(y)
#         y = y + random_tensor
#         if self.bias is not None:
#             bias = self.bias.view(1, self.out_channels, 1, 1).to('cpu')
#             y = y + bias

#         return y.to('cuda')
        

    """ this is  the  simulation of slalom """
    def forward(self, x):
        x = x[:, self.restore_params['shuffle_indices'], :, :]
        y = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        y = y.to('cpu')
        random_tensor = torch.ones_like(y)
        y = y+random_tensor
        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1).to('cpu')
            y = y + bias
        return y.to('cuda')

    # def forward(self, x):
    #     x = x[:, self.restore_params['shuffle_indices'], :, :]
    #     y = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation,   self.groups)
    #     y = y.to('cpu')
    #     cluster_index = self.restore_params['cluster_index']
    #     inv_A = self.restore_params['inv_A']
    #     for index, idlist in enumerate(cluster_index):
    #         Y = []
    #         for i in idlist:
    #             Y.append(y[:, i, :, :])
    #         Y_tensor = torch.stack(Y)  
    #         Y = Y_tensor.view(len(idlist), -1)
    #         revertY = torch.matmul(inv_A[index], Y)
    #         reverted_y_reshaped = revertY.view_as(Y_tensor)  
    #         for index, i in enumerate(idlist):
    #             y[:, i, :, :] = reverted_y_reshaped[index]

    #     # grouped_y = y.view(y.size(0), y.size(1) // 4, 4, y.size(2), y.size(3))
    #     # summed_y = grouped_y.sum(dim=2, keepdim=True)
    #     # expanded_summed_y = summed_y.repeat(1, 1, 4, 1, 1)
    #     # y = expanded_summed_y.view_as(y)
    #     # random_tensor = torch.ones_like(y)
    #     # y = y + random_tensor
    #     if self.bias is not None:
    #         bias = self.bias.view(1, self.out_channels, 1, 1).to('cpu')
    #         y = y + bias
    #     return y.to('cuda')
  

class AlexNet(nn.Module):

    def __init__(self, num_classes=10, img_size=32, restore_params=None):
        super(AlexNet, self).__init__()
        if restore_params!=None:
            self.features = nn.Sequential(
                # nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
                mod_conv(3, 64, restore_params = restore_params[0] ,name = "conv1",kernel_size=7, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                mod_conv(64, 192,restore_params = restore_params[1], name = "conv2",kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                mod_conv(192, 384, restore_params = restore_params[2],name = "conv3",kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                # new maxpool
                # nn.MaxPool2d(kernel_size=2, stride=2),
                mod_conv(384, 256, restore_params = restore_params[3],name = "conv4",kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                mod_conv(256, 256, restore_params = restore_params[4],name = "conv5", kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            self.features = nn.Sequential(
                # nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                # new maxpool
                # nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(256, 256,kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(256, num_classes)

        self.img_size=img_size
        self.restore_params = restore_params
        self.init_layer_config()
        self.config_block_params()
        self.config_block_flops()
        self.config_conv_layer_flops()

    def init_layer_config(self):
        self.forward_blocks = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                self.forward_blocks.append(name)
        self.forward_blocks.append("classifier")
        self.backward_blocks = copy.deepcopy(self.forward_blocks)
        self.backward_blocks.reverse()
        self.total_blocks = len(self.forward_blocks)
        print("Total blocks: ", self.total_blocks)
        self.forward_blocks.append('end')
        self.backward_blocks.append('start')
        print("Forward blocks: ", self.forward_blocks)
        print("Backward blocks: ", self.backward_blocks)

        self.parameter_names = []
        for name, _ in self.named_parameters():
            self.parameter_names.append(name)
        self.reverse_parameter_names = copy.deepcopy(self.parameter_names)
        self.reverse_parameter_names.reverse()
        # print("Forward parameters: ", self.parameter_names)
        # print("Backward parameters: ", self.reverse_parameter_names)
        

    def config_block_params(self):
        module_params = {}
        for name, module in self.named_modules():
            module_params[name] = 0
            for param in module.parameters():
                module_params[name] += np.prod(param.size())

        # ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20', 'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40', 'classifier', 'end']
        block_params = {}
        for bname in self.forward_blocks[:-1]:
            block_params[bname] = module_params[bname]
        block_name = None
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                block_name = name
            elif isinstance(module, nn.BatchNorm2d):
                block_params[block_name] += module_params[name]
                # print(f"{name} to {block_name}")

        self.forward_block_params = {}
        for idx, name in enumerate(self.forward_blocks):
            self.forward_block_params[name] = 0
            for prior_idx in range(idx):
                self.forward_block_params[name] += block_params[self.forward_blocks[prior_idx]]
        print("Forward block params: ", self.forward_block_params)
        
        self.backward_block_params = {}
        for idx, name in enumerate(self.backward_blocks):
            self.backward_block_params[name] = 0
            for prior_idx in range(idx):
                self.backward_block_params[name] += block_params[self.backward_blocks[prior_idx]]
        print("Backward block params: ", self.backward_block_params)

    def config_block_flops(self):
        self.block_flops = {}
        output_shape = self.img_size

        block_name = None
        for name, module in self.features.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.stride[0] > 1:
                    output_shape /= module.stride[0]
                block_name = f"features.{name}"
                print(f"{name} output {output_shape}")
                self.block_flops[block_name] = output_shape**2 * module.in_channels * module.out_channels * module.kernel_size[0]**2
            elif isinstance(module, nn.BatchNorm2d):
                self.block_flops[block_name] += output_shape**2 * module.num_features * 2
            elif isinstance(module, nn.MaxPool2d):
                output_shape /= module.stride
                print(f"{name} output {output_shape}")
        self.block_flops['classifier'] = self.classifier.in_features * self.classifier.out_features + self.classifier.out_features
        print("Block flops: ", self.block_flops)
        
        self.forward_block_flops = {}
        for idx, name in enumerate(self.forward_blocks):
            self.forward_block_flops[name] = 0
            for prior_idx in range(idx):
                self.forward_block_flops[name] += self.block_flops[self.forward_blocks[prior_idx]]
        print("Forward block flops: ", self.forward_block_flops)

        self.backward_block_flops = {}
        for idx, name in enumerate(self.backward_blocks):
            self.backward_block_flops[name] = 0
            for prior_idx in range(idx):
                self.backward_block_flops[name] += self.block_flops[self.backward_blocks[prior_idx]]
        print("Backward block flops: ", self.backward_block_flops)

    def config_conv_layer_flops(self):
        self.conv_layer_flops = self.block_flops


    # def forward(self, x):
    #     # print(x.shape)
    #     x = self.features(x)
    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)

    def forward(self, x):
        # print(x.shape)
        x = self.features(x)
        # for i in range(len(self.features)):
        #     if i in [0,3,6,8,10]:
        #         st = time.time()*1000
        #         x = self.features[i](x)
        #         et = time.time()*1000
        #         with open('ourscheme.txt', 'a') as file:
        #             file.write("conv{} +++ {}\n ".format(i,et-st))
        #     else:
        #         x = self.features[i](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
    if pretrained:
        ckp = model_zoo.load_url(url)
        state_dict = model.state_dict()
        ckp['classifier.weight'] = state_dict['classifier.weight']
        ckp['classifier.bias'] = state_dict['classifier.bias']
        if ckp['features.0.weight'].size(-1) != state_dict['features.0.weight'].size(-1):
            # st()
            ckp['features.0.weight'] = ckp['features.0.weight'][:,:,2:-2,2:-2]
        model.load_state_dict(ckp, strict=False)
    return model




    # def forward(self, x):
    #     x = x[:, self.restore_params['shuffle_indices'], :, :]
    #     y = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
    #     y = y.to('cpu')
    #     cluster_index = self.restore_params['cluster_index']
    #     inv_A = self.restore_params['inv_A']
        
    #     for index, idlist in enumerate(cluster_index):
    #         Y = []
    #         A = inv_A[index]
    #         for i in idlist:
    #             Y.append(y[:, i, :, :])
    #     for index, idlist in enumerate(cluster_index):
    #         Y = []
    #         for i in idlist:
    #             Y.append(y[:, i, :, :])
    #         Y_tensor = torch.stack(Y)  
    #         Y = Y_tensor.view(len(idlist), -1)
    #         revertY = torch.matmul(inv_A[index], Y)  
    #         reverted_y_reshaped = revertY.view_as(Y_tensor)  
    #         for index, i in enumerate(idlist):
    #             y[:, i, :, :] = reverted_y_reshaped[index]
    #     if self.bias is not None:
    #         bias = self.bias.view(1, self.out_channels, 1, 1).to('cpu')
    #         y = y + bias
    #     return y.to('cuda')
    # def forward(self, x):
    #     x = x[:, self.restore_params['shuffle_indices'], :, :]
    #     y = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
    #     y = y.to('cpu')
    #     cluster_index = self.restore_params['cluster_index']
    #     inv_A = self.restore_params['inv_A']
        
    #     for index, idlist in enumerate(cluster_index):
    #         Y = []
    #         for i in idlist:
    #             Y.append(y[:, i, :, :])
    #         Y_tensor = torch.stack(Y)  
    #         Y = Y_tensor.view(len(idlist), -1)
    #         revertY = torch.matmul(inv_A[index], Y)  
    #         reverted_y_reshaped = revertY.view_as(Y_tensor)  
    #         for index, i in enumerate(idlist):
    #             y[:, i, :, :] = reverted_y_reshaped[index]
    #     if self.bias is not None:
    #         bias = self.bias.view(1, self.out_channels, 1, 1).to('cpu')
    #         y = y + bias
    #     return y.to('cuda')
        


if __name__=="__main__":
    model = alexnet(num_classes=10)


