import os
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch 
from torch import nn
import numpy as np

LAYER_OBJECT = {}

class Linear0inWorker1(nn.Module):
    def __init__(self, d_model=256, d_k=64, d_v=64, n_heads=8):
        super(Linear0inWorker1, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(d_v * n_heads, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V):
        batch = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2) # [batch, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch, -1, self.n_heads, self.d_v).transpose(1, 2) # [batch, n_heads, len_v, d_v]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        return scores, V
    
class Linear1inWorker1(nn.Module):
    def __init__(self, d_model=256, d_v=64, n_heads=8):
        super(Linear1inWorker1, self).__init__()
        self.n_heads = n_heads
        self.d_v = d_v
        self.fc = nn.Linear(d_v * n_heads, d_model, bias=False)
    def forward(self, attn,V,batch):
        prob = torch.matmul(attn, V) 
        prob = prob.transpose(1, 2).contiguous() 
        prob = prob.view(batch, -1, self.n_heads * self.d_v).contiguous() 
        output = self.fc(prob) 
        return output

    
class Linear2inWorker1(nn.Module):
    def __init__(self, d_model=256, d_ff=512):
        super(Linear2inWorker1, self).__init__()
        self.ff1 = nn.Conv1d(d_model, d_ff, 1)
    def forward(self, x):
        return self.ff1(x)
    
class Linear3inWorker1(nn.Module):
    def __init__(self, d_ff=512, d_model=256):
        super(Linear3inWorker1, self).__init__()
        self.ff2 = nn.Conv1d(d_ff, d_model, 1)
    def forward(self, x):
        return self.ff2(x)
  




def worker1_process_linear_layer(x, name):
    
    layer = LAYER_OBJECT[name]
    if isinstance(layer, Linear0inWorker1):
        input_Q, input_K, input_V = x
        result = layer(input_Q, input_K, input_V)
    elif isinstance(layer, Linear1inWorker1):
        attn, V, batch = x 
        result = layer(attn, V, batch)
    elif isinstance(layer, Linear2inWorker1) or isinstance(layer, Linear3inWorker1):
        result = layer(x)
    
    return result


def initialize_layer_objects(d_model=256,n_layers =6,n_heads=8,d_ff=512,d_k=64,d_v=64):
    global LAYER_OBJECT
    LAYER_OBJECT = {
        'll0': Linear0inWorker1( d_model, d_k, d_v, n_heads),
        'll1': Linear1inWorker1(d_model, d_v, n_heads),
        'll2': Linear2inWorker1( d_model, d_ff),
        'll3': Linear3inWorker1(d_ff, d_model)
    }

def main():

    d_model = 768
    n_layers = 12
    n_heads =12
    d_ff = 3072
    d_k = 64
    d_v = 64
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    os.environ['MASTER_ADDR']='192.168.122.127'
    os.environ['MASTER_PORT']='22222'
    os.environ["GLOO_SOCKET_IFNAME"] = "virbr0"  # Change to corresponding IFNAME
    os.environ["TP_SOCKET_IFNAME"] = "virbr0"
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    os.environ["OMP_NUM_THREADS"] ='64'
    torch.set_num_threads(64)
    initialize_layer_objects(d_model,n_layers,n_heads,d_ff,d_k,d_v)

    rpc.shutdown()

if __name__ == "__main__":
    main()
