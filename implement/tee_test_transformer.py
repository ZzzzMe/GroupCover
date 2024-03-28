import os
import torch
import torch.distributed.rpc as rpc
import time  
from torch import nn
from torch.utils import data as Data
import numpy as np

WORKER_RNAK= -1

def make_data(sentences,source_vocab,target_vocab):
    encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
    for i in range(len(sentences)):
        encoder_input = [source_vocab[word] for word in sentences[i][0].split()]
        decoder_input = [target_vocab[word] for word in sentences[i][1].split()]
        decoder_output = [target_vocab[word] for word in sentences[i][2].split()]
        encoder_inputs.append(encoder_input)
        decoder_inputs.append(decoder_input)
        decoder_outputs.append(decoder_output)
    return torch.LongTensor(encoder_inputs), torch.LongTensor(decoder_inputs), torch.LongTensor(decoder_outputs)

def get_attn_pad_mask(seq_q, seq_k):

    batch, len_q = seq_q.size()
    batch, len_k = seq_k.size()
    # we define index of PAD is 0, if tensor equals (zero) PAD tokens
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1).to(seq_q.device)  # [batch, 1, len_k]
    return pad_attn_mask.expand(batch, len_q, len_k)  # [batch, len_q, len_k]

def get_attn_subsequent_mask(seq):

    attn_shape = [seq.size(0), seq.size(1), seq.size(1)] # [batch, target_len, target_len]
    subsequent_mask = torch.triu(torch.ones(attn_shape, device=seq.device), diagonal=1)
    return subsequent_mask # [batch, target_len, target_len] 

class PositionalEncoding(nn.Module):

    def __init__(self, d_model=256, dropout=.1,p_drop = 0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=p_drop)

        positional_encoding = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len).float().unsqueeze(1) # [max_len, 1]

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-torch.log(torch.Tensor([10000])) / d_model)) # [max_len / 2]

        positional_encoding[:, 0::2] = torch.sin(position * div_term) # even
        positional_encoding[:, 1::2] = torch.cos(position * div_term) # odd

        # [max_len, d_model] -> [1, max_len, d_model] -> [max_len, 1, d_model]
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)

        # register pe to buffer and require no grads
        self.register_buffer('pe', positional_encoding)

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        # we can add positional encoding to x directly, and ignore other dimension
        x = x + self.pe[:x.size(0), ...]

        return self.dropout(x)

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
        
class NoneLinear0inWorker0(nn.Module):
    def __init__(self,n_heads=8):
        super(NoneLinear0inWorker0, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.n_heads = n_heads
    def forward(self, scores, attn_mask):
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # [batch, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)
        attn = self.softmax(scores)
        return attn

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads=8,d_model=256,d_k=64,d_v=64):
        super(MultiHeadAttention, self).__init__()
        # do not use more instance to implement multihead attention
        # it can be complete in one matrix
        self.n_heads = n_heads
        # self.Linear0inWorker1 = Linear0inWorker1(d_model, d_k, d_v, n_heads)
        self.NoneLinear0inWorker0 = NoneLinear0inWorker0(n_heads)
        # self.Linear1inWorker1 = Linear1inWorker1(d_model, d_v, n_heads)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        To make sure multihead attention can be used both in encoder and decoder, 
        we use Q, K, V respectively.
        input_Q: [batch, len_q, d_model]
        input_K: [batch, len_k, d_model]
        input_V: [batch, len_v, d_model]
        '''
        residual, batch = input_Q, input_Q.size(0)
        scores, V  = rpc.rpc_sync(f"worker{WORKER_RNAK}", worker1_process_linear_layer, args=((input_Q, input_K, input_V), 'll0'))
        # scores, V = self.Linear0inWorker1(input_Q, input_K, input_V)
        attn = self.NoneLinear0inWorker0(scores,attn_mask)
        tmp = rpc.rpc_sync(f"worker{WORKER_RNAK}", worker1_process_linear_layer, args=((attn,V,batch), 'll1'))
        # tmp = self.Linear1inWorker1(attn,V,batch)
        output = self.layer_norm(residual + tmp)
        return output, attn
    
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
  
class FeedForwardNetwork(nn.Module):
    '''
    Using nn.Conv1d replace nn.Linear to implements FFN.
    '''
    def __init__(self,d_model=256,d_ff=512,p_drop=0.1):
        super(FeedForwardNetwork, self).__init__()
        # self.ff1 = nn.Linear(d_model, d_ff)
        # self.ff2 = nn.Linear(d_ff, d_model)
        # self.Linear2inWorker1 = Linear2inWorker1(d_model, d_ff)
        # self.Linear3inWorker1 = Linear3inWorker1(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_drop)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        residual = x
        x = x.transpose(1, 2) # [batch, d_model, seq_len]
        x = rpc.rpc_sync(f"worker{WORKER_RNAK}", worker1_process_linear_layer, args=(x, 'll2'))
        # x = self.Linear2inWorker1(x)
        x = self.relu(x)
        x = rpc.rpc_sync(f"worker{WORKER_RNAK}", worker1_process_linear_layer, args=(x, 'll3'))
        # x = self.Linear3inWorker1(x)
        x = x.transpose(1, 2) # [batch, seq_len, d_model]
        return self.layer_norm(residual + x)

  
class EncoderLayer(nn.Module):
    def __init__(self,n_heads=8,d_model=256,d_ff=512,d_k=64,d_v=64):
        super(EncoderLayer, self).__init__()
        self.encoder_self_attn = MultiHeadAttention(n_heads,d_model,d_k,d_v)
        self.ffn = FeedForwardNetwork(d_model,d_ff)

    def forward(self, encoder_input, encoder_pad_mask):

        encoder_output, attn = self.encoder_self_attn(encoder_input, encoder_input, encoder_input, encoder_pad_mask)
        encoder_output = self.ffn(encoder_output) # [batch, source_len, d_model]
  
        return encoder_output, attn
class Encoder(nn.Module):

    def __init__(self,source_vocab_size,d_model=256,n_layers =6,n_heads=8,d_ff=512,d_k=64,d_v=64):
        super(Encoder, self).__init__()
        self.source_embedding = nn.Embedding(source_vocab_size, d_model)
        self.positional_embedding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(n_heads,d_model,d_ff,d_k,d_v) for _ in range(n_layers)])

    def forward(self, encoder_input):
        encoder_output = self.source_embedding(encoder_input)
        encoder_output = self.positional_embedding(encoder_output.transpose(0, 1)).transpose(0, 1)
        encoder_self_attn_mask = get_attn_pad_mask(encoder_input, encoder_input)
        encoder_self_attns = []

        for layer in self.layers:
            encoder_output, encoder_self_attn = layer(encoder_output, encoder_self_attn_mask)
            encoder_self_attns.append(encoder_self_attn)

        return encoder_output, encoder_self_attns

class DecoderLayer(nn.Module):

    def __init__(self,n_heads = 8,d_model=256,d_ff=512,d_k=64,d_v=64):
        super(DecoderLayer, self).__init__()
        self.decoder_self_attn = MultiHeadAttention(n_heads,d_model,d_k,d_v)
        self.encoder_decoder_attn = MultiHeadAttention(n_heads,d_model,d_k,d_v)
        self.ffn = FeedForwardNetwork(d_model,d_ff)

    def forward(self, decoder_input, encoder_output, decoder_self_mask, decoder_encoder_mask):
        decoder_output, decoder_self_attn = self.decoder_self_attn(decoder_input, decoder_input, decoder_input, decoder_self_mask)
        decoder_output, decoder_encoder_attn = self.encoder_decoder_attn(decoder_output, encoder_output, encoder_output, decoder_encoder_mask)
        decoder_output = self.ffn(decoder_output)

        return decoder_output, decoder_self_attn, decoder_encoder_attn

class Decoder(nn.Module):

    def __init__(self,target_vocab_size,d_model=256,n_layers =6,n_heads=8,d_ff=512,d_k=64,d_v=64):
        super(Decoder, self).__init__()
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_embedding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(n_heads,d_model,d_ff,d_k,d_v) for _ in range(n_layers)])

    def forward(self, decoder_input, encoder_input, encoder_output):
        decoder_output = self.target_embedding(decoder_input)
        decoder_output = self.positional_embedding(decoder_output.transpose(0, 1)).transpose(0, 1)
        decoder_self_attn_mask = get_attn_pad_mask(decoder_input, decoder_input)
        decoder_subsequent_mask = get_attn_subsequent_mask(decoder_input)
        decoder_encoder_attn_mask = get_attn_pad_mask(decoder_input, encoder_input)

        decoder_self_mask = torch.gt(decoder_self_attn_mask + decoder_subsequent_mask, 0)
        decoder_self_attns, decoder_encoder_attns = [], []

        for layer in self.layers:
            decoder_output, decoder_self_attn, decoder_encoder_attn = layer(decoder_output, encoder_output, decoder_self_mask, decoder_encoder_attn_mask)
            decoder_self_attns.append(decoder_self_attn)
            decoder_encoder_attns.append(decoder_encoder_attn)

        return decoder_output, decoder_self_attns, decoder_encoder_attns

class Transformer(nn.Module):

    def __init__(self,source_vocab_size,target_vocab_size,d_model=256,n_layers =6,n_heads=8,d_ff=512,d_k=64,d_v=64):
        super(Transformer, self).__init__()

        self.encoder = Encoder(source_vocab_size,d_model,n_layers,n_heads,d_ff,d_k,d_v)
        self.decoder = Decoder(target_vocab_size,d_model,n_layers,n_heads,d_ff,d_k,d_v)
        self.projection = nn.Linear(d_model, target_vocab_size, bias=False)

    def forward(self, encoder_input, decoder_input):
        encoder_output, encoder_attns = self.encoder(encoder_input)

        decoder_output, decoder_self_attns, decoder_encoder_attns = self.decoder(decoder_input, encoder_input, encoder_output)
        decoder_logits = self.projection(decoder_output)

        return decoder_logits.view(-1, decoder_logits.size(-1)), encoder_attns, decoder_self_attns, decoder_encoder_attns
    
class Seq2SeqDataset(Data.Dataset):

    def __init__(self, encoder_input, decoder_input, decoder_output):
        super(Seq2SeqDataset, self).__init__()
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.decoder_output = decoder_output    
    def __len__(self):
        return self.encoder_input.shape[0]  
    def __getitem__(self, idx):
        return self.encoder_input[idx], self.decoder_input[idx], self.decoder_output[idx]
  


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



def initialize_layer_objects(d_model=256,n_layers =6,n_heads=8,d_ff=512,d_k=64,d_v=64,work_rank=0):
    global LAYER_OBJECT
    global WORKER_RNAK

    LAYER_OBJECT = {
        'll0': Linear0inWorker1( d_model, d_k, d_v, n_heads),
        'll1': Linear1inWorker1(d_model, d_v, n_heads),
        'll2': Linear2inWorker1( d_model, d_ff),
        'll3': Linear3inWorker1(d_ff, d_model)
    }
    WORKER_RNAK = work_rank

def main():

    sentences = [
            # enc_input           dec_input         dec_output
            ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
            ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
    ]
    source_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
    source_vocab_size = len(source_vocab)

    target_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
    idx2word = {i: w for i, w in enumerate(target_vocab)}
    target_vocab_size = len(target_vocab)

    batch_size = 512
    epochs = 64
    lr = 1e-3

    d_model = 768
    n_layers = 12
    n_heads =12
    d_ff = 3072
    d_k = 64
    d_v = 64
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    encoder_inputs, decoder_inputs, decoder_outputs = make_data(sentences,source_vocab,target_vocab)
    dataset = Seq2SeqDataset(encoder_inputs, decoder_inputs, decoder_outputs)
    data_loader = Data.DataLoader(dataset, batch_size, True)
    
    os.environ['MASTER_ADDR']='192.168.122.127'
    os.environ['MASTER_PORT']='22222'
    os.environ["GLOO_SOCKET_IFNAME"] = "enp0s3"  # MUST MODIFY HERE!
    os.environ["TP_SOCKET_IFNAME"] = "enp0s3"
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    worker_rank = int(os.environ['WORKER'])
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    initialize_layer_objects(d_model,n_layers,n_heads,d_ff,d_k,d_v,rank)

    model = Transformer(source_vocab_size,target_vocab_size,d_model,n_layers ,n_heads,d_ff,d_k,d_v)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print(model)
    with torch.no_grad():
        
        for epoch in range(10):
            st = time.time()
            for encoder_input, decoder_input, decoder_output in data_loader:
                for i in range(10):
                    output, encoder_attns, decoder_attns, decoder_encoder_attns = model(encoder_input, decoder_input)
            print('throughput '+'-'*10,10*(batch_size)/(time.time()-st))
    rpc.shutdown()

if __name__ == "__main__":
    main()
