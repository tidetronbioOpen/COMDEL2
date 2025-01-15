import torch
import torch.nn as nn
import numpy as np
from configuration import config
import pickle
from util import util_freeze



class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table)
        self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]
        pos = torch.arange(seq_len, device=device, dtype=torch.long)  # [seq_len]
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding



class CNN(nn.Module):
    """"""

    def __init__(self, config):

        super(CNN, self).__init__()
        global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, vocab_size, device
        max_len = config.max_len
        n_layers = config.num_layer  # number of encoder blocks
        n_head = config.num_head  # number of head in multi-head attention
        d_model = config.dim_embedding  # residue embedding dimension 残基嵌入维度
        d_ff = config.dim_feedforward  # hidden layer dimension in feedforward layer 前馈层中的隐藏层维度
        d_k = config.dim_k  # 'embedding dimension of vector k or q'
        d_v = config.dim_v  # 'embedding dimension of vector v'
        vocab_size = config.vocab_size  #
        device = torch.device("cuda" if config.cuda else "cpu")

        self.embedding = Embedding()
        self.cnn = 1

    def forward(self, input):
        """"""