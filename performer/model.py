import torch
import math
import torch.nn as nn

import os
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.optim.lr_scheduler import OneCycleLR


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelfAttention(nn.Module): 

    def __init__(self, embedding_dim, num_heads, masked=False, dropout_rate=0):
        super(SelfAttention, self).__init__()

        assert embedding_dim % num_heads == 0
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim//num_heads
        self.masked = masked

        self.Wq = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)
        self.Wk = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)
        self.Wv = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)
        self.Wo = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        device = x.device

        batch_size, seq_len, embedding_dim = x.shape
        
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)  
        V = V.permute(0, 2, 1, 3)  

        dot_product = torch.matmul(Q, torch.transpose(K, -2, -1))

        dot_product = dot_product/(math.sqrt(self.head_dim))

        if self.masked:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            mask = mask.to(device)
            mask = mask.unsqueeze(0).unsqueeze(0) #broadcast head and batch dim
            dot_product = dot_product.masked_fill(mask, float('-inf'))
        
        softmax = torch.nn.Softmax(dim=3)
        
        dot_product = softmax(dot_product)
        dot_product = self.dropout(dot_product) 

        attention_scores = torch.matmul(dot_product, V)
        attention_scores = attention_scores.permute(0, 2, 1, 3)
        attention_scores = attention_scores.reshape(batch_size, seq_len, embedding_dim)

        output = self.Wo(attention_scores)

        return output

class FastAttention(nn.Module): 

    def __init__(self, embedding_dim, num_heads, num_random_features, masked = None):
        #this is imr
        super(FastAttention, self).__init__()

        assert embedding_dim % num_heads == 0

        self.masked= masked
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim//num_heads
        self.num_random_features = num_random_features 

        assert self.num_random_features <= self.head_dim

        self.Wq = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)
        self.Wk = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)
        self.Wv = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)
        self.Wo = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)

    def forward(self, x, mask=None):

        device = x.device

        batch_size, seq_len, embedding_dim = x.shape
        
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)  
        V = V.permute(0, 2, 1, 3)  

        if self.masked:
            if mask is None:
                mask = torch.triu(torch.ones(batch_size, seq_len), diagonal=1).bool().to(device)
                mask = ~mask
            mask = mask[:, None, :, None].to(device)
            V = V.masked_fill(~mask, 0.)

        Q = Q / np.sqrt(self.head_dim)  # Temperature scaling
        K = K / np.sqrt(self.head_dim)
        Q_phi = self.positive_random_feature_map(Q, device)
        K_phi = self.positive_random_feature_map(K, device)

        if self.masked:
            causal_mask = torch.triu(torch.ones(seq_len, 64), diagonal=1).bool().to(device)
            K_phi_masked = K_phi.masked_fill(causal_mask[None, None, :, :], 0.)
            KV = torch.matmul(K_phi_masked.transpose(-2,-1), V)
        else: 
            KV = torch.matmul(K_phi.transpose(-2,-1), V)
            
        QKV = torch.matmul(Q_phi, KV) 
        
        K_ones = torch.matmul(K_phi.transpose(-2,-1), torch.ones(100, device=device)) 
        K_ones = K_ones.unsqueeze(-1)  
        QK_ones = torch.matmul(Q_phi, K_ones)  
        QK_ones = QK_ones.squeeze(-1) 
        
        D_inv = 1.0 / QK_ones  
        D_inv = D_inv.unsqueeze(-1) 
        
        attention_scores = D_inv * QKV  
        attention_scores = attention_scores.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        output = self.Wo(attention_scores)

        return output

    def positive_random_feature_map(self, attention_vector, device):
        scaling_factor = 1.0 / math.sqrt(self.num_random_features)

        omega = torch.randn(attention_vector.shape[-1], self.num_random_features, device=device)
        omega = omega / np.sqrt(attention_vector.shape[-1])    
        # omega = omega / np.sqrt(self.num_random_features)
        omega, _ = torch.qr(omega) 
        
        random_feature_map = torch.matmul(attention_vector, omega)
        random_feature_map = random_feature_map.clamp(max=50) 
        
        l2_norm = torch.norm(attention_vector, p=2, dim=-1, keepdim=True)
        eps = 1e-8
        l2_norm = l2_norm.clamp(min=eps)
        normalization_term = torch.exp(-l2_norm**2 / 2).clamp(min=eps)
        
        result = normalization_term * torch.exp(random_feature_map)
        return result


class CrossAttention(nn.Module): 

    def __init__(self, embedding_dim, num_heads, dropout_rate=0):
        super(CrossAttention, self).__init__()

        assert embedding_dim % num_heads == 0
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim//num_heads

        self.Wq = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)
        self.Wk = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)
        self.Wv = nn.Linear(embedding_dim, self.head_dim * num_heads, bias=False)
        self.Wo = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.dropout = nn.Dropout(dropout_rate)  # Add this line

    def forward(self, encoder_output, decoder_input):
        batch_size, decoder_seq_len, embedding_dim = decoder_input.shape
        _, encoder_seq_len, _ = encoder_output.shape
        
        Q = self.Wq(decoder_input)
        K = self.Wk(encoder_output)
        V = self.Wv(encoder_output)

        Q = Q.reshape(batch_size, decoder_seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, encoder_seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, encoder_seq_len, self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)  
        V = V.permute(0, 2, 1, 3)  

        dot_product = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot_product = dot_product/(math.sqrt(self.head_dim))
        
        softmax = torch.nn.Softmax(dim=3)
        dot_product = softmax(dot_product)
        dot_product = self.dropout(dot_product) 

        attention_scores = torch.matmul(dot_product, V)
        attention_scores = attention_scores.permute(0, 2, 1, 3)
        attention_scores = attention_scores.reshape(batch_size, decoder_seq_len, embedding_dim)

        output = self.Wo(attention_scores)

        return output



class Encoder(nn.Module):

    def __init__(self, embedding_dim, num_heads, ff_dim, dropout_rate, attention_type='self_attention', num_random_features=64):
        super(Encoder, self).__init__()
    
        if attention_type == 'self_attention':
            self.attention = SelfAttention(embedding_dim, num_heads, dropout_rate=dropout_rate)
        elif attention_type == 'fast_attention':
            self.attention = FastAttention(embedding_dim, num_heads, num_random_features=num_random_features)  
        else:
            raise ValueError("Unknown attention type: {}".format(attention_type))
            
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    
    def forward(self, x):

        attention_output = self.attention(x)
        attention_output = self.dropout(attention_output)  # Add this line

        x = x + attention_output 
        x = self.norm1(x) 

        x_ = self.feedforward(x) 
        x_= self.dropout(x_)  # Add this line

        x = x_ + x 
        x = self.norm2(x) 
        encoder_output = x

        return encoder_output


class Decoder(nn.Module):

    def __init__(self, embedding_dim, num_heads, ff_dim, dropout_rate, attention_type='self_attention', num_random_features=64):
        super(Decoder, self).__init__()
    
        if attention_type == 'self_attention':
            self.masked_attention = SelfAttention(embedding_dim, num_heads, masked=True, dropout_rate=dropout_rate)
        elif attention_type == 'fast_attention':
            self.masked_attention = FastAttention(embedding_dim, num_heads, num_random_features=num_random_features, masked=True)  
        else:
            raise ValueError("Unknown attention type: {}".format(attention_type))
            
        self.cross_attention = CrossAttention(embedding_dim, num_heads)
        
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, encoder_output, decoder_input):

        masked_attention = self.masked_attention(decoder_input)

        #add & norm 
        decoder_input = masked_attention + decoder_input 
        decoder_input = self.norm1(decoder_input)
        
        cross_attention  = self.cross_attention(encoder_output, decoder_input)
        
        decoder_output = cross_attention + decoder_input
        decoder_output = self.norm2(decoder_output)
        
        ff_output = self.feedforward(decoder_output)

        decoder_output = ff_output + decoder_output
        decoder_output = self.norm3(decoder_output)

        return decoder_output


class Transformer(nn.Module):

    def __init__(self, embedding_dim, num_heads, ff_dim, dropout_rate, num_layers, vocab_size, encoder_attention_type='self_attention', decoder_attention_type='self_attention', num_random_features=64):
        super(Transformer,self).__init__()
        self.embedding_dim = embedding_dim
        
        #this one 
        self.embedding1 = nn.Embedding(vocab_size, embedding_dim)
        self.embedding2 = nn.Embedding(vocab_size, embedding_dim)
        
        self.encoders = nn.ModuleList([Encoder(embedding_dim, num_heads, ff_dim, dropout_rate, attention_type=encoder_attention_type, num_random_features=num_random_features) for i in range(num_layers)])
        self.decoders = nn.ModuleList([Decoder(embedding_dim, num_heads, ff_dim, dropout_rate, attention_type=decoder_attention_type, num_random_features=num_random_features) for i in range(num_layers)])
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, y):

        x_embedding = self.embedding1(x)
        y_embedding = self.embedding2(y)

        x_pe = self.sin_cos_positional_encoding(x.shape[1], self.embedding_dim)
        y_pe = self.sin_cos_positional_encoding(y.shape[1], self.embedding_dim)

        encoder_output = self.dropout(x_embedding + x_pe)  # Add dropout here
        decoder_output = self.dropout(y_embedding + y_pe)  # Add dropout here
        
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output)

        for decoder in self.decoders:
            decoder_output = decoder(encoder_output, decoder_output)

        output = self.linear(decoder_output)
        # softmax = nn.Softmax(dim=2)  
        # output = softmax(output)
        return output

    def sin_cos_positional_encoding(self, seq_len, embedding_dim):
        pos = torch.arange(seq_len).unsqueeze(1).to(device)
        even = torch.arange(0, embedding_dim, 2).unsqueeze(0).to(device)
        
        denominator = 10000**((2*even) / embedding_dim)
        #make 10,000 an argument k to pass
        
        pe = torch.zeros(seq_len,embedding_dim, device=device)
        pe[:, 0::2] = torch.sin(pos/denominator)   
        pe[:, 1::2] = torch.cos(pos/denominator)   

        return pe.unsqueeze(0)