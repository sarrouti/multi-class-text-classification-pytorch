#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:06:48 2020

@author: sarroutim2
"""

import torch.nn as nn
import torch
from .base_rnn import BaseRNN
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class Classifier(nn.Module):
    
    #define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, embedding, hidden_dim, output_dim, num_layers, 
                 bidirectional, dropout,rnn_cell='LSTM'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if self.embedding is not None:
            self.embedding.weight=nn.Parameter(self.embedding.weight, requires_grad=True)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        self.dropout = nn.Dropout(0.2)        
        #activation function
        self.act = nn.Softmax()
        
    def forward(self, x, l):
        x = self.embedding(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, l, batch_first=True)

        lstm_out, (ht, ct) = self.lstm(x_pack)
        dense_outputs=self.linear(ht[-1])

        #Final activation function
        outputs=self.act(dense_outputs)
        return outputs