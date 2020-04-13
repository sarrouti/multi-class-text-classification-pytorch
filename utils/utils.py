#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:56:39 2020

@author: sarroutim2
"""

import torch
import torchtext
import json
class Vocabulary (object):
    
    SYM_PAD = '<pad>'    # padding.
    SYM_UNK = '<unk>'    # Unknown word.
    
    def __init__(self):
        self.word2idx={}
        self.idx2word={}
        self.idx=0
        self.add_word(self.SYM_PAD)
        self.add_word(self.SYM_UNK)
    
    def add_word (self, word):
        
        if word not in self.word2idx:
            self.word2idx [word] = self.idx
            self.idx2word [self.idx] = word
            self.idx += 1
    
        
    def remove_word(self, word):
	
        """Removes a specified word and updates the total number of unique words.
	Args:
	    word: String representation of the word.
        """
        if word in self.word2idx:
    	    self.word2idx.pop(word)
    	    self.idx2word.pop(self.idx)
    	    self.idx -= 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.SYM_UNK]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def save(self, location):
        with open(location, 'w') as f:
            json.dump({'word2idx': self.word2idx,
                       'idx2word': self.idx2word,
                       'idx': self.idx}, f)

    def load(self, location):
        with open(location, 'rb') as f:
            data = json.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.idx = data['idx']        
            
def get_glove_embedding(name, embed_size, vocab):
    """Construct embedding tensor.

    Args:
        name (str): Which GloVe embedding to use.
        embed_size (int): Dimensionality of embeddings.
        vocab: Vocabulary to generate embeddings.
    Returns:
        embedding (vocab_size, embed_size): Tensor of
            GloVe word embeddings.
    """

    glove = torchtext.vocab.GloVe(name=name,
                                  dim=str(embed_size))
    vocab_size = len(vocab)
    embedding = torch.zeros(vocab_size, embed_size)
    for i in range(vocab_size):
        embedding[i] = glove[vocab.idx2word[str(i)]]
    return embedding


# ===========================================================
# Helpers.
# ===========================================================

def process_lengths(inputs, pad=0):
    """Calculates the lenght of all the sequences in inputs.

    Args:
        inputs: A batch of tensors containing the question or response
            sequences.

    Returns: A list of their lengths.
    """
    max_length = inputs.size(1)
    if inputs.size(0) == 1:
        lengths = list(max_length - inputs.data.eq(pad).sum(1))
    else:
        lengths = list(max_length - inputs.data.eq(pad).sum(1).squeeze())
    return lengths
