#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:05:54 2020

@author: sarroutim2
"""
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import csv

import h5py
import progressbar
import time
from utils import Vocabulary
from utils import load_vocab
from models import Classifier
from utils import get_loader
from utils import process_lengths
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import json
import torch
import torch.backends.cudnn as cudnn
from utils import get_glove_embedding

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    preds = torch.max(preds, 1)[1]
    correct=0
    total=0
    correct += (preds == y).float().sum()
    total += y.shape[0]
    acc = correct / total
    return acc

def create_model(args, vocab, embedding=None):
    """Creates the model.

    Args:
        args: Instance of Argument Parser.
        vocab: Instance of Vocabulary.

    Returns:
        A multi class classification model.
    """
    # Load GloVe embedding.
    if args.use_glove:
        embedding =  get_glove_embedding(args.embedding_name,
                                        300,
                                        vocab)
    else:
        embedding = None

    # Build the models
    logging.info('Creating multi-class classification model...')
    model = Classifier(len(vocab), 
                       embedding_dim=args.embedding_dim,
                       embedding=embedding,
                       hidden_dim=args.num_hidden_nodes,
                       output_dim=args.num_output_nodes, 
                       num_layers=args.num_layers, 
                       bidirectional=args.bidirectional, 
                       dropout=args.dropout,
                       rnn_cell=args.rnn_cell)

    return model


def evaluate(model, data_loader, criterion, args):
    
    model.eval()
    epoch_loss=0
    epoch_acc=0
    for i, (sentences, labels, qindices) in enumerate(data_loader):
            #n_steps += 1
            # Set mini-batch dataset.
        if torch.cuda.is_available():
            sentences = sentences.cuda()
            labels = labels.cuda()
            qindices = qindices.cuda()
        lengths = process_lengths(sentences)
        lengths.sort(reverse = True)    
            #convert to 1D tensor
        predictions = model(sentences, lengths)
        
        loss = criterion(predictions, labels)        
            #compute the binary accuracy
        acc = binary_accuracy(predictions, labels)   
            #backpropage the loss and compute the gradients 
            #loss and accuracy
        epoch_loss += loss.item()  
        epoch_acc+= acc.item()    
    logging.info('\t Val-Loss: %.4f | Val-Acc: %.2f '
                             
                             % (loss.item()  , acc.item()*100))

def train(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Save the arguments.
    with open(os.path.join(args.model_path, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)

    # Config logging.
    log_format = '%(levelname)-8s %(message)s'
    logfile = os.path.join(args.model_path, 'train.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(args.__dict__))
    
    vocab = load_vocab(args.vocab_path)

    # Build data loader
    logging.info("Building data loader...")
    train_sampler = None
    val_sampler = None
    
    if os.path.exists(args.train_dataset_weights):
        train_weights = json.load(open(args.train_dataset_weights))
        train_weights = torch.DoubleTensor(train_weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                train_weights, len(train_weights))
    if os.path.exists(args.val_dataset_weights):
        val_weights = json.load(open(args.val_dataset_weights))
        val_weights = torch.DoubleTensor(val_weights)
        val_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                val_weights, len(val_weights))
    data_loader = get_loader(args.dataset,
                                 args.batch_size, shuffle=True,
                                 num_workers=args.num_workers,
                                 max_examples=args.max_examples,
                                 sampler=train_sampler)
    val_data_loader = get_loader(args.val_dataset,
                                     args.batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     max_examples=args.max_examples,
                                     sampler=val_sampler)
    logging.info("Done")
    
    model = create_model(args, vocab)

    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model))
    logging.info("Done")
    
    criterion = nn.CrossEntropyLoss()

    #criterion = nn.BCELoss()
        # Setup GPUs.
    if torch.cuda.is_available():
        logging.info("Using available GPU...")
        model.cuda()
        criterion.cuda()
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True
    # Parameters to train.
    params = model.parameters()
   
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                  factor=0.1, patience=args.patience,
                                  verbose=True, min_lr=1e-7)

    # Train the model.
    total_steps = len(data_loader)
    start_time = time.time()
    n_steps = 0
    
    #initialize every epoch 
    epoch_loss = 0
    epoch_acc = 0
    
    #set the model in training phase
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        epoch_acc = 0
        model.train() 
        for i, (sentences, labels, qindices) in enumerate(data_loader):
            n_steps += 1

            # Set mini-batch dataset.
            if torch.cuda.is_available():
                sentences = sentences.cuda()
                labels = labels.cuda()
                qindices = qindices.cuda()
            lengths = process_lengths(sentences)
            lengths.sort(reverse = True)       
            #resets the gradients after every batch
            optimizer.zero_grad()   
            #convert to 1D tensor
            predictions = model(sentences, lengths)

            loss = criterion(predictions, labels)        
            #compute the binary accuracy
            acc = binary_accuracy(predictions, labels)   
            #backpropage the loss and compute the gradients
            loss.backward()       
            
            #update the weights
            optimizer.step()       
            #loss and accuracy
            epoch_loss += loss.item()  
            epoch_acc+= acc.item()    
        delta_time = time.time() - start_time
        start_time = time.time()
        logging.info('Epoch [%d/%d] | Step [%d/%d] | Time: %.4f  \n'
                             '\t Train-Loss: %.4f | Train-Acc: %.2f'
                             
                             % (epoch, args.num_epochs, i,
                                total_steps, delta_time, 
                                loss.item()  , acc.item()*100))
        evaluate(model, val_data_loader, criterion, args)
        torch.save(model.state_dict(),
                   os.path.join(args.model_path,
                  'model-tf-%d.pkl' % (epoch+1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters.
    parser.add_argument('--model-path', type=str, default='weights/tf1/',
                        help='Path for saving trained models')
    parser.add_argument('--save-step', type=int, default=None,
                        help='Step size for saving trained models')
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max-examples', type=int, default=None,
                        help='For debugging. Limit examples in database.')

    # Data parameters.
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--dataset', type=str,
                        default='data/processed/dataset_train.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--val-dataset', type=str,
                        default='data/processed/dataset_val.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--train-dataset-weights', type=str,
                        default='data/processed/train_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--val-dataset-weights', type=str,
                        default='data/processed/test_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Location of where the model weights are.')

    # Model parameters
    parser.add_argument('--rnn-cell', type=str, default='LSTM',
                        help='Type of rnn cell (GRU, RNN or LSTM).')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of layers in lstm.')
    parser.add_argument('--num-output-nodes', type=int, default=4,
                        help='Number of labels.')
    parser.add_argument('--num-hidden-nodes', type=int, default=100,
                        help='Number of hidden nodes.')

    parser.add_argument('--bidirectional', action='store_true', default=False,
                        help='Boolean whether the RNN is bidirectional.')
    parser.add_argument('--use-glove', action='store_true', default=True,
                        help='Whether to use GloVe embeddings.')
    parser.add_argument('--use-w2v', action='store_true', default=False,
                        help='Whether to use W2V embeddings.')
    parser.add_argument('--embedding-name', type=str, default='6B',
                        help='Name of the GloVe embedding to use. data/processed/PubMed-w2v.txt')
    parser.add_argument('--embedding-dim', type=int, default=100,
                        help='Embedding size.')
    
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout applied to the RNN model.')

    parser.add_argument('--num-att-layers', type=int, default=2,
                        help='Number of attention layers.')



    args = parser.parse_args()
    train(args)
    # Hack to disable errors for importing Voca


