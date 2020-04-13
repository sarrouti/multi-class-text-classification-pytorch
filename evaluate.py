#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 01:04:11 2020

@author: sarroutim2
"""
import argparse
import json
import logging
import os
import torch
import progressbar
from models import Classifier
from utils import Dict2Obj
from utils import Vocabulary
from utils import get_loader
from utils import load_vocab
from utils import process_lengths
from utils import get_glove_embedding
from sklearn.metrics import accuracy_score
def binary_accuracy(preds, y):
    #round predictions to the closest integer
    acc = accuracy_score(preds,y)
    return acc

def evaluate(model, data_loader, vocab, args, params):
    model.eval()
    preds = []
    gts = []
    bar = progressbar.ProgressBar(maxval=len(data_loader))
    
    for i, (sentences, labels, _) in enumerate(data_loader):
            #n_steps += 1
            # Set mini-batch dataset.
        if torch.cuda.is_available():
            sentences = sentences.cuda()
            labels = labels.cuda()
        lengths = process_lengths(sentences)
        lengths.sort(reverse = True)
        
        predictions = model(sentences, lengths)
        predictions = torch.max(predictions, 1)[1]
        for p in predictions:
            preds.append(p.item())
        
        for l in labels:
            gts.append(l.item())
    
    print ('='*80)
    print ('GROUND TRUTH')
    print (gts[:args.num_show])
    print ('-'*80)
    print ('PREDICTIONS')
    print (preds[:args.num_show])
    print ('='*80)
    acc = binary_accuracy(preds, gts)
    
    print('Acc : ', acc)
    return acc, gts, preds

def main(args):
    # Load the arguments.
    model_dir = os.path.dirname(args.model_path)
    params = Dict2Obj(json.load(
            open(os.path.join(model_dir, "args.json"), "r")))

    # Config logging
    log_format = '%(levelname)-8s %(message)s'
    logfile = os.path.join(model_dir, 'eval.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(args.__dict__))
    # Load vocabulary wrapper.
    vocab = load_vocab(params.vocab_path)

    # Build data loader
    logging.info("Building data loader...")

    # Load GloVe embedding.
    if params.use_glove:
        embedding = get_glove_embedding(params.embedding_name,
                                        300, vocab)
    else:
        embedding = None

    # Build data loader
    logging.info("Building data loader...")
    data_loader = get_loader(args.dataset,
                                 args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 max_examples=args.max_examples)
    logging.info("Done")    
    # Build the models
    logging.info('Creating a multi class classification model...')
    model = Classifier(len(vocab), 
                       embedding_dim=params.embedding_dim, 
                       embedding=embedding,
                       hidden_dim=params.num_hidden_nodes,
                       output_dim=params.num_output_nodes, 
                       num_layers=params.num_layers, 
                       bidirectional=params.bidirectional, 
                       dropout=params.dropout,
                       rnn_cell=params.rnn_cell)


    logging.info("Done")

    logging.info("Loading model.")
    model.load_state_dict(torch.load(args.model_path+"model-tf-"+args.state+".pkl"))

    # Setup GPUs.
    if torch.cuda.is_available():
        logging.info("Using available GPU...")
        model.cuda()
    scores, gts, preds = evaluate(model, data_loader, vocab, args, params)

    # Print and save the scores.
    print (scores)
    with open(os.path.join(model_dir, args.results_path), 'w') as results_file:
        json.dump(scores, results_file)
    with open(os.path.join(model_dir, args.preds_path), 'w') as preds_file:
        json.dump(preds, preds_file)
    with open(os.path.join(model_dir, args.gts_path), 'w') as gts_file:
        json.dump(gts, gts_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters.
    parser.add_argument('--model-path', type=str, default='weights/tf1/'
                        ,help='Path for loading trained models')
    parser.add_argument('--results-path', type=str, default='results.json',
                        help='Path for saving results.')
    parser.add_argument('--preds-path', type=str, default='preds.json',
                        help='Path for saving predictions.')
    parser.add_argument('--gts-path', type=str, default='gts.json',
                        help='Path for saving ground truth.')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=8)
    #parser.add_argument('--pin_memory', default=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max-examples', type=int, default=None,
                        help='When set, only evalutes that many data points.')
    parser.add_argument('--num-show', type=int, default=50,
                        help='Number of predictions to print.')

    parser.add_argument('--state', type=str, default='1',
                        help='Path for saving results.')
    # Data parameters.
    parser.add_argument('--dataset', type=str,
                        default='data/processed/dataset_test.hdf5',
                        help='path for test annotation file')

    args = parser.parse_args()
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
    # Hack to disable errors for importing Vocabulary. Ignore this line.
    Vocabulary()
