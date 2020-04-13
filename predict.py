#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:34:20 2020

@author: sarroutim2
"""

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
from utils import process_text
from utils import process_label
from utils import get_glove_embedding
from sklearn.metrics import accuracy_score
from torch.autograd import Variable

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    acc = accuracy_score(preds,y)
    return acc

def predict(model, d_text):
    model.eval()
    labels = {1:'yesno',
              2:'factoid',
              3:'list',
              4:'summary'
             }
    tensor = torch.tensor(d_text).cuda()
    tensor = Variable(tensor,volatile=True).cuda()

    print(tensor)
    length = [len(tensor)]
    tensor = tensor.unsqueeze(1)                          #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)
    predictions = model(tensor, length_tensor)
    predictions = torch.max(predictions, 1)[1]
    print(predictions)
    print(labels[predictions.item()])

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

    # Processing input text
    logging.info("Processing input text...")
    text, length = process_text(args.text, vocab,
                                 max_length=20)
    d_text = text

    logging.info("Done")    
    # Build the models
    logging.info('Creating IQ model...')
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
    
    predict(model,d_text)


 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters.
    parser.add_argument('--model-path', type=str, default='weights/tf2/'
                        ,help='Path for loading trained models')

    parser.add_argument('--text', type=str, default='what is the definition of covid-19'
                        ,help='input text') 
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--state', type=str, default='1',
                        help='Path for saving results.')
 

    args = parser.parse_args()
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
    # Hack to disable errors for importing Vocabulary. Ignore this line.
    Vocabulary()
