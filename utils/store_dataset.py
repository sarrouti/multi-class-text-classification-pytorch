#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:53:49 2020

@author: sarroutim2
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 13:27:26 2020

@author: sarroutim2
"""


import argparse
import csv

import h5py
import progressbar
from vocab import process_text
from vocab import process_label
from vocab import load_vocab
import json

def save_dataset(dataset, vocab, output,
                 max_length=20):
    sentences=[]
    labels=[]
    with open (dataset) as questions_file:
        
        csv_reader=csv.reader(questions_file, delimiter='\t')
        
        line_count=0
        column_names=[]
        for row in csv_reader:
            
            if line_count==0:
                column_names.append(row[0])
                column_names.append(row[1])
                line_count+=1
            else:
                sentences.append(row[1])
                labels.append(row[2])
                
    total_sentences=len(sentences)
    
    vocab=load_vocab(vocab)
    
    print('Number of sentences to be written: %d',total_sentences)

    h5file = h5py.File(output, "w")
    d_questions = h5file.create_dataset(
        "sentences", (total_sentences, max_length), dtype='i')
 
    d_labels = h5file.create_dataset(
        "labels", (total_sentences,), dtype='i')

    bar = progressbar.ProgressBar(maxval=total_sentences)
    
    q_index = 0
    for sentence,label in zip(sentences,labels):
        
        q, length = process_text(sentence, vocab,
                                 max_length=20)
        d_questions[q_index, :length] = q
        if label == 'yesno' or label == 'factoid' or label == 'list' or label == 'summary':
            x=label
        else:
            print('error')
        l = process_label(label)
        d_labels[q_index] = int(l)
        q_index += 1
        bar.update(q_index)
    h5file.close()
      


if __name__ == '__main__' :
    
    parser=argparse.ArgumentParser()
    
    
    # Inputs
    parser.add_argument('--dataset', type=str, default='data/bioasq/train.tsv')
    
    parser.add_argument('--vocab', type=str, default = 'data/processed/vocab.json')
    
    # Outputs
    
    parser.add_argument('--output', type=str,
                        default='data/processed/dataset_train.hdf5',
                        help='directory for resized images.')
    
    # Hyperparameters
    parser.add_argument('--max-length', type=int, default=20,
                        help='maximum sequence length for sentences.')
    
    args=parser.parse_args()
    
    save_dataset(args.dataset, args.vocab,
                 args.output, 
                 max_length=args.max_length)
    print('Wrote dataset to %s' % args.output)
    # Hack to avoid import errors.
    #Vocabulary()