#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:52:09 2020

@author: sarroutim2
"""

import argparse
import json
import logging
import nltk
import numpy as np
import re
import csv

from collections import Counter

from utils import Vocabulary

def tokenize(sentence):
    """ tokenize sentence into words.
    
    args : 
        senetence: a string for words.
        
    return : 
        a list of words.
    """
    if len(sentence)==0:
        return []
    sentence = re.sub('\.+', r'.', sentence)
    sentence = re.sub('([a-z])([.,!?()])', r'\1 \2 ', sentence)
    sentence = re.sub('\s+', ' ', sentence)
    tokens=nltk.tokenize.word_tokenize(
            sentence.strip().lower())
    
    
    return tokens

def build_vocab(questions,questions_val, questions_test, threshold):
    
    ''' read file
    
    '''
    counter=Counter()
    with open (questions) as questions_file:
        csv_reader=csv.reader(questions_file, delimiter='\t')
        
        line_count=0
        column_names=[]
        for row in csv_reader:
            
            if line_count==0:
                column_names.append(row[0])
                column_names.append(row[1])
                line_count+=1
            else:
                tokens=tokenize(row[1])
                counter.update(tokens)

    with open (questions_val) as questions_file:
        csv_reader=csv.reader(questions_file, delimiter='\t')
        
        line_count=0
        column_names=[]
        for row in csv_reader:
            
            if line_count==0:
                column_names.append(row[0])
                column_names.append(row[1])
                line_count+=1
            else:
                tokens=tokenize(row[1])
                counter.update(tokens)   

    with open (questions_test) as questions_file:
        csv_reader=csv.reader(questions_file, delimiter='\t')
        
        line_count=0
        column_names=[]
        for row in csv_reader:
            
            if line_count==0:
                column_names.append(row[0])
                column_names.append(row[1])
                line_count+=1
            else:
                tokens=tokenize(row[1])
                counter.update(tokens)
    # If word frequency is less than threshold then the word is discarded
    
    tokens=[]
    tokens=[word for word, cnt in counter.items() if cnt>=threshold]
    
    vocab=create_vocab(tokens)
    
    return vocab


def create_vocab(words):
    
    vocab=Vocabulary()
    
    for i,w in enumerate(words):
        vocab.add_word(w)
    
    return vocab
                
def load_vocab(vocab_path):
    """Load Vocabulary object from a pickle file.
    Args:
        vocab_path: The location of the vocab pickle file.
    Returns:
        A Vocabulary object.
    """
    vocab = Vocabulary()
    vocab.load(vocab_path)
    return vocab
              
def process_text(text, vocab, max_length=20):
    """Converts text into a list of tokens surrounded by <start> and <end>.
    Args:
        text: String text.
        vocab: The vocabulary instance.
        max_length: The max allowed length.
    Returns:
        output: An numpy array with tokenized text.
        length: The length of the text.
    """
    tokens = tokenize(text.lower().strip())
    output = []
    output.extend([vocab(token) for token in tokens])
    length = min(max_length, len(output))
    return np.array(output[:length]), length

def process_label(label):
    """Converts text into a list of tokens.
    Args:
        text: String text.
        vocab: The vocabulary instance.
        max_length: The max allowed length.
    Returns:
        output: An numpy array with tokenized text.
        length: The length of the text.
    """
    labels = {"yesno" : 1,
              "factoid": 2,
              "list": 3,
              "summary": 4
              }
    output = []
    output.extend([labels[label]])
    return np.array(output)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inputs.
    parser.add_argument('--questions', type=str,
                        default='data/bioasq/train.tsv',
                        help='Path for train questions file.')
    parser.add_argument('--questions_val', type=str,
                        default='data/bioasq/dev.tsv',
                        help='Path for train questions file.')
    parser.add_argument('--questions_test', type=str,
                        default='data/bioasq/test.tsv',
                        help='Path for train questions file.')
    # Hyperparameters.
    parser.add_argument('--threshold', type=int, default=2,
                        help='Minimum word count threshold.')

    # Outputs.
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab.json',
                        help='Path for saving vocabulary wrapper.')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    vocab = build_vocab(args.questions,args.questions_val,args.questions_test, args.threshold)
    logging.info("Total vocabulary size: %d" % len(vocab))
    vocab.save(args.vocab_path)
    logging.info("Saved the vocabulary wrapper to '%s'" % args.vocab_path)  