#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:58:00 2020

@author: sarroutim2
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:19:53 2020

@author: sarroutim2
"""

import h5py
import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable


class QDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, dataset, max_examples=None,
                 indices=None):
        """Set the path for images, captions and vocabulary wrapper.
        Args:
            dataset: hdf5 file with questions and images.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
            indices: List of indices to use.
        """
        self.dataset = dataset
        self.max_examples = max_examples
        self.indices = indices

    def __getitem__(self, index):
        
        if not hasattr(self, 'questions'):
            annos = h5py.File(self.dataset, 'r')
            self.sentences = annos['sentences']
            self.labels = annos['labels']
            
        if self.indices is not None:
            index = self.indices[index]
        sentence = self.sentences[index]
        label = self.labels[index]

        sentence = torch.from_numpy(sentence)
        qlength = sentence.size(0) - sentence.eq(0).sum(0).squeeze()

        return (sentence, label, qlength.item())

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        annos = h5py.File(self.dataset, 'r')
        return annos['sentences'].shape[0]


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples.
    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, question, answer, answer_type, length).
            - question: torch tensor of shape (?); variable length.
            - label: Int for category label
            - qlength: Int for question length.
    Returns:
        questions: torch tensor of shape (batch_size, padded_length).
        labels: torch tensor of shape (batch_size,).
        qindices: torch tensor of shape(batch_size,).
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: x[2], reverse=True)
    sentences, labels, qlengths = zip(*data)
    sentences = torch.stack(sentences, 0).long()
    labels = torch.Tensor(labels).long()
    qindices = np.flip(np.argsort(qlengths), axis=0).copy()
    qindices = torch.Tensor(qindices).long()
    return sentences, labels, qindices


def get_loader(dataset, batch_size, sampler=None,
                   shuffle=True, num_workers=1, max_examples=None,
                   indices=None):
    """Returns torch.utils.data.DataLoader for custom dataset.
    Args:
        dataset: Location of annotations hdf5 file.
        batch_size: How many data points per batch.
        sampler: Instance of WeightedRandomSampler.
        shuffle: Boolean that decides if the data should be returned in a
            random order.
        num_workers: Number of threads to use.
        max_examples: Used for debugging. Assumes that we have a
            maximum number of training examples.
        indices: List of indices to use.
    Returns:
        A torch.utils.data.DataLoader for custom engagement dataset.
    """
    iq = QDataset(dataset, max_examples=max_examples,
                    indices=indices)
    data_loader = torch.utils.data.DataLoader(dataset=iq,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader