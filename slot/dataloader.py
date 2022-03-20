import numpy as np
import os
import pickle
import spacy
from typing import List, Dict
from utils import *

import torch
from torch.utils.data import Dataset

class slot_tag(Dataset):
    def __init__(self, data, vocab, label_mapping, num_class):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.num_class = num_class
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # Convert tokens to ids
        text = self.data[index]['tokens']
        text = self.vocab.encode(text)
        
        # Generate label's ont-hot encoding
        label = self.label2idx(self.data[index]['tags'])
        label_one_hot = self.gen_one_hot(label)
        
        id = self.data[index]['id']
        
        return torch.tensor(text, dtype=torch.long), torch.FloatTensor(label_one_hot), id
    
    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def label2idx(self, label):
        tmp = []
        for i in range(len(label)):
            tmp.append(self.label_mapping[label[i]])
        return tmp

    def idx2label(self, idx: int):
        return self._idx2label[idx]
    
    def gen_one_hot(self, label):
        label_one_hot = torch.empty((len(label), self.num_class))
        for i in range(len(label)):
            tmp = torch.zeros(self.num_class)
            tmp[label[i]] = 1
            
            label_one_hot[i] = tmp
            
        return label_one_hot