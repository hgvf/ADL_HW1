import numpy as np
import os
import pickle
import spacy
from typing import List, Dict
from utils import *

import torch
from torch.utils.data import Dataset

class intent_cls(Dataset):
    def __init__(self, data, vocab, label_mapping, tokenizer, num_class):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.tokenizer = tokenizer
        self.num_class = num_class
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # Tokenize
        word_piece = []
        text = self.data[index]['text']
        doc = self.tokenizer(text)
        for token in doc:
            word_piece.append(token.text)
        
        # Convert wordpiece to ids
        word_piece = self.vocab.token_to_id(word_piece)
       
        # Generate intent's ont-hot encoding tensor
        intent_idx = self.label2idx(self.data[index]['intent'])
        intent = torch.zeros(self.num_class)
        intent[intent_idx] = 1
        
        id = self.data[index]['id']
    
        return torch.tensor(word_piece, dtype=torch.long), torch.FloatTensor(intent), id
    
    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def label2idx(self, label):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]