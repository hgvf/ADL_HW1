import json
import pickle
import spacy
import argparse
import csv

from pathlib import Path
from typing import Dict
from tqdm import tqdm
from model import *
from dataloader import *
from utils import *

import torch
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Test dataloader
class test_tag(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Convert tokens to ids
        text = self.data[index]['tokens']
        text = self.vocab.encode(text)

        id = self.data[index]['id']
    
        return torch.tensor(text, dtype=torch.long), id

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

# Load datasets & dataloader
def collate_fn(data):
    text, id = zip(*data)
    
    text = rnn_utils.pad_sequence(list(text), batch_first=True)
    
    return text, id

if __name__ == '__main__':
    # 訓練參數設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/test.json')
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--tag2idx_path', type=str, default='./cache/tagt2idx.json')
    parser.add_argument('--vocab_path', type=str, default='./cache/vocab.pkl')
    parser.add_argument('--embedding_path', type=str, default='./cache/embeddings.pt')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/3/model.pt')
    parser.add_argument('--output_path', type=str, default='./output.csv')
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bidirectional', type=str, default='True')
    opt = parser.parse_args()

    # Load vocab & intent
    with open(opt.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    f = open(opt.tag2idx_path)
    tag2idx = json.load(f)

    # get the num_class
    num_class = len(list(tag2idx.keys()))

    # Load train & valid files    
    f = open(opt.data_dir)
    data = json.load(f)

    dataset = test_tag(data, vocab)
    test_loader = DataLoader(dataset, batch_size=50, shuffle=False, collate_fn=collate_fn)

    # model
    embed = torch.load(opt.embedding_path)
    if opt.bidirectional == 'True':
        model = SeqClassifier(embed, opt.hidden_size, opt.num_layers, opt.dropout, True, num_class)
    else:
        model = SeqClassifier(embed, opt.hidden_size, opt.num_layers, opt.dropout, False, num_class)

    ckpt = torch.load(opt.ckpt_dir, map_location='cpu')
    model.load_state_dict(ckpt['model'])

    with open(opt.output_path, 'w') as csvfile:
            writer = csv.writer(csvfile)

            # 寫入一列資料
            writer.writerow(['id', 'tags'])

    for text, id in tqdm(test_loader):
        out = model(text)
        
        max_class = torch.argmax(out, dim=-1)
        
        with open(opt.output_path, 'a') as csvfile:
            writer = csv.writer(csvfile)

            for i in range(max_class.shape[0]):
                toWrite = []
                toWrite.append(id[i])

                output = ""
                for w in range(max_class.shape[1]):
                    # Skip mask
                    if text[i][w] == 0:
                        break

                    output += (list(tag2idx.keys())[list(tag2idx.values()).index(max_class[i][w])])
                    output += " "

                toWrite.append(output[:-1])
                writer.writerow(toWrite)