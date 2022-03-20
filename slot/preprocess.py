import json
import logging
import pickle
import re
import os
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from random import random, seed
from typing import List, Dict
from tqdm import tqdm
from utils import *

import torch

# 訓練參數設定
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='./data')
parser.add_argument("--output_dir", type=str, default='./cache')
parser.add_argument("--glove_path", type=str, default='../glove.840B.300d.txt')
parser.add_argument("--vocab_size", type=float, default=10000)
parser.add_argument("--random_seed", type=float, default=13)
opt = parser.parse_args()

def build_vocab(words, vocab_size, output_dir, glove_path):
    # 挑前 "vocab_size" 個常出現的 words
    common_words = {w for w, _ in words.most_common(vocab_size)}

    # 轉成 vocab
    vocab = Vocab(common_words)
    vocab_path = os.path.join(output_dir, "vocab.pkl")
    with open(vocab_path, 'wb') as file:
        pickle.dump(vocab, file)
        
    # creating glove embeddings
    glove = {}
    
    with open(glove_path) as fp:
        row1 = fp.readline()
        
        # if the first row is not header
        if not re.match("^[0-9]+ [0-9]+$", row1):
            # seek to 0
            fp.seek(0)
            
        for i, line in tqdm(enumerate(fp)):
            cols = line.rstrip().split(" ")
            word = cols[0]
            vector = [float(v) for v in cols[1:]]

            # skip word not in words if words are provided
            if word not in common_words:
                continue
            glove[word] = vector
            glove_dim = len(vector)

    assert all(len(v) == glove_dim for v in glove.values())
    assert len(glove) <= vocab_size
    
    num_matched = sum([token in glove for token in vocab.tokens])
    
    embeddings = [glove.get(token, [random() * 2 - 1 for _ in range(glove_dim)])
                    for token in vocab.tokens]
    
    # embeddings: (vocab_size, glove_dim)
    embeddings = torch.tensor(embeddings)
    embedding_path = os.path.join(output_dir, "embeddings.pt")
    torch.save(embeddings, str(embedding_path))

if __name__ == '__main__':
    seed(opt.random_seed)

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename='./preprocess.log', 
        filemode='a',
    )

    tags = set()
    words = Counter()
    for split in ["train", "eval"]:
        dataset_path = os.path.join(opt.data_dir, split)
        
        # open dataset json file
        f = open(dataset_path+'.json')
        dataset = json.load(f)

        tags.update({tag for instance in dataset for tag in instance["tags"]})
        words.update([token for instance in dataset for token in instance["tokens"]])

    tag2idx = {tag: i for i, tag in enumerate(tags)}
    tag_idx_path = os.path.join(opt.output_dir, "tag2idx.json")

    with open(tag_idx_path, 'w') as f:
        json.dump(tag2idx, f)

    build_vocab(words, opt.vocab_size, opt.output_dir, opt.glove_path)