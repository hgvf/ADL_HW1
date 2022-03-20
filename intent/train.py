import json
import pickle
import spacy
import argparse
import logging

from pathlib import Path
from typing import Dict
from tqdm import tqdm
from model import *
from dataloader import *
from utils import *

import torch
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.utils.data import DataLoader

# Load datasets & dataloader
def collate_fn(data):
    text, intent, id = zip(*data)
    
    text = rnn_utils.pad_sequence(list(text), batch_first=True)
    intent = rnn_utils.pad_sequence(list(intent), batch_first=True)
    
    return text, intent, id

def train(model, optimizer, loader, criterion, epoch, n_epochs, device):
    model.train()
    
    train_loss = 0.0
    
    loop = tqdm(enumerate(loader), total=len(loader))
    for _, (text, intent, _) in loop:
        text, intent = text.to(device), intent.to(device)

        out = model(text)

        loss = criterion(out, intent)
        
        loss.backward()
        optimizer.step()

        model.zero_grad()
        optimizer.zero_grad()
        
        # 計算總誤差
        train_loss += loss.item() * text.size(0)
        
        # 設定 tqdm 要顯示的東西
        loop.set_description(f"[Train Epoch {epoch}/{n_epochs}]")
        loop.set_postfix(loss=loss.item())
        
    epoch_loss = train_loss / len(loader)
    
    return epoch_loss

def valid(model, optimizer, loader, criterion, epoch, n_epochs, device):
    model.eval()
    
    valid_loss = 0.0
    
    loop = tqdm(enumerate(loader), total=len(loader))
    for _, (text, intent, _) in loop:
        #print(text, intent)
        text, intent = text.to(device), intent.to(device)
        
        with torch.no_grad():
            out = model(text)

            loss = criterion(out, intent)
         
        # 計算總誤差
        valid_loss += loss.item() * text.size(0)
        
        # 設定 tqdm 要顯示的東西
        loop.set_description(f"[Valid Epoch {epoch}/{n_epochs}]")
        loop.set_postfix(loss=loss.item())
        
    epoch_loss = valid_loss / len(loader)
    
    return epoch_loss

if __name__ == '__main__':
    # 訓練參數設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/4/')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bidirectional', type=str, default='True')
    opt = parser.parse_args()

    if not os.path.exists(opt.ckpt_dir):
        os.mkdir(opt.ckpt_dir)
        
    logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=os.path.join(opt.ckpt_dir, 'train.log'),
            filemode='a',
        )
    
    # 設定 device (opt.device = 'cpu' or 'cuda:X')
    if opt.device[:4] == 'cuda':
        gpu_id = opt.device[-1]
        #os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        torch.cuda.set_device(opt.device)
        device = torch.device(opt.device)
    else:
        print('device: cpu')
        device = torch.device('cpu')

    logging.info('start training...')
    logging.info('device: %s' %(opt.device))
    logging.info('======================================================')

    # Tokenizer
    tokenizer = spacy.load("en_core_web_sm")

    # Load vocab & intent
    with open(os.path.join(opt.cache_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)

    intent_idx_path = os.path.join(opt.cache_dir, "intent2idx.json")
    f = open(intent_idx_path)
    intent2idx = json.load(f)

    # get the num_class
    num_class = len(list(intent2idx.keys()))

    # Load train & valid files
    data_paths = {split: os.path.join(opt.data_dir, f"{split}.json") for split in ['train', 'eval']}

    data = {}
    for split, path in data_paths.items():
        f = open(path)
        data[split] = json.load(f)

    # train & eval datasets
    datasets = {split: intent_cls(split_data, vocab, intent2idx, tokenizer, num_class) for split, split_data in data.items()}

    # dataloaders
    train_loader = DataLoader(datasets['train'], batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(datasets['eval'], batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    # model
    embed = torch.load('./cache/embeddings.pt')
    if opt.bidirectional == 'True':
        model = SeqClassifier(embed, opt.hidden_size, opt.num_layers, opt.dropout, True, num_class).to(device)
    else:
        model = SeqClassifier(embed, opt.hidden_size, opt.num_layers, opt.dropout, False, num_class).to(device)

    # Create optimizer & criterion
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)

    min_loss = 1000000
    early_stop_cnt = 0

    for epoch in range(opt.n_epochs):
        train_loss = train(model, optimizer, train_loader, criterion, epoch, opt.n_epochs, device)
        
        valid_loss = valid(model, optimizer, valid_loader, criterion, epoch, opt.n_epochs, device)
        
        logging.info('[Train] epoch: %d -> loss: %.4f' %(epoch, train_loss))
        logging.info('[Valid] epoch: %d -> loss: %.4f' %(epoch, valid_loss))
        logging.info('======================================================')

        if valid_loss <= min_loss:
            targetPath = os.path.join(opt.ckpt_dir, 'model.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer,
                'min_loss': min_loss
            }, targetPath)
            
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
            
        if early_stop_cnt == 20:
            logging.info('early stopping...')
            break

    logging.info('finish training...')