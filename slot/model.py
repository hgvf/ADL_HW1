import torch
import torch.nn as nn 

from typing import Dict

class SeqClassifier(nn.Module):
    def __init__(self, 
                 embeddings: torch.tensor, 
                 hidden_size: int, 
                 num_layers: int, 
                 dropout: float, 
                 bidirectional: bool,
                 num_class: int,) -> None:
        super(SeqClassifier, self).__init__()
        
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False)
        
        self.rnn = nn.LSTM(input_size=300, hidden_size=hidden_size, dropout=dropout, bidirectional=bidirectional, 
                      num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size*2, num_class)
        
    def forward(self, x):
        x = self.embed(x)
        
        x, _ = self.rnn(x)
                
        x = self.fc(x).squeeze()
        
        return x