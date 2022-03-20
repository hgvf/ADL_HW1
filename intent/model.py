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
        
        self.lstm = nn.LSTM(input_size=300, hidden_size=hidden_size, dropout=dropout, bidirectional=bidirectional, 
                      num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size*2, num_class)
        #self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.embed(x)
        
        x, (h, _) = self.lstm(x)
        
        h_forward, h_backward = h[-2, :, :], h[-1, :, :]
      
        last_hidden = torch.cat((h_forward, h_backward), axis=-1)
        last_hidden = last_hidden.unsqueeze(0)
        
        out = self.fc(last_hidden).squeeze()
        # out = self.fc(x[:, -1]).squeeze()
        
        return out