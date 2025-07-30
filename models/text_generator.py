import torch
import torch.nn as nn
import numpy as np

class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=1, dropout=0.2):
        super(TextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
    def forward(self, x, hidden):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # Si hidden est un tuple (pour LSTM) ou un tenseur (pour GRU)
        if isinstance(hidden, tuple):
            hidden = tuple(h for h in hidden)
        
        # Forward pass LSTM
        output, hidden = self.lstm(embedded, hidden)
        
        # Passer à travers la couche linéaire
        output = self.fc(output)  # (batch_size, seq_length, vocab_size)
        
        # Détacher les états cachés pour éviter les fuites de mémoire
        if isinstance(hidden, tuple):
            hidden = tuple(h.detach() for h in hidden)
        else:
            hidden = hidden.detach()
            
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        # Initialisation de l'état caché
        # Créer des tenseurs non initialisés sur le bon appareil
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        
        # Pour une meilleure initialisation, on peut utiliser une initialisation xavier
        # nn.init.xavier_uniform_(h0)
        # nn.init.xavier_uniform_(c0)
        
        return (h0, c0)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, seq_length=100):
        self.seq_length = seq_length
        self.chars = sorted(list(set(text)))
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.encoded = [self.char_to_int[ch] for ch in text]
        
    def __len__(self):
        return len(self.encoded) - self.seq_length
    
    def __getitem__(self, idx):
        # Retourne une séquence d'entrée et une séquence cible décalée d'un pas
        seq_in = self.encoded[idx:idx + self.seq_length]
        seq_out = self.encoded[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(seq_in, dtype=torch.long), torch.tensor(seq_out, dtype=torch.long)
