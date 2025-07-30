import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.text_generator import TextGenerator, TextDataset
import numpy as np

def load_text_files(directory):
    """Charge tous les fichiers texte du répertoire et les concatène."""
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                text += f.read()
    return text

def train_model():
    # Paramètres
    seq_length = 50  # Réduit la longueur des séquences
    batch_size = 128  # Augmenté pour accélérer l'entraînement
    embedding_dim = 128  # Réduit la dimension des embeddings
    hidden_dim = 256  # Réduit la taille des couches cachées
    n_layers = 1  # Une seule couche LSTM
    dropout = 0.2
    n_epochs = 5  # Seulement 5 époques
    learning_rate = 0.01  # Taux d'apprentissage plus élevé
    
    # Charger et préparer les données
    print("Chargement des données...")
    text = load_text_files('text')
    
    # Créer le dataset
    dataset = TextDataset(text, seq_length)
    vocab_size = len(dataset.chars)
    
    # Diviser en ensembles d'entraînement et de validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Créer les DataLoaders
    # Désactiver le chargement parallèle pour éviter les problèmes de fork sous Windows
    num_workers = 0
    # Utiliser drop_last pour éviter les batchs de taille différente
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Initialiser le modèle sur le GPU si disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du périphérique: {device}")
    
    model = TextGenerator(vocab_size, embedding_dim, hidden_dim, n_layers, dropout).to(device)
    
    # Afficher des informations sur le GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU allouée: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"Mémoire GPU réservée: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    
    # Fonction de perte et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Boucle d'entraînement
    print("Début de l'entraînement...")
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Déplacer les données vers le GPU
            x, y = x.to(device), y.to(device)
            
            # Réinitialiser l'état caché pour chaque batch
            # pour éviter les problèmes de taille de batch variable
            hidden = model.init_hidden(x.size(0), device)
            
            # Mettre les gradients à zéro
            optimizer.zero_grad()
            
            # Forward pass
            output, hidden = model(x, hidden)
            
            # Calculer la perte
            loss = criterion(output.transpose(1, 2), y)
            
            # Libérer la mémoire du GPU si nécessaire
            torch.cuda.empty_cache()
            
            # Backward pass et optimisation
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Époque {epoch+1}/{n_epochs}, Batch {batch_idx}/{len(train_loader)}, Perte: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                # Initialiser un nouvel état caché pour chaque batch de validation
                hidden = model.init_hidden(x.size(0), device)
                output, _ = model(x, hidden)
                loss = criterion(output.transpose(1, 2), y)
                val_loss += loss.item()
                
                # Libérer la mémoire du GPU
                del hidden, output, loss
                torch.cuda.empty_cache()
        
        # Afficher les statistiques d'époque
        print(f'Époque {epoch+1}/{n_epochs}, Perte entraînement: {train_loss/len(train_loader):.4f}, Perte validation: {val_loss/len(val_loader):.4f}')
        
        # Sauvegarder le modèle
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss/len(train_loader),
                'char_to_int': dataset.char_to_int,
                'int_to_char': dataset.int_to_char,
                'vocab_size': vocab_size
            }
            torch.save(checkpoint, f'models/checkpoint_epoch_{epoch+1}.pth')
    
    print("Entraînement terminé !")
    return model, dataset.char_to_int, dataset.int_to_char

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    model, char_to_int, int_to_char = train_model()
