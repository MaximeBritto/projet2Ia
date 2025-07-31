import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.text_generator import TextGenerator, TextDataset



def load_books():
    """Charge les livres pour l'entraînement et la validation."""
    # Livres pour l'entraînement (les 5 premiers)
    train_books = [f'HPBook{i}.txt' for i in range(1, 6)]
    # Livres pour la validation (les 2 derniers)
    val_books = [f'HPBook{i}.txt' for i in range(6, 8)]
    
    def load_books_list(book_list, set_name):
        """Charge et concatène les livres spécifiés."""
        text = ""
        print(f"\nChargement des livres de {set_name}...")
        for book in book_list:
            path = os.path.join('text', book)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text += f.read() + "\n\n"
                print(f"Chargé: {book}")
            except FileNotFoundError:
                print(f"Avertissement: {book} non trouvé, ignoré.")
        return text
    
    return load_books_list(train_books, "l'entraînement"), load_books_list(val_books, "la validation")

def train_model():
    # Paramètres
    seq_length = 50  # Longueur des séquences
    batch_size = 128  # Taille du lot
    embedding_dim = 128  # Dimension des embeddings
    hidden_dim = 256  # Taille des couches cachées
    n_layers = 1  # Nombre de couches LSTM
    dropout = 0.2
    n_epochs = 5  # Nombre d'époques
    learning_rate = 0.01  # Taux d'apprentissage
    
    # Charger et préparer les données
    print("Préparation des données...")
    train_text, val_text = load_train_val_texts()
    
    # Créer les datasets
    train_dataset = TextDataset(train_text, seq_length)
    val_dataset = TextDataset(val_text, seq_length)
    
    # S'assurer que les deux datasets utilisent le même vocabulaire
    # (au cas où certains caractères ne seraient présents que dans un sous-ensemble)
    vocab_size = len(train_dataset.chars)
    print(f"Taille du vocabulaire: {vocab_size} caractères")
    print(f"Taille de l'ensemble d'entraînement: {len(train_dataset)} séquences")
    print(f"Taille de l'ensemble de validation: {len(val_dataset)} séquences")
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=0,
        drop_last=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de {device} pour l'entraînement")
    
    model = TextGenerator(vocab_size, embedding_dim, hidden_dim, n_layers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Pour suivre la meilleure perte de validation
    best_val_loss = float('inf')
    
    # Entraînement
    print("\nDébut de l'entraînement...")
    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        
        print(f'\nDébut de l\'époque {epoch+1}/{n_epochs}')
        total_batches = len(train_loader)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Afficher la progression toutes les 10 itérations
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                print(f'  Batch {batch_idx+1}/{total_batches}')
            # Déplacer les données sur le bon appareil
            data, target = data.to(device), target.to(device)
            
            # Initialiser l'état caché
            hidden = model.init_hidden(data.size(0), device)
            
            # Mettre à zéro les gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(data, hidden)
            
            # Calculer la perte
            loss = criterion(output.view(-1, vocab_size), target.view(-1))
            
            # Backward pass et optimisation
            loss.backward()
            # Écrêtage des gradients pour éviter l'explosion des gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Afficher la perte toutes les 10 itérations
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                print(f'  Batch {batch_idx+1}/{total_batches}, Perte: {loss.item():.4f}')
        
        # Calculer la perte moyenne sur l'ensemble d'entraînement
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            print('\nValidation en cours...')
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                hidden = model.init_hidden(data.size(0), device)
                output, _ = model(data, hidden)
                loss = criterion(output.view(-1, vocab_size), target.view(-1))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Afficher les métriques
        print(f'\nÉpoque {epoch+1}/{n_epochs}:')
        print(f'  Perte entraînement: {avg_train_loss:.4f}')
        print(f'  Perte validation: {avg_val_loss:.4f}')
        
        # Sauvegarder le meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'char_to_int': train_dataset.char_to_int,
                'int_to_char': train_dataset.int_to_char,
                'vocab_size': vocab_size,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, f'models/checkpoint_epoch_{epoch+1}.pth')
    
    print("Entraînement terminé !")
    return model, dataset.char_to_int, dataset.int_to_char

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    model, char_to_int, int_to_char = train_model()
