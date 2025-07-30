import torch
import argparse
from models.text_generator import TextGenerator
import random

def load_model(checkpoint_path):
    # Charger le checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Récupérer les paramètres
    char_to_int = checkpoint['char_to_int']
    int_to_char = checkpoint['int_to_char']
    vocab_size = checkpoint['vocab_size']
    
    # Initialiser le modèle
    model = TextGenerator(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        n_layers=2,
        dropout=0.2
    )
    
    # Charger les poids du modèle
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, char_to_int, int_to_char

def generate_text(model, char_to_int, int_to_char, prompt, length=500, temperature=0.8):
    # Convertir le prompt en séquence d'entiers
    input_seq = [char_to_int.get(char, 0) for char in prompt]
    
    with torch.no_grad():
        # Préparer l'entrée
        input_tensor = torch.tensor([input_seq], dtype=torch.long)
        hidden = model.init_hidden(1, 'cpu')
        
        # Générer les caractères
        output_text = prompt
        for _ in range(length):
            output, hidden = model(input_tensor, hidden)
            
            # Obtenir le prochain caractère (échantillonnage avec température)
            output_dist = output[0, -1].div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            
            # Ajouter le caractère généré à la sortie
            char = int_to_char[top_i.item()]
            output_text += char
            
            # Mettre à jour l'entrée pour la prochaine itération
            input_tensor = torch.tensor([[top_i.item()]], dtype=torch.long)
            
            # Arrêter si on atteint une fin de phrase
            if char in ['.', '!', '?'] and len(output_text) > len(prompt) + 50:
                break
    
    return output_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Générer du texte avec un modèle LSTM entraîné')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoint.pth',
                       help='Chemin vers le fichier de checkpoint du modèle')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                       help='Texte de départ pour la génération')
    parser.add_argument('--length', type=int, default=500,
                       help='Longueur maximale du texte généré')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Température pour l\'échantillonnage (plus élevé = plus créatif)')
    
    args = parser.parse_args()
    
    # Charger le modèle
    print(f"Chargement du modèle depuis {args.checkpoint}...")
    model, char_to_int, int_to_char = load_model(args.checkpoint)
    
    # Générer du texte
    print(f"\nGénération avec le prompt: '{args.prompt}'")
    print("-" * 80)
    generated_text = generate_text(model, char_to_int, int_to_char, args.prompt, args.length, args.temperature)
    print(generated_text)
    print("-" * 80)
