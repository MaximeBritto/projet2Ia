import os
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from models.text_generator import TextGenerator
import random

app = Flask(__name__)

# Charger le modèle et les mappages de caractères
checkpoint = None
model = None
char_to_int = None
int_to_char = None
vocab_size = 0

def load_model():
    global checkpoint, model, char_to_int, int_to_char, vocab_size
    
    # Vérifier si un modèle entraîné existe
    checkpoint_path = 'models/checkpoint_epoch_5.pth'  # Utiliser le dernier checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Erreur: Fichier de modèle introuvable: {checkpoint_path}")
        return False
    
    # Déterminer l'appareil à utiliser
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Chargement du modèle sur {device}...")
    
    # Charger le checkpoint sur le bon appareil
    print(f"Chargement du checkpoint depuis: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Récupérer les paramètres
    char_to_int = checkpoint['char_to_int']
    int_to_char = checkpoint['int_to_char']
    vocab_size = checkpoint['vocab_size']
    
    # Initialiser le modèle avec les mêmes paramètres que pendant l'entraînement
    model = TextGenerator(
        vocab_size=vocab_size,
        embedding_dim=128,  # Ajusté pour correspondre au modèle entraîné
        hidden_dim=256,     # Ajusté pour correspondre au modèle entraîné
        n_layers=1,         # Ajusté pour correspondre au modèle entraîné
        dropout=0.2
    ).to(device)
    
    # Charger les poids du modèle
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Afficher des informations sur le modèle et le GPU
    print(f"Modèle chargé sur {next(model.parameters()).device}")
    if torch.cuda.is_available():
        print(f"Mémoire GPU allouée: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"Mémoire GPU réservée: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    
    return True

# Page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# API pour générer du texte
@app.route('/generate', methods=['POST'])
def generate_text():
    print("\n=== Début de la génération ===")
    data = request.json
    prompt = data.get('prompt', '').strip()
    print(f"Prompt reçu: '{prompt}'")
    
    if not prompt:
        print("Erreur: Prompt vide")
        return jsonify({'error': 'Le prompt est vide'}), 400
    
    try:
        # Définir l'appareil à utiliser (GPU si disponible)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilisation du périphérique: {device}")
        
        # Vérifier si le modèle est chargé
        if model is None or char_to_int is None or int_to_char is None:
            error_msg = 'Modèle non chargé. Modèle: {}, char_to_int: {}, int_to_char: {}'.format(
                'chargé' if model else 'non chargé',
                'défini' if char_to_int else 'non défini',
                'défini' if int_to_char else 'non défini'
            )
            print(f"Erreur: {error_msg}")
            return jsonify({'error': error_msg}), 500
            
        # Afficher les informations sur le vocabulaire
        print(f"Taille du vocabulaire: {len(char_to_int)} caractères")
        print(f"Exemple de mappage: 'a' -> {char_to_int.get('a', 'non trouvé')}")
        
        # Convertir le prompt en séquence d'entiers
        input_seq = [char_to_int.get(char, 0) for char in prompt]
        print(f"Séquence d'entrée: {input_seq}")
            
        # Générer du texte
        with torch.no_grad():
            # Préparer l'entrée et l'état caché initial
            input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
            hidden = model.init_hidden(1, device)
            
            # Générer les caractères
            output_text = prompt
            
            with torch.no_grad():
                # Passe initiale avec le prompt
                if len(input_seq) > 1:
                    _, hidden = model(input_tensor[:, :-1], hidden)
                
                # Dernier caractère du prompt
                input_tensor = input_tensor[:, -1:]
                
                # Génération caractère par caractère
                for _ in range(200):
                    output, hidden = model(input_tensor, hidden)
                    
                    # Obtenir le prochain caractère (échantillonnage avec température)
                    output_dist = output[0, -1].div(0.8).exp()
                    top_i = torch.multinomial(output_dist, 1)[0]
                    
                    # Ajouter le caractère généré à la sortie
                    char = int_to_char[top_i.item()]
                    output_text += char
                    
                    # Mettre à jour l'entrée pour la prochaine itération
                    input_tensor = top_i.unsqueeze(0).unsqueeze(0)
                    
                    # Arrêter si on atteint une fin de phrase
                    if char in ['.', '!', '?'] and len(output_text) > len(prompt) + 50:
                        break
                    
                    # Libérer la mémoire
                    torch.cuda.empty_cache()
        
        return jsonify({'generated_text': output_text})
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n=== ERREUR ===\n{error_trace}\n==============")
        return jsonify({
            'error': 'Erreur lors de la génération du texte',
            'details': str(e),
            'traceback': error_trace
        }), 500

if __name__ == '__main__':
    # Créer les dossiers nécessaires
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Charger le modèle si disponible
    if not load_model():
        print("Aucun modèle entraîné trouvé. Veuillez d'abord exécuter train.py")
    
    # Démarrer l'application
    app.run(debug=True, port=5000)
