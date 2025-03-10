from flask import Flask, request, jsonify
import joblib
import numpy as np
from spacy.cli import download
from app.utils import load_spacy_model, process_text, predict_tags

# Initialiser l'application Flask
app = Flask(__name__)

# Chargement du modèle SpaCy
download("en_core_web_sm")
nlp = load_spacy_model("en_core_web_sm")

# Charger les fichiers
try:
    vectorizer_local_path = './artefact/vectorizer.pkl'
    binarizer_local_path = './artefact/binarizer.pkl'
    model_local_path = './artefact/model.pkl'

    vectorizer = joblib.load(vectorizer_local_path)
    binarizer = joblib.load(binarizer_local_path)
    model = joblib.load(model_local_path)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement des modèles : {e}")

@app.route('/')
def home_page():
    return 'Bienvenue sur l\'API de suggestion de tags'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    title = data.get('title')
    question = data.get('question')
    
    if not title or not question:
        return jsonify({'error': 'Titre et question sont requis.'}), 400

    try:
        texte = process_text(nlp, title + ' ' + question)
        texte = ' '.join(texte)  

        tags = predict_tags(vectorizer, binarizer, model, texte)
        tags = tags[0].tolist()  

        return jsonify({'suggested_tags': tags})
    except Exception as e:
        return jsonify({'error': f'Erreur lors de la prédiction : {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
