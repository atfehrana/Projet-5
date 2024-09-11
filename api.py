from flask import Flask, request, jsonify
import spacy
import joblib
import os
import requests
import joblib
from io import BytesIO
from app.utils import load_spacy_model, process_text, predict_tags

# Initialize the Flask app
app = Flask(__name__)

# Load SpaCy model (make sure 'en_core_web_sm' is installed)
try:
    nlp = spacy.load('en_core_web_sm')
    print("SpaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

response_binarizer = requests.get("https://mlflow-ratfeh.s3.eu-west-3.amazonaws.com/53ee1c48888743c28a5a733abe06a58f/artifacts/binarizer/binarizer.pkl")
binarizer = joblib.load(BytesIO(response_binarizer.content))

response_vectorizer = requests.get("https://mlflow-ratfeh.s3.eu-west-3.amazonaws.com/53ee1c48888743c28a5a733abe06a58f/artifacts/tfidf_vectorizer/vectorizer.pkl")
vectorizer = joblib.load(BytesIO(response_vectorizer.content))

response_model = requests.get("https://mlflow-ratfeh.s3.eu-west-3.amazonaws.com/53ee1c48888743c28a5a733abe06a58f/artifacts/model/model.pkl")
model = joblib.load(BytesIO(response_model.content))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Assurez-vous que les champs titre et question sont présents
    titre = data.get('titre')
    question = data.get('question')

    if titre and question:
        # Combinez le titre et la question
        texte_combine = titre + ' ' + question
        
        # Traitement du texte
        texte_propre = process_text(nlp, texte_combine)
        texte_propre = ' '.join(texte_propre)
        
        # Prédiction des tags
        tags_suggérés = predict_tags(vectorizer, binarizer, model, texte_propre)
        
        # Retourner les résultats sous forme JSON
        return jsonify({'tags_suggérés': tags_suggérés})
    
    return jsonify({'erreur': 'Veuillez fournir un titre et une question.'}), 400


if __name__ == '__main__':
    port = int(os.environ.get("API_PORT", 5000))
    app.run(host='0.0.0.0', port=port)
