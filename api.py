from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from app.utils import load_spacy_model, process_text

app = Flask(__name__)
CORS(app)  # Activer CORS pour toutes les routes

# Charger le modèle SpaCy
nlp = load_spacy_model("en_core_web_sm")

# Charger les artefacts de modèle
vectorizer_local_path = './artefact/vectorizer.pkl'
binarizer_local_path = './artefact/binarizer.pkl'
model_local_path = './artefact/model.pkl'

vectorizer = joblib.load(vectorizer_local_path)
binarizer = joblib.load(binarizer_local_path)
model = joblib.load(model_local_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'titre' not in data or 'question' not in data:
        return jsonify({'error': 'Invalid data'}), 400

    titre = data['titre']
    question = data['question']

    # Traiter le texte et faire la prédiction
    texte = process_text(nlp, titre + ' ' + question)
    texte = ' '.join(texte)
    vecteur = vectorizer.transform([texte])
    prediction = model.predict(vecteur)
    tags = binarizer.inverse_transform(prediction)

    return jsonify({'tags_predits': tags[0]})

if __name__ == '__main__':
    app.run(debug=True)



