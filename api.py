from flask import Flask, request, jsonify
import joblib
from spacy.cli import download
from app.utils import load_spacy_model, process_text

app = Flask(__name__)

# Charger le modèle SpaCy
download("en_core_web_sm")
nlp = load_spacy_model("en_core_web_sm")

# Chemins des fichiers du modèle
vectorizer_local_path = './artefact/vectorizer.pkl'
binarizer_local_path = './artefact/binarizer.pkl'
model_local_path = './artefact/model.pkl'

# Charger les fichiers
vectorizer = joblib.load(vectorizer_local_path)
binarizer = joblib.load(binarizer_local_path)
model = joblib.load(model_local_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    titre = data.get('titre')
    question = data.get('question')

    if titre and question:
        texte = process_text(nlp, titre + ' ' + question)
        texte = ' '.join(texte)
        vecteur = vectorizer.transform([texte])
        prediction = model.predict(vecteur)
        tags = binarizer.inverse_transform(prediction)
        return jsonify({'tags_predits': tags[0]})
    else:
        return jsonify({'error': 'Titre et question manquants'}), 400

if __name__ == '__main__':
    app.run(debug=True)


