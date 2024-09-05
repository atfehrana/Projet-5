import streamlit as st
import mlflow
import joblib
import os
import boto3
from spacy.cli import download
from app.utils import load_spacy_model, process_text, predict_tags

# Chargement du modèle SpaCy:
download("en_core_web_sm")
nlp = load_spacy_model("en_core_web_sm")

# Chemins absolus dans le répertoire Heroku
vectorizer_local_path = os.path.join(os.getcwd(), 'artifacts/vectorizer.pkl')
binarizer_local_path = os.path.join(os.getcwd(), 'artifacts/binarizer.pkl')
model_local_path = os.path.join(os.getcwd(), 'artifacts/model.pkl')

# Chargement des fichiers
vectorizer = joblib.load(vectorizer_local_path)
binarizer = joblib.load(binarizer_local_path)
model = joblib.load(model_local_path)

print("Tous les fichiers ont été chargés.")

print("Tous les fichiers ont été chargés.")

# Titre de l'interface Streamlit :
st.title('Classification de questions')

# Champ de saisie du titre :
titre = st.text_input('Entrer Titre :')

# Champ de saisie de la question :
question = st.text_input('Entrer Question :')

# Bouton de prédiction
if st.button('Suggestion des Tags'):
    if titre and question:
        texte = process_text(nlp, titre + ' ' + question)
        texte = ' '.join(texte)
        tags = predict_tags(vectorizer, binarizer, model, texte)
        tags = tags[0].tolist()
        st.write(f'Tags suggérés : {tags}')
    else:
        st.write('Veuillez saisir un titre ET une question.')

