import streamlit as st
import mlflow
import joblib
import spacy
import os
import re
from spacy.cli import download
from app.utils import load_spacy_model, process_text, predict_tags

# Définir l'URI de suivi MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://0.0.0.0:5000"))

# Chargement du modèle SpaCy:
download("en_core_web_sm")
nlp = load_spacy_model("en_core_web_sm")


binarizer_path = "./mlruns/1/a6da68aa450e4e9f8948baa7f4b61411/artifacts/binarizer"
vectorizer_path = "./mlruns/1/081864e183a54a2db9522707ad621bc6/artifacts/tfidf_vectorizer"
model_path = "./mlruns/1/0938a50bb1e04b4cb49204f0da4e37f7/artifacts/model/model.pkl"

# Charger les artefacts avec joblib
vectorizer = joblib.load(vectorizer_path)
binarizer = joblib.load(binarizer_path)
model = joblib.load(model_path)

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