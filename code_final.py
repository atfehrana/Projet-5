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


binarizer_path = "./mlruns/1/01d924a6c9a84f9c807d3d86a4c739c0/artifacts/binarizer/binarizer.pkl"
vectorizer_path = "./mlruns/1/b76d4e5da477429ab13918eceaf65976/artifacts/tfidf_vectorizer/vectorizer.pkl"
model_path = "./mlruns/1/28989e27115f4f73acd9e18b0bf74671/artifacts/Regression_logistique/model.pkl"

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