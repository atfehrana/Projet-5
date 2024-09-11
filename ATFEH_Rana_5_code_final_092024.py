import streamlit as st
import joblib
import requests
from io import BytesIO
from app.utils import load_spacy_model, process_text, predict_tags
from spacy.cli import download

download("en_core_web_sm")
nlp = load_spacy_model("en_core_web_sm")

st.title('Classification de questions')

vectorizer_local_path = './artefact/vectorizer.pkl'
binarizer_local_path = './artefact/binarizer.pkl'
model_local_path = './artefact/model.pkl'

# Charger les fichiers
vectorizer = joblib.load(vectorizer_local_path)
binarizer = joblib.load(binarizer_local_path)
model = joblib.load(model_local_path)

titre = st.text_input('Entrer Titre :')

question = st.text_input('Entrer Question :')

if titre and question:
    texte = process_text(nlp, titre + ' ' + question)
    texte = ' '.join(texte)
    tags = predict_tags(vectorizer, binarizer, model, texte)
    tags = tags[0].tolist()
    st.write(f'Tags suggérés : {tags}')
else:
    st.write('Veuillez saisir un titre ET une question.')