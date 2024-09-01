import streamlit as st
import mlflow
import joblib
import spacy
import os
import re
from spacy.cli import download
from app.utils import load_spacy_model, process_text, predict_tags

# Définir l'URI de suivi MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://15.237.111.188:5000"))

# Chargement du modèle SpaCy:
download("en_core_web_sm")
nlp = load_spacy_model("en_core_web_sm")

run_id = "a6da68aa450e4e9f8948baa7f4b61411"
binarizer_path = "artifacts/binarizer/binarizer.pkl"
binarizer = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=binarizer_path)

run_id = "49a470a4460641bca21d1fae26787160"
binarizer_path = "artifacts/binarizer/binarizer.pkl"
binarizer = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=binarizer_path)

run_id = "5f5df97741c34193ad813305014b75d5"
model_path = "artifacts/model/model.pkl"
model = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=model_path)

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