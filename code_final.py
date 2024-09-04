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

def download_artifact(run_id, artifact_path):
    artifact = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    return artifact

adresse = "s3://mlflow-ratfeh/53ee1c48888743c28a5a733abe06a58f/artifacts/tfidf_vectorizer/vectorizer.pkl"
vectorizer_path = mlflow.artifacts.download_artifacts(adresse)
vectorizer = joblib.load(vectorizer_path)

# Chargement du binarizer :
adresse = "s3://mlflow-ratfeh/53ee1c48888743c28a5a733abe06a58f/artifacts/binarizer/binarizer.pkl"
binarizer_path = mlflow.artifacts.download_artifacts(adresse)
binarizer = joblib.load(binarizer_path)

# Chargement du modèle de classification :
adresse = "s3://mlflow-ratfeh/53ee1c48888743c28a5a733abe06a58f/artifacts/model/model.pkl"
model = mlflow.sklearn.load_model(adresse)

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