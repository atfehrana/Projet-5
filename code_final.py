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
    print(f"Successfully downloaded {artifact_path} from run {run_id}")
    return artifact

# Download the binarizer
vectorizer = download_artifact(run_id="53ee1c48888743c28a5a733abe06a58f", artifact_path="artifacts/tfidf_vectorizer/vectorizer.pkl")

# Attempt to download another binarizer
binarizer = download_artifact(run_id="49a470a4460641bca21d1fae26787160", artifact_path="artifacts/binarizer/binarizer.pkl")

# Download the model
model = download_artifact(run_id="5f5df97741c34193ad813305014b75d5", artifact_path="artifacts/model/model.pkl")
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