import streamlit as st
import mlflow
import joblib
import os
import boto3
from spacy.cli import download
from app.utils import load_spacy_model, process_text, predict_tags

# Définir l'URI de suivi MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://15.237.111.188:5000"))

# Chargement du modèle SpaCy:
download("en_core_web_sm")
nlp = load_spacy_model("en_core_web_sm")
import os
import joblib
import boto3

# Initialiser le client S3
s3 = boto3.client('s3')
bucket_name = 'mlflow-ratfeh'

# Chemins locaux dans /tmp
vectorizer_local_path = '/tmp/vectorizer.pkl'
binarizer_local_path = '/tmp/binarizer.pkl'
model_local_path = '/tmp/model.pkl'

vectorizer = joblib.load(vectorizer_local_path)
binarizer = joblib.load(binarizer_local_path)
model = joblib.load(model_local_path)

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

