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

import streamlit as st
import joblib
import os
import boto3
from spacy.cli import download
from app.utils import load_spacy_model, process_text, predict_tags

# Initialiser le client S3
s3 = boto3.client('s3')
bucket_name = 'mlflow-ratfeh'

# Clés S3
vectorizer_key = '53ee1c48888743c28a5a733abe06a58f/artifacts/tfidf_vectorizer/vectorizer.pkl'
binarizer_key = '53ee1c48888743c28a5a733abe06a58f/artifacts/binarizer/binarizer.pkl'
model_key = '53ee1c48888743c28a5a733abe06a58f/artifacts/model/model.pkl'

# Chemins locaux dans /tmp
vectorizer_local_path = '/tmp/vectorizer.pkl'
binarizer_local_path = '/tmp/binarizer.pkl'
model_local_path = '/tmp/model.pkl'

# Fonction pour télécharger si nécessaire
def download_if_not_exists(s3, bucket_name, s3_key, local_path):
    if not os.path.exists(local_path):
        try:
            st.write(f"Téléchargement de {s3_key} depuis S3 vers {local_path}")
            s3.download_file(bucket_name, s3_key, local_path)
            st.write(f"Téléchargement réussi de {s3_key}")
        except Exception as e:
            st.write(f"Erreur lors du téléchargement de {s3_key}: {e}")
    else:
        st.write(f"Fichier déjà présent : {local_path}")

# Télécharger les fichiers depuis S3 si nécessaire
download_if_not_exists(s3, bucket_name, vectorizer_key, vectorizer_local_path)
download_if_not_exists(s3, bucket_name, binarizer_key, binarizer_local_path)
download_if_not_exists(s3, bucket_name, model_key, model_local_path)

# Vérifier si les fichiers existent avant de les charger
if os.path.exists(vectorizer_local_path):
    vectorizer = joblib.load(vectorizer_local_path)
    st.write("Vectorizer chargé.")
else:
    st.write("Le fichier vectorizer.pkl est introuvable.")

if os.path.exists(binarizer_local_path):
    binarizer = joblib.load(binarizer_local_path)
    st.write("Binarizer chargé.")
else:
    st.write("Le fichier binarizer.pkl est introuvable.")

if os.path.exists(model_local_path):
    model = joblib.load(model_local_path)
    st.write("Modèle chargé.")
else:
    st.write("Le fichier model.pkl est introuvable.")


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

