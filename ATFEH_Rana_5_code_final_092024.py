import streamlit as st
import joblib
import requests
from io import BytesIO
from app.utils import load_spacy_model, process_text, get_use_embeddings, predict_tags
from spacy.cli import download

download("en_core_web_sm")
nlp = load_spacy_model("en_core_web_sm")

st.title('Classification de questions')

response_binarizer = requests.get("https://mlflow-ratfeh.s3.eu-west-3.amazonaws.com/53ee1c48888743c28a5a733abe06a58f/artifacts/binarizer/binarizer.pkl")
binarizer = joblib.load(BytesIO(response_binarizer.content))


response_model = requests.get("https://mlflow-ratfeh.s3.eu-west-3.amazonaws.com/53ee1c48888743c28a5a733abe06a58f/artifacts/model/model.pkl")
model = joblib.load(BytesIO(response_model.content))


titre = st.text_input('Entrer Titre :')

question = st.text_input('Entrer Question :')

if titre and question:
    texte = process_text(nlp, titre + ' ' + question) 
    texte = ' '.join(texte) 
    texte_list = [texte] 
    embeddings = get_use_embeddings(texte_list)
    tags = predict_tags(model, embeddings)
    tags = tags[0].tolist()
    st.write(f'Tags suggérés : {tags}')
else:
    st.write('Veuillez saisir un titre ET une question.')