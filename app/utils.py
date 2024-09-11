import spacy
import os
import re
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


def load_spacy_model(model_name: str):
    nlp = spacy.load(model_name)
    print(f"Successfully loaded SpaCy model: {model_name}")
    return nlp


def process_text(nlp, text, allowed_words=None, unique=False):

    # Suppression des caractères qui ne sont pas des lettres ou un des symboles (+, -, _) :
    text = re.sub(r'[^a-zA-Z\s\+\-_]', ' ', text)

    # Application du modèle linguistique au texte :
    doc = nlp(text)
    
    # Lemmatisation des mots, exclusion des stopwords, des mots de longueur inférieure à 3 et éventuellement, des mots non-autorisés :
    if allowed_words:
        words = [token.lemma_.lower() for token in doc\
                 if not token.is_stop and len(token.text) >= 3 and token.lemma_.lower() in allowed_words]
    else:
        words = [token.lemma_.lower() for token in doc if not token.is_stop and len(token.text) >= 3]

    # Suppression des doublons.
    if unique == True:
        words = list(set(words))

    # Supression des chaînes vides et des espaces supplémentaires.
    words = [word for word in words if word.strip()]
    
    return words

def predict_tags(binarizer, model, embeddings):
    y_pred = model.predict(embeddings)
    tags = binarizer.inverse_transform(y_pred)
    return tags

def get_use_embeddings(texts,  batch_size=8):
    
    model_use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model_use(batch_texts).numpy()
        all_embeddings.append(batch_embeddings)
    return np.concatenate(all_embeddings, axis=0)