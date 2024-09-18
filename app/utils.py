import spacy
import os
import re

def load_spacy_model(model_name: str):
    nlp = spacy.load(model_name)
    print(f"Successfully loaded SpaCy model: {model_name}")
    return nlp


def process_text(nlp, text, allowed_words=None, unique=False):


    text = re.sub(r'[^a-zA-Z\s\+\-_]', ' ', text)
    doc = nlp(text)
    
    if allowed_words:
        words = [token.lemma_.lower() for token in doc\
                 if not token.is_stop and len(token.text) >= 3 and token.lemma_.lower() in allowed_words]
    else:
        words = [token.lemma_.lower() for token in doc if not token.is_stop and len(token.text) >= 3]

    if unique == True:
        words = list(set(words))
    words = [word for word in words if word.strip()]
    
    return words

import numpy as np

def predict_tags(vectorizer, binarizer, model, texte):
    X = vectorizer.transform([texte])
    y_pred = model.predict(X)
    tags = binarizer.inverse_transform(y_pred)
    tags = tags[0] if len(tags) > 0 else []
    return tags