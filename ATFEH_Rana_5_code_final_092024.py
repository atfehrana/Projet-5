import streamlit as st
import requests

# Titre de l'interface Streamlit
st.title('Classification de questions')

# Champ de saisie du titre
titre = st.text_input('Entrer Titre :')

# Champ de saisie de la question
question = st.text_input('Entrer Question :')

# Bouton de prédiction
if st.button('Suggestion des Tags'):
    if titre and question:
        url = 'https://api-f11f.onrender.com/predict'
        data = {'title': titre, 'question': question}
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            tags = response.json().get('suggested_tags', [])
            if tags:
                st.success(f'Tags suggérés : {tags}')
            else:
                st.warning('Aucun tag suggéré.')
        else:
            st.error(f'Une erreur est survenue : {response.text}')
    else:
        st.error('Veuillez saisir un titre ET une question.')
