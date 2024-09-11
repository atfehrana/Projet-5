import streamlit as st
import requests

st.title('Classification de questions')

titre = st.text_input('Entrer Titre :')
question = st.text_input('Entrer Question :')

if st.button('Suggestion des Tags'):
    if titre and question:
        # Appeler l'API pour obtenir les tags prédits
        api_url = "https://projet5-ff24767a8c7c.herokuapp.com/predict"
        payload = {'titre': titre, 'question': question}
        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            tags = response.json().get('tags_predits', [])
            st.write(f'Tags suggérés : {tags}')
        else:
            st.write('Erreur dans la prédiction.')
    else:
        st.write('Veuillez saisir un titre ET une question.')
