import streamlit as st
import requests

# Titre de l'interface Streamlit :
st.title('Classification de questions')

# Champ de saisie du titre :
titre = st.text_input('Entrer Titre :')

# Champ de saisie de la question :
question = st.text_input('Entrer Question :')

# Bouton de prédiction
if st.button('Suggestion des Tags'):
    if titre and question:
        # Appel à l'API Flask pour obtenir les tags prédits
        api_url = "https://projet-5-14613fa14eef.herokuapp.com/predict"  
        payload = {'titre': titre, 'question': question}
        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            tags = response.json().get('tags_predits', [])
            st.write(f'Tags suggérés : {tags}')
        else:
            st.write('Erreur dans la prédiction.')
    else:
        st.write('Veuillez saisir un titre ET une question.')
