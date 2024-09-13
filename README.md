# Projet de Classification Automatique des Questions Stack Overflow

Ce projet vise à développer une application capable de suggérer des **tags pertinents** à des questions posées sur Stack Overflow en utilisant des modèles de machine learning. L'application a été développée en **Streamlit** et déployée sur **Render**. Elle utilise plusieurs approches, y compris des modèles de génération de topics (**LDA** et **NMF**) et des modèles supervisés pour prédire les tags des questions.

## Source des données

Les données utilisées pour entraîner les modèles proviennent de Stack Overflow. Un ensemble de 50 000 questions a été extrait via une requête SQL, en utilisant les critères suivants :

- Questions ayant reçu plus de **25 000 vues**.
- Questions ayant reçu au moins **une réponse**.
- Questions ayant au moins **5 tags**.
  
## Approches et algorithmes utilisés

### 1. **Modélisation Thématique (Non supervisée)**

- **BOW classique - LDA** : Extraction des thématiques à partir de Bag-of-Words (BOW) classique avec l'algorithme LDA.
- **BOW + TF-IDF - LDA** : Application de TF-IDF avant LDA pour améliorer la qualité des thématiques extraites.
- **BOW classique - NMF** : Utilisation de la Non-negative Matrix Factorization (NMF) pour réduire la dimensionnalité des données et extraire des thématiques.
- **BOW TF-IDF - NMF** : Application de TF-IDF avant l'utilisation de NMF pour identifier des thématiques pertinentes.

## Fonctionnalités

- **Saisie d'une question** : L'utilisateur peut entrer le titre et le corps de la question.
- **Suggestions de tags** : Une fois la question soumise, l'application retourne les tags suggérés basés sur le contenu.
- **Interface intuitive** : L'application est simple à utiliser grâce à Streamlit, avec un formulaire facile à comprendre.

## Déploiement

L'application est déployée sur **Render** et accessible via une URL publique.

## Installation locale

Si vous souhaitez exécuter cette application localement, voici les étapes à suivre :

### Prérequis

- **Python 3.8+**
- **Git**
- **pip** pour installer les dépendances

### Étapes d'installation

1. Installez les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

2. Exécutez l'application **Streamlit** :

   ```bash
   streamlit run ATFEH_Rana_5_code_final_092024.py
   ```

3. Accédez à l'application via l'URL fournie du navigateur.
