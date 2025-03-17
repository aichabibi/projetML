# Analyse et Prédiction du Churn Client dans le secteur des Télécommunications

Une application Streamlit pour visualiser, analyser et prédire le churn client dans le secteur des télécommunications.

## À propos du projet

Ce projet propose une analyse complète du churn client à partir du jeu de données IBM Telco Customer Churn. Le churn client, défini comme la perte de clients, est un enjeu crucial dans le secteur des télécommunications qui connaît un taux annuel de churn de 15-25%.

L'application permet de :
- Visualiser les données et les facteurs influençant le churn
- Analyser les corrélations entre différentes variables
- Préparer les données pour la modélisation
- Construire et évaluer différents modèles de prédiction
- Proposer des recommandations personnalisées pour réduire le risque de churn

## Fonctionnalités

L'application est structurée en plusieurs sections :

1. **Aperçu des données**
   - Exploration des données brutes
   - Statistiques descriptives
   - Analyse des valeurs manquantes

2. **Analyse exploratoire**
   - Visualisations interactives avec Plotly
   - Distribution du churn selon différentes variables
   - Identification des facteurs de risque

3. **Traitement des données**
   - Encodage des variables catégorielles
   - Normalisation des variables numériques
   - Analyse des corrélations

4. **Modélisation et prédiction**
   - Implémentation de plusieurs algorithmes de classification
   - Évaluation des performances des modèles
   - Outil de prédiction interactive avec recommandations personnalisées

##  Installation et utilisation

### Prérequis

- Python 3.7+
- pip

### Installation

1. Clonez ce dépôt :
```bash
git clone [https://github.com/votreUsername/analyse-churn-client.git](https://github.com/aichabibi/projetML)
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Téléchargez le jeu de données :
   - Téléchargez le fichier [WA_Fn-UseC_-Telco-Customer-Churn.csv](https://www.kaggle.com/blastchar/telco-customer-churn) depuis Kaggle
   - Placez-le dans le répertoire racine du projet

4. Lancez l'application :
```bash
streamlit run app.py
```

## Jeu de données

Le jeu de données IBM Telco Customer Churn contient des informations sur :
- Services souscrits (téléphone, internet, sécurité, etc.)
- Informations du compte (durée, contrat, paiement)
- Données démographiques (genre, âge, etc.)
- Comportement de churn (clients ayant quitté l'entreprise)

## Technologies utilisées

- **Streamlit** : Interface utilisateur interactive
- **Pandas** & **NumPy** : Manipulation et traitement des données
- **Matplotlib**, **Seaborn** & **Plotly** : Visualisations
- **Scikit-learn** : Modélisation et évaluation
- **Diverses bibliothèques ML** : Implémentation d'algorithmes de classification

## Principaux résultats

L'analyse des données révèle plusieurs facteurs importants influençant le churn client :
- Les clients avec contrat mensuel sont plus susceptibles de partir (75% de churn)
- Les clients utilisant le paiement par chèque électronique ont un taux de churn plus élevé
- Les clients avec service internet en fibre optique présentent un taux de churn important
- L'ancienneté du client est inversement corrélée au risque de churn
- Les services additionnels (sécurité, support technique) réduisent le risque de churn

## Modèles de prédiction

L'application implémente plusieurs algorithmes de classification :
- Régression Logistique
- Random Forest
- Gradient Boosting
- AdaBoost
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Voting Classifier (ensemble)

Les performances des modèles sont évaluées à l'aide de métriques standard (précision, rappel, F1-score) et de visualisations (matrice de confusion, courbe ROC).

##  Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou soumettre une pull request.

## Contact

Pour toute question ou suggestion, n'hésitez pas à me contacter à [aicha.bibi@ynov.com].
