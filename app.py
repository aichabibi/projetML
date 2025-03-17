import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io 
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report

# Configuration de la page
st.set_page_config(page_title="Analyse du Churn Client", layout="wide")

# Titre principal
st.title("Analyse et Prédiction du Churn Client")

# Introduction
st.markdown("""
## Qu'est-ce que le Churn Client ?
Le churn client est défini comme le fait qu'un client ou un abonné cesse de faire affaire avec une entreprise ou un service.

Dans l'industrie des télécommunications, les clients ont la possibilité de choisir parmi plusieurs fournisseurs de services et peuvent facilement passer de l'un à l'autre. Ce secteur, très concurrentiel, affiche un taux annuel de churn de 15 à 25 %.

La fidélisation individualisée des clients est un défi, car la plupart des entreprises ont un grand nombre de clients et ne peuvent pas se permettre de consacrer du temps à chacun d'entre eux. Les coûts seraient trop élevés et dépasseraient les revenus supplémentaires générés. Cependant, si une entreprise pouvait prédire à l'avance quels clients sont susceptibles de partir, elle pourrait concentrer ses efforts de rétention uniquement sur ces clients "à risque".
""")

# Chargement des données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        return df
    except FileNotFoundError:
        st.error("Fichier de données introuvable. Veuillez télécharger le fichier 'WA_Fn-UseC_-Telco-Customer-Churn.csv'.")
        return None

# Sidebar pour navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sélectionnez une page:", 
                        ["Aperçu des données", 
                         "Analyse exploratoire", 
                         "Traitement des données",
                         "Modélisation et prédiction"])

# Charger les données
df = load_data()

if df is not None:
    # Supprimer customerID car inutile pour l'analyse
    df_clean = df.copy()
    df_clean = df_clean.drop(['customerID'], axis=1)
    
    # Convertir TotalCharges en numérique
    df_clean['TotalCharges'] = pd.to_numeric(df_clean.TotalCharges, errors='coerce')
    
    # Traitement des valeurs manquantes (tenure = 0)
    df_clean = df_clean[df_clean['tenure'] != 0]
    
    # Remplacer les valeurs manquantes dans TotalCharges par la moyenne
    df_clean = df_clean.fillna(df_clean["TotalCharges"].mean())
    
    # Convertir SeniorCitizen en "Yes" ou "No"
    df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].map({0: "No", 1: "Yes"})
    
    # Page 1: Aperçu des données
    if page == "Aperçu des données":
        st.header("Aperçu des données")
        
        # Afficher les premières lignes
        st.subheader("Aperçu du jeu de données")
        st.write(df_clean.head())
        
        # Informations sur les données
        st.subheader("Structure du jeu de données")
        buffer = io.StringIO()
        df_clean.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        # Description des colonnes
        st.subheader("Description des colonnes")
        st.markdown("""
        - **Churn**: Les clients qui ont quitté le service au cours du dernier mois
        - **Services**: Téléphone, plusieurs lignes, internet, sécurité en ligne, sauvegarde en ligne, protection des appareils, support technique, streaming TV et films
        - **Informations du compte**: Durée de fidélité, type de contrat, mode de paiement, facturation sans papier, frais mensuels et frais totaux
        - **Informations démographiques**: Sexe, tranche d'âge, et s'ils ont des partenaires ou des personnes à charge
        """)
        
        # Statistiques descriptives
        st.subheader("Statistiques descriptives")
        st.write(df_clean.describe())
        
        # Vérification des valeurs manquantes
        st.subheader("Valeurs manquantes")
        st.write(df_clean.isnull().sum())
        
    # Page 2: Analyse exploratoire
    elif page == "Analyse exploratoire":
        st.header("Analyse exploratoire des données")
        
        # Distribution du churn
        st.subheader("Distribution du churn et du genre")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Création du graphique pour le churn avec Plotly
            fig_churn = px.pie(df_clean, names='Churn', title='Distribution du Churn')
            st.plotly_chart(fig_churn)
        
        with col2:
            # Création du graphique pour le genre avec Plotly
            fig_gender = px.pie(df_clean, names='gender', title='Distribution du Genre')
            st.plotly_chart(fig_gender)
        
        # Churn par contrat
        st.subheader("Distribution du churn par type de contrat")
        fig_contract = px.histogram(df_clean, x="Churn", color="Contract", barmode="group", 
                                    title="Distribution des contrats clients")
        st.plotly_chart(fig_contract)
        
        st.markdown("""
        **Observation:** Environ 75% des clients avec un contrat mensuel ont choisi de partir, contre 13% des clients avec un contrat d'un an et 3% avec un contrat de deux ans.
        """)
        
        # Churn par méthode de paiement
        st.subheader("Distribution du churn par méthode de paiement")
        fig_payment = px.histogram(df_clean, x="Churn", color="PaymentMethod", 
                                   title="Distribution des méthodes de paiement par rapport au churn")
        st.plotly_chart(fig_payment)
        
        st.markdown("""
        **Observation:** Les principaux clients qui sont partis utilisaient **Electronic Check** comme méthode de paiement.
        Les clients ayant opté pour **le transfert automatique par carte de crédit**, **le transfert automatique bancaire** 
        ou **le chèque postal** comme méthode de paiement étaient **moins susceptibles de partir**.
        """)
        
        # Churn par service Internet
        st.subheader("Distribution du churn par service Internet")
        internet_service_data = df_clean.groupby(['InternetService', 'Churn']).size().reset_index(name='count')
        fig_internet = px.bar(internet_service_data, x='InternetService', y='count', color='Churn', 
                             title="Distribution du service Internet par rapport au churn")
        st.plotly_chart(fig_internet)
        
        st.markdown("""
        **Observation:** Beaucoup de clients choisissent le service Fiber optic et il est également évident que les clients utilisant 
        la fibre optique ont un taux de churn élevé, ce qui pourrait suggérer une insatisfaction vis-à-vis de ce type de service Internet.
        """)
        
        # Churn par facturation sans papier
        st.subheader("Distribution du churn par facturation sans papier")
        fig_paperless = px.histogram(df_clean, x="Churn", color="PaperlessBilling", 
                                     title="Distribution de la facturation sans papier par rapport au churn")
        st.plotly_chart(fig_paperless)
        
        st.markdown("""
        **Observation:** Les clients avec **facturation sans papier** sont **plus susceptibles de partir** (churn).
        """)
        
        # Churn par ancienneté (tenure)
        st.subheader("Distribution du churn par ancienneté (tenure)")
        fig_tenure = px.box(df_clean, x='Churn', y='tenure', title='Ancienneté vs Churn')
        st.plotly_chart(fig_tenure)
        
        st.markdown("""
        **Observation:** Les **nouveaux clients** (avec une ancienneté plus faible) sont **plus susceptibles de partir** (churn).
        """)
        
    # Page 3: Traitement des données
    elif page == "Traitement des données":
        st.header("Traitement des données")
        
        # Encodage des variables catégorielles
        st.subheader("Encodage des variables catégorielles")
        
        if st.checkbox("Encoder les variables catégorielles"):
            # Fonction pour encoder les variables catégorielles
            def object_to_int(dataframe_series):
                if dataframe_series.dtype=='object':
                    dataframe_series = LabelEncoder().fit_transform(dataframe_series)
                return dataframe_series
            
            df_encoded = df_clean.apply(lambda x: object_to_int(x))
            st.write("Aperçu des données encodées:")
            st.write(df_encoded.head())
            
            # Afficher les corrélations
            st.subheader("Corrélations avec le Churn")
            corr_with_churn = df_encoded.corr()['Churn'].sort_values(ascending=False)
            st.write(corr_with_churn)
            
            # Visualisation des corrélations
            st.subheader("Matrice de corrélation")
            fig, ax = plt.subplots(figsize=(12, 8))
            corr_matrix = df_encoded.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)
            
        # Normalisation des variables numériques
        st.subheader("Normalisation des variables numériques")
        
        if st.checkbox("Normaliser les variables numériques"):
            num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
            
            # Avant normalisation
            st.write("Distribution avant normalisation:")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i, col in enumerate(num_cols):
                sns.histplot(df_clean[col], ax=axes[i], kde=True)
                axes[i].set_title(f"Distribution de {col}")
            st.pyplot(fig)
            
            # Après normalisation
            st.write("Distribution après normalisation:")
            scaler = StandardScaler()
            df_std = pd.DataFrame(scaler.fit_transform(df_clean[num_cols].astype('float64')),
                                  columns=num_cols)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i, col in enumerate(num_cols):
                sns.histplot(df_std[col], ax=axes[i], kde=True)
                axes[i].set_title(f"Distribution normalisée de {col}")
            st.pyplot(fig)
            
    # Page 4: Modélisation et prédiction
    elif page == "Modélisation et prédiction":
        st.header("Modélisation et prédiction du churn")
        
        # Encodage des données pour la modélisation
        def prepare_data(df):
            # Encodage des variables catégorielles
            df_model = df.copy()
            
            # Encodage des variables catégorielles
            for col in df_model.columns:
                if df_model[col].dtype == 'object':
                    df_model[col] = LabelEncoder().fit_transform(df_model[col])
            
            # Séparation des features et de la target
            X = df_model.drop(columns=['Churn'])
            y = df_model['Churn'].values
            
            # Séparation train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40, stratify=y)
            
            # Normalisation des features numériques
            num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
            
            return X_train, X_test, y_train, y_test
        
        X_train, X_test, y_train, y_test = prepare_data(df_clean)
        
        # Choix du modèle
        st.subheader("Sélection du modèle")
        model_option = st.selectbox(
            "Choisissez un modèle de classification:",
            ["Logistic Regression", "Random Forest", "Gradient Boosting", "AdaBoost", "KNN", "SVM", "Voting Classifier"]
        )
        
        # Entraînement et évaluation du modèle
        if st.button("Entraîner et évaluer le modèle"):
            with st.spinner("Entraînement du modèle en cours..."):
                if model_option == "Logistic Regression":
                    model = LogisticRegression(random_state=42)
                    model_name = "Régression Logistique"
                elif model_option == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model_name = "Random Forest"
                elif model_option == "Gradient Boosting":
                    model = GradientBoostingClassifier(random_state=42)
                    model_name = "Gradient Boosting"
                elif model_option == "AdaBoost":
                    model = AdaBoostClassifier(random_state=42)
                    model_name = "AdaBoost"
                elif model_option == "KNN":
                    model = KNeighborsClassifier(n_neighbors=11)
                    model_name = "K-Nearest Neighbors"
                elif model_option == "SVM":
                    model = SVC(probability=True, random_state=42)
                    model_name = "Support Vector Machine"
                elif model_option == "Voting Classifier":
                    clf1 = GradientBoostingClassifier(random_state=42)
                    clf2 = LogisticRegression(random_state=42)
                    clf3 = AdaBoostClassifier(random_state=42)
                    model = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft')
                    model_name = "Voting Classifier"
                
                # Entraînement du modèle
                model.fit(X_train, y_train)
                
                # Prédictions
                y_pred = model.predict(X_test)
                
                # Évaluation
                accuracy = accuracy_score(y_test, y_pred)
                
                # Affichage des résultats
                st.success(f"Modèle entraîné avec succès! Précision: {accuracy:.4f}")
                
                # Classification report
                st.subheader("Rapport de classification")
                report = classification_report(y_test, y_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.write(df_report)
                
                # Confusion Matrix
                st.subheader("Matrice de confusion")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cmap="Blues", ax=ax)
                plt.ylabel('Valeur réelle')
                plt.xlabel('Valeur prédite')
                plt.title(f'Matrice de confusion - {model_name}')
                st.pyplot(fig)
                
                # ROC Curve if model supports predict_proba
                try:
                    y_pred_prob = model.predict_proba(X_test)[:,1]
                    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
                    
                    st.subheader("Courbe ROC")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(fpr, tpr, label=f'{model_name}')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('Taux de faux positifs')
                    ax.set_ylabel('Taux de vrais positifs')
                    ax.set_title(f'Courbe ROC - {model_name}')
                    ax.legend()
                    st.pyplot(fig)
                except:
                    st.info("Ce modèle ne supporte pas le calcul de probabilités pour la courbe ROC.")
        
        # Section de prédiction interactive
        st.subheader("Prédiction interactive du churn")
        st.write("Utilisez les contrôles ci-dessous pour prédire si un client est susceptible de quitter le service.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Genre", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        with col2:
            tenure = st.slider("Tenure (mois)", 1, 72, 36)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        with col3:
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        with col5:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        with col6:
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 118.0, 70.0)
            # Total charges can be estimated as tenure * monthly_charges for simplicity
            total_charges = tenure * monthly_charges
        
        # Préparation des données pour la prédiction
        if st.button("Prédire le churn"):
            # Créer un dictionnaire avec les valeurs saisies
            input_data = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Créer un DataFrame à partir du dictionnaire
            input_df = pd.DataFrame([input_data])
            
            # Préparer le DataFrame pour la prédiction (encodage et mise à l'échelle)
            # On utilise le même pipeline que pour les données d'entraînement
            
            # Encodage des variables catégorielles
            le_dict = {}
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    le = LabelEncoder()
                    le.fit(df_clean[col])
                    le_dict[col] = le
            
            # Appliquer l'encodage
            input_df_encoded = input_df.copy()
            for col in input_df.columns:
                if col in le_dict:
                    try:
                        input_df_encoded[col] = le_dict[col].transform([input_df[col].iloc[0]])[0]
                    except:
                        st.error(f"Erreur d'encodage pour la colonne {col}. Valeur: {input_df[col].iloc[0]}")
                        continue
            
            # Mise à l'échelle des variables numériques
            num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
            scaler = StandardScaler()
            scaler.fit(df_clean[num_cols])
            input_df_encoded[num_cols] = scaler.transform(input_df_encoded[num_cols])
            
            # Faire la prédiction avec le modèle sélectionné
            try:
                if model_option == "Logistic Regression":
                    model = LogisticRegression(random_state=42)
                elif model_option == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_option == "Gradient Boosting":
                    model = GradientBoostingClassifier(random_state=42)
                elif model_option == "AdaBoost":
                    model = AdaBoostClassifier(random_state=42)
                elif model_option == "KNN":
                    model = KNeighborsClassifier(n_neighbors=11)
                elif model_option == "SVM":
                    model = SVC(probability=True, random_state=42)
                elif model_option == "Voting Classifier":
                    clf1 = GradientBoostingClassifier(random_state=42)
                    clf2 = LogisticRegression(random_state=42)
                    clf3 = AdaBoostClassifier(random_state=42)
                    model = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft')
                
                # Entraînement du modèle
                model.fit(X_train, y_train)
                
                # Prédiction
                prediction = model.predict(input_df_encoded)[0]
                
                # Probabilité si disponible
                try:
                    proba = model.predict_proba(input_df_encoded)[0, 1]
                    
                    # Affichage du résultat
                    if prediction == 1:
                        st.error(f"⚠️ Risque élevé de churn (Probabilité: {proba:.2%})")
                        st.markdown("""
                        ### Recommandations pour réduire le risque de churn:
                        
                        1. **Offrir une promotion ou une remise** sur les services actuels
                        2. **Améliorer le support client** pour ce client
                        3. **Proposer un contrat plus long** avec des avantages supplémentaires
                        4. **Contacter le client** pour évaluer sa satisfaction
                        """)
                    else:
                        st.success(f"✅ Client fidèle (Probabilité de rester: {1-proba:.2%})")
                        st.markdown("""
                        ### Recommandations pour maintenir la fidélité:
                        
                        1. **Programme de récompense** pour la fidélité client
                        2. **Proposer des services complémentaires** adaptés à son profil
                        3. **Communication régulière** sur les nouveaux services
                        """)
                except:
                    # Si le modèle ne supporte pas predict_proba
                    if prediction == 1:
                        st.error("⚠️ Risque élevé de churn")
                    else:
                        st.success("✅ Client fidèle")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")

    # Informations supplémentaires
    st.sidebar.header("À propos")
    st.sidebar.info("""
    Cette application analyse le churn client dans une entreprise de télécommunications.
    
    **Données:** IBM Sample Dataset - Telco Customer Churn
    
    **Autrice:** Aicha BIBI 
    """)
                
    # Import manquant pour le message d'erreur FileNotFoundError
else:
    import io
    st.warning("Veuillez télécharger le fichier de données pour continuer.")
    
    # Option pour télécharger un exemple de fichier
    st.info("""
    Vous pouvez télécharger le jeu de données ici:
    [IBM Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
    """)