import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import shap
import matplotlib.pyplot as plt

# Charger le modèle
model = load_model("models/credit_model.h5")

# Charger les données pour le scaler
data = pd.read_csv("data/clean_data.csv")
X = data.drop(columns=["Credit amount"])
scaler = StandardScaler()
scaler.fit(X)

st.title("Prédicteur de Risque de Crédit")

# Saisie utilisateur
age = st.number_input("Âge", min_value=18, max_value=100)
job = st.selectbox("nombre d'emplois", options=[0, 1, 2, 3])
housing = st.selectbox("Logement", options=["own", "free", "rent"])
saving_accounts = st.selectbox("Compte épargne", options=["little", "moderate", "quite rich", "rich", "nan"])
checking_account = st.selectbox("Compte courant", options=["little", "moderate", "rich", "nan"])
duration = st.slider("Durée du crédit (mois)", 4, 72)
purpose = st.selectbox("Objet du crédit", options=["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"])

# Préparer les données en dataframe
input_df = pd.DataFrame([{
    "Age": age,
    "Job": job,
    "Housing": housing,
    "Saving accounts": saving_accounts,
    "Checking account": checking_account,
    "Duration": duration,
    "Purpose": purpose
}])

# Encodage simple des variables catégorielles
cat_cols = ['Housing', 'Saving accounts', 'Checking account', 'Purpose']
for col in cat_cols:
    input_df[col] = input_df[col].astype("category").cat.codes

# Ajouter la colonne 'Sex' avec valeur fixe (à adapter selon ton modèle)
input_df["Sex"] = 1

# Réordonner colonnes comme dans l’entraînement
input_df = input_df[X.columns]

# Standardiser
input_scaled = scaler.transform(input_df)

# Prédiction
proba = model.predict(input_scaled)[0][0]
prediction = "Crédit accordé ✅" if proba < 0.5 else "Crédit refusé ❌"

st.subheader("Résultat de la prédiction")
st.write(prediction)
st.write(f"Probabilité de risque élevé : **{proba:.2f}**")

# Explicabilité avec SHAP
explainer = shap.Explainer(model, X[:100])
shap_values = explainer(input_scaled)

st.subheader("Variables les plus influentes")

# Afficher un graphique SHAP waterfall
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], max_display=3, show=False)
st.pyplot(fig)
