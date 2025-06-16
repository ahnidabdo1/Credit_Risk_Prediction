# 🤖 Credit Risk Prediction App

This project uses a **Neural Network (ANN)** to predict whether a client is likely to **default on a loan**, based on their profile and loan details. The app is built with **Streamlit** and incorporates **SHAP** for explainability.

🔗 **Live App**: [Streamlit App](https://abdelouahed-ahnid-credit-risk-prediction.streamlit.app/)

---

## 🚀 Features

- ✅ **Credit Risk Prediction** – Approve or refuse loan applications
- 📊 **Explainable AI** – SHAP waterfall plots to visualize prediction logic
- 🧠 **Machine Learning Stack** – Built with Keras (ANN) and Scikit-learn
- 🔧 **Robust Preprocessing** – Handles missing data, encodes categories, and scales features
- 🧾 **Transparent AI** – Model interpretability with SHAP values

---

## 📁 Project Structure

```bash
📦 credit-risk-app/
│
├── models/                     # Trained ANN model (.h5 format)
├── data/                       # Cleaned dataset (CSV format)
├── app.py                      # Streamlit web application
├── pred_risque_credit.ipynb    # Jupyter notebook: training + evaluation
├── requirements.txt            # List of required libraries
└── README.md                   # Project documentation (this file)
