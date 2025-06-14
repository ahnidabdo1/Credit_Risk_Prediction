# 🤖 Credit Risk Prediction App

This project uses a **Neural Network (ANN)** to predict whether a client is likely to default on a loan, based on their profile and loan details. The app is built using **Streamlit** and explains predictions using **SHAP** for interpretability.

---

## 🚀 Features

- ✅ Predict credit risk (loan approval or refusal)
- 📊 Visualize key client insights (SHAP waterfall plot)
- 🧠 Built with Keras (ANN) and Scikit-learn
- 📉 Handles missing data, categorical encoding, and feature scaling
- 🧾 SHAP-based interpretability for transparency

---

## 📁 Project Structure

```bash
📦credit-risk-app/
│
├── models/                   # Saved trained model (.h5)
├── data/                     # Clean dataset CSV
├── app.py                    # Streamlit web application
├── pred_risque_credit.ipynb  # Model training and evaluation
├── requirements.txt          # Dependencies
└── README.md                 # This file
