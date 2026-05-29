"""Test du modèle XGBoost sur des transactions réelles"""
import pandas as pd
import numpy as np
import joblib # type: ignore

print("📂 Chargement du modèle XGBoost...")
model = joblib.load('models/xgboost_model.pkl')

print("�� Chargement du dataset...")
df = pd.read_csv('data/raw/creditcard.csv')

# Features: TOUTES les colonnes sauf Time et Class (Amount est inclus)
# Le modèle XGBoost a été entraîné avec les colonnes V1 à V28 + Amount
feature_cols = [col for col in df.columns if col not in ['Time', 'Class']]
print(f"�� Nombre de features attendues: {len(feature_cols)}")

# Prendre 5 transactions légitimes et 5 frauduleuses
legit = df[df['Class'] == 0].head(5)
fraud = df[df['Class'] == 1].head(5)

print("\n" + "=" * 60)
print("✅ TRANSACTIONS LÉGITIMES (devraient être classées légitimes):")
print("=" * 60)

for i, row in legit.iterrows():
    features = row[feature_cols].values.reshape(1, -1)
    proba = model.predict_proba(features)[0][1]
    is_fraud = proba >= 0.5
    status = "❌ FRAUDE (ERREUR)" if is_fraud else "✅ LÉGITIME"
    print(f"   {status} | conf: {proba:.4f} | montant: {row['Amount']:.2f}€")

print("\n" + "=" * 60)
print("🚨 TRANSACTIONS FRAUDULEUSES (devraient être classées fraudes):")
print("=" * 60)

for i, row in fraud.iterrows():
    features = row[feature_cols].values.reshape(1, -1)
    proba = model.predict_proba(features)[0][1]
    is_fraud = proba >= 0.5
    status = "🚨 FRAUDE" if is_fraud else "✅ LÉGITIME (ERREUR)"
    print(f"   {status} | conf: {proba:.4f} | montant: {row['Amount']:.2f}€")
