"""Réentraînement du modèle pour obtenir des probabilités calibrées"""
import numpy as np
import joblib # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.calibration import CalibratedClassifierCV # type: ignore
from pathlib import Path

# Charger les données
print("📂 Chargement des données...")
X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

print(f"Train: {X_train.shape}, fraudes: {y_train.sum()}")
print(f"Test: {X_test.shape}, fraudes: {y_test.sum()}")

# Modèle de base avec régularisation forte
print("\n🔧 Entraînement du modèle calibré...")

base_model = LogisticRegression(
    class_weight='balanced',
    C=0.01,
    max_iter=2000,
    random_state=42,
    solver='liblinear'
)

base_model.fit(X_train, y_train)

# Calibration des probabilités
print("📊 Calibration des probabilités...")
calibrated_model = CalibratedClassifierCV(
    base_model, 
    method='sigmoid',
    cv=5
)
calibrated_model.fit(X_train, y_train)

# Évaluation
from sklearn.metrics import recall_score, precision_score, roc_auc_score # type: ignore

y_pred = calibrated_model.predict(X_test)
y_proba = calibrated_model.predict_proba(X_test)[:, 1]

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 50)
print("📊 RÉSULTATS DU MODÈLE CALIBRÉ")
print("=" * 50)
print(f"Recall: {recall:.4f} ({recall*100:.1f}%)")
print(f"Précision: {precision:.4f} ({precision*100:.1f}%)")
print(f"AUC: {auc:.4f}")

# Vérifier les probabilités
print("\n🔍 Vérification des probabilités:")
legit_samples = X_test[y_test == 0][:5]
fraud_samples = X_test[y_test == 1][:5]

for i, sample in enumerate(legit_samples):
    prob = calibrated_model.predict_proba(sample.reshape(1, -1))[0][1]
    print(f"   Transaction légitime {i+1}: proba fraude = {prob:.4f}")

for i, sample in enumerate(fraud_samples):
    prob = calibrated_model.predict_proba(sample.reshape(1, -1))[0][1]
    print(f"   Transaction frauduleuse {i+1}: proba fraude = {prob:.4f}")

# Sauvegarde
Path("models").mkdir(exist_ok=True)
joblib.dump(calibrated_model, "models/logistic_regression_calibrated.pkl")
print("\n💾 Modèle calibré sauvegardé: models/logistic_regression_calibrated.pkl")
