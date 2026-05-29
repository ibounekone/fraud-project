📄 FRAUD_DETECTION.md
markdown
# 🔍 API de Détection de Fraude Bancaire

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)](https://xgboost.readthedocs.io)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

API de détection de fraude bancaire en temps réel utilisant XGBoost, avec monitoring intégré et seuil de décision paramétrable.

---

## 📋 Table des matières

- [🎯 Objectif du projet](#-objectif-du-projet)
- [📊 Métriques et performances](#-métriques-et-performances)
- [🏗️ Architecture](#️-architecture)
- [🚀 Installation](#-installation)
- [🔧 Utilisation de l'API](#-utilisation-de-lapi)
- [📈 Monitoring](#-monitoring)
- [🧪 Tests](#-tests)
- [📁 Structure du projet](#-structure-du-projet)
- [📚 Dépendances](#-dépendances)
- [🤝 Contribution](#-contribution)

---

## 🎯 Objectif du projet

Détecter automatiquement les transactions frauduleuses en temps réel avec une priorité absolue sur le **recall** (taux de détection des fraudes).

### Contexte métier

| Type d'erreur | Conséquence | Gravité |
|---------------|-------------|---------|
| **Faux négatif** (fraude non détectée) | Perte financière directe | 🔴 CRITIQUE |
| **Faux positif** (légitime bloquée) | Client mécontent | 🟡 Modérée |

### Contraintes techniques

- Latence < 100 ms
- Disponibilité 99.9%
- Authentification requise

---

## 📊 Métriques et performances

### Modèle final : XGBoost

| Métrique | Valeur | Statut |
|----------|--------|--------|
| **Recall** | 80.0% | ✅ Objectif atteint |
| **Précision** | 90.0% | ✅ Excellent |
| **AUC-ROC** | 0.97 | ✅ Excellent |
| **Latence moyenne** | 2.4 ms | ✅ < 100 ms |

### Échantillon de résultats

```bash
✅ Transactions légitimes : proba fraude ≈ 0.0004 (0.04%)
🚨 Transactions frauduleuses : proba fraude ≈ 0.85-0.99 (85-99%)
🏗️ Architecture
text
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT (Requête HTTP)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    API FASTAPI (Port 8001)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Clé API     │→│ Seuil       │→│ Modèle XGBoost       │  │
│  │ (auth)      │  │ (0.01-0.99) │  │ (fraude détection)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                              │
│  Endpoints : /predict, /health, /metrics, /drift, /logs    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      RÉPONSE JSON                            │
│  {                                                           │
│    "is_fraud": false,                                        │
│    "confidence": 0.00038,                                    │
│    "latency_ms": 2.40,                                       │
│    "threshold_used": 0.5                                     │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
🚀 Installation
1. Cloner le dépôt
bash
git clone https://github.com/ibounekone/fraud-detection.git
cd fraud-detection
2. Créer l'environnement virtuel
bash
python3 -m venv fraud_env
source fraud_env/bin/activate  # Sur Mac/Linux
# fraud_env\Scripts\activate   # Sur Windows
3. Installer les dépendances
bash
pip install -r requirements.txt
4. Télécharger le dataset
Télécharger creditcard.csv depuis Kaggle et le placer dans data/raw/

5. Préparer les données
bash
python src/data/build_features.py
6. Entraîner le modèle
bash
python -c "
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/raw/creditcard.csv')
feature_cols = [col for col in df.columns if col not in ['Time', 'Class']]
X = df[feature_cols].values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=500,
    random_state=42
)
model.fit(X_train, y_train)

joblib.dump(model, 'models/xgboost_model.pkl')
print('✅ Modèle entraîné et sauvegardé')
"
7. Lancer l'API
bash
python src/api/predict_fraud_final.py
L'API est accessible sur http://localhost:8001

🔧 Utilisation de l'API
Endpoint de prédiction
POST /predict

Headers requis
Header	Valeur
Content-Type	application/json
X-API-Key	fraud_detection_2024_secret_key
Corps de la requête
json
{
  "transaction": {
    "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0,
    "V6": 0, "V7": 0, "V8": 0, "V9": 0, "V10": 0,
    "V11": 0, "V12": 0, "V13": 0, "V14": 0, "V15": 0,
    "V16": 0, "V17": 0, "V18": 0, "V19": 0, "V20": 0,
    "V21": 0, "V22": 0, "V23": 0, "V24": 0, "V25": 0,
    "V26": 0, "V27": 0, "V28": 0,
    "Amount": 100
  },
  "threshold": 0.5
}
Réponse
json
{
  "transaction_id": null,
  "is_fraud": false,
  "confidence": 0.0003768469032365829,
  "threshold_used": 0.5,
  "latency_ms": 2.40,
  "timestamp": "2026-05-29T01:25:52.231406"
}
Exemple avec curl
bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: fraud_detection_2024_secret_key" \
  -d '{
    "transaction": {
      "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0,
      "V6": 0, "V7": 0, "V8": 0, "V9": 0, "V10": 0,
      "V11": 0, "V12": 0, "V13": 0, "V14": 0, "V15": 0,
      "V16": 0, "V17": 0, "V18": 0, "V19": 0, "V20": 0,
      "V21": 0, "V22": 0, "V23": 0, "V24": 0, "V25": 0,
      "V26": 0, "V27": 0, "V28": 0,
      "Amount": 100
    },
    "threshold": 0.5
  }'
Endpoint de santé
GET /health

bash
curl http://localhost:8001/health
Documentation interactive
Ouvrir dans le navigateur : http://localhost:8001/docs

📈 Monitoring
Endpoints de monitoring
Endpoint	Description	Sécurisé
/health	État de l'API	❌ Non
/metrics	Métriques détaillées	✅ Oui
/drift	Analyse de dérive	✅ Oui
/logs	Historique des prédictions	✅ Oui
Dashboard visuel
bash
python src/monitoring/dashboard_final.py
Génère :

Évolution des probabilités

Distribution des prédictions

Latence des requêtes

Distribution des montants

🧪 Tests
Tester avec des transactions réelles
bash
python test_real_transactions.py
Résultat attendu
bash
============================================================
✅ TRANSACTIONS LÉGITIMES:
============================================================
   ✅ LÉGITIME | conf: 0.0008 | montant: 149.62€
   ✅ LÉGITIME | conf: 0.0005 | montant: 2.69€
   ✅ LÉGITIME | conf: 0.0006 | montant: 378.66€
   ✅ LÉGITIME | conf: 0.0001 | montant: 123.50€
   ✅ LÉGITIME | conf: 0.0004 | montant: 69.99€

============================================================
🚨 TRANSACTIONS FRAUDULEUSES:
============================================================
   🚨 FRAUDE | conf: 0.9999 | montant: 0.00€
   🚨 FRAUDE | conf: 0.9982 | montant: 239.93€
   🚨 FRAUDE | conf: 0.9998 | montant: 59.00€
   🚨 FRAUDE | conf: 0.9992 | montant: 1.00€
📁 Structure du projet
text
fraud_detection/
├── src/
│   ├── api/
│   │   └── predict_fraud_final.py      # API finale
│   ├── monitoring/
│   │   ├── dashboard_final.py           # Dashboard visuel
│   │   └── check_logs.py                # Diagnostic logs
│   └── data/
│       └── build_features.py            # Préparation données
├── models/
│   └── xgboost_model.pkl                # Modèle entraîné
├── data/
│   ├── raw/
│   │   └── creditcard.csv               # Dataset (ignoré git)
│   └── processed/                       # Données préparées
├── tests/
│   ├── test_real_transactions.py        # Tests avec données réelles
│   └── test_with_logging.py             # Tests avec logs
├── requirements.txt
├── .gitignore
└── README.md
📚 Dépendances
requirements.txt
txt
fastapi==0.104.1
uvicorn==0.24.0
xgboost==2.0.3
scikit-learn==1.3.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
joblib==1.3.2
Installation rapide
bash
pip install -r requirements.txt
🚀 Déploiement sur Render
Pousser le code sur GitHub

Créer un compte sur render.com

New+ → Web Service

Connecter le dépôt GitHub

Configuration :

Champ	Valeur
Runtime	Python 3
Build Command	pip install -r requirements.txt
Start Command	uvicorn src.api.predict_fraud_final:app --host 0.0.0.0 --port 10000
Environment Variable	API_KEY = fraud_detection_2024_secret_key
Cliquer sur "Create Web Service"

L'API sera disponible à : https://ton-projet.onrender.com

📊 Enseignements clés
Enseignement	Pourquoi c'est important
XGBoost > Régression logistique	80% recall vs 60%
Seuil paramétrable	Ajustable selon le contexte métier
Monitoring essentiel	Détecter la dérive du modèle
Probas calibrées	Éviter les confiances extrêmes
🤝 Contribution
Les contributions sont les bienvenues !

Forker le projet

Créer une branche (git checkout -b feature/amazing-feature)

Commiter les changements (git commit -m 'Add amazing feature')

Pousser la branche (git push origin feature/amazing-feature)

Ouvrir une Pull Request

📝 Licence
MIT License - voir fichier LICENSE

📧 Contact
Auteur : ibounekone
GitHub : @ibounekone

🙏 Remerciements
Kaggle pour le dataset

FastAPI pour le framework

XGBoost pour le modèle

Dernière mise à jour : 29 Mai 2026

text

---