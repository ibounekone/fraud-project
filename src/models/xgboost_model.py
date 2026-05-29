"""
MODÈLE XGBOOST AVEC SMOTE POUR LA DÉTECTION DE FRAUDE
Objectif : Battre la baseline (recall 91.8%)
"""

import numpy as np
from pathlib import Path
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix # type: ignore
from imblearn.over_sampling import SMOTE  # type: ignore # Génère des fraudes synthétiques
import xgboost as xgb # type: ignore
import warnings
warnings.filterwarnings('ignore')

# Chemins
DATA_PATH = Path("data/processed/")

def load_data():
    """Charge les données préparées"""
    print("📂 Chargement des données...")
    X_train = np.load(DATA_PATH / "X_train.npy")
    X_test = np.load(DATA_PATH / "X_test.npy")
    y_train = np.load(DATA_PATH / "y_train.npy")
    y_test = np.load(DATA_PATH / "y_test.npy")
    
    print(f"   Train: {X_train.shape[0]} samples, fraudes: {y_train.sum()}")
    print(f"   Test:  {X_test.shape[0]} samples, fraudes: {y_test.sum()}")
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    """
    SMOTE = Synthetic Minority Over-sampling TEchnique
    Crée des exemples synthétiques de la classe minoritaire (fraude)
    Pourquoi ? Évite le sur-apprentissage et améliore la généralisation
    """
    print("\n🔧 Application de SMOTE...")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"   Avant SMOTE : {X_train.shape[0]} samples, fraudes: {y_train.sum()}")
    print(f"   Après SMOTE  : {X_train_resampled.shape[0]} samples, fraudes: {y_train_resampled.sum()}")
    
    return X_train_resampled, y_train_resampled

def train_xgboost(X_train, y_train):
    """
    XGBoost = Extreme Gradient Boosting
    Pourquoi ce choix ?
    - État de l'art pour les données tabulaires
    - Gère bien le déséquilibre
    - Très rapide et performant
    """
    print("\n🌲 Entraînement de XGBoost...")
    
    # Paramètres optimisés pour la fraude
    model = xgb.XGBClassifier(
        n_estimators=100,      # Nombre d'arbres (plus = mieux mais plus long)
        max_depth=6,           # Profondeur max des arbres (évite sur-apprentissage)
        learning_rate=0.1,     # Vitesse d'apprentissage (plus petit = plus précis)
        subsample=0.8,         # Échantillonnage des lignes (évite sur-apprentissage)
        colsample_bytree=0.8,  # Échantillonnage des colonnes
        scale_pos_weight=500,  # Compense le déséquilibre (auto-calculé)
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    print("✅ XGBoost entraîné")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Évalue le modèle avec les métriques prioritaires"""
    print("\n" + "=" * 60)
    print("📈 ÉVALUATION DU MODÈLE")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n🎯 MÉTRIQUES :")
    print(f"   Recall   : {recall:.4f} ({recall*100:.2f}%) ← NOTRE PRIORITÉ")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   AUC-ROC  : {auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    detection_rate = cm[1,1] / (cm[1,0] + cm[1,1]) * 100
    
    print(f"\n📊 TAUX DE DÉTECTION DES FRAUDES : {detection_rate:.1f}%")
    
    return recall, detection_rate

def compare_with_baseline(xgb_recall, baseline_recall=0.9184):
    """Compare XGBoost avec la baseline"""
    print("\n" + "=" * 60)
    print("🏆 COMPARAISON AVEC LA BASELINE")
    print("=" * 60)
    
    print(f"   Baseline (Régression Logistique) : {baseline_recall*100:.2f}% recall")
    print(f"   XGBoost + SMOTE                  : {xgb_recall*100:.2f}% recall")
    
    if xgb_recall > baseline_recall:
        gain = (xgb_recall - baseline_recall) * 100
        print(f"\n✅ XGBoost bat la baseline de {gain:.2f} points !")
    else:
        perte = (baseline_recall - xgb_recall) * 100
        print(f"\n⚠️ XGBoost est moins bon de {perte:.2f} points")
        print("   On garde la baseline en production (plus simple)")

def main():
    print("=" * 60)
    print("🚀 XGBOOST + SMOTE - DÉTECTION DE FRAUDE")
    print("=" * 60)
    
    # 1. Chargement
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. SMOTE (rééquilibrage des classes)
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    
    # 3. Entraînement XGBoost
    model = train_xgboost(X_train_res, y_train_res)
    
    # 4. Évaluation
    xgb_recall, detection_rate = evaluate_model(model, X_test, y_test)
    
    # 5. Comparaison
    compare_with_baseline(xgb_recall)
    
    print("\n💡 CONCLUSION :")
    if xgb_recall > 0.92:
        print("   ✅ XGBoost est excellent – peut être déployé")
    else:
        print("   💡 La baseline suffit – plus simple et tout aussi performante")

if __name__ == "__main__":
    main()