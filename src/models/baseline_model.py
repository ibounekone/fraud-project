"""
BASELINE POUR LA DÉTECTION DE FRAUDE BANCAIRE
Objectif : Établir une référence minimale avant d'utiliser des modèles complexes
La baseline doit être simple, rapide, et interprétable
"""

# ========== IMPORTS ==========
import numpy as np  # Calculs numériques (moyennes, arrays)
import pandas as pd  # Manipulation des données (DataFrame)
from pathlib import Path  # Gestion portable des chemins de fichiers
from sklearn.linear_model import LogisticRegression  # type: ignore # Modèle simple de classification
from sklearn.metrics import ( # type: ignore
    accuracy_score,        # Pourcentage de bonnes prédictions (à ne pas utiliser seul)
    precision_score,      # Exactitude des alertes : VP / (VP + FP)
    recall_score,         # Détection des fraudes : VP / (VP + FN) → NOTRE PRIORITÉ
    f1_score,            # Moyenne harmonique entre précision et rappel
    roc_auc_score,       # Capacité à distinguer les classes (1 = parfait)
    confusion_matrix,    # Matrice : Vrais Pos, Faux Pos, Vrais Neg, Faux Neg
    classification_report # Rapport détaillé des métriques
)
import warnings
warnings.filterwarnings('ignore')  # Supprime les avertissements (plus lisible)

# ========== CHEMINS ==========
# Path gère les chemins de façon compatible Windows/Mac/Linux
DATA_PATH = Path("data/processed/")

# ========== FONCTION 1 : CHARGEMENT ==========
def load_data():
    """
    Charge les données préparées (fichiers .npy)
    Les .npy sont plus rapides et plus légers que les CSV
    """
    print("📂 Chargement des données préparées...")
    
    # np.load charge un fichier .npy (format binaire NumPy)
    X_train = np.load(DATA_PATH / "X_train.npy")
    X_test = np.load(DATA_PATH / "X_test.npy")
    y_train = np.load(DATA_PATH / "y_train.npy")
    y_test = np.load(DATA_PATH / "y_test.npy")
    
    print(f"✅ Train : {X_train.shape[0]} transactions")
    print(f"✅ Test  : {X_test.shape[0]} transactions")
    print(f"⚠️ Fraudes dans le test : {y_test.sum()} ({y_test.mean()*100:.4f}%)")
    
    return X_train, X_test, y_train, y_test

# ========== FONCTION 2 : BASELINE ==========
def train_baseline(X_train, y_train):
    """
    Entraîne un modèle baseline très simple : Régression Logistique
    Pourquoi ce choix ?
    1. C'est le modèle le plus simple pour la classification binaire
    2. Très rapide à entraîner (< 1 seconde sur 200k lignes)
    3. Interprétable (on peut voir le poids de chaque feature)
    4. Sert de référence : tout modèle complexe DOIT battre ce score
    """
    print("\n" + "=" * 60)
    print("📊 ENTRAÎNEMENT DE LA BASELINE (Régression Logistique)")
    print("=" * 60)
    
    # Création du modèle
    # class_weight='balanced' : Compense le déséquilibre des classes
    # Sans cela, le modèle ignorerait les fraudes (trop rares)
    # max_iter=1000 : Permet au modèle de converger (plus d'itérations)
    # random_state=42 : Fixe l'aléatoire pour résultats reproductibles
    model = LogisticRegression(
        class_weight='balanced',  # Donne plus de poids aux fraudes
        max_iter=1000,            # Suffisant pour convergence
        random_state=42,          # Reproductibilité
        C=1.0                     # Force de régularisation (1 = défaut)
    )
    
    print("🔧 Entraînement en cours...")
    model.fit(X_train, y_train)  # Apprentissage sur les données d'entraînement
    print("✅ Modèle entraîné avec succès")
    
    return model

# ========== FONCTION 3 : ÉVALUATION ==========
def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle avec TOUTES les métriques pertinentes
    Pour la fraude, le RAPPEL est le plus important
    """
    print("\n" + "=" * 60)
    print("📈 ÉVALUATION DU MODÈLE")
    print("=" * 60)
    
    # Prédiction des classes (0 = légitime, 1 = fraude)
    # Par défaut, seuil = 0.5 (si proba > 0.5 → frauduleux)
    y_pred = model.predict(X_test)
    
    # Probabilité d'être une fraude (entre 0 et 1)
    # Utile pour calculer l'AUC et ajuster le seuil
    y_proba = model.predict_proba(X_test)[:, 1]  # [:, 1] = probabilité classe 1 (fraude)
    
    # ----- MÉTRIQUES -----
    # accuracy = (VP + VN) / total
    # Problème : avec 99.83% de légitimes, prédire "tout légitime" donne 99.83%
    accuracy = accuracy_score(y_test, y_pred)
    
    # precision = VP / (VP + FP)
    # Sur toutes les alertes fraudes, combien sont vraiment fraudes ?
    precision = precision_score(y_test, y_pred)
    
    # recall = VP / (VP + FN) → NOTRE PRIORITÉ
    # Sur toutes les vraies fraudes, combien sont détectées ?
    recall = recall_score(y_test, y_pred)
    
    # f1 = moyenne harmonique entre précision et rappel
    f1 = f1_score(y_test, y_pred)
    
    # AUC = capacité à distinguer les deux classes
    # AUC = 0.5 → aléatoire, AUC = 1.0 → parfait
    auc = roc_auc_score(y_test, y_proba)
    
    # Matrice de confusion : [ [VN, FP],
    #                         [FN, VP] ]
    cm = confusion_matrix(y_test, y_pred)
    
    # ----- AFFICHAGE -----
    print(f"\n🎯 MÉTRIQUES GLOBALES :")
    print(f"   Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall   : {recall:.4f} ({recall*100:.2f}%) ← NOTRE PRIORITÉ")
    print(f"   F1-Score : {f1:.4f}")
    print(f"   AUC-ROC  : {auc:.4f}")
    
    print(f"\n📊 MATRICE DE CONFUSION :")
    print(f"                 Prédit Légitime   Prédit Fraude")
    print(f"   Réel Légitime     {cm[0,0]:>8,}        {cm[0,1]:>8,}")
    print(f"   Réel Fraude       {cm[1,0]:>8,}        {cm[1,1]:>8,}")
    
    # Interprétation pédagogique
    print(f"\n🔍 INTERPRÉTATION :")
    print(f"   ✅ Vrais Positifs (fraudes détectées)   : {cm[1,1]}")
    print(f"   ❌ Faux Négatifs (fraudes non détectées): {cm[1,0]} ← À minimiser")
    print(f"   ⚠️ Faux Positifs (alertes erronées)     : {cm[0,1]}")
    
    # Calcul du pourcentage de fraudes détectées
    detection_rate = cm[1,1] / (cm[1,0] + cm[1,1]) * 100
    print(f"\n📈 TAUX DE DÉTECTION DES FRAUDES : {detection_rate:.1f}%")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'detection_rate': detection_rate
    }

# ========== FONCTION 4 : INTERPRÉTATION DES FEATURES ==========
def interpret_features(model, feature_names=None):
    """
    Affiche les features les plus importantes (poids du modèle)
    Pour la régression logistique, plus |coefficient| est grand,
    plus la feature influence la décision
    """
    # Récupération des coefficients du modèle
    coefficients = model.coef_[0]  # [0] car 1 seule classe (fraude)
    
    if feature_names is None:
        # Si pas de noms, on crée des noms génériques
        feature_names = [f"V{i}" for i in range(len(coefficients))]
    
    # Création d'un DataFrame pour trier les coefficients
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)  # Valeur absolue pour trier
    })
    
    # Tri par importance décroissante
    importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
    
    print("\n" + "=" * 60)
    print("🔍 FEATURES LES PLUS IMPORTANTES")
    print("=" * 60)
    print("Plus |coefficient| est grand, plus la feature influence la prédiction")
    print("Coefficient > 0 → augmente la probabilité de fraude")
    print("Coefficient < 0 → diminue la probabilité de fraude")
    print("-" * 50)
    
    # Affichage des 10 premières features
    for i, row in importance_df.head(10).iterrows():
        direction = "➕" if row['coefficient'] > 0 else "➖"
        print(f"   {direction} {row['feature']:<20} : {row['coefficient']:>10.4f}")

# ========== FONCTION 5 : COMPARAISON ==========
def compare_with_random(model, y_test, y_pred, y_proba):
    """
    Compare le modèle avec un classifieur aléatoire
    Pour rappel : un modèle aléatoire donne AUC = 0.5
    """
    print("\n" + "=" * 60)
    print("🎯 COMPARAISON AVEC UN CLASSIFIEUR ALÉATOIRE")
    print("=" * 60)
    
    # Simulation d'un classifieur aléatoire
    np.random.seed(42)
    random_proba = np.random.random(len(y_test))
    random_auc = roc_auc_score(y_test, random_proba)
    
    print(f"   AUC du modèle aléatoire : {random_auc:.4f} (0.5 théorique)")
    print(f"   AUC de notre modèle      : {roc_auc_score(y_test, y_proba):.4f}")
    
    if roc_auc_score(y_test, y_proba) > random_auc + 0.1:
        print("   ✅ Notre modèle est bien meilleur qu'aléatoire")
    else:
        print("   ⚠️ Notre modèle n'est pas meilleur qu'aléatoire → problème")

# ========== MAIN ==========
def main():
    """
    Fonction principale qui orchestre tout le pipeline
    """
    print("=" * 60)
    print("🚨 BASELINE - DÉTECTION DE FRAUDE BANCAIRE")
    print("=" * 60)
    
    # 1. Chargement des données
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Entraînement de la baseline
    model = train_baseline(X_train, y_train)
    
    # 3. Évaluation
    metrics = evaluate_model(model, X_test, y_test)
    
    # 4. Interprétation des features (si on a des noms)
    # Pour le dataset anonymisé, les features s'appellent V1 à V28
    feature_names = [f"V{i+1}" for i in range(X_train.shape[1] - 1)] + ["Amount_scaled"]
    interpret_features(model, feature_names)
    
    # 5. Comparaison avec aléatoire
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    compare_with_random(model, y_test, y_pred, y_proba)
    
    # 6. Conclusion
    print("\n" + "=" * 60)
    print("📌 CONCLUSION DE LA BASELINE")
    print("=" * 60)
    
    recall = metrics['recall']
    detection_rate = metrics['detection_rate']
    
    if recall > 0.7:
        print(f"✅ Le modèle détecte {detection_rate:.1f}% des fraudes (rappel = {recall:.2f})")
        print("   C'est une bonne baseline ! On peut maintenant essayer XGBoost")
    elif recall > 0.5:
        print(f"⚠️ Le modèle détecte {detection_rate:.1f}% des fraudes (rappel = {recall:.2f})")
        print("   Améliorable avec XGBoost ou SMOTE")
    else:
        print(f"❌ Le modèle détecte seulement {detection_rate:.1f}% des fraudes")
        print("   La baseline est trop faible → besoin de SMOTE et XGBoost")
    
    print("\n💡 PROCHAINE ÉTAPE : ")
    print("   → XGBoost avec SMOTE pour améliorer le rappel")
    
    return model, metrics

# ========== POINT D'ENTRÉE ==========
if __name__ == "__main__":
    # Cette condition permet d'exécuter le script directement
    # Si on importe le fichier ailleurs, le main ne s'exécute pas
    model, metrics = main()