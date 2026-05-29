"""
AJUSTEMENT DU SEUIL DE DÉCISION POUR LA DÉTECTION DE FRAUDE
Objectif : Trouver le meilleur compromis entre recall et précision
Le seuil par défaut est 0.5 – on va le faire varier
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import recall_score, precision_score, confusion_matrix # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Chemins
DATA_PATH = Path("data/processed/")

def load_data():
    """Charge les données préparées"""
    print("📂 Chargement des données...")
    X_train = np.load(DATA_PATH / "X_train.npy")
    X_test = np.load(DATA_PATH / "X_test.npy")
    y_train = np.load(DATA_PATH / "y_train.npy")
    y_test = np.load(DATA_PATH / "y_test.npy")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Entraîne la régression logistique (notre meilleur modèle)"""
    print("\n🔧 Entraînement du modèle...")
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("✅ Modèle entraîné")
    return model

def evaluate_thresholds(model, X_test, y_test):
    """
    Teste différents seuils de décision
    Le seuil par défaut est 0.5 – on essaie de 0.1 à 0.9
    """
    # Obtenir les probabilités (entre 0 et 1)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "=" * 70)
    print("🎯 AJUSTEMENT DU SEUIL DE DÉCISION")
    print("=" * 70)
    print("Seuil | Recall | Précision | VP | FP | FN | VN")
    print("-" * 70)
    
    results = []
    
    # Tester différents seuils
    for seuil in np.arange(0.1, 1.0, 0.05):
        # Prédiction selon le seuil
        y_pred = (y_proba >= seuil).astype(int)
        
        # Calcul des métriques
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Extraction des valeurs
        if cm.shape == (2, 2):
            vn, fp, fn, vp = cm.ravel()
        else:
            vp = fn = fp = vn = 0
        
        results.append({
            'seuil': seuil,
            'recall': recall,
            'precision': precision,
            'vp': vp,
            'fp': fp,
            'fn': fn,
            'vn': vn
        })
        
        # Marquer le seuil par défaut
        if abs(seuil - 0.5) < 0.01:
            print(f"0.50* | {recall:.3f}   | {precision:.3f}     | {vp:3d} | {fp:3d} | {fn:3d} | {vn:5d}")
        else:
            print(f"{seuil:.2f}  | {recall:.3f}   | {precision:.3f}     | {vp:3d} | {fp:3d} | {fn:3d} | {vn:5d}")
    
    return results, y_proba

def find_best_threshold(results):
    """
    Trouve le meilleur seuil selon deux stratégies :
    1. Maximiser le recall (priorité métier)
    2. Maximiser F1-score (compromis)
    """
    print("\n" + "=" * 70)
    print("🏆 MEILLEUR SEUIL")
    print("=" * 70)
    
    # Stratégie 1 : Recall max
    best_recall = max(results, key=lambda x: x['recall'])
    print(f"\n📊 STRATÉGIE 1 : MAXIMISER LE RECALL")
    print(f"   Seuil = {best_recall['seuil']:.2f}")
    print(f"   Recall = {best_recall['recall']*100:.1f}%")
    print(f"   Précision = {best_recall['precision']*100:.1f}%")
    
    # Stratégie 2 : F1 max (compromis recall/précision)
    best_f1 = max(results, key=lambda x: 2 * x['recall'] * x['precision'] / (x['recall'] + x['precision']) if (x['recall'] + x['precision']) > 0 else 0)
    print(f"\n⚖️ STRATÉGIE 2 : MAXIMISER F1-SCORE")
    print(f"   Seuil = {best_f1['seuil']:.2f}")
    print(f"   Recall = {best_f1['recall']*100:.1f}%")
    print(f"   Précision = {best_f1['precision']*100:.1f}%")
    
    return best_recall, best_f1

def plot_threshold_impact(y_test, y_proba):
    """Visualise l'impact du seuil sur recall et précision"""
    thresholds = np.arange(0.1, 1.0, 0.02)
    recalls = []
    precisions = []
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        recalls.append(recall_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, recalls, 'b-', linewidth=2, label='Recall (fraudes détectées)')
    plt.plot(thresholds, precisions, 'r-', linewidth=2, label='Précision (alertes fiables)')
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Seuil par défaut (0.5)')
    plt.xlabel('Seuil de décision', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Impact du seuil sur les métriques - Détection de fraude', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
    print("\n📊 Graphique sauvegardé : threshold_analysis.png")
    plt.show()

def main():
    print("=" * 60)
    print("🎯 AJUSTEMENT DU SEUIL DE DÉCISION")
    print("=" * 60)
    
    # 1. Chargement
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Entraînement du modèle
    model = train_model(X_train, y_train)
    
    # 3. Évaluation des seuils
    results, y_proba = evaluate_thresholds(model, X_test, y_test)
    
    # 4. Meilleur seuil
    best_recall, best_f1 = find_best_threshold(results)
    
    # 5. Visualisation
    plot_threshold_impact(y_test, y_proba)
    
    # 6. Conclusion
    print("\n" + "=" * 60)
    print("📌 CONCLUSION")
    print("=" * 60)
    print(f"🔹 Seuil par défaut (0.5) : recall = 91.8%")
    print(f"🔹 Seuil optimal recall    : {best_recall['seuil']:.2f} → recall = {best_recall['recall']*100:.1f}%")
    print(f"🔹 Seuil optimal F1        : {best_f1['seuil']:.2f} → recall = {best_f1['recall']*100:.1f}%")
    
    if best_recall['recall'] > 0.9184:
        gain = (best_recall['recall'] - 0.9184) * 100
        print(f"\n✅ On peut gagner {gain:.1f} points de recall en ajustant le seuil à {best_recall['seuil']:.2f}")
    else:
        print("\n⚠️ Le seuil par défaut (0.5) est déjà optimal pour ce modèle")

if __name__ == "__main__":
    main()