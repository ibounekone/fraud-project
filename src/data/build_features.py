"""Chargement et exploration du dataset de fraude bancaire"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

DATA_PATH = Path("data/raw/creditcard.csv")
PROCESSED_PATH = Path("data/processed/")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

def load_data():
    """Charge le dataset"""
    print("📂 Chargement des données...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ {len(df)} transactions, {len(df.columns)} colonnes")
    print(f"📊 Fraudes : {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")
    return df

def explore_data(df):
    """Analyse exploratoire basique"""
    print("\n" + "=" * 50)
    print("📊 EXPLORATION DES DONNÉES")
    print("=" * 50)
    
    # Distribution des classes
    frauds = df[df['Class'] == 1]
    legit = df[df['Class'] == 0]
    
    print(f"\n✅ Transactions légitimes : {len(legit):,} ({len(legit)/len(df)*100:.2f}%)")
    print(f"🚨 Transactions frauduleuses : {len(frauds):,} ({len(frauds)/len(df)*100:.4f}%)")
    
    # Montant moyen
    print(f"\n💰 Montant moyen - légitime : {legit['Amount'].mean():.2f}€")
    print(f"💰 Montant moyen - fraude : {frauds['Amount'].mean():.2f}€")
    
    # Stats sur le temps
    print(f"\n⏰ Temps (secondes depuis première transaction)")
    print(f"   Min: {df['Time'].min()}, Max: {df['Time'].max()}")
    
    return frauds, legit

def prepare_data(df):
    """Prépare les features pour le modèle"""
    # Features (toutes sauf Time, Amount, Class)
    feature_cols = [col for col in df.columns if col not in ['Time', 'Amount', 'Class']]
    
    # Normalisation de Amount (optionnel car déjà échelonné dans le dataset)
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    feature_cols.append('Amount_scaled')
    
    X = df[feature_cols].values
    y = df['Class'].values
    
    print(f"\n🔧 Features : {len(feature_cols)} colonnes")
    print(f"   Forme de X : {X.shape}")
    print(f"   Forme de y : {y.shape}")
    
    return X, y, feature_cols, scaler

def split_data(X, y):
    """Sépare les données en train/test (en préservant le déséquilibre)"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Important : garde le même ratio de fraudes
    )
    
    print(f"\n📊 Répartition des classes :")
    print(f"   Train : {len(y_train)} samples, fraudes : {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"   Test  : {len(y_test)} samples, fraudes : {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test):
    """Sauvegarde les données préparées"""
    np.save(PROCESSED_PATH / "X_train.npy", X_train)
    np.save(PROCESSED_PATH / "X_test.npy", X_test)
    np.save(PROCESSED_PATH / "y_train.npy", y_train)
    np.save(PROCESSED_PATH / "y_test.npy", y_test)
    print(f"\n💾 Données sauvegardées dans {PROCESSED_PATH}")

def main():
    print("=" * 60)
    print("🔍 PRÉPARATION DONNÉES - DÉTECTION FRAUDE BANCAIRE")
    print("=" * 60)
    
    # 1. Chargement
    df = load_data()
    
    # 2. Exploration
    frauds, legit = explore_data(df)
    
    # 3. Préparation des features
    X, y, features, scaler = prepare_data(df)
    
    # 4. Séparation train/test
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 5. Sauvegarde
    save_processed_data(X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 60)
    print("✅ Données prêtes pour la modélisation !")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = main()