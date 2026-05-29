"""
Dashboard de monitoring pour la détection de fraude
Génère des graphiques et un rapport
"""

import json
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

METRICS_FILE = Path("data/metrics/fraud_logs.json")

def load_logs():
    """Charge les logs de monitoring"""
    if not METRICS_FILE.exists():
        print("❌ Aucun log trouvé. Lance d'abord l'API et fais des prédictions.")
        return None
    
    with open(METRICS_FILE, 'r') as f:
        logs = json.load(f)
    
    if not logs:
        return None
    
    df = pd.DataFrame(logs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def generate_report(df):
    """Génère un rapport texte"""
    if df is None or df.empty:
        return
    
    print("\n" + "=" * 60)
    print("📊 RAPPORT DE MONITORING - DÉTECTION DE FRAUDE")
    print("=" * 60)
    
    print(f"\n📅 Période: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"📊 Total prédictions: {len(df)}")
    
    # Statistiques
    frauds = df[df['is_fraud'] == True]
    legit = df[df['is_fraud'] == False]
    
    print(f"\n🚨 Transactions classées frauduleuses: {len(frauds)} ({len(frauds)/len(df)*100:.1f}%)")
    print(f"✅ Transactions classées légitimes: {len(legit)} ({len(legit)/len(df)*100:.1f}%)")
    
    # Confiance
    print(f"\n🎯 Confiance moyenne (fraudes): {frauds['probability'].mean():.3f}")
    print(f"🎯 Confiance moyenne (légitimes): {legit['probability'].mean():.3f}")
    
    # Montants
    print(f"\n💰 Montant moyen (fraudes): {frauds['amount'].mean():.2f}€")
    print(f"💰 Montant moyen (légitimes): {legit['amount'].mean():.2f}€")
    
    # Latence
    print(f"\n⚡ Latence moyenne: {df['latency_ms'].mean():.2f} ms")
    print(f"⚡ Latence max: {df['latency_ms'].max():.2f} ms")
    
    # Feedback (si disponible)
    if 'true_class' in df.columns:
        correct = df[df['was_correct'] == True]
        print(f"\n✅ Exactitude terrain: {len(correct)}/{len(df)} ({len(correct)/len(df)*100:.1f}%)")

def plot_dashboard(df):
    """Génère les graphiques"""
    if df is None or df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Évolution des prédictions
    ax1 = axes[0, 0]
    ax1.plot(df['timestamp'], df['probability'], 'b-', alpha=0.5, linewidth=0.5)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Seuil par défaut')
    ax1.set_title('Évolution des probabilités de fraude')
    ax1.set_xlabel('Temps')
    ax1.set_ylabel('Probabilité')
    ax1.legend()
    
    # 2. Distribution des probabilités
    ax2 = axes[0, 1]
    frauds = df[df['is_fraud'] == True]['probability']
    legit = df[df['is_fraud'] == False]['probability']
    ax2.hist(frauds, bins=20, alpha=0.5, label='Classées fraudes', color='red')
    ax2.hist(legit, bins=20, alpha=0.5, label='Classées légitimes', color='green')
    ax2.set_title('Distribution des probabilités')
    ax2.set_xlabel('Probabilité')
    ax2.set_ylabel('Fréquence')
    ax2.legend()
    
    # 3. Latence
    ax3 = axes[1, 0]
    ax3.plot(df['timestamp'], df['latency_ms'], 'g-', alpha=0.7, linewidth=1)
    ax3.set_title('Latence des requêtes')
    ax3.set_xlabel('Temps')
    ax3.set_ylabel('Latence (ms)')
    
    # 4. Montant par type
    ax4 = axes[1, 1]
    fraud_amounts = df[df['is_fraud'] == True]['amount']
    legit_amounts = df[df['is_fraud'] == False]['amount']
    ax4.boxplot([fraud_amounts, legit_amounts], labels=['Fraudes', 'Légitimes'])
    ax4.set_title('Distribution des montants')
    ax4.set_ylabel('Montant (€)')
    
    plt.tight_layout()
    plt.savefig('fraud_monitoring_dashboard.png', dpi=150, bbox_inches='tight')
    print("\n📊 Dashboard sauvegardé: fraud_monitoring_dashboard.png")
    plt.show()

def main():
    print("🔍 GÉNÉRATION DU DASHBOARD - DÉTECTION DE FRAUDE")
    print("=" * 60)
    
    df = load_logs()
    
    if df is None:
        print("\n❌ Aucune donnée trouvée.")
        print("   Lance d'abord: python src/api/predict_fraud_with_monitoring.py")
        print("   Puis fais quelques prédictions avec curl")
        return
    
    generate_report(df)
    plot_dashboard(df)

if __name__ == "__main__":
    main()