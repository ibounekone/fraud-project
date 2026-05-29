"""
Dashboard de monitoring pour la détection de fraude - Version robuste
Gère les fichiers JSON partiellement écrits
"""

import json
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

METRICS_FILE = Path("data/metrics/fraud_logs.json")

def load_logs_safe():
    """Charge les logs de manière robuste"""
    if not METRICS_FILE.exists():
        print("❌ Aucun log trouvé")
        return None
    
    try:
        with open(METRICS_FILE, 'r') as f:
            content = f.read().strip()
            
            # Si le fichier est vide
            if not content:
                print("⚠️ Fichier vide")
                return None
            
            # Tenter de parser le JSON
            logs = json.loads(content)
            
            # Vérifier que c'est une liste
            if not isinstance(logs, list):
                print("⚠️ Les logs ne sont pas une liste")
                return None
            
            if len(logs) == 0:
                print("⚠️ Liste de logs vide")
                return None
            
            print(f"✅ {len(logs)} logs chargés")
            return logs
            
    except json.JSONDecodeError as e:
        print(f"❌ Erreur JSON: {e}")
        # Afficher le début du fichier pour debug
        print("Contenu du fichier (début):")
        print(content[:200])
        return None
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return None

def create_dataframe(logs):
    """Convertit les logs en DataFrame pandas"""
    if not logs:
        return None
    
    rows = []
    for log in logs:
        row = {
            'timestamp': pd.to_datetime(log.get('timestamp')),
            'amount': log.get('amount', 0),
            'probability': log.get('probability', 0),
            'is_fraud': log.get('is_fraud', False),
            'threshold_used': log.get('threshold_used', 0.5),
            'latency_ms': log.get('latency_ms', 0)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('timestamp')
    return df

def generate_report(df):
    """Génère un rapport texte"""
    if df is None or df.empty:
        return
    
    print("\n" + "=" * 60)
    print("📊 RAPPORT DE MONITORING - DÉTECTION DE FRAUDE")
    print("=" * 60)
    
    print(f"\n📅 Période: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} → {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
    print(f"📊 Total prédictions: {len(df)}")
    
    # Statistiques
    frauds = df[df['is_fraud'] == True]
    legit = df[df['is_fraud'] == False]
    
    print(f"\n🚨 Transactions classées frauduleuses: {len(frauds)} ({len(frauds)/len(df)*100:.1f}%)")
    print(f"✅ Transactions classées légitimes: {len(legit)} ({len(legit)/len(df)*100:.1f}%)")
    
    # Confiance
    if len(frauds) > 0:
        print(f"\n🎯 Confiance moyenne (fraudes): {frauds['probability'].mean():.3f}")
    if len(legit) > 0:
        print(f"🎯 Confiance moyenne (légitimes): {legit['probability'].mean():.3f}")
    
    # Montants
    print(f"\n💰 Montant moyen: {df['amount'].mean():.2f}€")
    if len(frauds) > 0:
        print(f"💰 Montant moyen (fraudes): {frauds['amount'].mean():.2f}€")
    
    # Latence
    print(f"\n⚡ Latence moyenne: {df['latency_ms'].mean():.2f} ms")
    print(f"⚡ Latence max: {df['latency_ms'].max():.2f} ms")
    
    # Distribution des probabilités
    print(f"\n📊 Distribution des probabilités:")
    print(f"   < 0.3: {(df['probability'] < 0.3).sum()}")
    print(f"   0.3-0.5: {((df['probability'] >= 0.3) & (df['probability'] < 0.5)).sum()}")
    print(f"   0.5-0.7: {((df['probability'] >= 0.5) & (df['probability'] < 0.7)).sum()}")
    print(f"   0.7-0.9: {((df['probability'] >= 0.7) & (df['probability'] < 0.9)).sum()}")
    print(f"   >= 0.9: {(df['probability'] >= 0.9).sum()}")

def plot_dashboard(df):
    """Génère les graphiques"""
    if df is None or df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Évolution des probabilités
    ax1 = axes[0, 0]
    ax1.plot(df['timestamp'], df['probability'], 'b-', alpha=0.7, linewidth=1)
    ax1.axhline(y=0.5, color='r', linestyle='--', label='Seuil (0.5)')
    ax1.set_title('Évolution des probabilités de fraude')
    ax1.set_xlabel('Temps')
    ax1.set_ylabel('Probabilité')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Distribution des probabilités
    ax2 = axes[0, 1]
    frauds = df[df['is_fraud'] == True]['probability']
    legit = df[df['is_fraud'] == False]['probability']
    ax2.hist(frauds, bins=20, alpha=0.5, label='Fraudes', color='red')
    ax2.hist(legit, bins=20, alpha=0.5, label='Légitimes', color='green')
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
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Montant par type
    ax4 = axes[1, 1]
    fraud_amounts = df[df['is_fraud'] == True]['amount'] if len(frauds) > 0 else []
    legit_amounts = df[df['is_fraud'] == False]['amount'] if len(legit) > 0 else []
    
    box_data = []
    labels = []
    if len(fraud_amounts) > 0:
        box_data.append(fraud_amounts)
        labels.append('Fraudes')
    if len(legit_amounts) > 0:
        box_data.append(legit_amounts)
        labels.append('Légitimes')
    
    if box_data:
        ax4.boxplot(box_data, labels=labels)
    ax4.set_title('Distribution des montants')
    ax4.set_ylabel('Montant (€)')
    
    plt.tight_layout()
    plt.savefig('fraud_monitoring_dashboard.png', dpi=150, bbox_inches='tight')
    print("\n📊 Dashboard sauvegardé: fraud_monitoring_dashboard.png")
    plt.show()

def main():
    print("🔍 GÉNÉRATION DU DASHBOARD - DÉTECTION DE FRAUDE")
    print("=" * 60)
    
    logs = load_logs_safe()
    
    if logs is None:
        print("\n❌ Aucune donnée trouvée.")
        print("   Lance d'abord l'API et fais des prédictions:")
        print("   python src/api/predict_fraud_with_monitoring.py")
        return
    
    df = create_dataframe(logs)
    generate_report(df)
    plot_dashboard(df)
    
    print("\n✅ Dashboard généré avec succès!")

if __name__ == "__main__":
    main()