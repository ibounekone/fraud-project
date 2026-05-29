"""
Dashboard de monitoring - Version définitive
Lit toutes les métriques directement depuis l'API
Plus fiable que les fichiers JSON
"""

import urllib.request
import json
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from datetime import datetime

API_URL = "http://localhost:8005"
API_KEY = "fraud_detection_2024_secret_key"

def api_request(endpoint: str, data: dict = None):
    """Fait une requête à l'API"""
    req = urllib.request.Request(
        f"{API_URL}{endpoint}",
        headers={'X-API-Key': API_KEY, 'Content-Type': 'application/json'}
    )
    
    if data:
        req.data = json.dumps(data).encode('utf-8')
        req.method = 'POST'
    else:
        req.method = 'GET'
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"❌ Erreur API {endpoint}: {e}")
        return None

def get_metrics():
    """Récupère les métriques globales"""
    return api_request("/metrics")

def get_logs(limit: int = 50):
    """Récupère les logs récents"""
    return api_request(f"/logs?limit={limit}")

def get_drift():
    """Récupère l'analyse de dérive"""
    return api_request("/drift")

def print_report(metrics, drift, logs):
    """Affiche le rapport textuel"""
    print("\n" + "=" * 60)
    print("📊 RAPPORT DE MONITORING - DÉTECTION DE FRAUDE")
    print("=" * 60)
    
    if metrics:
        print(f"\n🎯 MÉTRIQUES GLOBALES:")
        print(f"   Total prédictions: {metrics.get('total_predictions')}")
        print(f"   Taux fraude détectée: {metrics.get('fraud_rate_detected'):.1f}%")
        print(f"   Confiance moyenne: {metrics.get('avg_confidence'):.3f}")
        print(f"   Latence moyenne: {metrics.get('avg_latency_ms'):.2f} ms")
    
    if drift:
        print(f"\n📈 ANALYSE DE DÉRIVE:")
        print(f"   Score drift: {drift.get('drift_score')}")
        print(f"   Statut: {drift.get('status')}")
        print(f"   Message: {drift.get('message')}")
        print(f"   Recommandation: {drift.get('recommendation')}")
    
    if logs and logs.get('logs'):
        logs_list = logs.get('logs', [])
        print(f"\n📋 DERNIÈRES PRÉDICTIONS ({len(logs_list)} affichées):")
        print("-" * 50)
        for log in logs_list[-10:]:
            timestamp = log.get('timestamp', 'N/A')[:19]
            is_fraud = "🚨 FRAUDE" if log.get('is_fraud') else "✅ LÉGITIME"
            prob = log.get('probability', 0)
            amount = log.get('amount', 0)
            print(f"   {timestamp} | {is_fraud} | prob: {prob:.3f} | montant: {amount:.0f}€")

def plot_metrics():
    """Génère des graphiques à partir des logs"""
    logs_data = api_request("/logs?limit=200")
    
    if not logs_data or not logs_data.get('logs'):
        print("⚠️ Pas assez de données pour les graphiques")
        return
    
    logs = logs_data['logs']
    
    if len(logs) < 5:
        print("⚠️ Minimum 5 prédictions requises pour les graphiques")
        return
    
    # Préparer les données
    timestamps = [log.get('timestamp', '')[:19] for log in logs]
    probabilities = [log.get('probability', 0) for log in logs]
    is_fraud = [log.get('is_fraud', False) for log in logs]
    amounts = [log.get('amount', 0) for log in logs]
    latencies = [log.get('latency_ms', 0) for log in logs]
    
    # Créer les graphiques
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Évolution des probabilités
    ax1 = axes[0, 0]
    colors = ['red' if f else 'green' for f in is_fraud]
    ax1.scatter(range(len(probabilities)), probabilities, c=colors, alpha=0.6, s=30)
    ax1.axhline(y=0.5, color='blue', linestyle='--', label='Seuil (0.5)')
    ax1.set_title('Évolution des probabilités de fraude')
    ax1.set_xlabel('Prédiction (ordre chronologique)')
    ax1.set_ylabel('Probabilité')
    ax1.legend()
    
    # 2. Distribution des probabilités
    ax2 = axes[0, 1]
    fraud_probs = [p for p, f in zip(probabilities, is_fraud) if f]
    legit_probs = [p for p, f in zip(probabilities, is_fraud) if not f]
    ax2.hist(fraud_probs, bins=15, alpha=0.5, label='Fraudes', color='red')
    ax2.hist(legit_probs, bins=15, alpha=0.5, label='Légitimes', color='green')
    ax2.set_title('Distribution des probabilités')
    ax2.set_xlabel('Probabilité')
    ax2.set_ylabel('Fréquence')
    ax2.legend()
    
    # 3. Latence
    ax3 = axes[1, 0]
    ax3.plot(range(len(latencies)), latencies, 'b-', alpha=0.7, linewidth=1)
    ax3.set_title('Latence des requêtes')
    ax3.set_xlabel('Prédiction')
    ax3.set_ylabel('Latence (ms)')
    
    # 4. Montants par type
    ax4 = axes[1, 1]
    fraud_amounts = [a for a, f in zip(amounts, is_fraud) if f]
    legit_amounts = [a for a, f in zip(amounts, is_fraud) if not f]
    
    box_data = []
    labels = []
    if fraud_amounts:
        box_data.append(fraud_amounts)
        labels.append('Fraudes')
    if legit_amounts:
        box_data.append(legit_amounts)
        labels.append('Légitimes')
    
    if box_data:
        ax4.boxplot(box_data, labels=labels)
    ax4.set_title('Distribution des montants')
    ax4.set_ylabel('Montant (€)')
    
    plt.tight_layout()
    plt.savefig('fraud_monitoring_dashboard.png', dpi=150, bbox_inches='tight')
    print("\n📊 Graphiques sauvegardés: fraud_monitoring_dashboard.png")
    plt.show()

def main():
    print("🔍 GÉNÉRATION DU DASHBOARD - DÉTECTION DE FRAUDE")
    print("=" * 60)
    
    # Vérifier que l'API est accessible
    try:
        urllib.request.urlopen(f"{API_URL}/health", timeout=2)
        print("✅ API accessible")
    except:
        print("❌ API non accessible. Lance d'abord:")
        print("   python src/api/predict_fraud_with_monitoring.py")
        return
    
    # Récupérer les données
    metrics = get_metrics()
    drift = get_drift()
    logs = get_logs(50)
    
    # Afficher le rapport
    print_report(metrics, drift, logs)
    
    # Générer les graphiques
    if metrics and metrics.get('total_predictions', 0) > 0:
        plot_metrics()
    else:
        print("\n⚠️ Fais quelques prédictions d'abord:")
        print("   curl -X POST http://localhost:8005/predict ...")
    
    print("\n✅ Dashboard généré avec succès!")

if __name__ == "__main__":
    main()