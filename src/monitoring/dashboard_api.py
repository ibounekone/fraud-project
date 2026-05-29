"""Dashboard qui lit les métriques directement depuis l'API"""
import urllib.request
import json
import matplotlib.pyplot as plt # type: ignore

API_URL = "http://localhost:8005"
API_KEY = "fraud_detection_2024_secret_key"

def get_metrics():
    """Récupère les métriques depuis l'API"""
    req = urllib.request.Request(
        f"{API_URL}/metrics",
        headers={'X-API-Key': API_KEY},
        method='GET'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def main():
    print("=" * 50)
    print("📊 DASHBOARD - MÉTRIQUES API")
    print("=" * 50)
    
    metrics = get_metrics()
    
    if metrics:
        print(f"\n📊 MÉTRIQUES:")
        print(f"   Total prédictions: {metrics.get('total_predictions')}")
        print(f"   Taux fraude: {metrics.get('fraud_rate_detected'):.1f}%")
        print(f"   Confiance moyenne: {metrics.get('avg_confidence'):.3f}")
        print(f"   Latence moyenne: {metrics.get('avg_latency_ms'):.2f} ms")
        print(f"   Score drift: {metrics.get('drift_score')}")
        print(f"   Statut: {metrics.get('status')}")
    else:
        print("❌ Impossible de récupérer les métriques")

if __name__ == "__main__":
    main()