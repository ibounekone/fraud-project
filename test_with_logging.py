"""
Script de test avec écriture JSON robuste
Ne plante pas si l'écriture échoue
"""
import urllib.request
import json
import time
from pathlib import Path

API_URL = "http://localhost:8005/predict"
API_KEY = "fraud_detection_2024_secret_key"

# Transaction de test
transaction = {
    "V1": -1.5, "V2": 0.5, "V3": -0.8, "V4": 1.2, "V5": -0.3,
    "V6": 0.7, "V7": -0.2, "V8": 0.1, "V9": -0.9, "V10": 0.4,
    "V11": -0.5, "V12": 1.1, "V13": -0.7, "V14": 0.2, "V15": -0.4,
    "V16": 0.8, "V17": -0.1, "V18": 0.5, "V19": -0.6, "V20": 0.3,
    "V21": -0.2, "V22": 0.6, "V23": -0.4, "V24": 0.1, "V25": -0.3,
    "V26": 0.2, "V27": -0.1, "V28": 0.4,
    "Amount": 1500,
    "transaction_id": f"test_{int(time.time())}"
}

def send_prediction():
    """Envoie une prédiction à l'API"""
    payload = json.dumps({
        "transaction": transaction,
        "threshold": 0.5
    }).encode('utf-8')
    
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'X-API-Key': API_KEY
        },
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode('utf-8'))
            print(f"✅ Prédiction: fraud={result.get('is_fraud')}, conf={result.get('confidence'):.3f}")
            return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    print("=" * 50)
    print("📊 TEST DE L'API AVEC LOGGING")
    print("=" * 50)
    
    # Vérifier que l'API tourne
    try:
        urllib.request.urlopen("http://localhost:8005/health", timeout=2)
        print("✅ API accessible")
    except:
        print("❌ API non accessible. Lance d'abord l'API:")
        print("   python src/api/predict_fraud_with_monitoring.py")
        return
    
    # Faire quelques prédictions
    for i in range(3):
        print(f"\n📝 Prédiction {i+1}/3")
        send_prediction()
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("✅ Tests terminés")
    
    # Vérifier les logs
    log_file = Path("data/metrics/fraud_logs.json")
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                logs = json.loads(content)
                print(f"\n📊 {len(logs)} logs enregistrés")
                if logs:
                    print(f"Dernier log: {logs[-1].get('timestamp')}")
        except json.JSONDecodeError as e:
            print(f"\n❌ Logs corrompus: {e}")
            print("Solution: rm data/metrics/fraud_logs.json && relance l'API")
    else:
        print("\n❌ Aucun fichier de logs")

if __name__ == "__main__":
    main()