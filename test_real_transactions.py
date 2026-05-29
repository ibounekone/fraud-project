"""Test de l'API avec de vraies transactions du dataset"""
import urllib.request
import json
import pandas as pd

API_URL = "http://localhost:8005/predict"
API_KEY = "fraud_detection_2024_secret_key"

# Charger le dataset
print("📂 Chargement du dataset...")
df = pd.read_csv('data/raw/creditcard.csv')

# Sélectionner 5 transactions légitimes et 5 frauduleuses
legit = df[df['Class'] == 0].head(5)
fraud = df[df['Class'] == 1].head(5)

# Features à envoyer
features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

def predict(transaction):
    """Envoie une transaction à l'API"""
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
    
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.loads(response.read().decode('utf-8'))

print("\n" + "=" * 60)
print("🧪 TEST AVEC VRAIES TRANSACTIONS")
print("=" * 60)

print("\n✅ TRANSACTIONS LÉGITIMES (devraient être classées légitimes):")
for i, row in legit.iterrows():
    transaction = {f: row[f] for f in features}
    transaction['Amount'] = float(row['Amount'])
    
    result = predict(transaction)
    status = "✅ LÉGITIME" if not result['is_fraud'] else "❌ FRAUDE (ERREUR)"
    print(f"   {status} | conf: {result['confidence']:.3f} | montant: {row['Amount']:.2f}€")

print("\n🚨 TRANSACTIONS FRAUDULEUSES (devraient être classées fraudes):")
for i, row in fraud.iterrows():
    transaction = {f: row[f] for f in features}
    transaction['Amount'] = float(row['Amount'])
    
    result = predict(transaction)
    status = "🚨 FRAUDE" if result['is_fraud'] else "✅ LÉGITIME (ERREUR)"
    print(f"   {status} | conf: {result['confidence']:.3f} | montant: {row['Amount']:.2f}€")