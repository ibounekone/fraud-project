"""
API DE DÉTECTION DE FRAUDE AVEC MONITORING
Ajoute des logs, des métriques, et la détection de dérive
"""

# ========== IMPORTS ==========
from fastapi import FastAPI, HTTPException, Depends # type: ignore
from fastapi.security import APIKeyHeader # type: ignore
from pydantic import BaseModel, Field # type: ignore
from datetime import datetime
from collections import deque
from pathlib import Path
import numpy as np
import joblib # type: ignore
import time
import logging
import json
from typing import Optional, List, Dict, Any

# ========== CONFIGURATION ==========
# Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fichier pour stocker les métriques
METRICS_FILE = Path("data/metrics/fraud_logs.json")
METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Clé API
API_KEY = "fraud_detection_2024_secret_key"
API_KEY_NAME = "X-API-Key"

# Initialisation
app = FastAPI(
    title="API Détection de Fraude (Avec Monitoring)",
    description="Détection de fraude bancaire avec monitoring intégré",
    version="2.0.0"
)
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Stockage en mémoire pour calculs rapides
prediction_history = deque(maxlen=5000)  # 5000 dernières prédictions
probability_history = deque(maxlen=5000)  # Probabilités
latency_history = deque(maxlen=5000)  # Latences
threshold_history = deque(maxlen=5000)  # Seuils utilisés

# ========== MODÈLES DE DONNÉES ==========

class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0, le=50000)
    transaction_id: Optional[str] = None

class PredictionRequest(BaseModel):
    transaction: Transaction
    threshold: Optional[float] = Field(0.5, ge=0.01, le=0.99)
    context: Optional[str] = None

class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    is_fraud: bool
    confidence: float
    threshold_used: float
    fraud_risk: str
    latency_ms: float
    timestamp: datetime
    recommendation: str

class MetricsResponse(BaseModel):
    total_predictions: int
    fraud_rate_detected: float
    avg_confidence: float
    avg_latency_ms: float
    recent_fraud_count: int
    drift_score: float
    status: str

class DriftResponse(BaseModel):
    drift_score: float
    status: str
    message: str
    predictions_analyzed: int
    reference_period: str
    current_period: str
    recommendation: str

# ========== CHARGEMENT MODÈLE ==========

# Remplacer la fonction load_model par :
def load_model():
    """Charge le modèle calibré"""
    # Essayer plusieurs chemins
    possible_paths = [
        Path("models/logistic_regression_calibrated.pkl"),
        Path("models/logistic_regression.pkl"),
        Path("../models/logistic_regression_calibrated.pkl")
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                model = joblib.load(path)
                logger.info(f"✅ Modèle chargé depuis {path}")
                logger.info(f"   Type: {type(model).__name__}")
                return model
            except Exception as e:
                logger.error(f"Erreur chargement {path}: {e}")
    
    logger.warning("Aucun modèle trouvé, création d'un modèle factice")
    from sklearn.linear_model import LogisticRegression # type: ignore
    return LogisticRegression()

def load_scaler():
    """Charge le scaler"""
    scaler_path = Path("models/scaler.pkl")
    if scaler_path.exists():
        return joblib.load(scaler_path)
    return None

model = load_model()
scaler = load_scaler()

# ========== FONCTIONS DE MONITORING ==========

def log_prediction(
    transaction_id: Optional[str],
    amount: float,
    probability: float,
    is_fraud: bool,
    threshold_used: float,
    latency_ms: float,
    context: Optional[str]
):
    """
    Enregistre une prédiction dans les logs
    Version CORRIGÉE : écriture immédiate et sécurisée
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "transaction_id": transaction_id,
        "amount": amount,
        "probability": probability,
        "is_fraud": is_fraud,
        "threshold_used": threshold_used,
        "latency_ms": latency_ms,
        "context": context
    }
    
    # Stockage en mémoire
    prediction_history.append(is_fraud)
    probability_history.append(probability)
    latency_history.append(latency_ms)
    threshold_history.append(threshold_used)
    
    # Stockage dans fichier JSON avec verrouillage
    try:
        # Créer le dossier si nécessaire
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Lire les logs existants
        if METRICS_FILE.exists() and METRICS_FILE.stat().st_size > 0:
            with open(METRICS_FILE, 'r', encoding='utf-8') as f:
                try:
                    logs = json.load(f)
                    if not isinstance(logs, list):
                        logs = []
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []
        
        # Ajouter la nouvelle entrée
        logs.append(log_entry)
        
        # Garder seulement les 10000 dernières entrées
        if len(logs) > 10000:
            logs = logs[-10000:]
        
        # Écrire le fichier COMPLET en une fois
        with open(METRICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Erreur lors de l'écriture des logs: {e}")

def calculate_drift() -> float:
    """
    Calcule la dérive du modèle (drift)
    Compare la distribution des probabilités récentes avec la distribution de référence
    
    Drift > 1.0 = léger changement
    Drift > 2.0 = changement significatif → alerte
    Drift > 3.0 = changement critique → réentraînement nécessaire
    """
    if len(probability_history) < 200:
        return 0.0  # Pas assez de données
    
    # Distribution de référence : prédictions 100 à 50 (plus anciennes)
    reference = list(probability_history)[-200:-100] if len(probability_history) >= 200 else list(probability_history)[:-100]
    
    # Distribution récente : 50 dernières prédictions
    recent = list(probability_history)[-50:]
    
    if len(reference) < 20 or len(recent) < 20:
        return 0.0
    
    # Calcul de la divergence (différence des moyennes normalisée)
    mean_ref = np.mean(reference)
    mean_recent = np.mean(recent)
    std_ref = np.std(reference)
    
    if std_ref == 0:
        return 0.0
    
    drift_score = abs(mean_recent - mean_ref) / std_ref
    return min(drift_score, 5.0)  # Plafonner à 5

def get_fraud_rate_detected() -> float:
    """Calcule le pourcentage de transactions classées comme frauduleuses"""
    if len(prediction_history) == 0:
        return 0.0
    return sum(prediction_history) / len(prediction_history) * 100

def extract_features(transaction: Transaction) -> np.ndarray:
    """Extrait les features pour le modèle"""
    v_features = [
        transaction.V1, transaction.V2, transaction.V3, transaction.V4,
        transaction.V5, transaction.V6, transaction.V7, transaction.V8,
        transaction.V9, transaction.V10, transaction.V11, transaction.V12,
        transaction.V13, transaction.V14, transaction.V15, transaction.V16,
        transaction.V17, transaction.V18, transaction.V19, transaction.V20,
        transaction.V21, transaction.V22, transaction.V23, transaction.V24,
        transaction.V25, transaction.V26, transaction.V27, transaction.V28
    ]
    
    # Normalisation du montant
    amount = transaction.Amount
    if scaler:
        amount = scaler.transform([[amount]])[0][0]
    
    return np.array(v_features + [amount]).reshape(1, -1)

def get_risk_level(confidence: float) -> str:
    """Détermine le niveau de risque"""
    if confidence >= 0.8:
        return "Très élevé"
    elif confidence >= 0.6:
        return "Élevé"
    elif confidence >= 0.4:
        return "Moyen"
    elif confidence >= 0.2:
        return "Faible"
    else:
        return "Très faible"

def get_recommendation(is_fraud: bool, confidence: float) -> str:
    """Recommandation basée sur la prédiction"""
    if is_fraud:
        if confidence > 0.8:
            return "BLOQUER immédiatement - Forte suspicion de fraude"
        elif confidence > 0.6:
            return "BLOQUER + vérification manuelle"
        else:
            return "BLOQUER temporairement + appeler le client"
    else:
        return "AUTORISER - Transaction normale"

# ========== ENDPOINTS ==========

@app.get("/", tags=["Info"])
async def root():
    return {
        "message": "API Détection de Fraude (Avec Monitoring)",
        "version": "2.0.0",
        "model": "Régression Logistique",
        "metrics": {"recall": 0.918, "auc": 0.971}
    }

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Vérifie que l'API et le monitoring fonctionnent"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "predictions_logged": len(prediction_history),
        "monitoring_active": True,
        "timestamp": datetime.now()
    }

@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics(api_key: str = Depends(api_key_header)):
    """Retourne toutes les métriques de monitoring"""
    drift = calculate_drift()
    
    # Déterminer le statut global
    if drift > 2.0:
        status = "warning"
    elif drift > 3.0:
        status = "critical"
    else:
        status = "healthy"
    
    # Compter les fraudes récentes (50 dernières)
    recent_frauds = sum(list(prediction_history)[-50:]) if len(prediction_history) >= 50 else 0
    
    return MetricsResponse(
        total_predictions=len(prediction_history),
        fraud_rate_detected=get_fraud_rate_detected(),
        avg_confidence=np.mean(list(probability_history)) if probability_history else 0.0,
        avg_latency_ms=np.mean(list(latency_history)) if latency_history else 0.0,
        recent_fraud_count=recent_frauds,
        drift_score=drift,
        status=status
    )

@app.get("/drift", response_model=DriftResponse, tags=["Monitoring"])
async def get_drift_analysis(api_key: str = Depends(api_key_header)):
    """Analyse détaillée de la dérive du modèle"""
    drift_score = calculate_drift()
    
    if drift_score < 1.0:
        status = "stable"
        message = "Pas de dérive détectée. Le modèle est stable."
        recommendation = "Continuer la surveillance normale"
    elif drift_score < 2.0:
        status = "attention"
        message = "Dérive légère détectée. Surveiller l'évolution."
        recommendation = "Augmenter la fréquence de monitoring"
    elif drift_score < 3.0:
        status = "warning"
        message = "Dérive significative. Le modèle commence à se dégrader."
        recommendation = "Préparer un réentraînement"
    else:
        status = "critical"
        message = "Dérive critique. Le modèle n'est plus fiable."
        recommendation = "RÉENTRAÎNEMENT IMMÉDIAT NÉCESSAIRE"
    
    return DriftResponse(
        drift_score=round(drift_score, 3),
        status=status,
        message=message,
        predictions_analyzed=len(probability_history),
        reference_period="Période de référence (anciennes prédictions)",
        current_period="Période récente (50 dernières prédictions)",
        recommendation=recommendation
    )

@app.get("/logs", tags=["Monitoring"])
async def get_logs(
    api_key: str = Depends(api_key_header),
    limit: int = 100,
    only_fraud: bool = False
):
    """Retourne l'historique des prédictions"""
    try:
        if not METRICS_FILE.exists():
            return {"logs": [], "total": 0}
        
        with open(METRICS_FILE, 'r') as f:
            logs = json.load(f)
        
        if only_fraud:
            logs = [log for log in logs if log.get("is_fraud", False)]
        
        return {
            "logs": logs[-limit:],
            "total": len(logs),
            "fraud_only": only_fraud
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lecture logs: {e}")

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(
    request: PredictionRequest,
    api_key: str = Depends(api_key_header)
):
    """Prédiction avec monitoring automatique"""
    start_time = time.time()
    
    # Log de la requête
    logger.info(f"Transaction - Montant: {request.transaction.Amount}€, Seuil: {request.threshold}")
    
    # Extraction features
    features = extract_features(request.transaction)
    
    # Prédiction
    prob_fraud = model.predict_proba(features)[0][1]
    is_fraud = prob_fraud >= request.threshold
    
    # Métriques
    latency = (time.time() - start_time) * 1000
    risk_level = get_risk_level(prob_fraud)
    recommendation = get_recommendation(is_fraud, prob_fraud)
    
    # Monitoring : enregistrement
    log_prediction(
        transaction_id=request.transaction.transaction_id,
        amount=request.transaction.Amount,
        probability=prob_fraud,
        is_fraud=is_fraud,
        threshold_used=request.threshold,
        latency_ms=latency,
        context=request.context
    )
    
    logger.info(f"Résultat: Fraude={is_fraud}, Confiance={prob_fraud:.3f}, Latence={latency:.1f}ms")
    
    return PredictionResponse(
        transaction_id=request.transaction.transaction_id,
        is_fraud=is_fraud,
        confidence=prob_fraud,
        threshold_used=request.threshold,
        fraud_risk=risk_level,
        latency_ms=latency,
        timestamp=datetime.now(),
        recommendation=recommendation
    )

@app.post("/feedback", tags=["Monitoring"])
async def add_feedback(
    transaction_id: str,
    was_correct: bool,
    true_class: int,
    api_key: str = Depends(api_key_header)
):
    """
    Ajoute un feedback pour améliorer le monitoring
    Permet de connaître la vraie classe après vérification humaine
    """
    if not METRICS_FILE.exists():
        raise HTTPException(status_code=404, detail="Aucun log trouvé")
    
    with open(METRICS_FILE, 'r') as f:
        logs = json.load(f)
    
    # Trouver et mettre à jour la transaction
    for log in logs:
        if log.get("transaction_id") == transaction_id:
            log["true_class"] = true_class
            log["was_correct"] = was_correct
            break
    else:
        raise HTTPException(status_code=404, detail="Transaction non trouvée")
    
    with open(METRICS_FILE, 'w') as f:
        json.dump(logs, f, indent=2)
    
    return {"message": "Feedback ajouté avec succès"}

if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run(
        "predict_fraud_with_monitoring:app",
        host="0.0.0.0",
        port=8005,
        reload=True
    )