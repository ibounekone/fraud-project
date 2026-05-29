"""
API DÉTECTION DE FRAUDE - VERSION FINALE STABLE
Utilise XGBoost avec seuil paramétrable
"""

from fastapi import FastAPI, HTTPException, Depends # type: ignore
from fastapi.security import APIKeyHeader # type: ignore
from pydantic import BaseModel, Field # type: ignore
from datetime import datetime
import numpy as np
import joblib # type: ignore
import time
import logging
from pathlib import Path
from typing import Optional

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration API
API_KEY = "fraud_detection_2024_secret_key"
API_KEY_NAME = "X-API-Key"

app = FastAPI(title="API Détection de Fraude Bancaire (Final)")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Modèles de données
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

class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    is_fraud: bool
    confidence: float
    threshold_used: float
    latency_ms: float
    timestamp: datetime

# Chargement du modèle
def load_model():
    """Charge le modèle XGBoost"""
    model_paths = [
        Path("models/xgboost_model.pkl"),
        Path("models/xgboost_model_fixed.pkl")
    ]
    
    for path in model_paths:
        if path.exists():
            try:
                model = joblib.load(path)
                logger.info(f"✅ Modèle chargé depuis {path}")
                return model
            except Exception as e:
                logger.error(f"Erreur chargement {path}: {e}")
    
    logger.error("❌ Aucun modèle trouvé!")
    return None

# Charger au démarrage
model = load_model()

if model is None:
    logger.warning("⚠️ Modèle non chargé, utilisation d'un modèle factice")
    from sklearn.linear_model import LogisticRegression # type: ignore
    model = LogisticRegression()

def extract_features(transaction: Transaction) -> np.ndarray:
    """Extrait les features dans l'ordre attendu par le modèle"""
    features = np.array([
        transaction.V1, transaction.V2, transaction.V3, transaction.V4, transaction.V5,
        transaction.V6, transaction.V7, transaction.V8, transaction.V9, transaction.V10,
        transaction.V11, transaction.V12, transaction.V13, transaction.V14, transaction.V15,
        transaction.V16, transaction.V17, transaction.V18, transaction.V19, transaction.V20,
        transaction.V21, transaction.V22, transaction.V23, transaction.V24, transaction.V25,
        transaction.V26, transaction.V27, transaction.V28, transaction.Amount
    ])
    return features.reshape(1, -1)

# Endpoints
@app.get("/")
async def root():
    return {"message": "API Détection de Fraude", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    api_key: str = Depends(api_key_header)
):
    start_time = time.time()
    
    # Extraction des features
    features = extract_features(request.transaction)
    
    # Prédiction
    proba = model.predict_proba(features)[0][1]
    is_fraud = proba >= request.threshold
    
    latency = (time.time() - start_time) * 1000
    
    return PredictionResponse(
        transaction_id=request.transaction.transaction_id,
        is_fraud=is_fraud,
        confidence=proba,
        threshold_used=request.threshold,
        latency_ms=latency,
        timestamp=datetime.now()
    )

if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run(
        "predict_fraud_final:app",
        host="0.0.0.0",
        port=8005,
        reload=True
    )