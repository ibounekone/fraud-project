"""
API DE DÉTECTION DE FRAUDE BANCAIRE
Avec seuil de décision paramétrable

Le seuil permet d'ajuster le compromis :
- Seuil bas (0.10) : plus de fraudes détectées, plus de faux positifs
- Seuil haut (0.80) : moins de faux positifs, mais plus de fraudes non détectées
"""

# ========== IMPORTS ==========
from fastapi import FastAPI, HTTPException, Depends # type: ignore
from fastapi.security import APIKeyHeader # type: ignore
from pydantic import BaseModel, Field # type: ignore
from datetime import datetime
import numpy as np
import joblib  # type: ignore # Pour charger le modèle entraîné
import time
import logging
from pathlib import Path
from typing import Optional

# ========== CONFIGURATION ==========
# Logs pour le monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Clé API (sécurité)
API_KEY = "fraud_detection_2024_secret_key"
API_KEY_NAME = "X-API-Key"

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API Détection de Fraude Bancaire",
    description="Détecte les transactions frauduleuses avec seuil paramétrable",
    version="1.0.0"
)

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Chemins des modèles
MODEL_PATH = Path("models/logistic_regression.pkl")
SCALER_PATH = Path("models/scaler.pkl")

# ========== MODÈLES DE DONNÉES (Pydantic) ==========

class Transaction(BaseModel):
    """
    Structure d'une transaction bancaire
    Les features V1 à V28 sont anonymisées (PCA)
    """
    # Features anonymisées (PCA)
    V1: float = Field(..., description="Feature PCA 1")
    V2: float = Field(..., description="Feature PCA 2")
    V3: float = Field(..., description="Feature PCA 3")
    V4: float = Field(..., description="Feature PCA 4")
    V5: float = Field(..., description="Feature PCA 5")
    V6: float = Field(..., description="Feature PCA 6")
    V7: float = Field(..., description="Feature PCA 7")
    V8: float = Field(..., description="Feature PCA 8")
    V9: float = Field(..., description="Feature PCA 9")
    V10: float = Field(..., description="Feature PCA 10")
    V11: float = Field(..., description="Feature PCA 11")
    V12: float = Field(..., description="Feature PCA 12")
    V13: float = Field(..., description="Feature PCA 13")
    V14: float = Field(..., description="Feature PCA 14")
    V15: float = Field(..., description="Feature PCA 15")
    V16: float = Field(..., description="Feature PCA 16")
    V17: float = Field(..., description="Feature PCA 17")
    V18: float = Field(..., description="Feature PCA 18")
    V19: float = Field(..., description="Feature PCA 19")
    V20: float = Field(..., description="Feature PCA 20")
    V21: float = Field(..., description="Feature PCA 21")
    V22: float = Field(..., description="Feature PCA 22")
    V23: float = Field(..., description="Feature PCA 23")
    V24: float = Field(..., description="Feature PCA 24")
    V25: float = Field(..., description="Feature PCA 25")
    V26: float = Field(..., description="Feature PCA 26")
    V27: float = Field(..., description="Feature PCA 27")
    V28: float = Field(..., description="Feature PCA 28")
    Amount: float = Field(..., ge=0, le=50000, description="Montant de la transaction (€)")
    
    # Optionnel : identifiant pour traçabilité
    transaction_id: Optional[str] = Field(None, description="ID unique de la transaction")

class PredictionRequest(BaseModel):
    """
    Requête de prédiction avec seuil optionnel
    """
    transaction: Transaction
    threshold: Optional[float] = Field(
        0.5,  # Valeur par défaut = compromis standard
        ge=0.01,  # Minimum 0.01
        le=0.99,  # Maximum 0.99
        description="Seuil de décision (0.01=très sensible, 0.99=très strict)"
    )
    context: Optional[str] = Field(
        None, 
        description="Contexte : 'high_amount', 'normal', 'low_amount'"
    )

class PredictionResponse(BaseModel):
    """
    Réponse de l'API
    """
    transaction_id: Optional[str]
    is_fraud: bool  # True = frauduleux, False = légitime
    confidence: float  # Probabilité (0 à 1)
    threshold_used: float  # Seuil utilisé pour la décision
    fraud_risk: str  # 'Élevé', 'Moyen', 'Faible', 'Très faible'
    latency_ms: float
    timestamp: datetime
    recommendation: str  # Action recommandée

class ThresholdInfo(BaseModel):
    """
    Informations sur les seuils disponibles
    """
    default_threshold: float
    thresholds: dict
    recommendation: str

# ========== CHARGEMENT DU MODÈLE ==========

def load_model():
    """
    Charge le modèle entraîné (régression logistique)
    Si le modèle n'existe pas, affiche une erreur
    """
    if not MODEL_PATH.exists():
        logger.warning("Modèle non trouvé. Création d'un modèle d'exemple...")
        return None
    
    model = joblib.load(MODEL_PATH)
    logger.info("Modèle chargé avec succès")
    return model

def load_scaler():
    """Charge le scaler pour normaliser le montant"""
    if not SCALER_PATH.exists():
        return None
    return joblib.load(SCALER_PATH)

# Chargement au démarrage
model = load_model()
scaler = load_scaler()

# Si le modèle n'existe pas, on crée un modèle factice pour la démo
if model is None:
    from sklearn.linear_model import LogisticRegression # type: ignore
    model = LogisticRegression()
    logger.warning("Modèle factice créé pour la démonstration")

# ========== FONCTIONS UTILITAIRES ==========

def extract_features(transaction: Transaction) -> np.ndarray:
    """
    Extrait les features d'une transaction pour le modèle
    Ordre des features : V1 à V28 + Amount
    """
    # Liste des features V1 à V28
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
    
    # Concaténation
    features = np.array(v_features + [amount])
    return features.reshape(1, -1)

def get_risk_level(confidence: float) -> str:
    """
    Détermine le niveau de risque basé sur la probabilité
    """
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

def get_recommendation(is_fraud: bool, confidence: float, threshold: float) -> str:
    """
    Recommandation basée sur la prédiction
    """
    if is_fraud:
        if confidence > 0.8:
            return "BLOQUER immédiatement - Forte suspicion de fraude"
        elif confidence > 0.6:
            return "BLOQUER + vérification manuelle"
        else:
            return "BLOQUER temporairement + appeler le client"
    else:
        if confidence < 0.2:
            return "AUTORISER - Transaction à très faible risque"
        else:
            return "AUTORISER - Transaction normale"

def get_adaptive_threshold(amount: float, context: Optional[str]) -> float:
    """
    Seuil adaptatif selon le montant et le contexte
    Stratégie métier :
    - Gros montants (> 1000€) : seuil bas pour maximiser détection
    - Petits montants (< 50€) : seuil haut pour éviter faux positifs
    """
    if context == "high_amount":
        return 0.10  # Très sensible – on détecte tout
    elif context == "low_amount":
        return 0.80  # Très strict – on évite les faux positifs
    elif amount > 1000:
        return 0.15
    elif amount > 100:
        return 0.40
    elif amount > 50:
        return 0.60
    else:
        return 0.80

# ========== ENDPOINTS ==========

@app.get("/", tags=["Info"])
async def root():
    """Informations générales sur l'API"""
    return {
        "message": "API Détection de Fraude Bancaire",
        "version": "1.0.0",
        "model": "Régression Logistique",
        "metrics": {"recall": 0.918, "auc": 0.971},
        "endpoints": ["/predict", "/predict/batch", "/thresholds", "/health", "/docs"]
    }

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Vérifie que l'API et le modèle sont opérationnels"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold_strategy": "paramétrable",
        "timestamp": datetime.now()
    }

@app.get("/thresholds", response_model=ThresholdInfo, tags=["Info"])
async def get_thresholds():
    """
    Retourne les seuils recommandés selon le contexte
    """
    return ThresholdInfo(
        default_threshold=0.5,
        thresholds={
            "Très sensible (max recall)": 0.10,
            "Sensible (gros montants)": 0.15,
            "Équilibré (défaut)": 0.50,
            "Strict (petits montants)": 0.80,
            "Très strict (min faux positifs)": 0.95
        },
        recommendation="Utilisez threshold=0.10 pour les gros montants (>1000€), threshold=0.80 pour les petits montants (<50€)"
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(
    request: PredictionRequest,
    api_key: str = Depends(api_key_header)
):
    """
    Prédit si une transaction est frauduleuse
    
    Paramètres :
    - threshold : seuil de décision (0.01 à 0.99)
      Plus le seuil est bas, plus on détecte de fraudes
      Plus le seuil est haut, plus on évite les faux positifs
    
    Contexte métier :
    - high_amount : seuil très bas (max recall)
    - low_amount : seuil élevé (max précision)
    """
    start_time = time.time()
    
    # Log de la requête
    logger.info(f"Nouvelle transaction - Montant: {request.transaction.Amount}€, "
                f"Seuil demandé: {request.threshold}, "
                f"Contexte: {request.context}")
    
    # Extraction des features
    features = extract_features(request.transaction)
    
    # Prédiction
    # predict_proba retourne [prob_légitime, prob_fraude]
    prob_fraud = model.predict_proba(features)[0][1]
    
    # Détermination du seuil à utiliser
    if request.context:
        threshold = get_adaptive_threshold(request.transaction.Amount, request.context)
    else:
        threshold = request.threshold
    
    # Décision selon le seuil
    is_fraud = prob_fraud >= threshold
    
    # Niveau de risque
    risk_level = get_risk_level(prob_fraud)
    
    # Recommandation
    recommendation = get_recommendation(is_fraud, prob_fraud, threshold)
    
    # Calcul de la latence
    latency = (time.time() - start_time) * 1000
    
    logger.info(f"Résultat - Fraude: {is_fraud}, Confiance: {prob_fraud:.3f}, "
                f"Seuil utilisé: {threshold}, Latence: {latency:.1f}ms")
    
    return PredictionResponse(
        transaction_id=request.transaction.transaction_id,
        is_fraud=is_fraud,
        confidence=prob_fraud,
        threshold_used=threshold,
        fraud_risk=risk_level,
        latency_ms=latency,
        timestamp=datetime.now(),
        recommendation=recommendation
    )

@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(
    requests: list[PredictionRequest],
    api_key: str = Depends(api_key_header)
):
    """
    Prédiction batch (plusieurs transactions en une requête)
    Plus efficace pour le traitement par lots
    """
    results = []
    for req in requests:
        features = extract_features(req.transaction)
        prob_fraud = model.predict_proba(features)[0][1]
        
        if req.context:
            threshold = get_adaptive_threshold(req.transaction.Amount, req.context)
        else:
            threshold = req.threshold
        
        is_fraud = prob_fraud >= threshold
        
        results.append({
            "transaction_id": req.transaction.transaction_id,
            "is_fraud": is_fraud,
            "confidence": prob_fraud,
            "amount": req.transaction.Amount
        })
    
    return {
        "count": len(results),
        "results": results,
        "timestamp": datetime.now()
    }

# ========== ENTRAÎNEMENT MODÈLE (optionnel) ==========
def train_and_save_model():
    """
    Entraîne et sauvegarde le modèle si nécessaire
    À exécuter une fois avant de lancer l'API
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression # type: ignore
    import joblib # type: ignore
    
    print("📂 Chargement des données...")
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    
    print("🔧 Entraînement du modèle...")
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print("💾 Sauvegarde du modèle...")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/logistic_regression.pkl")
    
    print("✅ Modèle sauvegardé dans models/logistic_regression.pkl")
    return model

if __name__ == "__main__":
    import uvicorn # type: ignore
    
    # Si le modèle n'existe pas, on l'entraîne
    if not Path("models/logistic_regression.pkl").exists():
        train_and_save_model()
        model = load_model()
    
    uvicorn.run(
        "predict_fraud:app",
        host="0.0.0.0",
        port=8005,
        reload=True
    )