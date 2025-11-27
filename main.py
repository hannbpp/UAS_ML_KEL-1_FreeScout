# ============================================
# EMAIL PRIORITY CLASSIFICATION API WITH UI
# Run: uvicorn main:app --reload
# ============================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import numpy as np
import pickle
import json
import pandas as pd
import lightgbm as lgb
import os

# ============================================
# 1. LightGBMWrapper Class
# ============================================
class LightGBMWrapper:
    """Wrapper untuk LightGBM Booster agar kompatibel dengan pickle"""
    def __init__(self, booster):
        self.booster = booster
    
    def predict(self, X):
        return self.booster.predict(X)
    
    def predict_proba(self, X):
        preds = self.booster.predict(X)
        if preds.ndim == 1:
            num_classes = 4
            prob_oh = np.zeros((len(preds), num_classes))
            prob_oh[np.arange(len(preds)), preds.astype(int)] = 1
            preds = prob_oh
        return preds

# ============================================
# 2. Pydantic Models
# ============================================
class PredictionInput(BaseModel):
    features: List[float] = Field(
        ...,
        description="List of 50 PCA features",
        min_items=50,
        max_items=50
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.1] * 50
            }
        }

class PredictionOutput(BaseModel):
    prediction_class: int
    prediction_label: str
    probabilities: Dict[str, float]
    confidence: float

class BatchInput(BaseModel):
    features_list: List[List[float]] = Field(
        ...,
        description="List of feature arrays (each with 50 PCA features)",
        min_items=1
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features_list": [
                    [0.1] * 50,
                    [0.2] * 50
                ]
            }
        }

class BatchOutput(BaseModel):
    predictions: List[Dict]
    total: int

class FlexiblePredictionInput(BaseModel):
    features_list: List[List[float]] = Field(
        ...,
        description="List of feature arrays (each with 50 PCA features)",
        min_items=1,
        max_items=1
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "features_list": [
                    [0.1] * 50
                ]
            }
        }

# ============================================
# 3. Load Models & Metadata
# ============================================
MODEL_PATH = "./models/"
OUTPUT_PATH = "./output/"

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load metadata
META_PATH = OUTPUT_PATH + 'ensemble_metadata.json'
try:
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    print("✅ Metadata loaded successfully")
except FileNotFoundError:
    print(f"⚠️ Warning: Metadata not found at {META_PATH}")
    metadata = {'label_mapping': {'0': "Low", '1': "Medium", '2': "High", '3': "Urgent"}}

label_mapping = metadata.get('label_mapping', {'0': "Low", '1': "Medium", '2': "High", '3': "Urgent"})
inv_label_mapping = {v: k for k, v in label_mapping.items()}

# Model paths
model_files = {
    'Naive Bayes': MODEL_PATH + 'model_nb.pkl',
    'Random Forest': MODEL_PATH + 'model_rf.pkl',
    'LightGBM': MODEL_PATH + 'model_lgbm.txt',
    'SVM': MODEL_PATH + 'model_svm.pkl',
    'XGBoost': MODEL_PATH + 'model_xgb.pkl'
}

# Load individual models
individual_models = {}
for model_name, model_path in model_files.items():
    try:
        if model_name == 'LightGBM':
            if os.path.exists(model_path):
                lgbm_model = lgb.Booster(model_file=model_path)
                individual_models[model_name] = LightGBMWrapper(lgbm_model)
                print(f"✅ {model_name} loaded")
        else:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    individual_models[model_name] = pickle.load(f)
                print(f"✅ {model_name} loaded")
            else:
                print(f"⚠️ {model_name} not found at {model_path}")
    except Exception as e:
        print(f"❌ Error loading {model_name}: {str(e)}")

# Load ensemble model (optional)
ensemble_model = None
ensemble_path = MODEL_PATH + 'ensemble_model.pkl'
try:
    if os.path.exists(ensemble_path):
        with open(ensemble_path, 'rb') as f:
            ensemble_model = pickle.load(f)
        print(f"✅ Ensemble model loaded")
except Exception as e:
    print(f"⚠️ Ensemble model not available: {str(e)}")

print(f"\n📦 Models available: {list(individual_models.keys())}")

# ============================================
# 3B. Check if VotingClassifier is fitted
# ============================================
def is_voting_fitted(model):
    try:
        getattr(model, "estimators_")
        return True
    except AttributeError:
        return False

# ============================================
# 4. Create FastAPI Application
# ============================================
app = FastAPI(
    title="Email Priority Classification API",
    description="Machine Learning API for email priority prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 5. API ENDPOINTS
# ============================================

@app.get("/")
def root():
    return {
        "message": "Welcome to Email Priority Classification API",
        "version": "1.0.0",
        "ui_url": "http://localhost:8000/ui",
        "documentation": "/docs",
        "endpoints": {
            "ui": "/ui (GET)",
            "health": "/health (GET)",
            "model_info": "/model_info (GET)",
            "predict": "/predict (POST)",
            "predict_single": "/predict_single (POST)",
            "predict_batch": "/predict_batch (POST)"
        }
    }

# ============================================
# SERVE UI (NEW)
# ============================================
@app.get("/ui")
def serve_ui():
    """Serve the web UI"""
    return FileResponse("ui.html", media_type="text/html")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "API is running successfully",
        "models_loaded": len(individual_models) + (1 if ensemble_model else 0),
        "available_models": list(individual_models.keys()),
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/model_info")
def get_model_info():
    return {
        "ensemble_method": "Soft Voting",
        "total_models": len(individual_models) + (1 if ensemble_model else 0),
        "individual_models": list(individual_models.keys()),
        "ensemble_available": ensemble_model is not None,
        "label_mapping": label_mapping,
        "expected_features": 50
    }

# ============================================
# PREDICT (Single)
# ============================================
@app.post("/predict", response_model=PredictionOutput)
def predict_priority(input_data: PredictionInput):
    try:
        X = np.array([input_data.features])

        if X.shape[1] != 50:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 50 features, got {X.shape[1]}"
            )

        if ensemble_model is not None and is_voting_fitted(ensemble_model):
            probs = ensemble_model.predict_proba(X)[0]
        else:
            if len(individual_models) == 0:
                raise HTTPException(status_code=500, detail="No models available")

            all_probs = []
            for model_name, model in individual_models.items():
                try:
                    model_probs = model.predict_proba(X)[0]
                    all_probs.append(model_probs)
                except Exception as e:
                    print(f"⚠️ Error with {model_name}: {e}")

            if len(all_probs) == 0:
                raise HTTPException(status_code=500, detail="All models failed")

            probs = np.mean(all_probs, axis=0)

        pred_class = int(np.argmax(probs))
        pred_label = label_mapping.get(str(pred_class), f"Class_{pred_class}")
        confidence = float(np.max(probs))

        prob_dict = {
            label_mapping.get(str(i), f"Class_{i}"): float(probs[i])
            for i in range(len(probs))
        }

        return PredictionOutput(
            prediction_class=pred_class,
            prediction_label=pred_label,
            probabilities=prob_dict,
            confidence=confidence
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ============================================
# PREDICT (Single with features_list)
# ============================================
@app.post("/predict_single", response_model=PredictionOutput)
def predict_single(input_data: FlexiblePredictionInput):
    try:
        if len(input_data.features_list) != 1:
            raise HTTPException(
                status_code=400,
                detail="features_list harus berisi exactly 1 sample"
            )
        
        X = np.array(input_data.features_list)

        if X.shape[1] != 50:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 50 features, got {X.shape[1]}"
            )

        if ensemble_model is not None and is_voting_fitted(ensemble_model):
            probs = ensemble_model.predict_proba(X)[0]
        else:
            if len(individual_models) == 0:
                raise HTTPException(status_code=500, detail="No models available")

            all_probs = []
            for model_name, model in individual_models.items():
                try:
                    model_probs = model.predict_proba(X)[0]
                    all_probs.append(model_probs)
                except Exception as e:
                    print(f"⚠️ Error with {model_name}: {e}")

            if len(all_probs) == 0:
                raise HTTPException(status_code=500, detail="All models failed")

            probs = np.mean(all_probs, axis=0)

        pred_class = int(np.argmax(probs))
        pred_label = label_mapping.get(str(pred_class), f"Class_{pred_class}")
        confidence = float(np.max(probs))

        prob_dict = {
            label_mapping.get(str(i), f"Class_{i}"): float(probs[i])
            for i in range(len(probs))
        }

        return PredictionOutput(
            prediction_class=pred_class,
            prediction_label=pred_label,
            probabilities=prob_dict,
            confidence=confidence
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ============================================
# PREDICT BATCH
# ============================================
@app.post("/predict_batch", response_model=BatchOutput)
def predict_batch(batch: BatchInput):
    try:
        X = np.array(batch.features_list)

        if X.ndim != 2:
            raise HTTPException(status_code=400, detail="features_list harus 2 dimensi")

        if X.shape[1] != 50:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 50 features per item, got {X.shape[1]}"
            )

        if ensemble_model is not None and is_voting_fitted(ensemble_model):
            probs = ensemble_model.predict_proba(X)
        else:
            all_probs = []
            for model_name, model in individual_models.items():
                try:
                    model_probs = model.predict_proba(X)
                    all_probs.append(model_probs)
                except Exception as e:
                    print(f"⚠️ Error with {model_name}: {e}")

            if len(all_probs) == 0:
                raise HTTPException(status_code=500, detail="All models failed")

            probs = np.mean(all_probs, axis=0)

        pred_classes = np.argmax(probs, axis=1)

        results = []
        for i in range(len(X)):
            pred_class = int(pred_classes[i])
            prob_dict = {
                label_mapping.get(str(j), f"Class_{j}"): float(probs[i][j])
                for j in range(len(probs[i]))
            }
            results.append({
                "prediction_class": pred_class,
                "prediction_label": label_mapping.get(str(pred_class), f"Class_{pred_class}"),
                "probabilities": prob_dict,
                "confidence": float(np.max(probs[i]))
            })

        return BatchOutput(predictions=results, total=len(results))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")