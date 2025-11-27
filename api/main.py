"""
Email Priority Classification API with LightGBM
Run: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pickle
import json
import pandas as pd
import lightgbm as lgb
import os
from pathlib import Path
from datetime import datetime

# ============================================
# 1. CONFIGURATION & PATHS
# ============================================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
STATIC_DIR = BASE_DIR / "static"

# Create directories if not exist
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ============================================
# 2. LightGBM Wrapper Class
# ============================================
class LightGBMWrapper:
    """Wrapper untuk LightGBM Booster agar kompatibel dengan pickle"""
    def __init__(self, booster):
        self.booster = booster
    
    def predict(self, X):
        """Prediksi kelas"""
        return self.booster.predict(X)
    
    def predict_proba(self, X):
        """Prediksi probabilitas"""
        preds = self.booster.predict(X)
        
        if preds.ndim == 1:
            # Convert to probabilities
            num_classes = 4
            prob_oh = np.zeros((len(preds), num_classes))
            prob_oh[np.arange(len(preds)), preds.astype(int)] = 1
            preds = prob_oh
        
        # Normalize to probabilities if needed
        if preds.max() > 1.0 or preds.min() < 0:
            preds = np.exp(preds) / np.exp(preds).sum(axis=1, keepdims=True)
        
        return preds

# ============================================
# 3. Pydantic Models
# ============================================
class PredictionInput(BaseModel):
    """Model untuk single prediction"""
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
    """Model output prediksi"""
    prediction_class: int
    prediction_label: str
    probabilities: Dict[str, float]
    confidence: float

class BatchInput(BaseModel):
    """Model untuk batch prediction"""
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
    """Model output batch prediction"""
    predictions: List[Dict]
    total: int
    timestamp: str

class ModelHealthCheck(BaseModel):
    """Model status kesehatan API"""
    status: str
    message: str
    models_loaded: int
    available_models: List[str]
    timestamp: str

# ============================================
# 4. Load Models & Metadata
# ============================================
def load_metadata():
    """Load metadata dari file atau set default"""
    meta_path = OUTPUT_DIR / "ensemble_metadata.json"
    try:
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        print("✅ Metadata loaded successfully")
        return metadata
    except FileNotFoundError:
        print(f"⚠️ Warning: Metadata not found at {meta_path}")
        default_metadata = {
            'label_mapping': {
                '0': "Low",
                '1': "Medium",
                '2': "High",
                '3': "Urgent"
            },
            'created_date': datetime.now().isoformat()
        }
        # Coba simpan default metadata
        try:
            with open(meta_path, "w") as f:
                json.dump(default_metadata, f, indent=2)
        except:
            pass
        return default_metadata

def load_models():
    """Load semua model yang tersedia"""
    models = {}
    
    model_files = {
        'Naive Bayes': MODEL_DIR / 'model_nb.pkl',
        'Random Forest': MODEL_DIR / 'model_rf.pkl',
        'LightGBM': MODEL_DIR / 'model_lgbm.txt',
        'SVM': MODEL_DIR / 'model_svm.pkl',
        'XGBoost': MODEL_DIR / 'model_xgb.pkl'
    }
    
    for model_name, model_path in model_files.items():
        try:
            if model_name == 'LightGBM':
                if model_path.exists():
                    lgbm_model = lgb.Booster(model_file=str(model_path))
                    models[model_name] = LightGBMWrapper(lgbm_model)
                    print(f"✅ {model_name} loaded from {model_path}")
                else:
                    print(f"⚠️ {model_name} not found at {model_path}")
            else:
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        models[model_name] = pickle.load(f)
                    print(f"✅ {model_name} loaded from {model_path}")
                else:
                    print(f"⚠️ {model_name} not found at {model_path}")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {str(e)}")
    
    return models

def load_ensemble():
    """Load ensemble model"""
    ensemble_path = MODEL_DIR / 'ensemble_model.pkl'
    try:
        if ensemble_path.exists():
            with open(ensemble_path, 'rb') as f:
                ensemble_model = pickle.load(f)
            print(f"✅ Ensemble model loaded from {ensemble_path}")
            return ensemble_model
        else:
            print(f"⚠️ Ensemble model not found at {ensemble_path}")
            return None
    except Exception as e:
        print(f"⚠️ Error loading ensemble model: {str(e)}")
        return None

# Load semua model pada startup
metadata = load_metadata()
label_mapping = metadata.get('label_mapping', {
    '0': "Low",
    '1': "Medium",
    '2': "High",
    '3': "Urgent"
})
inv_label_mapping = {v: k for k, v in label_mapping.items()}

individual_models = load_models()
ensemble_model = load_ensemble()

print(f"\n📦 Models available: {list(individual_models.keys())}")
print(f"📊 Label mapping: {label_mapping}")

# ============================================
# 5. Helper Functions
# ============================================
def is_voting_fitted(model):
    """Check jika VotingClassifier sudah fitted"""
    try:
        getattr(model, "estimators_")
        return True
    except AttributeError:
        return False

def get_predictions(X: np.ndarray) -> np.ndarray:
    """Get probabilitas prediksi dari ensemble atau individual models"""
    
    if ensemble_model is not None and is_voting_fitted(ensemble_model):
        print("Using ensemble model for prediction")
        return ensemble_model.predict_proba(X)
    
    if len(individual_models) == 0:
        raise HTTPException(
            status_code=500,
            detail="No models available for prediction"
        )
    
    print(f"Using {len(individual_models)} individual models for soft voting")
    all_probs = []
    
    for model_name, model in individual_models.items():
        try:
            model_probs = model.predict_proba(X)
            all_probs.append(model_probs)
            print(f"  ✅ {model_name} prediction shape: {model_probs.shape}")
        except Exception as e:
            print(f"  ⚠️ Error with {model_name}: {e}")
    
    if len(all_probs) == 0:
        raise HTTPException(
            status_code=500,
            detail="All models failed to generate predictions"
        )
    
    # Soft voting: average probabilities
    probs = np.mean(all_probs, axis=0)
    return probs

def validate_features(X: np.ndarray, expected_features: int = 50):
    """Validate shape dari features"""
    if X.shape[-1] != expected_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_features} features, got {X.shape[-1]}"
        )

# ============================================
# 6. Create FastAPI Application
# ============================================
app = FastAPI(
    title="Email Priority Classification API",
    description="Machine Learning API for email priority prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
except:
    print("⚠️ Static directory not found")

# ============================================
# 7. API ENDPOINTS - ROOT & INFO
# ============================================
@app.get("/")
def root():
    """Root endpoint dengan informasi API"""
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
            "predict_batch": "/predict_batch (POST)"
        }
    }

@app.get("/ui")
def serve_ui():
    """Serve web UI"""
    ui_path = BASE_DIR / "ui.html"
    if ui_path.exists():
        return FileResponse(str(ui_path), media_type="text/html")
    return {
        "message": "UI not found. Create ui.html in project root",
        "api_docs": "/docs"
    }

@app.get("/health", response_model=ModelHealthCheck)
def health_check():
    """Health check endpoint"""
    return ModelHealthCheck(
        status="healthy",
        message="API is running successfully",
        models_loaded=len(individual_models) + (1 if ensemble_model else 0),
        available_models=list(individual_models.keys()),
        timestamp=datetime.now().isoformat()
    )

@app.get("/model_info")
def get_model_info():
    """Get detailed model information"""
    return {
        "ensemble_method": "Soft Voting (Average Probabilities)",
        "total_models_loaded": len(individual_models) + (1 if ensemble_model else 0),
        "individual_models": list(individual_models.keys()),
        "ensemble_available": ensemble_model is not None,
        "ensemble_fitted": is_voting_fitted(ensemble_model) if ensemble_model else False,
        "label_mapping": label_mapping,
        "expected_features": 50,
        "supported_predictions": ["/predict", "/predict_batch"],
        "timestamp": datetime.now().isoformat()
    }

# ============================================
# 8. API ENDPOINTS - PREDICTIONS
# ============================================
@app.post("/predict", response_model=PredictionOutput)
def predict_priority(input_data: PredictionInput):
    """Prediksi prioritas email dari single sample"""
    try:
        X = np.array([input_data.features])
        validate_features(X)
        
        probs = get_predictions(X)[0]
        
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
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict_batch", response_model=BatchOutput)
def predict_batch(batch: BatchInput):
    """Prediksi prioritas email dari multiple samples"""
    try:
        X = np.array(batch.features_list)
        
        if X.ndim != 2:
            raise HTTPException(
                status_code=400,
                detail="features_list must be 2-dimensional array"
            )
        
        validate_features(X)
        
        print(f"Processing batch of {X.shape[0]} samples")
        probs = get_predictions(X)
        pred_classes = np.argmax(probs, axis=1)
        
        results = []
        for i in range(len(X)):
            pred_class = int(pred_classes[i])
            prob_dict = {
                label_mapping.get(str(j), f"Class_{j}"): float(probs[i][j])
                for j in range(len(probs[i]))
            }
            results.append({
                "sample_index": i,
                "prediction_class": pred_class,
                "prediction_label": label_mapping.get(str(pred_class), f"Class_{pred_class}"),
                "probabilities": prob_dict,
                "confidence": float(np.max(probs[i]))
            })
        
        return BatchOutput(
            predictions=results,
            total=len(results),
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )

# ============================================
# 9. API ENDPOINTS - FILE UPLOAD (OPTIONAL)
# ============================================
@app.post("/predict_from_csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """Prediksi dari CSV file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="File must be CSV format"
            )
        
        contents = await file.read()
        df = pd.read_csv(__import__('io').StringIO(contents.decode('utf-8')))
        
        # Validasi kolom
        if df.shape[1] != 50:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 50 columns, got {df.shape[1]}"
            )
        
        X = np.array(df.values)
        probs = get_predictions(X)
        pred_classes = np.argmax(probs, axis=1)
        
        results = []
        for i in range(len(X)):
            pred_class = int(pred_classes[i])
            results.append({
                "row": i,
                "prediction_class": pred_class,
                "prediction_label": label_mapping.get(str(pred_class), f"Class_{pred_class}"),
                "confidence": float(np.max(probs[i]))
            })
        
        return {
            "total_predictions": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ CSV prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"CSV prediction error: {str(e)}"
        )

# ============================================
# 10. API ENDPOINTS - UTILITY
# ============================================
@app.get("/status")
def get_status():
    """Get detailed API status"""
    return {
        "api_running": True,
        "models_status": {
            "individual_models": {
                name: "loaded" for name in individual_models.keys()
            },
            "ensemble_model": "loaded" if ensemble_model else "not_loaded",
            "total_loaded": len(individual_models) + (1 if ensemble_model else 0)
        },
        "label_mapping": label_mapping,
        "feature_count": 50,
        "class_count": len(label_mapping),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metadata")
def get_metadata():
    """Get model metadata"""
    return metadata

# ============================================
# 11. ERROR HANDLERS
# ============================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": True,
        "status_code": exc.status_code,
        "detail": exc.detail,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    print(f"❌ Unhandled exception: {str(exc)}")
    return {
        "error": True,
        "status_code": 500,
        "detail": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }

# ============================================
# 12. STARTUP EVENT
# ============================================
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("\n" + "="*50)
    print("🚀 Email Priority Classification API")
    print("="*50)
    print(f"✅ API Started at {datetime.now().isoformat()}")
    print(f"📦 Models loaded: {len(individual_models)}")
    print(f"📊 Ensemble model: {'Yes' if ensemble_model else 'No'}")
    print(f"🏷️  Classes: {list(label_mapping.values())}")
    print("="*50 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )