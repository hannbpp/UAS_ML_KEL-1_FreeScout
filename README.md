# 🚀 ML API Project - Email Priority Classification

Machine Learning API untuk klasifikasi prioritas email menggunakan ensemble methods.

## 📊 Project Structure

```
ML-API-PROJECT/
├── models/                    # Model files (5 models)
│   ├── ensemble_metadata.json
│   ├── ensemble_model.pkl
│   ├── model_lgbm.txt
│   ├── model_nb.pkl
│   ├── model_rf.pkl
│   ├── model_svm.pkl
│   └── model_xgb.pkl
├── output/                    # Results & metadata
│   └── ensemble_metadata.json
├── main.py                    # FastAPI application
├── lightgbm_wrapper.py        # LightGBM wrapper class
├── test_models.py             # Model testing script
└── ui.html                    # Web UI
```

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"
```

## 🚀 Run API

```bash
# Development
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

- `POST /predict_from_text` - Predict from raw email text
- `POST /predict_batch_text` - Batch prediction
- `GET /health` - Health check
- `GET /ui` - Web interface

## 👥 Team - Kelompok 1

Universitas Tanjungpura - Machine Learning

## 📄 License

MIT License

```

---

### **STEP 3: Buat requirements.txt**

Buat file `requirements.txt`:
```

fastapi==0.118.3
uvicorn[standard]==0.38.0
python-multipart==0.0.20
pydantic==2.10.6
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
lightgbm==4.5.0
xgboost==2.2.0
Sastrawi==1.0.1
nltk==3.9.1
