import pickle
import lightgbm as lgb
import numpy as np

from glob import glob

# Test input dummy
X = np.array([[0.1] * 52])

print("\n=== Testing Each Model Individually ===")

model_paths = glob("models/*")

for path in model_paths:
    print(f"\nTesting: {path}")
    try:
        if path.endswith(".txt"):
            model = lgb.Booster(model_file=path)
            preds = model.predict(X)
            print("OK → LightGBM output:", preds[:5])
        else:
            with open(path, "rb") as f:
                model = pickle.load(f)
            preds = model.predict_proba(X)
            print("OK → proba shape:", preds.shape)
    except Exception as e:
        print("❌ ERROR →", e)
