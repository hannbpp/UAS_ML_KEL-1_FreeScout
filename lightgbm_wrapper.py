import numpy as np

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
