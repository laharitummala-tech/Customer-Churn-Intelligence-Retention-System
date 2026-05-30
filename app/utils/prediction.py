import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "artifacts" / "churn_model.pkl"

model = joblib.load(MODEL_PATH)

def predict_churn(df):
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    return predictions, probabilities

def risk_segment(prob):
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"