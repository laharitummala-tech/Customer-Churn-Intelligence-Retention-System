import joblib
from pathlib import Path

ARTIFACT_PATH = Path(__file__).parent.parent / "artifacts" / "preprocessor.pkl"

preprocessor = joblib.load(ARTIFACT_PATH)
print("Current file:", Path(__file__))
print("Model path:", ARTIFACT_PATH)
print("Exists:", ARTIFACT_PATH.exists())
def preprocess_data(df):
    return preprocessor.transform(df)