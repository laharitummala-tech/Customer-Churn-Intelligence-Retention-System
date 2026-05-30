import shap
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "artifacts" / "churn_model.pkl"
model = joblib.load(MODEL_PATH)

explainer = shap.TreeExplainer(model)


def clean_feature_name(feature):
    feature = feature.replace("_", " ")

    if feature.startswith("Gender "):
        return "Gender: " + feature.replace("Gender ", "")

    if feature.startswith("Country "):
        return "Country: " + feature.replace("Country ", "")

    if feature.startswith("Signup Quarter "):
        return "Signup Quarter: " + feature.replace("Signup Quarter ", "")

    if feature == "City freq":
        return "City frequency"

    return feature


def extract_top_reasons_from_shap(churn_shap_values, feature_names, top_n=3):
    cleaned_names = np.array([clean_feature_name(col) for col in feature_names])
    reasons_list = []

    for row_values in churn_shap_values:
        positive_idx = np.where(row_values > 0)[0]

        if len(positive_idx) == 0:
            reasons_list.append("No strong churn-increasing reason")
            continue

        positive_values = row_values[positive_idx]

        top_idx = positive_idx[
            np.argsort(positive_values)[::-1][:top_n]
        ]

        top_reasons = cleaned_names[top_idx]
        reasons_list.append(", ".join(top_reasons))

    return reasons_list


def get_top_churn_reasons_batched(processed_df, top_n=3, batch_size=200):
    all_reasons = []
    feature_names = processed_df.columns

    for start in range(0, len(processed_df), batch_size):
        batch_df = processed_df.iloc[start:start + batch_size]

        shap_values = explainer.shap_values(
            batch_df,
            check_additivity=False
        )

        if isinstance(shap_values, list):
            churn_shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            churn_shap_values = shap_values[:, :, 1]
        else:
            churn_shap_values = shap_values

        batch_reasons = extract_top_reasons_from_shap(
            churn_shap_values,
            feature_names,
            top_n=top_n
        )

        all_reasons.extend(batch_reasons)

    return all_reasons