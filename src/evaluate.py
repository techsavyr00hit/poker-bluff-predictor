"""Evaluate the saved model on a fresh test split."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from features import split_features_target

DATA_PATH = Path("data/poker_hands.csv")
MODEL_PATH = Path("models/bluff_model.joblib")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python src/train.py` first.")

    df = pd.read_csv(DATA_PATH)
    x, y = split_features_target(df)
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    print("Classification report")
    print(classification_report(y_test, predictions, target_names=["value", "bluff"]))
    print("Confusion matrix")
    print(confusion_matrix(y_test, predictions))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")


if __name__ == "__main__":
    main()
