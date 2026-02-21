import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path

def main(sport, test_data_path, model_path):
    df = pd.read_csv(test_data_path)
    X = df.drop("target", axis=1).values
    y_true = df["target"].values

    model = load_model(model_path)
    y_pred = model.predict(X).flatten()

    acc = accuracy_score(y_true, y_pred > 0.5)
    auc = roc_auc_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    # Backtest simple: assume odds in df['odds_home']
    if 'odds_home' in df:
        value_bets = (y_pred > 1/df['odds_home']) & (y_true == 1)
        profit = value_bets.sum() * (df['odds_home'] - 1) - (~value_bets & (y_pred > 1/df['odds_home'])).sum()
        print(f"Profit simulé: {profit}")

if __name__ == "__main__":
    # Ex: python src/evaluate.py football data/processed/football_test.csv models/football_model.h5
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
