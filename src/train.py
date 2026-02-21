import argparse
import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.football_model import FootballModel
from src.models.tennis_model import TennisModel
from src.models.basketball_model import BasketballModel

# Charger config
with open(Path(__file__).parent.parent / "config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_map = {
    "football": FootballModel,
    "tennis": TennisModel,
    "basketball": BasketballModel
}

def main():
    parser = argparse.ArgumentParser(description="Entraîne un modèle pour un sport")
    parser.add_argument("--sport", choices=["football", "tennis", "basketball"], required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # Chemin data processed (assume préprocessé)
    data_path = Path("data/processed") / f"{args.sport}_features.csv"
    if not data_path.exists():
        print(f"Erreur: {data_path} n'existe pas. Lance preprocess.py d'abord.")
        return

    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1).values
    y = df["target"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ModelClass = model_map[args.sport]
    model = ModelClass(input_dim=X.shape[1])
    model.train(X_train, y_train, args.epochs, args.batch_size)

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy pour {args.sport}: {acc:.4f}")

    save_path = Path("models") / f"{args.sport}_model.h5"
    model.save(save_path)
    print(f"Modèle sauvegardé : {save_path}")

if __name__ == "__main__":
    main()
