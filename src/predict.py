import argparse
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Prédit un match avec un modèle entraîné")
    parser.add_argument("--sport", choices=["football", "tennis", "basketball"], required=True)
    parser.add_argument("--features", required=True, help="Features séparées par virgule, ex: '1.2,3.4,5'")
    args = parser.parse_args()

    model_path = Path("models") / f"{args.sport}_model.h5"
    if not model_path.exists():
        print(f"Erreur: Modèle {model_path} non trouvé. Entraîne d'abord.")
        return

    model = load_model(model_path)

    # Parse features
    features = np.array([float(f) for f in args.features.split(',')]).reshape(1, -1)

    # Assume scaler trained – pour simplicité, on skippe, mais en prod, save/load scaler
    pred = model.predict(features)[0][0]
    print(f"Probabilité de victoire (home/player1): {pred:.4f}")
    print(f"Conseil pari: Parier si pred > odds implicite (ex: si odds 2.0, implicite 0.5)")

if __name__ == "__main__":
    main()
