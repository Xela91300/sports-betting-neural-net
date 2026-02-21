import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from pathlib import Path
import yaml

# ────────────────────────────────────────────────
# Configuration & chemins
# ────────────────────────────────────────────────
MODELS_DIR = Path("models")
CONFIG_PATH = Path("config/config.yaml")

# Charger la config YAML
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Mapping sport → modèle & features attendues
SPORT_CONFIG = {
    "Football": {
        "model_path": MODELS_DIR / "football_model.h5",
        "features": config["football"]["features"],
        "desc": "Victoire de l'équipe à domicile"
    },
    "Tennis": {
        "model_path": MODELS_DIR / "tennis_model.h5",
        "features": config["tennis"]["features"],
        "desc": "Victoire du joueur 1"
    },
    "Basketball": {
        "model_path": MODELS_DIR / "basketball_model.h5",
        "features": config["basketball"]["features"],
        "desc": "Victoire de l'équipe à domicile"
    }
}

# ────────────────────────────────────────────────
# Cache pour charger les modèles (très important pour perf)
# ────────────────────────────────────────────────
@st.cache_resource
def load_cached_model(sport):
    path = SPORT_CONFIG[sport]["model_path"]
    if not path.exists():
        st.error(f"Modèle pour {sport} introuvable : {path}")
        return None
    return load_model(path)

# ────────────────────────────────────────────────
# Interface principale
# ────────────────────────────────────────────────
st.title("Prédictions Paris Sportifs – Réseaux de Neurones")
st.markdown("Choisis un sport, remplis les features et obtiens la probabilité estimée.")

sport = st.selectbox("Sport", ["Football", "Tennis", "Basketball"])

if sport:
    model = load_cached_model(sport)
    if model is None:
        st.stop()

    st.subheader(f"Prédiction pour {sport} ({SPORT_CONFIG[sport]['desc']})")

    # Créer un dictionnaire pour stocker les inputs utilisateur
    user_input = {}

    # Générer des inputs dynamiques selon les features du sport
    for feat in SPORT_CONFIG[sport]["features"]:
        # Valeurs par défaut raisonnables (à affiner selon tes données normalisées)
        default_val = 0.0
        if "form" in feat or "avg" in feat:
            default_val = 2.5
        elif "rank_diff" in feat or "diff" in feat:
            default_val = 0.0
        elif "odds" in feat:
            default_val = 2.0
        elif "surface" in feat:
            user_input[feat] = st.checkbox(f"{feat} (coché = True)", value=False)
            continue

        user_input[feat] = st.number_input(
            label=feat,
            value=default_val,
            step=0.1,
            format="%.2f",
            help=f"Valeur pour {feat} (normalisée ou brute selon ton prétraitement)"
        )

    if st.button("Prédire", type="primary"):
        try:
            # Convertir en array numpy (ordre des features doit matcher l'entraînement !)
            features_list = []
            for feat in SPORT_CONFIG[sport]["features"]:
                if feat in user_input:
                    features_list.append(user_input[feat])
                else:
                    features_list.append(0.0)  # fallback

            X = np.array(features_list).reshape(1, -1)

            # Prédiction
            proba = model.predict(X, verbose=0)[0][0]

            st.success(f"Probabilité estimée : **{proba:.2%}**")

            # Conseil pari très simple (à améliorer avec vraies cotes)
            if proba > 0.65:
                st.balloons()
                st.info("→ **Valeur potentielle** : Parier si cote > 1 / proba")
            elif proba > 0.50:
                st.info("→ Match équilibré – prudence")
            else:
                st.warning("→ Faible probabilité – éviter ou chercher value côté opposé")

            # Afficher les inputs pour vérif
            st.markdown("**Inputs utilisés :**")
            st.json(user_input)

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# Footer
st.markdown("---")
st.caption("Projet éducatif – Pas de garantie de profit. Joue responsablement.")
