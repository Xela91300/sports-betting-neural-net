import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import joblib

# ────────────────────────────────────────────────
# Chemins
# ────────────────────────────────────────────────
ROOT_DIR          = Path(__file__).parent
MODELS_DIR        = ROOT_DIR / "models"
DATA_DIR          = ROOT_DIR / "src" / "data"
DATA_RAW_DIR      = DATA_DIR / "raw"
DATA_PROCESSED_DIR= DATA_DIR / "processed"
CONFIG_PATH       = ROOT_DIR / "config" / "config.yaml"

for directory in [MODELS_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Charger la configuration
config = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
else:
    st.warning("⚠️ config.yaml non trouvé → configuration par défaut utilisée")

# ────────────────────────────────────────────────
# Configuration des sports
# ────────────────────────────────────────────────
SPORT_CONFIG = {
    "Football": {
        "model_path": MODELS_DIR / "football_model.h5",
        "scaler_path": MODELS_DIR / "football_scaler.joblib",
        "features": config.get("football", {}).get("features", [
            "home_form_5", "away_form_5", "home_goals_avg", "away_goals_avg",
            "diff_classement", "odds_home", "odds_draw", "odds_away"
        ]),
        "desc": "Victoire de l'équipe à domicile",
        "type": "classification",
        "data_pattern": "*football*.csv"
    },
    "Tennis": {
        "model_path": MODELS_DIR / "tennis_model.h5",
        "scaler_path": MODELS_DIR / "tennis_scaler.joblib",
        "features": config.get("tennis", {}).get("features", [
            "rank_diff", "pts_diff", "age_diff",
            "surface_hard", "surface_clay", "surface_grass",
            "best_of", "ace_diff", "df_diff", "1st_pct_diff", "bp_pct_diff"
        ]),
        "desc": "Victoire du joueur 1",
        "type": "classification",
        "data_pattern": "*tennis*.csv"
    },
    "Basketball": {
        "model_path": MODELS_DIR / "basketball_model.h5",
        "scaler_path": MODELS_DIR / "basketball_scaler.joblib",
        "features": config.get("basketball", {}).get("features", [
            "points_avg_home", "reb_avg_home", "eff_rating_home",
            "back_to_back", "spread", "points_avg_away"
        ]),
        "desc": "Victoire de l'équipe à domicile",
        "type": "classification",
        "data_pattern": "*basket*.csv"
    }
}

# ────────────────────────────────────────────────
# Labels explicites pour les features Tennis
# ────────────────────────────────────────────────
TENNIS_FEATURE_LABELS = {
    "rank_diff":     "Différence de classement (J1 - J2)",
    "pts_diff":      "Différence de points ATP (J1 - J2)",
    "age_diff":      "Différence d'âge (J1 - J2)",
    "surface_hard":  "Surface : Hard",
    "surface_clay":  "Surface : Clay",
    "surface_grass": "Surface : Grass",
    "best_of":       "Format (Best of 3 ou 5)",
    "ace_diff":      "Différence d'aces (J1 - J2)",
    "df_diff":       "Différence de doubles fautes (J1 - J2)",
    "1st_pct_diff":  "Différence % 1ère balle (J1 - J2)",
    "bp_pct_diff":   "Différence % BP sauvées (J1 - J2)",
}

# ────────────────────────────────────────────────
# Chargement modèle
# ────────────────────────────────────────────────
@st.cache_resource
def load_cached_model(sport):
    path = SPORT_CONFIG[sport]["model_path"]
    if not path.exists():
        return None
    try:
        from tensorflow.keras.models import load_model
        return load_model(str(path))
    except Exception as e:
        st.error(f"Erreur chargement modèle {sport}: {e}")
        return None

@st.cache_resource
def load_cached_scaler(sport):
    path = SPORT_CONFIG[sport].get("scaler_path")
    if path and path.exists():
        try:
            return joblib.load(str(path))
        except:
            return None
    return None

# ────────────────────────────────────────────────
# Datasets
# ────────────────────────────────────────────────
@st.cache_data(ttl=600)
def list_available_datasets(sport):
    pattern = SPORT_CONFIG[sport]["data_pattern"]
    candidates = []

    for base in [DATA_RAW_DIR, DATA_PROCESSED_DIR]:
        if base.exists():
            candidates.extend(base.rglob(pattern))
            candidates.extend(base.rglob(f"**/*{sport.lower()}*.csv"))

    if sport == "Tennis":
        tml_dir = DATA_RAW_DIR / "tml-tennis"
        if tml_dir.exists():
            candidates.extend(tml_dir.glob("*.csv"))

    seen = set()
    unique = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    datasets = []
    for p in sorted(unique, key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_file():
            datasets.append({
                "name": p.name,
                "path": str(p),
                "size_kb": round(p.stat().st_size / 1024, 1),
                "modified": pd.Timestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "location": str(p.relative_to(ROOT_DIR))
            })
    return datasets

@st.cache_data
def load_dataset(path_str: str):
    try:
        return pd.read_csv(path_str)
    except Exception as e:
        st.error(f"Impossible de lire {Path(path_str).name}: {e}")
        return None

# ────────────────────────────────────────────────
# Interface
# ────────────────────────────────────────────────
st.set_page_config(page_title="Sports Betting NN", page_icon="🎾⚽🏀", layout="wide")

st.title("Prédictions Paris Sportifs – Réseaux de Neurones")
st.caption("Données dans src/data/raw – Modèles dans models/")

# Sidebar
with st.sidebar:
    st.header("Données détectées")
    for sport_name in SPORT_CONFIG:
        datasets = list_available_datasets(sport_name)
        icon = "✅" if datasets else "⚠️"
        st.write(f"{icon} **{sport_name}**: {len(datasets)} fichier(s)")
        if datasets and st.checkbox(f"Détails {sport_name}", key=f"chk_{sport_name}"):
            for ds in datasets[:6]:
                st.caption(f"• {ds['name']}  ({ds['size_kb']:.1f} KB)  – {ds['modified']}")

    st.markdown("---")
    st.caption("Chemins utilisés :\n• src/data/raw/\n• src/data/processed/\n• models/")

# Sélection sport
col_left, col_right = st.columns([1, 4])
with col_left:
    sport = st.selectbox("Sport", list(SPORT_CONFIG.keys()))

if sport:
    cfg    = SPORT_CONFIG[sport]
    model  = load_cached_model(sport)
    scaler = load_cached_scaler(sport)

    tab_pred, tab_data, tab_info = st.tabs(["Prédiction", "Données", "Infos"])

    # ── Onglet Prédiction ────────────────────────────────────────────────────
    with tab_pred:
        st.subheader(f"Prédiction – {sport}")

        if not model:
            st.warning(f"Modèle {sport} introuvable → {cfg['model_path']}")
        else:
            st.success(f"✅ Modèle chargé ({len(cfg['features'])} features)")
            st.info(f"Objectif : {cfg['desc']}")

            st.markdown("### Caractéristiques du match")
            user_values = {}
            cols = st.columns(3)

            for idx, feat in enumerate(cfg["features"]):
                with cols[idx % 3]:
                    if sport == "Tennis" and feat in TENNIS_FEATURE_LABELS:
                        label = TENNIS_FEATURE_LABELS[feat]
                    else:
                        label = feat.replace("_", " ").title()

                    if feat in ("surface_hard", "surface_clay", "surface_grass"):
                        user_values[feat] = int(st.checkbox(label, value=False))
                    elif feat == "best_of":
                        user_values[feat] = st.selectbox(label, [3, 5])
                    elif "odds" in feat or "cote" in feat:
                        user_values[feat] = st.number_input(label, 1.01, 50.0, 2.0, 0.1)
                    elif feat in ("rank_diff", "pts_diff"):
                        user_values[feat] = st.number_input(label, -2000, 2000, 0, 1, format="%d")
                    elif feat in ("ace_diff", "df_diff"):
                        user_values[feat] = st.number_input(label, -30, 30, 0, 1, format="%d")
                    elif feat in ("1st_pct_diff", "bp_pct_diff"):
                        user_values[feat] = st.number_input(label, -1.0, 1.0, 0.0, 0.01)
                    elif feat == "age_diff":
                        user_values[feat] = st.number_input(label, -20.0, 20.0, 0.0, 0.5)
                    else:
                        user_values[feat] = st.number_input(label, -100.0, 100.0, 0.0, 0.1)

            if st.button("🔮 Prédire", type="primary"):
                try:
                    X = np.array([user_values.get(f, 0.0) for f in cfg["features"]]).reshape(1, -1)

                    if scaler:
                        X = scaler.transform(X)
                        st.caption("Données normalisées (scaler appliqué)")

                    with st.spinner("Prédiction en cours..."):
                        pred  = model.predict(X, verbose=0)
                        value = float(pred[0][0])

                    if cfg["type"] == "classification":
                        proba = value
                        st.metric("Probabilité victoire Joueur 1", f"{proba:.1%}")
                        st.progress(proba)
                        if proba > 0.65:
                            st.success("✅ Valeur potentielle détectée")
                        elif proba > 0.5:
                            st.info("ℹ️ Légère faveur pour le Joueur 1")
                        else:
                            st.warning("⚠️ Joueur 2 favori selon le modèle")
                    else:
                        st.metric("Valeur prédite", f"{value:.2f}")

                except Exception as e:
                    st.error(f"Erreur prédiction : {e}")

    # ── Onglet Données ───────────────────────────────────────────────────────
    with tab_data:
        st.subheader(f"Données – {sport}")
        datasets = list_available_datasets(sport)

        if datasets:
            df_files = pd.DataFrame(datasets)
            st.dataframe(df_files[["name", "size_kb", "modified", "location"]])

            selected = st.selectbox("Fichier à explorer", [d["name"] for d in datasets])
            if selected:
                file = next(d for d in datasets if d["name"] == selected)
                df   = load_dataset(file["path"])
                if df is not None:
                    st.markdown(f"**Aperçu : {selected}** ({len(df)} lignes, {len(df.columns)} colonnes)")
                    st.dataframe(df.head(15))
                    with st.expander("Statistiques descriptives"):
                        st.dataframe(df.describe())
                    with st.expander("Colonnes disponibles"):
                        st.write(list(df.columns))
        else:
            st.info("Aucune donnée trouvée. Placez vos CSV dans src/data/raw/ ou src/data/raw/tml-tennis/")

    # ── Onglet Infos ─────────────────────────────────────────────────────────
    with tab_info:
        st.subheader("Informations techniques")

        st.markdown("**Features attendues :**")
        for f in cfg["features"]:
            label = TENNIS_FEATURE_LABELS.get(f, f) if sport == "Tennis" else f
            st.markdown(f"- `{f}` — {label}")

        st.markdown("**Modèle :**")
        st.code(cfg['model_path'].name if cfg['model_path'].exists() else '❌ Non trouvé')

        st.markdown("**Scaler :**")
        st.code("✅ Présent" if scaler else "⚠️ Absent (prédictions non normalisées)")

        st.markdown("**Chemins :**")
        st.code(f"RAW      : {DATA_RAW_DIR}\nPROCESSED: {DATA_PROCESSED_DIR}\nMODELS   : {MODELS_DIR}")

st.markdown("---")
st.caption("Projet éducatif – Pas de garantie de gain – Jouez responsablement")
