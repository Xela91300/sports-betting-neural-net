import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import sys

# Ajouter le dossier src au path Python
ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))  # Permet d'importer depuis src/

# Importer vos modèles personnalisés
try:
    from models.base_model import BaseModel
    from models.football_model import FootballModel
    from models.tennis_model import TennisModel
    from models.basketball_model import BasketballModel
    MODELS_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Erreur d'import des modèles: {e}")
    st.error("Vérifiez que le dossier src/models/ contient bien les fichiers Python")
    MODELS_AVAILABLE = False
    st.stop()

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"  # Dossier où sont sauvegardés les .h5
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

# Vérifier que la config existe
if not CONFIG_PATH.exists():
    st.error(f"❌ Fichier de configuration introuvable: {CONFIG_PATH}")
    st.stop()

# Charger la config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Mapping sport → modèle & configuration
SPORT_CONFIG = {
    "Football": {
        "model_class": FootballModel,
        "model_path": MODELS_DIR / "football_model.h5",
        "features": config.get("football", {}).get("features", []),
        "desc": "Victoire de l'équipe à domicile",
        "type": "classification"
    },
    "Tennis": {
        "model_class": TennisModel,
        "model_path": MODELS_DIR / "tennis_model.h5",
        "features": config.get("tennis", {}).get("features", []),
        "desc": "Victoire du joueur 1",
        "type": "classification"
    },
    "Basketball": {
        "model_class": BasketballModel,
        "model_path": MODELS_DIR / "basketball_model.h5",
        "features": config.get("basketball", {}).get("features", []),
        "desc": "Prédiction Over/Under (score total)",
        "type": "regression"  # Basketball utilise la régression pour Over/Under
    }
}

# ────────────────────────────────────────────────
# Fonctions utilitaires
# ────────────────────────────────────────────────
@st.cache_resource
def load_cached_model(sport):
    """Charge un modèle en utilisant votre classe personnalisée"""
    sport_config = SPORT_CONFIG[sport]
    model_class = sport_config["model_class"]
    model_path = sport_config["model_path"]
    
    if not model_path.exists():
        st.warning(f"⚠️ Modèle pour {sport} introuvable: {model_path}")
        st.info("Le modèle doit être entraîné d'abord avec src/train.py")
        
        # Option pour créer un modèle factice (démo)
        if st.button(f"Créer un modèle factice pour {sport} (démo)"):
            model = model_class()
            model.build_model(input_dim=len(sport_config["features"]))
            
            # Sauvegarder
            model_path.parent.mkdir(exist_ok=True)
            model.save(str(model_path))
            st.success(f"✅ Modèle factice créé pour {sport}")
            return model
        return None
    
    # Charger le modèle existant
    try:
        model = model_class()
        model.load(str(model_path))
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle {sport}: {e}")
        return None

def normalize_feature_name(feat):
    """Convertit un nom de feature en nom lisible"""
    return feat.replace('_', ' ').title()

# ────────────────────────────────────────────────
# Interface principale
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Sports Betting Neural Net",
    page_icon="🎲",
    layout="wide"
)

st.title("🎲 Prédictions Paris Sportifs - Réseaux de Neurones")
st.markdown("Utilisation de vos modèles personnalisés dans `src/models/`")

# Sidebar avec infos
with st.sidebar:
    st.header("📁 Structure du projet")
    st.code("""
    sports-betting-neural-net/
    ├── app.py
    ├── config/config.yaml
    ├── src/models/
    │   ├── base_model.py
    │   ├── football_model.py
    │   ├── tennis_model.py
    │   └── basketball_model.py
    └── models/ (vos .h5)
    """)
    
    st.header("ℹ️ Statut")
    if MODELS_AVAILABLE:
        st.success("✅ Classes de modèles chargées")
    else:
        st.error("❌ Classes de modèles non trouvées")

# Sélection du sport
col1, col2 = st.columns([1, 3])
with col1:
    sport = st.selectbox("Choisissez un sport", ["Football", "Tennis", "Basketball"])

if sport:
    sport_info = SPORT_CONFIG[sport]
    
    # Vérifier les features
    if not sport_info["features"]:
        st.error(f"❌ Aucune feature définie pour {sport} dans config/config.yaml")
        with st.expander("Format attendu pour config.yaml"):
            st.code("""
football:
  features:
    - home_form
    - away_form
    - home_rank
    - away_rank
    - home_odds
    - away_odds
            """)
        st.stop()
    
    # Charger le modèle
    with st.spinner(f"Chargement du modèle {sport}..."):
        model = load_cached_model(sport)
    
    if model is None:
        st.warning("⚠️ Modèle non disponible. Veuillez d'abord entraîner un modèle avec `src/train.py`")
        st.stop()
    
    # Interface de prédiction
    st.subheader(f"📊 Prédiction {sport}")
    
    # Description selon le type
    if sport_info["type"] == "regression":
        st.info(f"🎯 **Objectif**: {sport_info['desc']} (régression)")
    else:
        st.info(f"🎯 **Objectif**: {sport_info['desc']} (classification)")
    
    # Création des inputs utilisateur
    st.markdown("### Entrez les caractéristiques du match")
    
    user_input = {}
    
    # Organiser en colonnes
    num_features = len(sport_info["features"])
    cols = st.columns(min(3, num_features))
    
    for i, feat in enumerate(sport_info["features"]):
        with cols[i % len(cols)]:
            feat_display = normalize_feature_name(feat)
            
            # Déterminer le type d'input
            if any(keyword in feat.lower() for keyword in ['surface', 'indoor', 'is_', 'has_']):
                # Features booléennes
                user_input[feat] = st.checkbox(
                    f"🏷️ {feat_display}",
                    value=False,
                    key=f"checkbox_{feat}"
                )
            elif any(keyword in feat.lower() for keyword in ['odds', 'cote']):
                # Cotes
                user_input[feat] = st.number_input(
                    f"💰 {feat_display}",
                    value=2.0,
                    min_value=1.01,
                    max_value=100.0,
                    step=0.1,
                    format="%.2f",
                    key=f"odds_{feat}"
                )
            elif any(keyword in feat.lower() for keyword in ['rank', 'classement']):
                # Classements
                user_input[feat] = st.number_input(
                    f"🏆 {feat_display}",
                    value=50,
                    min_value=1,
                    max_value=500,
                    step=1,
                    format="%d",
                    key=f"rank_{feat}"
                )
            else:
                # Features numériques standards
                user_input[feat] = st.number_input(
                    f"📊 {feat_display}",
                    value=0.0,
                    step=0.1,
                    format="%.2f",
                    key=f"num_{feat}"
                )
    
    # Bouton de prédiction
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔮 Prédire", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Préparer les features dans le bon ordre
            features_list = []
            for feat in sport_info["features"]:
                if feat in user_input:
                    features_list.append(float(user_input[feat]))
                else:
                    features_list.append(0.0)
            
            X = np.array(features_list).reshape(1, -1)
            
            # Faire la prédiction
            with st.spinner("Calcul en cours..."):
                prediction = model.predict(X)
            
            # Afficher selon le type
            if sport_info["type"] == "regression":
                # Pour Basketball (Over/Under)
                pred_value = float(prediction[0][0])
                
                st.metric(
                    label="🎯 Score total prédit",
                    value=f"{pred_value:.1f} points",
                    delta=None
                )
                
                # Interface Over/Under
                st.markdown("### 📈 Analyse Over/Under")
                threshold = st.number_input(
                    "Seuil Over/Under (ex: 210.5)",
                    value=210.5,
                    step=0.5,
                    format="%.1f",
                    key="threshold"
                )
                
                diff = pred_value - threshold
                
                # Barre de progression relative
                progress = min(1.0, max(0.0, (pred_value - 180) / 60))  # Normalisation approximative
                st.progress(progress)
                
                if diff > 0:
                    st.success(f"📈 **OVER {threshold}** (prédiction: {pred_value:.1f}, écart: +{diff:.1f})")
                else:
                    st.info(f"📉 **UNDER {threshold}** (prédiction: {pred_value:.1f}, écart: {diff:.1f})")
                
                # Suggestion de mise
                confidence = min(abs(diff) / 20, 1.0)  # Plus l'écart est grand, plus la confiance est haute
                if confidence > 0.5:
                    st.balloons()
                    st.success(f"💡 Confiance élevée ({confidence:.0%}) - Opportunité de value")
                elif confidence > 0.2:
                    st.info(f"💡 Confiance moyenne ({confidence:.0%}) - Prudence")
                else:
                    st.warning(f"💡 Confiance faible ({confidence:.0%}) - Éviter ou miser petit")
                    
            else:
                # Pour classification (Football/Tennis)
                proba = float(prediction[0][0])
                
                # Afficher la probabilité
                col1, col2, col3 = st.columns(3)
                with col2:
                    st.metric(
                        label="📊 Probabilité de victoire",
                        value=f"{proba:.1%}",
                        delta=None
                    )
                
                # Barre de progression
                st.progress(proba)
                
                # Interprétation
                st.markdown("### 📊 Analyse")
                
                if proba > 0.65:
                    st.success("✅ **Bonne opportunité de pari** (probabilité > 65%)")
                    if proba > 0.80:
                        st.balloons()
                        st.info("✨ Très forte probabilité - Vérifiez quand même les cotes")
                elif proba > 0.50:
                    st.info("⚖️ **Match équilibré** (probabilité entre 50% et 65%)")
                else:
                    st.warning(f"⚠️ **Probabilité faible** ({proba:.1%}) - Cherchez la value sur l'adversaire")
                
                # Suggestion de cote minimale
                if proba > 0:
                    min_odds = 1 / proba
                    st.info(f"💰 Pour être rentable, il faudrait une cote > **{min_odds:.2f}**")
            
            # Afficher les détails
            with st.expander("🔍 Voir les détails de la prédiction"):
                st.json({
                    "sport": sport,
                    "model_path": str(sport_info["model_path"]),
                    "prediction_type": sport_info["type"],
                    "features_used": sport_info["features"],
                    "feature_values": user_input,
                    "raw_prediction": float(prediction[0][0]) if prediction is not None else None
                })
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {e}")
            st.exception(e)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.caption(f"📁 Modèles sauvegardés dans: `{MODELS_DIR}`")
    st.caption("🎓 Projet éducatif - Jouez responsablement")
