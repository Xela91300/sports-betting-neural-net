import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import sys
import glob

# Ajouter le dossier src au path Python
ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

# Importer vos modèles personnalisés
try:
    from models.base_model import BaseModel
    from models.football_model import FootballModel
    from models.tennis_model import TennisModel
    from models.basketball_model import BasketballModel
    MODELS_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Erreur d'import des modèles: {e}")
    MODELS_AVAILABLE = False
    st.stop()

# ────────────────────────────────────────────────
# Configuration des chemins
# ────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
DATA_RAW_DIR = SRC_DIR / "data" / "raw"
DATA_PROCESSED_DIR = SRC_DIR / "data" / "processed"
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

# Créer les dossiers s'ils n'existent pas
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Charger la config
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
else:
    st.warning("⚠️ Fichier config.yaml non trouvé, utilisation de la configuration par défaut")
    config = {}

# Configuration des sports
SPORT_CONFIG = {
    "Football": {
        "model_class": FootballModel,
        "model_path": MODELS_DIR / "football_model.h5",
        "features": config.get("football", {}).get("features", [
            "home_form", "away_form", "home_rank", "away_rank", 
            "home_odds", "away_odds", "home_advantage"
        ]),
        "desc": "Victoire de l'équipe à domicile",
        "type": "classification",
        "data_pattern": "football*.csv"
    },
    "Tennis": {
        "model_class": TennisModel,
        "model_path": MODELS_DIR / "tennis_model.h5",
        "features": config.get("tennis", {}).get("features", [
            "player1_rank", "player2_rank", "player1_form", "player2_form",
            "surface_clay", "surface_grass", "head_to_head"
        ]),
        "desc": "Victoire du joueur 1",
        "type": "classification",
        "data_pattern": "tennis*.csv"  # Pattern pour trouver les fichiers tennis
    },
    "Basketball": {
        "model_class": BasketballModel,
        "model_path": MODELS_DIR / "basketball_model.h5",
        "features": config.get("basketball", {}).get("features", [
            "home_form", "away_form", "home_ppg", "away_ppg",
            "pace", "home_odds", "away_odds"
        ]),
        "desc": "Prédiction Over/Under (score total)",
        "type": "regression",
        "data_pattern": "basketball*.csv"
    }
}

# ────────────────────────────────────────────────
# Fonctions de chargement des données
# ────────────────────────────────────────────────
@st.cache_data
def list_available_datasets(sport):
    """Liste tous les fichiers CSV disponibles pour un sport donné"""
    pattern = SPORT_CONFIG[sport]["data_pattern"]
    
    # Chercher dans raw/
    raw_files = list(DATA_RAW_DIR.glob(f"**/{pattern}"))
    raw_files.extend(list(DATA_RAW_DIR.glob(f"**/*{sport.lower()}*.csv")))
    
    # Chercher spécifiquement dans tml-tennis/ pour le tennis
    if sport == "Tennis":
        tennis_dirs = list(DATA_RAW_DIR.glob("**/tennis*"))
        tennis_dirs.extend(list(DATA_RAW_DIR.glob("**/tml*")))
        for tennis_dir in tennis_dirs:
            if tennis_dir.is_dir():
                raw_files.extend(list(tennis_dir.glob("*.csv")))
    
    # Chercher dans processed/
    processed_files = list(DATA_PROCESSED_DIR.glob(f"**/{pattern}"))
    
    all_files = raw_files + processed_files
    
    # Extraire les noms de fichiers pour l'affichage
    datasets = []
    for file_path in all_files:
        rel_path = file_path.relative_to(ROOT_DIR)
        datasets.append({
            "name": file_path.name,
            "path": str(file_path),
            "size": file_path.stat().st_size / 1024,  # Taille en KB
            "modified": pd.Timestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "location": str(rel_path)
        })
    
    return datasets

@st.cache_data
def load_dataset(file_path):
    """Charge un fichier CSV en DataFrame"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement de {file_path}: {e}")
        return None

def get_feature_stats(df, feature_name):
    """Calcule des statistiques pour une feature donnée"""
    if feature_name in df.columns:
        return {
            "min": float(df[feature_name].min()),
            "max": float(df[feature_name].max()),
            "mean": float(df[feature_name].mean()),
            "std": float(df[feature_name].std())
        }
    return None

# ────────────────────────────────────────────────
# Fonctions pour les modèles
# ────────────────────────────────────────────────
@st.cache_resource
def load_cached_model(sport):
    """Charge un modèle en utilisant votre classe personnalisée"""
    sport_config = SPORT_CONFIG[sport]
    model_class = sport_config["model_class"]
    model_path = sport_config["model_path"]
    
    if not model_path.exists():
        return None
    
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
st.markdown("Utilisation de vos données dans `src/data/raw/`")

# Sidebar avec infos
with st.sidebar:
    st.header("📁 Données disponibles")
    
    # Afficher les datasets disponibles pour chaque sport
    for sport_name in SPORT_CONFIG.keys():
        datasets = list_available_datasets(sport_name)
        if datasets:
            st.success(f"✅ {sport_name}: {len(datasets)} fichier(s)")
            if st.checkbox(f"Voir fichiers {sport_name}", key=f"show_{sport_name}"):
                for ds in datasets[:5]:  # Limiter à 5 pour l'affichage
                    st.caption(f"📄 {ds['name']} ({ds['size']:.0f} KB)")
        else:
            st.warning(f"⚠️ {sport_name}: Aucune donnée")
    
    st.markdown("---")
    st.header("📁 Structure")
    st.code("""
    src/data/raw/
    ├── football/
    ├── tennis/
    │   └── tml-tennis/
    └── basketball/
    """)

# Sélection du sport
col1, col2 = st.columns([1, 3])
with col1:
    sport = st.selectbox("Choisissez un sport", ["Football", "Tennis", "Basketball"])

if sport:
    sport_info = SPORT_CONFIG[sport]
    
    # Onglets pour différentes fonctionnalités - CORRIGÉ ICI
    tab1, tab2, tab3 = st.tabs(["📊 Prédiction", "📂 Données", "ℹ️ Stats"])
    
    with tab1:
        # Interface de prédiction
        st.subheader(f"🔮 Prédiction {sport}")
        
        # Charger le modèle
        model = load_cached_model(sport)
        
        if model is None:
            st.warning(f"⚠️ Modèle {sport} non trouvé dans {MODELS_DIR}")
            st.info("Vous pouvez quand même explorer les données dans l'onglet 'Données'")
        else:
            st.success(f"✅ Modèle {sport} chargé")
            
            # Description selon le type
            if sport_info["type"] == "regression":
                st.info(f"🎯 **Objectif**: {sport_info['desc']} (régression)")
            else:
                st.info(f"🎯 **Objectif**: {sport_info['desc']} (classification)")
            
            # Création des inputs utilisateur
            st.markdown("### Entrez les caractéristiques du match")
            
            user_input = {}
            cols = st.columns(min(3, len(sport_info["features"])))
            
            for i, feat in enumerate(sport_info["features"]):
                with cols[i % len(cols)]:
                    feat_display = normalize_feature_name(feat)
                    
                    # Déterminer le type d'input
                    if any(keyword in feat.lower() for keyword in ['surface', 'indoor', 'is_', 'has_']):
                        user_input[feat] = st.checkbox(
                            f"🏷️ {feat_display}",
                            value=False,
                            key=f"checkbox_{feat}"
                        )
                    elif any(keyword in feat.lower() for keyword in ['odds', 'cote']):
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
                        user_input[feat] = st.number_input(
                            f"📊 {feat_display}",
                            value=0.0,
                            step=0.1,
                            format="%.2f",
                            key=f"num_{feat}"
                        )
            
            if st.button("🔮 Prédire", type="primary", use_container_width=True):
                try:
                    features_list = [float(user_input[feat]) for feat in sport_info["features"]]
                    X = np.array(features_list).reshape(1, -1)
                    
                    with st.spinner("Calcul en cours..."):
                        prediction = model.predict(X)
                    
                    if sport_info["type"] == "regression":
                        pred_value = float(prediction[0][0])
                        st.metric("🎯 Score prédit", f"{pred_value:.1f} points")
                    else:
                        proba = float(prediction[0][0])
                        st.metric("📊 Probabilité", f"{proba:.1%}")
                        st.progress(proba)
                        
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
    
    with tab2:
        st.subheader(f"📂 Données {sport}")
        
        # Lister les datasets disponibles
        datasets = list_available_datasets(sport)
        
        if datasets:
            st.success(f"✅ {len(datasets)} fichier(s) trouvé(s)")
            
            # Créer un DataFrame pour l'affichage
            df_list = pd.DataFrame(datasets)
            st.dataframe(
                df_list[["name", "size", "modified", "location"]],
                use_container_width=True,
                hide_index=True
            )
            
            # Sélectionner un fichier à visualiser
            selected_file = st.selectbox(
                "Choisissez un fichier à visualiser",
                options=[d["name"] for d in datasets],
                format_func=lambda x: f"📄 {x}"
            )
            
            if selected_file:
                file_info = next(d for d in datasets if d["name"] == selected_file)
                
                # Charger et afficher le fichier
                df = load_dataset(file_info["path"])
                if df is not None:
                    st.markdown(f"**Aperçu de {selected_file}**")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Statistiques
                    with st.expander("📊 Statistiques"):
                        st.dataframe(df.describe(), use_container_width=True)
                    
                    # Colonnes disponibles
                    with st.expander("📋 Colonnes"):
                        st.write(list(df.columns))
        else:
            st.warning(f"⚠️ Aucune donnée trouvée pour {sport}")
            st.info(f"Placez vos fichiers CSV dans: `{DATA_RAW_DIR}/`")
            
            # Afficher les chemins attendus
            st.code(f"""
            Chemins attendus pour {sport}:
            - {DATA_RAW_DIR}/{sport.lower()}/
            - {DATA_RAW_DIR}/raw/{sport.lower()}/
            - {DATA_RAW_DIR}/**/*{sport.lower()}*.csv
            - {DATA_RAW_DIR}/**/tml-{sport.lower()}/
            """)
    
    with tab3:
        st.subheader(f"ℹ️ Statistiques {sport}")
        
        st.markdown("**Features utilisées par le modèle :**")
        for feat in sport_info["features"]:
            st.markdown(f"- `{feat}`")
        
        st.markdown("**Type de modèle :**")
        st.info(f"{sport_info['type'].title()} - {sport_info['desc']}")
        
        st.markdown("**Chemins :**")
        st.code(f"""
        Modèle: {sport_info['model_path']}
        Données: {DATA_RAW_DIR}
        Pattern: {sport_info['data_pattern']}
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.caption(f"📁 Données: `{DATA_RAW_DIR}`")
    st.caption("🎓 Projet éducatif - Jouez responsablement")
