import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime, timedelta
import time
import hashlib
import base64
import warnings
import asyncio
import nest_asyncio
import os
import requests
import gzip  # Pour décompresser le modèle

nest_asyncio.apply()
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION DES CHEMINS
# ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "src" / "data" / "raw" / "tml-tennis"
HIST_DIR = ROOT_DIR / "history"

for dir_path in [MODELS_DIR, DATA_DIR, HIST_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# ─────────────────────────────────────────────────────────────
# TÉLÉCHARGEMENT DU MODÈLE DEPUIS GITHUB RELEASES
# ─────────────────────────────────────────────────────────────
@st.cache_resource(ttl=3600)
def download_model_from_github():
    """
    Télécharge le modèle depuis GitHub Releases
    """
    model_path = MODELS_DIR / "tennis_model.pkl"
    
    # Si le modèle existe déjà, le charger
    if model_path.exists():
        try:
            model_info = joblib.load(model_path)
            st.success(f"✅ Modèle chargé depuis le cache local")
            return model_info
        except:
            pass
    
    # Sinon, télécharger depuis GitHub
    with st.spinner("📥 Téléchargement du modèle depuis GitHub..."):
        try:
            # URL de la release GitHub (à ajuster selon ta release)
            base_url = "https://github.com/Xela91300/sports-betting-neural-net/releases/download/v1.0.0"
            
            # Télécharger les métadonnées d'abord
            meta_response = requests.get(f"{base_url}/model_metadata.json", timeout=30)
            if meta_response.status_code == 200:
                metadata = meta_response.json()
                st.info(f"📊 Modèle trouvé: Accuracy {metadata.get('accuracy', 0):.1%}")
            
            # Télécharger le modèle compressé
            response = requests.get(f"{base_url}/tennis_ml_model_complete.pkl.gz", timeout=60)
            if response.status_code == 200:
                # Sauvegarder temporairement
                temp_path = MODELS_DIR / "model_temp.pkl.gz"
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                
                # Décompresser
                with gzip.open(temp_path, "rb") as f:
                    model_info = joblib.load(f)
                
                # Sauvegarder en local
                joblib.dump(model_info, model_path)
                
                # Nettoyer
                temp_path.unlink()
                
                st.success(f"✅ Modèle téléchargé depuis GitHub avec succès !")
                return model_info
            else:
                st.warning("⚠️ Impossible de télécharger le modèle depuis GitHub")
                return None
                
        except Exception as e:
            st.error(f"❌ Erreur lors du téléchargement: {e}")
            return None

# ─────────────────────────────────────────────────────────────
# FONCTION DE PRÉDICTION ML
# ─────────────────────────────────────────────────────────────
def predict_with_ml_model(model_info, player1, player2, surface='Hard', level='A', best_of=3):
    """
    Fait une prédiction avec le modèle ML
    """
    if model_info is None:
        return None
    
    try:
        # Récupérer les composants du modèle
        model = model_info.get('model')
        scaler = model_info.get('scaler')
        player_stats = model_info.get('player_stats', {})
        features_list = model_info.get('features', [])
        
        if model is None or scaler is None:
            return None
        
        # Récupérer les stats des joueurs
        s1 = player_stats.get(player1, {})
        s2 = player_stats.get(player2, {})
        
        # Si un joueur n'est pas dans la base, retourner None
        if not s1 or not s2:
            return None
        
        # Calculer les features (adapté de ton code d'entraînement)
        r1 = max(s1.get('rank', 500.0), 1.0)
        r2 = max(s2.get('rank', 500.0), 1.0)
        log_rank_ratio = np.log(r2 / r1)
        
        pts_diff = (s1.get('rank_points', 0) - s2.get('rank_points', 0)) / 5000.0
        age_diff = s1.get('age', 25) - s2.get('age', 25)
        
        # Surface
        surf_clay = 1.0 if surface == 'Clay' else 0.0
        surf_grass = 1.0 if surface == 'Grass' else 0.0
        surf_hard = 1.0 if surface == 'Hard' else 0.0
        
        # Niveau du tournoi
        level_gs = 1.0 if level == 'G' else 0.0
        level_m = 1.0 if level == 'M' else 0.0
        best_of_5 = 1.0 if best_of == 5 else 0.0
        
        # Performances
        surf_wr_diff = s1.get('surface_wr', {}).get(surface, 0.5) - s2.get('surface_wr', {}).get(surface, 0.5)
        career_wr_diff = s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5)
        recent_form_diff = s1.get('recent_form', 0.5) - s2.get('recent_form', 0.5)
        h2h_ratio = 0.5  # À améliorer avec vrai H2H
        
        # Statistiques de service
        sp1 = s1.get('serve_pct', {})
        sp2 = s2.get('serve_pct', {})
        sr1 = s1.get('serve_raw', {})
        sr2 = s2.get('serve_raw', {})
        
        ace_diff = (sr1.get('ace', 0) - sr2.get('ace', 0)) / 10.0
        df_diff = (sr1.get('df', 0) - sr2.get('df', 0)) / 5.0
        
        pct_1st_in_diff = sp1.get('pct_1st_in', 0) - sp2.get('pct_1st_in', 0)
        pct_1st_won_diff = sp1.get('pct_1st_won', 0) - sp2.get('pct_1st_won', 0)
        pct_2nd_won_diff = sp1.get('pct_2nd_won', 0) - sp2.get('pct_2nd_won', 0)
        pct_bp_saved_diff = sp1.get('pct_bp_saved', 0) - sp2.get('pct_bp_saved', 0)
        
        # Fatigue
        days_diff = s1.get('days_since_last', 30) - s2.get('days_since_last', 30)
        fatigue_diff = s1.get('fatigue', 0) - s2.get('fatigue', 0)
        
        # Créer le tableau de features
        features = np.array([[
            log_rank_ratio, pts_diff, age_diff,
            surf_clay, surf_grass, surf_hard,
            level_gs, level_m, best_of_5,
            surf_wr_diff, career_wr_diff, recent_form_diff, h2h_ratio,
            ace_diff, df_diff,
            pct_1st_in_diff, pct_1st_won_diff, pct_2nd_won_diff, pct_bp_saved_diff,
            days_diff, fatigue_diff
        ]])
        
        # Normaliser
        features_scaled = scaler.transform(features)
        
        # Prédire
        proba = model.predict_proba(features_scaled)[0][1]
        
        # Borner entre 0.05 et 0.95 pour éviter les extrêmes
        proba = max(0.05, min(0.95, proba))
        
        return float(proba)
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction ML: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# FONCTIONS DE CALCUL DE PROBABILITÉ (FALLBACK)
# ─────────────────────────────────────────────────────────────
def calculate_probability_fallback(df, player1, player2, surface):
    """
    Calcul de probabilité basé sur les stats simples (fallback)
    """
    stats1 = get_player_stats(df, player1)
    stats2 = get_player_stats(df, player2)
    
    proba = 0.5
    
    if stats1 and stats2:
        # Différence de win rate
        proba += (stats1['win_rate'] - stats2['win_rate']) * 0.3
    
    # H2H
    h2h = get_h2h_stats(df, player1, player2)
    if h2h and h2h.get('total_matches', 0) > 0:
        wins1 = h2h.get(f'{player1}_wins', 0)
        proba += (wins1 / h2h['total_matches'] - 0.5) * 0.2
    
    return max(0.05, min(0.95, proba))

# ─────────────────────────────────────────────────────────────
# PAGE PRINCIPALE DE PRÉDICTION
# ─────────────────────────────────────────────────────────────
def show_prediction_page():
    st.markdown("<h2>🎯 Prédiction de match</h2>", unsafe_allow_html=True)
    
    # Charger le modèle
    if 'ml_model' not in st.session_state:
        with st.spinner("Chargement du modèle ML..."):
            st.session_state['ml_model'] = download_model_from_github()
    
    model_info = st.session_state.get('ml_model')
    
    # Afficher le statut du modèle
    if model_info:
        accuracy = model_info.get('accuracy', 0)
        st.sidebar.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0,223,162,0.1), rgba(0,121,255,0.1));
                    padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <div style="font-size: 0.8rem; color: #6C7A89;">MODÈLE ML ACTIF</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #00DFA2;">{accuracy:.1%}</div>
            <div style="font-size: 0.7rem; color: #6C7A89;">précision</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.warning("⚠️ Mode fallback - Stats simples")
    
    # Interface de prédiction
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.text_input("Joueur 1", value="Novak Djokovic")
        player2 = st.text_input("Joueur 2", value="Carlos Alcaraz")
        
    with col2:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
        
        odds1 = st.text_input("Cote Joueur 1 (optionnel)", placeholder="1.75")
        odds2 = st.text_input("Cote Joueur 2 (optionnel)", placeholder="2.10")
    
    if st.button("🎾 Prédire", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            # Essayer d'abord avec le modèle ML
            proba = None
            ml_used = False
            
            if model_info:
                proba = predict_with_ml_model(model_info, player1.strip(), player2.strip(), surface)
                if proba is not None:
                    ml_used = True
            
            # Fallback si le modèle ML n'a pas fonctionné
            if proba is None:
                # Charger les données ATP pour le fallback
                atp_data = load_atp_data()
                proba = calculate_probability_fallback(atp_data, player1.strip(), player2.strip(), surface)
            
            confidence = 50 + abs(proba - 0.5) * 40
            
            # Afficher le résultat
            st.markdown("### 📊 Résultat")
            
            # Barre de progression
            st.progress(float(proba))
            
            # Probabilités
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 1.2rem;">{player1}</div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: #00DFA2;">{proba:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem;">
                    <div style="font-size: 1.2rem;">{player2}</div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: #FF3B3F;">{1-proba:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Info sur le modèle utilisé
            if ml_used:
                st.success("🤖 Prédiction basée sur le modèle ML")
            else:
                st.info("📊 Prédiction basée sur les statistiques simples")
            
            # Calcul du value bet si les cotes sont fournies
            if odds1 and odds2:
                try:
                    o1 = float(odds1.replace(',', '.'))
                    o2 = float(odds2.replace(',', '.'))
                    
                    edge1 = proba - 1/o1
                    edge2 = (1-proba) - 1/o2
                    
                    if edge1 > 0.02:
                        st.success(f"✅ Value bet sur {player1}: {edge1*100:+.1f}%")
                    elif edge2 > 0.02:
                        st.success(f"✅ Value bet sur {player2}: {edge2*100:+.1f}%")
                except:
                    pass

# ─────────────────────────────────────────────────────────────
# FONCTIONS EXISTANTES (À GARDER)
# ─────────────────────────────────────────────────────────────

# [Garder toutes tes fonctions existantes ici : 
#  - load_atp_data()
#  - get_player_stats()
#  - get_h2h_stats()
#  - show_dashboard()
#  - show_multimatches()
#  - show_combines()
#  - show_history()
#  - show_statistics()
#  - show_telegram()
#  - show_configuration()
#  - etc.]

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="TennisIQ Pro - Prédictions IA",
        page_icon="🎾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0A1E2C 0%, #1A2E3C 100%); }
        .stProgress > div > div > div > div { background: linear-gradient(90deg, #00DFA2, #0079FF); }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #00DFA2, #0079FF); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            TennisIQ Pro
        </div>
        <div style="color: #6C7A89; text-transform: uppercase; letter-spacing: 3px;">
            Intelligence Artificielle pour le Tennis
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Menu",
            ["🎯 Prédiction", "📊 Multi-matchs", "🎰 Combinés", 
             "📜 Historique", "📈 Statistiques", "📱 Telegram", "⚙️ Configuration"],
            label_visibility="collapsed"
        )
    
    # Afficher la page sélectionnée
    if page == "🎯 Prédiction":
        show_prediction_page()
    elif page == "📊 Multi-matchs":
        show_multimatches(load_atp_data())
    elif page == "🎰 Combinés":
        show_combines(load_atp_data())
    elif page == "📜 Historique":
        show_history()
    elif page == "📈 Statistiques":
        show_statistics()
    elif page == "📱 Telegram":
        show_telegram()
    elif page == "⚙️ Configuration":
        show_configuration()

if __name__ == "__main__":
    main()
