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
import gzip

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

HIST_FILE = HIST_DIR / "predictions_history.json"
COMB_HIST_FILE = HIST_DIR / "combines_history.json"
USER_STATS_FILE = HIST_DIR / "user_stats.json"

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
SURFACES = ["Hard", "Clay", "Grass"]
MAX_MATCHES_ANALYSIS = 30
MAX_MATCHES_COMBINE = 30
MIN_PROBA_COMBINE = 0.55
MIN_EDGE_COMBINE = 0.02
MAX_SELECTIONS_COMBINE = 30

COLORS = {
    "primary": "#00DFA2",
    "success": "#00DFA2",
    "warning": "#FFB200",
    "danger": "#FF3B3F",
    "gray": "#6C7A89",
    "surface_hard": "#0079FF",
    "surface_clay": "#E67E22",
    "surface_grass": "#00DFA2",
}

SURFACE_CONFIG = {
    "Hard": {"color": COLORS["surface_hard"], "icon": "🟦"},
    "Clay": {"color": COLORS["surface_clay"], "icon": "🟧"},
    "Grass": {"color": COLORS["surface_grass"], "icon": "🟩"}
}

# ─────────────────────────────────────────────────────────────
# TELEGRAM INTEGRATION
# ─────────────────────────────────────────────────────────────
def get_telegram_config():
    try:
        token = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        return token, str(chat_id)
    except Exception:
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if token and chat_id:
            return token, chat_id
        return None, None

def send_telegram_message(message, parse_mode='HTML'):
    token, chat_id = get_telegram_config()
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True
        }
        response = requests.post(url, json=payload, timeout=15)
        return response.status_code == 200
    except Exception as e:
        print(f"Erreur Telegram: {e}")
        return False

def test_telegram_connection():
    token, chat_id = get_telegram_config()
    if not token:
        return False, "❌ Token manquant"
    if not chat_id:
        return False, "❌ Chat ID manquant"
    try:
        test_message = f"<b>✅ Test réussi !</b>\n\n📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {'chat_id': chat_id, 'text': test_message, 'parse_mode': 'HTML'}
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            return True, "✅ Connexion réussie !"
        else:
            return False, f"❌ Erreur: {resp.text}"
    except Exception as e:
        return False, f"❌ Exception: {str(e)}"

def format_prediction_message(pred_data):
    proba = pred_data.get('proba', 0.5)
    bar_length = 10
    filled = int(proba * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    message = f"""
<b>🎾 PRÉDICTION TENNISIQ</b>

<b>Match:</b> {pred_data.get('player1', '?')} vs {pred_data.get('player2', '?')}
<b>Surface:</b> {pred_data.get('surface', '?')}

<b>Probabilités:</b>
{bar}  {proba:.1%} / {1-proba:.1%}

• {pred_data.get('player1', 'J1')}: <b>{proba:.1%}</b>
• {pred_data.get('player2', 'J2')}: <b>{1-proba:.1%}</b>

<b>Favori:</b> {pred_data.get('favori_modele', '?')}
"""
    return message

def send_prediction_to_telegram(pred_data):
    return send_telegram_message(format_prediction_message(pred_data))

# ─────────────────────────────────────────────────────────────
# GROQ API
# ─────────────────────────────────────────────────────────────
def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return os.environ.get("GROQ_API_KEY", None)

def call_groq_api(prompt):
    api_key = get_groq_key()
    if not api_key:
        return "⚠️ Clé API Groq non configurée"
    try:
        # Simulation pour l'instant (à remplacer par vraie API Groq)
        return f"Analyse IA: {prompt[:100]}..."
    except Exception as e:
        return f"❌ Erreur: {str(e)}"

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_saved_model():
    """Charge le modèle ML depuis le dossier models/"""
    model_path = MODELS_DIR / "tennis_ml_model_complete.pkl"
    if model_path.exists():
        try:
            model_info = joblib.load(model_path)
            return model_info
        except Exception as e:
            st.warning(f"⚠️ Erreur chargement modèle: {e}")
            return None
    else:
        # Essayer de télécharger depuis GitHub
        try:
            url = "https://github.com/Xela91300/sports-betting-neural-net/releases/download/v1.0.0/tennis_ml_model_complete.pkl.gz"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                temp_path = MODELS_DIR / "model_temp.pkl.gz"
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                with gzip.open(temp_path, "rb") as f:
                    model_info = joblib.load(f)
                joblib.dump(model_info, model_path)
                temp_path.unlink()
                return model_info
        except:
            pass
    return None

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES ATP
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_atp_data():
    """Charge les données ATP depuis le dossier data/"""
    if not DATA_DIR.exists():
        return pd.DataFrame()
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()
    
    atp_dfs = []
    for f in csv_files[:5]:  # Limiter pour la vitesse
        if 'wta' in f.name.lower():
            continue
        try:
            df = pd.read_csv(f, encoding='utf-8', nrows=1000, on_bad_lines='skip')
            if 'winner_name' in df.columns and 'loser_name' in df.columns:
                atp_dfs.append(df)
        except:
            continue
    
    if atp_dfs:
        return pd.concat(atp_dfs, ignore_index=True)
    return pd.DataFrame()

# ─────────────────────────────────────────────────────────────
# FONCTIONS STATISTIQUES
# ─────────────────────────────────────────────────────────────
def get_player_stats(df, player):
    """Récupère les stats basiques d'un joueur"""
    if df.empty or not player:
        return None
    
    player_clean = player.strip()
    winner_col = 'winner_name' if 'winner_name' in df.columns else None
    loser_col = 'loser_name' if 'loser_name' in df.columns else None
    
    if not winner_col or not loser_col:
        return None
    
    matches = df[(df[winner_col] == player_clean) | (df[loser_col] == player_clean)]
    if len(matches) == 0:
        return None
    
    wins = len(matches[df[winner_col] == player_clean])
    total = len(matches)
    
    return {
        'name': player_clean,
        'matches_played': total,
        'wins': wins,
        'losses': total - wins,
        'win_rate': wins / total if total > 0 else 0
    }

def get_h2h_stats(df, player1, player2):
    """Récupère les stats H2H entre deux joueurs"""
    if df.empty or not player1 or not player2:
        return None
    
    p1 = player1.strip()
    p2 = player2.strip()
    winner_col = 'winner_name' if 'winner_name' in df.columns else None
    loser_col = 'loser_name' if 'loser_name' in df.columns else None
    
    if not winner_col or not loser_col:
        return None
    
    h2h = df[((df[winner_col] == p1) & (df[loser_col] == p2)) | 
             ((df[winner_col] == p2) & (df[loser_col] == p1))]
    
    if len(h2h) == 0:
        return None
    
    return {
        'total_matches': len(h2h),
        f'{p1}_wins': len(h2h[df[winner_col] == p1]),
        f'{p2}_wins': len(h2h[df[winner_col] == p2]),
    }

def calculate_probability(df, player1, player2, surface, h2h=None):
    """Calcule une probabilité simple basée sur les stats"""
    stats1 = get_player_stats(df, player1)
    stats2 = get_player_stats(df, player2)
    
    proba = 0.5
    
    if stats1 and stats2:
        proba += (stats1['win_rate'] - stats2['win_rate']) * 0.3
    
    if h2h and h2h.get('total_matches', 0) > 0:
        wins1 = h2h.get(f'{player1}_wins', 0)
        proba += (wins1 / h2h['total_matches'] - 0.5) * 0.2
    
    return max(0.05, min(0.95, proba))

def calculate_confidence(proba):
    """Calcule un score de confiance"""
    return 50 + abs(proba - 0.5) * 40

# ─────────────────────────────────────────────────────────────
# HISTORIQUE
# ─────────────────────────────────────────────────────────────
def load_history():
    if not HIST_FILE.exists():
        return []
    try:
        with open(HIST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_prediction(pred_data):
    try:
        history = load_history()
        pred_data['id'] = hashlib.md5(f"{datetime.now()}{pred_data.get('player1','')}".encode()).hexdigest()[:8]
        history.append(pred_data)
        if len(history) > 1000:
            history = history[-1000:]
        with open(HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        return True
    except:
        return False

def load_combines():
    if not COMB_HIST_FILE.exists():
        return []
    try:
        with open(COMB_HIST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_combine(combine_data):
    try:
        combines = load_combines()
        combine_data['date'] = datetime.now().isoformat()
        combine_data['id'] = hashlib.md5(f"{datetime.now()}".encode()).hexdigest()[:8]
        combines.append(combine_data)
        if len(combines) > 200:
            combines = combines[-200:]
        with open(COMB_HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(combines, f, indent=2)
        return True
    except:
        return False

def load_user_stats():
    if not USER_STATS_FILE.exists():
        return {
            'total_predictions': 0,
            'correct_predictions': 0,
            'total_combines': 0,
            'won_combines': 0,
            'total_invested': 0,
            'total_won': 0,
            'best_streak': 0,
            'current_streak': 0,
        }
    try:
        with open(USER_STATS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

# ─────────────────────────────────────────────────────────────
# PAGES DE L'APPLICATION
# ─────────────────────────────────────────────────────────────

def show_dashboard():
    """Page Dashboard"""
    st.markdown("<h2>🏠 Dashboard</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Matchs disponibles", "349K+")
    with col2:
        history = load_history()
        st.metric("Prédictions", len(history))
    with col3:
        stats = load_user_stats()
        accuracy = (stats.get('correct_predictions', 0) / max(stats.get('total_predictions', 1), 1)) * 100
        st.metric("Précision", f"{accuracy:.1f}%")
    with col4:
        st.metric("Joueurs", "15K+")

def show_prediction():
    """Page de prédiction simple"""
    st.markdown("<h2>🎯 Prédiction Simple</h2>", unsafe_allow_html=True)
    
    model_info = load_saved_model()
    if model_info:
        st.sidebar.success(f"✅ Modèle ML chargé (accuracy: {model_info.get('accuracy', 0):.1%})")
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.text_input("Joueur 1", "Novak Djokovic")
        player2 = st.text_input("Joueur 2", "Carlos Alcaraz")
    
    with col2:
        surface = st.selectbox("Surface", SURFACES)
        odds1 = st.text_input("Cote J1 (optionnel)", "")
        odds2 = st.text_input("Cote J2 (optionnel)", "")
    
    if st.button("🎾 Prédire", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            # Charger données pour stats
            atp_data = load_atp_data()
            
            # Calculer probabilité
            h2h = get_h2h_stats(atp_data, player1, player2)
            proba = calculate_probability(atp_data, player1, player2, surface, h2h)
            confidence = calculate_confidence(proba)
            
            # Afficher résultat
            st.markdown("### 📊 Résultat")
            st.progress(float(proba))
            
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
            
            # Value bet
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
            
            # Sauvegarder
            pred_data = {
                'player1': player1, 'player2': player2,
                'surface': surface, 'proba': proba,
                'confidence': confidence, 'date': datetime.now().isoformat(),
                'odds1': odds1, 'odds2': odds2,
                'favori_modele': player1 if proba >= 0.5 else player2
            }
            
            if st.button("💾 Sauvegarder la prédiction"):
                if save_prediction(pred_data):
                    st.success("✅ Sauvegardé !")
                else:
                    st.error("❌ Erreur")

def show_multimatches():
    """Page Multi-matchs"""
    st.markdown("<h2>📊 Multi-matchs</h2>", unsafe_allow_html=True)
    
    n_matches = st.number_input("Nombre de matchs", 2, 10, 3)
    
    atp_data = load_atp_data()
    matches = []
    
    for i in range(n_matches):
        with st.expander(f"Match {i+1}", expanded=i==0):
            col1, col2 = st.columns(2)
            with col1:
                p1 = st.text_input(f"Joueur 1 - Match {i+1}", key=f"mm_p1_{i}")
            with col2:
                p2 = st.text_input(f"Joueur 2 - Match {i+1}", key=f"mm_p2_{i}")
            
            surface = st.selectbox(f"Surface - Match {i+1}", SURFACES, key=f"mm_surf_{i}")
            matches.append({'p1': p1, 'p2': p2, 'surface': surface})
    
    if st.button("🔍 Analyser tout", use_container_width=True):
        for i, match in enumerate(matches):
            if match['p1'] and match['p2']:
                st.markdown(f"### Match {i+1}: {match['p1']} vs {match['p2']}")
                h2h = get_h2h_stats(atp_data, match['p1'], match['p2'])
                proba = calculate_probability(atp_data, match['p1'], match['p2'], match['surface'], h2h)
                st.progress(float(proba))
                st.caption(f"Probabilité {match['p1']}: {proba:.1%}")

def show_combines():
    """Page Combinés"""
    st.markdown("<h2>🎰 Générateur de Combinés</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        n_matches = st.number_input("Nombre de matchs", 2, 10, 3)
    with col2:
        mise = st.number_input("Mise (€)", 1.0, 1000.0, 10.0)
    
    atp_data = load_atp_data()
    selections = []
    
    for i in range(n_matches):
        with st.expander(f"Match {i+1}"):
            col1, col2 = st.columns(2)
            with col1:
                p1 = st.text_input(f"J1", key=f"comb_p1_{i}")
                odds1 = st.text_input(f"Cote {p1}", key=f"comb_odds1_{i}")
            with col2:
                p2 = st.text_input(f"J2", key=f"comb_p2_{i}")
                odds2 = st.text_input(f"Cote {p2}", key=f"comb_odds2_{i}")
            
            surface = st.selectbox("Surface", SURFACES, key=f"comb_surf_{i}")
            
            if p1 and p2 and odds1 and odds2:
                try:
                    o1 = float(odds1.replace(',', '.'))
                    o2 = float(odds2.replace(',', '.'))
                    h2h = get_h2h_stats(atp_data, p1, p2)
                    proba = calculate_probability(atp_data, p1, p2, surface, h2h)
                    
                    edge1 = proba - 1/o1
                    edge2 = (1-proba) - 1/o2
                    
                    if edge1 > MIN_EDGE_COMBINE and proba >= MIN_PROBA_COMBINE:
                        selections.append({
                            'match': f"{p1} vs {p2}",
                            'joueur': p1, 'proba': proba,
                            'cote': o1, 'edge': edge1
                        })
                    elif edge2 > MIN_EDGE_COMBINE and (1-proba) >= MIN_PROBA_COMBINE:
                        selections.append({
                            'match': f"{p1} vs {p2}",
                            'joueur': p2, 'proba': 1-proba,
                            'cote': o2, 'edge': edge2
                        })
                except:
                    pass
    
    if selections and st.button("🎯 Générer combiné", use_container_width=True):
        selections.sort(key=lambda x: x['edge'], reverse=True)
        selected = selections[:min(5, len(selections))]
        
        proba_combi = np.prod([s['proba'] for s in selected])
        cote_combi = np.prod([s['cote'] for s in selected])
        gain = mise * cote_combi
        esperance = proba_combi * gain - mise
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Probabilité", f"{proba_combi:.1%}")
        with col2:
            st.metric("Cote", f"{cote_combi:.2f}")
        with col3:
            st.metric("Gain potentiel", f"{gain:.2f}€")
        with col4:
            st.metric("Espérance", f"{esperance:+.2f}€")
        
        # Sauvegarder
        combine_data = {
            'selections': selected,
            'proba_globale': proba_combi,
            'cote_globale': cote_combi,
            'mise': mise,
            'gain_potentiel': gain,
            'esperance': esperance,
            'nb_matches': len(selected)
        }
        save_combine(combine_data)
        st.success("✅ Combiné sauvegardé !")

def show_history():
    """Page Historique"""
    st.markdown("<h2>📜 Historique</h2>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📋 Prédictions", "🎰 Combinés"])
    
    with tab1:
        history = load_history()
        if history:
            for pred in history[::-1][:20]:
                with st.expander(f"{pred.get('date', '')[:16]} - {pred.get('player1','?')} vs {pred.get('player2','?')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Surface", pred.get('surface', '—'))
                    with col2:
                        st.metric("Probabilité", f"{pred.get('proba', 0.5):.1%}")
                    with col3:
                        st.metric("Confiance", f"{pred.get('confidence', 0):.0f}")
        else:
            st.info("Aucune prédiction")
    
    with tab2:
        combines = load_combines()
        if combines:
            for comb in combines[::-1][:10]:
                with st.expander(f"{comb.get('date', '')[:16]} - {comb.get('nb_matches',0)} matchs"):
                    st.metric("Probabilité", f"{comb.get('proba_globale',0):.1%}")
                    st.metric("Cote", f"{comb.get('cote_globale',0):.2f}")
        else:
            st.info("Aucun combiné")

def show_statistics():
    """Page Statistiques"""
    st.markdown("<h2>📈 Statistiques</h2>", unsafe_allow_html=True)
    
    stats = load_user_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Prédictions", stats.get('total_predictions', 0))
    with col2:
        accuracy = (stats.get('correct_predictions', 0) / max(stats.get('total_predictions', 1), 1)) * 100
        st.metric("Précision", f"{accuracy:.1f}%")
    with col3:
        st.metric("Combinés", stats.get('total_combines', 0))
    with col4:
        profit = stats.get('total_won', 0) - stats.get('total_invested', 0)
        st.metric("Profit", f"{profit:+.2f}€")

def show_telegram():
    """Page Telegram"""
    st.markdown("<h2>📱 Telegram</h2>", unsafe_allow_html=True)
    
    token, chat_id = get_telegram_config()
    if not token or not chat_id:
        st.warning("⚠️ Telegram non configuré")
        st.code("""
        Ajoute dans les secrets Streamlit :
        TELEGRAM_BOT_TOKEN = "ton_token"
        TELEGRAM_CHAT_ID = "ton_chat_id"
        """)
        return
    
    st.success(f"✅ Telegram configuré (Chat ID: {chat_id})")
    
    if st.button("🔧 Tester la connexion"):
        success, msg = test_telegram_connection()
        if success:
            st.success(msg)
        else:
            st.error(msg)
    
    st.markdown("### 📝 Envoyer un message")
    with st.form("telegram_form"):
        message = st.text_area("Message", height=100)
        if st.form_submit_button("📤 Envoyer") and message:
            if send_telegram_message(message):
                st.success("✅ Message envoyé !")
            else:
                st.error("❌ Échec de l'envoi")

def show_configuration():
    """Page Configuration"""
    st.markdown("<h2>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    st.markdown("### 🤖 Modèle ML")
    model_info = load_saved_model()
    if model_info:
        st.success(f"✅ Modèle chargé (accuracy: {model_info.get('accuracy', 0):.1%})")
    else:
        st.warning("⚠️ Aucun modèle trouvé")
    
    st.markdown("### 📊 Données")
    atp_data = load_atp_data()
    st.info(f"📁 {len(atp_data)} matchs chargés")
    
    st.markdown("### 🗑️ Gestion")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Effacer prédictions"):
            if HIST_FILE.exists():
                HIST_FILE.unlink()
                st.rerun()
    with col2:
        if st.button("🗑️ Effacer combinés"):
            if COMB_HIST_FILE.exists():
                COMB_HIST_FILE.unlink()
                st.rerun()

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="TennisIQ Pro",
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
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 2rem; font-weight: 800; color: #00DFA2;">
                TennisIQ
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🎯 Prédiction", "📊 Multi-matchs", 
             "🎰 Combinés", "📜 Historique", "📈 Statistiques", 
             "📱 Telegram", "⚙️ Configuration"],
            label_visibility="collapsed"
        )
    
    # Afficher la page sélectionnée
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🎯 Prédiction":
        show_prediction()
    elif page == "📊 Multi-matchs":
        show_multimatches()
    elif page == "🎰 Combinés":
        show_combines()
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
