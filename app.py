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

nest_asyncio.apply()
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# TELEGRAM INTEGRATION
# ─────────────────────────────────────────────────────────────
try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

def get_telegram_config():
    """Récupère la config Telegram depuis les secrets Streamlit"""
    try:
        token = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        return token, str(chat_id)
    except Exception as e:
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if token and chat_id:
            return token, chat_id
        return None, None

async def send_telegram_message_async(message, parse_mode='HTML'):
    token, chat_id = get_telegram_config()
    if not token or not chat_id or not TELEGRAM_AVAILABLE:
        return False
    try:
        bot = Bot(token=token)
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode=parse_mode,
            disable_web_page_preview=True
        )
        return True
    except Exception as e:
        print(f"Erreur Telegram: {e}")
        return False

def send_telegram_message_requests(message, parse_mode='HTML'):
    """Envoie un message Telegram via requests (fallback)"""
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
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Erreur Telegram requests: {e}")
        return False

def send_telegram_message(message, parse_mode='HTML'):
    if TELEGRAM_AVAILABLE:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(send_telegram_message_async(message, parse_mode))
            loop.close()
            if result:
                return True
        except:
            pass
    return send_telegram_message_requests(message, parse_mode)

def format_prediction_message(pred_data, ai_comment=None):
    emoji_map = {'Hard': '🟦', 'Clay': '🟧', 'Grass': '🟩'}
    surface_emoji = emoji_map.get(pred_data.get('surface', ''), '🎾')
    ml_tag = "🤖 " if pred_data.get('ml_used') else ""
    proba = pred_data.get('proba', 0.5)
    bar_length = 10
    filled = int(proba * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    message = f"""
<b>{ml_tag}🎾 PRÉDICTION TENNISIQ</b>

<b>Match:</b> {pred_data.get('player1', '?')} vs {pred_data.get('player2', '?')}
<b>Tournoi:</b> {pred_data.get('tournament', '?')}
<b>Surface:</b> {surface_emoji} {pred_data.get('surface', '?')}

<b>Probabilités:</b>
{bar}  {proba:.1%} / {1-proba:.1%}

• {pred_data.get('player1', 'J1')}: <b>{proba:.1%}</b>
• {pred_data.get('player2', 'J2')}: <b>{1-proba:.1%}</b>

<b>Favori du modèle:</b> {pred_data.get('favori_modele', '?')}
<b>Confiance:</b> {'🟢' if pred_data.get('confidence', 0) >= 70 else '🟡' if pred_data.get('confidence', 0) >= 50 else '🔴'} {pred_data.get('confidence', 0):.0f}/100
"""
    
    if pred_data.get('odds1') and pred_data.get('odds2'):
        message += f"""
<b>Cotes bookmaker:</b>
• {pred_data.get('player1', 'J1')}: <code>{pred_data.get('odds1')}</code>
• {pred_data.get('player2', 'J2')}: <code>{pred_data.get('odds2')}</code>
"""
    
    if pred_data.get('best_value'):
        bv = pred_data['best_value']
        edge_color = '🟢' if bv['edge'] > 0.05 else '🟡'
        message += f"""
<b>🎯 VALUE BET DÉTECTÉ!</b>
{edge_color} <b>{bv['joueur']}</b> à <b>{bv['cote']:.2f}</b>
Edge: <b>{bv['edge']*100:+.1f}%</b>
"""
    
    if ai_comment:
        clean_comment = ai_comment.replace('<', '&lt;').replace('>', '&gt;')
        message += f"\n\n<b>🤖 ANALYSE IA:</b>\n{clean_comment}"
    
    message += f"\n\n#TennisIQ #{pred_data.get('surface', 'Tennis')}"
    return message

def format_combine_message(combine_data, ai_comment=None):
    proba = combine_data.get('proba_globale', 0)
    bar_length = 10
    filled = int(proba * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    message = f"""
<b>🎰 COMBINÉ TENNISIQ</b>

<b>📊 Statistiques:</b>
{bar}  {proba:.1%}
• {combine_data.get('nb_matches', 0)} sélections
• Cote combinée: <b>{combine_data.get('cote_globale', 0):.2f}</b>
• Espérance: <b>{combine_data.get('esperance', 0):+.2f}€</b>
• Kelly: <b>{combine_data.get('kelly', 0)*100:.1f}%</b>

<b>📋 Sélections:</b>
"""
    
    for i, sel in enumerate(combine_data.get('selections', [])[:5], 1):
        edge_color = '🟢' if sel.get('edge', 0) > 0.05 else '🟡'
        message += f"\n{i}. {edge_color} {sel.get('joueur', '?')} @ {sel.get('cote', 0):.2f} (edge: {sel.get('edge', 0)*100:+.1f}%)"
    
    if ai_comment:
        clean_comment = ai_comment.replace('<', '&lt;').replace('>', '&gt;')
        message += f"\n\n<b>🤖 ANALYSE IA:</b>\n{clean_comment}"
    
    message += f"\n\n#TennisIQ #Combiné"
    return message

def format_stats_message():
    stats = load_user_stats()
    history = load_history()
    
    total = stats.get('total_predictions', 0)
    correct = stats.get('correct_predictions', 0)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    recent = history[-10:] if len(history) >= 10 else history
    recent_correct = 0
    for pred in recent:
        if pred.get('statut') in ['joueur1_gagne', 'joueur2_gagne']:
            favori = pred.get('favori_modele', pred.get('player1'))
            if (pred.get('statut') == 'joueur1_gagne' and favori == pred.get('player1')) or \
               (pred.get('statut') == 'joueur2_gagne' and favori == pred.get('player2')):
                recent_correct += 1
    recent_acc = (recent_correct / len(recent) * 100) if recent else 0
    
    bar_length = 10
    filled = int(accuracy / 10)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    message = f"""
<b>📊 STATISTIQUES TENNISIQ</b>

<b>Global:</b>
{bar}  {accuracy:.1f}%
• Prédictions: {total}
• Correctes: {correct}
• Série actuelle: {stats.get('current_streak', 0)} {'🔥' if stats.get('current_streak', 0) > 3 else ''}
• Meilleure série: {stats.get('best_streak', 0)}

<b>Dernières 10:</b>
• Correctes: {recent_correct}/{len(recent)}
• Précision: {recent_acc:.1f}%

#TennisIQ #Stats
"""
    return message

def send_prediction_to_telegram(pred_data, ai_comment=None):
    return send_telegram_message(format_prediction_message(pred_data, ai_comment))

def send_combine_to_telegram(combine_data, ai_comment=None):
    return send_telegram_message(format_combine_message(combine_data, ai_comment))

def send_stats_to_telegram():
    return send_telegram_message(format_stats_message())

def test_telegram_connection():
    token, chat_id = get_telegram_config()
    if not token or not chat_id:
        return False, "❌ Configuration Telegram manquante"
    
    test_message = f"""
<b>🔧 TEST DE CONNEXION RÉUSSI!</b>

✅ Bot configuré
📱 Prêt à recevoir des prédictions
📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}

#TennisIQ #Test
"""
    if send_telegram_message(test_message):
        return True, "✅ Connexion réussie ! Message de test envoyé."
    else:
        return False, "❌ Échec de l'envoi. Vérifie ton token et chat_id"

def send_custom_message():
    st.markdown("### 📝 Message personnalisé")
    with st.form("custom_msg"):
        title = st.text_input("Titre", "Message TennisIQ")
        content = st.text_area("Contenu", height=100)
        urgent = st.checkbox("🔴 Urgent")
        if st.form_submit_button("📤 Envoyer") and content:
            urgent_tag = "🔴 URGENT - " if urgent else ""
            msg = f"<b>{urgent_tag}{title}</b>\n\n{content}\n\n📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n#TennisIQ"
            if send_telegram_message(msg):
                st.success("✅ Message envoyé !")
            else:
                st.error("❌ Échec de l'envoi. Vérifie la configuration Telegram.")

# ─────────────────────────────────────────────────────────────
# GROQ API
# ─────────────────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

def get_groq_key():
    """Récupère la clé Groq depuis les secrets"""
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return os.environ.get("GROQ_API_KEY", None)

def call_groq_api(prompt):
    """Appelle l'API Groq pour générer une analyse"""
    if not GROQ_AVAILABLE:
        return "⚠️ Bibliothèque Groq non installée. Installe avec: pip install groq"
    
    api_key = get_groq_key()
    if not api_key:
        return "⚠️ Clé API Groq non configurée. Ajoute GROQ_API_KEY dans les secrets."
    
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erreur API Groq: {str(e)}"

# ─────────────────────────────────────────────────────────────
# ML IMPORTS
# ─────────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    from sklearn.metrics import (accuracy_score, roc_auc_score, brier_score_loss,
                                  log_loss, confusion_matrix, classification_report)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# CONFIGURATION PAGE
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TennisIQ Pro - Prédictions IA",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CONSTANTES
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

# Features utilisées pour le modèle ML
ML_FEATURES = [
    "log_rank_ratio", "pts_diff_norm", "age_diff",
    "surf_clay", "surf_grass", "surf_hard",
    "level_gs", "level_m", "best_of_5",
    "surf_wr_diff", "career_wr_diff",
    "ace_diff_norm", "df_diff_norm",
    "pct_1st_in_diff", "pct_1st_won_diff",
    "pct_2nd_won_diff", "pct_bp_saved_diff",
]

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

TOURNAMENTS_ATP = [
    ("Australian Open", "Hard", "G", 5), ("Roland Garros", "Clay", "G", 5),
    ("Wimbledon", "Grass", "G", 5), ("US Open", "Hard", "G", 5),
    ("Indian Wells Masters", "Hard", "M", 3), ("Miami Open", "Hard", "M", 3),
    ("Monte-Carlo Masters", "Clay", "M", 3), ("Madrid Open", "Clay", "M", 3),
    ("Italian Open", "Clay", "M", 3), ("Canadian Open", "Hard", "M", 3),
    ("Cincinnati Masters", "Hard", "M", 3), ("Shanghai Masters", "Hard", "M", 3),
    ("Paris Masters", "Hard", "M", 3), ("Rotterdam", "Hard", "500", 3),
    ("Dubai Tennis Champs", "Hard", "500", 3), ("Acapulco", "Hard", "500", 3),
    ("Barcelona Open", "Clay", "500", 3), ("Halle Open", "Grass", "500", 3),
    ("Queen's Club", "Grass", "500", 3), ("Hamburg Open", "Clay", "500", 3),
    ("Washington Open", "Hard", "500", 3), ("Tokyo", "Hard", "500", 3),
    ("Vienna Open", "Hard", "500", 3), ("Basel", "Hard", "500", 3),
    ("Beijing", "Hard", "500", 3), ("Nitto ATP Finals", "Hard", "F", 3),
]

TOURN_DICT = {t[0]: (t[1], t[2], t[3]) for t in TOURNAMENTS_ATP}
TOURN_NAMES = [t[0] for t in TOURNAMENTS_ATP]

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; margin: 0; padding: 0; box-sizing: border-box; }
    .stApp { background: linear-gradient(135deg, #0A1E2C 0%, #1A2E3C 100%); }
    .result-card {
        background: linear-gradient(135deg, rgba(0,223,162,0.1), rgba(0,121,255,0.1));
        border: 2px solid #00DFA2;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    .badge {
        display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin: 0.25rem;
    }
    .progress-bar {
        width: 100%; height: 12px; background: rgba(255,255,255,0.1);
        border-radius: 6px; overflow: hidden; margin: 1rem 0;
    }
    .progress-fill {
        height: 100%; background: linear-gradient(90deg, #00DFA2, #0079FF);
        border-radius: 6px; transition: width 0.5s ease;
    }
    .metric-card {
        background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05);
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .ml-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(0,223,162,0.15), rgba(0,121,255,0.15));
        border: 1px solid rgba(0,223,162,0.3);
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.75rem;
        font-weight: 700;
        color: #00DFA2;
        letter-spacing: 1px;
    }
    .model-card {
        background: linear-gradient(135deg, rgba(0,223,162,0.05), rgba(0,121,255,0.05));
        border: 1px solid rgba(0,223,162,0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .header-title {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(135deg, #00DFA2, #0079FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.5rem;
    }
    .header-subtitle { color: #6C7A89; text-align: center; text-transform: uppercase; letter-spacing: 3px; }
    .divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FONCTIONS DE BASE
# ─────────────────────────────────────────────────────────────
def format_number(num, decimals=2):
    if num is None or pd.isna(num): return "—"
    if isinstance(num, (int, float)):
        if abs(num) >= 1e6: return f"{num/1e6:.1f}M"
        if abs(num) >= 1e3: return f"{num/1e3:.0f}K"
        return f"{num:,.{decimals}f}".replace(",", " ")
    return str(num)

def create_badge(text, color="#00DFA2"):
    bg = f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.1)"
    return f'<span class="badge" style="background: {bg}; color: {color};">{text}</span>'

def create_metric(label, value, unit="", color="#FFFFFF"):
    return f"""
    <div class="metric-card">
        <div style="font-size:0.7rem; color:#6C7A89; text-transform:uppercase;">{label}</div>
        <div style="font-size:1.8rem; font-weight:700; color:{color};">{value}<span style="font-size:0.8rem; color:#6C7A89;">{unit}</span></div>
    </div>
    """

def create_progress_bar(value):
    return f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {value*100:.1f}%;"></div>
    </div>
    """

def create_result_card(player1, player2, proba, confidence):
    favori = player1 if proba >= 0.5 else player2
    
    if confidence >= 70:
        conf_color = COLORS['success']
        conf_text = "🔋 CONFIANCE ÉLEVÉE"
    elif confidence >= 50:
        conf_color = COLORS['warning']
        conf_text = "⚡ CONFIANCE MODÉRÉE"
    else:
        conf_color = COLORS['danger']
        conf_text = "⚠️ CONFIANCE FAIBLE"
    
    return f"""
    <div class="result-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 1.5rem; color: #fff;">{player1}</div>
                <div style="font-size: 2.5rem; font-weight: 800; color: {COLORS['primary'] if proba >= 0.5 else COLORS['gray']};">{proba:.1%}</div>
            </div>
            <div style="font-size: 2rem; color: {COLORS['gray']};">VS</div>
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 1.5rem; color: #fff;">{player2}</div>
                <div style="font-size: 2.5rem; font-weight: 800; color: {COLORS['primary'] if proba < 0.5 else COLORS['gray']};">{1-proba:.1%}</div>
            </div>
        </div>
        {create_progress_bar(proba)}
        <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
            <div style="text-align: left;">
                <div style="color: {COLORS['gray']};">Favori</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {COLORS['primary']};">{favori}</div>
            </div>
            <div style="text-align: right;">
                <div style="color: {COLORS['gray']};">Confiance</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {conf_color};">{conf_text}</div>
            </div>
        </div>
    </div>
    """

# ─────────────────────────────────────────────────────────────
# FONCTIONS ML (NOUVELLES)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=7200)
def precompute_player_stats_ml(_df):
    """Pré-calcule les statistiques avancées pour le ML"""
    if _df is None or _df.empty: return {}
    df = _df.copy()
    df['_w_name'] = df['winner_name'].astype(str).str.strip()
    df['_l_name'] = df['loser_name'].astype(str).str.strip()
    all_players = set(df['_w_name'].unique()) | set(df['_l_name'].unique())
    stats = {}
    
    for player in all_players:
        if not player or player == 'nan': continue
        w_mask = df['_w_name'] == player
        l_mask = df['_l_name'] == player
        wins_df = df[w_mask]
        loss_df = df[l_mask]
        total = len(wins_df) + len(loss_df)
        if total == 0: continue
        
        rank = None
        if len(wins_df) > 0 and 'winner_rank' in df.columns:
            r = wins_df['winner_rank'].dropna()
            if len(r) > 0: rank = float(r.iloc[-1])
        if rank is None and len(loss_df) > 0 and 'loser_rank' in df.columns:
            r = loss_df['loser_rank'].dropna()
            if len(r) > 0: rank = float(r.iloc[-1])
        
        rank_points = None
        if len(wins_df) > 0 and 'winner_rank_points' in df.columns:
            p = wins_df['winner_rank_points'].dropna()
            if len(p) > 0: rank_points = float(p.iloc[-1])
        
        age = None
        if len(wins_df) > 0 and 'winner_age' in df.columns:
            a = wins_df['winner_age'].dropna()
            if len(a) > 0: age = float(a.mean())
        
        win_rate = len(wins_df) / total if total > 0 else 0.5
        
        recent_form = 0.5
        player_all = pd.concat([wins_df.assign(_result=1), loss_df.assign(_result=0)])
        if len(player_all) >= 5:
            last_20 = player_all.tail(20)
            recent_form = float(last_20['_result'].mean()) if len(last_20) > 0 else 0.5
        
        stats[player] = {
            'rank': rank or 500.0,
            'rank_points': rank_points or 0.0,
            'age': age or 25.0,
            'total_matches': total,
            'wins': len(wins_df),
            'losses': len(loss_df),
            'win_rate': win_rate,
            'recent_form': recent_form,
        }
    return stats

def prepare_ml_training_data(_df):
    """Prépare les données pour l'entraînement ML"""
    if _df is None or _df.empty:
        return None, None
    
    required = ['winner_rank', 'loser_rank']
    if not all(c in _df.columns for c in required):
        return None, None
    
    X_list, y_list = [], []
    
    for _, row in _df.iterrows():
        try:
            w_rank = float(row['winner_rank']) if pd.notna(row['winner_rank']) else 100.0
            l_rank = float(row['loser_rank']) if pd.notna(row['loser_rank']) else 100.0
            if w_rank <= 0: w_rank = 100.0
            if l_rank <= 0: l_rank = 100.0
            
            log_rank_ratio = np.log(l_rank / w_rank)
            
            feat = [log_rank_ratio, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            X_list.append(feat)
            y_list.append(1)
            
            feat_l = [-log_rank_ratio, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            X_list.append(feat_l)
            y_list.append(0)
        except:
            continue
    
    if len(X_list) < 500:
        return None, None
    
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

def train_ml_model(df):
    """Entraîne le modèle ML"""
    if not SKLEARN_AVAILABLE:
        return None
    
    with st.spinner("⏳ Préparation des données ML..."):
        X, y = prepare_ml_training_data(df)
    
    if X is None or len(X) < 500:
        return None
    
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    if len(X_train) > 60000:
        idx = np.random.choice(len(X_train), 60000, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    
    with st.spinner("🤖 Entraînement du modèle RandomForest..."):
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_split=20,
            min_samples_leaf=10, n_jobs=-1, random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train_sc, y_train)
        
        calibrated = CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=100, max_depth=8, min_samples_split=20,
                min_samples_leaf=10, n_jobs=-1, random_state=42
            ),
            cv=3, method='isotonic'
        )
        calibrated.fit(X_train_sc, y_train)
    
    with st.spinner("📊 Évaluation du modèle..."):
        y_pred = rf.predict(X_test_sc)
        y_proba = rf.predict_proba(X_test_sc)[:, 1]
        
        accuracy = float(accuracy_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_proba))
        
        roi_sims = []
        for i in range(len(y_test)):
            pred_win = y_proba[i] > 0.5
            actual_win = y_test[i] == 1
            roi_sims.append(1.0 if pred_win == actual_win else -1.0)
        simulated_roi = float(np.mean(roi_sims))
    
    return {
        'model': calibrated,
        'scaler': scaler,
        'accuracy': accuracy,
        'auc': auc,
        'simulated_roi': simulated_roi,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'trained_at': datetime.now().isoformat(),
    }

def predict_with_ml(model_info, player_stats, p1, p2, surface, level='A', best_of=3, h2h=None):
    """Prédit avec le modèle ML"""
    if model_info is None or player_stats is None:
        return None
    
    try:
        s1 = player_stats.get(p1, {})
        s2 = player_stats.get(p2, {})
        
        r1 = max(s1.get('rank', 500.0), 1.0)
        r2 = max(s2.get('rank', 500.0), 1.0)
        log_rank_ratio = np.log(r2 / r1)
        
        features = [log_rank_ratio, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        X = np.array(features).reshape(1, -1)
        X_sc = model_info['scaler'].transform(X)
        ml_proba = float(model_info['model'].predict_proba(X_sc)[0][1])
        
        return max(0.05, min(0.95, ml_proba))
    except:
        return None

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DONNÉES
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_atp_data():
    if not DATA_DIR.exists(): 
        return pd.DataFrame()
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files: 
        return pd.DataFrame()
    atp_dfs = []
    for f in csv_files:
        if 'wta' in f.name.lower(): 
            continue
        try:
            df = None
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(f, encoding=enc, on_bad_lines='skip', low_memory=False)
                    break
                except:
                    try:
                        df = pd.read_csv(f, sep=';', encoding=enc, on_bad_lines='skip', low_memory=False)
                        break
                    except: 
                        continue
            if df is not None and 'winner_name' in df.columns and 'loser_name' in df.columns:
                atp_dfs.append(df)
        except: 
            continue
    if atp_dfs:
        return pd.concat(atp_dfs, ignore_index=True)
    return pd.DataFrame()

def get_player_stats(df, player, surface=None):
    if df is None or df.empty or player is None: 
        return None
    player_clean = player.strip()
    winner_col = 'winner_name' if 'winner_name' in df.columns else None
    loser_col = 'loser_name' if 'loser_name' in df.columns else None
    if not winner_col or not loser_col: 
        return None
    dw = df[winner_col].astype(str).str.strip()
    dl = df[loser_col].astype(str).str.strip()
    matches = df[(dw == player_clean) | (dl == player_clean)]
    if len(matches) == 0: 
        return None
    wins = len(matches[dw == player_clean])
    total = len(matches)
    return {
        'name': player_clean, 'matches_played': total,
        'wins': wins, 'losses': total - wins,
        'win_rate': wins / total if total > 0 else 0
    }

def get_h2h_stats(df, player1, player2):
    if df is None or df.empty or player1 is None or player2 is None: 
        return None
    p1 = player1.strip()
    p2 = player2.strip()
    winner_col = 'winner_name' if 'winner_name' in df.columns else None
    loser_col = 'loser_name' if 'loser_name' in df.columns else None
    if not winner_col or not loser_col: 
        return None
    dw = df[winner_col].astype(str).str.strip()
    dl = df[loser_col].astype(str).str.strip()
    h2h = df[((dw == p1) & (dl == p2)) | ((dw == p2) & (dl == p1))]
    if len(h2h) == 0: 
        return None
    return {
        'total_matches': len(h2h),
        f'{p1}_wins': len(h2h[dw == p1]),
        f'{p2}_wins': len(h2h[dw == p2]),
    }

def calculate_probability(df, player1, player2, surface, h2h=None, model_info=None, player_stats=None):
    """Calcule la probabilité avec ou sans ML"""
    if model_info is not None and player_stats is not None:
        ml_proba = predict_with_ml(model_info, player_stats, player1, player2, surface)
        if ml_proba is not None:
            return ml_proba
    
    stats1 = get_player_stats(df, player1, surface)
    stats2 = get_player_stats(df, player2, surface)
    score = 0.5
    if stats1 and stats2:
        score += (stats1['win_rate'] - stats2['win_rate']) * 0.3
    if h2h and h2h.get('total_matches', 0) > 0:
        wins1 = h2h.get(f'{player1}_wins', 0)
        score += (wins1 / h2h['total_matches'] - 0.5) * 0.2
    return max(0.05, min(0.95, score))

def calculate_confidence(proba, player1, player2, h2h):
    confidence = 50
    if h2h and h2h.get('total_matches', 0) >= 3:
        confidence += 10
    confidence += abs(proba - 0.5) * 40
    return min(100, confidence)

# ─────────────────────────────────────────────────────────────
# HISTORIQUE
# ─────────────────────────────────────────────────────────────
def load_history():
    if not HIST_FILE.exists(): 
        return []
    try:
        with open(HIST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []
    except Exception as e:
        print(f"Erreur chargement historique: {e}")
        return []

def save_prediction(pred_data):
    try:
        history = load_history()
        if 'id' not in pred_data:
            pred_data['id'] = hashlib.md5(
                f"{pred_data.get('date', datetime.now().isoformat())}{pred_data.get('player1','')}{pred_data.get('player2','')}".encode()
            ).hexdigest()[:8]
        history.append(pred_data)
        if len(history) > 1000:
            history = history[-1000:]
        with open(HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Erreur sauvegarde: {e}")
        return False

def update_prediction_status(pred_id, statut):
    try:
        history = load_history()
        for pred in history:
            if pred.get('id') == pred_id:
                pred['statut'] = statut
                break
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
        combine_data['statut'] = 'en_attente'
        combine_data['id'] = hashlib.md5(f"{combine_data['date']}{len(combines)}".encode()).hexdigest()[:8]
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
            'total_predictions': 0, 'correct_predictions': 0,
            'total_combines': 0, 'won_combines': 0,
            'total_invested': 0, 'total_won': 0,
            'best_streak': 0, 'current_streak': 0,
        }
    try:
        with open(USER_STATS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except: 
        return {}

# ─────────────────────────────────────────────────────────────
# INTERFACE PRINCIPALE
# ─────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="header-title">TennisIQ Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Intelligence Artificielle pour le Tennis</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    with st.spinner("Chargement des données..."):
        atp_data = load_atp_data()

    if not atp_data.empty and 'player_stats_cache' not in st.session_state:
        with st.spinner("Calcul des statistiques avancées..."):
            st.session_state['player_stats_cache'] = precompute_player_stats_ml(atp_data)

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 2rem; font-weight: 800; background: linear-gradient(135deg, #00DFA2, #0079FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                TennisIQ
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🎯 Prédictions", "📊 Multi-matchs", "🎰 Combinés",
             "📜 Historique", "📈 Statistiques", "🤖 Modèle ML", "📱 Telegram", "⚙️ Configuration"],
            label_visibility="collapsed"
        )
        
        if not atp_data.empty:
            st.markdown(create_badge(f"ATP: {len(atp_data):,} matchs", COLORS['primary']), unsafe_allow_html=True)
        
        token, _ = get_telegram_config()
        if token:
            st.markdown(create_badge("📱 Telegram: OK", COLORS['success']), unsafe_allow_html=True)

    if page == "🏠 Dashboard":
        show_dashboard(atp_data)
    elif page == "🎯 Prédictions":
        show_predictions(atp_data)
    elif page == "📊 Multi-matchs":
        show_multimatches(atp_data)
    elif page == "🎰 Combinés":
        show_combines(atp_data)
    elif page == "📜 Historique":
        show_history()
    elif page == "📈 Statistiques":
        show_statistics()
    elif page == "🤖 Modèle ML":
        show_model_page(atp_data)
    elif page == "📱 Telegram":
        show_telegram()
    elif page == "⚙️ Configuration":
        show_configuration()

# ─────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────
def show_dashboard(atp_data):
    st.markdown("<h2>🏠 Tableau de Bord</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_metric("Matchs ATP", format_number(len(atp_data) if not atp_data.empty else 0)), unsafe_allow_html=True)
    with col2:
        history = load_history()
        st.markdown(create_metric("Prédictions", len(history)), unsafe_allow_html=True)
    with col3:
        stats = load_user_stats()
        accuracy = (stats.get('correct_predictions', 0) / max(stats.get('total_predictions', 1), 1)) * 100
        st.markdown(create_metric("Précision", f"{accuracy:.1f}", "%"), unsafe_allow_html=True)
    with col4:
        streak = stats.get('current_streak', 0)
        st.markdown(create_metric("Série", streak, "", COLORS['success'] if streak > 0 else COLORS['gray']), unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if not atp_data.empty and 'surface' in atp_data.columns:
        st.markdown("<h3>📊 Répartition des surfaces</h3>", unsafe_allow_html=True)
        surface_counts = atp_data['surface'].value_counts()
        st.bar_chart(surface_counts)

# ─────────────────────────────────────────────────────────────
# PAGE MODÈLE ML
# ─────────────────────────────────────────────────────────────
def show_model_page(atp_data):
    st.markdown("<h2>🤖 Modèle Machine Learning</h2>", unsafe_allow_html=True)

    if not SKLEARN_AVAILABLE:
        st.error("⚠️ **scikit-learn non installé.** Exécutez : `pip install scikit-learn`")
        return

    if atp_data.empty:
        st.warning("Aucune donnée ATP disponible pour entraîner le modèle.")
        return

    model_info = st.session_state.get('ml_model')

    st.markdown("""
    <div class="model-card">
        <h4>🧠 Architecture du modèle</h4>
        <p>
        RandomForest (150 arbres, profondeur max 10) + <strong>calibration isotonique</strong><br>
        Features : ratio de classement (log), âge, surface, niveau, best-of, win rate surface, 
        win rate carrière, stats de service (ace%, 1er service%, sauvegarde BP%)<br>
        Split temporel 80/20 pour éviter le leakage. Données équilibrées (gagnant/perdant = 50/50).
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        if model_info is None:
            st.info("👆 Le modèle n'a pas encore été entraîné. Cliquez sur **Entraîner** pour démarrer.")
            if st.button("🚀 Entraîner le modèle ML", use_container_width=True):
                model_info = train_ml_model(atp_data)
                if model_info:
                    st.session_state['ml_model'] = model_info
                    st.success(f"✅ Modèle entraîné avec succès ! Précision : **{model_info['accuracy']:.1%}**")
                    st.rerun()
                else:
                    st.error("❌ Entraînement impossible (données insuffisantes ou colonnes manquantes).")
        else:
            st.success(f"✅ Modèle actif — entraîné le {model_info.get('trained_at', '')[:16]}")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if st.button("🔄 Ré-entraîner", use_container_width=True):
                    model_info = train_ml_model(atp_data)
                    if model_info:
                        st.session_state['ml_model'] = model_info
                        st.success("✅ Modèle mis à jour !")
                        st.rerun()
            with col_r2:
                if st.button("🗑️ Supprimer le modèle", use_container_width=True):
                    st.session_state['ml_model'] = None
                    st.rerun()

    with col2:
        if model_info:
            acc_color = COLORS['success'] if model_info['accuracy'] >= 0.65 else COLORS['warning']
            st.markdown(create_metric("Précision", f"{model_info['accuracy']:.1%}", "", acc_color), unsafe_allow_html=True)

    if model_info is None:
        return

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    acc = model_info['accuracy']
    auc = model_info['auc']

    with col1:
        c = COLORS['success'] if acc >= 0.66 else COLORS['warning'] if acc >= 0.62 else COLORS['danger']
        st.markdown(create_metric("Précision", f"{acc:.1%}", "", c), unsafe_allow_html=True)
        st.caption("% de matchs correctement prédits")

    with col2:
        c = COLORS['success'] if auc >= 0.70 else COLORS['warning'] if auc >= 0.65 else COLORS['danger']
        st.markdown(create_metric("AUC-ROC", f"{auc:.3f}", "", c), unsafe_allow_html=True)
        st.caption("Discrimination (1.0 = parfait, 0.5 = aléatoire)")

    with col3:
        roi = model_info.get('simulated_roi', 0)
        c = COLORS['success'] if roi > 0 else COLORS['danger']
        st.markdown(create_metric("ROI simulé", f"{roi:+.1%}", "", c), unsafe_allow_html=True)
        st.caption("Si on mise 1€ sur chaque favori du modèle")

    with col4:
        st.markdown(create_metric("Matchs test", format_number(model_info['n_test'] // 2, 0)), unsafe_allow_html=True)
        st.caption("Échantillon de test")

# ─────────────────────────────────────────────────────────────
# PRÉDICTIONS SIMPLES (CORRIGÉE)
# ─────────────────────────────────────────────────────────────
def show_predictions(atp_data):
    st.markdown("<h2>🎯 Prédiction Simple</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    player1 = player2 = tournament = None
    surface = "Hard"
    
    with col1:
        if not atp_data.empty:
            winner_col = 'winner_name' if 'winner_name' in atp_data.columns else None
            loser_col = 'loser_name' if 'loser_name' in atp_data.columns else None
            if winner_col and loser_col:
                players = sorted(set(str(p).strip() for p in atp_data[winner_col].dropna().unique() if pd.notna(p)) |
                               set(str(p).strip() for p in atp_data[loser_col].dropna().unique() if pd.notna(p)))
                if players:
                    player1 = st.selectbox("Joueur 1", players, key="pred_p1")
                    players2 = [p for p in players if p != player1]
                    player2 = st.selectbox("Joueur 2", players2, key="pred_p2") if players2 else None
                    
                    if 'tourney_name' in atp_data.columns:
                        tournaments = sorted(atp_data['tourney_name'].dropna().unique())
                        tournament = st.selectbox("Tournoi", tournaments, key="pred_tournament") if tournaments else None
                        if tournament and 'surface' in atp_data.columns:
                            surface_df = atp_data[atp_data['tourney_name'] == tournament]['surface']
                            if not surface_df.empty:
                                surface = surface_df.iloc[0]
                    
                    with st.expander("📊 Cotes bookmaker (optionnel)"):
                        odds1 = st.text_input(f"Cote {player1}", key="pred_odds1", placeholder="1.75")
                        odds2 = st.text_input(f"Cote {player2}", key="pred_odds2", placeholder="2.10") if player2 else st.text_input("Cote J2", key="pred_odds2", placeholder="2.10")
                    
                    if surface in SURFACE_CONFIG:
                        st.markdown(create_badge(f"{SURFACE_CONFIG[surface]['icon']} {surface}", SURFACE_CONFIG[surface]['color']), unsafe_allow_html=True)
    
    with col2:
        if player1 and player2:
            p1 = player1.strip()
            p2 = player2.strip()
            h2h = get_h2h_stats(atp_data, p1, p2)
            
            model_info = st.session_state.get('ml_model')
            player_stats = st.session_state.get('player_stats_cache')
            
            proba = calculate_probability(atp_data, p1, p2, surface, h2h, model_info, player_stats)
            confidence = calculate_confidence(proba, p1, p2, h2h)
            
            ml_used = model_info is not None and player_stats is not None
            
            best_value = None
            if 'odds1' in locals() and odds1 and odds2:
                try:
                    o1 = float(odds1.replace(',', '.'))
                    o2 = float(odds2.replace(',', '.'))
                    edge1 = proba - 1/o1
                    edge2 = (1 - proba) - 1/o2
                    if edge1 > edge2 and edge1 > MIN_EDGE_COMBINE:
                        best_value = {'joueur': p1, 'edge': edge1, 'cote': o1, 'proba': proba}
                    elif edge2 > edge1 and edge2 > MIN_EDGE_COMBINE:
                        best_value = {'joueur': p2, 'edge': edge2, 'cote': o2, 'proba': 1 - proba}
                except: 
                    pass
            
            favori = p1 if proba >= 0.5 else p2
            
            ml_tag = '<span class="ml-badge">🤖 ML</span>' if ml_used else ''
            st.markdown(f"### Résultat {ml_tag}", unsafe_allow_html=True)
            st.markdown(create_result_card(p1, p2, proba, confidence), unsafe_allow_html=True)
            
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                send_tg = st.checkbox("📤 Envoyer Telegram", key="pred_send_tg")
            with col_t2:
                send_ai = st.checkbox("🤖 Ajouter analyse IA", key="pred_send_ai")
            with col_t3:
                if st.button("🤖 Générer IA", key="pred_gen_ai", use_container_width=True):
                    with st.spinner("Analyse IA en cours..."):
                        vb_txt = f"Value bet sur {best_value['joueur']} (edge {best_value['edge']*100:+.1f}%)" if best_value else "Aucun value bet"
                        prompt = f"Analyse ce match ATP : {p1} vs {p2} sur {surface}. Proba: {p1} {proba:.1%} | {p2} {1-proba:.1%}. {vb_txt}. Donne une analyse concise en 3 points en français."
                        ai_analysis = call_groq_api(prompt)
                        if ai_analysis:
                            st.session_state['last_ai'] = ai_analysis
                            st.info(ai_analysis)
            
            if best_value:
                st.success(f"✅ Value bet! {best_value['joueur']} @ {best_value['cote']:.2f} (edge: {best_value['edge']*100:+.1f}%)")
            
            if st.button("💾 Sauvegarder", key="pred_save", use_container_width=True):
                pred_data = {
                    'player1': p1,
                    'player2': p2,
                    'tournament': tournament if tournament else "Inconnu",
                    'surface': surface,
                    'proba': float(proba),
                    'confidence': float(confidence),
                    'odds1': odds1 if 'odds1' in locals() and odds1 else None,
                    'odds2': odds2 if 'odds2' in locals() and odds2 else None,
                    'favori_modele': favori,
                    'best_value': best_value,
                    'ml_used': ml_used,
                    'date': datetime.now().isoformat(),
                    'statut': 'en_attente'
                }
                
                if save_prediction(pred_data):
                    st.success("✅ Prédiction sauvegardée dans l'historique !")
                    
                    if send_tg:
                        with st.spinner("📤 Envoi sur Telegram..."):
                            ai_comment = st.session_state.get('last_ai') if send_ai else None
                            if send_prediction_to_telegram(pred_data, ai_comment):
                                st.success("📱 Prédiction envoyée sur Telegram !")
                            else:
                                st.error("❌ Échec de l'envoi Telegram. Vérifie la configuration.")
                else:
                    st.error("❌ Erreur lors de la sauvegarde dans l'historique")

# ─────────────────────────────────────────────────────────────
# MULTI-MATCHS
# ─────────────────────────────────────────────────────────────
def show_multimatches(atp_data):
    st.markdown("<h2>📊 Multi-matchs</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_matches = st.number_input("Nombre de matchs", 2, MAX_MATCHES_ANALYSIS, 3, key="mm_n")
    with col2:
        use_ai = st.checkbox("Analyses IA", True, key="mm_use_ai")
    with col3:
        send_all = st.checkbox("📱 Envoyer tout sur Telegram", False, key="mm_send_all")
    
    if atp_data.empty:
        st.warning("Données non disponibles")
        return
    
    winner_col = 'winner_name' if 'winner_name' in atp_data.columns else None
    loser_col = 'loser_name' if 'loser_name' in atp_data.columns else None
    if not winner_col or not loser_col:
        st.warning("Colonnes joueurs non trouvées")
        return
    
    players = sorted(set(str(p).strip() for p in atp_data[winner_col].dropna().unique() if pd.notna(p)) |
                     set(str(p).strip() for p in atp_data[loser_col].dropna().unique() if pd.notna(p)))
    tournaments = sorted(atp_data['tourney_name'].dropna().unique()) if 'tourney_name' in atp_data.columns else []
    
    matches = []
    for i in range(n_matches):
        with st.expander(f"Match {i+1}", expanded=i==0):
            col1, col2, col3 = st.columns(3)
            with col1:
                p1 = st.selectbox(f"J1", players, key=f"mm_p1_{i}")
            with col2:
                p2_options = [p for p in players if p != p1]
                p2 = st.selectbox(f"J2", p2_options, key=f"mm_p2_{i}") if p2_options else None
            with col3:
                tourn = st.selectbox(f"Tournoi", tournaments, key=f"mm_tourn_{i}") if tournaments else None
            
            surface = "Hard"
            if tourn and 'surface' in atp_data.columns:
                s_df = atp_data[atp_data['tourney_name'] == tourn]['surface']
                if not s_df.empty:
                    surface = s_df.iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                odds1 = st.text_input(f"Cote {p1}", key=f"mm_odds1_{i}", placeholder="1.75")
            with col2:
                odds2 = st.text_input(f"Cote {p2}", key=f"mm_odds2_{i}", placeholder="2.10") if p2 else st.text_input(f"Cote J2", key=f"mm_odds2_{i}", placeholder="2.10")
            
            if surface in SURFACE_CONFIG:
                st.markdown(create_badge(f"{SURFACE_CONFIG[surface]['icon']} {surface}", SURFACE_CONFIG[surface]['color']), unsafe_allow_html=True)
            
            matches.append({
                'player1': p1.strip() if p1 else None, 
                'player2': p2.strip() if p2 else None,
                'tournament': tourn, 'surface': surface,
                'odds1': odds1, 'odds2': odds2,
            })
    
    if st.button(f"🔍 Analyser", key="mm_analyze", use_container_width=True):
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        valid_matches = [m for m in matches if m['player1'] and m['player2']]
        
        if not valid_matches:
            st.warning("Veuillez remplir au moins un match complet")
            return
        
        model_info = st.session_state.get('ml_model')
        player_stats = st.session_state.get('player_stats_cache')
        
        for i, match in enumerate(valid_matches):
            h2h = get_h2h_stats(atp_data, match['player1'], match['player2'])
            proba = calculate_probability(atp_data, match['player1'], match['player2'], match['surface'], h2h, model_info, player_stats)
            confidence = calculate_confidence(proba, match['player1'], match['player2'], h2h)
            ml_used = model_info is not None and player_stats is not None
            
            best_value = None
            if match['odds1'] and match['odds2']:
                try:
                    o1 = float(match['odds1'].replace(',', '.'))
                    o2 = float(match['odds2'].replace(',', '.'))
                    edge1 = proba - 1/o1
                    edge2 = (1 - proba) - 1/o2
                    if edge1 > edge2 and edge1 > MIN_EDGE_COMBINE:
                        best_value = {'joueur': match['player1'], 'edge': edge1, 'cote': o1, 'proba': proba}
                    elif edge2 > edge1 and edge2 > MIN_EDGE_COMBINE:
                        best_value = {'joueur': match['player2'], 'edge': edge2, 'cote': o2, 'proba': 1 - proba}
                except: 
                    pass
            
            ml_tag = '<span class="ml-badge">🤖 ML</span>' if ml_used else ''
            st.markdown(f"### Match {i+1}: {match['player1']} vs {match['player2']} {ml_tag}")
            st.markdown(create_result_card(match['player1'], match['player2'], proba, confidence), unsafe_allow_html=True)
            
            if best_value:
                st.success(f"✅ Value bet: {best_value['joueur']} @ {best_value['cote']:.2f} (edge: {best_value['edge']*100:+.1f}%)")
            
            if use_ai and GROQ_AVAILABLE:
                vb_txt = f"Value bet sur {best_value['joueur']}" if best_value else "Aucun value bet"
                prompt = f"Analyse ce match: {match['player1']} vs {match['player2']} sur {match['surface']}. Proba: {match['player1']} {proba:.1%}. {vb_txt}. 3 points clés."
                with st.spinner("Analyse IA..."):
                    ai = call_groq_api(prompt)
                    if ai:
                        with st.expander("🤖 Analyse IA"):
                            st.markdown(ai)
                        if send_all:
                            pred_data = {
                                'player1': match['player1'], 'player2': match['player2'],
                                'tournament': match['tournament'], 'surface': match['surface'],
                                'proba': proba, 'confidence': confidence,
                                'odds1': match['odds1'], 'odds2': match['odds2'],
                                'favori_modele': match['player1'] if proba >= 0.5 else match['player2'],
                                'best_value': best_value,
                                'ml_used': ml_used,
                                'date': datetime.now().isoformat(),
                                'statut': 'en_attente'
                            }
                            send_prediction_to_telegram(pred_data, ai)

# ─────────────────────────────────────────────────────────────
# COMBINÉS
# ─────────────────────────────────────────────────────────────
def show_combines(atp_data):
    st.markdown("<h2>🎰 Générateur de Combinés</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_matches = st.number_input("Matchs", 2, MAX_MATCHES_COMBINE, 3, key="comb_n")
    with col2:
        mise = st.number_input("Mise (€)", 1.0, 10000.0, 10.0, key="comb_mise")
    with col3:
        use_ai = st.checkbox("Analyses IA", True, key="comb_use_ai")
    with col4:
        send_tg = st.checkbox("📱 Envoyer Telegram", False, key="comb_send_tg")
    
    if atp_data.empty:
        st.warning("Données non disponibles")
        return
    
    winner_col = 'winner_name' if 'winner_name' in atp_data.columns else None
    loser_col = 'loser_name' if 'loser_name' in atp_data.columns else None
    if not winner_col or not loser_col:
        st.warning("Colonnes joueurs non trouvées")
        return
    
    players = sorted(set(str(p).strip() for p in atp_data[winner_col].dropna().unique() if pd.notna(p)) |
                     set(str(p).strip() for p in atp_data[loser_col].dropna().unique() if pd.notna(p)))
    tournaments = sorted(atp_data['tourney_name'].dropna().unique()) if 'tourney_name' in atp_data.columns else []
    
    matches = []
    st.markdown(f"### Saisie des {n_matches} matchs")
    
    for i in range(n_matches):
        with st.container():
            st.markdown(f"**Match {i+1}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                p1 = st.selectbox("J1", players, key=f"comb_p1_{i}", label_visibility="collapsed")
            with col2:
                p2_options = [p for p in players if p != p1]
                p2 = st.selectbox("J2", p2_options, key=f"comb_p2_{i}", label_visibility="collapsed") if p2_options else None
            with col3:
                tourn = st.selectbox("T", tournaments, key=f"comb_tourn_{i}", label_visibility="collapsed") if tournaments else None
            
            col1, col2 = st.columns(2)
            with col1:
                odds1 = st.text_input(f"Cote {p1}", key=f"comb_odds1_{i}", placeholder="1.75")
            with col2:
                odds2 = st.text_input(f"Cote {p2}", key=f"comb_odds2_{i}", placeholder="2.10") if p2 else st.text_input(f"Cote J2", key=f"comb_odds2_{i}", placeholder="2.10")
            
            surface = "Hard"
            if tourn and 'surface' in atp_data.columns:
                s_df = atp_data[atp_data['tourney_name'] == tourn]['surface']
                if not s_df.empty:
                    surface = s_df.iloc[0]
            
            if surface in SURFACE_CONFIG:
                st.markdown(create_badge(surface, SURFACE_CONFIG[surface]['color']), unsafe_allow_html=True)
            
            if i < n_matches - 1:
                st.markdown("---")
            
            matches.append({
                'player1': p1.strip() if p1 else None, 
                'player2': p2.strip() if p2 else None,
                'tournament': tourn, 'surface': surface,
                'odds1': odds1, 'odds2': odds2,
            })
    
    if st.button("🎯 Générer le combiné", key="comb_generate", use_container_width=True):
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        selections = []
        
        model_info = st.session_state.get('ml_model')
        player_stats = st.session_state.get('player_stats_cache')
        
        with st.spinner("Analyse des matchs..."):
            for match in matches:
                if match['player1'] and match['player2'] and match['odds1'] and match['odds2']:
                    try:
                        o1 = float(match['odds1'].replace(',', '.'))
                        o2 = float(match['odds2'].replace(',', '.'))
                        h2h = get_h2h_stats(atp_data, match['player1'], match['player2'])
                        proba = calculate_probability(atp_data, match['player1'], match['player2'], match['surface'], h2h, model_info, player_stats)
                        edge1 = proba - 1/o1
                        edge2 = (1 - proba) - 1/o2
                        
                        if edge1 > MIN_EDGE_COMBINE and proba >= MIN_PROBA_COMBINE:
                            selections.append({
                                'match': f"{match['player1']} vs {match['player2']}",
                                'joueur': match['player1'], 'proba': proba, 'cote': o1, 'edge': edge1
                            })
                        elif edge2 > MIN_EDGE_COMBINE and (1 - proba) >= MIN_PROBA_COMBINE:
                            selections.append({
                                'match': f"{match['player1']} vs {match['player2']}",
                                'joueur': match['player2'], 'proba': 1 - proba, 'cote': o2, 'edge': edge2
                            })
                    except: 
                        pass
        
        if len(selections) >= 2:
            selections.sort(key=lambda x: x['edge'], reverse=True)
            selected = selections[:min(MAX_SELECTIONS_COMBINE, len(selections))]
            proba_combi = 1.0
            cote_combi = 1.0
            for sel in selected:
                proba_combi *= sel['proba']
                cote_combi *= sel['cote']
            gain = mise * cote_combi
            esperance = proba_combi * gain - mise
            kelly = (proba_combi * cote_combi - 1) / (cote_combi - 1) if cote_combi > 1 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                c = COLORS['success'] if proba_combi >= 0.3 else COLORS['warning'] if proba_combi >= 0.15 else COLORS['danger']
                st.markdown(create_metric("Probabilité", f"{proba_combi:.1%}", "", c), unsafe_allow_html=True)
            with col2:
                st.markdown(create_metric("Cote", f"{cote_combi:.2f}"), unsafe_allow_html=True)
            with col3:
                c = COLORS['success'] if esperance > 0 else COLORS['danger']
                st.markdown(create_metric("Espérance", f"{esperance:+.2f}€", "", c), unsafe_allow_html=True)
            with col4:
                st.markdown(create_metric("Kelly", f"{kelly*100:.1f}", "%"), unsafe_allow_html=True)
            
            st.markdown("### 📋 Sélections")
            df_sel = pd.DataFrame([{
                '#': i+1, 'Joueur': s['joueur'], 'Match': s['match'],
                'Proba': f"{s['proba']:.1%}", 'Cote': f"{s['cote']:.2f}",
                'Edge': f"{s['edge']*100:+.1f}%"
            } for i, s in enumerate(selected)])
            st.dataframe(df_sel, use_container_width=True, hide_index=True)
            
            combine_data = {
                'selections': selected, 'proba_globale': proba_combi,
                'cote_globale': cote_combi, 'mise': mise,
                'gain_potentiel': gain, 'esperance': esperance,
                'kelly': kelly, 'nb_matches': len(selected),
                'ml_used': model_info is not None
            }
            save_combine(combine_data)
            st.success("✅ Combiné sauvegardé !")
            
            if send_tg and GROQ_AVAILABLE and use_ai:
                with st.spinner("Analyse IA du combiné..."):
                    prompt = f"Analyse ce combiné de {len(selected)} matchs. Proba: {proba_combi:.1%}, cote: {cote_combi:.2f}, espérance: {esperance:+.2f}€. Sélections: {[s['joueur'] for s in selected]}. Avis en 3 points."
                    ai = call_groq_api(prompt)
                    if ai:
                        send_combine_to_telegram(combine_data, ai)
        else:
            st.warning(f"⚠️ Pas assez de sélections valides ({len(selections)} trouvées)")

# ─────────────────────────────────────────────────────────────
# HISTORIQUE
# ─────────────────────────────────────────────────────────────
def show_history():
    st.markdown("<h2>📜 Historique</h2>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📋 Prédictions", "🎰 Combinés"])
    
    with tab1:
        history = load_history()
        if history:
            filtered = history[::-1][:20]
            for pred in filtered:
                ml_tag = ' <span class="ml-badge">🤖 ML</span>' if pred.get('ml_used') else ''
                with st.expander(f"{pred.get('date', '')[:16]} - {pred.get('player1','?')} vs {pred.get('player2','?')}{ml_tag}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(create_metric("Tournoi", pred.get('tournament','—')), unsafe_allow_html=True)
                    with col2:
                        st.markdown(create_metric("Surface", pred.get('surface','—')), unsafe_allow_html=True)
                    with col3:
                        proba = pred.get('proba', 0.5)
                        st.markdown(create_metric("Probabilité", f"{proba:.1%}"), unsafe_allow_html=True)
                    
                    st.markdown(create_progress_bar(proba), unsafe_allow_html=True)
                    
                    if pred.get('best_value'):
                        st.success(f"🎯 Value bet détecté")
                    
                    if pred.get('statut') == 'en_attente':
                        col_b1, col_b2 = st.columns(2)
                        with col_b1:
                            if st.button(f"✅ {pred['player1']} gagne", key=f"hist_win1_{pred.get('id','')}"):
                                update_prediction_status(pred.get('id',''), 'joueur1_gagne')
                                st.rerun()
                        with col_b2:
                            if st.button(f"✅ {pred['player2']} gagne", key=f"hist_win2_{pred.get('id','')}"):
                                update_prediction_status(pred.get('id',''), 'joueur2_gagne')
                                st.rerun()
        else:
            st.info("Aucune prédiction")
    
    with tab2:
        combines = load_combines()
        if combines:
            for comb in combines[::-1][:10]:
                ml_tag = ' 🤖' if comb.get('ml_used') else ''
                with st.expander(f"{comb.get('date','')[:16]} - {comb.get('nb_matches',0)} matchs - Proba {comb.get('proba_globale',0):.1%}{ml_tag}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(create_metric("Probabilité", f"{comb.get('proba_globale',0):.1%}"), unsafe_allow_html=True)
                    with col2:
                        st.markdown(create_metric("Cote", f"{comb.get('cote_globale',0):.2f}"), unsafe_allow_html=True)
                    with col3:
                        st.markdown(create_metric("Espérance", f"{comb.get('esperance',0):+.2f}€"), unsafe_allow_html=True)
        else:
            st.info("Aucun combiné")

# ─────────────────────────────────────────────────────────────
# STATISTIQUES
# ─────────────────────────────────────────────────────────────
def show_statistics():
    st.markdown("<h2>📈 Statistiques</h2>", unsafe_allow_html=True)
    stats = load_user_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_metric("Prédictions", stats.get('total_predictions', 0)), unsafe_allow_html=True)
    with col2:
        accuracy = (stats.get('correct_predictions', 0) / max(stats.get('total_predictions', 1), 1)) * 100
        st.markdown(create_metric("Précision", f"{accuracy:.1f}", "%"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric("Combinés", stats.get('total_combines', 0)), unsafe_allow_html=True)
    with col4:
        profit = stats.get('total_won', 0) - stats.get('total_invested', 0)
        st.markdown(create_metric("Profit", f"{profit:+.2f}", "€", COLORS['success'] if profit > 0 else COLORS['danger']), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────────
def show_telegram():
    st.markdown("<h2>📱 Messages Telegram</h2>", unsafe_allow_html=True)
    
    token, chat_id = get_telegram_config()
    if not token or not chat_id:
        st.warning("⚠️ Telegram non configuré. Ajoute les secrets TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID")
        return
    
    st.success(f"✅ Telegram configuré (Chat ID: {chat_id})")
    
    tab1, tab2, tab3 = st.tabs(["✏️ Message simple", "📊 Stats", "⚡ Test"])
    
    with tab1:
        send_custom_message()
    
    with tab2:
        if st.button("📊 Envoyer les statistiques", key="tg_send_stats", use_container_width=True):
            with st.spinner("Envoi en cours..."):
                if send_stats_to_telegram():
                    st.success("✅ Stats envoyées !")
                else:
                    st.error("❌ Échec de l'envoi")
    
    with tab3:
        if st.button("🔧 Tester la connexion", key="tg_test", use_container_width=True):
            with st.spinner("Test en cours..."):
                success, msg = test_telegram_connection()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
def show_configuration():
    st.markdown("<h2>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    st.markdown("### 🤖 Intelligence Artificielle")
    groq_status = "✅ Connecté" if get_groq_key() else "❌ Non configuré"
    st.markdown(f"**Groq API:** {groq_status}")
    
    st.markdown("### 📱 Telegram")
    token, chat_id = get_telegram_config()
    if token and chat_id:
        st.success("✅ Telegram configuré")
        st.code(f"Chat ID: {chat_id}")
    else:
        st.warning("⚠️ Telegram non configuré")
        st.markdown("""
        **Configuration:**
        1. Va sur Telegram, cherche @BotFather
        2. Crée un bot avec /newbot
        3. Ajoute dans les secrets Streamlit:
        ```toml
        TELEGRAM_BOT_TOKEN = "ton_token"
        TELEGRAM_CHAT_ID = "ton_chat_id"
        ```
        """)
    
    st.markdown("### 🗑️ Gestion des données")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Effacer prédictions", key="config_clear_pred"):
            if HIST_FILE.exists():
                HIST_FILE.unlink()
                st.rerun()
    with col2:
        if st.button("🗑️ Effacer combinés", key="config_clear_comb"):
            if COMB_HIST_FILE.exists():
                COMB_HIST_FILE.unlink()
                st.rerun()
    with col3:
        if st.button("🗑️ Réinit. stats", key="config_clear_stats"):
            if USER_STATS_FILE.exists():
                USER_STATS_FILE.unlink()
                st.rerun()

# ─────────────────────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
