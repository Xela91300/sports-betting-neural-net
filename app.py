import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime, timedelta
import time
import hashlib
import warnings
import asyncio
import nest_asyncio
import os
import requests
import gzip
import plotly.express as px
import plotly.graph_objects as go
import shutil
import random

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

nest_asyncio.apply()
warnings.filterwarnings('ignore')

ROOT_DIR   = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR   = ROOT_DIR / "src" / "data" / "raw" / "tml-tennis"
HIST_DIR   = ROOT_DIR / "history"
BACKUP_DIR = ROOT_DIR / "backups"

for d in [MODELS_DIR, DATA_DIR, HIST_DIR, BACKUP_DIR]:
    d.mkdir(exist_ok=True, parents=True)

HIST_FILE        = HIST_DIR / "predictions_history.json"
COMB_HIST_FILE   = HIST_DIR / "combines_history.json"
USER_STATS_FILE  = HIST_DIR / "user_stats.json"
ACHIEVEMENTS_FILE= HIST_DIR / "achievements.json"
TRENDS_FILE      = HIST_DIR / "trends.json"

SURFACES = ["Hard", "Clay", "Grass"]
MIN_EDGE_COMBINE = 0.02
MAX_MATCHES_ANALYSIS = 30

ACHIEVEMENTS = {
    'first_win':          {'name': '🎯 Première victoire',  'desc': 'Première prédiction gagnante',         'icon': '🎯'},
    'streak_5':           {'name': '🔥 En forme',           'desc': '5 gagnantes consécutives',             'icon': '🔥'},
    'streak_10':          {'name': '⚡ Imbattable',          'desc': '10 gagnantes consécutives',            'icon': '⚡'},
    'pred_100':           {'name': '🏆 Expert',             'desc': '100 prédictions',                      'icon': '🏆'},
    'value_master':       {'name': '💎 Value Master',        'desc': '10 value bets gagnants',               'icon': '💎'},
    'surface_specialist': {'name': '🌍 Spécialiste surface', 'desc': 'Gagnant sur les 3 surfaces',           'icon': '🌍'},
}

TOURNAMENTS_DB = {
    "Australian Open": "Hard", "Roland Garros": "Clay",
    "Wimbledon": "Grass",      "US Open": "Hard",
    "Nitto ATP Finals": "Hard",
    "Indian Wells Masters": "Hard", "Miami Open": "Hard",
    "Monte-Carlo Masters": "Clay",  "Madrid Open": "Clay",
    "Italian Open": "Clay",   "Canadian Open": "Hard",
    "Cincinnati Masters": "Hard",   "Shanghai Masters": "Hard",
    "Paris Masters": "Hard",
    "Rotterdam Open": "Hard",  "Rio Open": "Clay",
    "Dubai Tennis Championships": "Hard", "Mexican Open": "Hard",
    "Barcelona Open": "Clay",  "Halle Open": "Grass",
    "Queen's Club Championships": "Grass", "Hamburg Open": "Clay",
    "Washington Open": "Hard", "China Open": "Hard",
    "Japan Open": "Hard",      "Vienna Open": "Hard",
    "Swiss Indoors": "Hard",   "Dallas Open": "Hard",
    "Qatar Open": "Hard",      "St. Petersburg Open": "Hard",
    "Adelaide International": "Hard", "Auckland Open": "Hard",
    "Brisbane International": "Hard", "Cordoba Open": "Clay",
    "Buenos Aires": "Clay",    "Delray Beach": "Hard",
    "Marseille Open": "Hard",  "Santiago": "Clay",
    "Houston": "Clay",         "Marrakech": "Clay",
    "Estoril": "Clay",         "Munich": "Clay",
    "Geneva": "Clay",          "Lyon": "Clay",
    "Stuttgart": "Grass",      "Mallorca": "Grass",
    "Eastbourne": "Grass",     "Newport": "Grass",
    "Atlanta": "Hard",         "Croatia Open Umag": "Clay",
    "Nordea Open": "Clay",     "Kitzbühel": "Clay",
    "Los Cabos": "Hard",       "Winston-Salem": "Hard",
    "Chengdu Open": "Hard",    "Sofia": "Hard",
    "Metz": "Hard",            "San Diego": "Hard",
    "Seoul": "Hard",           "Tel Aviv": "Hard",
    "Florence": "Hard",        "Antwerp": "Hard",
    "Stockholm": "Hard",       "Belgrade Open": "Clay",
    "Autre tournoi": "Hard",
}

TOURNAMENT_LEVEL = {
    "Australian Open": ("G", 5), "Roland Garros": ("G", 5),
    "Wimbledon": ("G", 5),       "US Open": ("G", 5),
    "Nitto ATP Finals": ("F", 3),
    "Indian Wells Masters": ("M", 3), "Miami Open": ("M", 3),
    "Monte-Carlo Masters": ("M", 3),  "Madrid Open": ("M", 3),
    "Italian Open": ("M", 3),   "Canadian Open": ("M", 3),
    "Cincinnati Masters": ("M", 3),   "Shanghai Masters": ("M", 3),
    "Paris Masters": ("M", 3),
}

TOURNAMENT_ALIASES = {
    "acapulco": "Mexican Open", "mexico": "Mexican Open",
    "australian": "Australian Open", "melbourne": "Australian Open",
    "roland garros": "Roland Garros", "french": "Roland Garros",
    "wimbledon": "Wimbledon", "wimby": "Wimbledon",
    "us open": "US Open", "new york": "US Open",
    "indian wells": "Indian Wells Masters", "miami": "Miami Open",
    "monte carlo": "Monte-Carlo Masters", "madrid": "Madrid Open",
    "rome": "Italian Open", "italy": "Italian Open",
    "canada": "Canadian Open", "cincinnati": "Cincinnati Masters",
    "shanghai": "Shanghai Masters", "paris masters": "Paris Masters", "bercy": "Paris Masters",
    "rotterdam": "Rotterdam Open", "dubai": "Dubai Tennis Championships",
    "barcelona": "Barcelona Open", "halle": "Halle Open",
    "queens": "Queen's Club Championships", "hamburg": "Hamburg Open",
    "washington": "Washington Open", "beijing": "China Open",
    "tokyo": "Japan Open", "vienna": "Vienna Open", "basel": "Swiss Indoors",
}

COLORS = {
    "primary": "#00DFA2", "success": "#00DFA2", "warning": "#FFB200",
    "danger": "#FF3B3F", "gray": "#6C7A89",
    "surface_hard": "#0079FF", "surface_clay": "#E67E22", "surface_grass": "#00DFA2",
}
SURFACE_CONFIG = {
    "Hard":  {"color": COLORS["surface_hard"],  "icon": "🟦"},
    "Clay":  {"color": COLORS["surface_clay"],  "icon": "🟧"},
    "Grass": {"color": COLORS["surface_grass"], "icon": "🟩"},
}

STATUS_OPTIONS = {
    "en_attente": "⏳ En attente", "gagne": "✅ Gagné",
    "perdu": "❌ Perdu", "annule": "⚠️ Annulé"
}

def get_tournament_surface(name):
    return TOURNAMENTS_DB.get(name, "Hard")

def get_tournament_level(name):
    return TOURNAMENT_LEVEL.get(name, ("A", 3))

def find_tournament(search):
    if not search:
        return None
    sl = search.lower().strip()
    if sl in TOURNAMENT_ALIASES:
        return TOURNAMENT_ALIASES[sl]
    for t in TOURNAMENTS_DB:
        if sl == t.lower():
            return t
    matches = [t for t in TOURNAMENTS_DB if sl in t.lower()]
    return min(matches, key=len) if matches else None

# ─────────────────────────────────────────────────────────────
# BACKUP & ACHIEVEMENTS
# ─────────────────────────────────────────────────────────────
def auto_backup():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    for f in [HIST_FILE, COMB_HIST_FILE, USER_STATS_FILE]:
        if f.exists():
            try:
                shutil.copy(f, BACKUP_DIR / f"{f.stem}_{ts}{f.suffix}")
            except: pass
    try:
        for b in BACKUP_DIR.glob("*"):
            if (datetime.now() - datetime.fromtimestamp(b.stat().st_mtime)).days > 30:
                b.unlink()
    except: pass

def load_achievements():
    if not ACHIEVEMENTS_FILE.exists(): return {}
    try:
        with open(ACHIEVEMENTS_FILE) as f: return json.load(f)
    except: return {}

def save_achievements(a):
    try:
        with open(ACHIEVEMENTS_FILE, 'w') as f: json.dump(a, f)
    except: pass

def check_and_unlock_achievements():
    stats = load_user_stats(); history = load_history(); ach = load_achievements(); new = []
    if stats.get('correct_predictions', 0) >= 1 and 'first_win' not in ach:
        ach['first_win'] = {'unlocked_at': datetime.now().isoformat()}; new.append(ACHIEVEMENTS['first_win'])
    if stats.get('best_streak', 0) >= 5 and 'streak_5' not in ach:
        ach['streak_5'] = {'unlocked_at': datetime.now().isoformat()}; new.append(ACHIEVEMENTS['streak_5'])
    if stats.get('best_streak', 0) >= 10 and 'streak_10' not in ach:
        ach['streak_10'] = {'unlocked_at': datetime.now().isoformat()}; new.append(ACHIEVEMENTS['streak_10'])
    if stats.get('total_predictions', 0) >= 100 and 'pred_100' not in ach:
        ach['pred_100'] = {'unlocked_at': datetime.now().isoformat()}; new.append(ACHIEVEMENTS['pred_100'])
    value_wins = sum(1 for p in history if p.get('best_value') and p.get('statut') == 'gagne')
    if value_wins >= 10 and 'value_master' not in ach:
        ach['value_master'] = {'unlocked_at': datetime.now().isoformat()}; new.append(ACHIEVEMENTS['value_master'])
    surfaces_won = {p.get('surface') for p in history if p.get('statut') == 'gagne'}
    if len(surfaces_won) >= 3 and 'surface_specialist' not in ach:
        ach['surface_specialist'] = {'unlocked_at': datetime.now().isoformat()}; new.append(ACHIEVEMENTS['surface_specialist'])
    if new:
        save_achievements(ach)
    return new

# ─────────────────────────────────────────────────────────────
# TELEGRAM - VERSION AMÉLIORÉE
# ─────────────────────────────────────────────────────────────
def get_telegram_config():
    try:
        return st.secrets["TELEGRAM_BOT_TOKEN"], str(st.secrets["TELEGRAM_CHAT_ID"])
    except:
        t = os.environ.get("TELEGRAM_BOT_TOKEN"); c = os.environ.get("TELEGRAM_CHAT_ID")
        return (t, c) if t and c else (None, None)

def send_telegram_message(message, parse_mode='HTML'):
    token, chat_id = get_telegram_config()
    if not token or not chat_id: 
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={'chat_id': chat_id, 'text': message,
                                'parse_mode': parse_mode, 'disable_web_page_preview': True},
                          timeout=15)
        return r.status_code == 200
    except Exception as e:
        st.error(f"Erreur Telegram: {e}")
        return False

def format_prediction_message(pred_data, bet_suggestions=None, ai_comment=None):
    """Formate un message de prédiction pour Telegram avec style pro"""
    proba = pred_data.get('proba', 0.5)
    bar_length = 10
    filled = int(proba * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    surf_icon = {'Hard': '🟦', 'Clay': '🟧', 'Grass': '🟩'}.get(pred_data.get('surface', ''), '🎾')
    ml_tag = "🤖 " if pred_data.get('ml_used') else ""
    gagnant = pred_data.get('favori', '?')
    
    message = f"""
<b>{ml_tag}🎾 PRÉDICTION TENNISIQ</b>

<b>Match:</b> {pred_data.get('player1','?')} vs {pred_data.get('player2','?')}
<b>Tournoi:</b> {pred_data.get('tournament','Inconnu')}
<b>Surface:</b> {surf_icon} {pred_data.get('surface','?')}

<b>📊 ANALYSE DU MATCH:</b>
<code>{bar}</code>  {proba:.1%} / {1-proba:.1%}

• {pred_data.get('player1','J1')}: <b>{proba:.1%}</b>
• {pred_data.get('player2','J2')}: <b>{1-proba:.1%}</b>

<b>🏆 GAGNANT PRÉDIT: <u>{gagnant}</u></b>
<b>Confiance:</b> {'🟢' if pred_data.get('confidence',0)>=70 else '🟡' if pred_data.get('confidence',0)>=50 else '🔴'} {pred_data.get('confidence',0):.0f}/100
"""
    if pred_data.get('odds1') and pred_data.get('odds2'):
        message += f"""
<b>Cotes:</b>
• {pred_data.get('player1','J1')}: <code>{pred_data.get('odds1')}</code>
• {pred_data.get('player2','J2')}: <code>{pred_data.get('odds2')}</code>
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
    
    message += f"\n\n#TennisIQ #{pred_data.get('surface','Tennis')}"
    return message

def format_stats_message():
    """Formate un message de statistiques pour Telegram avec style pro"""
    stats = load_user_stats()
    history = load_history()
    
    total = stats.get('total_predictions', 0)
    correct = stats.get('correct_predictions', 0)
    incorrect = stats.get('incorrect_predictions', 0)
    annules = stats.get('annules_predictions', 0)
    
    total_valide = correct + incorrect
    accuracy = (correct / total_valide * 100) if total_valide > 0 else 0
    
    bar_length = 10
    filled = int(accuracy / 10)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    recent = [p for p in history[-20:] if p.get('statut') in ['gagne', 'perdu']]
    recent_correct = sum(1 for p in recent if p.get('statut') == 'gagne')
    recent_acc = (recent_correct / len(recent) * 100) if recent else 0
    
    diff = recent_acc - accuracy
    
    # Calcul des performances par type de pari
    value_bets_total = sum(1 for p in history if p.get('best_value'))
    value_bets_won = sum(1 for p in history if p.get('best_value') and p.get('statut') == 'gagne')
    value_accuracy = (value_bets_won / value_bets_total * 100) if value_bets_total > 0 else 0
    
    favorites = sum(1 for p in history if p.get('favori') and p.get('statut') in ['gagne', 'perdu'])
    favorites_won = sum(1 for p in history if p.get('favori') and p.get('statut') == 'gagne' and 
                       ((p.get('proba',0.5) >= 0.5 and p.get('statut') == 'gagne') or
                        (p.get('proba',0.5) < 0.5 and p.get('statut') == 'gagne')))
    fav_accuracy = (favorites_won / favorites * 100) if favorites > 0 else 0
    
    message = f"""
<b>📊 STATISTIQUES TENNISIQ</b>

<b>🎯 Performance globale:</b>
<code>{bar}</code>  {accuracy:.1f}%

<b>📈 Détail:</b>
• Total prédictions: <b>{total}</b>
• ✅ Gagnées: <b>{correct}</b> ({accuracy:.1f}%)
• ❌ Perdues: <b>{incorrect}</b>
• ⚠️ Annulées: <b>{annules}</b>

<b>🔥 Dernières 20:</b>
• Correctes: <b>{recent_correct}/{len(recent)}</b>
• Précision: <b>{recent_acc:.1f}%</b> ({diff:+.1f}% vs globale)

<b>💎 Value Bets:</b>
• Total: <b>{value_bets_total}</b>
• Gagnés: <b>{value_bets_won}</b> ({value_accuracy:.1f}%)

<b>🏆 Favoris:</b>
• Total: <b>{favorites}</b>
• Gagnés: <b>{favorites_won}</b> ({fav_accuracy:.1f}%)

<b>🔥 Records:</b>
• Série actuelle: <b>{stats.get('current_streak', 0)}</b>
• Meilleure série: <b>{stats.get('best_streak', 0)}</b>

📅 Mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}

#TennisIQ #Stats
"""
    return message

def send_prediction_to_telegram(pred_data, bet_suggestions=None, ai_comment=None):
    return send_telegram_message(format_prediction_message(pred_data, bet_suggestions, ai_comment))

def send_stats_to_telegram():
    return send_telegram_message(format_stats_message())

def test_telegram_connection():
    token, chat_id = get_telegram_config()
    if not token: 
        return False, "❌ Token manquant"
    if not chat_id: 
        return False, "❌ Chat ID manquant"
    try:
        msg = f"""
<b>✅ TEST DE CONNEXION RÉUSSI !</b>

📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}
🤖 Bot TennisIQ opérationnel

📊 Statistiques actuelles:
• Prédictions: {len(load_history())}
• Précision: {calculate_global_accuracy():.1f}%

#TennisIQ #Test
"""
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={'chat_id': chat_id, 'text': msg, 'parse_mode': 'HTML'}, 
                          timeout=15)
        if r.status_code == 200:
            return True, "✅ Connexion réussie ! Message de test envoyé."
        else:
            return False, f"❌ Erreur: {r.text}"
    except Exception as e:
        return False, f"❌ Exception: {e}"

def send_custom_message():
    """Envoie un message personnalisé sur Telegram"""
    st.markdown("### 📝 Message personnalisé")
    with st.form("custom_msg_form"):
        title = st.text_input("Titre", "Message TennisIQ")
        content = st.text_area("Contenu", height=100, placeholder="Écris ton message ici...")
        urgent = st.checkbox("🔴 Urgent")
        
        col1, col2 = st.columns(2)
        with col1:
            include_stats = st.checkbox("📊 Inclure les stats")
        with col2:
            include_time = st.checkbox("🕐 Inclure la date", True)
        
        if st.form_submit_button("📤 Envoyer le message") and content:
            urgent_tag = "🔴 URGENT - " if urgent else ""
            date_tag = f"\n\n📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}" if include_time else ""
            stats_tag = f"\n\n{format_stats_message()}" if include_stats else ""
            
            msg = f"<b>{urgent_tag}{title}</b>\n\n{content}{date_tag}{stats_tag}\n\n#TennisIQ"
            
            if send_telegram_message(msg):
                st.success("✅ Message envoyé avec succès sur Telegram !")
            else:
                st.error("❌ Échec de l'envoi. Vérifie la configuration Telegram.")

# ─────────────────────────────────────────────────────────────
# GROQ IA
# ─────────────────────────────────────────────────────────────
def get_groq_key():
    try: return st.secrets["GROQ_API_KEY"]
    except: return os.environ.get("GROQ_API_KEY")

def call_groq_api(prompt):
    key = get_groq_key()
    if not key: return None
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json={"model": "llama-3.3-70b-versatile",
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.3, "max_tokens": 500},
                          timeout=30)
        return r.json()['choices'][0]['message']['content'] if r.status_code==200 else None
    except: return None

def analyze_match_with_ai(player1, player2, surface, tournament, proba, best_value=None):
    gagnant = player1 if proba >= 0.5 else player2
    perdant  = player2 if proba >= 0.5 else player1
    vb_txt = f" Value bet: {best_value['joueur']} @ {best_value['cote']:.2f} (edge +{best_value['edge']*100:.1f}%)" if best_value else ""
    return call_groq_api(f"""Analyse ce match de tennis:
Match: {player1} vs {player2} | Tournoi: {tournament} | Surface: {surface}
Proba: {player1} {proba:.1%} - {player2} {1-proba:.1%}
GAGNANT PRÉDIT: {gagnant} ({max(proba,1-proba):.1%}){vb_txt}

4 points: (1) Pourquoi {gagnant} est favori (2) Faiblesses de {perdant}
(3) Conseil pari{' / ' + vb_txt if best_value else ''} (4) Pronostic final.
Sois direct, concis, en français.""")

# ─────────────────────────────────────────────────────────────
# MODÈLE ML
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_saved_model():
    model_path = MODELS_DIR / "tennis_ml_model_complete.pkl"
    if model_path.exists():
        try: return joblib.load(model_path)
        except: return None
    try:
        with st.spinner("📥 Téléchargement du modèle..."):
            r = requests.get(
                "https://github.com/Xela91300/sports-betting-neural-net/releases/download/v1.0.0/tennis_ml_model_complete.pkl.gz",
                timeout=60)
            if r.status_code == 200:
                tmp = MODELS_DIR / "model_temp.pkl.gz"
                tmp.write_bytes(r.content)
                import gzip
                with gzip.open(tmp, "rb") as f:
                    model_info = joblib.load(f)
                joblib.dump(model_info, model_path)
                tmp.unlink()
                return model_info
    except: pass
    return None

def _extract_all_features(player_stats, p1, p2, surface, level='A', best_of=3, h2h_ratio=0.5):
    """Extrait les 21 features pour le modèle ML"""
    s1 = player_stats.get(p1, {})
    s2 = player_stats.get(p2, {})

    r1 = max(s1.get('rank', 500.0), 1.0)
    r2 = max(s2.get('rank', 500.0), 1.0)
    log_rank_ratio = float(np.log(r2 / r1))

    pts_diff = (s1.get('rank_points', 0) - s2.get('rank_points', 0)) / 5000.0
    age_diff = float(s1.get('age', 25) - s2.get('age', 25))

    surf_clay  = 1.0 if surface == 'Clay'  else 0.0
    surf_grass = 1.0 if surface == 'Grass' else 0.0
    surf_hard  = 1.0 if surface == 'Hard'  else 0.0

    level_gs  = 1.0 if level == 'G' else 0.0
    level_m   = 1.0 if level == 'M' else 0.0
    best_of_5 = 1.0 if best_of == 5 else 0.0

    surf_wr_diff   = float(s1.get('surface_wr', {}).get(surface, 0.5) - s2.get('surface_wr', {}).get(surface, 0.5))
    career_wr_diff = float(s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5))
    recent_form_diff = float(s1.get('recent_form', 0.5) - s2.get('recent_form', 0.5))

    sp1 = s1.get('serve_pct', {})
    sp2 = s2.get('serve_pct', {})
    sr1 = s1.get('serve_raw', {})
    sr2 = s2.get('serve_raw', {})

    ace_diff = (sr1.get('ace', 0) - sr2.get('ace', 0)) / 10.0
    df_diff  = (sr1.get('df',  0) - sr2.get('df',  0)) / 5.0

    pct_1st_in_diff  = float(sp1.get('pct_1st_in',  0) - sp2.get('pct_1st_in',  0))
    pct_1st_won_diff = float(sp1.get('pct_1st_won', 0) - sp2.get('pct_1st_won', 0))
    pct_2nd_won_diff = float(sp1.get('pct_2nd_won', 0) - sp2.get('pct_2nd_won', 0))
    pct_bp_saved_diff= float(sp1.get('pct_bp_saved',0) - sp2.get('pct_bp_saved',0))

    days_diff    = float(s1.get('days_since_last', 30) - s2.get('days_since_last', 30))
    fatigue_diff = float(s1.get('fatigue', 0) - s2.get('fatigue', 0))

    features = [
        log_rank_ratio, pts_diff, age_diff,
        surf_clay, surf_grass, surf_hard,
        level_gs, level_m, best_of_5,
        surf_wr_diff, career_wr_diff, recent_form_diff, h2h_ratio,
        ace_diff, df_diff,
        pct_1st_in_diff, pct_1st_won_diff, pct_2nd_won_diff, pct_bp_saved_diff,
        days_diff, fatigue_diff
    ]
    return np.nan_to_num(np.array(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

def predict_with_ml_model(model_info, player1, player2, surface='Hard',
                           tournament='', h2h_ratio=0.5):
    if model_info is None:
        return None
    try:
        model        = model_info.get('model')
        scaler       = model_info.get('scaler')
        player_stats = model_info.get('player_stats', {})

        if model is None or scaler is None:
            return None

        if player1 not in player_stats or player2 not in player_stats:
            return None

        level, best_of = get_tournament_level(tournament)

        feat = _extract_all_features(
            player_stats, player1, player2,
            surface, level, best_of, h2h_ratio
        )

        feat_scaled = scaler.transform(feat.reshape(1, -1))
        proba = float(model.predict_proba(feat_scaled)[0][1])
        return max(0.05, min(0.95, proba))

    except Exception as e:
        return None

# ─────────────────────────────────────────────────────────────
# DONNÉES ATP
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_atp_data():
    if not DATA_DIR.exists(): return []
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files: return []
    all_players = set()
    pb = st.progress(0); status = st.empty()
    for idx, f in enumerate(csv_files):
        if 'wta' in f.name.lower(): continue
        status.text(f"Chargement: {f.name}")
        try:
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(f, encoding=enc, usecols=['winner_name','loser_name'], on_bad_lines='skip')
                    all_players.update(df['winner_name'].dropna().astype(str).str.strip())
                    all_players.update(df['loser_name'].dropna().astype(str).str.strip())
                    break
                except: continue
        except: pass
        pb.progress((idx+1)/len(csv_files))
    pb.empty(); status.empty()
    return sorted(p for p in all_players if p and p.lower() != 'nan' and len(p) > 1)

@st.cache_data(ttl=3600)
def get_h2h_stats_df():
    if not DATA_DIR.exists(): return pd.DataFrame()
    dfs = []
    for f in list(DATA_DIR.glob("*.csv"))[:20]:
        if 'wta' in f.name.lower(): continue
        try:
            df = pd.read_csv(f, encoding='utf-8', usecols=['winner_name','loser_name'], on_bad_lines='skip')
            df['winner_name'] = df['winner_name'].astype(str).str.strip()
            df['loser_name']  = df['loser_name'].astype(str).str.strip()
            dfs.append(df)
        except: continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def get_h2h_stats(player1, player2):
    df = get_h2h_stats_df()
    if df.empty: return None
    p1, p2 = player1.strip(), player2.strip()
    h2h = df[((df['winner_name']==p1)&(df['loser_name']==p2))|
             ((df['winner_name']==p2)&(df['loser_name']==p1))]
    if len(h2h) == 0: return None
    return {
        'total_matches': len(h2h),
        f'{p1}_wins': len(h2h[h2h['winner_name']==p1]),
        f'{p2}_wins': len(h2h[h2h['winner_name']==p2]),
    }

def calculate_h2h_ratio(h2h, player1):
    if not h2h or h2h.get('total_matches', 0) == 0:
        return 0.5
    wins1 = h2h.get(f'{player1}_wins', 0)
    return wins1 / h2h['total_matches']

def calculate_probability(player1, player2, surface, tournament='', h2h=None, model_info=None):
    h2h_ratio = calculate_h2h_ratio(h2h, player1)

    if model_info:
        ml_proba = predict_with_ml_model(
            model_info, player1, player2, surface,
            tournament=tournament, h2h_ratio=h2h_ratio
        )
        if ml_proba is not None:
            return ml_proba, True

    proba = 0.5
    if h2h and h2h.get('total_matches', 0) > 0:
        proba += (h2h_ratio - 0.5) * 0.3
    return max(0.05, min(0.95, proba)), False

def calculate_confidence(proba, h2h=None):
    conf = 50.0
    if h2h and h2h.get('total_matches', 0) >= 3: conf += 10
    conf += abs(proba - 0.5) * 40
    return min(100.0, conf)

def calculate_global_accuracy():
    stats = load_user_stats()
    tv = stats.get('correct_predictions',0) + stats.get('incorrect_predictions',0)
    return (stats.get('correct_predictions',0)/tv*100) if tv > 0 else 0

# ─────────────────────────────────────────────────────────────
# HISTORIQUE & STATS
# ─────────────────────────────────────────────────────────────
def load_history():
    if not HIST_FILE.exists(): return []
    try:
        with open(HIST_FILE,'r',encoding='utf-8') as f: return json.load(f)
    except: return []

def save_prediction(pred_data):
    try:
        h = load_history()
        pred_data['id'] = hashlib.md5(f"{datetime.now()}{pred_data.get('player1','')}".encode()).hexdigest()[:8]
        pred_data['statut'] = 'en_attente'
        h.append(pred_data)
        with open(HIST_FILE,'w',encoding='utf-8') as f: json.dump(h[-1000:], f, indent=2, ensure_ascii=False)
        return True
    except: return False

def update_prediction_status(pred_id, status):
    try:
        h = load_history()
        for p in h:
            if p.get('id') == pred_id:
                p['statut'] = status
                p['date_maj'] = datetime.now().isoformat()
                # Si le résultat est validé, mettre à jour les stats immédiatement
                if status in ['gagne', 'perdu']:
                    update_user_stats()
                break
        with open(HIST_FILE,'w',encoding='utf-8') as f: json.dump(h, f, indent=2, ensure_ascii=False)
        return True
    except: return False

def load_user_stats():
    if not USER_STATS_FILE.exists():
        return {'total_predictions':0,'correct_predictions':0,'incorrect_predictions':0,
                'annules_predictions':0,'current_streak':0,'best_streak':0,
                'value_bets_total':0,'value_bets_won':0,'favorites_total':0,'favorites_won':0}
    try:
        with open(USER_STATS_FILE) as f: return json.load(f)
    except: return {}

def update_user_stats():
    h = load_history()
    correct = sum(1 for p in h if p.get('statut')=='gagne')
    incorrect = sum(1 for p in h if p.get('statut')=='perdu')
    annules = sum(1 for p in h if p.get('statut')=='annule')
    
    # Calcul des séries
    streak = cur = best = 0
    for p in reversed(h):
        if p.get('statut')=='gagne': 
            streak+=1
            cur=streak
            best=max(best,streak)
        elif p.get('statut')=='perdu': 
            streak=0
            cur=0
    
    # Stats value bets
    value_bets_total = sum(1 for p in h if p.get('best_value'))
    value_bets_won = sum(1 for p in h if p.get('best_value') and p.get('statut')=='gagne')
    
    # Stats favoris
    favorites_total = 0
    favorites_won = 0
    for p in h:
        if p.get('statut') in ['gagne', 'perdu']:
            favorites_total += 1
            if p.get('favori') and p.get('statut') == 'gagne':
                if (p.get('proba',0.5) >= 0.5 and p.get('player1') == p.get('favori')) or \
                   (p.get('proba',0.5) < 0.5 and p.get('player2') == p.get('favori')):
                    favorites_won += 1
    
    stats = {
        'total_predictions': len(h),
        'correct_predictions': correct,
        'incorrect_predictions': incorrect,
        'annules_predictions': annules,
        'current_streak': cur,
        'best_streak': best,
        'value_bets_total': value_bets_total,
        'value_bets_won': value_bets_won,
        'favorites_total': favorites_total,
        'favorites_won': favorites_won
    }
    with open(USER_STATS_FILE,'w') as f: json.dump(stats, f)
    return stats

def load_combines():
    if not COMB_HIST_FILE.exists(): return []
    try:
        with open(COMB_HIST_FILE,'r',encoding='utf-8') as f: return json.load(f)
    except: return []

def save_combine(data):
    try:
        c = load_combines()
        data['date'] = datetime.now().isoformat()
        data['id'] = hashlib.md5(f"{datetime.now()}".encode()).hexdigest()[:8]
        data['statut'] = 'en_attente'
        c.append(data)
        with open(COMB_HIST_FILE,'w',encoding='utf-8') as f: json.dump(c[-200:], f, indent=2, ensure_ascii=False)
        return True
    except: return False

# ─────────────────────────────────────────────────────────────
# SCRAPING & MATCHS DU JOUR
# ─────────────────────────────────────────────────────────────
def get_mock_matches():
    return [
        {'p1':'Novak Djokovic','p2':'Carlos Alcaraz','surface':'Clay','tournament':'Roland Garros'},
        {'p1':'Jannik Sinner','p2':'Daniil Medvedev','surface':'Hard','tournament':'Miami Open'},
        {'p1':'Alexander Zverev','p2':'Stefanos Tsitsipas','surface':'Clay','tournament':'Madrid Open'},
        {'p1':'Holger Rune','p2':'Casper Ruud','surface':'Grass','tournament':'Wimbledon'},
    ]

@st.cache_data(ttl=1800)
def get_daily_matches(force_refresh=False):
    if force_refresh: st.cache_data.clear()
    if BS4_AVAILABLE:
        try:
            r = requests.get("https://www.flashscore.fr/tennis/",
                             headers={'User-Agent':'Mozilla/5.0'}, timeout=10)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.content, 'html.parser')
            matches = []
            for item in soup.find_all('div', class_='event__match')[:15]:
                home = item.find('div', class_='event__participant--home')
                away = item.find('div', class_='event__participant--away')
                if home and away:
                    text = str(item).lower()
                    surf = 'Clay' if 'clay' in text or 'terre' in text else 'Grass' if 'grass' in text else 'Hard'
                    matches.append({'p1':home.text.strip(),'p2':away.text.strip(),'surface':surf,'tournament':'ATP'})
            if matches: return matches
        except: pass
    return get_mock_matches()

def auto_load_today_matches():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📅 Charger les matchs ATP du jour", use_container_width=True, key="load_today_matches"):
            with st.spinner("🌐 Récupération des matchs en direct..."):
                matches = get_daily_matches(force_refresh=True)
                if matches:
                    st.session_state['today_matches'] = matches
                    st.success(f"✅ {len(matches)} matchs chargés!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Aucun match trouvé")
            return matches
    return None

def scan_for_value_bets(matches):
    model_info = load_saved_model()
    vbs = []
    for m in matches:
        proba, _ = calculate_probability(m['p1'], m['p2'], m['surface'], m.get('tournament',''), None, model_info)
        o1 = round(1/proba*(0.9+0.2*np.random.random()), 2)
        o2 = round(1/(1-proba)*(0.9+0.2*np.random.random()), 2)
        e1 = proba - 1/o1
        e2 = (1-proba) - 1/o2
        if e1 > MIN_EDGE_COMBINE:
            vbs.append({
                'match':f"{m['p1']} vs {m['p2']}",
                'joueur':m['p1'],
                'edge':e1*100,
                'cote':o1,
                'proba':proba,
                'surface':m['surface'],
                'tournament':m.get('tournament','')
            })
        elif e2 > MIN_EDGE_COMBINE:
            vbs.append({
                'match':f"{m['p1']} vs {m['p2']}",
                'joueur':m['p2'],
                'edge':e2*100,
                'cote':o2,
                'proba':1-proba,
                'surface':m['surface'],
                'tournament':m.get('tournament','')
            })
    return sorted(vbs, key=lambda x: x['edge'], reverse=True)

def generate_alternative_bets(player1, player2, surface, proba, h2h=None):
    bets = []
    if proba > 0.6 or proba < 0.4:
        bets.append({'type':'📊 Under 22.5 games','description':'Moins de 22.5 jeux','proba':0.65,'cote':1.75,'confidence':70})
    else:
        bets.append({'type':'📊 Over 22.5 games','description':'Plus de 22.5 jeux','proba':0.62,'cote':1.80,'confidence':65})
    if proba > 0.65:
        bets.append({'type':'⚖️ Handicap -3.5','description':f'{player1} gagne avec écart','proba':0.58,'cote':2.10,'confidence':60})
    elif proba < 0.35:
        bets.append({'type':'⚖️ Handicap +3.5','description':f'{player2} perd par moins de 4','proba':0.62,'cote':1.95,'confidence':65})
    if 0.3 < proba < 0.7:
        bets.append({'type':'🔄 Les deux gagnent un set','description':'Chaque joueur gagne un set','proba':0.55,'cote':2.20,'confidence':55})
    return bets

# ─────────────────────────────────────────────────────────────
# COMPOSANTS UI
# ─────────────────────────────────────────────────────────────
def player_selector(label, all_players, key, default=None):
    search = st.text_input(f"🔍 {label}", key=f"search_{key}", placeholder="Tapez un nom...")
    if search:
        filtered = [p for p in all_players if search.lower() in p.lower()]
        if not filtered:
            filtered = [p for p in all_players if p[0].lower() == search[0].lower()][:100]
    else:
        filtered = all_players[:200]
    st.caption(f"{len(filtered)} sur {len(all_players):,}")
    if not filtered: return st.text_input(label, key=key)
    idx = 0
    if default:
        for i, p in enumerate(filtered):
            if default.lower() in p.lower(): idx = i; break
    return st.selectbox(label, filtered, index=idx, key=key)

def tournament_selector(label, key, default=None):
    search = st.text_input(f"🔍 {label}", key=f"search_tourn_{key}", placeholder="Tapez le tournoi...")
    all_t = sorted(TOURNAMENTS_DB.keys())
    if search:
        sl = search.lower().strip()
        res = set()
        if sl in TOURNAMENT_ALIASES: res.add(TOURNAMENT_ALIASES[sl])
        for t in all_t:
            if sl in t.lower(): res.add(t)
        for alias, official in TOURNAMENT_ALIASES.items():
            if sl in alias: res.add(official)
        filtered = sorted(res) if res else [t for t in all_t if t[0].lower()==sl[0]]
    else:
        filtered = all_t[:100]
    idx = 0
    if default and default in filtered: idx = filtered.index(default)
    return st.selectbox(label, filtered, index=idx, key=key)

# ─────────────────────────────────────────────────────────────
# DASHBOARD PRO
# ─────────────────────────────────────────────────────────────
def create_interactive_dashboard():
    history = load_history()
    if not history or len(history) < 3: 
        st.info("Pas assez de données pour le dashboard")
        return
    
    df = pd.DataFrame(history)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')
    df['correct'] = (df['statut']=='gagne').astype(int)
    df['cum_correct'] = df['correct'].expanding().sum()
    df['cum_total'] = df['statut'].isin(['gagne','perdu']).expanding().sum()
    df['accuracy'] = (df['cum_correct']/df['cum_total']*100).fillna(0)

    # Graphique 1: Évolution de la précision
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df['date'], 
        y=df['accuracy'], 
        mode='lines+markers',
        name='Précision', 
        line=dict(color='#00DFA2', width=3),
        fill='tozeroy', 
        fillcolor='rgba(0,223,162,0.1)'
    ))
    fig1.update_layout(
        title='📈 Évolution de la précision',
        xaxis_title='Date',
        yaxis_title='Précision (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Graphique 2: Performance par surface
    if len(df) > 0:
        surface_stats = []
        for surface in SURFACES:
            surf_preds = df[df['surface'] == surface]
            if len(surf_preds) > 0:
                correct = len(surf_preds[surf_preds['statut'] == 'gagne'])
                total = len(surf_preds[surf_preds['statut'].isin(['gagne', 'perdu'])])
                accuracy = (correct / total * 100) if total > 0 else 0
                surface_stats.append({
                    'Surface': surface,
                    'Accuracy': accuracy,
                    'Total': total,
                    'Correct': correct
                })
        
        if surface_stats:
            df_surface = pd.DataFrame(surface_stats)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=df_surface['Surface'],
                y=df_surface['Accuracy'],
                text=df_surface['Accuracy'].round(1).astype(str) + '%',
                textposition='outside',
                marker_color=[SURFACE_CONFIG[s]['color'] for s in df_surface['Surface']],
                hovertemplate='<b>%{x}</b><br>Précision: %{y:.1f}%<br>Matchs: %{customdata[0]}<br>Victoires: %{customdata[1]}<extra></extra>',
                customdata=df_surface[['Total', 'Correct']]
            ))
            fig2.update_layout(
                title='🎾 Performance par surface',
                xaxis_title='Surface',
                yaxis_title='Précision (%)',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Graphique 3: Distribution des confiances
    if 'confidence' in df.columns:
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=df['confidence'].dropna(),
            nbinsx=20,
            marker_color='#00DFA2',
            opacity=0.7,
            name='Confiance'
        ))
        fig3.update_layout(
            title='📊 Distribution des confiances',
            xaxis_title='Confiance',
            yaxis_title='Nombre de prédictions',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)

def show_dashboard():
    st.markdown("## 🏠 Dashboard")
    
    stats = load_user_stats()
    history = load_history()
    ach = load_achievements()
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    accuracy = calculate_global_accuracy()
    
    with col1:
        st.metric(
            "📊 Total prédictions", 
            stats.get('total_predictions', 0),
            help="Nombre total de prédictions effectuées"
        )
    with col2:
        st.metric(
            "🎯 Précision globale", 
            f"{accuracy:.1f}%",
            delta=f"{stats.get('correct_predictions', 0)}/{stats.get('incorrect_predictions', 0)}",
            help="Pourcentage de prédictions correctes"
        )
    with col3:
        pending = len([p for p in history if p.get('statut') == 'en_attente'])
        st.metric(
            "⏳ En attente", 
            pending,
            help="Prédictions en attente de validation"
        )
    with col4:
        st.metric(
            "🏆 Badges", 
            len(ach),
            help="Badges débloqués"
        )
    
    # Stats secondaires
    col1, col2, col3 = st.columns(3)
    with col1:
        value_acc = (stats.get('value_bets_won', 0) / stats.get('value_bets_total', 1) * 100) if stats.get('value_bets_total', 0) > 0 else 0
        st.metric(
            "💎 Value bets", 
            f"{stats.get('value_bets_won', 0)}/{stats.get('value_bets_total', 0)}",
            f"{value_acc:.1f}%",
            help="Value bets gagnés / total"
        )
    with col2:
        fav_acc = (stats.get('favorites_won', 0) / stats.get('favorites_total', 1) * 100) if stats.get('favorites_total', 0) > 0 else 0
        st.metric(
            "🏆 Favoris", 
            f"{stats.get('favorites_won', 0)}/{stats.get('favorites_total', 0)}",
            f"{fav_acc:.1f}%",
            help="Favoris gagnés / total"
        )
    with col3:
        st.metric(
            "🔥 Série actuelle", 
            stats.get('current_streak', 0),
            f"Record: {stats.get('best_streak', 0)}",
            help="Série de victoires consécutives"
        )
    
    # Statut des services
    model_info = load_saved_model()
    groq = get_groq_key()
    tg_token, tg_chat = get_telegram_config()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if model_info:
            ps = model_info.get('player_stats', {})
            st.success(f"✅ Modèle ML · {model_info.get('accuracy',0):.1%} · {len(ps):,} joueurs")
        else:
            st.warning("⚠️ Modèle ML non chargé")
    with col2:
        if groq:
            st.success("✅ IA Groq connectée")
        else:
            st.warning("⚠️ IA non configurée")
    with col3:
        if tg_token and tg_chat:
            st.success("✅ Telegram connecté")
        else:
            st.warning("⚠️ Telegram non configuré")
    
    # Dashboard interactif
    create_interactive_dashboard()

# ─────────────────────────────────────────────────────────────
# PAGE ANALYSE MULTI-MATCHS
# ─────────────────────────────────────────────────────────────
def show_prediction():
    st.markdown("## 🎯 Analyse Multi-matchs")

    model_info = load_saved_model()
    if model_info:
        ps = model_info.get('player_stats', {})
        n_known = len(ps)
        st.success(f"✅ Modèle ML chargé · {model_info.get('accuracy',0):.1%} accuracy · {n_known:,} joueurs connus")
    else:
        st.warning("⚠️ Modèle ML non chargé — mode fallback H2H actif")

    with st.spinner("Chargement des joueurs CSV..."):
        all_players = load_atp_data()
    st.info(f"📋 {len(all_players):,} joueurs disponibles · {len(TOURNAMENTS_DB)} tournois")

    # Auto-load des matchs du jour
    auto_load_today_matches()

    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        n_matches = st.number_input("Nombre de matchs", 1, MAX_MATCHES_ANALYSIS, 3)
    with col2: 
        mise = st.number_input("Mise (€)", 1.0, 1000.0, 10.0)
    with col3: 
        use_ai = st.checkbox("🤖 Analyse IA", True)
    with col4:
        send_tg = st.checkbox("📱 Envoyer Telegram", True)

    today_matches = st.session_state.get('today_matches', [])
    matches_input = []
    st.markdown("### 📝 Saisie des matchs")

    for i in range(n_matches):
        with st.expander(f"Match {i+1}", expanded=(i==0)):
            # Tournoi + surface auto
            col_t, col_s = st.columns([3,1])
            with col_t:
                dflt_tourn = today_matches[i]['tournament'] if i < len(today_matches) else "Roland Garros"
                tournament = tournament_selector("Tournoi", key=f"tourn_{i}", default=dflt_tourn)
            with col_s:
                surface = get_tournament_surface(tournament)
                level, best_of = get_tournament_level(tournament)
                st.markdown(f"**Surface**")
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.15);'
                    f'border-radius:8px;padding:0.5rem 0.75rem;">'
                    f'{SURFACE_CONFIG[surface]["icon"]} {surface}</div>',
                    unsafe_allow_html=True
                )
                if best_of == 5:
                    st.caption("🏆 Best of 5")

            # Joueurs
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                dflt_p1 = today_matches[i]['p1'] if i < len(today_matches) else ""
                p1 = player_selector("Joueur 1", all_players, f"p1_{i}", default=dflt_p1)
            with col_p2:
                players2 = [p for p in all_players if p != p1]
                dflt_p2 = today_matches[i]['p2'] if i < len(today_matches) else ""
                p2 = player_selector("Joueur 2", players2, f"p2_{i}", default=dflt_p2)

            # Cotes
            col_o1, col_o2 = st.columns(2)
            with col_o1: 
                odds1 = st.text_input(f"Cote {p1[:18] if p1 else 'J1'}", key=f"odds1_{i}", placeholder="1.75")
            with col_o2: 
                odds2 = st.text_input(f"Cote {p2[:18] if p2 else 'J2'}", key=f"odds2_{i}", placeholder="2.10")

            # Infos ML pour ce match
            if model_info and p1 and p2:
                ps = model_info.get('player_stats', {})
                p1_known = "✅" if p1 in ps else "⚠️"
                p2_known = "✅" if p2 in ps else "⚠️"
                st.caption(f"Modèle: {p1_known} {p1[:20]} · {p2_known} {p2[:20]}")

            matches_input.append({
                'player1':p1,'player2':p2,'surface':surface,
                'tournament':tournament,'odds1':odds1,'odds2':odds2
            })

    if not st.button("🔍 Analyser tous les matchs", type="primary", use_container_width=True):
        return

    valid = [m for m in matches_input if m['player1'] and m['player2']]
    if not valid:
        st.warning("Remplis au moins un match complet")
        return

    st.markdown("---")
    st.markdown("## 📊 Résultats")
    all_selections = []

    for i, match in enumerate(valid):
        p1, p2 = match['player1'], match['player2']
        surf   = match['surface']
        tourn  = match['tournament']

        st.markdown(f"### Match {i+1}: **{p1}** vs **{p2}**")
        st.caption(f"🏆 {tourn} · {SURFACE_CONFIG[surf]['icon']} {surf}")

        h2h = get_h2h_stats(p1, p2)
        proba, ml_used = calculate_probability(p1, p2, surf, tourn, h2h, model_info)
        confidence = calculate_confidence(proba, h2h)
        gagnant = p1 if proba >= 0.5 else p2

        # Badge ML ou fallback
        if ml_used:
            st.success("🤖 Prédiction ML avec TOUTES les 21 features")
        else:
            ps = model_info.get('player_stats', {}) if model_info else {}
            if model_info and (p1 not in ps or p2 not in ps):
                unknowns = [x for x in [p1, p2] if x not in ps]
                st.warning(f"⚠️ {', '.join(unknowns)} inconnu(s) du modèle — fallback H2H")
            else:
                st.info("📊 Mode fallback (modèle non chargé)")

        st.markdown(f"#### 🏆 GAGNANT PRÉDIT: **{gagnant}**")
        col1, col2 = st.columns(2)
        with col1: 
            st.metric(p1, f"{proba:.1%}")
        with col2: 
            st.metric(p2, f"{1-proba:.1%}")
        st.progress(float(proba))

        col1, col2, col3 = st.columns(3)
        with col1: 
            st.caption(f"{'🤖 ML 21 features' if ml_used else '📊 Fallback'}")
        with col2:
            ci = "🟢" if confidence>=70 else "🟡" if confidence>=50 else "🔴"
            st.caption(f"Confiance: {ci} {confidence:.0f}/100")
        with col3:
            if h2h:
                w1 = h2h.get(f"{p1}_wins",0)
                w2 = h2h.get(f"{p2}_wins",0)
                st.caption(f"H2H: {w1}-{w2} ({h2h['total_matches']} matchs)")

        # Value bet
        best_value = None
        if match['odds1'] and match['odds2']:
            try:
                o1 = float(match['odds1'].replace(',','.'))
                o2 = float(match['odds2'].replace(',','.'))
                e1 = proba - 1/o1
                e2 = (1-proba) - 1/o2
                if e1 > MIN_EDGE_COMBINE:
                    best_value = {'joueur':p1,'edge':e1,'cote':o1,'proba':proba}
                    st.success(f"🎯 Value bet: {p1} @ {o1} (edge: +{e1*100:.1f}%)")
                    all_selections.append(best_value)
                elif e2 > MIN_EDGE_COMBINE:
                    best_value = {'joueur':p2,'edge':e2,'cote':o2,'proba':1-proba}
                    st.success(f"🎯 Value bet: {p2} @ {o2} (edge: +{e2*100:.1f}%)")
                    all_selections.append(best_value)
            except Exception as ex:
                st.warning(f"Cotes invalides: {ex}")

        # Paris alternatifs
        bets = generate_alternative_bets(p1, p2, surf, proba, h2h)
        if bets:
            with st.expander("🎯 Paris alternatifs"):
                for b in bets:
                    st.info(f"{b['type']}: {b['description']} (proba {b['proba']:.1%} · cote {b['cote']:.2f})")

        # IA
        ai_comment = None
        if use_ai and get_groq_key():
            with st.spinner("🤖 Analyse IA..."):
                ai_comment = analyze_match_with_ai(p1, p2, surf, tourn, proba, best_value)
                if ai_comment:
                    with st.expander("🤖 Analyse IA"):
                        st.write(ai_comment)

        pred_data = {
            'player1':p1, 'player2':p2, 'tournament':tourn, 'surface':surf,
            'proba':float(proba), 'confidence':float(confidence),
            'odds1':match['odds1'], 'odds2':match['odds2'],
            'favori':gagnant, 'best_value':best_value, 'ml_used':ml_used,
            'date':datetime.now().isoformat()
        }

        # Sauvegarde automatique
        save_prediction(pred_data)
        st.success("✅ Sauvegardé automatiquement en attente")

        # Envoi Telegram
        if send_tg:
            tg_token, _ = get_telegram_config()
            if tg_token:
                if send_prediction_to_telegram(pred_data, bets, ai_comment):
                    st.success("📱 Envoyé sur Telegram")
                else:
                    st.warning("⚠️ Échec envoi Telegram")
            else:
                st.warning("⚠️ Telegram non configuré")

        st.divider()

    new_badges = check_and_unlock_achievements()
    if new_badges: 
        st.balloons()
        st.success("🏆 Nouveaux badges débloqués!")

# ─────────────────────────────────────────────────────────────
# PAGE EN ATTENTE PRO
# ─────────────────────────────────────────────────────────────
def show_pending():
    st.markdown("## ⏳ Prédictions en attente")
    
    h = load_history()
    pending = [p for p in h if p.get('statut') == 'en_attente']
    
    if not pending:
        st.info("🎉 Aucune prédiction en attente !")
        return
    
    # Statistiques des en attente
    total_pending = len(pending)
    avg_proba = np.mean([p.get('proba', 0.5) for p in pending])
    value_bets_pending = sum(1 for p in pending if p.get('best_value'))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 En attente", total_pending)
    with col2:
        st.metric("📈 Proba moyenne", f"{avg_proba:.1%}")
    with col3:
        st.metric("💎 Value bets", value_bets_pending)
    
    st.markdown("---")
    
    for pred in pending[::-1]:
        with st.expander(f"📅 {pred.get('date','')[:16]} · {pred['player1']} vs {pred['player2']}", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1: 
                st.metric("Tournoi", pred.get('tournament','?')[:20])
            with col2:
                s = pred.get('surface','?')
                st.metric("Surface", f"{SURFACE_CONFIG.get(s,{}).get('icon','🎾')} {s}")
            with col3: 
                st.metric("Probabilité", f"{pred.get('proba',0.5):.1%}")
            
            if pred.get('best_value'):
                bv = pred['best_value']
                st.info(f"🎯 Value bet: {bv['joueur']} @ {bv.get('cote','?')} (edge +{bv.get('edge',0)*100:.1f}%)")
            
            # Stats supplémentaires
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Confiance: {pred.get('confidence',0):.0f}/100")
            with col2:
                if pred.get('ml_used'):
                    st.caption("🤖 Modèle ML")
                else:
                    st.caption("📊 Stats CSV")
            
            st.markdown("**Résultat du match :**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button(f"✅ {pred['player1'][:15]} gagne", key=f"w1_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'gagne')
                    st.rerun()
            with c2:
                if st.button(f"✅ {pred['player2'][:15]} gagne", key=f"w2_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'gagne')
                    st.rerun()
            with c3:
                if st.button("❌ Perdu", key=f"loss_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'perdu')
                    st.rerun()
            with c4:
                if st.button("⚠️ Annuler", key=f"cancel_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'annule')
                    st.rerun()

# ─────────────────────────────────────────────────────────────
# PAGE BADGES
# ─────────────────────────────────────────────────────────────
def show_achievements():
    st.markdown("## 🏆 Badges")
    ach = load_achievements()
    st.progress(len(ach)/len(ACHIEVEMENTS))
    st.caption(f"{len(ach)}/{len(ACHIEVEMENTS)} badges débloqués")
    
    cols = st.columns(2)
    for i, (aid, adata) in enumerate(ACHIEVEMENTS.items()):
        with cols[i%2]:
            unlocked = aid in ach
            bg = "rgba(0,223,162,0.1)" if unlocked else "rgba(108,124,137,0.1)"
            border = f"2px solid #00DFA2" if unlocked else "1px solid rgba(255,255,255,0.1)"
            date_str = ""
            if unlocked:
                try: 
                    date_str = f'<div style="font-size:0.75rem;color:#6C7A89;">Débloqué le {datetime.fromisoformat(ach[aid]["unlocked_at"]).strftime("%d/%m/%Y")}</div>'
                except: pass
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:10px;padding:1rem;margin-bottom:1rem;">
                <span style="font-size:2rem;">{adata['icon']}</span>
                <strong style="color:{'#00DFA2' if unlocked else '#6C7A89'}"> {adata['name']}</strong>
                <div style="font-size:0.85rem;">{adata['desc']}</div>{date_str}
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PAGE TELEGRAM
# ─────────────────────────────────────────────────────────────
def show_telegram():
    st.markdown("## 📱 Telegram")
    token, chat_id = get_telegram_config()
    
    if not token or not chat_id:
        st.warning("⚠️ Telegram non configuré")
        st.code("""
        # Dans les secrets Streamlit (Settings → Secrets)
        TELEGRAM_BOT_TOKEN = "ton_token_ici"
        TELEGRAM_CHAT_ID = "ton_chat_id_ici"
        """, language='toml')
        
        st.info("""
        **Comment configurer :**
        1. Va sur Telegram et cherche @BotFather
        2. Envoie `/newbot` et suis les instructions
        3. Copie le token fourni
        4. Envoie un message à @userinfobot pour obtenir ton chat_id
        """)
        return
    
    st.success(f"✅ Bot configuré (Chat ID: {chat_id})")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔧 Tester la connexion", use_container_width=True):
            with st.spinner("Test en cours..."):
                ok, msg = test_telegram_connection()
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
    with col2:
        if st.button("📊 Envoyer les stats", use_container_width=True):
            with st.spinner("Envoi en cours..."):
                if send_stats_to_telegram():
                    st.success("✅ Statistiques envoyées !")
                else:
                    st.error("❌ Échec de l'envoi")
    
    st.markdown("---")
    st.markdown("### ✍️ Message personnalisé")
    
    with st.form("tg_form"):
        msg = st.text_area("Message", height=100, placeholder="Écris ton message ici...")
        urgent = st.checkbox("🔴 Urgent")
        include_stats = st.checkbox("📊 Inclure les stats")
        
        if st.form_submit_button("📤 Envoyer le message") and msg:
            full_msg = ("🔴 URGENT\n\n" if urgent else "") + msg
            if include_stats:
                full_msg += f"\n\n{format_stats_message()}"
            
            with st.spinner("Envoi en cours..."):
                if send_telegram_message(full_msg):
                    st.success("✅ Message envoyé !")
                else:
                    st.error("❌ Échec de l'envoi")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
def show_configuration():
    st.markdown("## ⚙️ Configuration")
    
    # Modèle ML
    st.markdown("### 🤖 Modèle Machine Learning")
    model_info = load_saved_model()
    if model_info:
        ps = model_info.get('player_stats', {})
        st.success(f"""✅ Modèle chargé avec succès
- Accuracy: **{model_info.get('accuracy',0):.1%}**
- AUC-ROC: **{model_info.get('auc',0):.3f}**
- Joueurs connus: **{len(ps):,}**
- Matchs d'entraînement: **{model_info.get('n_matches',0):,}**
- Entraîné le: {model_info.get('trained_at','?')[:10]}""")

        st.markdown("#### 🔍 Top 10 features utilisées")
        feats = model_info.get('feature_importance', {})
        if feats:
            sorted_feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)[:10]
            for f, imp in sorted_feats:
                st.progress(float(imp), text=f"{f}: {imp:.1%}")
        
        if st.button("🔄 Recharger le modèle"):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.warning("⚠️ Aucun modèle trouvé dans le dossier models/")
    
    st.markdown("---")
    st.markdown("### 🗑️ Gestion des données")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Effacer historique", use_container_width=True):
            if HIST_FILE.exists(): 
                HIST_FILE.unlink()
                update_user_stats()
                st.rerun()
    with col2:
        if st.button("🗑️ Effacer badges", use_container_width=True):
            if ACHIEVEMENTS_FILE.exists(): 
                ACHIEVEMENTS_FILE.unlink()
                st.rerun()
    with col3:
        if st.button("🔄 Recalculer stats", use_container_width=True):
            update_user_stats()
            st.rerun()
    
    st.markdown("### 💾 Backup")
    if st.button("📀 Faire un backup maintenant", use_container_width=True):
        auto_backup()
        st.success("✅ Backup effectué !")

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
    
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0A1E2C 0%, #1A2E3C 100%); }
        .stProgress > div > div > div > div { background: linear-gradient(90deg, #00DFA2, #0079FF); }
        .stButton > button { background: linear-gradient(90deg, #00DFA2, #0079FF); color: white; border: none; }
        .stButton > button:hover { background: linear-gradient(90deg, #00DFA2, #0079FF); opacity: 0.9; }
        h1, h2, h3 { color: #00DFA2; }
        div[data-testid="stMetricValue"] { font-size: 2rem; }
    </style>
    """, unsafe_allow_html=True)

    if 'last_backup' not in st.session_state: 
        st.session_state['last_backup'] = datetime.now()
    
    if (datetime.now() - st.session_state['last_backup']).seconds >= 86400:
        auto_backup()
        st.session_state['last_backup'] = datetime.now()

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; margin-bottom:2rem;">
            <div style="font-size:2.5rem;font-weight:800;color:#00DFA2;">TennisIQ</div>
            <div style="font-size:0.85rem;color:#6C7A89;">ML • 21 Features • Pro Edition</div>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🎯 Analyse", "⏳ En Attente", "💎 Value Bets", "🏆 Badges", "📱 Telegram", "⚙️ Configuration"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        stats = load_user_stats()
        pending = len([p for p in load_history() if p.get('statut') == 'en_attente'])
        tv = stats.get('correct_predictions',0) + stats.get('incorrect_predictions',0)
        acc = (stats.get('correct_predictions',0)/tv*100) if tv>0 else 0
        
        col1, col2 = st.columns(2)
        with col1: 
            st.metric("Précision", f"{acc:.1f}%")
            st.metric("Badges", len(load_achievements()))
        with col2: 
            st.metric("En attente", pending)
            st.metric("Série", stats.get('current_streak', 0))

    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🎯 Analyse":
        show_prediction()
    elif page == "⏳ En Attente":
        show_pending()
    elif page == "💎 Value Bets":
        st.markdown("## 💎 Value Bets en direct")
        matches = get_daily_matches()
        vbs = scan_for_value_bets(matches)
        
        if vbs:
            for vb in vbs:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    with col1: 
                        st.markdown(f"**{vb['joueur']}**")
                        st.caption(vb['match'])
                    with col2: 
                        st.metric("Cote", f"{vb['cote']:.2f}")
                    with col3: 
                        st.metric("Edge", f"+{vb['edge']:.1f}%")
                    with col4: 
                        st.metric("Proba", f"{vb['proba']:.1%}")
                    st.divider()
        else:
            st.info("Aucun value bet détecté pour le moment")
    
    elif page == "🏆 Badges":
        show_achievements()
    elif page == "📱 Telegram":
        show_telegram()
    elif page == "⚙️ Configuration":
        show_configuration()

if __name__ == "__main__":
    main()
