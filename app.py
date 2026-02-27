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

# ─────────────────────────────────────────────────────────────
# CONFIGURATION DES CHEMINS
# ─────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR   = ROOT_DIR / "src" / "data" / "raw" / "tml-tennis"
HIST_DIR   = ROOT_DIR / "history"
BACKUP_DIR = ROOT_DIR / "backups"

for d in [MODELS_DIR, DATA_DIR, HIST_DIR, BACKUP_DIR]:
    d.mkdir(exist_ok=True, parents=True)

HIST_FILE         = HIST_DIR / "predictions_history.json"
COMB_HIST_FILE    = HIST_DIR / "combines_history.json"
USER_STATS_FILE   = HIST_DIR / "user_stats.json"
ACHIEVEMENTS_FILE = HIST_DIR / "achievements.json"

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
SURFACES          = ["Hard", "Clay", "Grass"]
MIN_EDGE_COMBINE  = 0.02
MAX_MATCHES       = 30

ACHIEVEMENTS = {
    'first_win':          {'name': '🎯 Première victoire',  'desc': 'Première prédiction gagnante',  'icon': '🎯'},
    'streak_5':           {'name': '🔥 En forme',           'desc': '5 victoires consécutives',      'icon': '🔥'},
    'streak_10':          {'name': '⚡ Imbattable',          'desc': '10 victoires consécutives',     'icon': '⚡'},
    'pred_100':           {'name': '🏆 Expert',             'desc': '100 prédictions',               'icon': '🏆'},
    'value_master':       {'name': '💎 Value Master',        'desc': '10 value bets gagnants',        'icon': '💎'},
    'surface_specialist': {'name': '🌍 Multi-surface',       'desc': 'Gagnant sur les 3 surfaces',    'icon': '🌍'},
}

TOURNAMENTS_DB = {
    "Australian Open": "Hard", "Roland Garros": "Clay",
    "Wimbledon": "Grass",      "US Open": "Hard",
    "Nitto ATP Finals": "Hard",
    "Indian Wells Masters": "Hard", "Miami Open": "Hard",
    "Monte-Carlo Masters": "Clay",  "Madrid Open": "Clay",
    "Italian Open": "Clay",   "Canadian Open": "Hard",
    "Cincinnati Masters": "Hard",   "Shanghai Masters": "Hard",
    "Paris Masters": "Hard",  "Rotterdam Open": "Hard",
    "Rio Open": "Clay",       "Dubai Tennis Championships": "Hard",
    "Mexican Open": "Hard",   "Barcelona Open": "Clay",
    "Halle Open": "Grass",    "Queen's Club Championships": "Grass",
    "Hamburg Open": "Clay",   "Washington Open": "Hard",
    "China Open": "Hard",     "Japan Open": "Hard",
    "Vienna Open": "Hard",    "Swiss Indoors": "Hard",
    "Dallas Open": "Hard",    "Qatar Open": "Hard",
    "Adelaide International": "Hard", "Auckland Open": "Hard",
    "Brisbane International": "Hard", "Cordoba Open": "Clay",
    "Buenos Aires": "Clay",   "Delray Beach": "Hard",
    "Marseille Open": "Hard", "Santiago": "Clay",
    "Houston": "Clay",        "Marrakech": "Clay",
    "Estoril": "Clay",        "Munich": "Clay",
    "Geneva": "Clay",         "Lyon": "Clay",
    "Stuttgart": "Grass",     "Mallorca": "Grass",
    "Eastbourne": "Grass",    "Newport": "Grass",
    "Atlanta": "Hard",        "Croatia Open Umag": "Clay",
    "Kitzbühel": "Clay",      "Los Cabos": "Hard",
    "Winston-Salem": "Hard",  "Chengdu Open": "Hard",
    "Sofia": "Hard",          "Metz": "Hard",
    "San Diego": "Hard",      "Seoul": "Hard",
    "Tel Aviv": "Hard",       "Florence": "Hard",
    "Antwerp": "Hard",        "Stockholm": "Hard",
    "Belgrade Open": "Clay",  "Autre tournoi": "Hard",
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
    "acapulco": "Mexican Open", "mexican": "Mexican Open",
    "australian": "Australian Open", "melbourne": "Australian Open",
    "roland garros": "Roland Garros", "french open": "Roland Garros",
    "wimbledon": "Wimbledon", "us open": "US Open",
    "flushing": "US Open", "new york": "US Open",
    "indian wells": "Indian Wells Masters", "miami": "Miami Open",
    "monte carlo": "Monte-Carlo Masters", "madrid": "Madrid Open",
    "rome": "Italian Open", "canada": "Canadian Open",
    "cincinnati": "Cincinnati Masters", "shanghai": "Shanghai Masters",
    "paris masters": "Paris Masters", "bercy": "Paris Masters",
    "rotterdam": "Rotterdam Open", "dubai": "Dubai Tennis Championships",
    "barcelona": "Barcelona Open", "halle": "Halle Open",
    "queens": "Queen's Club Championships", "hamburg": "Hamburg Open",
    "washington": "Washington Open", "beijing": "China Open",
    "tokyo": "Japan Open", "vienna": "Vienna Open", "basel": "Swiss Indoors",
}

COLORS = {
    "primary": "#00DFA2", "secondary": "#0079FF",
    "warning": "#FFB200", "danger": "#FF3B3F", "gray": "#6C7A89",
    "hard": "#0079FF", "clay": "#E67E22", "grass": "#00DFA2",
    "card_bg": "rgba(255,255,255,0.04)", "card_border": "rgba(255,255,255,0.10)",
}

SURFACE_CFG = {
    "Hard":  {"color": "#0079FF", "icon": "🟦", "bg": "rgba(0,121,255,0.12)"},
    "Clay":  {"color": "#E67E22", "icon": "🟧", "bg": "rgba(230,126,34,0.12)"},
    "Grass": {"color": "#00DFA2", "icon": "🟩", "bg": "rgba(0,223,162,0.12)"},
}

# ─────────────────────────────────────────────────────────────
# CSS GLOBAL PRO
# ─────────────────────────────────────────────────────────────
PRO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --primary: #00DFA2;
    --secondary: #0079FF;
    --bg: #080E1A;
    --card: rgba(255,255,255,0.035);
    --card-hover: rgba(255,255,255,0.06);
    --border: rgba(255,255,255,0.08);
    --text: #E8EDF5;
    --muted: #7A8599;
    --success: #00DFA2;
    --warning: #FFB200;
    --danger: #FF4757;
}

.stApp {
    background: var(--bg);
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(8,14,26,0.97) !important;
    border-right: 1px solid var(--border) !important;
}

/* Headers */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00DFA2 0%, #0079FF 100%) !important;
    color: #080E1A !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,223,162,0.25) !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1rem 1.25rem !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00DFA2, #0079FF) !important;
}

/* Expander */
details {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Input */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Radio */
.stRadio > div { gap: 0.5rem; }

/* Alerts */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 10px !important;
    border-left-width: 3px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
</style>
"""

# ─────────────────────────────────────────────────────────────
# COMPOSANTS UI PRO
# ─────────────────────────────────────────────────────────────
def card(content, border_color=None, padding="1.5rem"):
    bc = border_color or COLORS["card_border"]
    return f"""
    <div style="background:{COLORS['card_bg']};border:1px solid {bc};
    border-radius:16px;padding:{padding};margin-bottom:1rem;">
    {content}</div>"""

def stat_pill(label, value, color="#00DFA2", icon=""):
    return f"""
    <div style="display:inline-flex;align-items:center;gap:0.5rem;
    background:rgba(0,0,0,0.3);border:1px solid {color}33;
    border-radius:100px;padding:0.35rem 0.85rem;margin:0.2rem;">
        <span style="font-size:0.75rem;color:{color};">{icon}</span>
        <span style="font-size:0.8rem;font-weight:600;color:{COLORS['gray']};">{label}</span>
        <span style="font-size:0.9rem;font-weight:700;color:{color};">{value}</span>
    </div>"""

def section_title(title, subtitle=""):
    sub = f'<p style="color:{COLORS["gray"]};font-size:0.9rem;margin:0.25rem 0 0;">{subtitle}</p>' if subtitle else ""
    return f"""
    <div style="margin-bottom:1.5rem;">
        <h2 style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;
        color:#E8EDF5;margin:0;letter-spacing:-0.02em;">{title}</h2>
        {sub}
    </div>"""

def big_metric(label, value, delta=None, icon="", color="#00DFA2"):
    delta_html = ""
    if delta is not None:
        dcolor = "#00DFA2" if delta >= 0 else "#FF4757"
        darrow = "↑" if delta >= 0 else "↓"
        delta_html = f'<span style="font-size:0.8rem;color:{dcolor};margin-left:0.5rem;">{darrow} {abs(delta):.1f}%</span>'
    return f"""
    <div style="background:{COLORS['card_bg']};border:1px solid {color}33;
    border-radius:16px;padding:1.25rem 1.5rem;text-align:center;">
        <div style="font-size:1.6rem;margin-bottom:0.25rem;">{icon}</div>
        <div style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
        color:{color};">{value}{delta_html}</div>
        <div style="font-size:0.8rem;color:{COLORS['gray']};margin-top:0.25rem;
        text-transform:uppercase;letter-spacing:0.08em;">{label}</div>
    </div>"""

def match_result_badge(statut, pronostic_correct=None):
    configs = {
        "gagne":     ("✅", "#00DFA2", "rgba(0,223,162,0.12)", "GAGNÉ"),
        "perdu":     ("❌", "#FF4757", "rgba(255,71,87,0.12)",  "PERDU"),
        "annule":    ("⚠️", "#FFB200", "rgba(255,178,0,0.12)",  "ABANDONNÉ"),
        "en_attente":("⏳", "#7A8599", "rgba(122,133,153,0.12)","EN ATTENTE"),
    }
    icon, color, bg, label = configs.get(statut, configs["en_attente"])
    prono_html = ""
    if pronostic_correct is not None and statut in ["gagne","perdu"]:
        if pronostic_correct:
            prono_html = f'<span style="font-size:0.65rem;color:#00DFA2;margin-left:0.4rem;">🎯 Pronostic ✓</span>'
        else:
            prono_html = f'<span style="font-size:0.65rem;color:#FF4757;margin-left:0.4rem;">🎯 Pronostic ✗</span>'
    return f"""
    <span style="background:{bg};color:{color};border:1px solid {color}44;
    border-radius:100px;padding:0.25rem 0.7rem;font-size:0.75rem;
    font-weight:700;white-space:nowrap;">{icon} {label}{prono_html}</span>"""

def surface_badge(surface):
    cfg = SURFACE_CFG.get(surface, SURFACE_CFG["Hard"])
    return f"""<span style="background:{cfg['bg']};color:{cfg['color']};
    border:1px solid {cfg['color']}44;border-radius:100px;
    padding:0.2rem 0.6rem;font-size:0.75rem;font-weight:600;">
    {cfg['icon']} {surface}</span>"""

# ─────────────────────────────────────────────────────────────
# UTILITAIRES TOURNOIS
# ─────────────────────────────────────────────────────────────
def get_surface(name): return TOURNAMENTS_DB.get(name, "Hard")
def get_level(name): return TOURNAMENT_LEVEL.get(name, ("A", 3))

def find_tournament(s):
    if not s: return None
    sl = s.lower().strip()
    if sl in TOURNAMENT_ALIASES: return TOURNAMENT_ALIASES[sl]
    for t in TOURNAMENTS_DB:
        if sl == t.lower(): return t
    m = [t for t in TOURNAMENTS_DB if sl in t.lower()]
    return min(m, key=len) if m else None

# ─────────────────────────────────────────────────────────────
# TELEGRAM — CORRIGÉ ET FIABLE
# ─────────────────────────────────────────────────────────────
def get_tg_config():
    try:
        return st.secrets["TELEGRAM_BOT_TOKEN"], str(st.secrets["TELEGRAM_CHAT_ID"])
    except:
        t = os.environ.get("TELEGRAM_BOT_TOKEN")
        c = os.environ.get("TELEGRAM_CHAT_ID")
        return (t, c) if t and c else (None, None)

def tg_send(message, parse_mode="HTML"):
    """Envoi Telegram direct via requests — pas d'async, pas de checkbox"""
    token, chat_id = get_tg_config()
    if not token or not chat_id:
        return False, "❌ Telegram non configuré (secrets manquants)"
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            return True, "✅ Envoyé sur Telegram"
        else:
            data = r.json()
            err_desc = data.get("description", r.text[:100])
            return False, f"❌ Telegram API: {err_desc}"
    except requests.exceptions.Timeout:
        return False, "❌ Timeout Telegram (>15s)"
    except Exception as e:
        return False, f"❌ Erreur: {str(e)[:80]}"

def tg_test():
    token, chat_id = get_tg_config()
    if not token: return False, "❌ Token manquant"
    if not chat_id: return False, "❌ Chat ID manquant"
    # Vérifier le bot
    try:
        r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
        if r.status_code != 200:
            return False, f"❌ Token invalide: {r.json().get('description','')}"
        bot_name = r.json().get("result", {}).get("first_name", "Bot")
    except Exception as e:
        return False, f"❌ Impossible de joindre Telegram: {e}"
    # Envoyer message test
    msg = f"""<b>✅ TennisIQ — Test de connexion</b>

🤖 Bot: <b>{bot_name}</b>
📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}
📊 Prédictions enregistrées: <b>{len(load_history())}</b>
🎯 Précision globale: <b>{calc_accuracy():.1f}%</b>

<i>Connexion opérationnelle !</i>
#TennisIQ"""
    return tg_send(msg)

def format_pred_msg(pred, bet_suggestions=None, ai_comment=None):
    proba = pred.get("proba", 0.5)
    bar   = "█" * int(proba * 10) + "░" * (10 - int(proba * 10))
    surf  = pred.get("surface", "Hard")
    s_icon = {"Hard": "🟦", "Clay": "🟧", "Grass": "🟩"}.get(surf, "🎾")
    ml_tag = "🤖 " if pred.get("ml_used") else ""
    fav    = pred.get("favori", "?")
    conf   = pred.get("confidence", 50)
    conf_icon = "🟢" if conf >= 70 else "🟡" if conf >= 50 else "🔴"

    msg = f"""<b>{ml_tag}🎾 PRÉDICTION TENNISIQ</b>

🆚 <b>{pred.get('player1','?')} vs {pred.get('player2','?')}</b>
🏆 {pred.get('tournament','?')} | {s_icon} {surf}

<code>{bar}</code>
• {pred.get('player1','J1')}: <b>{proba:.1%}</b>
• {pred.get('player2','J2')}: <b>{1-proba:.1%}</b>

🏅 <b>FAVORI: {fav}</b>
{conf_icon} Confiance: <b>{conf:.0f}/100</b>"""

    if pred.get("odds1") and pred.get("odds2"):
        msg += f"\n💰 Cotes: {pred['player1']} @ <b>{pred['odds1']}</b> | {pred['player2']} @ <b>{pred['odds2']}</b>"

    if pred.get("best_value"):
        bv = pred["best_value"]
        msg += f"\n\n🎯 <b>VALUE BET !</b> {bv['joueur']} @ {bv['cote']:.2f} | Edge: <b>+{bv['edge']*100:.1f}%</b>"

    if bet_suggestions:
        msg += "\n\n<b>📊 Paris alternatifs:</b>"
        for b in bet_suggestions[:2]:
            msg += f"\n• {b['type']}: {b['proba']:.1%} @ {b['cote']:.2f}"

    if ai_comment:
        clean = ai_comment.replace("<","&lt;").replace(">","&gt;")
        msg += f"\n\n🤖 <b>Analyse IA:</b>\n{clean[:600]}"

    msg += f"\n\n#TennisIQ #{surf.replace(' ','')}"
    return msg

def format_stats_msg():
    stats = load_user_stats(); h = load_history()
    correct = stats.get("correct_predictions", 0)
    wrong   = stats.get("incorrect_predictions", 0)
    cancel  = stats.get("annules_predictions", 0)
    tv      = correct + wrong
    acc     = (correct / tv * 100) if tv > 0 else 0
    bar     = "█" * int(acc / 10) + "░" * (10 - int(acc / 10))
    recent  = [p for p in h[-20:] if p.get("statut") in ["gagne","perdu"]]
    r_ok    = sum(1 for p in recent if p.get("statut") == "gagne")
    r_acc   = (r_ok / len(recent) * 100) if recent else 0
    diff    = r_acc - acc
    trend   = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"

    return f"""<b>📊 STATISTIQUES TENNISIQ</b>

<code>{bar}</code> {acc:.1f}%

<b>Vue d'ensemble:</b>
• 📝 Total: <b>{stats.get('total_predictions',0)}</b>
• ✅ Gagnés: <b>{correct}</b> ({acc:.1f}%)
• ❌ Perdus: <b>{wrong}</b>
• ⚠️ Abandons: <b>{cancel}</b>

<b>Forme récente (20 derniers):</b>
{trend} <b>{r_acc:.1f}%</b> ({diff:+.1f}% vs global)

<b>Records:</b>
• 🔥 Série actuelle: <b>{stats.get('current_streak',0)}</b>
• ⚡ Meilleure série: <b>{stats.get('best_streak',0)}</b>

📅 {datetime.now().strftime('%d/%m/%Y %H:%M')} #TennisIQ"""

# ─────────────────────────────────────────────────────────────
# GROQ IA
# ─────────────────────────────────────────────────────────────
def get_groq_key():
    try: return st.secrets["GROQ_API_KEY"]
    except: return os.environ.get("GROQ_API_KEY")

def call_groq(prompt):
    key = get_groq_key()
    if not key: return None
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role":"user","content":prompt}],
                  "temperature": 0.3, "max_tokens": 500}, timeout=30)
        return r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else None
    except: return None

def ai_analysis(p1, p2, surface, tournament, proba, best_value=None):
    fav = p1 if proba >= 0.5 else p2
    und = p2 if proba >= 0.5 else p1
    vb  = f"Value bet: {best_value['joueur']} @ {best_value['cote']:.2f} (edge +{best_value['edge']*100:.1f}%)" if best_value else ""
    return call_groq(f"""Analyse ce match ATP en 4 points précis:

{p1} vs {p2} | {tournament} | Surface: {surface}
Proba: {p1} {proba:.1%} — {p2} {1-proba:.1%}
FAVORI: {fav} ({max(proba,1-proba):.1%}){f' | {vb}' if vb else ''}

1. Pourquoi {fav} est favori (2-3 arguments clés)
2. Points faibles de {und} dans ce contexte
3. {'Value bet: ' + vb if vb else 'Conseil pari optimal'}
4. Pronostic final et niveau de confiance

Réponds en français, sois concis et factuel.""")

# ─────────────────────────────────────────────────────────────
# MODÈLE ML
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    mp = MODELS_DIR / "tennis_ml_model_complete.pkl"
    if mp.exists():
        try: return joblib.load(mp)
        except: return None
    try:
        with st.spinner("📥 Téléchargement du modèle ML..."):
            r = requests.get(
                "https://github.com/Xela91300/sports-betting-neural-net/releases/download/v1.0.0/tennis_ml_model_complete.pkl.gz",
                timeout=60)
            if r.status_code == 200:
                tmp = MODELS_DIR / "tmp.pkl.gz"
                tmp.write_bytes(r.content)
                with gzip.open(tmp, "rb") as f:
                    mi = joblib.load(f)
                joblib.dump(mi, mp); tmp.unlink()
                return mi
    except: pass
    return None

def extract_21_features(ps, p1, p2, surface, level="A", best_of=3, h2h_ratio=0.5):
    s1, s2 = ps.get(p1, {}), ps.get(p2, {})
    r1 = max(s1.get("rank", 500.0), 1.0)
    r2 = max(s2.get("rank", 500.0), 1.0)
    sp1, sp2 = s1.get("serve_pct", {}), s2.get("serve_pct", {})
    sr1, sr2 = s1.get("serve_raw", {}), s2.get("serve_raw", {})
    feats = [
        float(np.log(r2/r1)),
        (s1.get("rank_points",0) - s2.get("rank_points",0)) / 5000.0,
        float(s1.get("age",25) - s2.get("age",25)),
        1.0 if surface=="Clay"  else 0.0,
        1.0 if surface=="Grass" else 0.0,
        1.0 if surface=="Hard"  else 0.0,
        1.0 if level=="G" else 0.0,
        1.0 if level=="M" else 0.0,
        1.0 if best_of==5 else 0.0,
        float(s1.get("surface_wr",{}).get(surface,0.5) - s2.get("surface_wr",{}).get(surface,0.5)),
        float(s1.get("win_rate",0.5) - s2.get("win_rate",0.5)),
        float(s1.get("recent_form",0.5) - s2.get("recent_form",0.5)),
        float(h2h_ratio),
        (sr1.get("ace",0) - sr2.get("ace",0)) / 10.0,
        (sr1.get("df",0) - sr2.get("df",0)) / 5.0,
        float(sp1.get("pct_1st_in",0) - sp2.get("pct_1st_in",0)),
        float(sp1.get("pct_1st_won",0) - sp2.get("pct_1st_won",0)),
        float(sp1.get("pct_2nd_won",0) - sp2.get("pct_2nd_won",0)),
        float(sp1.get("pct_bp_saved",0) - sp2.get("pct_bp_saved",0)),
        float(s1.get("days_since_last",30) - s2.get("days_since_last",30)),
        float(s1.get("fatigue",0) - s2.get("fatigue",0)),
    ]
    return np.nan_to_num(np.array(feats, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

def predict_ml(mi, p1, p2, surface, tournament="", h2h_ratio=0.5):
    if mi is None: return None
    try:
        m = mi.get("model"); sc = mi.get("scaler"); ps = mi.get("player_stats", {})
        if m is None or sc is None: return None
        if p1 not in ps or p2 not in ps: return None
        lv, bo = get_level(tournament)
        f = extract_21_features(ps, p1, p2, surface, lv, bo, h2h_ratio)
        p = float(m.predict_proba(sc.transform(f.reshape(1,-1)))[0][1])
        return max(0.05, min(0.95, p))
    except: return None

# ─────────────────────────────────────────────────────────────
# DONNÉES CSV
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_players():
    if not DATA_DIR.exists(): return []
    players = set()
    for f in DATA_DIR.glob("*.csv"):
        if "wta" in f.name.lower(): continue
        try:
            for enc in ["utf-8","latin-1","cp1252"]:
                try:
                    df = pd.read_csv(f, encoding=enc, usecols=["winner_name","loser_name"], on_bad_lines="skip")
                    players.update(df["winner_name"].dropna().astype(str).str.strip())
                    players.update(df["loser_name"].dropna().astype(str).str.strip())
                    break
                except: continue
        except: pass
    return sorted(p for p in players if p and p.lower() != "nan" and len(p) > 1)

@st.cache_data(ttl=3600)
def load_h2h_df():
    if not DATA_DIR.exists(): return pd.DataFrame()
    dfs = []
    for f in list(DATA_DIR.glob("*.csv"))[:20]:
        if "wta" in f.name.lower(): continue
        try:
            df = pd.read_csv(f, encoding="utf-8", usecols=["winner_name","loser_name"], on_bad_lines="skip")
            df["winner_name"] = df["winner_name"].astype(str).str.strip()
            df["loser_name"]  = df["loser_name"].astype(str).str.strip()
            dfs.append(df)
        except: continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def get_h2h(p1, p2):
    df = load_h2h_df()
    if df.empty: return None
    mask = (((df.winner_name==p1)&(df.loser_name==p2))|
            ((df.winner_name==p2)&(df.loser_name==p1)))
    h = df[mask]
    if len(h) == 0: return None
    return {"total": len(h), "p1_wins": len(h[h.winner_name==p1]), "p2_wins": len(h[h.winner_name==p2])}

def h2h_ratio(h2h, p1):
    if not h2h or h2h["total"] == 0: return 0.5
    return h2h["p1_wins"] / h2h["total"]

def calc_proba(p1, p2, surface, tournament="", h2h=None, mi=None):
    ratio = h2h_ratio(h2h, p1)
    if mi:
        p = predict_ml(mi, p1, p2, surface, tournament, ratio)
        if p is not None: return p, True
    proba = 0.5 + (ratio - 0.5) * 0.3
    return max(0.05, min(0.95, proba)), False

def calc_confidence(proba, h2h=None):
    c = 50.0
    if h2h and h2h.get("total",0) >= 3: c += 10
    c += abs(proba - 0.5) * 40
    return min(100.0, c)

# ─────────────────────────────────────────────────────────────
# HISTORIQUE & STATISTIQUES
# ─────────────────────────────────────────────────────────────
def load_history():
    if not HIST_FILE.exists(): return []
    try:
        with open(HIST_FILE,"r",encoding="utf-8") as f: return json.load(f)
    except: return []

def save_pred(pred):
    try:
        h = load_history()
        pred["id"] = hashlib.md5(f"{datetime.now()}{pred.get('player1','')}".encode()).hexdigest()[:8]
        pred["statut"] = "en_attente"
        pred["vainqueur_reel"] = None
        pred["pronostic_correct"] = None
        h.append(pred)
        with open(HIST_FILE,"w",encoding="utf-8") as f: json.dump(h[-1000:], f, indent=2, ensure_ascii=False)
        return True
    except: return False

def update_pred_result(pred_id, statut, vainqueur_reel=None):
    """
    Met à jour le résultat d'une prédiction.
    statut: gagne | perdu | annule
    vainqueur_reel: nom du joueur qui a vraiment gagné (ou None pour abandon)
    """
    try:
        h = load_history()
        for p in h:
            if p.get("id") == pred_id:
                p["statut"] = statut
                p["date_maj"] = datetime.now().isoformat()
                p["vainqueur_reel"] = vainqueur_reel
                # Calcul automatique si le pronostic était correct
                if vainqueur_reel:
                    p["pronostic_correct"] = (vainqueur_reel == p.get("favori"))
                else:
                    p["pronostic_correct"] = None
                break
        with open(HIST_FILE,"w",encoding="utf-8") as f: json.dump(h, f, indent=2, ensure_ascii=False)
        update_stats()
        return True
    except: return False

def load_user_stats():
    default = {"total_predictions":0,"correct_predictions":0,
               "incorrect_predictions":0,"annules_predictions":0,
               "current_streak":0,"best_streak":0}
    if not USER_STATS_FILE.exists(): return default
    try:
        with open(USER_STATS_FILE) as f: return json.load(f)
    except: return default

def update_stats():
    h = load_history()
    correct   = sum(1 for p in h if p.get("statut") == "gagne")
    incorrect = sum(1 for p in h if p.get("statut") == "perdu")
    cancel    = sum(1 for p in h if p.get("statut") == "annule")
    streak = cur = best = 0
    for p in reversed(h):
        if p.get("statut") == "gagne": streak+=1; cur=streak; best=max(best,streak)
        elif p.get("statut") == "perdu": streak=0; cur=0
    stats = {"total_predictions": len(h), "correct_predictions": correct,
             "incorrect_predictions": incorrect, "annules_predictions": cancel,
             "current_streak": cur, "best_streak": best}
    with open(USER_STATS_FILE,"w") as f: json.dump(stats, f)
    return stats

def calc_accuracy():
    s = load_user_stats()
    tv = s.get("correct_predictions",0) + s.get("incorrect_predictions",0)
    return (s.get("correct_predictions",0)/tv*100) if tv > 0 else 0

# ─────────────────────────────────────────────────────────────
# ACHIEVEMENTS
# ─────────────────────────────────────────────────────────────
def load_ach():
    if not ACHIEVEMENTS_FILE.exists(): return {}
    try:
        with open(ACHIEVEMENTS_FILE) as f: return json.load(f)
    except: return {}

def save_ach(a):
    try:
        with open(ACHIEVEMENTS_FILE,"w") as f: json.dump(a, f)
    except: pass

def check_achievements():
    s = load_user_stats(); h = load_history(); a = load_ach(); new = []
    checks = [
        ("first_win", s.get("correct_predictions",0) >= 1),
        ("streak_5",  s.get("best_streak",0) >= 5),
        ("streak_10", s.get("best_streak",0) >= 10),
        ("pred_100",  s.get("total_predictions",0) >= 100),
    ]
    for aid, cond in checks:
        if cond and aid not in a:
            a[aid] = {"unlocked_at": datetime.now().isoformat()}
            new.append(ACHIEVEMENTS[aid])
    value_wins = sum(1 for p in h if p.get("best_value") and p.get("statut")=="gagne")
    if value_wins >= 10 and "value_master" not in a:
        a["value_master"] = {"unlocked_at": datetime.now().isoformat()}
        new.append(ACHIEVEMENTS["value_master"])
    surfs = {p.get("surface") for p in h if p.get("statut")=="gagne"}
    if len(surfs) >= 3 and "surface_specialist" not in a:
        a["surface_specialist"] = {"unlocked_at": datetime.now().isoformat()}
        new.append(ACHIEVEMENTS["surface_specialist"])
    if new: save_ach(a)
    return new

# ─────────────────────────────────────────────────────────────
# BACKUP
# ─────────────────────────────────────────────────────────────
def backup():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for f in [HIST_FILE, USER_STATS_FILE]:
        if f.exists():
            try: shutil.copy(f, BACKUP_DIR / f"{f.stem}_{ts}{f.suffix}")
            except: pass

# ─────────────────────────────────────────────────────────────
# MATCHS DU JOUR
# ─────────────────────────────────────────────────────────────
def mock_matches():
    return [
        {"p1":"Novak Djokovic","p2":"Carlos Alcaraz","surface":"Clay","tournament":"Roland Garros"},
        {"p1":"Jannik Sinner","p2":"Daniil Medvedev","surface":"Hard","tournament":"Miami Open"},
        {"p1":"Alexander Zverev","p2":"Stefanos Tsitsipas","surface":"Clay","tournament":"Madrid Open"},
        {"p1":"Holger Rune","p2":"Casper Ruud","surface":"Grass","tournament":"Wimbledon"},
    ]

@st.cache_data(ttl=1800)
def get_matches(force=False):
    if force: st.cache_data.clear()
    return mock_matches()

def alt_bets(p1, p2, surface, proba):
    bets = []
    if proba > 0.6 or proba < 0.4:
        bets.append({"type":"📊 Under 22.5 games","description":"Moins de 22.5 jeux au total","proba":0.64,"cote":1.78,"confidence":68})
    else:
        bets.append({"type":"📊 Over 22.5 games","description":"Plus de 22.5 jeux au total","proba":0.61,"cote":1.82,"confidence":63})
    if proba > 0.65:
        bets.append({"type":"⚖️ Handicap -3.5 jeux","description":f"{p1} gagne avec écart","proba":0.57,"cote":2.15,"confidence":58})
    elif proba < 0.35:
        bets.append({"type":"⚖️ Handicap +3.5 jeux","description":f"{p2} perd par moins de 4","proba":0.60,"cote":1.98,"confidence":62})
    if 0.3 < proba < 0.7:
        bets.append({"type":"🔄 Chaque joueur gagne un set","description":"Match en au moins 2 sets chacun","proba":0.54,"cote":2.25,"confidence":54})
    return bets

# ─────────────────────────────────────────────────────────────
# COMPOSANTS SÉLECTEURS
# ─────────────────────────────────────────────────────────────
def player_sel(label, all_players, key, default=None):
    search = st.text_input(f"🔍 {label}", key=f"srch_{key}", placeholder="Tapez un nom...")
    filtered = ([p for p in all_players if search.lower() in p.lower()]
                if search else all_players[:200])
    if not filtered: filtered = [p for p in all_players if p[0].lower()==search[0].lower()][:50] if search else []
    st.caption(f"{len(filtered)} sur {len(all_players):,} joueurs")
    if not filtered: return st.text_input(label, key=key)
    idx = 0
    if default:
        for i, p in enumerate(filtered):
            if default.lower() in p.lower(): idx = i; break
    return st.selectbox(label, filtered, index=idx, key=key)

def tourn_sel(label, key, default=None):
    search = st.text_input(f"🔍 {label}", key=f"srcht_{key}", placeholder="ex: Roland Garros, wimbledon...")
    all_t = sorted(TOURNAMENTS_DB.keys())
    if search:
        sl = search.lower().strip()
        res = set()
        if sl in TOURNAMENT_ALIASES: res.add(TOURNAMENT_ALIASES[sl])
        for t in all_t:
            if sl in t.lower(): res.add(t)
        for a, o in TOURNAMENT_ALIASES.items():
            if sl in a: res.add(o)
        filtered = sorted(res) if res else all_t[:50]
    else:
        filtered = all_t[:100]
    idx = filtered.index(default) if default and default in filtered else 0
    return st.selectbox(label, filtered, index=idx, key=key)

# ═══════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# DASHBOARD PRO
# ─────────────────────────────────────────────────────────────
def show_dashboard():
    st.markdown(section_title("🏠 Dashboard", "Vue d'ensemble de vos performances"), unsafe_allow_html=True)

    stats = load_user_stats()
    h = load_history()
    a = load_ach()
    mi = load_model()

    correct   = stats.get("correct_predictions", 0)
    wrong     = stats.get("incorrect_predictions", 0)
    cancel    = stats.get("annules_predictions", 0)
    pending   = len([p for p in h if p.get("statut") == "en_attente"])
    tv        = correct + wrong
    acc       = (correct / tv * 100) if tv > 0 else 0

    # Forme récente pour delta
    recent = [p for p in h[-20:] if p.get("statut") in ["gagne","perdu"]]
    r_acc  = (sum(1 for p in recent if p.get("statut")=="gagne") / len(recent) * 100) if recent else 0

    # ── KPI Row ──────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(big_metric("PRÉCISION", f"{acc:.1f}%", r_acc-acc if tv>0 else None, "🎯", "#00DFA2"), unsafe_allow_html=True)
    with c2: st.markdown(big_metric("GAGNÉS", str(correct), None, "✅", "#00DFA2"), unsafe_allow_html=True)
    with c3: st.markdown(big_metric("PERDUS", str(wrong), None, "❌", "#FF4757"), unsafe_allow_html=True)
    with c4: st.markdown(big_metric("ABANDONS", str(cancel), None, "⚠️", "#FFB200"), unsafe_allow_html=True)
    with c5: st.markdown(big_metric("EN ATTENTE", str(pending), None, "⏳", "#7A8599"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Série + Statut services ───────────────────
    col_l, col_r = st.columns([1, 2])

    with col_l:
        streak = stats.get("current_streak", 0)
        best   = stats.get("best_streak", 0)
        fire = "🔥" if streak >= 5 else "⚡" if streak >= 3 else ""
        st.markdown(f"""
        <div style="background:{COLORS['card_bg']};border:1px solid {'#00DFA244' if streak>0 else COLORS['card_border']};
        border-radius:16px;padding:1.5rem;text-align:center;height:100%;">
            <div style="font-size:2rem;">{fire or '🎾'}</div>
            <div style="font-family:Syne,sans-serif;font-size:3rem;font-weight:800;
            color:{'#00DFA2' if streak > 0 else '#7A8599'};">{streak}</div>
            <div style="color:{COLORS['gray']};font-size:0.85rem;text-transform:uppercase;
            letter-spacing:0.1em;">Série actuelle</div>
            <div style="margin-top:0.75rem;padding-top:0.75rem;border-top:1px solid {COLORS['card_border']};">
                <span style="color:{COLORS['gray']};font-size:0.8rem;">Record: </span>
                <span style="color:#FFB200;font-weight:700;font-size:1rem;">⚡ {best}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_r:
        # Statut services
        tg_token, _ = get_tg_config()
        groq_key    = get_groq_key()
        st.markdown(f"""
        <div style="background:{COLORS['card_bg']};border:1px solid {COLORS['card_border']};
        border-radius:16px;padding:1.5rem;">
            <div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;
            color:#E8EDF5;margin-bottom:1rem;">⚙️ STATUT DES SERVICES</div>
            <div style="display:grid;gap:0.75rem;">""", unsafe_allow_html=True)

        services = []
        if mi:
            ps = mi.get("player_stats", {})
            services.append(("🤖 Modèle ML", f"{mi.get('accuracy',0):.1%} acc · {len(ps):,} joueurs", True))
        else:
            services.append(("🤖 Modèle ML", "Non chargé — mode CSV actif", False))
        services.append(("🧠 IA Groq", "Connectée" if groq_key else "Non configurée", bool(groq_key)))
        services.append(("📱 Telegram", "Configuré" if tg_token else "Non configuré", bool(tg_token)))

        for svc, desc, ok in services:
            color = "#00DFA2" if ok else "#FF4757"
            dot   = "●" if ok else "○"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem;
            background:rgba(255,255,255,0.03);border-radius:8px;">
                <span style="color:{color};font-size:0.8rem;">{dot}</span>
                <span style="font-weight:600;color:#E8EDF5;flex:1;">{svc}</span>
                <span style="color:{COLORS['gray']};font-size:0.8rem;">{desc}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Graphique évolution ────────────────────────
    finished = [p for p in h if p.get("statut") in ["gagne","perdu"]]
    if len(finished) >= 3:
        st.markdown(f"""<div style="font-family:Syne,sans-serif;font-size:1.1rem;
        font-weight:700;color:#E8EDF5;margin-bottom:0.75rem;">📈 Évolution de la précision</div>""",
        unsafe_allow_html=True)

        df_h = pd.DataFrame(finished)
        df_h["ok"] = (df_h["statut"] == "gagne").astype(int)
        df_h["cum_ok"] = df_h["ok"].expanding().sum()
        df_h["cum_n"]  = range(1, len(df_h)+1)
        df_h["acc"]    = df_h["cum_ok"] / df_h["cum_n"] * 100
        df_h["n"]      = range(1, len(df_h)+1)

        fig = go.Figure()
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.15)", annotation_text="50%")
        fig.add_trace(go.Scatter(
            x=df_h["n"], y=df_h["acc"],
            mode="lines", name="Précision",
            line=dict(color="#00DFA2", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,223,162,0.07)"
        ))
        fig.update_layout(
            height=260, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#7A8599", family="DM Sans"),
            xaxis=dict(showgrid=False, title="Prédiction #", color="#7A8599"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       title="Précision (%)", color="#7A8599", range=[0,100]),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Performance par surface ───────────────────
    col_surf, col_badges = st.columns([3, 2])

    with col_surf:
        surf_data = []
        for surf in SURFACES:
            sp = [p for p in h if p.get("surface")==surf and p.get("statut") in ["gagne","perdu"]]
            if sp:
                ok = sum(1 for p in sp if p.get("statut")=="gagne")
                surf_data.append({"Surface":surf, "Précision":ok/len(sp)*100,
                                  "Total":len(sp), "Gagnés":ok})
        if surf_data:
            st.markdown(f"""<div style="font-family:Syne,sans-serif;font-size:1.1rem;
            font-weight:700;color:#E8EDF5;margin-bottom:0.75rem;">🎾 Par surface</div>""",
            unsafe_allow_html=True)
            df_s = pd.DataFrame(surf_data)
            fig2 = go.Figure(go.Bar(
                x=df_s["Surface"], y=df_s["Précision"],
                text=df_s["Précision"].round(0).astype(int).astype(str) + "%",
                textposition="outside",
                marker_color=[SURFACE_CFG[s]["color"] for s in df_s["Surface"]],
                hovertemplate="<b>%{x}</b><br>Précision: %{y:.1f}%<br>Matchs: %{customdata}<extra></extra>",
                customdata=df_s["Total"]
            ))
            fig2.update_layout(
                height=220, margin=dict(l=0,r=0,t=30,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#7A8599", family="DM Sans"),
                xaxis=dict(showgrid=False, color="#7A8599"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                           color="#7A8599", range=[0,110]),
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

    with col_badges:
        st.markdown(f"""<div style="font-family:Syne,sans-serif;font-size:1.1rem;
        font-weight:700;color:#E8EDF5;margin-bottom:0.75rem;">🏆 Badges ({len(a)}/{len(ACHIEVEMENTS)})</div>""",
        unsafe_allow_html=True)
        if a:
            for aid, adata_val in list(a.items())[:4]:
                ach_meta = ACHIEVEMENTS.get(aid, {})
                try:
                    d = datetime.fromisoformat(adata_val["unlocked_at"]).strftime("%d/%m/%Y")
                except: d = "?"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.75rem;
                background:rgba(0,223,162,0.06);border:1px solid rgba(0,223,162,0.2);
                border-radius:10px;padding:0.6rem 0.9rem;margin-bottom:0.5rem;">
                    <span style="font-size:1.5rem;">{ach_meta.get('icon','🏆')}</span>
                    <div>
                        <div style="font-weight:700;color:#00DFA2;font-size:0.85rem;">{ach_meta.get('name','')}</div>
                        <div style="color:{COLORS['gray']};font-size:0.72rem;">Débloqué le {d}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:2rem;color:{COLORS['gray']};
            border:1px dashed {COLORS['card_border']};border-radius:12px;">
                Aucun badge encore<br><small>Faites des prédictions !</small>
            </div>""", unsafe_allow_html=True)

    # ── Bouton Telegram stats ─────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📱 Envoyer les stats sur Telegram", use_container_width=False):
        ok, msg = tg_send(format_stats_msg())
        st.success(msg) if ok else st.error(msg)


# ─────────────────────────────────────────────────────────────
# ANALYSE MULTI-MATCHS
# ─────────────────────────────────────────────────────────────
def show_prediction():
    st.markdown(section_title("🎯 Analyse Multi-matchs", "Prédictions ML avec toutes les features"), unsafe_allow_html=True)

    mi = load_model()
    if mi:
        ps = mi.get("player_stats", {})
        st.markdown(f"""
        <div style="background:rgba(0,223,162,0.08);border:1px solid rgba(0,223,162,0.25);
        border-radius:12px;padding:0.75rem 1rem;margin-bottom:1rem;display:flex;align-items:center;gap:0.75rem;">
            <span style="font-size:1.2rem;">🤖</span>
            <div>
                <span style="font-weight:700;color:#00DFA2;">Modèle ML actif</span>
                <span style="color:{COLORS['gray']};font-size:0.85rem;margin-left:0.75rem;">
                    {mi.get('accuracy',0):.1%} accuracy · {len(ps):,} joueurs · 21 features
                </span>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Modèle ML non chargé — prédictions en mode statistiques CSV")

    with st.spinner("Chargement des joueurs..."):
        all_p = load_players()

    c1, c2, c3 = st.columns(3)
    with c1: n = st.number_input("Nombre de matchs", 1, MAX_MATCHES, 2)
    with c2: use_ai = st.checkbox("🤖 Analyse IA", True)
    with c3: send_tg = st.checkbox("📱 Envoi Telegram auto", False)

    today = st.session_state.get("today_matches", [])
    inputs = []
    st.markdown(f"""<div style="font-family:Syne,sans-serif;font-size:1.1rem;
    font-weight:700;color:#E8EDF5;margin:1.5rem 0 0.75rem;">📝 Saisie des matchs</div>""",
    unsafe_allow_html=True)

    for i in range(n):
        with st.expander(f"Match {i+1}", expanded=(i==0)):
            ct, cs = st.columns([3,1])
            with ct:
                tourn = tourn_sel("Tournoi", f"t{i}", today[i]["tournament"] if i<len(today) else "Roland Garros")
            with cs:
                surf = get_surface(tourn)
                lv, bo = get_level(tourn)
                cfg = SURFACE_CFG[surf]
                st.markdown(f"""<div style="background:{cfg['bg']};border:1px solid {cfg['color']}55;
                border-radius:10px;padding:0.6rem;text-align:center;margin-top:1.75rem;">
                    <div style="font-size:1.3rem;">{cfg['icon']}</div>
                    <div style="font-weight:700;color:{cfg['color']};font-size:0.9rem;">{surf}</div>
                    {'<div style="font-size:0.7rem;color:#7A8599;">Best of 5</div>' if bo==5 else ''}
                </div>""", unsafe_allow_html=True)

            cp1, cp2 = st.columns(2)
            with cp1:
                p1 = player_sel("Joueur 1", all_p, f"p1_{i}", today[i]["p1"] if i<len(today) else "")
                o1 = st.text_input(f"Cote {p1[:15] if p1 else 'J1'}", key=f"o1_{i}", placeholder="1.75")
            with cp2:
                p2_list = [p for p in all_p if p != p1]
                p2 = player_sel("Joueur 2", p2_list, f"p2_{i}", today[i]["p2"] if i<len(today) else "")
                o2 = st.text_input(f"Cote {p2[:15] if p2 else 'J2'}", key=f"o2_{i}", placeholder="2.10")

            if mi and p1 and p2:
                ps_d = mi.get("player_stats", {})
                p1k  = "✅" if p1 in ps_d else "⚠️ inconnu"
                p2k  = "✅" if p2 in ps_d else "⚠️ inconnu"
                st.caption(f"ML: {p1[:20]} {p1k} · {p2[:20]} {p2k}")

            inputs.append({"p1":p1,"p2":p2,"surf":surf,"tourn":tourn,"o1":o1,"o2":o2})

    if not st.button("🔍 Analyser tous les matchs", type="primary", use_container_width=True):
        return

    valid = [m for m in inputs if m["p1"] and m["p2"]]
    if not valid: st.warning("Remplis au moins un match"); return

    st.markdown("---")
    st.markdown(section_title("📊 Résultats de l'analyse"), unsafe_allow_html=True)

    for i, m in enumerate(valid):
        p1, p2, surf, tourn = m["p1"], m["p2"], m["surf"], m["tourn"]

        h2h_data = get_h2h(p1, p2)
        proba, ml_used = calc_proba(p1, p2, surf, tourn, h2h_data, mi)
        conf = calc_confidence(proba, h2h_data)
        fav  = p1 if proba >= 0.5 else p2
        und  = p2 if proba >= 0.5 else p1

        cfg = SURFACE_CFG[surf]
        # Header match
        st.markdown(f"""
        <div style="background:{COLORS['card_bg']};border:1px solid {COLORS['card_border']};
        border-radius:16px;padding:1.5rem;margin-bottom:0.5rem;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;">
                <div>
                    <span style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;
                    color:#E8EDF5;">Match {i+1}</span>
                    <span style="margin-left:0.75rem;">{surface_badge(surf)}</span>
                    <span style="color:{COLORS['gray']};font-size:0.85rem;margin-left:0.5rem;">🏆 {tourn}</span>
                </div>
                <span style="color:{'#00DFA2' if ml_used else '#7A8599'};font-size:0.8rem;
                font-weight:600;">{'🤖 ML · 21 features' if ml_used else '📊 Fallback CSV'}</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:1rem;align-items:center;margin-bottom:1rem;">
                <div style="text-align:center;">
                    <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                    color:#E8EDF5;">{p1}</div>
                    <div style="font-size:1.8rem;font-weight:800;
                    color:{'#00DFA2' if fav==p1 else '#7A8599'};">{proba:.1%}</div>
                    {'<div style="color:#00DFA2;font-size:0.75rem;font-weight:700;">⭐ FAVORI</div>' if fav==p1 else ''}
                </div>
                <div style="text-align:center;color:{COLORS['gray']};font-weight:700;font-size:1.2rem;">VS</div>
                <div style="text-align:center;">
                    <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                    color:#E8EDF5;">{p2}</div>
                    <div style="font-size:1.8rem;font-weight:800;
                    color:{'#00DFA2' if fav==p2 else '#7A8599'};">{1-proba:.1%}</div>
                    {'<div style="color:#00DFA2;font-size:0.75rem;font-weight:700;">⭐ FAVORI</div>' if fav==p2 else ''}
                </div>
            </div>""", unsafe_allow_html=True)

        st.progress(float(proba))

        # Confiance + H2H
        ci = "🟢" if conf>=70 else "🟡" if conf>=50 else "🔴"
        h2h_str = f"H2H {h2h_data['p1_wins']}-{h2h_data['p2_wins']} ({h2h_data['total']} matchs)" if h2h_data else "H2H: aucun"
        st.markdown(f"""
            <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.75rem;">
                {stat_pill("Confiance", f"{conf:.0f}/100", "#00DFA2", ci)}
                {stat_pill("H2H", h2h_str, "#0079FF", "📊")}
                {stat_pill("Format", f"Best of {get_level(tourn)[1]}", "#7A8599", "📋")}
            </div>
        </div>""", unsafe_allow_html=True)

        # Value bet
        best_val = None
        if m["o1"] and m["o2"]:
            try:
                o1f = float(m["o1"].replace(",",".")); o2f = float(m["o2"].replace(",","."))
                e1 = proba - 1/o1f; e2 = (1-proba) - 1/o2f
                if e1 > MIN_EDGE_COMBINE:
                    best_val = {"joueur":p1,"edge":e1,"cote":o1f,"proba":proba}
                elif e2 > MIN_EDGE_COMBINE:
                    best_val = {"joueur":p2,"edge":e2,"cote":o2f,"proba":1-proba}
                if best_val:
                    st.success(f"🎯 **VALUE BET !** {best_val['joueur']} @ {best_val['cote']:.2f} · Edge: **+{best_val['edge']*100:.1f}%**")
            except: pass

        # Paris alternatifs
        bets = alt_bets(p1, p2, surf, proba)
        with st.expander("📊 Paris alternatifs"):
            for b in bets:
                ci2 = "🟢" if b["confidence"]>=65 else "🟡"
                st.markdown(f"{ci2} **{b['type']}** — {b['description']} · Proba {b['proba']:.1%} · Cote {b['cote']:.2f}")

        # IA
        ai_txt = None
        if use_ai and get_groq_key():
            with st.spinner("🤖 Analyse IA..."):
                ai_txt = ai_analysis(p1, p2, surf, tourn, proba, best_val)
                if ai_txt:
                    with st.expander("🤖 Analyse IA"):
                        st.write(ai_txt)

        pred_data = {
            "player1":p1, "player2":p2, "tournament":tourn, "surface":surf,
            "proba":float(proba), "confidence":float(conf),
            "odds1":m["o1"], "odds2":m["o2"], "favori":fav,
            "best_value":best_val, "ml_used":ml_used,
            "date":datetime.now().isoformat()
        }

        # Boutons SÉPARÉS et INDÉPENDANTS
        cb1, cb2 = st.columns(2)
        with cb1:
            if st.button(f"💾 Sauvegarder", key=f"save_{i}", use_container_width=True):
                if save_pred(pred_data):
                    st.success("✅ Prédiction sauvegardée !")
                else:
                    st.error("❌ Erreur de sauvegarde")
        with cb2:
            if st.button(f"📱 Envoyer sur Telegram", key=f"tg_{i}", use_container_width=True):
                msg = format_pred_msg(pred_data, bets, ai_txt)
                ok, resp = tg_send(msg)
                st.success(resp) if ok else st.error(resp)

        # Envoi auto si coché (mais après avoir sauvegardé)
        if send_tg and i == 0:  # Seulement si case cochée au moment de l'analyse
            save_pred(pred_data)
            tg_send(format_pred_msg(pred_data, bets, ai_txt))

        st.markdown("---")

    nb = check_achievements()
    if nb:
        st.balloons()
        st.success(f"🏆 {len(nb)} nouveau(x) badge(s) débloqué(s) !")


# ─────────────────────────────────────────────────────────────
# EN ATTENTE — PRO avec vainqueur + pronostic
# ─────────────────────────────────────────────────────────────
def show_pending():
    st.markdown(section_title("⏳ Prédictions en attente", "Validez les résultats pour mettre à jour les statistiques"), unsafe_allow_html=True)

    h = load_history()
    pending = [p for p in h if p.get("statut") == "en_attente"]

    if not pending:
        st.markdown(f"""
        <div style="text-align:center;padding:3rem;background:{COLORS['card_bg']};
        border:1px dashed {COLORS['card_border']};border-radius:16px;">
            <div style="font-size:3rem;">🎉</div>
            <div style="font-size:1.2rem;font-weight:700;color:#E8EDF5;margin-top:0.5rem;">Aucune prédiction en attente !</div>
            <div style="color:{COLORS['gray']};margin-top:0.25rem;">Toutes vos prédictions ont un résultat.</div>
        </div>""", unsafe_allow_html=True)
        return

    # Compteur
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;">
        <span style="background:rgba(255,178,0,0.15);border:1px solid rgba(255,178,0,0.35);
        color:#FFB200;border-radius:100px;padding:0.35rem 0.9rem;font-weight:700;font-size:0.9rem;">
            ⏳ {len(pending)} prédiction{'s' if len(pending)>1 else ''} en attente
        </span>
    </div>""", unsafe_allow_html=True)

    for pred in reversed(pending):
        pid   = pred.get("id","?")
        p1    = pred.get("player1","?")
        p2    = pred.get("player2","?")
        fav   = pred.get("favori","?")
        surf  = pred.get("surface","Hard")
        tourn = pred.get("tournament","?")
        proba = pred.get("proba", 0.5)
        conf  = pred.get("confidence", 50)
        date_str = pred.get("date","")[:16].replace("T"," ")

        cfg = SURFACE_CFG.get(surf, SURFACE_CFG["Hard"])

        st.markdown(f"""
        <div style="background:{COLORS['card_bg']};border:1px solid {COLORS['card_border']};
        border-radius:16px;padding:1.5rem;margin-bottom:1.25rem;">

            <!-- Header -->
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;">
                <div style="display:flex;align-items:center;gap:0.75rem;">
                    <span style="font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:#E8EDF5;">
                        {p1} <span style="color:{COLORS['gray']};font-weight:400;">vs</span> {p2}
                    </span>
                    {surface_badge(surf)}
                </div>
                <span style="color:{COLORS['gray']};font-size:0.78rem;">📅 {date_str}</span>
            </div>

            <!-- Tournoi + proba -->
            <div style="display:flex;align-items:center;gap:1.5rem;margin-bottom:0.75rem;">
                <span style="color:{COLORS['gray']};font-size:0.85rem;">🏆 {tourn}</span>
                <span style="color:#E8EDF5;font-size:0.85rem;">
                    Favori: <strong style="color:#00DFA2;">{fav}</strong> ({proba if fav==p1 else 1-proba:.1%})
                </span>
                <span style="color:{COLORS['gray']};font-size:0.8rem;">
                    {'🟢' if conf>=70 else '🟡' if conf>=50 else '🔴'} {conf:.0f}/100
                </span>
            </div>""", unsafe_allow_html=True)

        if pred.get("best_value"):
            bv = pred["best_value"]
            st.markdown(f"""
            <div style="background:rgba(0,223,162,0.07);border:1px solid rgba(0,223,162,0.2);
            border-radius:8px;padding:0.5rem 0.75rem;margin-bottom:0.75rem;font-size:0.8rem;">
                🎯 Value bet noté: <strong style="color:#00DFA2;">{bv['joueur']} @ {bv.get('cote','?')}</strong>
                · Edge <strong>+{bv.get('edge',0)*100:.1f}%</strong>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
            <!-- Instruction -->
            <div style="font-weight:600;color:#E8EDF5;font-size:0.9rem;margin-bottom:0.75rem;">
                Qui a gagné ce match ?
            </div>
        </div>""", unsafe_allow_html=True)

        # Boutons résultat — layout propre
        c1, c2, c3 = st.columns([2, 2, 1])

        with c1:
            btn_label = f"✅ {p1[:22]} a gagné"
            btn_color = "✅ Notre pronostic ✓" if fav == p1 else "✅ Notre pronostic ✗"
            if st.button(btn_label, key=f"w1_{pid}", use_container_width=True, type="primary" if fav==p1 else "secondary"):
                statut = "gagne" if fav == p1 else "perdu"
                update_pred_result(pid, statut, vainqueur_reel=p1)
                check_achievements()
                st.rerun()

        with c2:
            btn_label2 = f"✅ {p2[:22]} a gagné"
            if st.button(btn_label2, key=f"w2_{pid}", use_container_width=True, type="primary" if fav==p2 else "secondary"):
                statut = "gagne" if fav == p2 else "perdu"
                update_pred_result(pid, statut, vainqueur_reel=p2)
                check_achievements()
                st.rerun()

        with c3:
            if st.button("⚠️ Abandon", key=f"ab_{pid}", use_container_width=True):
                update_pred_result(pid, "annule", vainqueur_reel=None)
                st.rerun()

        # Aide visuelle sur la logique
        if fav == p1:
            st.caption(f"💡 Notre pronostic: {p1} → si {p1} gagne = ✅ GAGNÉ | si {p2} gagne = ❌ PERDU")
        else:
            st.caption(f"💡 Notre pronostic: {p2} → si {p2} gagne = ✅ GAGNÉ | si {p1} gagne = ❌ PERDU")

        st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# STATISTIQUES COMPLÈTES
# ─────────────────────────────────────────────────────────────
def show_statistics():
    st.markdown(section_title("📊 Statistiques", "Analyse complète de vos performances"), unsafe_allow_html=True)

    h = load_history()
    if not h:
        st.info("Aucune prédiction enregistrée pour le moment.")
        return

    df = pd.DataFrame(h)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["pronostic_correct"] = df["pronostic_correct"].fillna(False)

    fini     = df[df["statut"].isin(["gagne","perdu","annule"])]
    gagnes   = df[df["statut"] == "gagne"]
    perdus   = df[df["statut"] == "perdu"]
    abandons = df[df["statut"] == "annule"]
    pending  = df[df["statut"] == "en_attente"]

    tv   = len(gagnes) + len(perdus)
    acc  = (len(gagnes) / tv * 100) if tv > 0 else 0

    # ── KPI ──────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(big_metric("TOTAL", str(len(df)), None, "📝", "#0079FF"), unsafe_allow_html=True)
    with c2: st.markdown(big_metric("GAGNÉS ✅", str(len(gagnes)), None, "✅", "#00DFA2"), unsafe_allow_html=True)
    with c3: st.markdown(big_metric("PERDUS ❌", str(len(perdus)), None, "❌", "#FF4757"), unsafe_allow_html=True)
    with c4: st.markdown(big_metric("ABANDONS ⚠️", str(len(abandons)), None, "⚠️", "#FFB200"), unsafe_allow_html=True)
    with c5: st.markdown(big_metric("PRÉCISION", f"{acc:.1f}%", None, "🎯", "#00DFA2"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Donut + Table résultats ───────────────────────────
    col_pie, col_table = st.columns([1, 2])

    with col_pie:
        if tv > 0:
            fig_d = go.Figure(go.Pie(
                labels=["Gagnés ✅", "Perdus ❌", "Abandons ⚠️"],
                values=[len(gagnes), len(perdus), len(abandons)],
                hole=0.65,
                marker_colors=["#00DFA2", "#FF4757", "#FFB200"],
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>%{value} matchs (%{percent})<extra></extra>"
            ))
            fig_d.update_layout(
                height=240, margin=dict(l=0,r=0,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#7A8599"),
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, font=dict(size=11, color="#E8EDF5")),
                annotations=[dict(text=f"<b>{acc:.0f}%</b>", x=0.5, y=0.5,
                                   font=dict(size=22, color="#00DFA2", family="Syne"), showarrow=False)]
            )
            st.plotly_chart(fig_d, use_container_width=True)

    with col_table:
        # Table résumée par statut + pronostic
        if not fini.empty:
            st.markdown(f"""<div style="font-family:Syne,sans-serif;font-size:0.95rem;
            font-weight:700;color:#E8EDF5;margin-bottom:0.75rem;">📋 Résultats récents avec pronostic</div>""",
            unsafe_allow_html=True)

            recent_fini = fini.sort_values("date", ascending=False).head(10)
            for _, row in recent_fini.iterrows():
                s     = row.get("statut","?")
                pc    = row.get("pronostic_correct")
                fav_r = row.get("favori","?")
                vr    = row.get("vainqueur_reel","?")
                date_ = str(row.get("date",""))[:10]
                surf_ = row.get("surface","?")

                # Couleur selon résultat
                if s == "gagne":   sc,si = "#00DFA244","✅"
                elif s == "perdu": sc,si = "#FF475744","❌"
                else:              sc,si = "#FFB20044","⚠️"

                # Pronostic correct badge
                if pc is True:   pb = f'<span style="color:#00DFA2;font-size:0.72rem;">🎯 Pronostic ✓</span>'
                elif pc is False: pb = f'<span style="color:#FF4757;font-size:0.72rem;">🎯 Pronostic ✗</span>'
                else:             pb = f'<span style="color:#7A8599;font-size:0.72rem;">⚠️ Abandon</span>'

                vr_str = f"Vainqueur: <strong>{vr}</strong>" if vr else "Match non joué"
                fav_str = f"Notre prono: <strong>{fav_r}</strong>"

                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.75rem;
                background:{sc};border-radius:10px;padding:0.6rem 0.9rem;margin-bottom:0.4rem;">
                    <span style="font-size:1rem;">{si}</span>
                    <div style="flex:1;">
                        <div style="font-size:0.85rem;font-weight:600;color:#E8EDF5;">
                            {row.get('player1','?')} vs {row.get('player2','?')}
                        </div>
                        <div style="font-size:0.75rem;color:{COLORS['gray']};">
                            {fav_str} · {vr_str} · {surface_badge(surf_)}
                        </div>
                    </div>
                    <div style="text-align:right;">
                        {pb}
                        <div style="font-size:0.7rem;color:{COLORS['gray']};margin-top:0.2rem;">{date_}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── Performance par surface ───────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div style="font-family:Syne,sans-serif;font-size:1.1rem;
    font-weight:700;color:#E8EDF5;margin-bottom:1rem;">🎾 Performance par surface</div>""", unsafe_allow_html=True)

    surf_cols = st.columns(3)
    for si, surf in enumerate(SURFACES):
        cfg = SURFACE_CFG[surf]
        sp = df[df["surface"]==surf]
        s_g = len(sp[sp["statut"]=="gagne"])
        s_p = len(sp[sp["statut"]=="perdu"])
        s_a = len(sp[sp["statut"]=="annule"])
        s_tv = s_g + s_p
        s_acc = (s_g/s_tv*100) if s_tv > 0 else 0
        with surf_cols[si]:
            st.markdown(f"""
            <div style="background:{cfg['bg']};border:1px solid {cfg['color']}44;
            border-radius:14px;padding:1.25rem;text-align:center;">
                <div style="font-size:1.8rem;margin-bottom:0.25rem;">{cfg['icon']}</div>
                <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                color:{cfg['color']};">{surf}</div>
                <div style="font-size:2rem;font-weight:800;color:#E8EDF5;margin:0.5rem 0;">{s_acc:.0f}%</div>
                <div style="display:flex;justify-content:center;gap:1rem;font-size:0.8rem;">
                    <span style="color:#00DFA2;">✅ {s_g}</span>
                    <span style="color:#FF4757;">❌ {s_p}</span>
                    <span style="color:#FFB200;">⚠️ {s_a}</span>
                </div>
                <div style="color:{COLORS['gray']};font-size:0.75rem;margin-top:0.25rem;">{len(sp)} matchs total</div>
            </div>""", unsafe_allow_html=True)

    # ── Détail avec pronostics ────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if not fini.empty and "pronostic_correct" in fini.columns:
        correct_pred = fini[fini["pronostic_correct"]==True]
        wrong_pred   = fini[fini["pronostic_correct"]==False]
        abandon_pred = fini[fini["statut"]=="annule"]

        st.markdown(f"""
        <div style="background:{COLORS['card_bg']};border:1px solid {COLORS['card_border']};
        border-radius:16px;padding:1.5rem;">
            <div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;
            color:#E8EDF5;margin-bottom:1rem;">📋 Résumé des pronostics enregistrés</div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">
                <div style="text-align:center;background:rgba(0,223,162,0.08);
                border-radius:12px;padding:1rem;">
                    <div style="font-size:2rem;font-weight:800;color:#00DFA2;">{len(correct_pred)}</div>
                    <div style="font-size:0.8rem;color:#00DFA2;font-weight:600;">🎯 PRONOSTIC CORRECT</div>
                    <div style="font-size:0.75rem;color:{COLORS['gray']};margin-top:0.25rem;">
                        {len(correct_pred)/max(len(fini)-len(abandon_pred),1)*100:.1f}% des matchs joués</div>
                </div>
                <div style="text-align:center;background:rgba(255,71,87,0.08);
                border-radius:12px;padding:1rem;">
                    <div style="font-size:2rem;font-weight:800;color:#FF4757;">{len(wrong_pred)}</div>
                    <div style="font-size:0.8rem;color:#FF4757;font-weight:600;">🎯 PRONOSTIC INCORRECT</div>
                    <div style="font-size:0.75rem;color:{COLORS['gray']};margin-top:0.25rem;">
                        {len(wrong_pred)/max(len(fini)-len(abandon_pred),1)*100:.1f}% des matchs joués</div>
                </div>
                <div style="text-align:center;background:rgba(255,178,0,0.08);
                border-radius:12px;padding:1rem;">
                    <div style="font-size:2rem;font-weight:800;color:#FFB200;">{len(abandon_pred)}</div>
                    <div style="font-size:0.8rem;color:#FFB200;font-weight:600;">⚠️ ABANDONS / ANNULÉS</div>
                    <div style="font-size:0.75rem;color:{COLORS['gray']};margin-top:0.25rem;">
                        Matchs non comptabilisés</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Exporter ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📥 Exporter l'historique en CSV"):
        csv = df.to_csv(index=False, encoding="utf-8")
        st.download_button("⬇️ Télécharger CSV", csv, "tennisiq_history.csv", "text/csv")


# ─────────────────────────────────────────────────────────────
# TELEGRAM PAGE
# ─────────────────────────────────────────────────────────────
def show_telegram():
    st.markdown(section_title("📱 Telegram", "Configuration et envoi des notifications"), unsafe_allow_html=True)

    token, chat_id = get_tg_config()

    if not token or not chat_id:
        st.markdown(f"""
        <div style="background:rgba(255,178,0,0.08);border:1px solid rgba(255,178,0,0.3);
        border-radius:14px;padding:1.5rem;margin-bottom:1.5rem;">
            <div style="font-weight:700;color:#FFB200;margin-bottom:0.75rem;">⚠️ Telegram non configuré</div>
            <div style="color:#E8EDF5;font-size:0.9rem;">Ajoute ces variables dans les secrets Streamlit :</div>
        </div>""", unsafe_allow_html=True)
        st.code('TELEGRAM_BOT_TOKEN = "1234567890:AAExxxxxxxxxxxxxxxx"\nTELEGRAM_CHAT_ID = "-100xxxxxxxxxx"', language="toml")
        with st.expander("📖 Comment obtenir ces valeurs ?"):
            st.markdown("""
1. Sur Telegram, cherche **@BotFather**
2. Tape `/newbot` et suis les instructions
3. BotFather te donnera le **TOKEN**
4. Cherche **@userinfobot** pour obtenir ton **Chat ID**
5. Si tu utilises un canal, le Chat ID commence par `-100`
            """)
        return

    st.markdown(f"""
    <div style="background:rgba(0,223,162,0.08);border:1px solid rgba(0,223,162,0.25);
    border-radius:12px;padding:1rem 1.25rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:0.75rem;">
        <span style="font-size:1.5rem;">✅</span>
        <div>
            <div style="font-weight:700;color:#00DFA2;">Telegram configuré</div>
            <div style="color:{COLORS['gray']};font-size:0.85rem;">Chat ID: {chat_id}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Actions rapides
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔧 Tester la connexion", use_container_width=True):
            with st.spinner("Test en cours..."):
                ok, msg = tg_test()
            st.success(msg) if ok else st.error(msg)
    with c2:
        if st.button("📊 Envoyer les stats", use_container_width=True):
            with st.spinner("Envoi..."):
                ok, msg = tg_send(format_stats_msg())
            st.success(msg) if ok else st.error(msg)
    with c3:
        if st.button("🔄 Vider le cache réseau", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Cache vidé")

    st.markdown("<br>", unsafe_allow_html=True)

    # Message personnalisé
    st.markdown(f"""<div style="font-family:Syne,sans-serif;font-size:1rem;
    font-weight:700;color:#E8EDF5;margin-bottom:0.75rem;">✍️ Message personnalisé</div>""",
    unsafe_allow_html=True)

    with st.form("tg_custom"):
        title = st.text_input("Titre du message", "📢 Message TennisIQ")
        body  = st.text_area("Contenu", height=100, placeholder="Votre message ici...")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1: urgent = st.checkbox("🔴 Marquer URGENT")
        with col_opt2: incl_stats = st.checkbox("📊 Inclure les statistiques")
        submitted = st.form_submit_button("📤 Envoyer", use_container_width=True)

    if submitted:
        if not body:
            st.warning("Le message ne peut pas être vide")
        else:
            prefix = "🔴 <b>URGENT</b> — " if urgent else ""
            stats_section = f"\n\n{format_stats_msg()}" if incl_stats else ""
            msg = f"<b>{prefix}{title}</b>\n\n{body}{stats_section}\n\n📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}"
            with st.spinner("Envoi en cours..."):
                ok, resp = tg_send(msg)
            st.success(resp) if ok else st.error(resp)

    # Diagnostic
    with st.expander("🔍 Diagnostic avancé"):
        if st.button("Vérifier le bot"):
            try:
                r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
                if r.status_code == 200:
                    bot = r.json().get("result", {})
                    st.json({"username": bot.get("username"), "name": bot.get("first_name"), "id": bot.get("id")})
                else:
                    st.error(f"Erreur API: {r.text[:200]}")
            except Exception as e:
                st.error(f"Erreur réseau: {e}")


# ─────────────────────────────────────────────────────────────
# CONFIGURATION PAGE
# ─────────────────────────────────────────────────────────────
def show_config():
    st.markdown(section_title("⚙️ Configuration", "Gestion du modèle et des données"), unsafe_allow_html=True)

    mi = load_model()
    if mi:
        ps  = mi.get("player_stats", {})
        imp = mi.get("feature_importance", {})
        st.markdown(f"""
        <div style="background:rgba(0,223,162,0.06);border:1px solid rgba(0,223,162,0.2);
        border-radius:14px;padding:1.25rem;margin-bottom:1.5rem;">
            <div style="font-family:Syne,sans-serif;font-weight:700;color:#00DFA2;margin-bottom:0.75rem;">
                🤖 Modèle ML actif
            </div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;">
                {big_metric('Accuracy', f"{mi.get('accuracy',0):.1%}", None, '', '#00DFA2')}
                {big_metric('AUC-ROC', f"{mi.get('auc',0):.3f}", None, '', '#0079FF')}
                {big_metric('Joueurs', f"{len(ps):,}", None, '', '#7A8599')}
                {big_metric('Matchs train', f"{mi.get('n_matches',0):,}", None, '', '#7A8599')}
            </div>
            <div style="color:{COLORS['gray']};font-size:0.8rem;margin-top:0.75rem;">
                Entraîné le {mi.get('trained_at','?')[:10]} · Version {mi.get('version','?')}
            </div>
        </div>""", unsafe_allow_html=True)

        if imp:
            st.markdown(f"""<div style="font-family:Syne,sans-serif;font-size:0.95rem;
            font-weight:700;color:#E8EDF5;margin-bottom:0.5rem;">🔍 Top features (importance)</div>""",
            unsafe_allow_html=True)
            sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, val in sorted_imp:
                st.progress(float(val), text=f"{feat}: {val:.1%}")

        if st.button("🔄 Recharger le modèle depuis le disque"):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.warning("⚠️ Aucun modèle ML chargé.")
        st.info("Place le fichier `tennis_ml_model_complete.pkl` dans le dossier `models/`")

    st.markdown("---")
    st.markdown(f"""<div style="font-family:Syne,sans-serif;font-size:1rem;
    font-weight:700;color:#E8EDF5;margin-bottom:0.75rem;">🗑️ Gestion des données</div>""",
    unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🗑️ Effacer l'historique", use_container_width=True):
            if HIST_FILE.exists(): HIST_FILE.unlink()
            update_stats(); st.rerun()
    with c2:
        if st.button("🔄 Recalculer les stats", use_container_width=True):
            update_stats(); st.success("✅ Stats recalculées")
    with c3:
        if st.button("💾 Backup maintenant", use_container_width=True):
            backup(); st.success("✅ Backup effectué")


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
    st.markdown(PRO_CSS, unsafe_allow_html=True)

    # Auto-backup quotidien
    if "last_backup" not in st.session_state:
        st.session_state["last_backup"] = datetime.now()
    if (datetime.now() - st.session_state["last_backup"]).seconds >= 86400:
        backup()
        st.session_state["last_backup"] = datetime.now()

    # ── SIDEBAR ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem 0 1rem;">
            <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
            background:linear-gradient(135deg,#00DFA2,#0079FF);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;">TennisIQ</div>
            <div style="font-size:0.75rem;color:#7A8599;letter-spacing:0.1em;
            text-transform:uppercase;margin-top:0.25rem;">ML · Pro Edition</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        page = st.radio(
            "Nav",
            ["🏠 Dashboard", "🎯 Analyse", "⏳ En Attente", "📊 Statistiques",
             "💎 Value Bets", "📱 Telegram", "⚙️ Configuration"],
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        # Mini stats sidebar
        s = load_user_stats()
        h = load_history()
        acc  = calc_accuracy()
        pend = len([p for p in h if p.get("statut") == "en_attente"])

        st.markdown(f"""
        <div style="padding:0.5rem 0;">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;text-align:center;">
                <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:0.6rem;">
                    <div style="font-size:1.1rem;font-weight:800;color:#00DFA2;">{acc:.1f}%</div>
                    <div style="font-size:0.65rem;color:#7A8599;text-transform:uppercase;">Précision</div>
                </div>
                <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:0.6rem;">
                    <div style="font-size:1.1rem;font-weight:800;color:#FFB200;">{pend}</div>
                    <div style="font-size:0.65rem;color:#7A8599;text-transform:uppercase;">En attente</div>
                </div>
                <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:0.6rem;">
                    <div style="font-size:1.1rem;font-weight:800;color:#00DFA2;">{s.get('correct_predictions',0)}</div>
                    <div style="font-size:0.65rem;color:#7A8599;text-transform:uppercase;">Gagnés</div>
                </div>
                <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:0.6rem;">
                    <div style="font-size:1.1rem;font-weight:800;color:{'#FF4757' if s.get('current_streak',0)==0 else '#00DFA2'};">{s.get('current_streak',0)}</div>
                    <div style="font-size:0.65rem;color:#7A8599;text-transform:uppercase;">Série</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── ROUTING ───────────────────────────────────────────
    if   page == "🏠 Dashboard":      show_dashboard()
    elif page == "🎯 Analyse":         show_prediction()
    elif page == "⏳ En Attente":      show_pending()
    elif page == "📊 Statistiques":    show_statistics()
    elif page == "💎 Value Bets":
        st.markdown(section_title("💎 Value Bets", "Opportunités détectées automatiquement"), unsafe_allow_html=True)
        mi = load_model()
        from_cache = get_matches()
        vbs = []
        for m in from_cache:
            proba, _ = calc_proba(m["p1"], m["p2"], m["surface"], m["tournament"], None, mi)
            o1 = round(1/proba*(0.9+0.2*random.random()), 2)
            o2 = round(1/(1-proba)*(0.9+0.2*random.random()), 2)
            e1 = proba - 1/o1; e2 = (1-proba) - 1/o2
            if e1 > MIN_EDGE_COMBINE:
                vbs.append({"match":f"{m['p1']} vs {m['p2']}","joueur":m["p1"],"edge":e1,"cote":o1,"proba":proba,"surf":m["surface"]})
            elif e2 > MIN_EDGE_COMBINE:
                vbs.append({"match":f"{m['p1']} vs {m['p2']}","joueur":m["p2"],"edge":e2,"cote":o2,"proba":1-proba,"surf":m["surface"]})
        vbs.sort(key=lambda x: x["edge"], reverse=True)
        if vbs:
            for vb in vbs:
                cfg = SURFACE_CFG.get(vb["surf"], SURFACE_CFG["Hard"])
                c1,c2,c3,c4 = st.columns([3,1,1,1])
                with c1:
                    st.markdown(f"**{vb['joueur']}**")
                    st.caption(f"{vb['match']} · {cfg['icon']} {vb['surf']}")
                with c2: st.metric("Cote", f"{vb['cote']:.2f}")
                with c3: st.metric("Edge", f"+{vb['edge']*100:.1f}%")
                with c4: st.metric("Proba", f"{vb['proba']:.1%}")
                st.divider()
        else:
            st.info("Aucun value bet détecté sur les matchs du jour.")
    elif page == "📱 Telegram":        show_telegram()
    elif page == "⚙️ Configuration":   show_config()


if __name__ == "__main__":
    main()
