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
warnings.filterwarnings(‘ignore’)

# ─────────────────────────────────────────────────────────────

# CONFIGURATION DES CHEMINS

# ─────────────────────────────────────────────────────────────

ROOT_DIR   = Path(**file**).parent
MODELS_DIR = ROOT_DIR / “models”
DATA_DIR   = ROOT_DIR / “src” / “data” / “raw” / “tml-tennis”
HIST_DIR   = ROOT_DIR / “history”
BACKUP_DIR = ROOT_DIR / “backups”

for d in [MODELS_DIR, DATA_DIR, HIST_DIR, BACKUP_DIR]:
d.mkdir(exist_ok=True, parents=True)

HIST_FILE         = HIST_DIR / “predictions_history.json”
USER_STATS_FILE   = HIST_DIR / “user_stats.json”
ACHIEVEMENTS_FILE = HIST_DIR / “achievements.json”
METADATA_FILE     = MODELS_DIR / “model_metadata.json”

# ─────────────────────────────────────────────────────────────

# CONSTANTES

# ─────────────────────────────────────────────────────────────

SURFACES         = [“Hard”, “Clay”, “Grass”]
MIN_EDGE_COMBINE = 0.02
MAX_MATCHES      = 30

ACHIEVEMENTS = {
‘first_win’:          {‘name’: ‘🎯 Première victoire’,  ‘desc’: ‘Première prédiction gagnante’,  ‘icon’: ‘🎯’},
‘streak_5’:           {‘name’: ‘🔥 En forme’,           ‘desc’: ‘5 victoires consécutives’,      ‘icon’: ‘🔥’},
‘streak_10’:          {‘name’: ‘⚡ Imbattable’,          ‘desc’: ‘10 victoires consécutives’,     ‘icon’: ‘⚡’},
‘pred_100’:           {‘name’: ‘🏆 Expert’,             ‘desc’: ‘100 prédictions’,               ‘icon’: ‘🏆’},
‘value_master’:       {‘name’: ‘💎 Value Master’,        ‘desc’: ‘10 value bets gagnants’,        ‘icon’: ‘💎’},
‘surface_specialist’: {‘name’: ‘🌍 Multi-surface’,       ‘desc’: ‘Gagnant sur les 3 surfaces’,    ‘icon’: ‘🌍’},
}

TOURNAMENTS_DB = {
“Australian Open”: “Hard”, “Roland Garros”: “Clay”,
“Wimbledon”: “Grass”,      “US Open”: “Hard”,
“Nitto ATP Finals”: “Hard”,
“Indian Wells Masters”: “Hard”, “Miami Open”: “Hard”,
“Monte-Carlo Masters”: “Clay”,  “Madrid Open”: “Clay”,
“Italian Open”: “Clay”,   “Canadian Open”: “Hard”,
“Cincinnati Masters”: “Hard”,   “Shanghai Masters”: “Hard”,
“Paris Masters”: “Hard”,  “Rotterdam Open”: “Hard”,
“Rio Open”: “Clay”,       “Dubai Tennis Championships”: “Hard”,
“Mexican Open”: “Hard”,   “Barcelona Open”: “Clay”,
“Halle Open”: “Grass”,    “Queen’s Club Championships”: “Grass”,
“Hamburg Open”: “Clay”,   “Washington Open”: “Hard”,
“China Open”: “Hard”,     “Japan Open”: “Hard”,
“Vienna Open”: “Hard”,    “Swiss Indoors”: “Hard”,
“Dallas Open”: “Hard”,    “Qatar Open”: “Hard”,
“Adelaide International”: “Hard”, “Auckland Open”: “Hard”,
“Brisbane International”: “Hard”, “Cordoba Open”: “Clay”,
“Buenos Aires”: “Clay”,   “Delray Beach”: “Hard”,
“Marseille Open”: “Hard”, “Santiago”: “Clay”,
“Houston”: “Clay”,        “Marrakech”: “Clay”,
“Estoril”: “Clay”,        “Munich”: “Clay”,
“Geneva”: “Clay”,         “Lyon”: “Clay”,
“Stuttgart”: “Grass”,     “Mallorca”: “Grass”,
“Eastbourne”: “Grass”,    “Newport”: “Grass”,
“Atlanta”: “Hard”,        “Croatia Open Umag”: “Clay”,
“Kitzbühel”: “Clay”,      “Los Cabos”: “Hard”,
“Winston-Salem”: “Hard”,  “Chengdu Open”: “Hard”,
“Sofia”: “Hard”,          “Metz”: “Hard”,
“San Diego”: “Hard”,      “Seoul”: “Hard”,
“Tel Aviv”: “Hard”,       “Florence”: “Hard”,
“Antwerp”: “Hard”,        “Stockholm”: “Hard”,
“Belgrade Open”: “Clay”,  “Autre tournoi”: “Hard”,
}

TOURNAMENT_LEVEL = {
“Australian Open”: (“G”, 5), “Roland Garros”: (“G”, 5),
“Wimbledon”: (“G”, 5),       “US Open”: (“G”, 5),
“Nitto ATP Finals”: (“F”, 3),
“Indian Wells Masters”: (“M”, 3), “Miami Open”: (“M”, 3),
“Monte-Carlo Masters”: (“M”, 3),  “Madrid Open”: (“M”, 3),
“Italian Open”: (“M”, 3),   “Canadian Open”: (“M”, 3),
“Cincinnati Masters”: (“M”, 3),   “Shanghai Masters”: (“M”, 3),
“Paris Masters”: (“M”, 3),
}

TOURNAMENT_ALIASES = {
“acapulco”: “Mexican Open”, “mexican”: “Mexican Open”,
“australian”: “Australian Open”, “melbourne”: “Australian Open”,
“roland garros”: “Roland Garros”, “french open”: “Roland Garros”,
“wimbledon”: “Wimbledon”, “us open”: “US Open”,
“flushing”: “US Open”, “new york”: “US Open”,
“indian wells”: “Indian Wells Masters”, “miami”: “Miami Open”,
“monte carlo”: “Monte-Carlo Masters”, “madrid”: “Madrid Open”,
“rome”: “Italian Open”, “canada”: “Canadian Open”,
“cincinnati”: “Cincinnati Masters”, “shanghai”: “Shanghai Masters”,
“paris masters”: “Paris Masters”, “bercy”: “Paris Masters”,
“rotterdam”: “Rotterdam Open”, “dubai”: “Dubai Tennis Championships”,
“barcelona”: “Barcelona Open”, “halle”: “Halle Open”,
“queens”: “Queen’s Club Championships”, “hamburg”: “Hamburg Open”,
“washington”: “Washington Open”, “beijing”: “China Open”,
“tokyo”: “Japan Open”, “vienna”: “Vienna Open”, “basel”: “Swiss Indoors”,
}

COLORS = {
“primary”: “#00DFA2”, “secondary”: “#0079FF”,
“warning”: “#FFB200”, “danger”: “#FF3B3F”, “gray”: “#6C7A89”,
“hard”: “#0079FF”, “clay”: “#E67E22”, “grass”: “#00DFA2”,
“card_bg”: “rgba(255,255,255,0.04)”, “card_border”: “rgba(255,255,255,0.10)”,
}

SURFACE_CFG = {
“Hard”:  {“color”: “#0079FF”, “icon”: “🟦”, “bg”: “rgba(0,121,255,0.12)”},
“Clay”:  {“color”: “#E67E22”, “icon”: “🟧”, “bg”: “rgba(230,126,34,0.12)”},
“Grass”: {“color”: “#00DFA2”, “icon”: “🟩”, “bg”: “rgba(0,223,162,0.12)”},
}

# ─────────────────────────────────────────────────────────────

# CSS GLOBAL — RESPONSIVE MOBILE + DESKTOP

# ─────────────────────────────────────────────────────────────

PRO_CSS = “””

<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
    --primary: #00DFA2;
    --secondary: #0079FF;
    --bg: #080E1A;
    --card: rgba(255,255,255,0.035);
    --border: rgba(255,255,255,0.08);
    --text: #E8EDF5;
    --muted: #7A8599;
    --success: #00DFA2;
    --warning: #FFB200;
    --danger: #FF4757;
}

/* BASE */
.stApp { background: var(--bg) !important; font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: rgba(8,14,26,0.97) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: var(--text) !important;
    font-size: 0.9rem !important;
    padding: 0.4rem 0 !important;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(135deg, #00DFA2 0%, #0079FF 100%) !important;
    color: #080E1A !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1rem !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,223,162,0.25) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.06) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

/* METRICS */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1rem 1.25rem !important;
}

/* PROGRESS BAR */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00DFA2, #0079FF) !important;
}

/* INPUTS */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input::placeholder { color: var(--muted) !important; }
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ALERTS */
.stSuccess { border-radius: 10px !important; border-left: 3px solid #00DFA2 !important; }
.stInfo    { border-radius: 10px !important; border-left: 3px solid #0079FF !important; }
.stWarning { border-radius: 10px !important; border-left: 3px solid #FFB200 !important; }
.stError   { border-radius: 10px !important; border-left: 3px solid #FF4757 !important; }

/* EXPANDER */
details {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* DIVIDER */
hr { border-color: var(--border) !important; }

/* CHECKBOX */
.stCheckbox label { color: var(--text) !important; font-size: 0.9rem !important; }

/* SCROLLBAR */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }

/* PLOTLY CHARTS */
.js-plotly-plot .plotly { border-radius: 12px; }

/* MOBILE RESPONSIVE */
@media (max-width: 768px) {
    .stButton > button { font-size: 0.82rem !important; padding: 0.45rem 0.75rem !important; }
    h2 { font-size: 1.3rem !important; }
    .block-container { padding: 0.75rem !important; }
}

/* AI RESULT BOX */
.ai-result-box {
    background: rgba(0,121,255,0.06);
    border: 1px solid rgba(0,121,255,0.25);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
    color: #E8EDF5;
    line-height: 1.7;
}

/* VALUE BET BOX */
.value-bet-box {
    background: linear-gradient(135deg, rgba(0,223,162,0.15), rgba(0,121,255,0.10));
    border: 2px solid #00DFA2;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin: 0.75rem 0;
}
</style>

“””

# ─────────────────────────────────────────────────────────────

# COMPOSANTS UI

# ─────────────────────────────────────────────────────────────

def card(content, border_color=None, padding=“1.5rem”):
bc = border_color or COLORS[“card_border”]
return (
f’<div style=“background:{COLORS[“card_bg”]};border:1px solid {bc};’
f’border-radius:16px;padding:{padding};margin-bottom:1rem;”>’
f’{content}</div>’
)

def stat_pill(label, value, color=”#00DFA2”, icon=””):
return (
f’<span style="display:inline-flex;align-items:center;gap:0.4rem;'
f'background:rgba(0,0,0,0.3);border:1px solid {color}33;'
f'border-radius:100px;padding:0.3rem 0.75rem;margin:0.2rem;">’
f’<span style="font-size:0.75rem;color:{color};">{icon}</span>’
f’<span style=“font-size:0.78rem;font-weight:600;color:{COLORS[“gray”]};”>{label}</span>’
f’<span style="font-size:0.85rem;font-weight:700;color:{color};">{value}</span>’
f’</span>’
)

def section_title(title, subtitle=””):
sub = f’<p style=“color:{COLORS[“gray”]};font-size:0.9rem;margin:0.25rem 0 0;”>{subtitle}</p>’ if subtitle else “”
return (
f’<div style="margin-bottom:1.5rem;">’
f’<h2 style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;'
f'color:#E8EDF5;margin:0;letter-spacing:-0.02em;">{title}</h2>’
f’{sub}</div>’
)

def big_metric(label, value, delta=None, icon=””, color=”#00DFA2”):
delta_html = “”
if delta is not None:
dcolor = “#00DFA2” if delta >= 0 else “#FF4757”
darrow = “↑” if delta >= 0 else “↓”
delta_html = f’<div style="font-size:0.75rem;color:{dcolor};margin-top:0.1rem;">{darrow} {abs(delta):.1f}%</div>’
return (
f’<div style=“background:{COLORS[“card_bg”]};border:1px solid {color}33;’
f’border-radius:14px;padding:1rem;text-align:center;”>’
f’<div style="font-size:1.3rem;margin-bottom:0.15rem;">{icon}</div>’
f’<div style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;color:{color};">{value}</div>’
f’{delta_html}’
f’<div style=“font-size:0.72rem;color:{COLORS[“gray”]};margin-top:0.25rem;’
f’text-transform:uppercase;letter-spacing:0.07em;”>{label}</div>’
f’</div>’
)

def surface_badge(surface):
cfg = SURFACE_CFG.get(surface, SURFACE_CFG[“Hard”])
return (
f’<span style=“background:{cfg[“bg”]};color:{cfg[“color”]};’
f’border:1px solid {cfg[“color”]}44;border-radius:100px;’
f’padding:0.18rem 0.55rem;font-size:0.72rem;font-weight:600;”>’
f’{cfg[“icon”]} {surface}</span>’
)

def api_status_bar(mi, groq_ok, serper_ok, tg_ok):
“”“Bandeau de statut des APIs — visible en haut de chaque page.”””
items = []
items.append((“🤖 ML”, mi is not None))
items.append((“🧠 IA Groq”, groq_ok))
items.append((“🔍 Web”, serper_ok))
items.append((“📱 Telegram”, tg_ok))
parts = []
for name, ok in items:
color = “#00DFA2” if ok else “#FF4757”
dot   = “●” if ok else “○”
parts.append(
f’<span style="color:{color};font-size:0.8rem;margin-right:1rem;">’
f’{dot} {name}</span>’
)
st.markdown(
f’<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
f'border-radius:10px;padding:0.5rem 1rem;margin-bottom:1.25rem;display:flex;flex-wrap:wrap;">’
f’{””.join(parts)}</div>’,
unsafe_allow_html=True
)

def value_bet_box_html(joueur, cote, edge, proba):
return (
f’<div style="background:linear-gradient(135deg,rgba(0,223,162,0.18),rgba(0,121,255,0.12));'
f'border:2px solid #00DFA2;border-radius:14px;padding:1.25rem 1.5rem;margin:0.75rem 0;">’
f’<div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:800;'
f'color:#00DFA2;margin-bottom:0.5rem;">💎 VALUE BET DÉTECTÉ</div>’
f’<div style="font-size:1.3rem;font-weight:800;color:#E8EDF5;margin-bottom:0.5rem;">’
f’👉 MISER SUR : <span style="color:#00DFA2;">{joueur}</span></div>’
f’<div style="display:flex;flex-wrap:wrap;gap:1rem;margin-top:0.25rem;">’
f’<span style="color:#FFB200;font-weight:700;">Cote : {cote:.2f}</span>’
f’<span style="color:#00DFA2;font-weight:700;">Edge : +{edge*100:.1f}%</span>’
f’<span style="color:#7A8599;">Proba modèle : {proba:.1%}</span>’
f’</div>’
f’<div style="font-size:0.75rem;color:#7A8599;margin-top:0.5rem;">’
f’La cote du bookmaker sous-estime la probabilité réelle → avantage mathématique confirmé</div>’
f’</div>’
)

# ─────────────────────────────────────────────────────────────

# UTILITAIRES TOURNOIS

# ─────────────────────────────────────────────────────────────

def get_surface(name): return TOURNAMENTS_DB.get(name, “Hard”)
def get_level(name):   return TOURNAMENT_LEVEL.get(name, (“A”, 3))

def find_tournament(s):
if not s: return None
sl = s.lower().strip()
if sl in TOURNAMENT_ALIASES: return TOURNAMENT_ALIASES[sl]
for t in TOURNAMENTS_DB:
if sl == t.lower(): return t
m = [t for t in TOURNAMENTS_DB if sl in t.lower()]
return min(m, key=len) if m else None

# ─────────────────────────────────────────────────────────────

# SECRETS — LECTURE ROBUSTE

# ─────────────────────────────────────────────────────────────

def get_secret(key):
“”“Lit un secret Streamlit, fallback sur variable d’environnement.”””
try:
val = st.secrets.get(key)
if val:
return str(val).strip()
except Exception:
pass
return os.environ.get(key, “”).strip() or None

def get_tg_config():
return get_secret(“TELEGRAM_BOT_TOKEN”), get_secret(“TELEGRAM_CHAT_ID”)

def get_groq_key():    return get_secret(“GROQ_API_KEY”)
def get_serper_key():  return get_secret(“SERPER_API_KEY”)

# ─────────────────────────────────────────────────────────────

# CHARGEMENT DU MODÈLE ML

# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_rf_model():
model_path = MODELS_DIR / “tennis_ml_model_complete.pkl”
if model_path.exists():
try:
model_info = joblib.load(model_path)
if model_info.get(‘model’) and model_info.get(‘scaler’):
return model_info
except Exception as e:
st.error(f”Erreur chargement modèle: {e}”)
try:
with st.spinner(“📥 Téléchargement du modèle ML (77.77% accuracy)…”):
url = “https://github.com/Xela91300/sports-betting-neural-net/releases/latest/download/tennis_ml_model_complete.pkl.gz”
r = requests.get(url, timeout=90)
if r.status_code == 200:
temp_path = MODELS_DIR / “model_temp.pkl.gz”
with open(temp_path, “wb”) as f:
f.write(r.content)
with gzip.open(temp_path, “rb”) as f:
model_info = joblib.load(f)
joblib.dump(model_info, model_path)
temp_path.unlink()
return model_info
except Exception as e:
st.warning(f”⚠️ Impossible de télécharger le modèle: {e}”)
return None

@st.cache_data
def load_model_metadata():
if METADATA_FILE.exists():
try:
with open(METADATA_FILE, ‘r’) as f:
return json.load(f)
except:
return {}
return {}

# ─────────────────────────────────────────────────────────────

# UTILITAIRE

# ─────────────────────────────────────────────────────────────

def _safe_float(val, default=0.0):
try:
return float(val)
except (TypeError, ValueError):
return default

# ─────────────────────────────────────────────────────────────

# TELEGRAM

# ─────────────────────────────────────────────────────────────

def tg_send(message, parse_mode=“HTML”):
token, chat_id = get_tg_config()
if not token or not chat_id:
return False, “❌ Telegram non configuré (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID absents des secrets)”
message = str(message)
if len(message) > 4000:
message = message[:3990] + “\n…<i>(tronqué)</i>”
try:
url     = f”https://api.telegram.org/bot{token}/sendMessage”
payload = {
“chat_id”: chat_id,
“text”: message,
“parse_mode”: parse_mode,
“disable_web_page_preview”: True
}
r    = requests.post(url, json=payload, timeout=25)
data = r.json()
if r.status_code == 200 and data.get(“ok”):
return True, “✅ Message envoyé sur Telegram”
err = data.get(“description”, f”HTTP {r.status_code}”)
# Retry sans HTML si erreur de parsing
if “can’t parse” in err.lower() or “parse” in err.lower():
plain = (message
.replace(”<b>”,””).replace(”</b>”,””)
.replace(”<i>”,””).replace(”</i>”,””)
.replace(”<code>”,””).replace(”</code>”,””))
r2 = requests.post(url, json={“chat_id”: chat_id, “text”: plain,
“disable_web_page_preview”: True}, timeout=25)
if r2.status_code == 200 and r2.json().get(“ok”):
return True, “✅ Message envoyé (texte brut)”
return False, f”❌ Telegram: {err}”
except requests.exceptions.ConnectionError:
return False, “❌ Pas de connexion réseau vers Telegram”
except requests.exceptions.Timeout:
return False, “❌ Timeout Telegram (>25s)”
except Exception as e:
return False, f”❌ Erreur: {str(e)[:100]}”

def tg_test():
token, chat_id = get_tg_config()
if not token:  return False, “❌ TELEGRAM_BOT_TOKEN manquant dans les secrets”
if not chat_id: return False, “❌ TELEGRAM_CHAT_ID manquant dans les secrets”
try:
r = requests.get(f”https://api.telegram.org/bot{token}/getMe”, timeout=12)
if r.status_code != 200:
return False, f”❌ Token invalide: {r.json().get(‘description’,‘Erreur’)}”
bot_name = r.json().get(“result”, {}).get(“first_name”, “Bot”)
except Exception as e:
return False, f”❌ Impossible de joindre Telegram: {e}”
h   = load_history()
msg = (
f”<b>✅ TennisIQ — Test de connexion</b>\n\n”
f”🤖 Bot: <b>{bot_name}</b>\n”
f”📅 {datetime.now().strftime(’%d/%m/%Y %H:%M’)}\n”
f”📊 Prédictions: <b>{len(h)}</b>\n”
f”🎯 Précision: <b>{calc_accuracy():.1f}%</b>\n\n”
f”<i>Connexion opérationnelle !</i>\n#TennisIQ”
)
return tg_send(msg)

def format_pred_msg(pred, bet_suggestions=None, ai_comment=None):
proba     = _safe_float(pred.get(“proba”), 0.5)
bar       = “█” * int(proba * 10) + “░” * (10 - int(proba * 10))
surf      = pred.get(“surface”, “Hard”)
s_icon    = {“Hard”: “🟦”, “Clay”: “🟧”, “Grass”: “🟩”}.get(surf, “🎾”)
ml_tag    = “🤖 “ if pred.get(“ml_used”) else “”
fav       = pred.get(“favori”, “?”)
conf      = _safe_float(pred.get(“confidence”), 50)
conf_icon = “🟢” if conf >= 70 else “🟡” if conf >= 50 else “🔴”
p1        = pred.get(“player1”, “J1”)
p2        = pred.get(“player2”, “J2”)

```
msg = (
    f"<b>{ml_tag}🎾 PRÉDICTION TENNISIQ</b>\n\n"
    f"🆚 <b>{p1} vs {p2}</b>\n"
    f"🏆 {pred.get('tournament','?')} | {s_icon} {surf}\n\n"
    f"<code>{bar}</code>\n"
    f"• {p1}: <b>{proba:.1%}</b>\n"
    f"• {p2}: <b>{1-proba:.1%}</b>\n\n"
    f"🏅 <b>FAVORI: {fav}</b>\n"
    f"{conf_icon} Confiance: <b>{conf:.0f}/100</b>"
)

o1 = pred.get("odds1"); o2 = pred.get("odds2")
if o1 and o2:
    try:
        o1f = _safe_float(str(o1).replace(",","."))
        o2f = _safe_float(str(o2).replace(",","."))
        if o1f > 0 and o2f > 0:
            msg += f"\n💰 Cotes: {p1} @ <b>{o1f:.2f}</b> | {p2} @ <b>{o2f:.2f}</b>"
    except: pass

bv = pred.get("best_value")
if bv and isinstance(bv, dict):
    msg += (
        f"\n\n🔥🔥 <b>VALUE BET — MISER SUR {bv.get('joueur','?').upper()} !</b> 🔥🔥\n"
        f"👉 Cote: <b>{_safe_float(bv.get('cote')):.2f}</b> | "
        f"Edge: <b>+{_safe_float(bv.get('edge'))*100:.1f}%</b> | "
        f"Proba: {_safe_float(bv.get('proba')):.1%}\n"
        f"<i>Avantage mathématique confirmé</i>"
    )

if bet_suggestions:
    msg += "\n\n<b>📊 Paris alternatifs:</b>"
    for b in (bet_suggestions or [])[:2]:
        try:
            msg += f"\n• {b.get('type','?')}: {_safe_float(b.get('proba')):.1%} @ {_safe_float(b.get('cote')):.2f}"
        except: pass

if ai_comment:
    clean = str(ai_comment).replace("<","&lt;").replace(">","&gt;")[:600]
    msg += f"\n\n🤖 <b>Analyse IA:</b>\n{clean}"

msg += f"\n\n#TennisIQ #{surf.replace(' ','')}"
return msg
```

def format_stats_msg():
s       = load_user_stats()
h       = load_history()
correct = s.get(“correct_predictions”, 0)
wrong   = s.get(“incorrect_predictions”, 0)
tv      = correct + wrong
acc     = (correct / tv * 100) if tv > 0 else 0
bar     = “█” * int(acc / 10) + “░” * (10 - int(acc / 10))
recent  = [p for p in h[-20:] if p.get(“statut”) in [“gagne”,“perdu”]]
r_ok    = sum(1 for p in recent if p.get(“statut”) == “gagne”)
r_acc   = (r_ok / len(recent) * 100) if recent else 0
diff    = r_acc - acc
trend   = “📈” if diff > 0 else “📉” if diff < 0 else “➡️”

```
return (
    f"<b>📊 STATISTIQUES TENNISIQ</b>\n\n"
    f"<code>{bar}</code> {acc:.1f}%\n\n"
    f"<b>Vue d'ensemble:</b>\n"
    f"• 📝 Total: <b>{s.get('total_predictions',0)}</b>\n"
    f"• ✅ Gagnés: <b>{correct}</b> ({acc:.1f}%)\n"
    f"• ❌ Perdus: <b>{wrong}</b>\n"
    f"• ⚠️ Abandons: <b>{s.get('annules_predictions',0)}</b>\n\n"
    f"<b>Forme récente (20 derniers):</b>\n"
    f"{trend} <b>{r_acc:.1f}%</b> ({diff:+.1f}% vs global)\n\n"
    f"<b>Records:</b>\n"
    f"• 🔥 Série actuelle: <b>{s.get('current_streak',0)}</b>\n"
    f"• ⚡ Meilleure série: <b>{s.get('best_streak',0)}</b>\n\n"
    f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')} #TennisIQ"
)
```

# ─────────────────────────────────────────────────────────────

# RECHERCHE WEB — SERPER

# ─────────────────────────────────────────────────────────────

def search_match_web(p1, p2, tournament):
“”“Recherche infos récentes via Serper API (Google Search).”””
key = get_serper_key()
if not key:
return None
query = f”{p1} vs {p2} {tournament} 2025 tennis”
try:
r = requests.post(
“https://google.serper.dev/search”,
headers={“X-API-KEY”: key, “Content-Type”: “application/json”},
json={“q”: query, “num”: 5, “hl”: “fr”},
timeout=20,
)
if r.status_code != 200:
return None
snippets = []
for item in r.json().get(“organic”, [])[:4]:
snippet = item.get(“snippet”, “”)
title   = item.get(“title”, “”)
if snippet:
snippets.append(f”• {title}: {snippet}”)
return “\n”.join(snippets) if snippets else None
except Exception:
return None

# ─────────────────────────────────────────────────────────────

# GROQ IA

# ─────────────────────────────────────────────────────────────

def call_groq(prompt, max_tokens=800):
key = get_groq_key()
if not key:
return None
try:
r = requests.post(
“https://api.groq.com/openai/v1/chat/completions”,
headers={“Authorization”: f”Bearer {key}”, “Content-Type”: “application/json”},
json={
“model”: “llama-3.3-70b-versatile”,
“messages”: [{“role”: “user”, “content”: prompt}],
“temperature”: 0.3,
“max_tokens”: max_tokens
},
timeout=45,
)
if r.status_code == 200:
return r.json()[“choices”][0][“message”][“content”]
return None
except Exception:
return None

def ai_analysis(p1, p2, surface, tournament, proba, best_value=None, h2h=None, web_info=None):
fav       = p1 if proba >= 0.5 else p2
und       = p2 if proba >= 0.5 else p1
fav_proba = max(proba, 1-proba)

```
h2h_str = "Pas de données H2H"
if h2h and h2h.get("total", 0) > 0:
    w1  = h2h.get('p1_wins', 0)
    w2  = h2h.get('p2_wins', 0)
    tot = h2h['total']
    leader = p1 if w1 > w2 else p2
    h2h_str = f"H2H: {w1}-{w2} en faveur de {leader} ({tot} matchs)"

vb_str = ""
if best_value and isinstance(best_value, dict):
    bv_j = best_value.get("joueur", "?")
    bv_c = _safe_float(best_value.get("cote"), 0)
    bv_e = _safe_float(best_value.get("edge"), 0)
    vb_str = f"\n⚠️ VALUE BET: Miser sur {bv_j} @ {bv_c:.2f} (edge +{bv_e*100:.1f}%)"

web_section = ""
if web_info:
    web_section = f"\n\n📰 INFOS RÉCENTES (internet):\n{web_info}"

vb_instruction = ""
if best_value and isinstance(best_value, dict):
    bv_j = best_value.get("joueur", "?")
    bv_c = _safe_float(best_value.get("cote"), 0)
    bv_e = _safe_float(best_value.get("edge"), 0)
    vb_instruction = f"🔥 VALUE BET CONFIRMÉ: Miser sur {bv_j} @ {bv_c:.2f} — edge +{bv_e*100:.1f}%. Explique pourquoi ce pari est intéressant."
else:
    vb_instruction = "Quel pari recommandes-tu ? Vainqueur, sets, ou jeux ? Sois précis."

prompt = f"""Tu es un expert analyste tennis ATP avec accès aux données statistiques.
```

MATCH:
{p1} vs {p2}
Tournoi: {tournament} | Surface: {surface}
Probabilités ML: {p1} {proba:.1%} — {p2} {1-proba:.1%}
FAVORI ML: {fav} ({fav_proba:.1%})
{h2h_str}{vb_str}{web_section}

CONSIGNE: Réponds en français, 4 sections claires:

1. 🏆 POURQUOI {fav.upper()} EST FAVORI
   (2-3 arguments: forme, surface, classement)
1. ⚠️ RISQUES
   (vulnérabilités de {fav}, atouts de {und})
1. 💰 RECOMMANDATION
   {vb_instruction}
1. 🎯 VERDICT
   Pronostic en 1 phrase + confiance (1-5 étoiles) + % bankroll à risquer

Sois concis et factuel.”””

```
return call_groq(prompt, max_tokens=800)
```

# ─────────────────────────────────────────────────────────────

# FEATURES ML (21)

# ─────────────────────────────────────────────────────────────

def extract_21_features(ps, p1, p2, surface, level=“A”, best_of=3, h2h_ratio=0.5):
s1, s2 = ps.get(p1, {}), ps.get(p2, {})
r1 = max(s1.get(“rank”, 500.0), 1.0)
r2 = max(s2.get(“rank”, 500.0), 1.0)
sp1, sp2 = s1.get(“serve_pct”, {}), s2.get(“serve_pct”, {})
sr1, sr2 = s1.get(“serve_raw”, {}), s2.get(“serve_raw”, {})
feats = [
float(np.log(r2 / r1)),
(s1.get(“rank_points”, 0) - s2.get(“rank_points”, 0)) / 5000.0,
float(s1.get(“age”, 25) - s2.get(“age”, 25)),
1.0 if surface == “Clay”  else 0.0,
1.0 if surface == “Grass” else 0.0,
1.0 if surface == “Hard”  else 0.0,
1.0 if level == “G” else 0.0,
1.0 if level == “M” else 0.0,
1.0 if best_of == 5 else 0.0,
float(s1.get(“surface_wr”, {}).get(surface, 0.5) - s2.get(“surface_wr”, {}).get(surface, 0.5)),
float(s1.get(“win_rate”, 0.5) - s2.get(“win_rate”, 0.5)),
float(s1.get(“recent_form”, 0.5) - s2.get(“recent_form”, 0.5)),
float(h2h_ratio),
(sr1.get(“ace”, 0) - sr2.get(“ace”, 0)) / 10.0,
(sr1.get(“df”,  0) - sr2.get(“df”,  0)) / 5.0,
float(sp1.get(“pct_1st_in”,   0) - sp2.get(“pct_1st_in”,   0)),
float(sp1.get(“pct_1st_won”,  0) - sp2.get(“pct_1st_won”,  0)),
float(sp1.get(“pct_2nd_won”,  0) - sp2.get(“pct_2nd_won”,  0)),
float(sp1.get(“pct_bp_saved”, 0) - sp2.get(“pct_bp_saved”, 0)),
float(s1.get(“days_since_last”, 30) - s2.get(“days_since_last”, 30)),
float(s1.get(“fatigue”, 0) - s2.get(“fatigue”, 0)),
]
return np.nan_to_num(np.array(feats, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

# ─────────────────────────────────────────────────────────────

# PRÉDICTION ML

# ─────────────────────────────────────────────────────────────

def predict_rf(p1, p2, surface, tournament=””, h2h_ratio_val=0.5):
mi = load_rf_model()
if mi is None:
return None, “rf_absent”
try:
m  = mi.get(“model”)
sc = mi.get(“scaler”)
ps = mi.get(“player_stats”, {})
if m is None or sc is None:
return None, “rf_incomplet”
if p1 not in ps or p2 not in ps:
return None, “rf_joueurs_inconnus”
lv, bo = get_level(tournament)
f = extract_21_features(ps, p1, p2, surface, lv, bo, h2h_ratio_val)
p = float(m.predict_proba(sc.transform(f.reshape(1, -1)))[0][1])
return max(0.05, min(0.95, p)), “rf_ok”
except Exception as e:
return None, f”rf_erreur:{str(e)[:40]}”

# ─────────────────────────────────────────────────────────────

# DONNÉES CSV

# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_players():
if not DATA_DIR.exists():
return []
players = set()
for f in DATA_DIR.glob(”*.csv”):
if “wta” in f.name.lower(): continue
try:
for enc in [“utf-8”,“latin-1”,“cp1252”]:
try:
df = pd.read_csv(f, encoding=enc,
usecols=[“winner_name”,“loser_name”],
on_bad_lines=“skip”)
players.update(df[“winner_name”].dropna().astype(str).str.strip())
players.update(df[“loser_name”].dropna().astype(str).str.strip())
break
except: continue
except: pass
return sorted(p for p in players if p and p.lower() != “nan” and len(p) > 1)

@st.cache_data(ttl=3600)
def load_h2h_df():
if not DATA_DIR.exists():
return pd.DataFrame()
dfs = []
for f in list(DATA_DIR.glob(”*.csv”))[:20]:
if “wta” in f.name.lower(): continue
try:
df = pd.read_csv(f, encoding=“utf-8”,
usecols=[“winner_name”,“loser_name”],
on_bad_lines=“skip”)
df[“winner_name”] = df[“winner_name”].astype(str).str.strip()
df[“loser_name”]  = df[“loser_name”].astype(str).str.strip()
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
return {“total”: len(h), “p1_wins”: len(h[h.winner_name==p1]),
“p2_wins”: len(h[h.winner_name==p2])}

def h2h_ratio(h2h, p1):
if not h2h or h2h[“total”] == 0: return 0.5
return h2h[“p1_wins”] / h2h[“total”]

def calc_proba(p1, p2, surface, tournament=””, h2h=None, mi=None):
ratio = h2h_ratio(h2h, p1)
if mi:
p, status = predict_rf(p1, p2, surface, tournament, ratio)
if p is not None:
return p, True
proba = 0.5 + (ratio - 0.5) * 0.3
return max(0.05, min(0.95, proba)), False

def calc_confidence(proba, h2h=None):
c = 50.0
if h2h and h2h.get(“total”, 0) >= 3: c += 10
c += abs(proba - 0.5) * 40
return min(100.0, c)

# ─────────────────────────────────────────────────────────────

# HISTORIQUE & STATS

# ─────────────────────────────────────────────────────────────

def load_history():
if not HIST_FILE.exists(): return []
try:
with open(HIST_FILE, “r”, encoding=“utf-8”) as f:
return json.load(f)
except: return []

def save_pred(pred):
try:
h = load_history()
pred[“id”]              = hashlib.md5(f”{datetime.now()}{pred.get(‘player1’,’’)}”.encode()).hexdigest()[:8]
pred[“statut”]         = “en_attente”
pred[“vainqueur_reel”] = None
pred[“pronostic_correct”] = None
h.append(pred)
with open(HIST_FILE, “w”, encoding=“utf-8”) as f:
json.dump(h[-1000:], f, indent=2, ensure_ascii=False)
return True
except: return False

def update_pred_result(pred_id, statut, vainqueur_reel=None):
try:
h = load_history()
for p in h:
if p.get(“id”) == pred_id:
p[“statut”]         = statut
p[“date_maj”]       = datetime.now().isoformat()
p[“vainqueur_reel”] = vainqueur_reel
p[“pronostic_correct”] = (
(vainqueur_reel == p.get(“favori”)) if vainqueur_reel else None
)
break
with open(HIST_FILE, “w”, encoding=“utf-8”) as f:
json.dump(h, f, indent=2, ensure_ascii=False)
update_stats()
return True
except: return False

def load_user_stats():
default = {
“total_predictions”: 0, “correct_predictions”: 0,
“incorrect_predictions”: 0, “annules_predictions”: 0,
“current_streak”: 0, “best_streak”: 0
}
if not USER_STATS_FILE.exists(): return default
try:
with open(USER_STATS_FILE) as f: return json.load(f)
except: return default

def update_stats():
h = load_history()
correct  = sum(1 for p in h if p.get(“statut”) == “gagne”)
wrong    = sum(1 for p in h if p.get(“statut”) == “perdu”)
cancel   = sum(1 for p in h if p.get(“statut”) == “annule”)
streak = cur = best = 0
for p in reversed(h):
if p.get(“statut”) == “gagne”:
streak += 1; cur = streak; best = max(best, streak)
elif p.get(“statut”) == “perdu”:
streak = 0; cur = 0
stats = {
“total_predictions”: len(h),
“correct_predictions”: correct,
“incorrect_predictions”: wrong,
“annules_predictions”: cancel,
“current_streak”: cur,
“best_streak”: best
}
with open(USER_STATS_FILE, “w”) as f:
json.dump(stats, f)
return stats

def calc_accuracy():
s = load_user_stats()
tv = s.get(“correct_predictions”, 0) + s.get(“incorrect_predictions”, 0)
return (s.get(“correct_predictions”, 0) / tv * 100) if tv > 0 else 0

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
with open(ACHIEVEMENTS_FILE, “w”) as f: json.dump(a, f)
except: pass

def check_achievements():
s = load_user_stats(); h = load_history(); a = load_ach(); new = []
checks = [
(“first_win”, s.get(“correct_predictions”, 0) >= 1),
(“streak_5”,  s.get(“best_streak”, 0) >= 5),
(“streak_10”, s.get(“best_streak”, 0) >= 10),
(“pred_100”,  s.get(“total_predictions”, 0) >= 100),
]
for aid, cond in checks:
if cond and aid not in a:
a[aid] = {“unlocked_at”: datetime.now().isoformat()}
new.append(ACHIEVEMENTS[aid])
value_wins = sum(1 for p in h if p.get(“best_value”) and p.get(“statut”) == “gagne”)
if value_wins >= 10 and “value_master” not in a:
a[“value_master”] = {“unlocked_at”: datetime.now().isoformat()}
new.append(ACHIEVEMENTS[“value_master”])
surfs = {p.get(“surface”) for p in h if p.get(“statut”) == “gagne”}
if len(surfs) >= 3 and “surface_specialist” not in a:
a[“surface_specialist”] = {“unlocked_at”: datetime.now().isoformat()}
new.append(ACHIEVEMENTS[“surface_specialist”])
if new: save_ach(a)
return new

# ─────────────────────────────────────────────────────────────

# BACKUP

# ─────────────────────────────────────────────────────────────

def backup():
ts = datetime.now().strftime(”%Y%m%d_%H%M%S”)
for f in [HIST_FILE, USER_STATS_FILE]:
if f.exists():
try: shutil.copy(f, BACKUP_DIR / f”{f.stem}_{ts}{f.suffix}”)
except: pass

# ─────────────────────────────────────────────────────────────

# PARIS ALTERNATIFS

# ─────────────────────────────────────────────────────────────

def alt_bets(p1, p2, surface, proba):
bets = []
if proba > 0.6 or proba < 0.4:
bets.append({“type”:“📊 Under 22.5 games”,“description”:“Moins de 22.5 jeux”,
“proba”:0.64,“cote”:1.78,“confidence”:68})
else:
bets.append({“type”:“📊 Over 22.5 games”,“description”:“Plus de 22.5 jeux”,
“proba”:0.61,“cote”:1.82,“confidence”:63})
if proba > 0.65:
bets.append({“type”:f”⚖️ {p1} -3.5 jeux”,“description”:“Gagne avec écart”,
“proba”:0.57,“cote”:2.15,“confidence”:58})
elif proba < 0.35:
bets.append({“type”:f”⚖️ {p2} +3.5 jeux”,“description”:“Perd par moins de 4”,
“proba”:0.60,“cote”:1.98,“confidence”:62})
if 0.3 < proba < 0.7:
bets.append({“type”:“🔄 Chaque joueur gagne un set”,“description”:“Match en 3 sets min”,
“proba”:0.54,“cote”:2.25,“confidence”:54})
return bets

# ─────────────────────────────────────────────────────────────

# MATCHS DEMO

# ─────────────────────────────────────────────────────────────

def mock_matches():
return [
{“p1”:“Novak Djokovic”,“p2”:“Carlos Alcaraz”,“surface”:“Clay”,“tournament”:“Roland Garros”},
{“p1”:“Jannik Sinner”,“p2”:“Daniil Medvedev”,“surface”:“Hard”,“tournament”:“Miami Open”},
{“p1”:“Alexander Zverev”,“p2”:“Stefanos Tsitsipas”,“surface”:“Clay”,“tournament”:“Madrid Open”},
{“p1”:“Holger Rune”,“p2”:“Casper Ruud”,“surface”:“Grass”,“tournament”:“Wimbledon”},
]

# ─────────────────────────────────────────────────────────────

# COMPOSANTS SÉLECTEURS

# ─────────────────────────────────────────────────────────────

def player_sel(label, all_players, key, default=None):
search   = st.text_input(f”🔍 {label}”, key=f”srch_{key}”, placeholder=“Tapez un nom…”)
filtered = ([p for p in all_players if search.lower() in p.lower()]
if search else all_players[:200])
if not filtered and search:
filtered = [p for p in all_players if p[0].lower() == search[0].lower()][:50] if search else []
st.caption(f”{len(filtered)} sur {len(all_players):,} joueurs”)
if not filtered:
return st.text_input(label, key=key)
idx = 0
if default:
for i, p in enumerate(filtered):
if default.lower() in p.lower(): idx = i; break
return st.selectbox(label, filtered, index=idx, key=key)

def tourn_sel(label, key, default=None):
search = st.text_input(f”🔍 {label}”, key=f”srcht_{key}”, placeholder=“ex: Roland Garros…”)
all_t  = sorted(TOURNAMENTS_DB.keys())
if search:
sl  = search.lower().strip()
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

# PAGE : DASHBOARD

# ═══════════════════════════════════════════════════════════════

def show_dashboard():
st.markdown(section_title(“🏠 Dashboard”, “Vue d’ensemble de vos performances”),
unsafe_allow_html=True)

```
mi       = load_rf_model()
metadata = load_model_metadata()
stats    = load_user_stats()
h        = load_history()
a        = load_ach()

token, _ = get_tg_config()
api_status_bar(mi, bool(get_groq_key()), bool(get_serper_key()), bool(token))

correct  = stats.get("correct_predictions", 0)
wrong    = stats.get("incorrect_predictions", 0)
cancel   = stats.get("annules_predictions", 0)
pending  = len([p for p in h if p.get("statut") == "en_attente"])
tv       = correct + wrong
acc      = (correct / tv * 100) if tv > 0 else 0
recent   = [p for p in h[-20:] if p.get("statut") in ["gagne","perdu"]]
r_acc    = (sum(1 for p in recent if p.get("statut") == "gagne") / len(recent) * 100) if recent else 0

# KPIs
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.markdown(big_metric("PRÉCISION",   f"{acc:.1f}%",  r_acc-acc if tv>0 else None, "🎯","#00DFA2"), unsafe_allow_html=True)
with c2: st.markdown(big_metric("GAGNÉS",      str(correct),   None,"✅","#00DFA2"),  unsafe_allow_html=True)
with c3: st.markdown(big_metric("PERDUS",      str(wrong),     None,"❌","#FF4757"),  unsafe_allow_html=True)
with c4: st.markdown(big_metric("ABANDONS",    str(cancel),    None,"⚠️","#FFB200"),  unsafe_allow_html=True)
with c5: st.markdown(big_metric("EN ATTENTE",  str(pending),   None,"⏳","#7A8599"),  unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_l, col_r = st.columns([1, 2])

with col_l:
    streak = stats.get("current_streak", 0)
    best   = stats.get("best_streak", 0)
    fire   = "🔥" if streak >= 5 else "⚡" if streak >= 3 else ""
    sc     = "#00DFA2" if streak > 0 else "#7A8599"
    sb     = "#00DFA244" if streak > 0 else COLORS["card_border"]
    st.markdown(
        f'<div style="background:{COLORS["card_bg"]};border:1px solid {sb};'
        f'border-radius:16px;padding:1.5rem;text-align:center;">'
        f'<div style="font-size:2rem;">{fire or "🎾"}</div>'
        f'<div style="font-family:Syne,sans-serif;font-size:2.5rem;font-weight:800;color:{sc};">{streak}</div>'
        f'<div style="color:{COLORS["gray"]};font-size:0.82rem;text-transform:uppercase;letter-spacing:0.1em;">Série actuelle</div>'
        f'<div style="margin-top:0.75rem;padding-top:0.75rem;border-top:1px solid {COLORS["card_border"]};">'
        f'<span style="color:{COLORS["gray"]};font-size:0.8rem;">Record: </span>'
        f'<span style="color:#FFB200;font-weight:700;">⚡ {best}</span>'
        f'</div></div>',
        unsafe_allow_html=True
    )

with col_r:
    services = []
    if mi:
        ps        = mi.get("player_stats", {})
        acc_model = _safe_float(mi.get("accuracy", metadata.get("accuracy", 0)))
        services.append(("🤖 Modèle ML", f"{acc_model:.1%} acc · {len(ps):,} joueurs", True))
    else:
        services.append(("🤖 Modèle ML", "Non chargé — mode CSV", False))
    services.append(("🧠 IA Groq",       "Connectée"          if get_groq_key()    else "GROQ_API_KEY manquante",    bool(get_groq_key())))
    services.append(("🔍 Recherche web", "Activée (Serper)"   if get_serper_key()  else "SERPER_API_KEY manquante",  bool(get_serper_key())))
    services.append(("📱 Telegram",      "Configuré"          if token             else "Clés manquantes",           bool(token)))

    rows_html = ""
    for svc, desc, ok in services:
        color = "#00DFA2" if ok else "#FF4757"
        dot   = "●" if ok else "○"
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem;'
            f'background:rgba(255,255,255,0.03);border-radius:8px;margin-bottom:0.35rem;">'
            f'<span style="color:{color};font-size:0.8rem;">{dot}</span>'
            f'<span style="font-weight:600;color:#E8EDF5;flex:1;font-size:0.88rem;">{svc}</span>'
            f'<span style="color:{COLORS["gray"]};font-size:0.78rem;">{desc}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div style="background:{COLORS["card_bg"]};border:1px solid {COLORS["card_border"]};'
        f'border-radius:16px;padding:1.25rem;">'
        f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;'
        f'color:#E8EDF5;margin-bottom:0.85rem;">⚙️ STATUT DES SERVICES</div>'
        f'{rows_html}</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Graphique évolution
finished = [p for p in h if p.get("statut") in ["gagne","perdu"]]
if len(finished) >= 3:
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;'
        'color:#E8EDF5;margin-bottom:0.6rem;">📈 Évolution de la précision</div>',
        unsafe_allow_html=True
    )
    df_h = pd.DataFrame(finished)
    df_h["ok"]     = (df_h["statut"] == "gagne").astype(int)
    df_h["cum_ok"] = df_h["ok"].expanding().sum()
    df_h["cum_n"]  = range(1, len(df_h)+1)
    df_h["acc"]    = df_h["cum_ok"] / df_h["cum_n"] * 100
    df_h["n"]      = range(1, len(df_h)+1)

    fig = go.Figure()
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.15)", annotation_text="50%")
    fig.add_trace(go.Scatter(
        x=df_h["n"], y=df_h["acc"], mode="lines",
        line=dict(color="#00DFA2", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,223,162,0.07)"
    ))
    fig.update_layout(
        height=240, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7A8599", family="DM Sans"),
        xaxis=dict(showgrid=False, title="Prédiction #", color="#7A8599"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   title="Précision (%)", color="#7A8599", range=[0,100]),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# Surface + badges
col_surf, col_badges = st.columns([3, 2])

with col_surf:
    surf_data = []
    for surf in SURFACES:
        sp = [p for p in h if p.get("surface") == surf and p.get("statut") in ["gagne","perdu"]]
        if sp:
            ok = sum(1 for p in sp if p.get("statut") == "gagne")
            surf_data.append({"Surface": surf, "Précision": ok/len(sp)*100, "Total": len(sp)})
    if surf_data:
        st.markdown(
            '<div style="font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;'
            'color:#E8EDF5;margin-bottom:0.6rem;">🎾 Par surface</div>',
            unsafe_allow_html=True
        )
        df_s  = pd.DataFrame(surf_data)
        fig2  = go.Figure(go.Bar(
            x=df_s["Surface"], y=df_s["Précision"],
            text=df_s["Précision"].round(0).astype(int).astype(str) + "%",
            textposition="outside",
            marker_color=[SURFACE_CFG[s]["color"] for s in df_s["Surface"]],
            hovertemplate="<b>%{x}</b><br>Précision: %{y:.1f}%<br>Matchs: %{customdata}<extra></extra>",
            customdata=df_s["Total"]
        ))
        fig2.update_layout(
            height=200, margin=dict(l=0,r=0,t=25,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#7A8599", family="DM Sans"),
            xaxis=dict(showgrid=False, color="#7A8599"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                       color="#7A8599", range=[0,110]),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

with col_badges:
    st.markdown(
        f'<div style="font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;'
        f'color:#E8EDF5;margin-bottom:0.6rem;">🏆 Badges ({len(a)}/{len(ACHIEVEMENTS)})</div>',
        unsafe_allow_html=True
    )
    if a:
        for aid, adata_val in list(a.items())[:4]:
            ach_meta = ACHIEVEMENTS.get(aid, {})
            try: d = datetime.fromisoformat(adata_val["unlocked_at"]).strftime("%d/%m/%Y")
            except: d = "?"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.6rem;'
                f'background:rgba(0,223,162,0.06);border:1px solid rgba(0,223,162,0.2);'
                f'border-radius:10px;padding:0.55rem 0.8rem;margin-bottom:0.45rem;">'
                f'<span style="font-size:1.4rem;">{ach_meta.get("icon","🏆")}</span>'
                f'<div><div style="font-weight:700;color:#00DFA2;font-size:0.82rem;">{ach_meta.get("name","")}</div>'
                f'<div style="color:{COLORS["gray"]};font-size:0.7rem;">Débloqué le {d}</div></div>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            f'<div style="text-align:center;padding:1.5rem;color:{COLORS["gray"]};'
            f'border:1px dashed {COLORS["card_border"]};border-radius:12px;font-size:0.85rem;">'
            f'Aucun badge encore<br><small>Faites des prédictions !</small></div>',
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)
if st.button("📱 Envoyer les stats sur Telegram", use_container_width=True):
    ok, msg = tg_send(format_stats_msg())
    st.success(msg) if ok else st.error(msg)
```

# ═══════════════════════════════════════════════════════════════

# PAGE : ANALYSE MULTI-MATCHS  ← CORRECTION PRINCIPALE MOBILE

# ═══════════════════════════════════════════════════════════════

def show_prediction():
st.markdown(section_title(“🎯 Analyse Multi-matchs”, “Prédictions ML + IA + Recherche web”),
unsafe_allow_html=True)

```
mi       = load_rf_model()
metadata = load_model_metadata()
token, _ = get_tg_config()
groq_ok  = bool(get_groq_key())
serp_ok  = bool(get_serper_key())
tg_ok    = bool(token)

# ── Statut APIs — toujours visible ──────────────────────────
api_status_bar(mi, groq_ok, serp_ok, tg_ok)

if mi:
    ps        = mi.get("player_stats", {})
    acc_model = _safe_float(mi.get("accuracy", metadata.get("accuracy", 0)))
    n_matches = metadata.get("n_matches", mi.get("n_matches", 0))
    st.markdown(
        f'<div style="background:rgba(0,223,162,0.07);border:1px solid rgba(0,223,162,0.2);'
        f'border-radius:10px;padding:0.6rem 1rem;margin-bottom:1rem;">'
        f'🤖 <b style="color:#00DFA2;">Modèle ML actif ({acc_model:.1%} accuracy)</b>'
        f'<span style="color:{COLORS["gray"]};font-size:0.82rem;margin-left:0.6rem;">'
        f'{len(ps):,} joueurs · {n_matches:,} matchs</span>'
        f'</div>',
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ Modèle ML non chargé — prédictions en mode statistiques CSV uniquement")

if not groq_ok:
    st.warning("⚠️ GROQ_API_KEY absente des secrets → Analyse IA désactivée")
if not serp_ok:
    st.info("💡 SERPER_API_KEY absente → Recherche web désactivée (non bloquant)")

with st.spinner("Chargement des joueurs..."):
    all_p = load_players()

# ── Options ──────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1: n_matches_input = st.number_input("Nombre de matchs", 1, MAX_MATCHES, 1)
with c2: use_ai  = st.checkbox("🤖 Analyse IA",    value=groq_ok, disabled=not groq_ok)
with c3: send_tg = st.checkbox("📱 Envoi Telegram", value=False,    disabled=not tg_ok)

# ── Saisie des matchs ────────────────────────────────────────
st.markdown(
    '<div style="font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;'
    'color:#E8EDF5;margin:1.25rem 0 0.6rem;">📝 Saisie des matchs</div>',
    unsafe_allow_html=True
)

inputs = []
for i in range(n_matches_input):
    with st.expander(f"⚽ Match {i+1}", expanded=(i == 0)):
        ct, cs = st.columns([3, 1])
        with ct:
            tourn = tourn_sel("Tournoi", f"t{i}", "Roland Garros")
        with cs:
            surf   = get_surface(tourn)
            lv, bo = get_level(tourn)
            cfg    = SURFACE_CFG[surf]
            bo_tag = '<div style="font-size:0.7rem;color:#7A8599;">Best of 5</div>' if bo == 5 else ""
            st.markdown(
                f'<div style="background:{cfg["bg"]};border:1px solid {cfg["color"]}55;'
                f'border-radius:10px;padding:0.5rem;text-align:center;margin-top:1.6rem;">'
                f'<div style="font-size:1.2rem;">{cfg["icon"]}</div>'
                f'<div style="font-weight:700;color:{cfg["color"]};font-size:0.85rem;">{surf}</div>'
                f'{bo_tag}</div>',
                unsafe_allow_html=True
            )

        cp1, cp2 = st.columns(2)
        with cp1:
            p1 = player_sel("Joueur 1", all_p, f"p1_{i}")
            o1 = st.text_input(f"Cote {p1[:12] if p1 else 'J1'}", key=f"o1_{i}", placeholder="1.75")
        with cp2:
            p2_list = [p for p in all_p if p != p1]
            p2 = player_sel("Joueur 2", p2_list, f"p2_{i}")
            o2 = st.text_input(f"Cote {p2[:12] if p2 else 'J2'}", key=f"o2_{i}", placeholder="2.10")

        if mi and p1 and p2:
            ps_d = mi.get("player_stats", {})
            p1k  = "✅" if p1 in ps_d else "⚠️ inconnu"
            p2k  = "✅" if p2 in ps_d else "⚠️ inconnu"
            st.caption(f"ML: {p1[:18]} {p1k}  ·  {p2[:18]} {p2k}")

        inputs.append({"p1": p1, "p2": p2, "surf": surf, "tourn": tourn, "o1": o1, "o2": o2})

# ── Bouton principal ─────────────────────────────────────────
if not st.button("🔍 Analyser tous les matchs", type="primary", use_container_width=True):
    return

valid = [m for m in inputs if m["p1"] and m["p2"]]
if not valid:
    st.warning("⚠️ Remplis au moins un match complet (joueur 1 et joueur 2)")
    return

st.markdown("---")
st.markdown(section_title("📊 Résultats"), unsafe_allow_html=True)

for i, m in enumerate(valid):
    p1, p2, surf, tourn = m["p1"], m["p2"], m["surf"], m["tourn"]

    # ── Calcul de la prédiction ───────────────────────────────
    # Clé unique dans session_state pour STOCKER les résultats
    result_key  = f"pred_result_{p1}_{p2}_{tourn}"
    ai_key      = f"ai_result_{p1}_{p2}_{tourn}"
    web_key     = f"web_result_{p1}_{p2}_{tourn}"

    # Calcul ML (rapide, on le refait à chaque analyse)
    h2h_data = get_h2h(p1, p2)
    proba, ml_used = calc_proba(p1, p2, surf, tourn, h2h_data, mi)
    conf   = calc_confidence(proba, h2h_data)
    fav    = p1 if proba >= 0.5 else p2
    und    = p2 if proba >= 0.5 else p1

    # Stocker le résultat de base dans session_state
    st.session_state[result_key] = {
        "proba": proba, "ml_used": ml_used, "conf": conf,
        "fav": fav, "h2h": h2h_data
    }

    cfg          = SURFACE_CFG[surf]
    lv, bo       = get_level(tourn)
    fav_proba    = max(proba, 1-proba)
    ml_label     = "🤖 ML · 21 features" if ml_used else "📊 Fallback CSV"
    ml_color     = "#00DFA2" if ml_used else "#7A8599"
    p1_proba     = f"{proba:.1%}"
    p2_proba     = f"{1-proba:.1%}"
    p1_color     = "#00DFA2" if fav == p1 else "#7A8599"
    p2_color     = "#00DFA2" if fav == p2 else "#7A8599"
    p1_fav_html  = '<div style="color:#00DFA2;font-size:0.72rem;font-weight:700;">⭐ FAVORI</div>' if fav == p1 else ""
    p2_fav_html  = '<div style="color:#00DFA2;font-size:0.72rem;font-weight:700;">⭐ FAVORI</div>' if fav == p2 else ""
    h2h_str      = (f"H2H {h2h_data['p1_wins']}-{h2h_data['p2_wins']} ({h2h_data['total']} matchs)"
                    if h2h_data else "H2H: aucun")
    conf_icon    = "🟢" if conf >= 70 else "🟡" if conf >= 50 else "🔴"

    # ── Carte principale ─────────────────────────────────────
    st.markdown(
        f'<div style="background:{COLORS["card_bg"]};border:1px solid {COLORS["card_border"]};'
        f'border-radius:16px;padding:1.25rem;margin-bottom:0.5rem;">'
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;margin-bottom:0.85rem;">'
        f'<div style="display:flex;align-items:center;gap:0.6rem;flex-wrap:wrap;">'
        f'<span style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#E8EDF5;">Match {i+1}</span>'
        f'<span>{surface_badge(surf)}</span>'
        f'<span style="color:{COLORS["gray"]};font-size:0.82rem;">🏆 {tourn}</span>'
        f'</div>'
        f'<span style="color:{ml_color};font-size:0.78rem;font-weight:600;">{ml_label}</span>'
        f'</div>'
        # Joueurs
        f'<div style="display:grid;grid-template-columns:1fr auto 1fr;gap:0.75rem;align-items:center;margin-bottom:0.85rem;">'
        f'<div style="text-align:center;">'
        f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#E8EDF5;">{p1}</div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:{p1_color};">{p1_proba}</div>'
        f'{p1_fav_html}</div>'
        f'<div style="text-align:center;color:{COLORS["gray"]};font-weight:700;font-size:1.1rem;">VS</div>'
        f'<div style="text-align:center;">'
        f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#E8EDF5;">{p2}</div>'
        f'<div style="font-size:1.6rem;font-weight:800;color:{p2_color};">{p2_proba}</div>'
        f'{p2_fav_html}</div></div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.progress(float(proba))

    st.markdown(
        f'<div style="display:flex;gap:0.4rem;flex-wrap:wrap;margin:0.5rem 0;">'
        f'{stat_pill("Confiance", f"{conf:.0f}/100", "#00DFA2", conf_icon)}'
        f'{stat_pill("H2H", h2h_str, "#0079FF", "📊")}'
        f'{stat_pill("Format", f"Best of {bo}", "#7A8599", "📋")}'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Value Bet ─────────────────────────────────────────────
    best_val = None
    if m["o1"] and m["o2"]:
        try:
            o1f = _safe_float(str(m["o1"]).replace(",","."))
            o2f = _safe_float(str(m["o2"]).replace(",","."))
            if o1f > 1.0 and o2f > 1.0:
                e1 = proba - 1/o1f
                e2 = (1-proba) - 1/o2f
                if e1 > MIN_EDGE_COMBINE:
                    best_val = {"joueur": p1, "edge": e1, "cote": o1f, "proba": proba}
                elif e2 > MIN_EDGE_COMBINE:
                    best_val = {"joueur": p2, "edge": e2, "cote": o2f, "proba": 1-proba}
                if best_val:
                    st.markdown(
                        value_bet_box_html(best_val["joueur"], best_val["cote"],
                                           best_val["edge"], best_val["proba"]),
                        unsafe_allow_html=True
                    )
        except: pass

    # ── Paris alternatifs ─────────────────────────────────────
    bets = alt_bets(p1, p2, surf, proba)
    with st.expander("📊 Paris alternatifs"):
        for b in bets:
            ci2 = "🟢" if b["confidence"] >= 65 else "🟡"
            st.markdown(
                f"{ci2} **{b['type']}** — {b['description']} "
                f"· Proba {b['proba']:.1%} · Cote {b['cote']:.2f} · Confiance {b['confidence']}%"
            )

    # ═══════════════════════════════════════════════════════
    # ANALYSE IA — STOCKÉE DANS SESSION_STATE (fix mobile)
    # ═══════════════════════════════════════════════════════
    if use_ai and groq_ok:
        # Récupérer le résultat déjà calculé si disponible
        ai_txt  = st.session_state.get(ai_key)
        web_txt = st.session_state.get(web_key)

        if ai_txt is None:
            # Bouton séparé pour éviter de refaire le calcul à chaque rerun
            btn_label = f"🤖 Lancer l'analyse IA — Match {i+1}"
            if st.button(btn_label, key=f"btn_ai_{i}", use_container_width=True):
                with st.spinner(f"🤖 Analyse IA en cours pour {p1} vs {p2}..."):
                    # Recherche web
                    if serp_ok:
                        web_info = search_match_web(p1, p2, tourn)
                        if web_info:
                            st.session_state[web_key] = web_info
                            web_txt = web_info
                    # Appel IA
                    result = ai_analysis(
                        p1, p2, surf, tourn, proba,
                        best_val, h2h_data,
                        web_txt
                    )
                    if result:
                        st.session_state[ai_key] = result
                        ai_txt = result
                    else:
                        st.error("❌ L'analyse IA a échoué — vérifiez GROQ_API_KEY dans les secrets")
        else:
            st.caption("✅ Analyse IA déjà calculée (récupérée depuis la session)")

        # Affichage DIRECT (pas dans un expander fermé = visible sur mobile)
        if ai_txt:
            st.markdown(
                f'<div style="background:rgba(0,121,255,0.06);border:1px solid rgba(0,121,255,0.2);'
                f'border-radius:14px;padding:1.25rem 1.5rem;margin:0.75rem 0;">'
                f'<div style="font-family:Syne,sans-serif;font-weight:700;color:#0079FF;margin-bottom:0.75rem;">'
                f'🤖 Analyse IA</div>'
                f'<div style="color:#E8EDF5;line-height:1.7;white-space:pre-wrap;">{ai_txt}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            if web_txt:
                st.caption("🔍 Analyse enrichie avec des données web récentes")

    elif use_ai and not groq_ok:
        st.warning("⚠️ GROQ_API_KEY absente des secrets Streamlit — IA non disponible")

    # ── Sauvegarde + Telegram ─────────────────────────────────
    pred_data = {
        "player1": p1, "player2": p2, "tournament": tourn, "surface": surf,
        "proba": float(proba), "confidence": float(conf),
        "odds1": m["o1"], "odds2": m["o2"], "favori": fav,
        "best_value": best_val, "ml_used": bool(ml_used),
        "date": datetime.now().isoformat()
    }

    cb1, cb2 = st.columns(2)
    with cb1:
        if st.button(f"💾 Sauvegarder match {i+1}", key=f"save_{i}", use_container_width=True):
            if save_pred(pred_data):
                st.success("✅ Prédiction sauvegardée !")
            else:
                st.error("❌ Erreur de sauvegarde")

    with cb2:
        if st.button(f"📱 Envoyer sur Telegram", key=f"tg_{i}", use_container_width=True):
            if not tg_ok:
                st.error("❌ Telegram non configuré dans les secrets")
            else:
                ai_content = st.session_state.get(ai_key)
                msg = format_pred_msg(pred_data, bets, ai_content)
                ok2, resp = tg_send(msg)
                st.success(resp) if ok2 else st.error(resp)

    if send_tg and i == 0:
        save_pred(pred_data)
        ai_content = st.session_state.get(ai_key)
        ok2, resp = tg_send(format_pred_msg(pred_data, bets, ai_content))
        if not ok2: st.error(resp)

    st.markdown("---")

nb = check_achievements()
if nb:
    st.balloons()
    st.success(f"🏆 {len(nb)} nouveau(x) badge(s) débloqué(s) !")
```

# ═══════════════════════════════════════════════════════════════

# PAGE : EN ATTENTE

# ═══════════════════════════════════════════════════════════════

def show_pending():
st.markdown(section_title(“⏳ Prédictions en attente”,
“Validez les résultats pour mettre à jour les statistiques”),
unsafe_allow_html=True)

```
h       = load_history()
pending = [p for p in h if p.get("statut") == "en_attente"]

if not pending:
    st.markdown(
        f'<div style="text-align:center;padding:2.5rem;background:{COLORS["card_bg"]};'
        f'border:1px dashed {COLORS["card_border"]};border-radius:16px;">'
        f'<div style="font-size:2.5rem;">🎉</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:#E8EDF5;margin-top:0.4rem;">Aucune prédiction en attente !</div>'
        f'<div style="color:{COLORS["gray"]};margin-top:0.2rem;font-size:0.88rem;">Toutes vos prédictions ont un résultat.</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    return

st.markdown(
    f'<div style="margin-bottom:1.25rem;">'
    f'<span style="background:rgba(255,178,0,0.15);border:1px solid rgba(255,178,0,0.35);'
    f'color:#FFB200;border-radius:100px;padding:0.3rem 0.85rem;font-weight:700;font-size:0.88rem;">'
    f'⏳ {len(pending)} prédiction{"s" if len(pending)>1 else ""} en attente</span>'
    f'</div>',
    unsafe_allow_html=True
)

for pred in reversed(pending):
    pid      = pred.get("id", "?")
    p1       = pred.get("player1", "?")
    p2       = pred.get("player2", "?")
    fav      = pred.get("favori", "?")
    surf     = pred.get("surface", "Hard")
    tourn    = pred.get("tournament", "?")
    proba    = _safe_float(pred.get("proba"), 0.5)
    conf     = _safe_float(pred.get("confidence"), 50)
    date_str = pred.get("date", "")[:16].replace("T", " ")

    fav_proba_display = f"{(proba if fav == p1 else 1 - proba):.1%}"
    conf_icon = "🟢" if conf >= 70 else "🟡" if conf >= 50 else "🔴"

    st.markdown(
        f'<div style="background:{COLORS["card_bg"]};border:1px solid {COLORS["card_border"]};'
        f'border-radius:16px;padding:1.25rem;margin-bottom:1rem;">'
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;margin-bottom:0.75rem;">'
        f'<div style="display:flex;align-items:center;gap:0.6rem;flex-wrap:wrap;">'
        f'<span style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#E8EDF5;">'
        f'{p1} <span style="color:{COLORS["gray"]};font-weight:400;">vs</span> {p2}</span>'
        f'<span>{surface_badge(surf)}</span>'
        f'</div>'
        f'<span style="color:{COLORS["gray"]};font-size:0.76rem;">📅 {date_str}</span>'
        f'</div>'
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:1rem;margin-bottom:0.75rem;">'
        f'<span style="color:{COLORS["gray"]};font-size:0.82rem;">🏆 {tourn}</span>'
        f'<span style="color:#E8EDF5;font-size:0.82rem;">'
        f'Favori: <strong style="color:#00DFA2;">{fav}</strong> ({fav_proba_display})</span>'
        f'<span style="color:{COLORS["gray"]};font-size:0.78rem;">{conf_icon} {conf:.0f}/100</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    bv = pred.get("best_value")
    if bv and isinstance(bv, dict):
        bv_j    = bv.get("joueur", "?")
        bv_cote = _safe_float(bv.get("cote"), 0)
        bv_edge = _safe_float(bv.get("edge"), 0)
        st.markdown(
            f'<div style="background:rgba(0,223,162,0.07);border:1px solid rgba(0,223,162,0.2);'
            f'border-radius:8px;padding:0.45rem 0.7rem;margin-bottom:0.7rem;font-size:0.78rem;">'
            f'💎 Value bet: <strong style="color:#00DFA2;">{bv_j} @ {bv_cote:.2f}</strong>'
            f' · Edge <strong>+{bv_edge*100:.1f}%</strong></div>',
            unsafe_allow_html=True
        )

    st.markdown(
        f'<div style="font-weight:600;color:#E8EDF5;font-size:0.88rem;margin-bottom:0.6rem;">'
        f'Qui a gagné ce match ?</div></div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        if st.button(f"✅ {p1[:20]} gagné", key=f"w1_{pid}", use_container_width=True,
                     type="primary" if fav == p1 else "secondary"):
            statut = "gagne" if fav == p1 else "perdu"
            update_pred_result(pid, statut, vainqueur_reel=p1)
            check_achievements()
            st.rerun()
    with c2:
        if st.button(f"✅ {p2[:20]} gagné", key=f"w2_{pid}", use_container_width=True,
                     type="primary" if fav == p2 else "secondary"):
            statut = "gagne" if fav == p2 else "perdu"
            update_pred_result(pid, statut, vainqueur_reel=p2)
            check_achievements()
            st.rerun()
    with c3:
        if st.button("⚠️ Abandon", key=f"ab_{pid}", use_container_width=True):
            update_pred_result(pid, "annule", vainqueur_reel=None)
            st.rerun()

    opp = p2 if fav == p1 else p1
    st.caption(f"💡 Pronostic: {fav} → si {fav} gagne = ✅ GAGNÉ | si {opp} gagne = ❌ PERDU")
    st.markdown("<br>", unsafe_allow_html=True)
```

# ═══════════════════════════════════════════════════════════════

# PAGE : STATISTIQUES

# ═══════════════════════════════════════════════════════════════

def show_statistics():
st.markdown(section_title(“📊 Statistiques”, “Analyse complète de vos performances”),
unsafe_allow_html=True)

```
h = load_history()
if not h:
    st.info("Aucune prédiction enregistrée pour le moment.")
    return

df = pd.DataFrame(h)
df["date"]              = pd.to_datetime(df["date"], errors="coerce")
df["pronostic_correct"] = df["pronostic_correct"].fillna(False)

fini     = df[df["statut"].isin(["gagne","perdu","annule"])]
gagnes   = df[df["statut"] == "gagne"]
perdus   = df[df["statut"] == "perdu"]
abandons = df[df["statut"] == "annule"]
tv  = len(gagnes) + len(perdus)
acc = (len(gagnes) / tv * 100) if tv > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.markdown(big_metric("TOTAL",      str(len(df)),        None,"📝","#0079FF"), unsafe_allow_html=True)
with c2: st.markdown(big_metric("GAGNÉS",     str(len(gagnes)),    None,"✅","#00DFA2"), unsafe_allow_html=True)
with c3: st.markdown(big_metric("PERDUS",     str(len(perdus)),    None,"❌","#FF4757"), unsafe_allow_html=True)
with c4: st.markdown(big_metric("ABANDONS",   str(len(abandons)),  None,"⚠️","#FFB200"), unsafe_allow_html=True)
with c5: st.markdown(big_metric("PRÉCISION",  f"{acc:.1f}%",       None,"🎯","#00DFA2"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col_pie, col_table = st.columns([1, 2])

with col_pie:
    if tv > 0:
        fig_d = go.Figure(go.Pie(
            labels=["Gagnés ✅", "Perdus ❌", "Abandons ⚠️"],
            values=[len(gagnes), len(perdus), len(abandons)],
            hole=0.65,
            marker_colors=["#00DFA2","#FF4757","#FFB200"],
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value} (%{percent})<extra></extra>"
        ))
        fig_d.update_layout(
            height=220, margin=dict(l=0,r=0,t=5,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#7A8599"),
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5,
                        font=dict(size=10, color="#E8EDF5")),
            annotations=[dict(text=f"<b>{acc:.0f}%</b>", x=0.5, y=0.5,
                              font=dict(size=20, color="#00DFA2", family="Syne"),
                              showarrow=False)]
        )
        st.plotly_chart(fig_d, use_container_width=True)

with col_table:
    if not fini.empty:
        st.markdown(
            f'<div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;'
            f'color:#E8EDF5;margin-bottom:0.6rem;">📋 Résultats récents</div>',
            unsafe_allow_html=True
        )
        recent_fini = fini.sort_values("date", ascending=False).head(10)
        for _, row in recent_fini.iterrows():
            s    = row.get("statut", "?")
            pc   = row.get("pronostic_correct")
            fav_ = row.get("favori", "?")
            vr   = row.get("vainqueur_reel", "?")
            date_ = str(row.get("date", ""))[:10]
            surf_ = row.get("surface", "Hard")

            if s == "gagne":    sc, si = "#00DFA222","✅"
            elif s == "perdu":  sc, si = "#FF475722","❌"
            else:               sc, si = "#FFB20022","⚠️"

            if pc is True:    pb = '<span style="color:#00DFA2;font-size:0.7rem;">🎯 ✓</span>'
            elif pc is False: pb = '<span style="color:#FF4757;font-size:0.7rem;">🎯 ✗</span>'
            else:             pb = '<span style="color:#7A8599;font-size:0.7rem;">⚠️</span>'

            vr_str  = f"Vainqueur: <b>{vr}</b>" if vr else "—"
            fav_str = f"Prono: <b>{fav_}</b>"

            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.6rem;'
                f'background:{sc};border-radius:10px;padding:0.55rem 0.8rem;margin-bottom:0.35rem;">'
                f'<span style="font-size:0.9rem;">{si}</span>'
                f'<div style="flex:1;">'
                f'<div style="font-size:0.82rem;font-weight:600;color:#E8EDF5;">'
                f'{row.get("player1","?")} vs {row.get("player2","?")}</div>'
                f'<div style="font-size:0.72rem;color:{COLORS["gray"]};">'
                f'{fav_str} · {vr_str} · {surface_badge(surf_)}</div></div>'
                f'<div style="text-align:right;">{pb}'
                f'<div style="font-size:0.68rem;color:{COLORS["gray"]};margin-top:0.15rem;">{date_}</div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

st.markdown("<br>", unsafe_allow_html=True)

surf_cols = st.columns(3)
for si, surf in enumerate(SURFACES):
    cfg   = SURFACE_CFG[surf]
    sp    = df[df["surface"] == surf]
    s_g   = len(sp[sp["statut"] == "gagne"])
    s_p   = len(sp[sp["statut"] == "perdu"])
    s_a   = len(sp[sp["statut"] == "annule"])
    s_tv  = s_g + s_p
    s_acc = (s_g / s_tv * 100) if s_tv > 0 else 0
    with surf_cols[si]:
        st.markdown(
            f'<div style="background:{cfg["bg"]};border:1px solid {cfg["color"]}44;'
            f'border-radius:14px;padding:1rem;text-align:center;">'
            f'<div style="font-size:1.6rem;margin-bottom:0.2rem;">{cfg["icon"]}</div>'
            f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:{cfg["color"]};">{surf}</div>'
            f'<div style="font-size:1.8rem;font-weight:800;color:#E8EDF5;margin:0.4rem 0;">{s_acc:.0f}%</div>'
            f'<div style="display:flex;justify-content:center;gap:0.75rem;font-size:0.78rem;">'
            f'<span style="color:#00DFA2;">✅ {s_g}</span>'
            f'<span style="color:#FF4757;">❌ {s_p}</span>'
            f'<span style="color:#FFB200;">⚠️ {s_a}</span></div>'
            f'<div style="color:{COLORS["gray"]};font-size:0.72rem;margin-top:0.2rem;">{len(sp)} matchs</div>'
            f'</div>',
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)
if st.button("📥 Exporter l'historique en CSV", use_container_width=True):
    csv = df.to_csv(index=False, encoding="utf-8")
    st.download_button("⬇️ Télécharger CSV", csv, "tennisiq_history.csv", "text/csv")
```

# ═══════════════════════════════════════════════════════════════

# PAGE : VALUE BETS

# ═══════════════════════════════════════════════════════════════

def show_value_bets():
st.markdown(section_title(“💎 Value Bets”, “Opportunités détectées avec avantage mathématique”),
unsafe_allow_html=True)

```
mi   = load_rf_model()
vbs  = []

for m in mock_matches():
    proba, _ = calc_proba(m["p1"], m["p2"], m["surface"], m["tournament"], None, mi)
    seed = hash(f"{m['p1']}{m['p2']}") % 1000 / 1000
    o1   = round(1/proba * (0.88 + 0.15 * seed), 2)
    o2   = round(1/(1-proba) * (0.88 + 0.15 * (1 - seed)), 2)
    e1   = proba - 1/o1
    e2   = (1-proba) - 1/o2
    if e1 > MIN_EDGE_COMBINE:
        vbs.append({"match": f"{m['p1']} vs {m['p2']}", "joueur": m["p1"],
                    "edge": e1, "cote": o1, "proba": proba,
                    "surf": m["surface"], "tournament": m["tournament"]})
    elif e2 > MIN_EDGE_COMBINE:
        vbs.append({"match": f"{m['p1']} vs {m['p2']}", "joueur": m["p2"],
                    "edge": e2, "cote": o2, "proba": 1-proba,
                    "surf": m["surface"], "tournament": m["tournament"]})

vbs.sort(key=lambda x: x["edge"], reverse=True)

if not vbs:
    st.info("Aucun value bet détecté sur les matchs du jour.")
    return

st.markdown(
    f'<div style="font-size:0.88rem;color:{COLORS["gray"]};margin-bottom:1.25rem;">'
    f'🎯 {len(vbs)} opportunité(s) — triées par edge décroissant</div>',
    unsafe_allow_html=True
)

for rank, vb in enumerate(vbs, 1):
    edge_pct   = vb["edge"] * 100
    edge_color = "#00DFA2" if edge_pct >= 5 else "#FFB200"

    st.markdown(
        f'<div style="background:linear-gradient(135deg,rgba(0,223,162,0.10),rgba(0,121,255,0.06));'
        f'border:2px solid {edge_color};border-radius:16px;padding:1.25rem;margin-bottom:0.85rem;">'
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
        f'<div style="display:flex;align-items:center;gap:0.6rem;flex-wrap:wrap;">'
        f'<span style="background:{edge_color}22;border:1px solid {edge_color}55;color:{edge_color};'
        f'border-radius:100px;padding:0.15rem 0.55rem;font-weight:800;font-size:0.8rem;">#{rank}</span>'
        f'<span style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#E8EDF5;">{vb["match"]}</span>'
        f'<span>{surface_badge(vb["surf"])}</span></div>'
        f'<span style="color:{COLORS["gray"]};font-size:0.78rem;">🏆 {vb["tournament"]}</span>'
        f'</div>'
        f'<div style="font-size:1.25rem;font-weight:800;color:#E8EDF5;margin-bottom:0.5rem;">'
        f'👉 MISER SUR : <span style="color:#00DFA2;">{vb["joueur"].upper()}</span></div>'
        f'<div style="display:flex;flex-wrap:wrap;gap:1rem;">'
        f'<div style="text-align:center;background:rgba(255,255,255,0.05);border-radius:10px;padding:0.4rem 0.9rem;">'
        f'<div style="font-size:1.2rem;font-weight:800;color:#FFB200;">{vb["cote"]:.2f}</div>'
        f'<div style="font-size:0.68rem;color:{COLORS["gray"]};text-transform:uppercase;">Cote</div></div>'
        f'<div style="text-align:center;background:rgba(0,223,162,0.1);border-radius:10px;padding:0.4rem 0.9rem;">'
        f'<div style="font-size:1.2rem;font-weight:800;color:{edge_color};">+{edge_pct:.1f}%</div>'
        f'<div style="font-size:0.68rem;color:{COLORS["gray"]};text-transform:uppercase;">Edge</div></div>'
        f'<div style="text-align:center;background:rgba(255,255,255,0.05);border-radius:10px;padding:0.4rem 0.9rem;">'
        f'<div style="font-size:1.2rem;font-weight:800;color:#0079FF;">{vb["proba"]:.1%}</div>'
        f'<div style="font-size:0.68rem;color:{COLORS["gray"]};text-transform:uppercase;">Proba ML</div></div>'
        f'</div>'
        f'<div style="font-size:0.75rem;color:{COLORS["gray"]};margin-top:0.6rem;">'
        f'Probabilité modèle ({vb["proba"]:.1%}) &gt; probabilité implicite cote ({1/vb["cote"]:.1%}) '
        f'→ avantage de +{edge_pct:.1f}%</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Stocker résultat IA en session_state ──────────────────
    vb_ai_key = f"vb_ai_{rank}_{vb['joueur']}"
    vb_ai_txt = st.session_state.get(vb_ai_key)

    tg_col, ai_col = st.columns(2)
    with tg_col:
        if st.button(f"📱 Telegram #{rank}", key=f"vb_tg_{rank}", use_container_width=True):
            msg = (
                f"💎 <b>VALUE BET #{rank} — TENNISIQ</b>\n\n"
                f"🆚 {vb['match']} | {vb['surf']} | {vb['tournament']}\n\n"
                f"🔥🔥 <b>MISER SUR {vb['joueur'].upper()} !</b> 🔥🔥\n"
                f"👉 Cote: <b>{vb['cote']:.2f}</b>\n"
                f"📊 Edge: <b>+{edge_pct:.1f}%</b>\n"
                f"🎯 Proba: <b>{vb['proba']:.1%}</b>\n\n"
                f"<i>Avantage mathématique confirmé</i>\n#TennisIQ #ValueBet"
            )
            ok, resp = tg_send(msg)
            st.success(resp) if ok else st.error(resp)

    with ai_col:
        if vb_ai_txt is None:
            if st.button(f"🤖 Analyse IA #{rank}", key=f"vb_ai_{rank}", use_container_width=True):
                if get_groq_key():
                    with st.spinner("Analyse IA..."):
                        parts = vb["match"].split(" vs ", 1)
                        vb_p1 = parts[0] if len(parts) > 0 else ""
                        vb_p2 = parts[1] if len(parts) > 1 else ""
                        web_info = search_match_web(vb_p1, vb_p2, vb["tournament"]) if get_serper_key() else None
                        bv_dict  = {"joueur": vb["joueur"], "cote": vb["cote"],
                                    "edge": vb["edge"], "proba": vb["proba"]}
                        result = ai_analysis(vb_p1, vb_p2, vb["surf"], vb["tournament"],
                                             vb["proba"], bv_dict, None, web_info)
                        if result:
                            st.session_state[vb_ai_key] = result
                            vb_ai_txt = result
                        else:
                            st.error("❌ Analyse IA échouée")
                else:
                    st.warning("⚠️ GROQ_API_KEY manquante dans les secrets")
        else:
            st.caption("✅ IA déjà calculée")

    # Affichage direct (pas d'expander)
    if vb_ai_txt:
        st.markdown(
            f'<div style="background:rgba(0,121,255,0.06);border:1px solid rgba(0,121,255,0.2);'
            f'border-radius:12px;padding:1.1rem 1.3rem;margin:0.5rem 0;">'
            f'<div style="font-weight:700;color:#0079FF;font-size:0.85rem;margin-bottom:0.5rem;">🤖 Analyse IA</div>'
            f'<div style="color:#E8EDF5;line-height:1.7;font-size:0.88rem;white-space:pre-wrap;">{vb_ai_txt}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
```

# ═══════════════════════════════════════════════════════════════

# PAGE : TELEGRAM

# ═══════════════════════════════════════════════════════════════

def show_telegram():
st.markdown(section_title(“📱 Telegram”, “Configuration et envoi des notifications”),
unsafe_allow_html=True)

```
token, chat_id = get_tg_config()

if not token or not chat_id:
    st.error(
        "❌ Telegram non configuré.\n\n"
        "Ajoutez ces variables dans vos **secrets Streamlit** (`.streamlit/secrets.toml`) :\n\n"
        "```toml\nTELEGRAM_BOT_TOKEN = \"1234567890:AAExxxxxxxxxxxxxxxx\"\n"
        "TELEGRAM_CHAT_ID   = \"-100xxxxxxxxxx\"\n```"
    )
    with st.expander("📖 Comment obtenir ces valeurs ?"):
        st.markdown("""
```

1. Sur Telegram, cherchez **@BotFather** → `/newbot` → copiez le **TOKEN**
1. Cherchez **@userinfobot** pour votre **Chat ID** personnel
1. Pour un canal privé: ajoutez le bot, envoyez un message, visitez `https://api.telegram.org/bot<TOKEN>/getUpdates`
1. Le Chat ID d’un canal privé commence par `-100`
   “””)
   return
   
   st.markdown(
   f’<div style="background:rgba(0,223,162,0.07);border:1px solid rgba(0,223,162,0.22);'
f'border-radius:10px;padding:0.7rem 1rem;margin-bottom:1.25rem;">’
   f’<span style="font-size:1.2rem;">✅</span>’
   f’<span style="font-weight:700;color:#00DFA2;margin-left:0.6rem;">Telegram configuré</span>’
   f’<span style=“color:{COLORS[“gray”]};font-size:0.82rem;margin-left:0.6rem;”>Chat ID: {chat_id}</span>’
   f’</div>’,
   unsafe_allow_html=True
   )
   
   c1, c2, c3 = st.columns(3)
   with c1:
   if st.button(“🔧 Tester la connexion”, use_container_width=True):
   with st.spinner(“Test…”):
   ok, msg = tg_test()
   st.success(msg) if ok else st.error(msg)
   with c2:
   if st.button(“📊 Envoyer les stats”, use_container_width=True):
   with st.spinner(“Envoi…”):
   ok, msg = tg_send(format_stats_msg())
   st.success(msg) if ok else st.error(msg)
   with c3:
   if st.button(“🔄 Vider le cache”, use_container_width=True):
   st.cache_data.clear()
   st.success(“✅ Cache vidé”)
   
   st.markdown(”<br>”, unsafe_allow_html=True)
   st.markdown(
   ‘<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;'
'color:#E8EDF5;margin-bottom:0.6rem;">✍️ Message personnalisé</div>’,
   unsafe_allow_html=True
   )
   
   with st.form(“tg_custom”):
   title    = st.text_input(“Titre du message”, “📢 Message TennisIQ”)
   body     = st.text_area(“Contenu”, height=100, placeholder=“Votre message ici…”)
   col_opt1, col_opt2 = st.columns(2)
   with col_opt1: urgent     = st.checkbox(“🔴 Marquer URGENT”)
   with col_opt2: incl_stats = st.checkbox(“📊 Inclure les stats”)
   submitted = st.form_submit_button(“📤 Envoyer”, use_container_width=True)
   
   if submitted:
   if not body:
   st.warning(“Le message ne peut pas être vide”)
   else:
   prefix        = “🔴 <b>URGENT</b> — “ if urgent else “”
   stats_section = f”\n\n{format_stats_msg()}” if incl_stats else “”
   msg = (
   f”<b>{prefix}{title}</b>\n\n{body}{stats_section}\n\n”
   f”📅 {datetime.now().strftime(’%d/%m/%Y %H:%M’)}”
   )
   with st.spinner(“Envoi en cours…”):
   ok, resp = tg_send(msg)
   st.success(resp) if ok else st.error(resp)
   
   with st.expander(“🔍 Diagnostic avancé”):
   if st.button(“Vérifier le bot Telegram”):
   try:
   r = requests.get(f”https://api.telegram.org/bot{token}/getMe”, timeout=12)
   if r.status_code == 200:
   bot = r.json().get(“result”, {})
   st.json({
   “username”: bot.get(“username”),
   “name”:     bot.get(“first_name”),
   “id”:       bot.get(“id”),
   “can_join_groups”: bot.get(“can_join_groups”),
   })
   st.success(f”✅ Bot valide: @{bot.get(‘username’)}”)
   else:
   st.error(f”Erreur API: {r.text[:200]}”)
   except Exception as e:
   st.error(f”Erreur réseau: {e}”)
   
   ```
    if st.button("Test message simple"):
        ok, resp = tg_send("🎾 TennisIQ — test simple")
        st.success(resp) if ok else st.error(resp)
   ```

# ═══════════════════════════════════════════════════════════════

# PAGE : CONFIGURATION

# ═══════════════════════════════════════════════════════════════

def show_config():
st.markdown(section_title(“⚙️ Configuration”, “Gestion du modèle ML et des données”),
unsafe_allow_html=True)

```
mi       = load_rf_model()
metadata = load_model_metadata()

if mi:
    ps        = mi.get("player_stats", {})
    imp       = mi.get("feature_importance", {})
    acc_model = _safe_float(mi.get("accuracy", metadata.get("accuracy", 0)))
    auc       = _safe_float(mi.get("auc", 0))
    n_matches = metadata.get("n_matches", mi.get("n_matches", 0))
    trained   = mi.get("trained_at", metadata.get("trained_at", "?"))
    version   = mi.get("version",    metadata.get("version", "?"))

    m1 = big_metric("Accuracy", f"{acc_model:.1%}", None,"","#00DFA2")
    m2 = big_metric("AUC-ROC",  f"{auc:.3f}",       None,"","#0079FF")
    m3 = big_metric("Joueurs",  f"{len(ps):,}",     None,"","#7A8599")
    m4 = big_metric("Matchs",   f"{n_matches:,}",   None,"","#7A8599")

    st.markdown(
        f'<div style="background:rgba(0,223,162,0.05);border:1px solid rgba(0,223,162,0.18);'
        f'border-radius:14px;padding:1.1rem;margin-bottom:1.25rem;">'
        f'<div style="font-family:Syne,sans-serif;font-weight:700;color:#00DFA2;margin-bottom:0.6rem;">'
        f'🤖 Modèle ML actif ({acc_model:.1%} accuracy)</div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.6rem;">'
        f'{m1}{m2}{m3}{m4}</div>'
        f'<div style="color:{COLORS["gray"]};font-size:0.78rem;margin-top:0.6rem;">'
        f'Entraîné le {str(trained)[:10]} · Version {version}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    if imp:
        st.markdown(
            '<div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;'
            'color:#E8EDF5;margin-bottom:0.5rem;">🔍 Top features (importance)</div>',
            unsafe_allow_html=True
        )
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
        for feat, val in sorted_imp:
            st.progress(float(val), text=f"{feat}: {val:.1%}")

    if st.button("🔄 Recharger le modèle"):
        st.cache_resource.clear()
        st.rerun()
else:
    st.warning("⚠️ Aucun modèle ML chargé.")
    st.info("Placez `tennis_ml_model_complete.pkl` dans le dossier `models/`")

st.markdown("---")

# Secrets vérification
st.markdown(
    '<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;'
    'color:#E8EDF5;margin-bottom:0.75rem;">🔑 Vérification des secrets Streamlit</div>',
    unsafe_allow_html=True
)

secrets_check = [
    ("GROQ_API_KEY",          get_groq_key(),    "IA Groq (analyse des matchs)"),
    ("SERPER_API_KEY",        get_serper_key(),  "Recherche web (Serper)"),
    ("TELEGRAM_BOT_TOKEN",    get_tg_config()[0],"Notifications Telegram"),
    ("TELEGRAM_CHAT_ID",      get_tg_config()[1],"ID du chat Telegram"),
]

for key_name, val, desc in secrets_check:
    ok    = bool(val)
    color = "#00DFA2" if ok else "#FF4757"
    dot   = "●" if ok else "○"
    label = "Configuré ✓" if ok else "MANQUANT ✗"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.75rem;padding:0.5rem 0.75rem;'
        f'background:rgba(255,255,255,0.03);border-radius:8px;margin-bottom:0.35rem;">'
        f'<span style="color:{color};font-size:0.85rem;">{dot}</span>'
        f'<code style="color:#E8EDF5;flex:1;font-size:0.82rem;">{key_name}</code>'
        f'<span style="color:{COLORS["gray"]};font-size:0.78rem;">{desc}</span>'
        f'<span style="color:{color};font-size:0.78rem;font-weight:700;">{label}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

st.info("💡 Les secrets se configurent dans `.streamlit/secrets.toml` ou dans l'interface Streamlit Cloud → Settings → Secrets")

st.markdown("---")
st.markdown(
    '<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;'
    'color:#E8EDF5;margin-bottom:0.75rem;">🗑️ Gestion des données</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        if HIST_FILE.exists(): HIST_FILE.unlink()
        update_stats()
        # Nettoyer aussi la session_state
        for key in list(st.session_state.keys()):
            if key.startswith("pred_result_") or key.startswith("ai_result_") or key.startswith("web_result_"):
                del st.session_state[key]
        st.rerun()
with c2:
    if st.button("🔄 Recalculer les stats", use_container_width=True):
        update_stats()
        st.success("✅ Stats recalculées")
with c3:
    if st.button("💾 Backup maintenant", use_container_width=True):
        backup()
        st.success("✅ Backup effectué")

if st.button("🧹 Vider le cache session IA", use_container_width=True):
    for key in list(st.session_state.keys()):
        if key.startswith("ai_result_") or key.startswith("web_result_") or key.startswith("vb_ai_"):
            del st.session_state[key]
    st.success("✅ Cache IA vidé — les prochaines analyses seront recalculées")
```

# ═══════════════════════════════════════════════════════════════

# MAIN

# ═══════════════════════════════════════════════════════════════

def main():
st.set_page_config(
page_title=“TennisIQ Pro”,
page_icon=“🎾”,
layout=“wide”,
initial_sidebar_state=“expanded”
)
st.markdown(PRO_CSS, unsafe_allow_html=True)

```
# Backup quotidien
if "last_backup" not in st.session_state:
    st.session_state["last_backup"] = datetime.now()
if (datetime.now() - st.session_state["last_backup"]).seconds >= 86400:
    backup()
    st.session_state["last_backup"] = datetime.now()

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.25rem 0 0.75rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;
        background:linear-gradient(135deg,#00DFA2,#0079FF);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;">TennisIQ</div>
        <div style="font-size:0.72rem;color:#7A8599;letter-spacing:0.1em;
        text-transform:uppercase;margin-top:0.2rem;">ML · Pro Edition</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:0.5rem 0;'>", unsafe_allow_html=True)

    page = st.radio(
        "Nav",
        ["🏠 Dashboard", "🎯 Analyse", "⏳ En Attente", "📊 Statistiques",
         "💎 Value Bets", "📱 Telegram", "⚙️ Configuration"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:0.5rem 0;'>", unsafe_allow_html=True)

    s    = load_user_stats()
    h    = load_history()
    acc  = calc_accuracy()
    pend = len([p for p in h if p.get("statut") == "en_attente"])
    streak_color = "#00DFA2" if s.get("current_streak", 0) > 0 else "#FF4757"

    st.markdown(
        f'<div style="padding:0.25rem 0;">'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;text-align:center;">'
        f'<div style="background:rgba(255,255,255,0.04);border-radius:9px;padding:0.5rem;">'
        f'<div style="font-size:1rem;font-weight:800;color:#00DFA2;">{acc:.1f}%</div>'
        f'<div style="font-size:0.62rem;color:#7A8599;text-transform:uppercase;">Précision</div></div>'
        f'<div style="background:rgba(255,255,255,0.04);border-radius:9px;padding:0.5rem;">'
        f'<div style="font-size:1rem;font-weight:800;color:#FFB200;">{pend}</div>'
        f'<div style="font-size:0.62rem;color:#7A8599;text-transform:uppercase;">En attente</div></div>'
        f'<div style="background:rgba(255,255,255,0.04);border-radius:9px;padding:0.5rem;">'
        f'<div style="font-size:1rem;font-weight:800;color:#00DFA2;">{s.get("correct_predictions",0)}</div>'
        f'<div style="font-size:0.62rem;color:#7A8599;text-transform:uppercase;">Gagnés</div></div>'
        f'<div style="background:rgba(255,255,255,0.04);border-radius:9px;padding:0.5rem;">'
        f'<div style="font-size:1rem;font-weight:800;color:{streak_color};">{s.get("current_streak",0)}</div>'
        f'<div style="font-size:0.62rem;color:#7A8599;text-transform:uppercase;">Série</div></div>'
        f'</div></div>',
        unsafe_allow_html=True
    )

# ── ROUTING ──────────────────────────────────────────────────
if   page == "🏠 Dashboard":    show_dashboard()
elif page == "🎯 Analyse":       show_prediction()
elif page == "⏳ En Attente":    show_pending()
elif page == "📊 Statistiques":  show_statistics()
elif page == "💎 Value Bets":    show_value_bets()
elif page == "📱 Telegram":      show_telegram()
elif page == "⚙️ Configuration": show_config()
```

if **name** == “**main**”:
main()