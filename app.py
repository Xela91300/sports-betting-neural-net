import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime
import hashlib
import warnings
import nest_asyncio
import os
import requests
import gzip
import plotly.graph_objects as go
import shutil
import random

nest_asyncio.apply()
warnings.filterwarnings("ignore")

ROOT_DIR   = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR   = ROOT_DIR / "src" / "data" / "raw" / "tml-tennis"
HIST_DIR   = ROOT_DIR / "history"
BACKUP_DIR = ROOT_DIR / "backups"

for d in [MODELS_DIR, DATA_DIR, HIST_DIR, BACKUP_DIR]:
    d.mkdir(exist_ok=True, parents=True)

HIST_FILE         = HIST_DIR / "predictions_history.json"
USER_STATS_FILE   = HIST_DIR / "user_stats.json"
ACHIEVEMENTS_FILE = HIST_DIR / "achievements.json"
METADATA_FILE     = MODELS_DIR / "model_metadata.json"

SURFACES         = ["Hard", "Clay", "Grass"]
MIN_EDGE_COMBINE = 0.02
MAX_MATCHES      = 30

ACHIEVEMENTS = {
    "first_win":          {"name": "Premiere victoire",  "icon": "T1"},
    "streak_5":           {"name": "En forme",           "icon": "T2"},
    "streak_10":          {"name": "Imbattable",         "icon": "T3"},
    "pred_100":           {"name": "Expert",             "icon": "T4"},
    "value_master":       {"name": "Value Master",       "icon": "T5"},
    "surface_specialist": {"name": "Multi-surface",      "icon": "T6"},
}

TOURNAMENTS_DB = {
    "Australian Open": "Hard",
    "Roland Garros": "Clay",
    "Wimbledon": "Grass",
    "US Open": "Hard",
    "Nitto ATP Finals": "Hard",
    "Indian Wells Masters": "Hard",
    "Miami Open": "Hard",
    "Monte-Carlo Masters": "Clay",
    "Madrid Open": "Clay",
    "Italian Open": "Clay",
    "Canadian Open": "Hard",
    "Cincinnati Masters": "Hard",
    "Shanghai Masters": "Hard",
    "Paris Masters": "Hard",
    "Rotterdam Open": "Hard",
    "Rio Open": "Clay",
    "Dubai Tennis Championships": "Hard",
    "Mexican Open": "Hard",
    "Barcelona Open": "Clay",
    "Halle Open": "Grass",
    "Queen Club Championships": "Grass",
    "Hamburg Open": "Clay",
    "Washington Open": "Hard",
    "China Open": "Hard",
    "Japan Open": "Hard",
    "Vienna Open": "Hard",
    "Swiss Indoors": "Hard",
    "Dallas Open": "Hard",
    "Qatar Open": "Hard",
    "Adelaide International": "Hard",
    "Auckland Open": "Hard",
    "Brisbane International": "Hard",
    "Cordoba Open": "Clay",
    "Buenos Aires": "Clay",
    "Delray Beach": "Hard",
    "Marseille Open": "Hard",
    "Santiago": "Clay",
    "Houston": "Clay",
    "Marrakech": "Clay",
    "Estoril": "Clay",
    "Munich": "Clay",
    "Geneva": "Clay",
    "Lyon": "Clay",
    "Stuttgart": "Grass",
    "Mallorca": "Grass",
    "Eastbourne": "Grass",
    "Newport": "Grass",
    "Atlanta": "Hard",
    "Croatia Open Umag": "Clay",
    "Los Cabos": "Hard",
    "Winston-Salem": "Hard",
    "Chengdu Open": "Hard",
    "Sofia": "Hard",
    "Metz": "Hard",
    "San Diego": "Hard",
    "Seoul": "Hard",
    "Tel Aviv": "Hard",
    "Florence": "Hard",
    "Antwerp": "Hard",
    "Stockholm": "Hard",
    "Belgrade Open": "Clay",
    "Autre tournoi": "Hard",
}

TOURNAMENT_LEVEL = {
    "Australian Open": ("G", 5),
    "Roland Garros": ("G", 5),
    "Wimbledon": ("G", 5),
    "US Open": ("G", 5),
    "Nitto ATP Finals": ("F", 3),
    "Indian Wells Masters": ("M", 3),
    "Miami Open": ("M", 3),
    "Monte-Carlo Masters": ("M", 3),
    "Madrid Open": ("M", 3),
    "Italian Open": ("M", 3),
    "Canadian Open": ("M", 3),
    "Cincinnati Masters": ("M", 3),
    "Shanghai Masters": ("M", 3),
    "Paris Masters": ("M", 3),
}

TOURNAMENT_ALIASES = {
    "australian": "Australian Open",
    "melbourne": "Australian Open",
    "roland garros": "Roland Garros",
    "french open": "Roland Garros",
    "wimbledon": "Wimbledon",
    "us open": "US Open",
    "indian wells": "Indian Wells Masters",
    "miami": "Miami Open",
    "monte carlo": "Monte-Carlo Masters",
    "madrid": "Madrid Open",
    "rome": "Italian Open",
    "canada": "Canadian Open",
    "cincinnati": "Cincinnati Masters",
    "shanghai": "Shanghai Masters",
    "paris masters": "Paris Masters",
    "bercy": "Paris Masters",
    "rotterdam": "Rotterdam Open",
    "dubai": "Dubai Tennis Championships",
    "barcelona": "Barcelona Open",
    "halle": "Halle Open",
    "hamburg": "Hamburg Open",
    "washington": "Washington Open",
    "beijing": "China Open",
    "tokyo": "Japan Open",
    "vienna": "Vienna Open",
    "basel": "Swiss Indoors",
}

COLORS = {
    "primary": "#00DFA2", "secondary": "#0079FF",
    "warning": "#FFB200", "danger": "#FF3B3F", "gray": "#6C7A89",
    "card_bg": "rgba(255,255,255,0.04)", "card_border": "rgba(255,255,255,0.10)",
}

SURFACE_CFG = {
    "Hard":  {"color": "#0079FF", "icon": "H", "bg": "rgba(0,121,255,0.12)"},
    "Clay":  {"color": "#E67E22", "icon": "C", "bg": "rgba(230,126,34,0.12)"},
    "Grass": {"color": "#00DFA2", "icon": "G", "bg": "rgba(0,223,162,0.12)"},
}

PRO_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap");
:root {
    --primary: #00DFA2; --secondary: #0079FF; --bg: #080E1A;
    --card: rgba(255,255,255,0.035); --border: rgba(255,255,255,0.08);
    --text: #E8EDF5; --muted: #7A8599;
}
.stApp { background: var(--bg); font-family: "DM Sans", sans-serif; }
section[data-testid="stSidebar"] { background: rgba(8,14,26,0.97) !important; border-right: 1px solid var(--border) !important; }
h1, h2, h3 { font-family: "Syne", sans-serif !important; color: var(--text) !important; }
.stButton > button { background: linear-gradient(135deg, #00DFA2 0%, #0079FF 100%) !important; color: #080E1A !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; transition: all 0.2s ease !important; }
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 24px rgba(0,223,162,0.25) !important; }
[data-testid="metric-container"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; padding: 1rem 1.25rem !important; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #00DFA2, #0079FF) !important; }
details { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
/* Responsive adjustments */
@media (max-width: 768px) {
    .stButton > button { font-size: 0.8rem; padding: 0.3rem 0.6rem; }
    [data-testid="column"] { min-width: 100% !important; }
    .st-expander { padding: 0 !important; }
}
</style>
"""

def surface_badge(surface):
    cfg = SURFACE_CFG.get(surface, SURFACE_CFG["Hard"])
    return (
        "<span style=\"background:" + cfg["bg"] + ";color:" + cfg["color"] + ";"
        "border:1px solid " + cfg["color"] + "44;border-radius:100px;"
        "padding:0.2rem 0.6rem;font-size:0.75rem;font-weight:600;\">"
        + cfg["icon"] + " " + surface + "</span>"
    )

def section_title(title, subtitle=""):
    sub = (
        "<p style=\"color:#6C7A89;font-size:0.9rem;margin:0.25rem 0 0;\">"
        + subtitle + "</p>"
    ) if subtitle else ""
    return (
        "<div style=\"margin-bottom:1.5rem;\">"
        "<h2 style=\"font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;"
        "color:#E8EDF5;margin:0;letter-spacing:-0.02em;\">" + title + "</h2>"
        + sub + "</div>"
    )

def big_metric(label, value, delta=None, icon="", color="#00DFA2"):
    delta_html = ""
    if delta is not None:
        dcolor = "#00DFA2" if delta >= 0 else "#FF4757"
        sign = "+" if delta >= 0 else ""
        delta_html = (
            "<span style=\"font-size:0.8rem;color:" + dcolor + ";margin-left:0.5rem;\">"
            + sign + str(round(delta, 1)) + "%</span>"
        )
    return (
        "<div style=\"background:rgba(255,255,255,0.04);border:1px solid " + color + "33;"
        "border-radius:16px;padding:1.25rem 1.5rem;text-align:center;\">"
        "<div style=\"font-size:1.6rem;margin-bottom:0.25rem;\">" + icon + "</div>"
        "<div style=\"font-family:Syne,sans-serif;font-size:2rem;font-weight:800;"
        "color:" + color + "\">" + value + delta_html + "</div>"
        "<div style=\"font-size:0.8rem;color:#6C7A89;margin-top:0.25rem;"
        "text-transform:uppercase;letter-spacing:0.08em;\">" + label + "</div>"
        "</div>"
    )

def stat_pill(label, value, color="#00DFA2", icon=""):
    return (
        "<div style=\"display:inline-flex;align-items:center;gap:0.5rem;"
        "background:rgba(0,0,0,0.3);border:1px solid " + color + "33;"
        "border-radius:100px;padding:0.35rem 0.85rem;margin:0.2rem;\">"
        "<span style=\"font-size:0.75rem;color:" + color + "\">" + icon + "</span>"
        "<span style=\"font-size:0.8rem;font-weight:600;color:#6C7A89;\">" + label + "</span>"
        "<span style=\"font-size:0.9rem;font-weight:700;color:" + color + "\">" + value + "</span>"
        "</div>"
    )

def html_progress_bar(proba, color="#00DFA2"):
    pct  = int(proba * 100)
    pct2 = 100 - pct
    return (
        "<div style=\"background:rgba(255,255,255,0.07);border-radius:100px;"
        "height:8px;margin:0.75rem 0;overflow:hidden;\">"
        "<div style=\"background:linear-gradient(90deg," + color + ",#0079FF);"
        "width:" + str(pct) + "%;height:100%;border-radius:100px;\"></div></div>"
        "<div style=\"display:flex;justify-content:space-between;"
        "font-size:0.8rem;margin-top:4px;\">"
        "<span style=\"color:" + color + ";font-weight:700;\">" + str(pct) + "%</span>"
        "<span style=\"color:#7A8599;\">" + str(pct2) + "%</span></div>"
    )

def get_surface(name):
    return TOURNAMENTS_DB.get(name, "Hard")

def get_level(name):
    return TOURNAMENT_LEVEL.get(name, ("A", 3))

def _download_model_gz(model_path):
    url = (
        "https://github.com/Xela91300/sports-betting-neural-net"
        "/releases/latest/download/tennis_ml_model_complete.pkl.gz"
    )
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        return None
    temp_path = MODELS_DIR / "model_temp.pkl.gz"
    temp_path.write_bytes(response.content)
    with gzip.open(temp_path, "rb") as gz:
        model_info = joblib.load(gz)
    joblib.dump(model_info, model_path)
    temp_path.unlink(missing_ok=True)
    return model_info

def load_rf_model():
    if "rf_model_cache" in st.session_state:
        return st.session_state["rf_model_cache"]
    model_path = MODELS_DIR / "tennis_ml_model_complete.pkl"
    model_info = None
    if model_path.exists():
        try:
            candidate = joblib.load(model_path)
            if candidate.get("model") and candidate.get("scaler"):
                model_info = candidate
        except Exception as e:
            st.error("Erreur chargement modele: " + str(e))
    if model_info is None:
        try:
            with st.spinner("Telechargement du modele..."):
                model_info = _download_model_gz(model_path)
        except Exception as e:
            st.warning("Impossible de telecharger le modele: " + str(e))
    st.session_state["rf_model_cache"] = model_info
    return model_info

def load_model_metadata():
    if "model_metadata_cache" in st.session_state:
        return st.session_state["model_metadata_cache"]
    result = {}
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r") as fh:
                result = json.load(fh)
        except Exception:
            result = {}
    st.session_state["model_metadata_cache"] = result
    return result

def get_tg_config():
    try:
        return st.secrets["TELEGRAM_BOT_TOKEN"], str(st.secrets["TELEGRAM_CHAT_ID"])
    except Exception:
        t = os.environ.get("TELEGRAM_BOT_TOKEN")
        c = os.environ.get("TELEGRAM_CHAT_ID")
        return (t, c) if t and c else (None, None)

def tg_send(message, parse_mode="HTML"):
    token, chat_id = get_tg_config()
    if not token or not chat_id:
        return False, "Telegram non configure"
    try:
        url = "https://api.telegram.org/bot" + token + "/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": parse_mode,
                   "disable_web_page_preview": True}
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            return True, "Envoye sur Telegram"
        return False, "Telegram API: " + r.json().get("description", r.text[:100])
    except requests.exceptions.Timeout:
        return False, "Timeout Telegram"
    except Exception as e:
        return False, "Erreur: " + str(e)[:80]

def tg_test():
    token, chat_id = get_tg_config()
    if not token:
        return False, "Token manquant"
    if not chat_id:
        return False, "Chat ID manquant"
    try:
        r = requests.get("https://api.telegram.org/bot" + token + "/getMe", timeout=10)
        if r.status_code != 200:
            return False, "Token invalide"
        bot_name = r.json().get("result", {}).get("first_name", "Bot")
    except Exception as e:
        return False, "Erreur: " + str(e)
    msg = (
        "<b>TennisIQ - Test connexion</b>\n\n"
        "Bot: <b>" + bot_name + "</b>\n"
        "Date: " + datetime.now().strftime("%d/%m/%Y %H:%M") + "\n"
        "Predictions: <b>" + str(len(load_history())) + "</b>\n"
        "Precision: <b>" + str(round(calc_accuracy(), 1)) + "%</b>"
    )
    return tg_send(msg)

def format_pred_msg(pred, bet_suggestions=None, ai_comment=None):
    proba    = pred.get("proba", 0.5)
    bar      = chr(9608) * int(proba * 10) + chr(9617) * (10 - int(proba * 10))
    surf     = pred.get("surface", "Hard")
    fav      = pred.get("favori", "?")
    conf     = pred.get("confidence", 50)
    ml_tag   = "[ML] " if pred.get("ml_used") else ""
    msg = (
        "<b>" + ml_tag + "PREDICTION TENNISIQ</b>\n\n"
        "<b>" + pred.get("player1","?") + " vs " + pred.get("player2","?") + "</b>\n"
        + pred.get("tournament","?") + " | " + surf + "\n\n"
        "<code>" + bar + "</code>\n"
        + pred.get("player1","J1") + ": <b>" + str(round(proba*100,1)) + "%</b>\n"
        + pred.get("player2","J2") + ": <b>" + str(round((1-proba)*100,1)) + "%</b>\n\n"
        "FAVORI: <b>" + fav + "</b>\n"
        "Confiance: <b>" + str(int(conf)) + "/100</b>"
    )
    if pred.get("odds1") and pred.get("odds2"):
        msg += ("\nCotes: " + pred["player1"] + " @ <b>" + str(pred["odds1"]) + "</b>"
                " | " + pred["player2"] + " @ <b>" + str(pred["odds2"]) + "</b>")
    if pred.get("best_value"):
        bv = pred["best_value"]
        msg += ("\n\nVALUE BET: " + bv["joueur"] + " @ " + str(round(bv["cote"],2))
                + " Edge:+" + str(round(bv["edge"]*100,1)) + "%")
    if ai_comment:
        clean = ai_comment.replace("<","&lt;").replace(">","&gt;")
        msg += "\n\nAnalyse IA:\n" + clean[:600]
    msg += "\n\n#TennisIQ"
    return msg

def format_stats_msg():
    stats = load_user_stats()
    h = load_history()
    correct = stats.get("correct_predictions", 0)
    wrong   = stats.get("incorrect_predictions", 0)
    cancel  = stats.get("annules_predictions", 0)
    tv      = correct + wrong
    acc     = (correct / tv * 100) if tv > 0 else 0
    recent  = [p for p in h[-20:] if p.get("statut") in ["gagne","perdu"]]
    r_ok    = sum(1 for p in recent if p.get("statut") == "gagne")
    r_acc   = (r_ok / len(recent) * 100) if recent else 0
    diff    = r_acc - acc
    return (
        "<b>STATISTIQUES TENNISIQ</b>\n\n"
        "Total: <b>" + str(stats.get("total_predictions",0)) + "</b>\n"
        "Gagnes: <b>" + str(correct) + "</b> (" + str(round(acc,1)) + "%)\n"
        "Perdus: <b>" + str(wrong) + "</b>\n"
        "Abandons: <b>" + str(cancel) + "</b>\n\n"
        "Forme recente: <b>" + str(round(r_acc,1)) + "%</b> ("
        + ("+" if diff >= 0 else "") + str(round(diff,1)) + "% vs global)\n"
        "Serie: <b>" + str(stats.get("current_streak",0)) + "</b>\n"
        "Record: <b>" + str(stats.get("best_streak",0)) + "</b>\n\n"
        + datetime.now().strftime("%d/%m/%Y %H:%M") + " #TennisIQ"
    )

# --- Fonctions pour les clés API ---
def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")

def get_deepseek_key():
    try:
        return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        return os.environ.get("DEEPSEEK_API_KEY")

def get_claude_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")

# --- Appels API pour chaque IA ---
def call_groq(prompt):
    key = get_groq_key()
    if not key:
        st.error("Clé API Groq non configurée.")
        return None
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            json={"model": "llama3-8b-8192",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3, "max_tokens": 500},
            timeout=30
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            st.error(f"Erreur Groq ({r.status_code}): {r.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("Timeout de l'API Groq.")
        return None
    except Exception as e:
        st.error(f"Exception lors de l'appel Groq : {e}")
        return None

def call_deepseek(prompt):
    key = get_deepseek_key()
    if not key:
        st.error("Clé API DeepSeek non configurée.")
        return None
    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500
        }
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            st.error(f"Erreur DeepSeek ({r.status_code}): {r.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("Timeout de l'API DeepSeek.")
        return None
    except Exception as e:
        st.error(f"Exception lors de l'appel DeepSeek : {e}")
        return None

def call_claude(prompt):
    key = get_claude_key()
    if not key:
        st.error("Clé API Claude (Anthropic) non configurée.")
        return None
    try:
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        }
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=30
        )
        if r.status_code == 200:
            return r.json()["content"][0]["text"]
        else:
            st.error(f"Erreur Claude ({r.status_code}): {r.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("Timeout de l'API Claude.")
        return None
    except Exception as e:
        st.error(f"Exception lors de l'appel Claude : {e}")
        return None

def ai_analysis(p1, p2, surface, tournament, proba, best_value=None, ia_choice="Groq"):
    """Appelle l'IA choisie pour analyser le match."""
    fav = p1 if proba >= 0.5 else p2
    und = p2 if proba >= 0.5 else p1
    vb  = ""
    if best_value:
        vb = (best_value["joueur"] + " @ " + str(round(best_value["cote"],2))
              + " edge+" + str(round(best_value["edge"]*100,1)) + "%")
    prompt = (
        "Analyse ce match ATP en 4 points:\n\n"
        + p1 + " vs " + p2 + " | " + tournament + " | " + surface + "\n"
        "Proba: " + p1 + " " + str(round(proba*100,1)) + "% - "
        + p2 + " " + str(round((1-proba)*100,1)) + "%\n"
        "FAVORI: " + fav + (" | VB: " + vb if vb else "") + "\n\n"
        "1. Pourquoi " + fav + " est favori\n"
        "2. Points faibles de " + und + "\n"
        "3. Conseil pari\n"
        "4. Pronostic final\n\n"
        "Reponds en francais, sois concis."
    )

    if ia_choice == "Groq":
        return call_groq(prompt)
    elif ia_choice == "DeepSeek":
        return call_deepseek(prompt)
    elif ia_choice == "Claude":
        return call_claude(prompt)
    else:
        return None

def extract_21_features(ps, p1, p2, surface, level="A", best_of=3, h2h_ratio=0.5):
    s1, s2 = ps.get(p1, {}), ps.get(p2, {})
    r1 = max(s1.get("rank", 500.0), 1.0)
    r2 = max(s2.get("rank", 500.0), 1.0)
    sp1, sp2 = s1.get("serve_pct", {}), s2.get("serve_pct", {})
    sr1, sr2 = s1.get("serve_raw", {}), s2.get("serve_raw", {})
    feats = [
        float(np.log(r2 / r1)),
        (s1.get("rank_points", 0) - s2.get("rank_points", 0)) / 5000.0,
        float(s1.get("age", 25) - s2.get("age", 25)),
        1.0 if surface == "Clay"  else 0.0,
        1.0 if surface == "Grass" else 0.0,
        1.0 if surface == "Hard"  else 0.0,
        1.0 if level == "G" else 0.0,
        1.0 if level == "M" else 0.0,
        1.0 if best_of == 5 else 0.0,
        float(s1.get("surface_wr", {}).get(surface, 0.5) - s2.get("surface_wr", {}).get(surface, 0.5)),
        float(s1.get("win_rate", 0.5) - s2.get("win_rate", 0.5)),
        float(s1.get("recent_form", 0.5) - s2.get("recent_form", 0.5)),
        float(h2h_ratio),
        (sr1.get("ace", 0) - sr2.get("ace", 0)) / 10.0,
        (sr1.get("df",  0) - sr2.get("df",  0)) / 5.0,
        float(sp1.get("pct_1st_in",   0) - sp2.get("pct_1st_in",   0)),
        float(sp1.get("pct_1st_won",  0) - sp2.get("pct_1st_won",  0)),
        float(sp1.get("pct_2nd_won",  0) - sp2.get("pct_2nd_won",  0)),
        float(sp1.get("pct_bp_saved", 0) - sp2.get("pct_bp_saved", 0)),
        float(s1.get("days_since_last", 30) - s2.get("days_since_last", 30)),
        float(s1.get("fatigue", 0) - s2.get("fatigue", 0)),
    ]
    return np.nan_to_num(np.array(feats, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

def predict_rf(p1, p2, surface, tournament="", h2h_ratio_val=0.5):
    mi = load_rf_model()
    if mi is None:
        return None, "rf_absent"
    try:
        m  = mi.get("model")
        sc = mi.get("scaler")
        ps = mi.get("player_stats", {})
        if m is None or sc is None:
            return None, "rf_incomplet"
        if p1 not in ps or p2 not in ps:
            return None, "rf_joueurs_inconnus"
        lv, bo = get_level(tournament)
        f = extract_21_features(ps, p1, p2, surface, lv, bo, h2h_ratio_val)
        p = float(m.predict_proba(sc.transform(f.reshape(1, -1)))[0][1])
        return max(0.05, min(0.95, p)), "rf_ok"
    except Exception as e:
        return None, "rf_erreur:" + str(e)[:40]

@st.cache_data(ttl=3600)
def load_players():
    if not DATA_DIR.exists():
        return []
    players = set()
    for f in DATA_DIR.glob("*.csv"):
        if "wta" in f.name.lower():
            continue
        try:
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(f, encoding=enc,
                                     usecols=["winner_name","loser_name"],
                                     on_bad_lines="skip")
                    players.update(df["winner_name"].dropna().astype(str).str.strip())
                    players.update(df["loser_name"].dropna().astype(str).str.strip())
                    break
                except Exception:
                    continue
        except Exception:
            pass
    return sorted(p for p in players if p and p.lower() != "nan" and len(p) > 1)

@st.cache_data(ttl=3600)
def load_h2h_df():
    if not DATA_DIR.exists():
        return pd.DataFrame()
    dfs = []
    for f in list(DATA_DIR.glob("*.csv"))[:20]:
        if "wta" in f.name.lower():
            continue
        try:
            df = pd.read_csv(f, encoding="utf-8",
                             usecols=["winner_name","loser_name"],
                             on_bad_lines="skip")
            df["winner_name"] = df["winner_name"].astype(str).str.strip()
            df["loser_name"]  = df["loser_name"].astype(str).str.strip()
            dfs.append(df)
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def get_h2h(p1, p2):
    df = load_h2h_df()
    if df.empty:
        return None
    mask = (
        ((df.winner_name == p1) & (df.loser_name == p2)) |
        ((df.winner_name == p2) & (df.loser_name == p1))
    )
    h = df[mask]
    if len(h) == 0:
        return None
    return {"total": len(h),
            "p1_wins": len(h[h.winner_name == p1]),
            "p2_wins": len(h[h.winner_name == p2])}

def h2h_ratio(h2h, p1):
    if not h2h or h2h["total"] == 0:
        return 0.5
    return h2h["p1_wins"] / h2h["total"]

def calc_proba(p1, p2, surface, tournament="", h2h=None, mi=None):
    ratio = h2h_ratio(h2h, p1)
    if mi:
        p, status = predict_rf(p1, p2, surface, tournament, ratio)
        if p is not None:
            return p, True
    proba = 0.5 + (ratio - 0.5) * 0.3
    return max(0.05, min(0.95, proba)), False

def calc_confidence(proba, h2h=None):
    c = 50.0
    if h2h and h2h.get("total", 0) >= 3:
        c += 10
    c += abs(proba - 0.5) * 40
    return min(100.0, c)

def load_history():
    if not HIST_FILE.exists():
        return []
    try:
        with open(HIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_pred(pred):
    try:
        h = load_history()
        pred["id"] = hashlib.md5(
            (str(datetime.now()) + pred.get("player1","")).encode()
        ).hexdigest()[:8]
        pred["statut"] = "en_attente"
        pred["vainqueur_reel"] = None
        pred["pronostic_correct"] = None
        h.append(pred)
        with open(HIST_FILE, "w", encoding="utf-8") as f:
            json.dump(h[-1000:], f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

def update_pred_result(pred_id, statut, vainqueur_reel=None):
    try:
        h = load_history()
        for p in h:
            if p.get("id") == pred_id:
                p["statut"] = statut
                p["date_maj"] = datetime.now().isoformat()
                p["vainqueur_reel"] = vainqueur_reel
                p["pronostic_correct"] = (vainqueur_reel == p.get("favori")) if vainqueur_reel else None
                break
        with open(HIST_FILE, "w", encoding="utf-8") as f:
            json.dump(h, f, indent=2, ensure_ascii=False)
        update_stats()
        return True
    except Exception:
        return False

def load_user_stats():
    default = {"total_predictions": 0, "correct_predictions": 0,
               "incorrect_predictions": 0, "annules_predictions": 0,
               "current_streak": 0, "best_streak": 0}
    if not USER_STATS_FILE.exists():
        return default
    try:
        with open(USER_STATS_FILE) as f:
            return json.load(f)
    except Exception:
        return default

def update_stats():
    h = load_history()
    correct   = sum(1 for p in h if p.get("statut") == "gagne")
    incorrect = sum(1 for p in h if p.get("statut") == "perdu")
    cancel    = sum(1 for p in h if p.get("statut") == "annule")
    streak = cur = best = 0
    for p in reversed(h):
        if p.get("statut") == "gagne":
            streak += 1; cur = streak; best = max(best, streak)
        elif p.get("statut") == "perdu":
            streak = 0; cur = 0
    stats = {"total_predictions": len(h), "correct_predictions": correct,
             "incorrect_predictions": incorrect, "annules_predictions": cancel,
             "current_streak": cur, "best_streak": best}
    with open(USER_STATS_FILE, "w") as f:
        json.dump(stats, f)
    return stats

def calc_accuracy():
    s = load_user_stats()
    tv = s.get("correct_predictions", 0) + s.get("incorrect_predictions", 0)
    return (s.get("correct_predictions", 0) / tv * 100) if tv > 0 else 0

def load_ach():
    if not ACHIEVEMENTS_FILE.exists():
        return {}
    try:
        with open(ACHIEVEMENTS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def save_ach(a):
    try:
        with open(ACHIEVEMENTS_FILE, "w") as f:
            json.dump(a, f)
    except Exception:
        pass

def check_achievements():
    s = load_user_stats(); h = load_history(); a = load_ach(); new = []
    checks = [
        ("first_win", s.get("correct_predictions", 0) >= 1),
        ("streak_5",  s.get("best_streak", 0) >= 5),
        ("streak_10", s.get("best_streak", 0) >= 10),
        ("pred_100",  s.get("total_predictions", 0) >= 100),
    ]
    for aid, cond in checks:
        if cond and aid not in a:
            a[aid] = {"unlocked_at": datetime.now().isoformat()}
            new.append(ACHIEVEMENTS[aid])
    vw = sum(1 for p in h if p.get("best_value") and p.get("statut") == "gagne")
    if vw >= 10 and "value_master" not in a:
        a["value_master"] = {"unlocked_at": datetime.now().isoformat()}
        new.append(ACHIEVEMENTS["value_master"])
    surfs = {p.get("surface") for p in h if p.get("statut") == "gagne"}
    if len(surfs) >= 3 and "surface_specialist" not in a:
        a["surface_specialist"] = {"unlocked_at": datetime.now().isoformat()}
        new.append(ACHIEVEMENTS["surface_specialist"])
    if new:
        save_ach(a)
    return new

def backup():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for f in [HIST_FILE, USER_STATS_FILE]:
        if f.exists():
            try:
                shutil.copy(f, BACKUP_DIR / (f.stem + "_" + ts + f.suffix))
            except Exception:
                pass

def mock_matches():
    return [
        {"p1":"Novak Djokovic","p2":"Carlos Alcaraz","surface":"Clay","tournament":"Roland Garros"},
        {"p1":"Jannik Sinner","p2":"Daniil Medvedev","surface":"Hard","tournament":"Miami Open"},
        {"p1":"Alexander Zverev","p2":"Stefanos Tsitsipas","surface":"Clay","tournament":"Madrid Open"},
        {"p1":"Holger Rune","p2":"Casper Ruud","surface":"Grass","tournament":"Wimbledon"},
    ]

@st.cache_data(ttl=1800)
def get_matches():
    return mock_matches()

def alt_bets(p1, p2, surface, proba):
    bets = []
    if proba > 0.6 or proba < 0.4:
        bets.append({"type":"Under 22.5 games","description":"Moins de 22.5 jeux","proba":0.64,"cote":1.78,"confidence":68})
    else:
        bets.append({"type":"Over 22.5 games","description":"Plus de 22.5 jeux","proba":0.61,"cote":1.82,"confidence":63})
    if proba > 0.65:
        bets.append({"type":"Handicap -3.5","description":p1 + " gagne avec ecart","proba":0.57,"cote":2.15,"confidence":58})
    elif proba < 0.35:
        bets.append({"type":"Handicap +3.5","description":p2 + " favori quand meme","proba":0.60,"cote":1.98,"confidence":62})
    if 0.3 < proba < 0.7:
        bets.append({"type":"Set chacun","description":"Match en 3 sets","proba":0.54,"cote":2.25,"confidence":54})
    return bets

def player_sel(label, all_players, key, default=None):
    search = st.text_input("Rechercher " + label, key="srch_" + key, placeholder="Tapez un nom...")
    if search:
        filtered = [p for p in all_players if search.lower() in p.lower()]
    else:
        filtered = all_players[:200]
    if not filtered and search:
        filtered = [p for p in all_players if p and p[0].lower() == search[0].lower()][:50]
    st.caption(str(len(filtered)) + " / " + str(len(all_players)) + " joueurs")
    if not filtered:
        return st.text_input(label, key=key)
    idx = 0
    if default:
        for i, p in enumerate(filtered):
            if default.lower() in p.lower():
                idx = i
                break
    return st.selectbox(label, filtered, index=idx, key=key)

def tourn_sel(label, key, default=None):
    search = st.text_input("Rechercher " + label, key="srcht_" + key,
                           placeholder="ex: Roland Garros...")
    all_t = sorted(TOURNAMENTS_DB.keys())
    if search:
        sl = search.lower().strip()
        res = set()
        if sl in TOURNAMENT_ALIASES:
            res.add(TOURNAMENT_ALIASES[sl])
        for t in all_t:
            if sl in t.lower():
                res.add(t)
        filtered = sorted(res) if res else all_t[:50]
    else:
        filtered = all_t[:100]
    idx = filtered.index(default) if default and default in filtered else 0
    return st.selectbox(label, filtered, index=idx, key=key)

def show_dashboard():
    st.markdown(section_title("Dashboard", "Vue d ensemble de vos performances"),
                unsafe_allow_html=True)
    stats = load_user_stats()
    h = load_history()
    a = load_ach()
    mi = load_rf_model()
    metadata = load_model_metadata()
    correct = stats.get("correct_predictions", 0)
    wrong   = stats.get("incorrect_predictions", 0)
    cancel  = stats.get("annules_predictions", 0)
    pending = len([p for p in h if p.get("statut") == "en_attente"])
    tv  = correct + wrong
    acc = (correct / tv * 100) if tv > 0 else 0
    recent = [p for p in h[-20:] if p.get("statut") in ["gagne","perdu"]]
    r_acc = (sum(1 for p in recent if p.get("statut")=="gagne") / len(recent) * 100) if recent else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(big_metric("PRECISION", str(round(acc,1)) + "%", r_acc-acc if tv>0 else None, "", "#00DFA2"), unsafe_allow_html=True)
    with c2: st.markdown(big_metric("GAGNES", str(correct), None, "", "#00DFA2"), unsafe_allow_html=True)
    with c3: st.markdown(big_metric("PERDUS", str(wrong), None, "", "#FF4757"), unsafe_allow_html=True)
    with c4: st.markdown(big_metric("ABANDONS", str(cancel), None, "", "#FFB200"), unsafe_allow_html=True)
    with c5: st.markdown(big_metric("EN ATTENTE", str(pending), None, "", "#7A8599"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 2])
    with col_l:
        streak = stats.get("current_streak", 0)
        best   = stats.get("best_streak", 0)
        sc = "#00DFA2" if streak > 0 else "#7A8599"
        st.markdown(
            "<div style=\"background:rgba(255,255,255,0.04);border:1px solid " + sc + "44;"
            "border-radius:16px;padding:1.5rem;text-align:center;\">"
            "<div style=\"font-family:Syne,sans-serif;font-size:3rem;font-weight:800;"
            "color:" + sc + "\">" + str(streak) + "</div>"
            "<div style=\"color:#6C7A89;font-size:0.85rem;text-transform:uppercase;"
            "letter-spacing:0.1em;\">Serie actuelle</div>"
            "<div style=\"margin-top:0.75rem;padding-top:0.75rem;"
            "border-top:1px solid rgba(255,255,255,0.10);\">"
            "<span style=\"color:#6C7A89;font-size:0.8rem;\">Record: </span>"
            "<span style=\"color:#FFB200;font-weight:700;\">" + str(best) + "</span>"
            "</div></div>",
            unsafe_allow_html=True
        )
    with col_r:
        tg_token, _ = get_tg_config()
        groq_key    = get_groq_key()
        deepseek_key = get_deepseek_key()
        claude_key   = get_claude_key()
        services = []
        if mi:
            ps = mi.get("player_stats", {})
            acc_model = mi.get("accuracy", metadata.get("accuracy", 0))
            services.append(("Modele ML", str(round(acc_model*100,1)) + "% acc - " + str(len(ps)) + " joueurs", True))
        else:
            services.append(("Modele ML", "Non charge", False))
        services.append(("IA Groq", "Connectee" if groq_key else "Non configuree", bool(groq_key)))
        services.append(("IA DeepSeek", "Connectee" if deepseek_key else "Non configuree", bool(deepseek_key)))
        services.append(("IA Claude", "Connectee" if claude_key else "Non configuree", bool(claude_key)))
        services.append(("Telegram", "Configure" if tg_token else "Non configure", bool(tg_token)))
        svc_html = ("<div style=\"background:rgba(255,255,255,0.04);"
                    "border:1px solid rgba(255,255,255,0.10);"
                    "border-radius:16px;padding:1.5rem;\">"
                    "<div style=\"font-family:Syne,sans-serif;font-size:1rem;font-weight:700;"
                    "color:#E8EDF5;margin-bottom:1rem;\">STATUT DES SERVICES</div>"
                    "<div style=\"display:grid;gap:0.75rem;\">")
        for svc, desc, ok in services:
            color = "#00DFA2" if ok else "#FF4757"
            dot   = "ON" if ok else "OFF"
            svc_html += ("<div style=\"display:flex;align-items:center;gap:0.75rem;"
                         "padding:0.5rem 0.75rem;background:rgba(255,255,255,0.03);"
                         "border-radius:8px;\">"
                         "<span style=\"color:" + color + ";font-size:0.75rem;font-weight:700;\">"
                         + dot + "</span>"
                         "<span style=\"font-weight:600;color:#E8EDF5;flex:1;\">" + svc + "</span>"
                         "<span style=\"color:#6C7A89;font-size:0.8rem;\">" + desc + "</span>"
                         "</div>")
        svc_html += "</div></div>"
        st.markdown(svc_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    finished = [p for p in h if p.get("statut") in ["gagne","perdu"]]
    if len(finished) >= 3:
        df_h = pd.DataFrame(finished)
        df_h["ok"]     = (df_h["statut"] == "gagne").astype(int)
        df_h["cum_ok"] = df_h["ok"].expanding().sum()
        df_h["cum_n"]  = range(1, len(df_h)+1)
        df_h["acc"]    = df_h["cum_ok"] / df_h["cum_n"] * 100
        df_h["n"]      = range(1, len(df_h)+1)
        fig = go.Figure()
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.15)")
        fig.add_trace(go.Scatter(x=df_h["n"], y=df_h["acc"], mode="lines",
                                  line=dict(color="#00DFA2", width=2.5),
                                  fill="tozeroy", fillcolor="rgba(0,223,162,0.07)"))
        fig.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#7A8599"), showlegend=False,
                          xaxis=dict(showgrid=False, title="Prediction #", color="#7A8599"),
                          yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                                     title="Precision (%)", color="#7A8599", range=[0,100]))
        st.plotly_chart(fig, use_container_width=True)

    col_surf, col_badges = st.columns([3, 2])
    with col_surf:
        surf_data = []
        for surf in SURFACES:
            sp = [p for p in h if p.get("surface")==surf and p.get("statut") in ["gagne","perdu"]]
            if sp:
                ok = sum(1 for p in sp if p.get("statut")=="gagne")
                surf_data.append({"Surface":surf,"Precision":ok/len(sp)*100,"Total":len(sp)})
        if surf_data:
            df_s = pd.DataFrame(surf_data)
            fig2 = go.Figure(go.Bar(
                x=df_s["Surface"], y=df_s["Precision"],
                text=df_s["Precision"].round(0).astype(int).astype(str) + "%",
                textposition="outside",
                marker_color=[SURFACE_CFG[s]["color"] for s in df_s["Surface"]],
                customdata=df_s["Total"],
                hovertemplate="<b>%{x}</b><br>%{y:.1f}%<br>%{customdata} matchs<extra></extra>"
            ))
            fig2.update_layout(height=220, margin=dict(l=0,r=0,t=30,b=0),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#7A8599"), showlegend=False,
                               xaxis=dict(showgrid=False, color="#7A8599"),
                               yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                                          color="#7A8599", range=[0,110]))
            st.plotly_chart(fig2, use_container_width=True)
    with col_badges:
        st.markdown("<div style=\"font-family:Syne,sans-serif;font-size:1.1rem;"
                    "font-weight:700;color:#E8EDF5;margin-bottom:0.75rem;\">Badges ("
                    + str(len(a)) + "/" + str(len(ACHIEVEMENTS)) + ")</div>",
                    unsafe_allow_html=True)
        if a:
            for aid, adata_val in list(a.items())[:4]:
                ach_meta = ACHIEVEMENTS.get(aid, {})
                try:
                    d = datetime.fromisoformat(adata_val["unlocked_at"]).strftime("%d/%m/%Y")
                except Exception:
                    d = "?"
                st.markdown(
                    "<div style=\"background:rgba(0,223,162,0.06);"
                    "border:1px solid rgba(0,223,162,0.2);"
                    "border-radius:10px;padding:0.6rem 0.9rem;margin-bottom:0.5rem;\">"
                    "<div style=\"font-weight:700;color:#00DFA2;font-size:0.85rem;\">"
                    + ach_meta.get("name","") + "</div>"
                    "<div style=\"color:#6C7A89;font-size:0.72rem;\">Debloque le " + d + "</div>"
                    "</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style=\"text-align:center;padding:2rem;color:#6C7A89;"
                        "border:1px dashed rgba(255,255,255,0.10);"
                        "border-radius:12px;\">Aucun badge encore</div>",
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Envoyer les stats sur Telegram"):
        ok, msg = tg_send(format_stats_msg())
        st.success(msg) if ok else st.error(msg)

def show_prediction():
    st.markdown(section_title("Analyse Multi-matchs", "Predictions ML"), unsafe_allow_html=True)
    mi = load_rf_model()
    metadata = load_model_metadata()
    if mi:
        ps = mi.get("player_stats", {})
        acc_model = mi.get("accuracy", metadata.get("accuracy", 0))
        st.markdown(
            "<div style=\"background:rgba(0,223,162,0.08);"
            "border:1px solid rgba(0,223,162,0.25);border-radius:12px;"
            "padding:0.75rem 1rem;margin-bottom:1rem;\">"
            "<span style=\"font-weight:700;color:#00DFA2;\">Modele ML actif ("
            + str(round(acc_model*100,1)) + "% accuracy)</span>"
            "<span style=\"color:#6C7A89;font-size:0.85rem;margin-left:0.75rem;\">"
            + str(len(ps)) + " joueurs - 21 features</span></div>",
            unsafe_allow_html=True)
    else:
        st.warning("Modele ML non charge - mode CSV")

    with st.spinner("Chargement des joueurs..."):
        all_p = load_players()

    # Paramètres d'analyse dans la barre latérale
    with st.sidebar:
        st.markdown("### Paramètres d'analyse")
        n = st.number_input("Nombre de matchs", 1, MAX_MATCHES, 2)
        ia_options = ["Aucune"]
        if get_groq_key(): ia_options.append("Groq")
        if get_deepseek_key(): ia_options.append("DeepSeek")
        if get_claude_key(): ia_options.append("Claude")
        default_ia = "Aucune"
        if len(ia_options) > 1:
            default_ia = ia_options[1]  # première IA disponible
        ia_choice = st.selectbox("IA pour analyse", ia_options, index=ia_options.index(default_ia))
        send_tg = st.checkbox("Envoi Telegram auto", False)

    inputs = []
    for i in range(n):
        with st.expander("Match " + str(i+1), expanded=(i==0)):
            # Première ligne : tournoi et surface
            ct, cs = st.columns([3, 1])
            with ct:
                tourn = tourn_sel("Tournoi", "t"+str(i), "Roland Garros")
            with cs:
                surf = get_surface(tourn)
                lv, bo = get_level(tourn)
                cfg = SURFACE_CFG[surf]
                st.markdown(
                    "<div style=\"background:" + cfg["bg"] + ";"
                    "border:1px solid " + cfg["color"] + "55;"
                    "border-radius:10px;padding:0.6rem;text-align:center;"
                    "margin-top:1.75rem;\">"
                    "<div style=\"font-weight:700;color:" + cfg["color"] + ";font-size:0.9rem;\">"
                    + surf + "</div>"
                    + ("<div style=\"font-size:0.7rem;color:#7A8599;\">Best of 5</div>" if bo==5 else "")
                    + "</div>", unsafe_allow_html=True)

            # Deuxième ligne : joueurs et cotes
            cp1, cp2 = st.columns(2)
            with cp1:
                p1 = player_sel("Joueur 1", all_p, "p1_"+str(i))
                o1 = st.text_input("Cote " + (p1[:15] if p1 else "J1"), key="o1_"+str(i), placeholder="1.75")
            with cp2:
                p2_list = [p for p in all_p if p != p1]
                p2 = player_sel("Joueur 2", p2_list, "p2_"+str(i))
                o2 = st.text_input("Cote " + (p2[:15] if p2 else "J2"), key="o2_"+str(i), placeholder="2.10")

            if mi and p1 and p2:
                ps_d = mi.get("player_stats", {})
                st.caption("ML: " + p1[:20] + " " + ("OK" if p1 in ps_d else "inconnu")
                           + " / " + p2[:20] + " " + ("OK" if p2 in ps_d else "inconnu"))
            inputs.append({"p1":p1,"p2":p2,"surf":surf,"tourn":tourn,"o1":o1,"o2":o2})

    if not st.button("Analyser tous les matchs", type="primary", use_container_width=True):
        return

    valid = [m for m in inputs if m["p1"] and m["p2"]]
    if not valid:
        st.warning("Remplis au moins un match")
        return

    st.markdown("---")
    st.markdown(section_title("Resultats de l analyse"), unsafe_allow_html=True)

    for i, m in enumerate(valid):
        p1, p2, surf, tourn = m["p1"], m["p2"], m["surf"], m["tourn"]
        h2h_data = get_h2h(p1, p2)
        proba, ml_used = calc_proba(p1, p2, surf, tourn, h2h_data, mi)
        conf = calc_confidence(proba, h2h_data)
        fav  = p1 if proba >= 0.5 else p2
        cfg  = SURFACE_CFG[surf]
        h2h_str = ("H2H " + str(h2h_data["p1_wins"]) + "-" + str(h2h_data["p2_wins"])
                   + " (" + str(h2h_data["total"]) + ")") if h2h_data else "H2H: aucun"

        p1_color = "#00DFA2" if fav==p1 else "#7A8599"
        p2_color = "#00DFA2" if fav==p2 else "#7A8599"
        p1_bg = "rgba(0,223,162,0.07)" if fav==p1 else "transparent"
        p2_bg = "rgba(0,223,162,0.07)" if fav==p2 else "transparent"
        p1_border = "rgba(0,223,162,0.2)" if fav==p1 else "transparent"
        p2_border = "rgba(0,223,162,0.2)" if fav==p2 else "transparent"
        p1_tag = "<div style=\"color:#00DFA2;font-size:0.75rem;font-weight:700;\">FAVORI</div>" if fav==p1 else "<div style=\"color:#7A8599;font-size:0.75rem;\">outsider</div>"
        p2_tag = "<div style=\"color:#00DFA2;font-size:0.75rem;font-weight:700;\">FAVORI</div>" if fav==p2 else "<div style=\"color:#7A8599;font-size:0.75rem;\">outsider</div>"

        st.markdown(
            "<div style=\"background:rgba(255,255,255,0.04);"
            "border:1px solid rgba(255,255,255,0.10);"
            "border-radius:16px;padding:1.5rem;margin-bottom:1rem;\">"
            "<div style=\"display:flex;align-items:center;"
            "justify-content:space-between;margin-bottom:1rem;\">"
            "<div><span style=\"font-family:Syne,sans-serif;font-size:1.2rem;"
            "font-weight:800;color:#E8EDF5;\">Match " + str(i+1) + "</span>"
            "<span style=\"margin-left:0.75rem;\">" + surface_badge(surf) + "</span>"
            "<span style=\"color:#6C7A89;font-size:0.85rem;margin-left:0.5rem;\">"
            + tourn + "</span></div>"
            "<span style=\"color:" + ("#00DFA2" if ml_used else "#7A8599") + ";"
            "font-size:0.8rem;font-weight:600;\">"
            + ("ML 21 features" if ml_used else "CSV fallback") + "</span></div>"
            "<div style=\"display:grid;grid-template-columns:1fr auto 1fr;"
            "gap:1rem;align-items:center;\">"
            "<div style=\"text-align:center;background:" + p1_bg + ";"
            "border-radius:12px;padding:0.75rem;"
            "border:1px solid " + p1_border + "\">"
            "<div style=\"font-family:Syne,sans-serif;font-size:1.05rem;"
            "font-weight:700;color:#E8EDF5;\">" + p1 + "</div>"
            "<div style=\"font-size:2rem;font-weight:800;color:" + p1_color + ";\">"
            + str(round(proba*100,1)) + "%</div>" + p1_tag + "</div>"
            "<div style=\"text-align:center;color:#6C7A89;"
            "font-weight:700;font-size:1.4rem;\">VS</div>"
            "<div style=\"text-align:center;background:" + p2_bg + ";"
            "border-radius:12px;padding:0.75rem;"
            "border:1px solid " + p2_border + "\">"
            "<div style=\"font-family:Syne,sans-serif;font-size:1.05rem;"
            "font-weight:700;color:#E8EDF5;\">" + p2 + "</div>"
            "<div style=\"font-size:2rem;font-weight:800;color:" + p2_color + ";\">"
            + str(round((1-proba)*100,1)) + "%</div>" + p2_tag + "</div></div>"
            "<div style=\"margin:0.75rem 0;\">"
            "<div style=\"display:flex;justify-content:space-between;"
            "font-size:0.72rem;color:#6C7A89;margin-bottom:4px;\">"
            "<span>" + p1 + "</span><span>" + p2 + "</span></div>"
            + html_progress_bar(proba, cfg["color"])
            + "</div>"
            "<div style=\"display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.75rem;\">"
            + stat_pill("Confiance", str(int(conf)) + "/100", "#00DFA2", "")
            + stat_pill("H2H", h2h_str, "#0079FF", "")
            + stat_pill("Format", "Best of " + str(get_level(tourn)[1]), "#7A8599", "")
            + "</div></div>",
            unsafe_allow_html=True
        )

        best_val = None
        if m["o1"] and m["o2"]:
            try:
                o1f = float(m["o1"].replace(",","."))
                o2f = float(m["o2"].replace(",","."))
                e1 = proba - 1/o1f
                e2 = (1-proba) - 1/o2f
                if e1 > MIN_EDGE_COMBINE:
                    best_val = {"joueur":p1,"edge":e1,"cote":o1f,"proba":proba}
                elif e2 > MIN_EDGE_COMBINE:
                    best_val = {"joueur":p2,"edge":e2,"cote":o2f,"proba":1-proba}
                if best_val:
                    st.success("VALUE BET: " + best_val["joueur"]
                               + " @ " + str(round(best_val["cote"],2))
                               + "  Edge:+" + str(round(best_val["edge"]*100,1)) + "%")
            except Exception:
                pass
        else:
            st.caption("Renseignez les cotes pour détecter des value bets")

        bets = alt_bets(p1, p2, surf, proba)
        with st.expander("Paris alternatifs"):
            for b in bets:
                ci2 = "OK" if b["confidence"] >= 65 else "~"
                st.markdown(ci2 + " **" + b["type"] + "** - " + b["description"]
                            + "  Proba " + str(round(b["proba"]*100,1)) + "%"
                            + "  Cote " + str(b["cote"]))

        ai_txt = None
        if ia_choice != "Aucune":
            with st.spinner(f"Analyse IA ({ia_choice}) en cours..."):
                ai_txt = ai_analysis(p1, p2, surf, tourn, proba, best_val, ia_choice)
            if ai_txt:
                with st.expander("Analyse IA", expanded=True):
                    st.markdown(
                        "<div style=\"background:rgba(0,121,255,0.06);"
                        "border:1px solid rgba(0,121,255,0.2);border-radius:10px;"
                        "padding:1rem;font-size:0.9rem;line-height:1.6;color:#E8EDF5;\">"
                        + ai_txt.replace("\n","<br>") + "</div>",
                        unsafe_allow_html=True)
            else:
                st.caption("Analyse IA non disponible")

        pred_data = {"player1":p1,"player2":p2,"tournament":tourn,"surface":surf,
                     "proba":float(proba),"confidence":float(conf),
                     "odds1":m["o1"],"odds2":m["o2"],"favori":fav,
                     "best_value":best_val,"ml_used":ml_used,
                     "date":datetime.now().isoformat()}

        cb1, cb2 = st.columns(2)
        with cb1:
            if st.button("Sauvegarder", key="save_"+str(i), use_container_width=True):
                st.success("Sauvegarde!") if save_pred(pred_data) else st.error("Erreur sauvegarde")
        with cb2:
            if st.button("Envoyer Telegram", key="tg_"+str(i), use_container_width=True):
                ok, resp = tg_send(format_pred_msg(pred_data, bets, ai_txt))
                st.success(resp) if ok else st.error(resp)

        if send_tg and i == 0:
            save_pred(pred_data)
            tg_send(format_pred_msg(pred_data, bets, ai_txt))
        st.markdown("---")

    nb = check_achievements()
    if nb:
        st.balloons()
        st.success(str(len(nb)) + " badge(s) debloque(s)!")

def show_pending():
    st.markdown(section_title("En attente", "Validez les resultats"), unsafe_allow_html=True)
    h = load_history()
    pending = [p for p in h if p.get("statut") == "en_attente"]
    if not pending:
        st.markdown("<div style=\"text-align:center;padding:3rem;"
                    "background:rgba(255,255,255,0.04);"
                    "border:1px dashed rgba(255,255,255,0.10);"
                    "border-radius:16px;\">"
                    "<div style=\"font-size:1.2rem;font-weight:700;"
                    "color:#E8EDF5;\">Aucune prediction en attente!</div></div>",
                    unsafe_allow_html=True)
        return
    st.info(str(len(pending)) + " prediction(s) en attente")
    for pred in reversed(pending):
        pid      = pred.get("id","?")
        p1       = pred.get("player1","?")
        p2       = pred.get("player2","?")
        fav      = pred.get("favori","?")
        surf     = pred.get("surface","Hard")
        tourn    = pred.get("tournament","?")
        proba    = pred.get("proba", 0.5)
        conf     = pred.get("confidence", 50)
        date_str = pred.get("date","")[:16].replace("T"," ")
        fav_proba = proba if fav == p1 else (1 - proba)

        st.markdown(
            "<div style=\"background:rgba(255,255,255,0.04);"
            "border:1px solid rgba(255,255,255,0.10);"
            "border-radius:16px;padding:1.5rem;margin-bottom:0.75rem;\">"
            "<div style=\"display:flex;align-items:center;"
            "justify-content:space-between;margin-bottom:1rem;\">"
            "<div><span style=\"font-family:Syne,sans-serif;font-size:1.05rem;"
            "font-weight:700;color:#E8EDF5;\">" + p1 + " vs " + p2 + "</span>"
            "<span style=\"margin-left:0.75rem;\">" + surface_badge(surf) + "</span></div>"
            "<span style=\"color:#6C7A89;font-size:0.78rem;\">" + date_str + "</span></div>"
            "<div style=\"margin-bottom:0.75rem;\">"
            "<span style=\"color:#6C7A89;font-size:0.85rem;\">" + tourn + "</span>"
            "<span style=\"color:#E8EDF5;font-size:0.85rem;margin-left:1rem;\">"
            "Favori: <strong style=\"color:#00DFA2;\">" + fav + "</strong>"
            " (" + str(round(fav_proba*100,1)) + "%) | Confiance: " + str(int(conf)) + "/100"
            "</span></div>"
            "<div style=\"font-weight:600;color:#E8EDF5;font-size:0.9rem;"
            "background:rgba(255,255,255,0.04);border-radius:8px;"
            "padding:0.5rem 0.75rem;\">Qui a gagne ce match?</div></div>",
            unsafe_allow_html=True
        )
        if pred.get("best_value"):
            bv = pred["best_value"]
            st.markdown(
                "<div style=\"background:rgba(0,223,162,0.07);"
                "border:1px solid rgba(0,223,162,0.2);"
                "border-radius:8px;padding:0.5rem 0.75rem;"
                "margin-bottom:0.5rem;font-size:0.8rem;\">"
                "Value bet: <strong style=\"color:#00DFA2;\">"
                + bv["joueur"] + " @ " + str(round(bv.get("cote",0),2))
                + "</strong> Edge:+" + str(round(bv.get("edge",0)*100,1)) + "%</div>",
                unsafe_allow_html=True)

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            if st.button(p1[:22] + " gagne", key="w1_"+pid,
                         use_container_width=True,
                         type="primary" if fav==p1 else "secondary"):
                update_pred_result(pid, "gagne" if fav==p1 else "perdu", vainqueur_reel=p1)
                check_achievements(); st.rerun()
        with c2:
            if st.button(p2[:22] + " gagne", key="w2_"+pid,
                         use_container_width=True,
                         type="primary" if fav==p2 else "secondary"):
                update_pred_result(pid, "gagne" if fav==p2 else "perdu", vainqueur_reel=p2)
                check_achievements(); st.rerun()
        with c3:
            if st.button("Abandon", key="ab_"+pid, use_container_width=True):
                update_pred_result(pid, "annule"); st.rerun()

        st.caption("Prono: " + fav + " favori")
        st.markdown("<br>", unsafe_allow_html=True)

def show_statistics():
    st.markdown(section_title("Statistiques", "Analyse complete"), unsafe_allow_html=True)
    h = load_history()
    if not h:
        st.info("Aucune prediction enregistree.")
        return
    df = pd.DataFrame(h)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "pronostic_correct" not in df.columns:
        df["pronostic_correct"] = False
    df["pronostic_correct"] = df["pronostic_correct"].fillna(False)
    gagnes   = df[df["statut"] == "gagne"]
    perdus   = df[df["statut"] == "perdu"]
    abandons = df[df["statut"] == "annule"]
    fini     = df[df["statut"].isin(["gagne","perdu","annule"])]
    tv  = len(gagnes) + len(perdus)
    acc = (len(gagnes) / tv * 100) if tv > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.markdown(big_metric("TOTAL", str(len(df)), None, "", "#0079FF"), unsafe_allow_html=True)
    with c2: st.markdown(big_metric("GAGNES", str(len(gagnes)), None, "", "#00DFA2"), unsafe_allow_html=True)
    with c3: st.markdown(big_metric("PERDUS", str(len(perdus)), None, "", "#FF4757"), unsafe_allow_html=True)
    with c4: st.markdown(big_metric("ABANDONS", str(len(abandons)), None, "", "#FFB200"), unsafe_allow_html=True)
    with c5: st.markdown(big_metric("PRECISION", str(round(acc,1)) + "%", None, "", "#00DFA2"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_pie, col_table = st.columns([1, 2])
    with col_pie:
        if tv > 0:
            fig_d = go.Figure(go.Pie(
                labels=["Gagnes","Perdus","Abandons"],
                values=[len(gagnes),len(perdus),len(abandons)],
                hole=0.65, marker_colors=["#00DFA2","#FF4757","#FFB200"],
                textinfo="none"))
            fig_d.update_layout(
                height=240, margin=dict(l=0,r=0,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#7A8599"),
                legend=dict(font=dict(size=11, color="#E8EDF5")),
                annotations=[dict(text="<b>" + str(int(acc)) + "%</b>",
                                   x=0.5, y=0.5,
                                   font=dict(size=22, color="#00DFA2", family="Syne"),
                                   showarrow=False)])
            st.plotly_chart(fig_d, use_container_width=True)
    with col_table:
        if not fini.empty:
            recent_fini = fini.sort_values("date", ascending=False).head(10)
            for _, row in recent_fini.iterrows():
                s     = row.get("statut","?")
                pc    = row.get("pronostic_correct")
                fav_r = row.get("favori","?")
                vr    = row.get("vainqueur_reel","?")
                date_ = str(row.get("date",""))[:10]
                surf_ = str(row.get("surface","?"))
                sc = "#00DFA244" if s=="gagne" else "#FF475744" if s=="perdu" else "#FFB20044"
                si = "V" if s=="gagne" else "D" if s=="perdu" else "~"
                pb = ("<span style=\"color:#00DFA2;font-size:0.72rem;\">OK</span>" if pc is True
                      else "<span style=\"color:#FF4757;font-size:0.72rem;\">X</span>" if pc is False
                      else "<span style=\"color:#7A8599;font-size:0.72rem;\">~</span>")
                st.markdown(
                    "<div style=\"display:flex;align-items:center;gap:0.75rem;"
                    "background:" + sc + ";border-radius:10px;"
                    "padding:0.6rem 0.9rem;margin-bottom:0.4rem;\">"
                    "<span>" + si + "</span>"
                    "<div style=\"flex:1;\">"
                    "<div style=\"font-size:0.85rem;font-weight:600;color:#E8EDF5;\">"
                    + str(row.get("player1","?")) + " vs " + str(row.get("player2","?")) + "</div>"
                    "<div style=\"font-size:0.75rem;color:#6C7A89;\">"
                    "Prono: <strong>" + str(fav_r) + "</strong>"
                    " | Vainqueur: <strong>" + str(vr) + "</strong>"
                    " | " + surface_badge(surf_) + "</div></div>"
                    "<div style=\"text-align:right;\">" + pb
                    + "<div style=\"font-size:0.7rem;color:#6C7A89;\">" + str(date_) + "</div>"
                    "</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    surf_cols = st.columns(3)
    for si, surf in enumerate(SURFACES):
        cfg = SURFACE_CFG[surf]
        sp = df[df["surface"]==surf]
        s_g = len(sp[sp["statut"]=="gagne"])
        s_p = len(sp[sp["statut"]=="perdu"])
        s_a = len(sp[sp["statut"]=="annule"])
        s_acc = (s_g / (s_g+s_p) * 100) if (s_g+s_p) > 0 else 0
        with surf_cols[si]:
            st.markdown(
                "<div style=\"background:" + cfg["bg"] + ";"
                "border:1px solid " + cfg["color"] + "44;"
                "border-radius:14px;padding:1.25rem;text-align:center;\">"
                "<div style=\"font-family:Syne,sans-serif;font-size:1.1rem;"
                "font-weight:700;color:" + cfg["color"] + "\">" + surf + "</div>"
                "<div style=\"font-size:2rem;font-weight:800;color:#E8EDF5;"
                "margin:0.5rem 0;\">" + str(int(s_acc)) + "%</div>"
                "<div style=\"display:flex;justify-content:center;gap:1rem;font-size:0.8rem;\">"
                "<span style=\"color:#00DFA2;\">V " + str(s_g) + "</span>"
                "<span style=\"color:#FF4757;\">D " + str(s_p) + "</span>"
                "<span style=\"color:#FFB200;\">A " + str(s_a) + "</span>"
                "</div><div style=\"color:#6C7A89;font-size:0.75rem;margin-top:0.25rem;\">"
                + str(len(sp)) + " matchs</div></div>",
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Exporter en CSV"):
        csv = df.to_csv(index=False, encoding="utf-8")
        st.download_button("Telecharger CSV", csv, "tennisiq_history.csv", "text/csv")

def show_telegram():
    st.markdown(section_title("Telegram", "Notifications"), unsafe_allow_html=True)
    token, chat_id = get_tg_config()
    if not token or not chat_id:
        st.warning("Telegram non configure. Ajouter dans les secrets Streamlit:")
        st.code("TELEGRAM_BOT_TOKEN = YOUR_TOKEN" + chr(10) + "TELEGRAM_CHAT_ID = YOUR_CHAT_ID")
        return
    st.success("Telegram configure - Chat ID: " + str(chat_id))
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Tester connexion", use_container_width=True):
            with st.spinner("Test..."):
                ok, msg = tg_test()
            st.success(msg) if ok else st.error(msg)
    with c2:
        if st.button("Envoyer stats", use_container_width=True):
            ok, msg = tg_send(format_stats_msg())
            st.success(msg) if ok else st.error(msg)
    with c3:
        if st.button("Vider cache", use_container_width=True):
            st.cache_data.clear(); st.success("Cache vide")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("tg_custom"):
        title      = st.text_input("Titre", "Message TennisIQ")
        body       = st.text_area("Contenu", height=100)
        urgent     = st.checkbox("URGENT")
        incl_stats = st.checkbox("Inclure stats")
        if st.form_submit_button("Envoyer", use_container_width=True):
            if not body:
                st.warning("Message vide")
            else:
                prefix = "URGENT - " if urgent else ""
                stats_section = ("\n\n" + format_stats_msg()) if incl_stats else ""
                msg = ("<b>" + prefix + title + "</b>\n\n"
                       + body + stats_section + "\n\n"
                       + datetime.now().strftime("%d/%m/%Y %H:%M"))
                ok, resp = tg_send(msg)
                st.success(resp) if ok else st.error(resp)

def show_config():
    st.markdown(section_title("Configuration", "Gestion du modele"), unsafe_allow_html=True)
    mi = load_rf_model()
    metadata = load_model_metadata()
    if mi:
        ps  = mi.get("player_stats", {})
        imp = mi.get("feature_importance", {})
        acc_model = mi.get("accuracy", metadata.get("accuracy", 0))
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(big_metric("Accuracy", str(round(acc_model*100,1)) + "%", None, "", "#00DFA2"), unsafe_allow_html=True)
        with c2: st.markdown(big_metric("AUC-ROC", str(round(mi.get("auc",0),3)), None, "", "#0079FF"), unsafe_allow_html=True)
        with c3: st.markdown(big_metric("Joueurs", str(len(ps)), None, "", "#7A8599"), unsafe_allow_html=True)
        with c4: st.markdown(big_metric("Matchs", str(metadata.get("n_matches", mi.get("n_matches",0))), None, "", "#7A8599"), unsafe_allow_html=True)
        st.caption("Entraine le " + mi.get("trained_at", metadata.get("trained_at","?"))[:10])
        if imp:
            st.markdown("**Top features:**")
            sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, val in sorted_imp:
                st.progress(float(val), text=feat + ": " + str(round(val*100,1)) + "%")
        if st.button("Recharger le modele"):
            for k in ["rf_model_cache","model_metadata_cache"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
    else:
        st.warning("Aucun modele ML charge.")
        st.info("Placer tennis_ml_model_complete.pkl dans models/")

    st.markdown("---")
    st.subheader("Tests des IA")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Tester Groq"):
            with st.spinner("Test Groq..."):
                resp = call_groq("Dis 'OK' en un mot.")
            st.write("Réponse :", resp)
    with col2:
        if st.button("Tester DeepSeek"):
            with st.spinner("Test DeepSeek..."):
                resp = call_deepseek("Dis 'OK' en un mot.")
            st.write("Réponse :", resp)
    with col3:
        if st.button("Tester Claude"):
            with st.spinner("Test Claude..."):
                resp = call_claude("Dis 'OK' en un mot.")
            st.write("Réponse :", resp)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Effacer historique", use_container_width=True):
            if HIST_FILE.exists(): HIST_FILE.unlink()
            update_stats(); st.rerun()
    with c2:
        if st.button("Recalculer stats", use_container_width=True):
            update_stats(); st.success("Stats recalculees")
    with c3:
        if st.button("Backup", use_container_width=True):
            backup(); st.success("Backup effectue")

def show_value_bets():
    st.markdown(section_title("Value Bets", "Opportunites detectees"), unsafe_allow_html=True)
    mi = load_rf_model()
    vbs = []
    for m in get_matches():
        proba, _ = calc_proba(m["p1"], m["p2"], m["surface"], m["tournament"], None, mi)
        o1 = round(1/proba * (0.9 + 0.2*random.random()), 2)
        o2 = round(1/(1-proba) * (0.9 + 0.2*random.random()), 2)
        e1 = proba - 1/o1
        e2 = (1-proba) - 1/o2
        if e1 > MIN_EDGE_COMBINE:
            vbs.append({"match": m["p1"] + " vs " + m["p2"],
                        "joueur": m["p1"], "edge": e1, "cote": o1,
                        "proba": proba, "surf": m["surface"]})
        elif e2 > MIN_EDGE_COMBINE:
            vbs.append({"match": m["p1"] + " vs " + m["p2"],
                        "joueur": m["p2"], "edge": e2, "cote": o2,
                        "proba": 1-proba, "surf": m["surface"]})
    vbs.sort(key=lambda x: x["edge"], reverse=True)
    if vbs:
        for vb in vbs:
            c1, c2, c3, c4 = st.columns([3,1,1,1])
            with c1: st.markdown("**" + vb["joueur"] + "**"); st.caption(vb["match"] + " " + vb["surf"])
            with c2: st.metric("Cote", str(round(vb["cote"],2)))
            with c3: st.metric("Edge", "+" + str(round(vb["edge"]*100,1)) + "%")
            with c4: st.metric("Proba", str(round(vb["proba"]*100,1)) + "%")
            st.divider()
    else:
        st.info("Aucun value bet detecte.")

def main():
    st.set_page_config(page_title="TennisIQ Pro", page_icon="T",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(PRO_CSS, unsafe_allow_html=True)

    if "last_backup" not in st.session_state:
        st.session_state["last_backup"] = datetime.now()
    if (datetime.now() - st.session_state["last_backup"]).seconds >= 86400:
        backup()
        st.session_state["last_backup"] = datetime.now()

    with st.sidebar:
        st.markdown(
            "<div style=\"text-align:center;padding:1.5rem 0 1rem;\">"
            "<div style=\"font-family:Syne,sans-serif;font-size:2rem;font-weight:800;"
            "background:linear-gradient(135deg,#00DFA2,#0079FF);"
            "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
            "background-clip:text;\">TennisIQ</div>"
            "<div style=\"font-size:0.75rem;color:#7A8599;text-transform:uppercase;\">"
            "ML Pro Edition</div></div>",
            unsafe_allow_html=True)

        page = st.radio("Navigation",
                        ["Dashboard","Analyse","En Attente","Statistiques",
                         "Value Bets","Telegram","Configuration"],
                        label_visibility="collapsed")

        s    = load_user_stats()
        h    = load_history()
        acc  = calc_accuracy()
        pend = len([p for p in h if p.get("statut") == "en_attente"])
        sc   = "#FF4757" if s.get("current_streak",0)==0 else "#00DFA2"
        st.markdown(
            "<div style=\"padding:0.5rem 0;\">"
            "<div style=\"display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;text-align:center;\">"
            "<div style=\"background:rgba(255,255,255,0.04);border-radius:10px;padding:0.6rem;\">"
            "<div style=\"font-size:1.1rem;font-weight:800;color:#00DFA2;\">" + str(round(acc,1)) + "%</div>"
            "<div style=\"font-size:0.65rem;color:#7A8599;text-transform:uppercase;\">Precision</div></div>"
            "<div style=\"background:rgba(255,255,255,0.04);border-radius:10px;padding:0.6rem;\">"
            "<div style=\"font-size:1.1rem;font-weight:800;color:#FFB200;\">" + str(pend) + "</div>"
            "<div style=\"font-size:0.65rem;color:#7A8599;text-transform:uppercase;\">Attente</div></div>"
            "<div style=\"background:rgba(255,255,255,0.04);border-radius:10px;padding:0.6rem;\">"
            "<div style=\"font-size:1.1rem;font-weight:800;color:#00DFA2;\">" + str(s.get("correct_predictions",0)) + "</div>"
            "<div style=\"font-size:0.65rem;color:#7A8599;text-transform:uppercase;\">Gagnes</div></div>"
            "<div style=\"background:rgba(255,255,255,0.04);border-radius:10px;padding:0.6rem;\">"
            "<div style=\"font-size:1.1rem;font-weight:800;color:" + sc + ";\">" + str(s.get("current_streak",0)) + "</div>"
            "<div style=\"font-size:0.65rem;color:#7A8599;text-transform:uppercase;\">Serie</div></div>"
            "</div></div>",
            unsafe_allow_html=True)

    if   page == "Dashboard":     show_dashboard()
    elif page == "Analyse":       show_prediction()
    elif page == "En Attente":    show_pending()
    elif page == "Statistiques":  show_statistics()
    elif page == "Value Bets":    show_value_bets()
    elif page == "Telegram":      show_telegram()
    elif page == "Configuration": show_config()

if __name__ == "__main__":
    main()