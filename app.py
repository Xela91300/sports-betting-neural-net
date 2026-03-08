"""
TennisIQ Pro — Version Précision Maximale
Améliorations : ELO dynamique par surface, ensemble de modèles,
critère de Kelly, score de momentum, calibration des probabilités.
"""
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime, timedelta
import hashlib
import warnings
import nest_asyncio
import os
import requests
import gzip
import plotly.graph_objects as go
import shutil
import random
import math

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
ELO_CACHE_FILE    = HIST_DIR / "elo_ratings.json"

SURFACES         = ["Hard", "Clay", "Grass"]
MIN_EDGE_COMBINE = 0.02
MAX_MATCHES      = 30

# ─── ELO configuration ───────────────────────────────────────
ELO_K_BASE    = 32       # K-factor de base
ELO_K_GRAND   = 40       # K-factor Grand Chelem
ELO_K_MASTERS = 36       # K-factor Masters
ELO_BASE      = 1500     # ELO de départ
ELO_DECAY     = 0.998    # Décroissance hebdomadaire

ACHIEVEMENTS = {
    "first_win":          {"name": "Premiere victoire",  "icon": "T1"},
    "streak_5":           {"name": "En forme",           "icon": "T2"},
    "streak_10":          {"name": "Imbattable",         "icon": "T3"},
    "pred_100":           {"name": "Expert",             "icon": "T4"},
    "value_master":       {"name": "Value Master",       "icon": "T5"},
    "surface_specialist": {"name": "Multi-surface",      "icon": "T6"},
}

TOURNAMENTS_DB = {
    "Australian Open": "Hard", "Roland Garros": "Clay", "Wimbledon": "Grass",
    "US Open": "Hard", "Nitto ATP Finals": "Hard", "Indian Wells Masters": "Hard",
    "Miami Open": "Hard", "Monte-Carlo Masters": "Clay", "Madrid Open": "Clay",
    "Italian Open": "Clay", "Canadian Open": "Hard", "Cincinnati Masters": "Hard",
    "Shanghai Masters": "Hard", "Paris Masters": "Hard", "Rotterdam Open": "Hard",
    "Rio Open": "Clay", "Dubai Tennis Championships": "Hard", "Mexican Open": "Hard",
    "Barcelona Open": "Clay", "Halle Open": "Grass", "Queen Club Championships": "Grass",
    "Hamburg Open": "Clay", "Washington Open": "Hard", "China Open": "Hard",
    "Japan Open": "Hard", "Vienna Open": "Hard", "Swiss Indoors": "Hard",
    "Dallas Open": "Hard", "Qatar Open": "Hard", "Adelaide International": "Hard",
    "Auckland Open": "Hard", "Brisbane International": "Hard", "Cordoba Open": "Clay",
    "Buenos Aires": "Clay", "Delray Beach": "Hard", "Marseille Open": "Hard",
    "Santiago": "Clay", "Houston": "Clay", "Marrakech": "Clay", "Estoril": "Clay",
    "Munich": "Clay", "Geneva": "Clay", "Lyon": "Clay", "Stuttgart": "Grass",
    "Mallorca": "Grass", "Eastbourne": "Grass", "Newport": "Grass", "Atlanta": "Hard",
    "Croatia Open Umag": "Clay", "Los Cabos": "Hard", "Winston-Salem": "Hard",
    "Chengdu Open": "Hard", "Sofia": "Hard", "Metz": "Hard", "San Diego": "Hard",
    "Seoul": "Hard", "Tel Aviv": "Hard", "Florence": "Hard", "Antwerp": "Hard",
    "Stockholm": "Hard", "Belgrade Open": "Clay", "Autre tournoi": "Hard",
}

TOURNAMENT_LEVEL = {
    "Australian Open": ("G", 5), "Roland Garros": ("G", 5),
    "Wimbledon": ("G", 5), "US Open": ("G", 5), "Nitto ATP Finals": ("F", 3),
    "Indian Wells Masters": ("M", 3), "Miami Open": ("M", 3),
    "Monte-Carlo Masters": ("M", 3), "Madrid Open": ("M", 3),
    "Italian Open": ("M", 3), "Canadian Open": ("M", 3),
    "Cincinnati Masters": ("M", 3), "Shanghai Masters": ("M", 3),
    "Paris Masters": ("M", 3),
}

TOURNAMENT_ALIASES = {
    "australian": "Australian Open", "melbourne": "Australian Open",
    "roland garros": "Roland Garros", "french open": "Roland Garros",
    "wimbledon": "Wimbledon", "us open": "US Open",
    "indian wells": "Indian Wells Masters", "miami": "Miami Open",
    "monte carlo": "Monte-Carlo Masters", "madrid": "Madrid Open",
    "rome": "Italian Open", "canada": "Canadian Open",
    "cincinnati": "Cincinnati Masters", "shanghai": "Shanghai Masters",
    "paris masters": "Paris Masters", "bercy": "Paris Masters",
    "rotterdam": "Rotterdam Open", "dubai": "Dubai Tennis Championships",
    "barcelona": "Barcelona Open", "halle": "Halle Open",
    "hamburg": "Hamburg Open", "washington": "Washington Open",
    "beijing": "China Open", "tokyo": "Japan Open",
    "vienna": "Vienna Open", "basel": "Swiss Indoors",
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
section[data-testid="stSidebar"] {
    background: rgba(8,14,26,0.97) !important;
    border-right: 1px solid var(--border) !important;
}
h1, h2, h3 { font-family: "Syne", sans-serif !important; color: var(--text) !important; }
.stButton > button {
    background: linear-gradient(135deg, #00DFA2 0%, #0079FF 100%) !important;
    color: #080E1A !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,223,162,0.25) !important;
}
[data-testid="metric-container"] {
    background: var(--card) !important; border: 1px solid var(--border) !important;
    border-radius: 14px !important; padding: 1rem 1.25rem !important;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00DFA2, #0079FF) !important;
}
details { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
</style>
"""

# ═══════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════
def surface_badge(surface):
    cfg = SURFACE_CFG.get(surface, SURFACE_CFG["Hard"])
    return (
        "<span style='background:" + cfg["bg"] + ";color:" + cfg["color"] + ";"
        "border:1px solid " + cfg["color"] + "44;border-radius:100px;"
        "padding:0.2rem 0.6rem;font-size:0.75rem;font-weight:600;'>"
        + cfg["icon"] + " " + surface + "</span>"
    )

def section_title(title, subtitle=""):
    sub = ("<p style='color:#6C7A89;font-size:0.9rem;margin:0.25rem 0 0;'>"
           + subtitle + "</p>") if subtitle else ""
    return ("<div style='margin-bottom:1.5rem;'>"
            "<h2 style='font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;"
            "color:#E8EDF5;margin:0;letter-spacing:-0.02em;'>" + title + "</h2>"
            + sub + "</div>")

def big_metric(label, value, delta=None, color="#00DFA2"):
    delta_html = ""
    if delta is not None:
        dcolor = "#00DFA2" if delta >= 0 else "#FF4757"
        sign = "+" if delta >= 0 else ""
        delta_html = ("<span style='font-size:0.8rem;color:" + dcolor
                      + ";margin-left:0.5rem;'>" + sign + str(round(delta, 1)) + "%</span>")
    return ("<div style='background:rgba(255,255,255,0.04);border:1px solid " + color + "33;"
            "border-radius:16px;padding:1.25rem 1.5rem;text-align:center;'>"
            "<div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;"
            "color:" + color + "'>" + value + delta_html + "</div>"
            "<div style='font-size:0.8rem;color:#6C7A89;margin-top:0.25rem;"
            "text-transform:uppercase;letter-spacing:0.08em;'>" + label + "</div>"
            "</div>")

def stat_pill(label, value, color="#00DFA2"):
    return ("<div style='display:inline-flex;align-items:center;gap:0.5rem;"
            "background:rgba(0,0,0,0.3);border:1px solid " + color + "33;"
            "border-radius:100px;padding:0.35rem 0.85rem;margin:0.2rem;'>"
            "<span style='font-size:0.8rem;font-weight:600;color:#6C7A89;'>" + label + "</span>"
            "<span style='font-size:0.9rem;font-weight:700;color:" + color + "'>" + value + "</span>"
            "</div>")

def confidence_bar_html(value, label="", color="#00DFA2"):
    pct = min(100, max(0, int(value)))
    bar_color = "#00DFA2" if pct >= 70 else "#FFB200" if pct >= 50 else "#FF4757"
    return ("<div style='margin:0.3rem 0;'>"
            + ("<div style='font-size:0.75rem;color:#7A8599;margin-bottom:2px;'>" + label + "</div>" if label else "")
            + "<div style='background:rgba(255,255,255,0.07);border-radius:100px;height:6px;'>"
            "<div style='background:" + bar_color + ";width:" + str(pct) + "%;height:100%;border-radius:100px;'></div>"
            "</div>"
            "<span style='font-size:0.72rem;color:" + bar_color + ";font-weight:700;'>" + str(pct) + "%</span>"
            "</div>")

def kelly_badge(fraction, label="Kelly"):
    if fraction <= 0:
        return ""
    color = "#00DFA2" if fraction >= 0.05 else "#FFB200"
    pct   = round(fraction * 100, 1)
    return ("<span style='background:rgba(0,223,162,0.12);border:1px solid rgba(0,223,162,0.3);"
            "border-radius:100px;padding:0.25rem 0.7rem;font-size:0.75rem;font-weight:700;"
            "color:" + color + ";'>" + label + " " + str(pct) + "% bankroll</span>")

# ═══════════════════════════════════════════════════════════════
# UTILITAIRES TOURNOIS
# ═══════════════════════════════════════════════════════════════
def get_surface(name):  return TOURNAMENTS_DB.get(name, "Hard")
def get_level(name):    return TOURNAMENT_LEVEL.get(name, ("A", 3))

def _safe_float(v, default=0.0):
    try:    return float(v)
    except: return default

# ═══════════════════════════════════════════════════════════════
# SYSTÈME ELO DYNAMIQUE PAR SURFACE
# Calculé depuis les CSV historiques + cache JSON
# ═══════════════════════════════════════════════════════════════
def _elo_expected(ra, rb):
    """Probabilité attendue pour le joueur A contre B."""
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

def _k_factor(level_code):
    return {"G": ELO_K_GRAND, "M": ELO_K_MASTERS, "F": ELO_K_MASTERS}.get(level_code, ELO_K_BASE)

@st.cache_data(ttl=7200, show_spinner=False)
def compute_elo_from_csv():
    """
    Calcule les ELO globaux et par surface depuis les CSV historiques.
    Retourne un dict : {player: {global, Hard, Clay, Grass, matches, last_date}}
    """
    if not DATA_DIR.exists():
        return {}

    all_rows = []
    cols_needed = ["winner_name", "loser_name", "tourney_date", "surface",
                   "tourney_level", "score"]
    for f in sorted(DATA_DIR.glob("*.csv")):
        if "wta" in f.name.lower():
            continue
        try:
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(f, encoding=enc, on_bad_lines="skip",
                                     usecols=[c for c in cols_needed
                                              if c in pd.read_csv(f, nrows=0, encoding=enc).columns])
                    all_rows.append(df)
                    break
                except Exception:
                    continue
        except Exception:
            continue

    if not all_rows:
        return {}

    df = pd.concat(all_rows, ignore_index=True)
    # Normalisation
    for col in ["winner_name", "loser_name", "surface"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "tourney_date" in df.columns:
        df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["tourney_date"]).sort_values("tourney_date")
    else:
        df = df.reset_index(drop=True)

    # Init ELO
    elo_global  = {}
    elo_surface = {s: {} for s in SURFACES}
    matches_cnt = {}
    last_date   = {}

    def _get(d, player):
        return d.setdefault(player, ELO_BASE)

    for _, row in df.iterrows():
        w   = row.get("winner_name", "")
        l   = row.get("loser_name",  "")
        sur = row.get("surface",     "Hard")
        lv  = str(row.get("tourney_level", "A"))
        dt  = row.get("tourney_date", None)

        if not w or not l or w == "nan" or l == "nan":
            continue
        if sur not in SURFACES:
            sur = "Hard"

        k = _k_factor(lv)

        # Global ELO
        ew = _get(elo_global, w)
        el = _get(elo_global, l)
        exp_w = _elo_expected(ew, el)
        elo_global[w] = ew + k * (1 - exp_w)
        elo_global[l] = el + k * (0 - (1 - exp_w))

        # Surface ELO
        ews = _get(elo_surface[sur], w)
        els = _get(elo_surface[sur], l)
        exp_ws = _elo_expected(ews, els)
        elo_surface[sur][w] = ews + k * (1 - exp_ws)
        elo_surface[sur][l] = els + k * (0 - (1 - exp_ws))

        # Metadata
        matches_cnt[w] = matches_cnt.get(w, 0) + 1
        matches_cnt[l] = matches_cnt.get(l, 0) + 1
        if dt is not None:
            last_date[w] = str(dt)[:10]
            last_date[l] = str(dt)[:10]

    # Assemble
    all_players = set(elo_global.keys())
    result = {}
    for p in all_players:
        result[p] = {
            "global":  round(elo_global.get(p, ELO_BASE), 1),
            "Hard":    round(elo_surface["Hard"].get(p, ELO_BASE), 1),
            "Clay":    round(elo_surface["Clay"].get(p, ELO_BASE), 1),
            "Grass":   round(elo_surface["Grass"].get(p, ELO_BASE), 1),
            "matches": matches_cnt.get(p, 0),
            "last":    last_date.get(p, ""),
        }
    return result

def get_elo_ratings():
    """Retourne le dict ELO (depuis cache session ou recalcul)."""
    if "elo_ratings" not in st.session_state:
        with st.spinner("Calcul des cotes ELO depuis l'historique..."):
            st.session_state["elo_ratings"] = compute_elo_from_csv()
    return st.session_state["elo_ratings"]

def elo_proba(p1, p2, surface):
    """
    Probabilité basée sur l'ELO.
    Combine ELO global (30%) + ELO surface (70%).
    """
    elo = get_elo_ratings()
    if p1 not in elo or p2 not in elo:
        return None
    d1 = elo[p1]; d2 = elo[p2]
    # ELO global
    p_global  = _elo_expected(d1["global"], d2["global"])
    # ELO surface
    p_surface = _elo_expected(d1.get(surface, ELO_BASE), d2.get(surface, ELO_BASE))
    # Mix pondéré
    p = 0.30 * p_global + 0.70 * p_surface
    return max(0.05, min(0.95, p))

def elo_diff_info(p1, p2, surface):
    """Retourne texte info ELO pour affichage."""
    elo = get_elo_ratings()
    if p1 not in elo or p2 not in elo:
        return None, None, None
    d1, d2 = elo[p1], elo[p2]
    diff = d1.get(surface, ELO_BASE) - d2.get(surface, ELO_BASE)
    return round(d1.get(surface, ELO_BASE)), round(d2.get(surface, ELO_BASE)), round(diff)

# ═══════════════════════════════════════════════════════════════
# SCORE DE MOMENTUM (forme récente pondérée)
# Calcule un score 0-1 sur les N derniers matchs dans les CSV
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def compute_momentum():
    """
    Pour chaque joueur, calcule un score de momentum (0-1) basé
    sur les 10 derniers matchs avec décroissance exponentielle.
    """
    if not DATA_DIR.exists():
        return {}

    all_rows = []
    for f in sorted(DATA_DIR.glob("*.csv")):
        if "wta" in f.name.lower():
            continue
        try:
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(f, encoding=enc, on_bad_lines="skip",
                                     usecols=lambda c: c in ["winner_name","loser_name","tourney_date"])
                    all_rows.append(df)
                    break
                except Exception:
                    continue
        except Exception:
            continue

    if not all_rows:
        return {}

    df = pd.concat(all_rows, ignore_index=True)
    for col in ["winner_name", "loser_name"]:
        df[col] = df[col].astype(str).str.strip()
    if "tourney_date" in df.columns:
        df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["tourney_date"]).sort_values("tourney_date")

    # Derniers matchs par joueur (résultat = 1 si gagné, 0 si perdu)
    from collections import defaultdict
    results = defaultdict(list)
    for _, row in df.iterrows():
        w = row.get("winner_name",""); l = row.get("loser_name","")
        if w and w != "nan": results[w].append(1)
        if l and l != "nan": results[l].append(0)

    N     = 10
    decay = 0.85  # les matchs récents comptent plus
    momentum = {}
    for player, res in results.items():
        last = res[-N:]  # 10 derniers (plus récents en dernier)
        if not last:
            continue
        weights     = [decay ** (len(last) - 1 - i) for i in range(len(last))]
        total_w     = sum(weights)
        score       = sum(r * w for r, w in zip(last, weights)) / total_w
        win_streak  = 0
        for r in reversed(last):
            if r == 1: win_streak += 1
            else:       break
        # Bonus série victoires
        score = min(1.0, score + win_streak * 0.03)
        momentum[player] = round(score, 4)

    return momentum

def get_momentum():
    if "momentum_cache" not in st.session_state:
        st.session_state["momentum_cache"] = compute_momentum()
    return st.session_state["momentum_cache"]

def momentum_diff(p1, p2):
    """Retourne (score_p1, score_p2, avantage_p1_en_proba)."""
    m = get_momentum()
    s1 = m.get(p1, 0.5)
    s2 = m.get(p2, 0.5)
    # Conversion diff → avantage probabiliste (max ±8%)
    diff = (s1 - s2) * 0.16
    return round(s1, 3), round(s2, 3), diff

# ═══════════════════════════════════════════════════════════════
# H2H PONDÉRÉ (H2H récent compte plus)
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def load_h2h_full():
    """Charge les CSV avec date pour H2H pondéré."""
    if not DATA_DIR.exists():
        return pd.DataFrame()
    dfs = []
    for f in sorted(DATA_DIR.glob("*.csv")):
        if "wta" in f.name.lower():
            continue
        try:
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(f, encoding=enc, on_bad_lines="skip",
                                     usecols=lambda c: c in ["winner_name","loser_name","tourney_date","surface"])
                    df["winner_name"] = df["winner_name"].astype(str).str.strip()
                    df["loser_name"]  = df["loser_name"].astype(str).str.strip()
                    if "tourney_date" in df.columns:
                        df["tourney_date"] = pd.to_datetime(
                            df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
                    dfs.append(df)
                    break
                except Exception:
                    continue
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def get_h2h(p1, p2, surface=None):
    """H2H avec pondération temporelle et optionnellement filtré par surface."""
    df = load_h2h_full()
    if df.empty:
        return None
    mask = (((df.winner_name == p1) & (df.loser_name == p2)) |
            ((df.winner_name == p2) & (df.loser_name == p1)))
    h = df[mask].copy()
    if len(h) == 0:
        return None

    # H2H sur cette surface
    h_surf = h[h["surface"] == surface] if surface and "surface" in h.columns else pd.DataFrame()

    # Pondération temporelle : matchs récents × 2, plus de 3 ans × 0.5
    now = pd.Timestamp.now()
    weights = []
    if "tourney_date" in h.columns:
        for dt in h["tourney_date"]:
            if pd.isna(dt):
                w = 1.0
            else:
                days = (now - dt).days
                if days < 365:     w = 2.0
                elif days < 730:   w = 1.5
                elif days < 1095:  w = 1.0
                else:              w = 0.5
            weights.append(w)
    else:
        weights = [1.0] * len(h)

    p1_wins = sum(w for w, row in zip(weights, h.itertuples()) if row.winner_name == p1)
    p2_wins = sum(w for w, row in zip(weights, h.itertuples()) if row.winner_name == p2)
    total_w = p1_wins + p2_wins

    return {
        "total":      len(h),
        "p1_wins":    len(h[h.winner_name == p1]),
        "p2_wins":    len(h[h.winner_name == p2]),
        "p1_wins_w":  round(p1_wins, 2),
        "p2_wins_w":  round(p2_wins, 2),
        "total_w":    round(total_w, 2),
        "surf_total": len(h_surf),
        "surf_p1":    len(h_surf[h_surf.winner_name == p1]) if not h_surf.empty else 0,
        "surf_p2":    len(h_surf[h_surf.winner_name == p2]) if not h_surf.empty else 0,
    }

def h2h_proba(h2h, p1):
    """Probabilité H2H pondérée."""
    if not h2h or h2h["total_w"] == 0:
        return 0.5
    return min(0.9, max(0.1, h2h["p1_wins_w"] / h2h["total_w"]))

# ═══════════════════════════════════════════════════════════════
# ENSEMBLE DE MODÈLES — FUSION RF + ELO + MOMENTUM + H2H
# ═══════════════════════════════════════════════════════════════
def logit(p):
    p = max(0.001, min(0.999, p))
    return math.log(p / (1 - p))

def inv_logit(x):
    return 1.0 / (1.0 + math.exp(-x))

def ensemble_proba(p1, p2, surface, tournament, h2h_data, mi):
    """
    Combine plusieurs signaux en log-odds pour une proba calibrée.

    Signaux utilisés :
      - Modèle RF 21 features   (poids 45%)
      - ELO surface dynamique   (poids 30%)
      - Momentum récent         (poids 15%)
      - H2H pondéré             (poids 10%)

    Si un signal est absent, son poids est redistribué.
    Retourne (proba, détails_dict, sources_utilisées)
    """
    details  = {}
    log_odds = 0.0
    total_w  = 0.0

    # ── Signal 1 : Modèle RF ─────────────────────────────────
    rf_p = None
    if mi:
        ratio = h2h_proba(h2h_data, p1) if h2h_data else 0.5
        rf_p, rf_status = predict_rf(p1, p2, surface, tournament, ratio, mi)
    if rf_p is not None:
        w = 0.45
        log_odds += w * logit(rf_p)
        total_w  += w
        details["RF"] = {"proba": round(rf_p, 4), "weight": w, "status": "ok"}
    else:
        details["RF"] = {"proba": None, "weight": 0, "status": "absent"}

    # ── Signal 2 : ELO surface ───────────────────────────────
    elo_p = elo_proba(p1, p2, surface)
    if elo_p is not None:
        w = 0.30
        log_odds += w * logit(elo_p)
        total_w  += w
        e1, e2, ediff = elo_diff_info(p1, p2, surface)
        details["ELO"] = {"proba": round(elo_p, 4), "weight": w,
                           "elo_p1": e1, "elo_p2": e2, "diff": ediff}
    else:
        details["ELO"] = {"proba": None, "weight": 0}

    # ── Signal 3 : Momentum ──────────────────────────────────
    m1, m2, mom_diff = momentum_diff(p1, p2)
    mom_p = max(0.05, min(0.95, 0.5 + mom_diff))
    w_mom = 0.15
    log_odds += w_mom * logit(mom_p)
    total_w  += w_mom
    details["Momentum"] = {"proba": round(mom_p, 4), "weight": w_mom,
                            "score_p1": m1, "score_p2": m2}

    # ── Signal 4 : H2H pondéré ───────────────────────────────
    h2h_p = h2h_proba(h2h_data, p1)
    if h2h_data and h2h_data["total"] >= 2:
        w = 0.10
        log_odds += w * logit(h2h_p)
        total_w  += w
        details["H2H"] = {"proba": round(h2h_p, 4), "weight": w,
                           "total": h2h_data["total"]}
    else:
        details["H2H"] = {"proba": round(h2h_p, 4), "weight": 0, "total": 0}

    # ── Normalisation si total_w < 1 ─────────────────────────
    if total_w == 0:
        return 0.5, details, []
    log_odds_norm = log_odds / total_w
    proba = inv_logit(log_odds_norm)
    proba = max(0.05, min(0.95, proba))

    sources = [k for k, v in details.items() if v.get("weight", 0) > 0]
    return round(proba, 4), details, sources

# ═══════════════════════════════════════════════════════════════
# CONFIANCE MULTI-FACTEURS
# ═══════════════════════════════════════════════════════════════
def calc_confidence_v2(proba, details, h2h_data):
    """
    Score de confiance 0-100 basé sur :
    - Écart de proba par rapport à 0.5
    - Nombre de sources d'accord
    - Quantité de données H2H
    - Cohérence entre les signaux
    - ELO diff
    """
    score = 40.0

    # Écart par rapport à 0.5 → max +25
    score += abs(proba - 0.5) * 50

    # Nombre de sources actives → max +10
    active = sum(1 for v in details.values() if v.get("weight", 0) > 0)
    score += active * 2.5

    # Cohérence entre les signaux → pénalité si divergence
    proba_vals = [v["proba"] for v in details.values()
                  if v.get("proba") is not None and v.get("weight", 0) > 0]
    if len(proba_vals) >= 2:
        std = np.std(proba_vals)
        # std élevé = signaux divergents → pénalité
        score -= std * 40

    # H2H data
    if h2h_data and h2h_data.get("total", 0) >= 5:
        score += 8
    elif h2h_data and h2h_data.get("total", 0) >= 2:
        score += 4

    # ELO diff important → signe clair
    elo_d = details.get("ELO", {})
    if elo_d.get("diff") is not None:
        score += min(8, abs(elo_d["diff"]) / 50)

    return round(min(100.0, max(10.0, score)), 1)

# ═══════════════════════════════════════════════════════════════
# CRITÈRE DE KELLY — DIMENSIONNEMENT DE LA MISE
# ═══════════════════════════════════════════════════════════════
def kelly_fraction(proba, cote, fraction=0.25):
    """
    Calcule la fraction Kelly.
    fraction=0.25 = Kelly fractionné (recommandé pour limiter le risque).
    Retourne 0 si pas de value bet.
    """
    if cote <= 1.0 or proba <= 0.0:
        return 0.0
    b   = cote - 1.0        # profit net pour 1 misé
    q   = 1.0 - proba
    kf  = (b * proba - q) / b
    if kf <= 0:
        return 0.0
    return round(kf * fraction, 4)  # Kelly fractionné

def compute_value_bets(p1, p2, proba, o1_str, o2_str):
    """
    Calcule les value bets avec edge, Kelly, et classification qualité.
    Retourne (best_val_dict_or_None, analyse_dict)
    """
    best_val = None
    analyse  = {}

    o1f = _safe_float(str(o1_str).replace(",","."))
    o2f = _safe_float(str(o2_str).replace(",","."))
    if o1f <= 1.0 or o2f <= 1.0:
        return None, {}

    # Probabilité implicite des cotes (marges comprises)
    impl1 = 1.0 / o1f
    impl2 = 1.0 / o2f
    marge = impl1 + impl2 - 1.0  # marge bookmaker

    e1 = proba - impl1
    e2 = (1 - proba) - impl2

    kf1 = kelly_fraction(proba,        o1f)
    kf2 = kelly_fraction(1 - proba,    o2f)

    def quality(edge):
        if edge >= 0.08: return "A", "#00DFA2", "Excellent"
        if edge >= 0.05: return "B", "#0079FF", "Bon"
        if edge >= 0.02: return "C", "#FFB200", "Acceptable"
        return "D", "#FF4757", "Faible"

    for player, edge, cote, proba_b, kf in [
        (p1, e1, o1f, proba,     kf1),
        (p2, e2, o2f, 1-proba,   kf2),
    ]:
        grade, color, label = quality(edge)
        analyse[player] = {
            "edge": round(edge, 4), "cote": cote,
            "proba": round(proba_b, 4), "implied": round(1/cote, 4),
            "kelly": kf, "grade": grade, "color": color, "label": label,
            "marge": round(marge, 4),
        }

    best = max([(e1, p1, kf1, o1f, proba),
                (e2, p2, kf2, o2f, 1-proba)], key=lambda x: x[0])
    if best[0] > MIN_EDGE_COMBINE:
        best_val = {
            "joueur": best[1], "edge": best[0], "cote": best[3],
            "proba":  best[4], "kelly": best[2],
        }
    return best_val, analyse

# ═══════════════════════════════════════════════════════════════
# MODÈLE RF — inchangé
# ═══════════════════════════════════════════════════════════════
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
            with st.spinner("Telechargement du modele RF..."):
                url = ("https://github.com/Xela91300/sports-betting-neural-net"
                       "/releases/latest/download/tennis_ml_model_complete.pkl.gz")
                r = requests.get(url, timeout=60)
                if r.status_code == 200:
                    tmp = MODELS_DIR / "model_temp.pkl.gz"
                    tmp.write_bytes(r.content)
                    with gzip.open(tmp, "rb") as gz:
                        model_info = joblib.load(gz)
                    joblib.dump(model_info, model_path)
                    tmp.unlink(missing_ok=True)
        except Exception as e:
            st.warning("Modele non telecharge: " + str(e))
    st.session_state["rf_model_cache"] = model_info
    return model_info

def load_model_metadata():
    if "model_metadata_cache" in st.session_state:
        return st.session_state["model_metadata_cache"]
    result = {}
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE) as fh:
                result = json.load(fh)
        except Exception:
            pass
    st.session_state["model_metadata_cache"] = result
    return result

def extract_21_features(ps, p1, p2, surface, level="A", best_of=3, h2h_r=0.5):
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
        float(h2h_r),
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

def predict_rf(p1, p2, surface, tournament, h2h_r, mi):
    if mi is None: return None, "absent"
    try:
        m, sc, ps = mi.get("model"), mi.get("scaler"), mi.get("player_stats", {})
        if m is None or sc is None: return None, "incomplet"
        if p1 not in ps or p2 not in ps: return None, "joueurs_inconnus"
        lv, bo = get_level(tournament)
        f = extract_21_features(ps, p1, p2, surface, lv, bo, h2h_r)
        p = float(m.predict_proba(sc.transform(f.reshape(1, -1)))[0][1])
        return max(0.05, min(0.95, p)), "ok"
    except Exception as e:
        return None, str(e)[:30]

# ═══════════════════════════════════════════════════════════════
# DONNÉES CSV
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def load_players():
    if not DATA_DIR.exists(): return []
    players = set()
    for f in DATA_DIR.glob("*.csv"):
        if "wta" in f.name.lower(): continue
        try:
            for enc in ["utf-8","latin-1","cp1252"]:
                try:
                    df = pd.read_csv(f, encoding=enc,
                                     usecols=["winner_name","loser_name"],
                                     on_bad_lines="skip")
                    players.update(df["winner_name"].dropna().astype(str).str.strip())
                    players.update(df["loser_name"].dropna().astype(str).str.strip())
                    break
                except Exception: continue
        except Exception: pass
    return sorted(p for p in players if p and p.lower() != "nan" and len(p) > 1)

# ═══════════════════════════════════════════════════════════════
# HISTORIQUE & STATS
# ═══════════════════════════════════════════════════════════════
def load_history():
    if not HIST_FILE.exists(): return []
    try:
        with open(HIST_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return []

def save_pred(pred):
    try:
        h = load_history()
        pred["id"] = hashlib.md5((str(datetime.now())+pred.get("player1","")).encode()).hexdigest()[:8]
        pred["statut"] = "en_attente"
        pred["vainqueur_reel"] = None
        pred["pronostic_correct"] = None
        h.append(pred)
        with open(HIST_FILE, "w", encoding="utf-8") as f:
            json.dump(h[-1000:], f, indent=2, ensure_ascii=False)
        return True
    except Exception: return False

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
    except Exception: return False

def load_user_stats():
    default = {"total_predictions":0,"correct_predictions":0,
               "incorrect_predictions":0,"annules_predictions":0,
               "current_streak":0,"best_streak":0}
    if not USER_STATS_FILE.exists(): return default
    try:
        with open(USER_STATS_FILE) as f: return json.load(f)
    except Exception: return default

def update_stats():
    h = load_history()
    correct   = sum(1 for p in h if p.get("statut")=="gagne")
    incorrect = sum(1 for p in h if p.get("statut")=="perdu")
    cancel    = sum(1 for p in h if p.get("statut")=="annule")
    streak = cur = best = 0
    for p in reversed(h):
        if p.get("statut")=="gagne":   streak+=1; cur=streak; best=max(best,streak)
        elif p.get("statut")=="perdu": streak=0;  cur=0
    stats = {"total_predictions":len(h),"correct_predictions":correct,
             "incorrect_predictions":incorrect,"annules_predictions":cancel,
             "current_streak":cur,"best_streak":best}
    with open(USER_STATS_FILE,"w") as f: json.dump(stats,f)
    return stats

def calc_accuracy():
    s = load_user_stats()
    tv = s.get("correct_predictions",0)+s.get("incorrect_predictions",0)
    return (s.get("correct_predictions",0)/tv*100) if tv>0 else 0

# ═══════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════
def get_tg_config():
    try: return st.secrets["TELEGRAM_BOT_TOKEN"], str(st.secrets["TELEGRAM_CHAT_ID"])
    except Exception:
        t = os.environ.get("TELEGRAM_BOT_TOKEN")
        c = os.environ.get("TELEGRAM_CHAT_ID")
        return (t,c) if t and c else (None,None)

def tg_send(message, parse_mode="HTML"):
    token, chat_id = get_tg_config()
    if not token or not chat_id: return False, "Telegram non configure"
    message = str(message)
    if len(message) > 4000: message = message[:3990] + "\n..."
    try:
        r = requests.post(
            "https://api.telegram.org/bot" + token + "/sendMessage",
            json={"chat_id":chat_id,"text":message,"parse_mode":parse_mode,
                  "disable_web_page_preview":True}, timeout=15)
        if r.status_code == 200: return True, "Envoye sur Telegram"
        err = r.json().get("description", r.text[:100])
        if "parse" in err.lower():
            plain = message.replace("<b>","").replace("</b>","").replace("<i>","").replace("</i>","")
            r2 = requests.post(
                "https://api.telegram.org/bot" + token + "/sendMessage",
                json={"chat_id":chat_id,"text":plain,"disable_web_page_preview":True}, timeout=15)
            if r2.status_code == 200: return True, "Envoye (texte brut)"
        return False, "Telegram: " + err
    except Exception as e: return False, "Erreur: " + str(e)[:60]

def format_pred_msg(pred, ai_txt=None):
    proba = _safe_float(pred.get("proba"), 0.5)
    bar   = chr(9608)*int(proba*10) + chr(9617)*(10-int(proba*10))
    fav   = pred.get("favori","?")
    bv    = pred.get("best_value")
    msg   = (
        "<b>[ML+ELO] PREDICTION TENNISIQ</b>\n\n"
        "<b>" + pred.get("player1","?") + " vs " + pred.get("player2","?") + "</b>\n"
        + pred.get("tournament","?") + " | " + pred.get("surface","?") + "\n\n"
        "<code>" + bar + "</code>\n"
        + pred.get("player1","J1") + ": <b>" + str(round(proba*100,1)) + "%</b>\n"
        + pred.get("player2","J2") + ": <b>" + str(round((1-proba)*100,1)) + "%</b>\n\n"
        "FAVORI: <b>" + fav + "</b>  Confiance: <b>"
        + str(int(pred.get("confidence",50))) + "/100</b>\n"
        "Sources: " + str(pred.get("sources","?"))
    )
    if pred.get("odds1") and pred.get("odds2"):
        msg += "\nCotes: " + str(pred["odds1"]) + " / " + str(pred["odds2"])
    if bv:
        kf = _safe_float(bv.get("kelly"),0)
        msg += ("\n\nVALUE BET - MISER SUR <b>" + str(bv.get("joueur","?")).upper() + "</b>\n"
                "Cote: <b>" + str(round(_safe_float(bv.get("cote")),2)) + "</b>"
                "  Edge: <b>+" + str(round(_safe_float(bv.get("edge"))*100,1)) + "%</b>"
                + ("  Kelly: <b>" + str(round(kf*100,1)) + "% bankroll</b>" if kf>0 else ""))
    if ai_txt:
        msg += "\n\nAnalyse IA:\n" + str(ai_txt)[:500]
    msg += "\n\n#TennisIQ"
    return msg

def format_stats_msg():
    s=load_user_stats(); h=load_history()
    c=s.get("correct_predictions",0); w=s.get("incorrect_predictions",0)
    tv=c+w; acc=(c/tv*100) if tv>0 else 0
    recent=[p for p in h[-20:] if p.get("statut") in ["gagne","perdu"]]
    r_acc=(sum(1 for p in recent if p.get("statut")=="gagne")/len(recent)*100) if recent else 0
    return ("<b>STATS TENNISIQ</b>\n"
            "Precision: <b>"+str(round(acc,1))+"%</b>  "
            "Forme: <b>"+str(round(r_acc,1))+"%</b>\n"
            "V:"+str(c)+" D:"+str(w)+" A:"+str(s.get("annules_predictions",0))+"\n"
            "Serie: <b>"+str(s.get("current_streak",0))+"</b>  "
            "Record: <b>"+str(s.get("best_streak",0))+"</b>\n"
            + datetime.now().strftime("%d/%m/%Y %H:%M") + " #TennisIQ")

# ═══════════════════════════════════════════════════════════════
# IA ANALYSIS (Groq / DeepSeek / Claude)
# ═══════════════════════════════════════════════════════════════
def get_groq_key():
    try: return st.secrets["GROQ_API_KEY"]
    except Exception: return os.environ.get("GROQ_API_KEY")
def get_deepseek_key():
    try: return st.secrets["DEEPSEEK_API_KEY"]
    except Exception: return os.environ.get("DEEPSEEK_API_KEY")
def get_claude_key():
    try: return st.secrets["ANTHROPIC_API_KEY"]
    except Exception: return os.environ.get("ANTHROPIC_API_KEY")

def call_groq(prompt):
    key = get_groq_key()
    if not key: return None
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},
            json={"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":prompt}],
                  "temperature":0.3,"max_tokens":600}, timeout=30)
        return r.json()["choices"][0]["message"]["content"] if r.status_code==200 else None
    except Exception: return None

def call_deepseek(prompt):
    key = get_deepseek_key()
    if not key: return None
    try:
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},
            json={"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],
                  "temperature":0.3,"max_tokens":600}, timeout=30)
        return r.json()["choices"][0]["message"]["content"] if r.status_code==200 else None
    except Exception: return None

def call_claude_api(prompt):
    key = get_claude_key()
    if not key: return None
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"},
            json={"model":"claude-3-haiku-20240307","max_tokens":600,
                  "messages":[{"role":"user","content":prompt}]}, timeout=30)
        return r.json()["content"][0]["text"] if r.status_code==200 else None
    except Exception: return None

def ai_analysis(p1, p2, surface, tournament, proba, details, best_value, ia_choice):
    """Prompt enrichi avec données ELO, momentum, H2H, edge."""
    fav  = p1 if proba >= 0.5 else p2
    und  = p2 if proba >= 0.5 else p1

    elo_info = ""
    elo_d = details.get("ELO",{})
    if elo_d.get("elo_p1"):
        elo_info = ("ELO surface: " + p1 + " " + str(elo_d["elo_p1"])
                    + " vs " + p2 + " " + str(elo_d["elo_p2"])
                    + " (diff " + str(elo_d["diff"]) + ")")

    mom_d = details.get("Momentum",{})
    mom_info = ""
    if mom_d.get("score_p1") is not None:
        mom_info = ("Momentum: " + p1 + " " + str(round(mom_d["score_p1"]*100,0)) + "%"
                    + " vs " + p2 + " " + str(round(mom_d["score_p2"]*100,0)) + "%")

    h2h_d = details.get("H2H",{})
    h2h_info = ("H2H: " + str(h2h_d.get("total",0)) + " matchs"
                + (" - " + p1 + " domine" if _safe_float(h2h_d.get("proba")) > 0.55 else "")) if h2h_d.get("total",0) > 0 else "H2H: aucun"

    vb_str = ""
    if best_value:
        vb_str = ("VALUE BET DETECTE: " + str(best_value.get("joueur","?"))
                  + " @ " + str(round(_safe_float(best_value.get("cote")),2))
                  + " Edge+" + str(round(_safe_float(best_value.get("edge"))*100,1)) + "%"
                  + " Kelly=" + str(round(_safe_float(best_value.get("kelly"))*100,1)) + "% bankroll")

    prompt = (
        "Tu es un expert analyste tennis ATP. Analyse ce match en francais, 4 points:\n\n"
        + p1 + " vs " + p2 + " | " + tournament + " | " + surface + "\n"
        "Proba ML+ELO: " + p1 + " " + str(round(proba*100,1)) + "% | "
        + p2 + " " + str(round((1-proba)*100,1)) + "%\n"
        "FAVORI: " + fav + "\n"
        + elo_info + "\n"
        + mom_info + "\n"
        + h2h_info + "\n"
        + (vb_str + "\n" if vb_str else "")
        + "\n1. Pourquoi " + fav + " est favori (arguments ELO, forme, surface)\n"
        "2. Risques et points faibles de " + fav + "\n"
        "3. RECOMMANDATION PARI: " + (vb_str if vb_str else "pari optimal") + "\n"
        "4. Pronostic final + confiance (etoiles 1-5) + % bankroll conseille\n\n"
        "Sois factuel et concis."
    )
    if ia_choice == "Groq":     return call_groq(prompt)
    if ia_choice == "DeepSeek": return call_deepseek(prompt)
    if ia_choice == "Claude":   return call_claude_api(prompt)
    return None

# ═══════════════════════════════════════════════════════════════
# ACHIEVEMENTS
# ═══════════════════════════════════════════════════════════════
def load_ach():
    if not ACHIEVEMENTS_FILE.exists(): return {}
    try:
        with open(ACHIEVEMENTS_FILE) as f: return json.load(f)
    except Exception: return {}

def save_ach(a):
    try:
        with open(ACHIEVEMENTS_FILE,"w") as f: json.dump(a,f)
    except Exception: pass

def check_achievements():
    s=load_user_stats(); h=load_history(); a=load_ach(); new=[]
    for aid, cond in [
        ("first_win",s.get("correct_predictions",0)>=1),
        ("streak_5", s.get("best_streak",0)>=5),
        ("streak_10",s.get("best_streak",0)>=10),
        ("pred_100", s.get("total_predictions",0)>=100),
    ]:
        if cond and aid not in a:
            a[aid]={"unlocked_at":datetime.now().isoformat()}; new.append(ACHIEVEMENTS[aid])
    vw=sum(1 for p in h if p.get("best_value") and p.get("statut")=="gagne")
    if vw>=10 and "value_master" not in a:
        a["value_master"]={"unlocked_at":datetime.now().isoformat()}; new.append(ACHIEVEMENTS["value_master"])
    surfs={p.get("surface") for p in h if p.get("statut")=="gagne"}
    if len(surfs)>=3 and "surface_specialist" not in a:
        a["surface_specialist"]={"unlocked_at":datetime.now().isoformat()}; new.append(ACHIEVEMENTS["surface_specialist"])
    if new: save_ach(a)
    return new

def backup():
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    for f in [HIST_FILE,USER_STATS_FILE]:
        if f.exists():
            try: shutil.copy(f,BACKUP_DIR/(f.stem+"_"+ts+f.suffix))
            except Exception: pass

# ═══════════════════════════════════════════════════════════════
# SÉLECTEURS
# ═══════════════════════════════════════════════════════════════
def player_sel(label, all_players, key, default=None):
    search = st.text_input("Rechercher " + label, key="srch_"+key, placeholder="Tapez un nom...")
    filtered = [p for p in all_players if search.lower() in p.lower()] if search else all_players[:200]
    if not filtered and search:
        filtered = [p for p in all_players if p and p[0].lower()==search[0].lower()][:50]
    st.caption(str(len(filtered))+" / "+str(len(all_players)))
    if not filtered: return st.text_input(label, key=key)
    idx=0
    if default:
        for i,p in enumerate(filtered):
            if default.lower() in p.lower(): idx=i; break
    return st.selectbox(label, filtered, index=idx, key=key)

def tourn_sel(label, key, default=None):
    search = st.text_input("Rechercher "+label, key="srcht_"+key, placeholder="ex: Roland Garros...")
    all_t  = sorted(TOURNAMENTS_DB.keys())
    if search:
        sl  = search.lower().strip()
        res = set()
        if sl in TOURNAMENT_ALIASES: res.add(TOURNAMENT_ALIASES[sl])
        for t in all_t:
            if sl in t.lower(): res.add(t)
        filtered = sorted(res) if res else all_t[:50]
    else:
        filtered = all_t[:100]
    idx = filtered.index(default) if default and default in filtered else 0
    return st.selectbox(label, filtered, index=idx, key=key)

# ═══════════════════════════════════════════════════════════════
# PAGE : DASHBOARD
# ═══════════════════════════════════════════════════════════════
def show_dashboard():
    st.markdown(section_title("Dashboard", "Vue d ensemble"), unsafe_allow_html=True)
    stats=load_user_stats(); h=load_history(); a=load_ach()
    mi=load_rf_model(); metadata=load_model_metadata()
    correct=stats.get("correct_predictions",0); wrong=stats.get("incorrect_predictions",0)
    cancel=stats.get("annules_predictions",0)
    pending=len([p for p in h if p.get("statut")=="en_attente"])
    tv=correct+wrong; acc=(correct/tv*100) if tv>0 else 0
    recent=[p for p in h[-20:] if p.get("statut") in ["gagne","perdu"]]
    r_acc=(sum(1 for p in recent if p.get("statut")=="gagne")/len(recent)*100) if recent else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(big_metric("PRECISION",str(round(acc,1))+"%",r_acc-acc if tv>0 else None), unsafe_allow_html=True)
    with c2: st.markdown(big_metric("GAGNES",str(correct),color="#00DFA2"), unsafe_allow_html=True)
    with c3: st.markdown(big_metric("PERDUS",str(wrong),color="#FF4757"), unsafe_allow_html=True)
    with c4: st.markdown(big_metric("ABANDONS",str(cancel),color="#FFB200"), unsafe_allow_html=True)
    with c5: st.markdown(big_metric("EN ATTENTE",str(pending),color="#7A8599"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1,2])
    with col_l:
        streak=stats.get("current_streak",0); best=stats.get("best_streak",0)
        sc="#00DFA2" if streak>0 else "#7A8599"
        st.markdown(
            "<div style='background:rgba(255,255,255,0.04);border:1px solid "+sc+"44;"
            "border-radius:16px;padding:1.5rem;text-align:center;'>"
            "<div style='font-family:Syne,sans-serif;font-size:3rem;font-weight:800;color:"+sc+";'>"
            +str(streak)+"</div>"
            "<div style='color:#6C7A89;font-size:0.85rem;text-transform:uppercase;'>Serie actuelle</div>"
            "<div style='margin-top:0.75rem;padding-top:0.75rem;border-top:1px solid rgba(255,255,255,0.10);'>"
            "<span style='color:#6C7A89;'>Record: </span>"
            "<span style='color:#FFB200;font-weight:700;'>"+str(best)+"</span></div></div>",
            unsafe_allow_html=True)

    with col_r:
        tg_token,_=get_tg_config(); groq_key=get_groq_key()
        deepseek_key=get_deepseek_key(); claude_key=get_claude_key()
        elo_count = len(get_elo_ratings())
        services=[]
        if mi:
            ps=mi.get("player_stats",{}); acc_m=mi.get("accuracy",metadata.get("accuracy",0))
            services.append(("Modele RF",str(round(acc_m*100,1))+"% acc - "+str(len(ps))+" joueurs",True))
        else: services.append(("Modele RF","Non charge",False))
        services.append(("ELO dynamique",str(elo_count)+" joueurs calcules",elo_count>0))
        services.append(("Groq","OK" if groq_key else "Non configure",bool(groq_key)))
        services.append(("DeepSeek","OK" if deepseek_key else "Non configure",bool(deepseek_key)))
        services.append(("Claude","OK" if claude_key else "Non configure",bool(claude_key)))
        services.append(("Telegram","OK" if tg_token else "Non configure",bool(tg_token)))
        rows=""
        for svc,desc,ok in services:
            col=("#00DFA2" if ok else "#FF4757"); dot=("ON" if ok else "OFF")
            rows+=("<div style='display:flex;align-items:center;gap:0.75rem;padding:0.4rem 0.75rem;"
                   "background:rgba(255,255,255,0.03);border-radius:8px;margin-bottom:0.3rem;'>"
                   "<span style='color:"+col+";font-size:0.7rem;font-weight:700;'>"+dot+"</span>"
                   "<span style='font-weight:600;color:#E8EDF5;flex:1;'>"+svc+"</span>"
                   "<span style='color:#6C7A89;font-size:0.8rem;'>"+desc+"</span></div>")
        st.markdown(
            "<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.10);"
            "border-radius:16px;padding:1.5rem;'>"
            "<div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;"
            "color:#E8EDF5;margin-bottom:0.75rem;'>STATUT DES SERVICES</div>"
            +rows+"</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    finished=[p for p in h if p.get("statut") in ["gagne","perdu"]]
    if len(finished)>=3:
        df_h=pd.DataFrame(finished)
        df_h["ok"]=(df_h["statut"]=="gagne").astype(int)
        df_h["cum_ok"]=df_h["ok"].expanding().sum()
        df_h["cum_n"]=range(1,len(df_h)+1)
        df_h["acc"]=df_h["cum_ok"]/df_h["cum_n"]*100
        df_h["n"]=range(1,len(df_h)+1)
        fig=go.Figure()
        fig.add_hline(y=50,line_dash="dot",line_color="rgba(255,255,255,0.15)")
        fig.add_trace(go.Scatter(x=df_h["n"],y=df_h["acc"],mode="lines",
                                  line=dict(color="#00DFA2",width=2.5),
                                  fill="tozeroy",fillcolor="rgba(0,223,162,0.07)"))
        fig.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=0),
                          paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#7A8599"),showlegend=False,
                          xaxis=dict(showgrid=False,title="Prediction #",color="#7A8599"),
                          yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.05)",
                                     title="Precision (%)",color="#7A8599",range=[0,100]))
        st.plotly_chart(fig,use_container_width=True)

    if st.button("Envoyer stats Telegram"):
        ok,msg=tg_send(format_stats_msg())
        st.success(msg) if ok else st.error(msg)

# ═══════════════════════════════════════════════════════════════
# PAGE : ANALYSE — CŒUR DE L'APPLICATION
# ═══════════════════════════════════════════════════════════════
def show_prediction():
    st.markdown(section_title("Analyse Multi-matchs",
                              "Ensemble RF + ELO + Momentum + H2H | Kelly criterion"),
                unsafe_allow_html=True)
    mi=load_rf_model(); metadata=load_model_metadata()

    if mi:
        ps=mi.get("player_stats",{}); acc_m=mi.get("accuracy",metadata.get("accuracy",0))
        st.markdown(
            "<div style='background:rgba(0,223,162,0.08);border:1px solid rgba(0,223,162,0.25);"
            "border-radius:12px;padding:0.75rem 1rem;margin-bottom:0.5rem;'>"
            "<span style='font-weight:700;color:#00DFA2;'>RF "+str(round(acc_m*100,1))+"% accuracy</span>"
            "<span style='color:#6C7A89;margin-left:0.75rem;'>"+str(len(ps))+" joueurs · 21 features</span></div>",
            unsafe_allow_html=True)
    else:
        st.warning("Modele RF absent — ELO + Momentum actifs")

    elo_count=len(get_elo_ratings())
    st.markdown(
        "<div style='background:rgba(0,121,255,0.08);border:1px solid rgba(0,121,255,0.25);"
        "border-radius:12px;padding:0.75rem 1rem;margin-bottom:1rem;'>"
        "<span style='font-weight:700;color:#0079FF;'>ELO dynamique</span>"
        "<span style='color:#6C7A89;margin-left:0.75rem;'>"+str(elo_count)+" joueurs · par surface · pondéré</span></div>",
        unsafe_allow_html=True)

    with st.spinner("Chargement des joueurs..."):
        all_p=load_players()

    with st.sidebar:
        st.markdown("### Parametres")
        n=st.slider("Nombre de matchs",1,MAX_MATCHES,2)
        ia_opts=["Aucune"]
        if get_groq_key():    ia_opts.append("Groq")
        if get_deepseek_key(): ia_opts.append("DeepSeek")
        if get_claude_key():   ia_opts.append("Claude")
        ia_choice=st.selectbox("IA",ia_opts,index=min(1,len(ia_opts)-1))
        send_tg=st.checkbox("Envoi Telegram auto",False)
        show_details=st.checkbox("Afficher details ensemble",True)

    inputs=[]
    for i in range(n):
        with st.expander("Match "+str(i+1), expanded=(i==0)):
            ct,cs=st.columns([3,1])
            with ct: tourn=tourn_sel("Tournoi","t"+str(i),"Roland Garros")
            with cs:
                surf=get_surface(tourn); lv,bo=get_level(tourn)
                cfg=SURFACE_CFG[surf]
                st.markdown(
                    "<div style='background:"+cfg["bg"]+";border:1px solid "+cfg["color"]+"55;"
                    "border-radius:10px;padding:0.6rem;text-align:center;margin-top:1.75rem;'>"
                    "<div style='font-weight:700;color:"+cfg["color"]+";'>"+surf+"</div>"
                    +("<div style='font-size:0.7rem;color:#7A8599;'>Best of 5</div>" if bo==5 else "")
                    +"</div>", unsafe_allow_html=True)
            cp1,cp2=st.columns(2)
            with cp1:
                p1=player_sel("Joueur 1",all_p,"p1_"+str(i))
                o1=st.text_input("Cote "+(p1[:15] if p1 else "J1"),key="o1_"+str(i),placeholder="1.75")
            with cp2:
                p2_list=[p for p in all_p if p!=p1]
                p2=player_sel("Joueur 2",p2_list,"p2_"+str(i))
                o2=st.text_input("Cote "+(p2[:15] if p2 else "J2"),key="o2_"+str(i),placeholder="2.10")

            # Preview ELO en temps réel
            if p1 and p2:
                e1,e2,ediff=elo_diff_info(p1,p2,surf)
                if e1 and e2:
                    st.caption("ELO "+surf+": "+p1[:15]+" "+str(e1)+" vs "+p2[:15]+" "+str(e2)
                               +" (diff "+str(ediff)+")")
            inputs.append({"p1":p1,"p2":p2,"surf":surf,"tourn":tourn,"o1":o1,"o2":o2})

    if not st.button("Analyser",type="primary",use_container_width=True):
        return

    valid=[m for m in inputs if m["p1"] and m["p2"]]
    if not valid: st.warning("Remplis au moins un match"); return

    st.markdown("---")
    st.markdown(section_title("Resultats"), unsafe_allow_html=True)

    for i,m in enumerate(valid):
        p1,p2,surf,tourn=m["p1"],m["p2"],m["surf"],m["tourn"]

        # ── CALCUL ENSEMBLE ─────────────────────────────────
        h2h_data=get_h2h(p1,p2,surf)
        proba,details,sources=ensemble_proba(p1,p2,surf,tourn,h2h_data,mi)
        conf=calc_confidence_v2(proba,details,h2h_data)
        fav=p1 if proba>=0.5 else p2
        fav_p=max(proba,1-proba)
        cfg=SURFACE_CFG[surf]
        lv,bo=get_level(tourn)

        # Couleurs selon favori
        p1_col="#00DFA2" if fav==p1 else "#7A8599"
        p2_col="#00DFA2" if fav==p2 else "#7A8599"
        p1_bg ="rgba(0,223,162,0.07)" if fav==p1 else "transparent"
        p2_bg ="rgba(0,223,162,0.07)" if fav==p2 else "transparent"
        p1_tag=("<div style='color:#00DFA2;font-size:0.75rem;font-weight:700;'>FAVORI</div>" if fav==p1
                else "<div style='color:#7A8599;font-size:0.72rem;'>outsider</div>")
        p2_tag=("<div style='color:#00DFA2;font-size:0.75rem;font-weight:700;'>FAVORI</div>" if fav==p2
                else "<div style='color:#7A8599;font-size:0.72rem;'>outsider</div>")

        h2h_str=("H2H "+str(h2h_data["p1_wins"])+"-"+str(h2h_data["p2_wins"])
                 +" ("+str(h2h_data["total"])+")" if h2h_data else "H2H: aucun")

        # Barre de probabilité HTML
        pct=int(proba*100); pct2=100-pct
        bar_html=(
            "<div style='background:rgba(255,255,255,0.07);border-radius:100px;height:8px;overflow:hidden;margin:0.5rem 0;'>"
            "<div style='background:linear-gradient(90deg,"+cfg["color"]+",#0079FF);width:"+str(pct)+"%;height:100%;border-radius:100px;'></div></div>"
            "<div style='display:flex;justify-content:space-between;font-size:0.8rem;'>"
            "<span style='color:"+cfg["color"]+";font-weight:700;'>"+str(pct)+"%</span>"
            "<span style='color:#7A8599;'>"+str(pct2)+"%</span></div>"
        )

        st.markdown(
            "<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.10);"
            "border-radius:16px;padding:1.5rem;margin-bottom:1rem;'>"
            # Header
            "<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;'>"
            "<div><span style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;color:#E8EDF5;'>"
            "Match "+str(i+1)+"</span>"
            "<span style='margin-left:0.75rem;'>"+surface_badge(surf)+"</span>"
            "<span style='color:#6C7A89;font-size:0.85rem;margin-left:0.5rem;'>"+tourn+"</span></div>"
            "<span style='background:rgba(0,121,255,0.15);border:1px solid rgba(0,121,255,0.3);"
            "border-radius:100px;padding:0.2rem 0.6rem;font-size:0.72rem;font-weight:700;color:#0079FF;'>"
            +"+".join(sources)+"</span></div>"
            # Joueurs
            "<div style='display:grid;grid-template-columns:1fr auto 1fr;gap:1rem;align-items:center;'>"
            "<div style='text-align:center;background:"+p1_bg+";border-radius:12px;padding:0.75rem;'>"
            "<div style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:#E8EDF5;'>"+p1+"</div>"
            "<div style='font-size:2rem;font-weight:800;color:"+p1_col+";'>"+str(round(proba*100,1))+"%</div>"
            +p1_tag+"</div>"
            "<div style='text-align:center;color:#6C7A89;font-weight:700;font-size:1.4rem;'>VS</div>"
            "<div style='text-align:center;background:"+p2_bg+";border-radius:12px;padding:0.75rem;'>"
            "<div style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:#E8EDF5;'>"+p2+"</div>"
            "<div style='font-size:2rem;font-weight:800;color:"+p2_col+";'>"+str(round((1-proba)*100,1))+"%</div>"
            +p2_tag+"</div></div>"
            # Barre
            "<div style='margin:0.5rem 0;'>"
            "<div style='display:flex;justify-content:space-between;font-size:0.72rem;color:#6C7A89;margin-bottom:2px;'>"
            "<span>"+p1+"</span><span>"+p2+"</span></div>"
            +bar_html+"</div>"
            # Pills
            "<div style='display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.75rem;'>"
            +stat_pill("Confiance",str(int(conf))+"/100","#00DFA2" if conf>=70 else "#FFB200" if conf>=50 else "#FF4757")
            +stat_pill("H2H",h2h_str,"#0079FF")
            +stat_pill("Format","Bo"+str(bo),"#7A8599")
            +"</div></div>",
            unsafe_allow_html=True)

        # ── Détail des signaux ───────────────────────────────
        if show_details:
            with st.expander("Detail des signaux de prediction"):
                cols_d = st.columns(len([k for k,v in details.items() if v.get("weight",0)>0 or v.get("proba")]))
                col_idx=0
                for signal, d in details.items():
                    p_val=d.get("proba")
                    w    =d.get("weight",0)
                    if p_val is None and w==0: continue
                    with cols_d[col_idx]:
                        col_idx+=1
                        col_s="#00DFA2" if w>0 else "#6C7A89"
                        w_pct=str(int(w*100))+"%"
                        p_display="--" if p_val is None else str(round(p_val*100,1))+"%"
                        st.markdown(
                            "<div style='background:rgba(255,255,255,0.04);"
                            "border:1px solid rgba(255,255,255,0.10);"
                            "border-radius:12px;padding:0.75rem;text-align:center;'>"
                            "<div style='font-size:0.75rem;color:#6C7A89;text-transform:uppercase;'>"+signal+"</div>"
                            "<div style='font-size:1.4rem;font-weight:800;color:"+col_s+";'>"+p_display+"</div>"
                            "<div style='font-size:0.7rem;color:#6C7A89;'>Poids: "+w_pct+"</div>"
                            + (("<div style='font-size:0.65rem;color:#7A8599;margin-top:0.25rem;'>"
                                "ELO "+p1[:12]+":"+str(d.get("elo_p1","?"))+"<br>"
                                "ELO "+p2[:12]+":"+str(d.get("elo_p2","?"))+"</div>")
                               if signal=="ELO" and d.get("elo_p1") else "")
                            + (("<div style='font-size:0.65rem;color:#7A8599;margin-top:0.25rem;'>"
                                "Forme "+p1[:12]+":"+str(round(_safe_float(d.get("score_p1"))*100,0))+"%<br>"
                                "Forme "+p2[:12]+":"+str(round(_safe_float(d.get("score_p2"))*100,0))+"%</div>")
                               if signal=="Momentum" else "")
                            +"</div>",
                            unsafe_allow_html=True)

        # ── Value Bet avec Kelly ─────────────────────────────
        best_val, vb_analyse = compute_value_bets(p1,p2,proba,m["o1"],m["o2"])
        if best_val:
            kf=_safe_float(best_val.get("kelly"),0)
            edge_pct=round(_safe_float(best_val.get("edge"))*100,1)
            edge_col="#00DFA2" if edge_pct>=5 else "#FFB200"
            st.markdown(
                "<div style='background:linear-gradient(135deg,rgba(0,223,162,0.15),rgba(0,121,255,0.10));"
                "border:2px solid #00DFA2;border-radius:14px;padding:1.25rem;margin:0.75rem 0;'>"
                "<div style='font-size:1.35rem;font-weight:800;color:#E8EDF5;'>"
                "MISER SUR : <span style='color:#00DFA2;'>"+str(best_val["joueur"]).upper()+"</span></div>"
                "<div style='display:flex;gap:1.5rem;margin-top:0.5rem;flex-wrap:wrap;'>"
                "<span style='color:#FFB200;font-weight:700;'>Cote: "+str(round(_safe_float(best_val.get("cote")),2))+"</span>"
                "<span style='color:"+edge_col+";font-weight:700;'>Edge: +"+str(edge_pct)+"%</span>"
                "<span style='color:#0079FF;font-weight:700;'>Proba: "+str(round(_safe_float(best_val.get("proba"))*100,1))+"%</span>"
                +(kelly_badge(kf) if kf>0 else "")
                +"</div>"
                "<div style='font-size:0.72rem;color:#7A8599;margin-top:0.5rem;'>"
                "Marge bookmaker: "+str(round(_safe_float(vb_analyse.get(p1,{}).get("marge",0))*100,1))+"%  |  "
                "Kelly fractionne (25%) — ne jamais depasser 5% du bankroll sur un seul pari</div></div>",
                unsafe_allow_html=True)

            # Analyse détaillée des deux côtés
            if vb_analyse:
                with st.expander("Analyse value bet detaillee"):
                    ac1,ac2=st.columns(2)
                    for col_a,player,ostr in [(ac1,p1,m["o1"]),(ac2,p2,m["o2"])]:
                        with col_a:
                            va=vb_analyse.get(player,{})
                            grade=va.get("grade","D"); color_g=va.get("color","#7A8599")
                            edge_p=round(_safe_float(va.get("edge"))*100,1)
                            impl_p=round(_safe_float(va.get("implied"))*100,1)
                            prob_p=round(_safe_float(va.get("proba"))*100,1)
                            kf_p=round(_safe_float(va.get("kelly"))*100,1)
                            st.markdown(
                                "<div style='background:rgba(255,255,255,0.04);"
                                "border:1px solid "+color_g+"44;border-radius:12px;padding:0.75rem;'>"
                                "<div style='font-weight:700;color:"+color_g+";'>"+player[:20]
                                +" ["+grade+"]</div>"
                                "<div style='font-size:0.78rem;color:#E8EDF5;margin-top:0.25rem;'>"
                                "Proba modele: "+str(prob_p)+"%<br>"
                                "Proba cote: "+str(impl_p)+"%<br>"
                                "Edge: <b style='color:"+color_g+";'>"+str(edge_p)+"%</b><br>"
                                +"Kelly: "+str(kf_p)+"% bankroll<br>"
                                +"Qualite: "+str(va.get("label",""))+"</div></div>",
                                unsafe_allow_html=True)
        elif m["o1"] and m["o2"]:
            st.caption("Pas de value bet (edge < "+str(int(MIN_EDGE_COMBINE*100))+"% sur les deux joueurs)")

        # ── Paris alternatifs ────────────────────────────────
        with st.expander("Paris alternatifs"):
            bets=_alt_bets(p1,p2,surf,proba)
            for b in bets:
                ci="OK" if b["confidence"]>=65 else "~"
                st.markdown(ci+" **"+b["type"]+"** — "+b["description"]
                            +"  Proba "+str(round(b["proba"]*100,1))+"%  Cote "+str(b["cote"]))

        # ── Analyse IA ───────────────────────────────────────
        ai_txt=None
        if ia_choice!="Aucune":
            with st.spinner("Analyse "+ia_choice+"..."):
                ai_txt=ai_analysis(p1,p2,surf,tourn,proba,details,best_val,ia_choice)
            if ai_txt:
                with st.expander("Analyse IA",expanded=bool(best_val)):
                    st.markdown(
                        "<div style='background:rgba(0,121,255,0.06);"
                        "border:1px solid rgba(0,121,255,0.2);border-radius:10px;"
                        "padding:1rem;font-size:0.9rem;line-height:1.6;color:#E8EDF5;'>"
                        +ai_txt.replace("\n","<br>")+"</div>",
                        unsafe_allow_html=True)

        pred_data={"player1":p1,"player2":p2,"tournament":tourn,"surface":surf,
                   "proba":float(proba),"confidence":float(conf),
                   "odds1":m["o1"],"odds2":m["o2"],"favori":fav,
                   "best_value":best_val,"ml_used":bool(mi),
                   "sources":sources,"details":str(details),
                   "date":datetime.now().isoformat()}

        cb1,cb2=st.columns(2)
        with cb1:
            if st.button("Sauvegarder",key="save_"+str(i),use_container_width=True):
                st.success("Sauvegarde!") if save_pred(pred_data) else st.error("Erreur")
        with cb2:
            if st.button("Envoyer Telegram",key="tg_"+str(i),use_container_width=True):
                ok,resp=tg_send(format_pred_msg(pred_data,ai_txt))
                st.success(resp) if ok else st.error(resp)

        if send_tg and i==0:
            save_pred(pred_data); tg_send(format_pred_msg(pred_data,ai_txt))
        st.markdown("---")

    nb=check_achievements()
    if nb: st.balloons(); st.success(str(len(nb))+" badge(s) debloque(s)!")

def _alt_bets(p1,p2,surface,proba):
    bets=[]
    if proba>0.6 or proba<0.4:
        bets.append({"type":"Under 22.5 games","description":"Moins de 22.5 jeux","proba":0.64,"cote":1.78,"confidence":68})
    else:
        bets.append({"type":"Over 22.5 games","description":"Plus de 22.5 jeux","proba":0.61,"cote":1.82,"confidence":63})
    if proba>0.65:
        bets.append({"type":"Handicap -3.5","description":p1+" avec ecart","proba":0.57,"cote":2.15,"confidence":58})
    elif proba<0.35:
        bets.append({"type":"Handicap +3.5","description":p2+" outsider","proba":0.60,"cote":1.98,"confidence":62})
    if 0.3<proba<0.7:
        bets.append({"type":"Set chacun","description":"Match en 3 sets","proba":0.54,"cote":2.25,"confidence":54})
    return bets

# ═══════════════════════════════════════════════════════════════
# PAGE : EN ATTENTE
# ═══════════════════════════════════════════════════════════════
def show_pending():
    st.markdown(section_title("En attente","Validez les resultats"), unsafe_allow_html=True)
    h=load_history(); pending=[p for p in h if p.get("statut")=="en_attente"]
    if not pending:
        st.markdown("<div style='text-align:center;padding:3rem;background:rgba(255,255,255,0.04);"
                    "border:1px dashed rgba(255,255,255,0.10);border-radius:16px;'>"
                    "<div style='font-size:1.2rem;font-weight:700;color:#E8EDF5;'>Aucune prediction en attente!</div>"
                    "</div>", unsafe_allow_html=True)
        return
    st.info(str(len(pending))+" prediction(s) en attente")
    for pred in reversed(pending):
        pid=pred.get("id","?"); p1=pred.get("player1","?"); p2=pred.get("player2","?")
        fav=pred.get("favori","?"); surf=pred.get("surface","Hard")
        tourn=pred.get("tournament","?"); proba=_safe_float(pred.get("proba"),0.5)
        conf=_safe_float(pred.get("confidence"),50)
        date_str=pred.get("date","")[:16].replace("T"," ")
        fav_p=(proba if fav==p1 else 1-proba)
        sources_str=str(pred.get("sources",""))

        st.markdown(
            "<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.10);"
            "border-radius:16px;padding:1.5rem;margin-bottom:0.75rem;'>"
            "<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:0.75rem;'>"
            "<div><span style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:#E8EDF5;'>"
            +p1+" vs "+p2+"</span>"
            "<span style='margin-left:0.75rem;'>"+surface_badge(surf)+"</span></div>"
            "<span style='color:#6C7A89;font-size:0.78rem;'>"+date_str+"</span></div>"
            "<div style='color:#6C7A89;margin-bottom:0.75rem;'>"+tourn
            +" | Favori: <strong style='color:#00DFA2;'>"+fav+"</strong>"
            " ("+str(round(fav_p*100,1))+"%) | Conf: "+str(int(conf))+"/100"
            +((" | Sources: "+sources_str) if sources_str else "")+"</div>"
            "<div style='font-weight:600;color:#E8EDF5;'>Qui a gagne?</div></div>",
            unsafe_allow_html=True)
        if pred.get("best_value"):
            bv=pred["best_value"]
            kf=_safe_float(bv.get("kelly"),0)
            st.markdown(
                "<div style='background:rgba(0,223,162,0.07);border:1px solid rgba(0,223,162,0.2);"
                "border-radius:8px;padding:0.5rem 0.75rem;margin-bottom:0.5rem;font-size:0.8rem;'>"
                "Value bet: <strong style='color:#00DFA2;'>"+str(bv.get("joueur","?"))
                +" @ "+str(round(_safe_float(bv.get("cote")),2))+"</strong>"
                " Edge:+"+str(round(_safe_float(bv.get("edge"))*100,1))+"%"
                +(kelly_badge(kf) if kf>0 else "")+"</div>",
                unsafe_allow_html=True)
        c1,c2,c3=st.columns([2,2,1])
        with c1:
            if st.button(p1[:22]+" gagne",key="w1_"+pid,use_container_width=True,
                         type="primary" if fav==p1 else "secondary"):
                update_pred_result(pid,"gagne" if fav==p1 else "perdu",p1); check_achievements(); st.rerun()
        with c2:
            if st.button(p2[:22]+" gagne",key="w2_"+pid,use_container_width=True,
                         type="primary" if fav==p2 else "secondary"):
                update_pred_result(pid,"gagne" if fav==p2 else "perdu",p2); check_achievements(); st.rerun()
        with c3:
            if st.button("Abandon",key="ab_"+pid,use_container_width=True):
                update_pred_result(pid,"annule"); st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE : STATISTIQUES
# ═══════════════════════════════════════════════════════════════
def show_statistics():
    st.markdown(section_title("Statistiques","Analyse complete"), unsafe_allow_html=True)
    h=load_history()
    if not h: st.info("Aucune prediction."); return
    df=pd.DataFrame(h)
    df["date"]=pd.to_datetime(df["date"],errors="coerce")
    if "pronostic_correct" not in df.columns: df["pronostic_correct"]=False
    df["pronostic_correct"]=df["pronostic_correct"].fillna(False)
    gagnes=df[df["statut"]=="gagne"]; perdus=df[df["statut"]=="perdu"]
    abandons=df[df["statut"]=="annule"]; fini=df[df["statut"].isin(["gagne","perdu","annule"])]
    tv=len(gagnes)+len(perdus); acc=(len(gagnes)/tv*100) if tv>0 else 0

    c1,c2,c3,c4,c5=st.columns(5)
    with c1: st.markdown(big_metric("TOTAL",str(len(df)),color="#0079FF"), unsafe_allow_html=True)
    with c2: st.markdown(big_metric("GAGNES",str(len(gagnes)),color="#00DFA2"), unsafe_allow_html=True)
    with c3: st.markdown(big_metric("PERDUS",str(len(perdus)),color="#FF4757"), unsafe_allow_html=True)
    with c4: st.markdown(big_metric("ABANDONS",str(len(abandons)),color="#FFB200"), unsafe_allow_html=True)
    with c5: st.markdown(big_metric("PRECISION",str(round(acc,1))+"%"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_pie,col_table=st.columns([1,2])
    with col_pie:
        if tv>0:
            fig_d=go.Figure(go.Pie(labels=["Gagnes","Perdus","Abandons"],
                                    values=[len(gagnes),len(perdus),len(abandons)],
                                    hole=0.65,marker_colors=["#00DFA2","#FF4757","#FFB200"],textinfo="none"))
            fig_d.update_layout(height=240,margin=dict(l=0,r=0,t=10,b=0),
                                 paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                                 font=dict(color="#7A8599"),legend=dict(font=dict(size=11,color="#E8EDF5")),
                                 annotations=[dict(text="<b>"+str(int(acc))+"%</b>",x=0.5,y=0.5,
                                                    font=dict(size=22,color="#00DFA2",family="Syne"),showarrow=False)])
            st.plotly_chart(fig_d,use_container_width=True)
    with col_table:
        if not fini.empty:
            for _,row in fini.sort_values("date",ascending=False).head(10).iterrows():
                s=row.get("statut","?"); pc=row.get("pronostic_correct")
                sc="#00DFA244" if s=="gagne" else "#FF475744" if s=="perdu" else "#FFB20044"
                si="V" if s=="gagne" else "D" if s=="perdu" else "~"
                pb=("<span style='color:#00DFA2;'>OK</span>" if pc is True
                    else "<span style='color:#FF4757;'>X</span>" if pc is False
                    else "<span style='color:#7A8599;'>~</span>")
                src=str(row.get("sources",""))
                st.markdown(
                    "<div style='display:flex;align-items:center;gap:0.75rem;background:"+sc+";"
                    "border-radius:10px;padding:0.6rem 0.9rem;margin-bottom:0.4rem;'>"
                    "<span>"+si+"</span>"
                    "<div style='flex:1;'>"
                    "<div style='font-size:0.85rem;font-weight:600;color:#E8EDF5;'>"
                    +str(row.get("player1","?"))+" vs "+str(row.get("player2","?"))+"</div>"
                    "<div style='font-size:0.75rem;color:#6C7A89;'>"
                    "Prono: <strong>"+str(row.get("favori","?"))+"</strong>"
                    " | Vainqueur: <strong>"+str(row.get("vainqueur_reel","?"))+"</strong>"
                    +((" | "+src) if src else "")+"</div></div>"
                    "<div style='text-align:right;'>"+pb
                    +"<div style='font-size:0.7rem;color:#6C7A89;'>"+str(row.get("date",""))[:10]+"</div>"
                    "</div></div>",unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    surf_cols=st.columns(3)
    for si,surf in enumerate(SURFACES):
        cfg=SURFACE_CFG[surf]; sp=df[df["surface"]==surf]
        sg=len(sp[sp["statut"]=="gagne"]); sp2=len(sp[sp["statut"]=="perdu"])
        sa=len(sp[sp["statut"]=="annule"])
        s_acc=(sg/(sg+sp2)*100) if (sg+sp2)>0 else 0
        with surf_cols[si]:
            st.markdown(
                "<div style='background:"+cfg["bg"]+";border:1px solid "+cfg["color"]+"44;"
                "border-radius:14px;padding:1.25rem;text-align:center;'>"
                "<div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;"
                "color:"+cfg["color"]+"'>"+surf+"</div>"
                "<div style='font-size:2rem;font-weight:800;color:#E8EDF5;margin:0.5rem 0;'>"
                +str(int(s_acc))+"%</div>"
                "<div style='display:flex;justify-content:center;gap:1rem;font-size:0.8rem;'>"
                "<span style='color:#00DFA2;'>V "+str(sg)+"</span>"
                "<span style='color:#FF4757;'>D "+str(sp2)+"</span>"
                "<span style='color:#FFB200;'>A "+str(sa)+"</span></div>"
                "<div style='color:#6C7A89;font-size:0.75rem;'>"+str(len(sp))+" matchs</div></div>",
                unsafe_allow_html=True)
    if st.button("Exporter CSV"):
        st.download_button("Telecharger",df.to_csv(index=False),"tennisiq.csv","text/csv")

# ═══════════════════════════════════════════════════════════════
# PAGE : VALUE BETS
# ═══════════════════════════════════════════════════════════════
def show_value_bets():
    st.markdown(section_title("Value Bets","Avec edge + Kelly criterion"), unsafe_allow_html=True)
    mi=load_rf_model(); vbs=[]
    for m in _mock_matches():
        h2h_d=get_h2h(m["p1"],m["p2"],m["surface"])
        proba,_,_=ensemble_proba(m["p1"],m["p2"],m["surface"],m["tournament"],h2h_d,mi)
        seed=hash(m["p1"]+m["p2"])%1000/1000
        o1=round(1/proba*(0.88+0.15*seed),2)
        o2=round(1/(1-proba)*(0.88+0.15*(1-seed)),2)
        bv,analyse=compute_value_bets(m["p1"],m["p2"],proba,str(o1),str(o2))
        if bv:
            vbs.append({**bv,"match":m["p1"]+" vs "+m["p2"],
                        "surf":m["surface"],"tournament":m["tournament"],"proba_orig":proba})
    vbs.sort(key=lambda x:x["edge"],reverse=True)
    if not vbs: st.info("Aucun value bet."); return
    for rank,vb in enumerate(vbs,1):
        cfg=SURFACE_CFG.get(vb["surf"],SURFACE_CFG["Hard"])
        e_pct=round(_safe_float(vb["edge"])*100,1)
        kf=_safe_float(vb.get("kelly"),0)
        st.markdown(
            "<div style='background:rgba(255,255,255,0.04);border:1px solid rgba(0,223,162,0.3);"
            "border-radius:14px;padding:1.25rem;margin-bottom:1rem;'>"
            "<div style='display:flex;justify-content:space-between;margin-bottom:0.5rem;'>"
            "<span style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#E8EDF5;'>"
            "#"+str(rank)+" "+vb["match"]+"</span>"
            +surface_badge(vb["surf"])+"</div>"
            "<div style='font-size:1.2rem;font-weight:800;'>"
            "MISER SUR : <span style='color:#00DFA2;'>"+str(vb["joueur"]).upper()+"</span></div>"
            "<div style='display:flex;gap:1.5rem;margin-top:0.5rem;flex-wrap:wrap;'>"
            "<span style='color:#FFB200;font-weight:700;'>"+str(round(_safe_float(vb["cote"]),2))+"</span>"
            "<span style='color:#00DFA2;font-weight:700;'>Edge +"+str(e_pct)+"%</span>"
            "<span style='color:#0079FF;'>Proba "+str(round(_safe_float(vb["proba"])*100,1))+"%</span>"
            +(kelly_badge(kf) if kf>0 else "")
            +"</div></div>",
            unsafe_allow_html=True)
        if st.button("Telegram #"+str(rank),key="vbtg_"+str(rank)):
            msg=("<b>VALUE BET #"+str(rank)+" TENNISIQ</b>\n\n"
                 +str(vb["match"])+" | "+str(vb["surf"])+"\n\n"
                 "MISER SUR <b>"+str(vb["joueur"]).upper()+"</b>\n"
                 "Cote: <b>"+str(round(_safe_float(vb["cote"]),2))+"</b>"
                 " Edge: <b>+"+str(e_pct)+"%</b>"
                 +(" Kelly: <b>"+str(round(kf*100,1))+"% bankroll</b>" if kf>0 else "")
                 +"\n#TennisIQ #ValueBet")
            ok,resp=tg_send(msg); st.success(resp) if ok else st.error(resp)

def _mock_matches():
    return [
        {"p1":"Novak Djokovic","p2":"Carlos Alcaraz","surface":"Clay","tournament":"Roland Garros"},
        {"p1":"Jannik Sinner","p2":"Daniil Medvedev","surface":"Hard","tournament":"Miami Open"},
        {"p1":"Alexander Zverev","p2":"Stefanos Tsitsipas","surface":"Clay","tournament":"Madrid Open"},
        {"p1":"Holger Rune","p2":"Casper Ruud","surface":"Grass","tournament":"Wimbledon"},
    ]

# ═══════════════════════════════════════════════════════════════
# PAGE : CONFIGURATION
# ═══════════════════════════════════════════════════════════════
def show_config():
    st.markdown(section_title("Configuration","Modele + Tests IA"), unsafe_allow_html=True)
    mi=load_rf_model(); metadata=load_model_metadata()
    if mi:
        ps=mi.get("player_stats",{}); imp=mi.get("feature_importance",{})
        acc_m=mi.get("accuracy",metadata.get("accuracy",0))
        c1,c2,c3,c4=st.columns(4)
        with c1: st.markdown(big_metric("Accuracy",str(round(acc_m*100,1))+"%"), unsafe_allow_html=True)
        with c2: st.markdown(big_metric("AUC",str(round(mi.get("auc",0),3)),color="#0079FF"), unsafe_allow_html=True)
        with c3: st.markdown(big_metric("Joueurs",str(len(ps)),color="#7A8599"), unsafe_allow_html=True)
        with c4: st.markdown(big_metric("Matchs",str(metadata.get("n_matches",0)),color="#7A8599"), unsafe_allow_html=True)
        if imp:
            st.markdown("**Top features:**")
            for feat,val in sorted(imp.items(),key=lambda x:x[1],reverse=True)[:10]:
                st.progress(float(val),text=feat+": "+str(round(val*100,1))+"%")
        if st.button("Recharger modele"):
            for k in ["rf_model_cache","model_metadata_cache","elo_ratings","momentum_cache"]:
                st.session_state.pop(k,None)
            st.rerun()
    else:
        st.warning("Aucun modele RF.")

    st.markdown("---")
    st.subheader("ELO dynamique")
    elo=get_elo_ratings()
    st.success(str(len(elo))+" joueurs avec ELO calcule depuis les CSV")
    if elo:
        top_elo=sorted(elo.items(),key=lambda x:x[1]["global"],reverse=True)[:10]
        df_elo=pd.DataFrame([{"Joueur":p,"Global":int(d["global"]),
                               "Hard":int(d["Hard"]),"Clay":int(d["Clay"]),
                               "Grass":int(d["Grass"]),"Matchs":d["matches"]}
                              for p,d in top_elo])
        st.dataframe(df_elo,use_container_width=True)
    if st.button("Recalculer ELO"):
        st.session_state.pop("elo_ratings",None)
        st.session_state.pop("momentum_cache",None)
        st.cache_data.clear(); st.rerun()

    st.markdown("---")
    st.subheader("Tests IA")
    c1,c2,c3=st.columns(3)
    with c1:
        if st.button("Test Groq"):
            st.write(call_groq("Dis OK."))
    with c2:
        if st.button("Test DeepSeek"):
            st.write(call_deepseek("Dis OK."))
    with c3:
        if st.button("Test Claude"):
            st.write(call_claude_api("Dis OK."))

    st.markdown("---")
    c1,c2,c3=st.columns(3)
    with c1:
        if st.button("Effacer historique",use_container_width=True):
            if HIST_FILE.exists(): HIST_FILE.unlink()
            update_stats(); st.rerun()
    with c2:
        if st.button("Recalculer stats",use_container_width=True):
            update_stats(); st.success("OK")
    with c3:
        if st.button("Backup",use_container_width=True):
            backup(); st.success("OK")

def show_telegram():
    st.markdown(section_title("Telegram","Notifications"), unsafe_allow_html=True)
    token,chat_id=get_tg_config()
    if not token or not chat_id:
        st.warning("Telegram non configure.")
        st.code("TELEGRAM_BOT_TOKEN = ...\nTELEGRAM_CHAT_ID = ...")
        return
    st.success("Telegram configure - Chat ID: "+str(chat_id))
    c1,c2,c3=st.columns(3)
    with c1:
        if st.button("Tester",use_container_width=True):
            ok,msg=tg_send("<b>Test TennisIQ</b>\n"+datetime.now().strftime("%d/%m/%Y %H:%M"))
            st.success(msg) if ok else st.error(msg)
    with c2:
        if st.button("Stats",use_container_width=True):
            ok,msg=tg_send(format_stats_msg()); st.success(msg) if ok else st.error(msg)
    with c3:
        if st.button("Vider cache",use_container_width=True):
            st.cache_data.clear(); st.success("OK")
    with st.form("tg_form"):
        title=st.text_input("Titre","Message TennisIQ")
        body=st.text_area("Contenu",height=80)
        urgent=st.checkbox("URGENT")
        if st.form_submit_button("Envoyer"):
            if body:
                msg="<b>"+("URGENT - " if urgent else "")+title+"</b>\n\n"+body+"\n"+datetime.now().strftime("%d/%m/%Y %H:%M")
                ok,resp=tg_send(msg); st.success(resp) if ok else st.error(resp)

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="TennisIQ Pro",page_icon="T",
                       layout="wide",initial_sidebar_state="expanded")
    st.markdown(PRO_CSS,unsafe_allow_html=True)

    if "last_backup" not in st.session_state:
        st.session_state["last_backup"]=datetime.now()
    if (datetime.now()-st.session_state["last_backup"]).seconds>=86400:
        backup(); st.session_state["last_backup"]=datetime.now()

    with st.sidebar:
        st.markdown(
            "<div style='text-align:center;padding:1.5rem 0 1rem;'>"
            "<div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;"
            "background:linear-gradient(135deg,#00DFA2,#0079FF);"
            "-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>"
            "TennisIQ</div>"
            "<div style='font-size:0.75rem;color:#7A8599;text-transform:uppercase;'>"
            "ML+ELO Pro Edition</div></div>",unsafe_allow_html=True)

        page=st.radio("Nav",
                      ["Dashboard","Analyse","En Attente","Statistiques",
                       "Value Bets","Telegram","Configuration"],
                      label_visibility="collapsed")

        s=load_user_stats(); h=load_history()
        acc=calc_accuracy(); pend=len([p for p in h if p.get("statut")=="en_attente"])
        sc="#FF4757" if s.get("current_streak",0)==0 else "#00DFA2"
        st.markdown(
            "<div style='padding:0.5rem 0;'>"
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;text-align:center;'>"
            "<div style='background:rgba(255,255,255,0.04);border-radius:10px;padding:0.6rem;'>"
            "<div style='font-size:1.1rem;font-weight:800;color:#00DFA2;'>"+str(round(acc,1))+"%</div>"
            "<div style='font-size:0.65rem;color:#7A8599;text-transform:uppercase;'>Precision</div></div>"
            "<div style='background:rgba(255,255,255,0.04);border-radius:10px;padding:0.6rem;'>"
            "<div style='font-size:1.1rem;font-weight:800;color:#FFB200;'>"+str(pend)+"</div>"
            "<div style='font-size:0.65rem;color:#7A8599;text-transform:uppercase;'>Attente</div></div>"
            "<div style='background:rgba(255,255,255,0.04);border-radius:10px;padding:0.6rem;'>"
            "<div style='font-size:1.1rem;font-weight:800;color:#00DFA2;'>"+str(s.get("correct_predictions",0))+"</div>"
            "<div style='font-size:0.65rem;color:#7A8599;text-transform:uppercase;'>Gagnes</div></div>"
            "<div style='background:rgba(255,255,255,0.04);border-radius:10px;padding:0.6rem;'>"
            "<div style='font-size:1.1rem;font-weight:800;color:"+sc+";'>"+str(s.get("current_streak",0))+"</div>"
            "<div style='font-size:0.65rem;color:#7A8599;text-transform:uppercase;'>Serie</div></div>"
            "</div></div>",unsafe_allow_html=True)

    if   page=="Dashboard":     show_dashboard()
    elif page=="Analyse":       show_prediction()
    elif page=="En Attente":    show_pending()
    elif page=="Statistiques":  show_statistics()
    elif page=="Value Bets":    show_value_bets()
    elif page=="Telegram":      show_telegram()
    elif page=="Configuration": show_config()

if __name__=="__main__":
    main()
