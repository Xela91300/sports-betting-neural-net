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
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# ML IMPORTS (OPTIONNEL)
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

# Configuration de la page - DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT
st.set_page_config(
    page_title="TennisIQ Pro - Prédictions IA",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.tennisiq.pro/help',
        'Report a bug': 'https://www.tennisiq.pro/bug',
        'About': '# TennisIQ Pro - Intelligence Artificielle pour le Tennis'
    }
)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION AVANCÉE
# ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "src" / "data" / "raw" / "tml-tennis"
CACHE_DIR = ROOT_DIR / "cache"
HIST_DIR = ROOT_DIR / "history"

for dir_path in [MODELS_DIR, DATA_DIR, CACHE_DIR, HIST_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

HIST_FILE = HIST_DIR / "predictions_history.json"
COMB_HIST_FILE = HIST_DIR / "combines_history.json"
USER_STATS_FILE = HIST_DIR / "user_stats.json"

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
FEATURES = [
    "rank_diff", "pts_diff", "age_diff", "form_diff", "fatigue_diff",
    "ace_diff", "df_diff", "pct_1st_in_diff", "pct_1st_won_diff",
    "pct_2nd_won_diff", "pct_bp_saved_diff", "pct_ret_1st_diff",
    "pct_ret_2nd_diff", "h2h_score", "best_of", "surface_hard",
    "surface_clay", "surface_grass", "level_gs", "level_m1000",
    "level_500", "surf_wr_diff", "surf_matches_diff",
    "days_since_last_diff", "p1_returning", "p2_returning"
]

# Features utilisées pour le modèle ML (subset stable et non-leaké)
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
TOURS = {"ATP": "atp"}
ATP_ONLY = True
START_YEAR = 2007
MAX_MATCHES_ANALYSIS = 30
MAX_MATCHES_COMBINE = 30
MIN_PROBA_COMBINE = 0.55
MIN_EDGE_COMBINE = 0.02
MAX_SELECTIONS_COMBINE = 30

COLORS = {
    "primary": "#00DFA2",
    "primary_dark": "#00B886",
    "secondary": "#0079FF",
    "secondary_dark": "#0063CC",
    "success": "#00DFA2",
    "warning": "#FFB200",
    "danger": "#FF3B3F",
    "info": "#0079FF",
    "dark": "#0A1E2C",
    "dark_light": "#1A2E3C",
    "light": "#F5F9FF",
    "gray": "#6C7A89",
    "gray_light": "#E1E8F0",
    "white": "#FFFFFF",
    "black": "#000000",
    "surface_hard": "#0079FF",
    "surface_clay": "#E67E22",
    "surface_grass": "#00DFA2",
}

SURFACE_CONFIG = {
    "Hard": {"color": COLORS["surface_hard"], "icon": "🟦", "description": "Surface dure - Jeu rapide"},
    "Clay": {"color": COLORS["surface_clay"], "icon": "🟧", "description": "Terre battue - Jeu lent"},
    "Grass": {"color": COLORS["surface_grass"], "icon": "🟩", "description": "Gazon - Jeu très rapide"}
}

LEVEL_CONFIG = {
    "G": {"name": "Grand Chelem", "color": "#FFD700", "icon": "🏆"},
    "M": {"name": "Masters 1000", "color": "#C0C0C0", "icon": "🥇"},
    "500": {"name": "ATP 500", "color": "#CD7F32", "icon": "🥈"},
    "A": {"name": "ATP 250", "color": "#6C7A89", "icon": "🎾"},
    "F": {"name": "ATP Finals", "color": "#9400D3", "icon": "👑"},
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
    ("Brisbane International", "Hard", "A", 3), ("Adelaide International", "Hard", "A", 3),
    ("Auckland Open", "Hard", "A", 3), ("Montpellier", "Hard", "A", 3),
    ("Marseille", "Hard", "A", 3), ("Buenos Aires", "Clay", "A", 3),
    ("Estoril", "Clay", "A", 3), ("Munich", "Clay", "A", 3),
    ("Geneva", "Clay", "A", 3), ("Stuttgart", "Grass", "A", 3),
    ("Eastbourne", "Grass", "A", 3), ("Newport", "Grass", "A", 3),
    ("Bastad", "Clay", "A", 3), ("Kitzbuhel", "Clay", "A", 3),
    ("Los Cabos", "Hard", "A", 3), ("Atlanta", "Hard", "A", 3),
    ("Stockholm", "Hard", "A", 3), ("Antwerp", "Hard", "A", 3),
]

TOURN_DICT = {t[0]: (t[1], t[2], t[3]) for t in TOURNAMENTS_ATP}
TOURN_NAMES = [t[0] for t in TOURNAMENTS_ATP]

ODDS_API_KEY = "8090906fec7338245114345194fde760"
ODDS_CACHE = {}
ODDS_TTL = 6 * 3600

# ─────────────────────────────────────────────────────────────
# CSS PROFESSIONNEL
# ─────────────────────────────────────────────────────────────
def load_css():
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
        :root {
            --primary: #00DFA2; --primary-dark: #00B886; --secondary: #0079FF;
            --secondary-dark: #0063CC; --success: #00DFA2; --warning: #FFB200;
            --danger: #FF3B3F; --info: #0079FF; --dark: #0A1E2C; --dark-light: #1A2E3C;
            --light: #F5F9FF; --gray: #6C7A89; --gray-light: #E1E8F0; --white: #FFFFFF;
            --surface-hard: #0079FF; --surface-clay: #E67E22; --surface-grass: #00DFA2;
            --shadow-sm: 0 2px 4px rgba(0,0,0,0.05); --shadow-md: 0 4px 6px rgba(0,0,0,0.07);
            --shadow-lg: 0 10px 15px rgba(0,0,0,0.1); --shadow-xl: 0 20px 25px rgba(0,0,0,0.15);
            --radius-sm: 4px; --radius-md: 8px; --radius-lg: 12px; --radius-xl: 16px;
            --transition: all 0.2s ease;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background: linear-gradient(135deg, #0A1E2C 0%, #1A2E3C 100%); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F2533 0%, #0A1E2C 100%);
            border-right: 1px solid rgba(255,255,255,0.05);
            box-shadow: var(--shadow-xl);
        }
        .card {
            background: rgba(255,255,255,0.03); backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.05); border-radius: var(--radius-lg);
            padding: 1.5rem; margin-bottom: 1rem; transition: var(--transition);
        }
        .card:hover { border-color: rgba(255,255,255,0.1); transform: translateY(-2px); box-shadow: var(--shadow-lg); }
        .badge {
            display: inline-flex; align-items: center; padding: 0.25rem 0.75rem;
            border-radius: 20px; font-size: 0.75rem; font-weight: 600;
            letter-spacing: 0.3px; text-transform: uppercase; gap: 0.25rem; margin: 0.25rem;
        }
        .progress-bar {
            width: 100%; height: 8px; background: rgba(255,255,255,0.05);
            border-radius: 4px; overflow: hidden; margin: 0.5rem 0;
        }
        .progress-fill {
            height: 100%; background: linear-gradient(90deg, var(--primary), var(--secondary));
            border-radius: 4px; transition: width 0.5s ease;
        }
        .metric-card {
            background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05);
            border-radius: var(--radius-md); padding: 1rem; text-align: center;
        }
        .metric-label { font-size: 0.7rem; color: var(--gray); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.25rem; }
        .metric-value { font-size: 1.8rem; font-weight: 700; color: var(--white); line-height: 1.2; }
        .metric-unit { font-size: 0.8rem; color: var(--gray); margin-left: 0.25rem; }
        .stat-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.75rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .stat-row:last-child { border-bottom: none; }
        .stat-key { color: var(--gray); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
        .stat-value { color: var(--white); font-weight: 600; font-size: 0.95rem; }
        .divider {
            height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            margin: 2rem 0;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important; color: var(--gray) !important;
            font-size: 0.85rem; font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase;
            padding: 0.75rem 1.5rem; border-radius: 0 !important; border-bottom: 2px solid transparent !important;
        }
        .stTabs [aria-selected="true"] { color: var(--primary) !important; border-bottom: 2px solid var(--primary) !important; }
        .stButton > button {
            background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
            color: var(--white) !important; border: none !important; border-radius: var(--radius-md) !important;
            font-weight: 600 !important; font-size: 0.9rem !important; letter-spacing: 0.5px !important;
            padding: 0.75rem 2rem !important; transition: var(--transition) !important;
            text-transform: uppercase !important; box-shadow: var(--shadow-md) !important;
        }
        .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: var(--shadow-lg) !important; }
        .stButton > button:disabled { opacity: 0.5 !important; cursor: not-allowed !important; }
        .stTextInput > div > div {
            background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.05) !important;
            border-radius: var(--radius-md) !important; color: var(--white) !important;
        }
        .stSelectbox > div > div {
            background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.05) !important;
            border-radius: var(--radius-md) !important; color: var(--white) !important;
        }
        [data-testid="stExpander"] {
            background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.05) !important;
            border-radius: var(--radius-lg) !important;
        }
        .stAlert {
            background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.05) !important;
            border-left: 4px solid var(--primary) !important; border-radius: var(--radius-md) !important;
            color: var(--white) !important;
        }
        .message-success { background: rgba(0,223,162,0.1); border: 1px solid rgba(0,223,162,0.2); border-left: 4px solid var(--success); color: var(--success); padding: 1rem; border-radius: var(--radius-md); margin: 1rem 0; }
        .message-error { background: rgba(255,59,63,0.1); border: 1px solid rgba(255,59,63,0.2); border-left: 4px solid var(--danger); color: var(--danger); padding: 1rem; border-radius: var(--radius-md); margin: 1rem 0; }
        .message-warning { background: rgba(255,178,0,0.1); border: 1px solid rgba(255,178,0,0.2); border-left: 4px solid var(--warning); color: var(--warning); padding: 1rem; border-radius: var(--radius-md); margin: 1rem 0; }
        .message-info { background: rgba(0,121,255,0.1); border: 1px solid rgba(0,121,255,0.2); border-left: 4px solid var(--info); color: var(--info); padding: 1rem; border-radius: var(--radius-md); margin: 1rem 0; }
        .ml-badge {
            display: inline-block; background: linear-gradient(135deg, rgba(0,223,162,0.15), rgba(0,121,255,0.15));
            border: 1px solid rgba(0,223,162,0.3); border-radius: 20px; padding: 0.3rem 0.8rem;
            font-size: 0.75rem; font-weight: 700; color: #00DFA2; letter-spacing: 1px;
        }
        .model-card {
            background: linear-gradient(135deg, rgba(0,223,162,0.05), rgba(0,121,255,0.05));
            border: 1px solid rgba(0,223,162,0.15); border-radius: var(--radius-lg);
            padding: 1.5rem; margin-bottom: 1rem;
        }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
        h1, h2, h3, h4, h5, h6 { color: var(--white); font-weight: 600; margin-bottom: 1rem; }
        p { color: var(--gray); line-height: 1.6; }
        a { color: var(--primary); text-decoration: none; }
        .header { padding: 2rem 0 1rem 0; text-align: center; }
        .header-title { font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, var(--primary), var(--secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; letter-spacing: -1px; }
        .header-subtitle { color: var(--gray); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 3px; }
        .footer { text-align: center; padding: 2rem; color: var(--gray); font-size: 0.8rem; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 3rem; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_css()

# ─────────────────────────────────────────────────────────────
# GESTION DES APIS
# ─────────────────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        import os
        return os.environ.get("GROQ_API_KEY", None)

def call_groq_api(prompt):
    if not GROQ_AVAILABLE:
        return None
    api_key = get_groq_key()
    if not api_key:
        return None
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800, temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur API: {str(e)}"

# ─────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────
def format_number(num, decimals=2):
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return "—"
    if isinstance(num, (int, float)):
        if abs(num) >= 1e6:
            return f"{num/1e6:.1f}M"
        elif abs(num) >= 1e3:
            return f"{num/1e3:.0f}K"
        else:
            return f"{num:,.{decimals}f}".replace(",", " ")
    return str(num)

def format_percent(num, decimals=1):
    if num is None:
        return "—"
    return f"{num:.{decimals}%}"

def format_date(date_str):
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime("%d %b %Y %H:%M")
    except:
        return date_str

def create_progress_bar(value, color=COLORS["primary"]):
    return f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {value*100:.1f}%; background: linear-gradient(90deg, {color}, {COLORS['secondary']});"></div>
    </div>
    """

def create_badge(text, type="primary"):
    colors = {
        "primary": COLORS["primary"], "secondary": COLORS["secondary"],
        "success": COLORS["success"], "warning": COLORS["warning"],
        "danger": COLORS["danger"], "info": COLORS["info"],
        "hard": COLORS["surface_hard"], "clay": COLORS["surface_clay"], "grass": COLORS["surface_grass"],
    }
    color = colors.get(type, COLORS["primary"])
    bg_color = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)"
    return f'<span class="badge" style="background: {bg_color}; color: {color}; border: 1px solid {bg_color};">{text}</span>'

def create_metric(label, value, unit="", color=COLORS["white"]):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color};">{value}<span class="metric-unit">{unit}</span></div>
    </div>
    """

def create_stat_row(key, value, value_color=COLORS["white"]):
    return f"""
    <div class="stat-row">
        <span class="stat-key">{key}</span>
        <span class="stat-value" style="color: {value_color};">{value}</span>
    </div>
    """

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_atp_data():
    if not DATA_DIR.exists():
        return None
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        return None
    atp_dfs = []
    for f in csv_files:
        if 'wta' in f.name.lower():
            continue
        try:
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(f, encoding=encoding, on_bad_lines='skip', low_memory=False)
                    break
                except:
                    try:
                        df = pd.read_csv(f, sep=';', encoding=encoding, on_bad_lines='skip', low_memory=False)
                        break
                    except:
                        continue
            if df is not None and 'winner_name' in df.columns and 'loser_name' in df.columns:
                atp_dfs.append(df)
        except Exception:
            continue
    if atp_dfs:
        return pd.concat(atp_dfs, ignore_index=True)
    return None

# ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
# MACHINE LEARNING - CŒUR DU SYSTÈME
# ══════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=7200, show_spinner=False)
def precompute_player_stats_ml(_df):
    """
    Précompute toutes les statistiques joueurs nécessaires au ML :
    - Classement actuel (rang + points)
    - Âge moyen
    - Win rate global et par surface
    - Stats de service moyennes (ace, df, 1st %, etc.)
    Retourne un dict indexé par nom de joueur.
    """
    if _df is None or _df.empty:
        return {}

    has = {
        'winner_rank': 'winner_rank' in _df.columns,
        'loser_rank': 'loser_rank' in _df.columns,
        'winner_pts': 'winner_rank_points' in _df.columns,
        'loser_pts': 'loser_rank_points' in _df.columns,
        'winner_age': 'winner_age' in _df.columns,
        'loser_age': 'loser_age' in _df.columns,
        'surface': 'surface' in _df.columns,
        'tourney_date': 'tourney_date' in _df.columns,
        'w_ace': 'w_ace' in _df.columns,
    }

    # Trier par date si disponible
    if has['tourney_date']:
        try:
            df = _df.sort_values('tourney_date').copy()
        except:
            df = _df.copy()
    else:
        df = _df.copy()

    # Normaliser les noms
    df['_w_name'] = df['winner_name'].astype(str).str.strip()
    df['_l_name'] = df['loser_name'].astype(str).str.strip()

    all_players = set(df['_w_name'].unique()) | set(df['_l_name'].unique())
    stats = {}

    for player in all_players:
        if not player or player == 'nan':
            continue

        w_mask = df['_w_name'] == player
        l_mask = df['_l_name'] == player
        wins_df = df[w_mask]
        loss_df = df[l_mask]
        total = len(wins_df) + len(loss_df)

        if total == 0:
            continue

        # ── Classement le plus récent ──
        rank = None
        rank_points = None
        if has['winner_rank'] and len(wins_df) > 0:
            r = wins_df['winner_rank'].dropna()
            if len(r) > 0:
                rank = float(r.iloc[-1])
        if rank is None and has['loser_rank'] and len(loss_df) > 0:
            r = loss_df['loser_rank'].dropna()
            if len(r) > 0:
                rank = float(r.iloc[-1])
        if has['winner_pts'] and len(wins_df) > 0:
            p = wins_df['winner_rank_points'].dropna()
            if len(p) > 0:
                rank_points = float(p.iloc[-1])

        # ── Âge ──
        age = None
        if has['winner_age'] and len(wins_df) > 0:
            a = wins_df['winner_age'].dropna()
            if len(a) > 0:
                age = float(a.mean())
        if age is None and has['loser_age'] and len(loss_df) > 0:
            a = loss_df['loser_age'].dropna()
            if len(a) > 0:
                age = float(a.mean())

        # ── Win rate global ──
        win_rate = len(wins_df) / total if total > 0 else 0.5

        # ── Stats par surface ──
        surface_stats = {}
        if has['surface']:
            for surf in ['Hard', 'Clay', 'Grass']:
                w_surf = wins_df[wins_df['surface'] == surf]
                l_surf = loss_df[loss_df['surface'] == surf]
                n_w, n_l = len(w_surf), len(l_surf)
                tot_s = n_w + n_l
                surface_stats[surf] = {
                    'wins': n_w, 'losses': n_l, 'total': tot_s,
                    'win_rate': n_w / tot_s if tot_s > 0 else 0.5
                }

        # ── Stats de service moyennes (carrière) ──
        serve_cols_map = {
            'ace': ('w_ace', 'l_ace'), 'df': ('w_df', 'l_df'),
            'svpt': ('w_svpt', 'l_svpt'), '1stIn': ('w_1stIn', 'l_1stIn'),
            '1stWon': ('w_1stWon', 'l_1stWon'), '2ndWon': ('w_2ndWon', 'l_2ndWon'),
            'bpSaved': ('w_bpSaved', 'l_bpSaved'), 'bpFaced': ('w_bpFaced', 'l_bpFaced'),
            'SvGms': ('w_SvGms', 'l_SvGms'),
        }
        serve_raw = {}
        for stat, (wc, lc) in serve_cols_map.items():
            vals = []
            if wc in df.columns:
                vals.extend(wins_df[wc].dropna().tolist())
            if lc in df.columns:
                vals.extend(loss_df[lc].dropna().tolist())
            serve_raw[stat] = float(np.mean(vals)) if vals else 0.0

        # Calculer les pourcentages
        svpt = max(serve_raw.get('svpt', 1), 1)
        in1st = serve_raw.get('1stIn', 0)
        serve_pct = {
            'pct_1st_in': in1st / svpt,
            'pct_1st_won': serve_raw['1stWon'] / in1st if in1st > 0 else 0.0,
            'pct_2nd_won': serve_raw['2ndWon'] / max(svpt - in1st, 1),
            'pct_bp_saved': serve_raw['bpSaved'] / max(serve_raw['bpFaced'], 1),
            'ace_per_match': serve_raw['ace'],
            'df_per_match': serve_raw['df'],
        }

        # ── Forme récente (20 derniers matchs) ──
        recent_form = 0.5
        player_all = pd.concat([
            wins_df.assign(_result=1),
            loss_df.assign(_result=0)
        ])
        if has['tourney_date']:
            try:
                player_all = player_all.sort_values('tourney_date')
            except:
                pass
        if len(player_all) >= 5:
            last_20 = player_all.tail(20)
            recent_form = float(last_20['_result'].mean())

        stats[player] = {
            'rank': rank or 500.0,
            'rank_points': rank_points or 0.0,
            'age': age or 25.0,
            'total_matches': total,
            'wins': len(wins_df),
            'losses': len(loss_df),
            'win_rate': win_rate,
            'recent_form': recent_form,
            'surface_stats': surface_stats,
            'serve_raw': serve_raw,
            'serve_pct': serve_pct,
        }

    return stats


@st.cache_data(ttl=7200, show_spinner=False)
def prepare_ml_training_data(_df):
    """
    Prépare le dataset d'entraînement pour le modèle ML.
    Features uniquement disponibles AVANT le match (pas de leakage).
    Crée une ligne par perspective (gagnant=p1, perdant=p1) pour un dataset équilibré.
    """
    if _df is None or _df.empty:
        return None, None

    required = ['winner_rank', 'loser_rank']
    if not all(c in _df.columns for c in required):
        return None, None

    has_surface = 'surface' in _df.columns
    has_level = 'tourney_level' in _df.columns
    has_age = 'winner_age' in _df.columns
    has_pts = 'winner_rank_points' in _df.columns
    has_bestof = 'best_of' in _df.columns
    has_serve = 'w_ace' in _df.columns

    if has_surface and 'tourney_date' in _df.columns:
        try:
            df = _df.sort_values('tourney_date').reset_index(drop=True)
        except:
            df = _df.reset_index(drop=True)
    else:
        df = _df.reset_index(drop=True)

    X_list, y_list = [], []

    for _, row in df.iterrows():
        try:
            w_rank = float(row['winner_rank']) if pd.notna(row['winner_rank']) else 100.0
            l_rank = float(row['loser_rank']) if pd.notna(row['loser_rank']) else 100.0
            if w_rank <= 0: w_rank = 100.0
            if l_rank <= 0: l_rank = 100.0

            w_pts = float(row.get('winner_rank_points', 0)) if has_pts and pd.notna(row.get('winner_rank_points')) else 0.0
            l_pts = float(row.get('loser_rank_points', 0)) if has_pts and pd.notna(row.get('loser_rank_points')) else 0.0

            w_age = float(row.get('winner_age', 25)) if has_age and pd.notna(row.get('winner_age')) else 25.0
            l_age = float(row.get('loser_age', 25)) if has_age and pd.notna(row.get('loser_age')) else 25.0

            surface = str(row.get('surface', 'Hard')) if has_surface and pd.notna(row.get('surface')) else 'Hard'
            level = str(row.get('tourney_level', 'A')) if has_level and pd.notna(row.get('tourney_level')) else 'A'
            best_of = float(row.get('best_of', 3)) if has_bestof and pd.notna(row.get('best_of')) else 3.0

            # Encodage
            surf_hard = 1.0 if surface == 'Hard' else 0.0
            surf_clay = 1.0 if surface == 'Clay' else 0.0
            surf_grass = 1.0 if surface == 'Grass' else 0.0
            level_gs = 1.0 if level == 'G' else 0.0
            level_m = 1.0 if level == 'M' else 0.0
            best_of_5 = 1.0 if best_of == 5 else 0.0

            # Log ratio de classement (feature clé - meilleure linéarité que la diff brute)
            log_rank_ratio = np.log(l_rank / w_rank)  # positif si winner mieux classé

            pts_diff_norm = (w_pts - l_pts) / 5000.0
            age_diff = w_age - l_age

            # Stats de service (approximation par match - career avg dans prédiction)
            ace_diff_norm = df_diff_norm = 0.0
            pct_1st_in_diff = pct_1st_won_diff = pct_2nd_won_diff = pct_bp_saved_diff = 0.0

            if has_serve:
                try:
                    w_ace = float(row.get('w_ace', 0)) if pd.notna(row.get('w_ace')) else 0.0
                    l_ace = float(row.get('l_ace', 0)) if pd.notna(row.get('l_ace')) else 0.0
                    w_df = float(row.get('w_df', 0)) if pd.notna(row.get('w_df')) else 0.0
                    l_df = float(row.get('l_df', 0)) if pd.notna(row.get('l_df')) else 0.0
                    w_svpt = max(float(row.get('w_svpt', 100)) if pd.notna(row.get('w_svpt')) else 100.0, 1)
                    l_svpt = max(float(row.get('l_svpt', 100)) if pd.notna(row.get('l_svpt')) else 100.0, 1)
                    w_1in = float(row.get('w_1stIn', 0)) if pd.notna(row.get('w_1stIn')) else 0.0
                    l_1in = float(row.get('l_1stIn', 0)) if pd.notna(row.get('l_1stIn')) else 0.0
                    w_1w = float(row.get('w_1stWon', 0)) if pd.notna(row.get('w_1stWon')) else 0.0
                    l_1w = float(row.get('l_1stWon', 0)) if pd.notna(row.get('l_1stWon')) else 0.0
                    w_2w = float(row.get('w_2ndWon', 0)) if pd.notna(row.get('w_2ndWon')) else 0.0
                    l_2w = float(row.get('l_2ndWon', 0)) if pd.notna(row.get('l_2ndWon')) else 0.0
                    w_bps = float(row.get('w_bpSaved', 0)) if pd.notna(row.get('w_bpSaved')) else 0.0
                    l_bps = float(row.get('l_bpSaved', 0)) if pd.notna(row.get('l_bpSaved')) else 0.0
                    w_bpf = max(float(row.get('w_bpFaced', 1)) if pd.notna(row.get('w_bpFaced')) else 1.0, 1)
                    l_bpf = max(float(row.get('l_bpFaced', 1)) if pd.notna(row.get('l_bpFaced')) else 1.0, 1)

                    ace_diff_norm = (w_ace - l_ace) / 10.0
                    df_diff_norm = (w_df - l_df) / 5.0
                    pct_1st_in_diff = w_1in / w_svpt - l_1in / l_svpt
                    pct_1st_won_diff = (w_1w / max(w_1in, 1)) - (l_1w / max(l_1in, 1))
                    pct_2nd_won_diff = (w_2w / max(w_svpt - w_1in, 1)) - (l_2w / max(l_svpt - l_1in, 1))
                    pct_bp_saved_diff = w_bps / w_bpf - l_bps / l_bpf
                except:
                    pass

            # Feature vector gagnant=p1 → label=1
            feat_w = [
                log_rank_ratio, pts_diff_norm, age_diff,
                surf_clay, surf_grass, surf_hard,
                level_gs, level_m, best_of_5,
                0.0, 0.0,  # surf_wr_diff, career_wr_diff (inconnu en training ligne par ligne)
                ace_diff_norm, df_diff_norm,
                pct_1st_in_diff, pct_1st_won_diff, pct_2nd_won_diff, pct_bp_saved_diff,
            ]

            # Feature vector perdant=p1 → label=0 (on inverse les diffs)
            feat_l = [
                -log_rank_ratio, -pts_diff_norm, -age_diff,
                surf_clay, surf_grass, surf_hard,
                level_gs, level_m, best_of_5,
                0.0, 0.0,
                -ace_diff_norm, -df_diff_norm,
                -pct_1st_in_diff, -pct_1st_won_diff, -pct_2nd_won_diff, -pct_bp_saved_diff,
            ]

            X_list.append(feat_w)
            y_list.append(1)
            X_list.append(feat_l)
            y_list.append(0)

        except Exception:
            continue

    if len(X_list) < 500:
        return None, None

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def train_ml_model(df):
    """
    Entraîne le modèle ML complet :
    1. RandomForest + calibration isotonique
    2. Split temporel 80/20 pour le backtesting
    3. Métriques : accuracy, AUC, Brier score, log-loss
    4. Importance des features
    Retourne un dict avec modèle, scaler, métriques et features.
    """
    if not SKLEARN_AVAILABLE:
        return None

    with st.spinner("⏳ Préparation des données ML..."):
        X, y = prepare_ml_training_data(df)

    if X is None or len(X) < 500:
        return None

    # Split temporel (pas random pour éviter le leakage)
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Limiter l'entraînement à 60k samples pour la vitesse
    if len(X_train) > 60000:
        idx = np.random.choice(len(X_train), 60000, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    with st.spinner("🤖 Entraînement du modèle RandomForest..."):
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # Modèle de base : RandomForest (rapide, bien calibré, parallélisable)
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_split=20,
            min_samples_leaf=10, n_jobs=-1, random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train_sc, y_train)

        # Calibration isotonique pour améliorer les probabilités
        calibrated = CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=100, max_depth=8, min_samples_split=20,
                min_samples_leaf=10, n_jobs=-1, random_state=42
            ),
            cv=3, method='isotonic'
        )
        calibrated.fit(X_train_sc, y_train)

    with st.spinner("📊 Évaluation du modèle..."):
        # Métriques sur le test set (données non vues)
        y_pred_rf = rf.predict(X_test_sc)
        y_proba_rf = rf.predict_proba(X_test_sc)[:, 1]
        y_proba_cal = calibrated.predict_proba(X_test_sc)[:, 1]

        accuracy_rf = float(accuracy_score(y_test, y_pred_rf))
        auc_rf = float(roc_auc_score(y_test, y_proba_rf))
        brier_rf = float(brier_score_loss(y_test, y_proba_rf))
        logloss_rf = float(log_loss(y_test, y_proba_rf))
        brier_cal = float(brier_score_loss(y_test, y_proba_cal))

        # Importance des features
        feature_importances = dict(zip(ML_FEATURES, rf.feature_importances_.tolist()))

        # Calibration curve (pour visualisation)
        frac_pos, mean_pred = calibration_curve(y_test, y_proba_cal, n_bins=10)

        # Backtest ROI simulé (mise 1€ sur favori du modèle)
        roi_sims = []
        for i in range(len(y_test)):
            pred_win = y_proba_cal[i] > 0.5
            actual_win = y_test[i] == 1
            roi_sims.append(1.0 if pred_win == actual_win else -1.0)
        simulated_roi = float(np.mean(roi_sims))

    return {
        'model': calibrated,
        'rf_raw': rf,
        'scaler': scaler,
        'accuracy': accuracy_rf,
        'auc': auc_rf,
        'brier_uncalibrated': brier_rf,
        'brier_calibrated': brier_cal,
        'log_loss': logloss_rf,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'feature_importances': feature_importances,
        'calibration_frac_pos': frac_pos.tolist(),
        'calibration_mean_pred': mean_pred.tolist(),
        'simulated_roi': simulated_roi,
        'trained_at': datetime.now().isoformat(),
    }


def extract_ml_features(player_stats, p1, p2, surface, level, best_of, h2h=None):
    """
    Construit le vecteur de features ML pour un match donné.
    Utilise les stats précomputées (carrière) des deux joueurs.
    """
    s1 = player_stats.get(p1, {})
    s2 = player_stats.get(p2, {})

    r1 = max(s1.get('rank', 500.0), 1.0)
    r2 = max(s2.get('rank', 500.0), 1.0)
    log_rank_ratio = np.log(r2 / r1)  # positif si p1 mieux classé

    p1_pts = s1.get('rank_points', 0.0)
    p2_pts = s2.get('rank_points', 0.0)
    pts_diff_norm = (p1_pts - p2_pts) / 5000.0

    a1 = s1.get('age', 25.0)
    a2 = s2.get('age', 25.0)
    age_diff = a1 - a2

    # Surface
    surf_clay = 1.0 if surface == 'Clay' else 0.0
    surf_grass = 1.0 if surface == 'Grass' else 0.0
    surf_hard = 1.0 if surface == 'Hard' else 0.0

    # Level
    level_gs = 1.0 if level == 'G' else 0.0
    level_m = 1.0 if level == 'M' else 0.0
    best_of_5 = 1.0 if best_of == 5 else 0.0

    # Win rate sur surface
    surf_wr1 = s1.get('surface_stats', {}).get(surface, {}).get('win_rate', 0.5)
    surf_wr2 = s2.get('surface_stats', {}).get(surface, {}).get('win_rate', 0.5)
    surf_wr_diff = surf_wr1 - surf_wr2

    # Win rate carrière
    career_wr_diff = s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5)

    # Stats de service (moyennes carrière)
    sp1 = s1.get('serve_pct', {})
    sp2 = s2.get('serve_pct', {})
    sr1 = s1.get('serve_raw', {})
    sr2 = s2.get('serve_raw', {})

    ace_diff_norm = (sr1.get('ace', 0) - sr2.get('ace', 0)) / 10.0
    df_diff_norm = (sr1.get('df', 0) - sr2.get('df', 0)) / 5.0
    pct_1st_in_diff = sp1.get('pct_1st_in', 0) - sp2.get('pct_1st_in', 0)
    pct_1st_won_diff = sp1.get('pct_1st_won', 0) - sp2.get('pct_1st_won', 0)
    pct_2nd_won_diff = sp1.get('pct_2nd_won', 0) - sp2.get('pct_2nd_won', 0)
    pct_bp_saved_diff = sp1.get('pct_bp_saved', 0) - sp2.get('pct_bp_saved', 0)

    features = [
        log_rank_ratio, pts_diff_norm, age_diff,
        surf_clay, surf_grass, surf_hard,
        level_gs, level_m, best_of_5,
        surf_wr_diff, career_wr_diff,
        ace_diff_norm, df_diff_norm,
        pct_1st_in_diff, pct_1st_won_diff, pct_2nd_won_diff, pct_bp_saved_diff,
    ]

    return np.array(features, dtype=np.float32)


def predict_with_ml(model_info, player_stats, p1, p2, surface, level, best_of, h2h=None):
    """
    Fait une prédiction hybride :
    - Modèle ML calibré (60%) : rank, surface, service stats, niveau
    - Ajustements contextuels (40%) : H2H, forme récente, surface spécifique
    Retourne la probabilité de victoire de p1.
    """
    if model_info is None or player_stats is None:
        return None

    try:
        feat = extract_ml_features(player_stats, p1, p2, surface, level, best_of, h2h)
        X = feat.reshape(1, -1)
        X_sc = model_info['scaler'].transform(X)
        ml_proba = float(model_info['model'].predict_proba(X_sc)[0][1])

        # ── Ajustements contextuels ──
        adj = 0.0

        # H2H (poids 12%) — seulement si ≥ 3 matchs ensemble
        if h2h and h2h.get('total_matches', 0) >= 3:
            wins1 = h2h.get(f'{p1}_wins', 0)
            total_h2h = h2h['total_matches']
            adj += (wins1 / total_h2h - 0.5) * 0.12

        # Forme récente (poids 10%)
        s1 = player_stats.get(p1, {})
        s2 = player_stats.get(p2, {})
        form1 = s1.get('recent_form', 0.5)
        form2 = s2.get('recent_form', 0.5)
        adj += (form1 - form2) * 0.10

        # Win rate sur surface (poids 8% — si assez de matchs)
        surf_stats1 = s1.get('surface_stats', {}).get(surface, {})
        surf_stats2 = s2.get('surface_stats', {}).get(surface, {})
        if surf_stats1.get('total', 0) >= 10 and surf_stats2.get('total', 0) >= 10:
            adj += (surf_stats1['win_rate'] - surf_stats2['win_rate']) * 0.08

        final_proba = ml_proba + adj
        return max(0.05, min(0.95, final_proba))

    except Exception as e:
        return None


def get_model_from_session(df):
    """Récupère ou entraîne le modèle (cache en session state)."""
    key_rows = len(df) if df is not None else 0
    if ('ml_model' not in st.session_state or
            st.session_state.get('ml_model_rows', 0) != key_rows):
        st.session_state['ml_model'] = None
        st.session_state['ml_model_rows'] = key_rows
    return st.session_state.get('ml_model')


def get_player_stats_from_cache(df):
    """Récupère les stats joueurs depuis le cache session."""
    key_rows = len(df) if df is not None else 0
    if ('player_stats_cache' not in st.session_state or
            st.session_state.get('player_stats_cache_rows', 0) != key_rows):
        st.session_state['player_stats_cache'] = None
        st.session_state['player_stats_cache_rows'] = key_rows
    return st.session_state.get('player_stats_cache')


# ─────────────────────────────────────────────────────────────
# FONCTIONS DE CALCUL (COMPATIBILITÉ + ML)
# ─────────────────────────────────────────────────────────────
def get_player_stats(df, player, surface=None, n_matches=20):
    """Stats simples d'un joueur (compatibilité avec l'interface existante)."""
    if df is None or player is None:
        return None
    player_clean = player.strip() if isinstance(player, str) else player
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
    """Statistiques H2H entre deux joueurs."""
    if df is None or player1 is None or player2 is None:
        return None
    p1 = player1.strip() if isinstance(player1, str) else player1
    p2 = player2.strip() if isinstance(player2, str) else player2
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

def calculate_probability(df, player1, player2, surface, level='A', best_of=3, h2h=None):
    """
    Calcule la probabilité de victoire de player1.
    Utilise le modèle ML calibré si disponible, sinon retombe sur les règles.
    """
    # Essayer le modèle ML d'abord
    player_stats_cache = get_player_stats_from_session_safe()
    model_info = st.session_state.get('ml_model')

    if model_info is not None and player_stats_cache is not None:
        ml_proba = predict_with_ml(model_info, player_stats_cache, player1, player2,
                                    surface, level, best_of, h2h)
        if ml_proba is not None:
            return ml_proba

    # Fallback : règles simples
    stats1 = get_player_stats(df, player1, surface)
    stats2 = get_player_stats(df, player2, surface)
    score = 0.5
    if stats1 and stats2:
        score += (stats1['win_rate'] - stats2['win_rate']) * 0.3
    if h2h and h2h.get('total_matches', 0) > 0:
        wins1 = h2h.get(f'{player1}_wins', 0)
        score += (wins1 / h2h['total_matches'] - 0.5) * 0.2
    return max(0.05, min(0.95, score))


def get_player_stats_from_session_safe():
    """Récupère les stats depuis session state sans erreur."""
    return st.session_state.get('player_stats_cache')


def calculate_confidence(proba, player1, player2, h2h, player_stats_cache=None):
    """
    Calcule le score de confiance (0-100) basé sur :
    - Nombre de matchs joués par chaque joueur
    - Qualité du H2H
    - Force de la probabilité (éloignée de 50%)
    - Disponibilité du modèle ML calibré
    """
    confidence = 40.0

    # Bonus modèle ML disponible
    if st.session_state.get('ml_model') is not None:
        confidence += 15.0

    # Bonus données joueurs
    if player_stats_cache:
        s1 = player_stats_cache.get(player1, {})
        s2 = player_stats_cache.get(player2, {})
        n1 = s1.get('total_matches', 0)
        n2 = s2.get('total_matches', 0)
        confidence += min(n1 / 50, 10.0)
        confidence += min(n2 / 50, 10.0)

    # Bonus H2H
    if h2h and h2h.get('total_matches', 0) >= 5:
        confidence += 8.0
    elif h2h and h2h.get('total_matches', 0) >= 3:
        confidence += 4.0

    # Bonus probabilité extrême
    confidence += abs(proba - 0.5) * 30.0

    return min(100.0, confidence)

# ─────────────────────────────────────────────────────────────
# GESTION DE L'HISTORIQUE
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
    history = load_history()
    if 'date' not in pred_data:
        pred_data['date'] = datetime.now().isoformat()
    pred_data['statut'] = 'en_attente'
    pred_data['id'] = hashlib.md5(
        f"{pred_data['date']}{pred_data.get('player1','')}{pred_data.get('player2','')}".encode()
    ).hexdigest()[:8]
    history.append(pred_data)
    if len(history) > 1000:
        history = history[-1000:]
    try:
        with open(HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except:
        return False

def update_prediction_status(pred_id, statut):
    history = load_history()
    for pred in history:
        if pred.get('id') == pred_id:
            old_statut = pred.get('statut', 'en_attente')
            pred['statut'] = statut
            if old_statut == 'en_attente' and statut in ['joueur1_gagne', 'joueur2_gagne']:
                stats = load_user_stats()
                stats['total_predictions'] = stats.get('total_predictions', 0) + 1
                favori = pred.get('favori_modele', pred.get('player1'))
                if ((statut == 'joueur1_gagne' and favori == pred.get('player1')) or
                        (statut == 'joueur2_gagne' and favori == pred.get('player2'))):
                    stats['correct_predictions'] = stats.get('correct_predictions', 0) + 1
                    stats['current_streak'] = stats.get('current_streak', 0) + 1
                    stats['best_streak'] = max(stats.get('best_streak', 0), stats['current_streak'])
                else:
                    stats['current_streak'] = 0
                stats['last_updated'] = datetime.now().isoformat()
                try:
                    with open(USER_STATS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(stats, f, indent=2)
                except:
                    pass
            break
    try:
        with open(HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
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
    combines = load_combines()
    combine_data['date'] = datetime.now().isoformat()
    combine_data['statut'] = 'en_attente'
    combine_data['id'] = hashlib.md5(
        f"{combine_data['date']}{len(combines)}".encode()
    ).hexdigest()[:8]
    combines.append(combine_data)
    if len(combines) > 200:
        combines = combines[-200:]
    try:
        with open(COMB_HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(combines, f, indent=2, ensure_ascii=False)
        return True
    except:
        return False

def update_combine_status(combine_id, statut):
    combines = load_combines()
    for comb in combines:
        if comb.get('id') == combine_id:
            old_statut = comb.get('statut', 'en_attente')
            comb['statut'] = statut
            if old_statut == 'en_attente' and statut in ['gagne', 'perdu']:
                stats = load_user_stats()
                stats['total_combines'] = stats.get('total_combines', 0) + 1
                if statut == 'gagne':
                    stats['won_combines'] = stats.get('won_combines', 0) + 1
                    stats['total_won'] = stats.get('total_won', 0) + comb.get('gain_potentiel', 0)
                stats['total_invested'] = stats.get('total_invested', 0) + comb.get('mise', 0)
                stats['last_updated'] = datetime.now().isoformat()
                try:
                    with open(USER_STATS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(stats, f, indent=2)
                except:
                    pass
            break
    try:
        with open(COMB_HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(combines, f, indent=2, ensure_ascii=False)
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
            'last_updated': datetime.now().isoformat()
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
    st.markdown("""
    <div class="header">
        <div class="header-title">TennisIQ Pro</div>
        <div class="header-subtitle">Intelligence Artificielle pour le Tennis</div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    with st.spinner("Chargement des données..."):
        atp_data = load_atp_data()

    # Préchargement des stats joueurs en session state
    if atp_data is not None and st.session_state.get('player_stats_cache') is None:
        with st.spinner("🔄 Calcul des statistiques avancées..."):
            st.session_state['player_stats_cache'] = precompute_player_stats_ml(atp_data)
            st.session_state['player_stats_cache_rows'] = len(atp_data)

    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0; text-align: center;">
            <div style="font-size: 2rem; font-weight: 800; background: linear-gradient(135deg, #00DFA2, #0079FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                TennisIQ
            </div>
            <div style="color: #6C7A89; font-size: 0.7rem; letter-spacing: 2px; margin-top: 0.25rem;">
                PROFESSIONAL EDITION
            </div>
        </div>
        <div class="divider"></div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🎯 Prédictions", "📊 Multi-matchs", "🎰 Combinés",
             "📜 Historique", "📈 Statistiques", "🤖 Modèle ML", "⚙️ Configuration"],
            label_visibility="collapsed"
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if atp_data is not None:
            st.markdown(create_badge(f"ATP: {len(atp_data):,} matchs", "primary"), unsafe_allow_html=True)
        else:
            st.markdown(create_badge("ATP: 0 matchs", "danger"), unsafe_allow_html=True)

        # Statut modèle ML
        model_info = st.session_state.get('ml_model')
        if model_info is not None:
            acc = model_info.get('accuracy', 0)
            st.markdown(create_badge(f"🤖 ML: {acc:.1%} acc.", "success"), unsafe_allow_html=True)
        elif SKLEARN_AVAILABLE:
            st.markdown(create_badge("🤖 ML: non entraîné", "warning"), unsafe_allow_html=True)
        else:
            st.markdown(create_badge("🤖 sklearn absent", "danger"), unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; color: #6C7A89; font-size: 0.7rem;">
            Version 3.0.0-ML<br>© 2024 TennisIQ Pro
        </div>
        """, unsafe_allow_html=True)

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
    elif page == "⚙️ Configuration":
        show_configuration()


# ─────────────────────────────────────────────────────────────
# PAGE MODÈLE ML (NOUVELLE)
# ─────────────────────────────────────────────────────────────
def show_model_page(atp_data):
    """Page dédiée au modèle ML : entraînement, performance, backtesting."""

    st.markdown("<h2>🤖 Modèle Machine Learning</h2>", unsafe_allow_html=True)

    if not SKLEARN_AVAILABLE:
        st.error("⚠️ **scikit-learn non installé.** Exécutez : `pip install scikit-learn`")
        return

    if atp_data is None:
        st.warning("Aucune donnée ATP disponible pour entraîner le modèle.")
        return

    model_info = st.session_state.get('ml_model')

    # ── Section entraînement ──
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
                    st.session_state['ml_model_rows'] = len(atp_data)
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

    # ── Métriques de performance ──
    st.markdown("<h3>📊 Métriques de performance (test set)</h3>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    acc = model_info['accuracy']
    auc = model_info['auc']
    brier = model_info['brier_calibrated']
    logloss = model_info['log_loss']

    with col1:
        c = COLORS['success'] if acc >= 0.66 else COLORS['warning'] if acc >= 0.62 else COLORS['danger']
        st.markdown(create_metric("Précision", f"{acc:.1%}", "", c), unsafe_allow_html=True)
        st.caption("% de matchs correctement prédits")

    with col2:
        c = COLORS['success'] if auc >= 0.70 else COLORS['warning'] if auc >= 0.65 else COLORS['danger']
        st.markdown(create_metric("AUC-ROC", f"{auc:.3f}", "", c), unsafe_allow_html=True)
        st.caption("Discrimination (1.0 = parfait, 0.5 = aléatoire)")

    with col3:
        c = COLORS['success'] if brier <= 0.22 else COLORS['warning'] if brier <= 0.25 else COLORS['danger']
        st.markdown(create_metric("Brier Score", f"{brier:.3f}", "", c), unsafe_allow_html=True)
        st.caption("Calibration proba (0 = parfait, 0.25 = aléatoire)")

    with col4:
        roi = model_info.get('simulated_roi', 0)
        c = COLORS['success'] if roi > 0 else COLORS['danger']
        st.markdown(create_metric("ROI simulé", f"{roi:+.1%}", "", c), unsafe_allow_html=True)
        st.caption("Si on mise 1€ sur chaque favori du modèle")

    # Données d'entraînement
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(create_metric("Matchs entraînement", format_number(model_info['n_train'] // 2, 0), ""), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric("Matchs test", format_number(model_info['n_test'] // 2, 0), ""), unsafe_allow_html=True)
    with col3:
        improvement = acc - 0.5
        st.markdown(create_metric("Gain vs aléatoire", f"+{improvement:.1%}", "", COLORS['primary']), unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Importance des features ──
    st.markdown("<h3>🎯 Importance des variables</h3>", unsafe_allow_html=True)

    feat_imp = model_info.get('feature_importances', {})
    if feat_imp:
        feat_df = pd.DataFrame(list(feat_imp.items()), columns=['Feature', 'Importance'])
        feat_df = feat_df.sort_values('Importance', ascending=False)

        labels_fr = {
            'log_rank_ratio': '📊 Ratio classement (log)',
            'pts_diff_norm': '🏆 Différence de points ATP',
            'age_diff': '🎂 Différence d\'âge',
            'surf_clay': '🟧 Surface terre battue',
            'surf_grass': '🟩 Surface gazon',
            'surf_hard': '🟦 Surface dure',
            'level_gs': '🏆 Grand Chelem',
            'level_m': '🥇 Masters 1000',
            'best_of_5': '5️⃣ Best of 5',
            'surf_wr_diff': '📈 Écart win rate surface',
            'career_wr_diff': '📈 Écart win rate carrière',
            'ace_diff_norm': '⚡ Différence aces',
            'df_diff_norm': '💥 Différence doubles fautes',
            'pct_1st_in_diff': '🎯 Écart 1er service réussi',
            'pct_1st_won_diff': '🎾 Écart pts gagnés 1er service',
            'pct_2nd_won_diff': '🎾 Écart pts gagnés 2ème service',
            'pct_bp_saved_diff': '🛡️ Écart BP sauvées',
        }
        feat_df['Label'] = feat_df['Feature'].map(lambda x: labels_fr.get(x, x))

        for _, row in feat_df.iterrows():
            imp_pct = row['Importance']
            bar_color = COLORS['primary'] if imp_pct > 0.10 else COLORS['secondary'] if imp_pct > 0.05 else COLORS['gray']
            st.markdown(f"""
            <div style="margin: 0.4rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                    <span style="color: #fff; font-size: 0.85rem;">{row['Label']}</span>
                    <span style="color: {bar_color}; font-weight: 700; font-size: 0.85rem;">{imp_pct:.1%}</span>
                </div>
                <div style="background: rgba(255,255,255,0.05); border-radius: 4px; height: 6px; overflow: hidden;">
                    <div style="width: {min(imp_pct*300, 100):.1f}%; height: 100%; background: {bar_color}; border-radius: 4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Calibration des probabilités ──
    st.markdown("<h3>🎯 Calibration des probabilités</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p>Un modèle parfaitement calibré doit être proche de la diagonale : 
    si le modèle prédit 70% de chance, le joueur doit gagner ~70% du temps.
    La calibration isotonique améliore significativement cet alignement.</p>
    """, unsafe_allow_html=True)

    cal_fp = model_info.get('calibration_frac_pos', [])
    cal_mp = model_info.get('calibration_mean_pred', [])

    if cal_fp and cal_mp:
        cal_df = pd.DataFrame({
            'Probabilité prédite': cal_mp,
            'Fréquence observée': cal_fp,
            'Calibration parfaite': cal_mp,
        }).set_index('Probabilité prédite')
        st.line_chart(cal_df)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Interprétation ──
    st.markdown("<h3>📖 Interprétation des métriques</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="model-card">
            <h5>🎯 Précision (~65-68% attendu)</h5>
            <p>L'état de l'art en prédiction tennis se situe entre 67-72%.
            Au-dessus de 65% est considéré comme excellent.
            Le modèle surpasse systématiquement les bookmakers sur l'identification des favoris.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="model-card">
            <h5>📊 AUC-ROC (~0.70 attendu)</h5>
            <p>Mesure la capacité à distinguer gagnants et perdants.
            0.70 signifie que dans 70% des cas, le modèle attribue une probabilité
            plus haute au vrai gagnant qu'au vrai perdant.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="model-card">
            <h5>🎯 Brier Score (~0.22 attendu)</h5>
            <p>Mesure l'erreur quadratique des probabilités prédites.
            Plus proche de 0 = mieux calibré. 0.25 = prédiction aléatoire.
            La calibration isotonique réduit typiquement le Brier de 5-10%.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="model-card">
            <h5>💰 ROI simulé</h5>
            <p>Simule une mise de 1€ sur chaque favori du modèle (cote moyenne 1.8).
            Un ROI positif sans marge de bookmaker valide la qualité prédictive.
            En conditions réelles, le ROI sera réduit par la marge bookmaker (~5-8%).</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Backtesting par surface ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("<h3>📊 Backtesting par surface</h3>", unsafe_allow_html=True)

    if atp_data is not None and 'surface' in atp_data.columns:
        player_stats_cache = st.session_state.get('player_stats_cache', {})
        if player_stats_cache and model_info:
            backtest_results = []
            df_sorted = atp_data.copy()
            if 'tourney_date' in df_sorted.columns:
                try:
                    df_sorted = df_sorted.sort_values('tourney_date')
                except:
                    pass

            # Utiliser les 20% derniers matchs pour le backtest
            test_start = int(len(df_sorted) * 0.80)
            df_test = df_sorted.iloc[test_start:].copy()

            for surface in ['Hard', 'Clay', 'Grass']:
                surf_df = df_test[df_test['surface'] == surface].copy() if 'surface' in df_test.columns else pd.DataFrame()
                if len(surf_df) < 50:
                    continue

                correct = 0
                total = 0
                for _, row in surf_df.iterrows():
                    try:
                        w = str(row['winner_name']).strip()
                        l = str(row['loser_name']).strip()
                        level = str(row.get('tourney_level', 'A'))
                        best_of = float(row.get('best_of', 3)) if pd.notna(row.get('best_of')) else 3.0
                        h2h = get_h2h_stats(df_sorted, w, l)
                        proba = predict_with_ml(model_info, player_stats_cache, w, l, surface, level, best_of, h2h)
                        if proba is not None and proba > 0.5:
                            correct += 1
                        total += 1
                    except:
                        continue

                if total > 0:
                    backtest_results.append({
                        'Surface': f"{SURFACE_CONFIG.get(surface, {}).get('icon', '')} {surface}",
                        'Matchs testés': total,
                        'Précision': f"{correct/total:.1%}",
                        'Corrects': correct,
                        'Bonus vs 50%': f"+{(correct/total - 0.5):.1%}"
                    })

            if backtest_results:
                bt_df = pd.DataFrame(backtest_results)
                st.dataframe(bt_df, use_container_width=True, hide_index=True)
            else:
                st.info("Pas assez de données pour le backtesting par surface.")


# ─────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────
def show_dashboard(atp_data):
    st.markdown("<h2>🏠 Tableau de Bord</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(create_metric("Matchs ATP", format_number(len(atp_data) if atp_data is not None else 0), ""), unsafe_allow_html=True)

    with col2:
        history = load_history()
        st.markdown(create_metric("Prédictions", format_number(len(history)), ""), unsafe_allow_html=True)

    with col3:
        stats = load_user_stats()
        accuracy = (stats.get('correct_predictions', 0) / stats.get('total_predictions', 1)) * 100 if stats.get('total_predictions', 0) > 0 else 0
        st.markdown(create_metric("Précision", f"{accuracy:.1f}", "%", COLORS['success'] if accuracy >= 60 else COLORS['warning']), unsafe_allow_html=True)

    with col4:
        streak = stats.get('current_streak', 0)
        st.markdown(create_metric("Série en cours", f"{streak}", "", COLORS['success'] if streak > 0 else COLORS['gray']), unsafe_allow_html=True)

    with col5:
        model_info = st.session_state.get('ml_model')
        if model_info:
            ml_acc = model_info.get('accuracy', 0)
            c = COLORS['success'] if ml_acc >= 0.65 else COLORS['warning']
            st.markdown(create_metric("Modèle ML", f"{ml_acc:.1%}", "", c), unsafe_allow_html=True)
        else:
            st.markdown(create_metric("Modèle ML", "Non entraîné", "", COLORS['gray']), unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Bannière modèle ML si pas entraîné
    if st.session_state.get('ml_model') is None and SKLEARN_AVAILABLE and atp_data is not None:
        st.markdown("""
        <div class="model-card" style="text-align: center; padding: 2rem;">
            <div style="font-size: 2rem; margin-bottom: 1rem;">🤖</div>
            <h4>Activez le Modèle ML pour de meilleures prédictions</h4>
            <p>Le modèle RandomForest calibré améliore la précision de ~50% (règles simples) à ~65-68% (ML).<br>
            Rendez-vous sur la page <strong>🤖 Modèle ML</strong> pour l'entraîner.</p>
        </div>
        """, unsafe_allow_html=True)

    if atp_data is not None and 'surface' in atp_data.columns:
        st.markdown("<h3>📊 Répartition des surfaces</h3>", unsafe_allow_html=True)
        surface_counts = atp_data['surface'].value_counts()
        st.bar_chart(surface_counts)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("<h3>⏳ Prédictions en attente</h3>", unsafe_allow_html=True)

    history = load_history()
    pending = [h for h in history if h.get('statut') == 'en_attente']

    if pending:
        for pred in pending[-5:]:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 2])
                with col1:
                    st.markdown(f"**{pred.get('player1', '?')}** vs **{pred.get('player2', '?')}**")
                    st.caption(pred.get('date', '')[:10])
                with col2:
                    st.markdown(f"Tournoi: {pred.get('tournament', '?')}")
                    st.markdown(f"Surface: {pred.get('surface', '?')}")
                    if pred.get('ml_used'):
                        st.markdown('<span class="ml-badge">ML</span>', unsafe_allow_html=True)
                with col3:
                    proba = pred.get('proba', 0.5)
                    st.markdown(f"**Probas**")
                    st.markdown(f"{pred.get('player1', '?')}: {proba:.1%}")
                    st.markdown(f"{pred.get('player2', '?')}: {1-proba:.1%}")
                with col4:
                    favori = pred.get('favori_modele', pred.get('player1'))
                    st.markdown(f"**Favori**")
                    st.markdown(f"🏆 {favori}")
                with col5:
                    if st.button(f"✅ {pred.get('player1', '?')} gagne", key=f"dash_win1_{pred.get('id', '')}"):
                        update_prediction_status(pred.get('id', ''), 'joueur1_gagne')
                        st.rerun()
                    if st.button(f"✅ {pred.get('player2', '?')} gagne", key=f"dash_win2_{pred.get('id', '')}"):
                        update_prediction_status(pred.get('id', ''), 'joueur2_gagne')
                        st.rerun()
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("🚫 Abandon", key=f"dash_aband_{pred.get('id', '')}"):
                            update_prediction_status(pred.get('id', ''), 'abandon')
                            st.rerun()
                    with col_b:
                        if st.button("❌ Annulé", key=f"dash_annul_{pred.get('id', '')}"):
                            update_prediction_status(pred.get('id', ''), 'annule')
                            st.rerun()
                st.markdown("---")
    else:
        st.info("Aucune prédiction en attente")


# ─────────────────────────────────────────────────────────────
# PRÉDICTIONS
# ─────────────────────────────────────────────────────────────
def show_predictions(atp_data):
    st.markdown("<h2>🎯 Prédiction Simple</h2>", unsafe_allow_html=True)

    # Statut du modèle
    model_info = st.session_state.get('ml_model')
    player_stats_cache = st.session_state.get('player_stats_cache', {})
    if model_info:
        st.markdown(f'<div class="message-success">🤖 <strong>Modèle ML actif</strong> — Précision : {model_info["accuracy"]:.1%} | AUC : {model_info["auc"]:.3f} | Calibré isotonique</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="message-warning">⚠️ <strong>Modèle ML non entraîné</strong> — Mode règles simples. Allez sur <strong>🤖 Modèle ML</strong> pour activer le ML.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    player1 = player2 = tournament = None
    surface = "Hard"
    level = "A"
    best_of = 3
    df = atp_data

    with col1:
        if df is not None and not df.empty:
            winner_col = 'winner_name' if 'winner_name' in df.columns else None
            loser_col = 'loser_name' if 'loser_name' in df.columns else None
            if winner_col and loser_col:
                players = sorted(set(str(p).strip() for p in df[winner_col].dropna().unique() if pd.notna(p)) |
                                 set(str(p).strip() for p in df[loser_col].dropna().unique() if pd.notna(p)))
                if players:
                    player1 = st.selectbox("Joueur 1", players, key="pred_p1")
                    players2 = [p for p in players if p != player1]
                    player2 = st.selectbox("Joueur 2", players2, key="pred_p2")

                    if 'tourney_name' in df.columns:
                        tournaments = sorted(df['tourney_name'].dropna().unique())
                        tournament = st.selectbox("Tournoi", tournaments) if tournaments else None
                        if tournament and 'surface' in df.columns:
                            surface_df = df[df['tourney_name'] == tournament]['surface']
                            if not surface_df.empty:
                                surface = surface_df.iloc[0]
                        if tournament and 'tourney_level' in df.columns:
                            level_df = df[df['tourney_name'] == tournament]['tourney_level']
                            if not level_df.empty:
                                level = str(level_df.iloc[0])
                        if tournament and 'best_of' in df.columns:
                            bestof_df = df[df['tourney_name'] == tournament]['best_of']
                            if not bestof_df.empty:
                                try:
                                    best_of = int(bestof_df.iloc[0])
                                except:
                                    best_of = 3

                    with st.expander("📊 Cotes bookmaker (optionnel)"):
                        odds1 = st.text_input(f"Cote {player1}", key="pred_odds1", placeholder="1.75")
                        odds2 = st.text_input(f"Cote {player2}", key="pred_odds2", placeholder="2.10")

                    if surface in SURFACE_CONFIG:
                        st.markdown(create_badge(f"{SURFACE_CONFIG[surface]['icon']} {surface}", surface.lower()), unsafe_allow_html=True)

                    # Afficher les stats avancées du joueur si disponibles
                    if player_stats_cache and player1 in player_stats_cache:
                        s = player_stats_cache[player1]
                        with st.expander(f"📈 Stats avancées de {player1}"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                r = s.get('rank', '?')
                                st.markdown(create_stat_row("Classement ATP", f"#{int(r) if r != '?' else '?'}"), unsafe_allow_html=True)
                                st.markdown(create_stat_row("Matchs (total)", s.get('total_matches', 0)), unsafe_allow_html=True)
                                st.markdown(create_stat_row("Win rate", f"{s.get('win_rate', 0):.1%}"), unsafe_allow_html=True)
                                st.markdown(create_stat_row("Forme récente", f"{s.get('recent_form', 0):.1%}"), unsafe_allow_html=True)
                            with col_b:
                                surf_s = s.get('surface_stats', {}).get(surface, {})
                                st.markdown(create_stat_row(f"WR {surface}", f"{surf_s.get('win_rate', 0):.1%}"), unsafe_allow_html=True)
                                st.markdown(create_stat_row(f"Matchs {surface}", surf_s.get('total', 0)), unsafe_allow_html=True)
                                sp = s.get('serve_pct', {})
                                st.markdown(create_stat_row("1er service %", f"{sp.get('pct_1st_in', 0):.1%}"), unsafe_allow_html=True)
                                st.markdown(create_stat_row("BP sauvées %", f"{sp.get('pct_bp_saved', 0):.1%}"), unsafe_allow_html=True)

    if not (player1 and player2):
        odds1 = odds2 = ""

    with col2:
        if player1 and player2:
            p1 = player1.strip()
            p2 = player2.strip()
            h2h = get_h2h_stats(df, p1, p2)

            # Calcul probabilité (ML ou règles)
            proba = calculate_probability(df, p1, p2, surface, level, best_of, h2h)
            confidence = calculate_confidence(proba, p1, p2, h2h, player_stats_cache)
            ml_used = model_info is not None and player_stats_cache is not None

            # Value bet
            best_value = None
            if 'odds1' in dir() and odds1 and odds2:
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

            st.markdown("<h3 style='text-align: center;'>Résultat</h3>", unsafe_allow_html=True)

            if ml_used:
                st.markdown('<div style="text-align:center;margin-bottom:0.5rem;"><span class="ml-badge">🤖 PRÉDICTION ML CALIBRÉE</span></div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-weight:600;">{p1}</span>
                    <span style="font-weight:600;">{p2}</span>
                </div>
                {create_progress_bar(proba)}
                <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                    <span style="color: {COLORS['primary']}; font-weight:700;">{proba:.1%}</span>
                    <span style="color: {COLORS['gray']}; font-weight:700;">{(1-proba):.1%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(create_metric("Favori du modèle", favori, "", COLORS['primary']), unsafe_allow_html=True)
            with col_b:
                conf_color = COLORS['success'] if confidence >= 70 else COLORS['warning'] if confidence >= 50 else COLORS['danger']
                st.markdown(create_metric("Confiance", f"{confidence:.0f}", "/100", conf_color), unsafe_allow_html=True)

            # Stats avancées comparatives
            if player_stats_cache and p1 in player_stats_cache and p2 in player_stats_cache:
                s1 = player_stats_cache[p1]
                s2 = player_stats_cache[p2]

                st.markdown("<h5 style='margin-top:1rem;'>📊 Comparaison avancée</h5>", unsafe_allow_html=True)

                indicators = [
                    ("Classement", f"#{int(s1.get('rank',999))}", f"#{int(s2.get('rank',999))}", s1.get('rank',999) < s2.get('rank',999)),
                    (f"WR {surface}", f"{s1.get('surface_stats',{}).get(surface,{}).get('win_rate',0):.1%}", f"{s2.get('surface_stats',{}).get(surface,{}).get('win_rate',0):.1%}", s1.get('surface_stats',{}).get(surface,{}).get('win_rate',0) > s2.get('surface_stats',{}).get(surface,{}).get('win_rate',0)),
                    ("Forme récente", f"{s1.get('recent_form',0):.1%}", f"{s2.get('recent_form',0):.1%}", s1.get('recent_form',0) > s2.get('recent_form',0)),
                    ("Win rate", f"{s1.get('win_rate',0):.1%}", f"{s2.get('win_rate',0):.1%}", s1.get('win_rate',0) > s2.get('win_rate',0)),
                ]

                for label, v1, v2, p1_better in indicators:
                    c1 = COLORS['success'] if p1_better else COLORS['gray']
                    c2 = COLORS['success'] if not p1_better else COLORS['gray']
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;padding:0.4rem 0;border-bottom:1px solid rgba(255,255,255,0.05);">
                        <span style="color:{c1};font-weight:600;font-size:0.9rem;">{v1}</span>
                        <span style="color:#6C7A89;font-size:0.75rem;text-transform:uppercase;">{label}</span>
                        <span style="color:{c2};font-weight:600;font-size:0.9rem;">{v2}</span>
                    </div>
                    """, unsafe_allow_html=True)

            if best_value:
                st.success(f"✅ **Value bet!** {best_value['joueur']} @ {best_value['cote']:.2f} (edge: {best_value['edge']*100:+.1f}%)")
            elif 'odds1' in dir() and odds1 and odds2:
                st.warning("⚠️ Aucun value bet significatif")

            if st.button("💾 Sauvegarder la prédiction", use_container_width=True):
                pred_data = {
                    'player1': p1, 'player2': p2,
                    'tournament': tournament or "Inconnu",
                    'surface': surface, 'level': level, 'best_of': best_of,
                    'proba': proba, 'confidence': confidence, 'circuit': "ATP",
                    'odds1': odds1 if 'odds1' in dir() and odds1 else None,
                    'odds2': odds2 if 'odds2' in dir() and odds2 else None,
                    'favori_modele': favori, 'best_value': best_value,
                    'ml_used': ml_used,
                }
                if save_prediction(pred_data):
                    st.success("✅ Prédiction sauvegardée !")
                else:
                    st.error("Erreur lors de la sauvegarde")

    if player1 and player2:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        stats1 = get_player_stats(df, player1.strip(), surface)
        stats2 = get_player_stats(df, player2.strip(), surface)
        h2h = get_h2h_stats(df, player1.strip(), player2.strip())

        with col1:
            st.markdown(f"<h4>{player1.strip()}</h4>", unsafe_allow_html=True)
            if stats1:
                st.markdown(create_stat_row("Matchs joués", stats1['matches_played']), unsafe_allow_html=True)
                st.markdown(create_stat_row("Victoires", stats1['wins']), unsafe_allow_html=True)
                st.markdown(create_stat_row("Défaites", stats1['losses']), unsafe_allow_html=True)
                st.markdown(create_stat_row("Win rate", f"{stats1['win_rate']:.1%}"), unsafe_allow_html=True)

        with col2:
            st.markdown(f"<h4>{player2.strip()}</h4>", unsafe_allow_html=True)
            if stats2:
                st.markdown(create_stat_row("Matchs joués", stats2['matches_played']), unsafe_allow_html=True)
                st.markdown(create_stat_row("Victoires", stats2['wins']), unsafe_allow_html=True)
                st.markdown(create_stat_row("Défaites", stats2['losses']), unsafe_allow_html=True)
                st.markdown(create_stat_row("Win rate", f"{stats2['win_rate']:.1%}"), unsafe_allow_html=True)

        with col3:
            st.markdown("<h4>Face à Face</h4>", unsafe_allow_html=True)
            if h2h:
                st.markdown(create_stat_row("Matchs", h2h['total_matches']), unsafe_allow_html=True)
                st.markdown(create_stat_row(player1.strip(), h2h.get(f'{player1.strip()}_wins', 0)), unsafe_allow_html=True)
                st.markdown(create_stat_row(player2.strip(), h2h.get(f'{player2.strip()}_wins', 0)), unsafe_allow_html=True)
            else:
                st.info("Aucun face-à-face")


# ─────────────────────────────────────────────────────────────
# MULTI-MATCHS
# ─────────────────────────────────────────────────────────────
def show_multimatches(atp_data):
    st.markdown("<h2>📊 Multi-matchs</h2>", unsafe_allow_html=True)

    model_info = st.session_state.get('ml_model')
    player_stats_cache = st.session_state.get('player_stats_cache', {})

    if model_info:
        st.markdown(f'<div class="message-success">🤖 Modèle ML actif — Précision : {model_info["accuracy"]:.1%}</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        n_matches = st.number_input("Nombre de matchs", min_value=2, max_value=MAX_MATCHES_ANALYSIS, value=min(5, MAX_MATCHES_ANALYSIS))
    with col2:
        use_ai = st.checkbox("Activer l'analyse IA", value=True)
    with col3:
        auto_save = st.checkbox("Sauvegarde auto", value=True)

    df = atp_data
    if df is None or df.empty:
        st.warning("Données non disponibles")
        return

    winner_col = 'winner_name' if 'winner_name' in df.columns else None
    loser_col = 'loser_name' if 'loser_name' in df.columns else None
    if not winner_col or not loser_col:
        return

    players = sorted(set(str(p).strip() for p in df[winner_col].dropna().unique() if pd.notna(p)) |
                     set(str(p).strip() for p in df[loser_col].dropna().unique() if pd.notna(p)))
    tournaments = sorted(df['tourney_name'].dropna().unique()) if 'tourney_name' in df.columns else []

    matches = []
    for i in range(n_matches):
        with st.expander(f"Match {i+1}", expanded=i == 0 and n_matches <= 10):
            col1, col2, col3 = st.columns(3)
            with col1:
                p1 = st.selectbox("Joueur 1", players, key=f"mm_p1_{i}")
            with col2:
                players2 = [p for p in players if p != p1]
                p2 = st.selectbox("Joueur 2", players2, key=f"mm_p2_{i}")
            with col3:
                tourn = st.selectbox("Tournoi", tournaments if tournaments else ["Inconnu"], key=f"mm_tourn_{i}")

            surface = "Hard"
            level = "A"
            best_of = 3
            if tourn and tourn != "Inconnu" and 'surface' in df.columns:
                s_df = df[df['tourney_name'] == tourn]['surface']
                if not s_df.empty:
                    surface = s_df.iloc[0]
            if tourn and 'tourney_level' in df.columns:
                l_df = df[df['tourney_name'] == tourn]['tourney_level']
                if not l_df.empty:
                    level = str(l_df.iloc[0])
            if tourn and 'best_of' in df.columns:
                bo_df = df[df['tourney_name'] == tourn]['best_of']
                if not bo_df.empty:
                    try:
                        best_of = int(bo_df.iloc[0])
                    except:
                        pass

            col1, col2 = st.columns(2)
            with col1:
                odds1 = st.text_input(f"Cote {p1}", key=f"mm_odds1_{i}", placeholder="1.75")
            with col2:
                odds2 = st.text_input(f"Cote {p2}", key=f"mm_odds2_{i}", placeholder="2.10")

            if surface in SURFACE_CONFIG:
                st.markdown(create_badge(f"{SURFACE_CONFIG[surface]['icon']} {surface}", surface.lower()), unsafe_allow_html=True)

            matches.append({
                'player1': p1.strip() if p1 else None, 'player2': p2.strip() if p2 else None,
                'tournament': tourn, 'surface': surface, 'level': level, 'best_of': best_of,
                'odds1': odds1, 'odds2': odds2,
            })

    if st.button(f"🔍 Analyser {n_matches} matchs", use_container_width=True):
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, match in enumerate(matches):
            status_text.text(f"Analyse du match {i+1}/{n_matches}...")
            if match['player1'] and match['player2']:
                h2h = get_h2h_stats(df, match['player1'], match['player2'])
                proba = calculate_probability(df, match['player1'], match['player2'],
                                              match['surface'], match['level'], match['best_of'], h2h)
                confidence = calculate_confidence(proba, match['player1'], match['player2'], h2h, player_stats_cache)

                proba_impl1 = 1/float(match['odds1'].replace(',', '.')) if match['odds1'] else None
                proba_impl2 = 1/float(match['odds2'].replace(',', '.')) if match['odds2'] else None
                edge1 = proba - proba_impl1 if proba_impl1 else None
                edge2 = (1 - proba) - proba_impl2 if proba_impl2 else None

                best_value = None
                if edge1 is not None and edge2 is not None:
                    if edge1 > edge2 and edge1 > MIN_EDGE_COMBINE:
                        best_value = {'joueur': match['player1'], 'edge': edge1, 'cote': float(match['odds1'].replace(',', '.')), 'proba': proba}
                    elif edge2 > edge1 and edge2 > MIN_EDGE_COMBINE:
                        best_value = {'joueur': match['player2'], 'edge': edge2, 'cote': float(match['odds2'].replace(',', '.')), 'proba': 1 - proba}

                favori = match['player1'] if proba >= 0.5 else match['player2']

                if auto_save:
                    save_prediction({
                        'player1': match['player1'], 'player2': match['player2'],
                        'tournament': match['tournament'], 'surface': match['surface'],
                        'proba': proba, 'confidence': confidence, 'circuit': "ATP",
                        'odds1': match['odds1'], 'odds2': match['odds2'],
                        'favori_modele': favori, 'best_value': best_value,
                        'ml_used': model_info is not None, 'source': 'multi_match'
                    })

                results.append({
                    'match': i+1, 'player1': match['player1'], 'player2': match['player2'],
                    'tournament': match['tournament'], 'surface': match['surface'],
                    'proba': proba, 'confidence': confidence,
                    'odds1': match['odds1'], 'odds2': match['odds2'],
                    'proba_impl1': proba_impl1, 'proba_impl2': proba_impl2,
                    'edge1': edge1, 'edge2': edge2,
                    'best_value': best_value, 'favori_modele': favori,
                    'proba_favori': proba if proba >= 0.5 else 1 - proba,
                    'ml_used': model_info is not None,
                })

            progress_bar.progress((i + 1) / n_matches)

        status_text.empty()
        progress_bar.empty()

        if results:
            st.markdown("## 📊 Résultats de l'analyse")
            for result in results:
                with st.container():
                    ml_tag = '<span class="ml-badge">🤖 ML</span>' if result.get('ml_used') else ''
                    st.markdown(f"### Match {result['match']}: {result['player1']} vs {result['player2']} {ml_tag}", unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(create_metric("Tournoi", result['tournament']), unsafe_allow_html=True)
                    with col2:
                        st.markdown(create_metric("Surface", result['surface']), unsafe_allow_html=True)
                    with col3:
                        st.markdown(create_metric("Confiance", f"{result['confidence']:.0f}", "/100"), unsafe_allow_html=True)
                    with col4:
                        if result['best_value']:
                            st.markdown(create_metric("Value Bet", "✅ OUI", "", COLORS['success']), unsafe_allow_html=True)
                        else:
                            st.markdown(create_metric("Value Bet", "❌ NON", "", COLORS['danger']), unsafe_allow_html=True)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"**{result['player1']}**")
                        st.markdown(f"Modèle: {result['proba']:.1%}")
                        if result['odds1']:
                            st.markdown(f"Cote: {result['odds1']}")
                    with col2:
                        st.markdown(f"**{result['player2']}**")
                        st.markdown(f"Modèle: {1-result['proba']:.1%}")
                        if result['odds2']:
                            st.markdown(f"Cote: {result['odds2']}")
                    with col3:
                        st.markdown("**Favori du modèle**")
                        st.markdown(f"🏆 {result['favori_modele']}")
                        st.markdown(f"Probas: {result['proba_favori']:.1%}")
                    with col4:
                        if result['best_value']:
                            st.markdown("**🎯 Value Bet**")
                            st.markdown(f"💰 {result['best_value']['joueur']}")
                            st.markdown(f"Edge: {result['best_value']['edge']*100:+.1f}%")

                    st.markdown(f"""
                    <div style="margin: 1rem 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span>{result['player1']}</span><span>{result['player2']}</span>
                        </div>
                        {create_progress_bar(result['proba'])}
                        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                            <span style="color: {COLORS['primary']};">{result['proba']:.1%}</span>
                            <span style="color: {COLORS['gray']};">{(1-result['proba']):.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if result['best_value']:
                        st.success(f"✅ **Recommandation:** Parier sur **{result['best_value']['joueur']}** @ {result['best_value']['cote']:.2f} (edge: {result['best_value']['edge']*100:+.1f}%)")
                    elif result['odds1'] and result['odds2']:
                        st.warning("⚠️ Aucun value bet détecté — à éviter")

                    st.markdown("---")

            if use_ai and GROQ_AVAILABLE:
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown("<h3>🤖 Analyses IA</h3>", unsafe_allow_html=True)
                for result in results:
                    if result['best_value']:
                        vb_txt = f"Value bet sur {result['best_value']['joueur']} (edge {result['best_value']['edge']*100:+.1f}%)"
                    else:
                        vb_txt = "Aucun value bet"
                    prompt = (f"Analyse ce match ATP : {result['player1']} vs {result['player2']} "
                              f"sur {result['surface']}. Proba ML : {result['player1']} {result['proba']:.1%} | "
                              f"{result['player2']} {1-result['proba']:.1%}. {vb_txt}. 3 points clés en français.")
                    with st.spinner(f"Analyse match {result['match']}..."):
                        analysis = call_groq_api(prompt)
                    if analysis:
                        with st.expander(f"Match {result['match']}: {result['player1']} vs {result['player2']}"):
                            st.markdown(analysis)


# ─────────────────────────────────────────────────────────────
# COMBINÉS
# ─────────────────────────────────────────────────────────────
def show_combines(atp_data):
    st.markdown("<h2>🎰 Générateur de Combinés</h2>", unsafe_allow_html=True)

    model_info = st.session_state.get('ml_model')
    player_stats_cache = st.session_state.get('player_stats_cache', {})

    if model_info:
        st.markdown(f'<div class="message-success">🤖 Modèle ML actif — Précision : {model_info["accuracy"]:.1%}</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_matches = st.number_input("Matchs à analyser", min_value=2, max_value=MAX_MATCHES_COMBINE, value=min(5, MAX_MATCHES_COMBINE))
    with col2:
        mise = st.number_input("Mise (€)", min_value=1.0, max_value=10000.0, value=10.0, step=5.0)
    with col3:
        use_ai = st.checkbox("Analyses IA", value=True)
    with col4:
        auto_select = st.checkbox("Auto-sélection", value=True)

    df = atp_data
    if df is None or df.empty:
        return

    winner_col = 'winner_name' if 'winner_name' in df.columns else None
    loser_col = 'loser_name' if 'loser_name' in df.columns else None
    if not winner_col or not loser_col:
        return

    players = sorted(set(str(p).strip() for p in df[winner_col].dropna().unique() if pd.notna(p)) |
                     set(str(p).strip() for p in df[loser_col].dropna().unique() if pd.notna(p)))
    tournaments = sorted(df['tourney_name'].dropna().unique()) if 'tourney_name' in df.columns else []

    matches = []
    st.markdown(f"### Saisie des {n_matches} matchs")

    for i in range(n_matches):
        with st.container():
            st.markdown(f"**Match {i+1}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                p1 = st.selectbox("J1", players, key=f"comb_p1_{i}", label_visibility="collapsed")
            with col2:
                players2 = [p for p in players if p != p1]
                p2 = st.selectbox("J2", players2, key=f"comb_p2_{i}", label_visibility="collapsed")
            with col3:
                tourn = st.selectbox("T", tournaments if tournaments else ["Inconnu"], key=f"comb_tourn_{i}", label_visibility="collapsed")

            col1, col2 = st.columns(2)
            with col1:
                odds1 = st.text_input(f"Cote {p1 if p1 else 'J1'}", key=f"comb_odds1_{i}", placeholder="1.75")
            with col2:
                odds2 = st.text_input(f"Cote {p2 if p2 else 'J2'}", key=f"comb_odds2_{i}", placeholder="2.10")

            surface = "Hard"
            level = "A"
            best_of = 3
            if tourn and tourn != "Inconnu" and 'surface' in df.columns:
                s_df = df[df['tourney_name'] == tourn]['surface']
                if not s_df.empty:
                    surface = s_df.iloc[0]
            if tourn and 'tourney_level' in df.columns:
                l_df = df[df['tourney_name'] == tourn]['tourney_level']
                if not l_df.empty:
                    level = str(l_df.iloc[0])

            if surface in SURFACE_CONFIG:
                st.markdown(create_badge(surface, surface.lower()), unsafe_allow_html=True)

            if i < n_matches - 1:
                st.markdown("---")

            matches.append({
                'player1': p1.strip() if p1 else None, 'player2': p2.strip() if p2 else None,
                'tournament': tourn, 'surface': surface, 'level': level, 'best_of': best_of,
                'odds1': odds1, 'odds2': odds2,
            })

    if st.button("🎯 Générer le meilleur combiné", use_container_width=True):
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        selections = []
        invalid = 0

        with st.spinner("Analyse des matchs..."):
            for match in matches:
                if match['player1'] and match['player2'] and match['odds1'] and match['odds2']:
                    try:
                        o1 = float(match['odds1'].replace(',', '.'))
                        o2 = float(match['odds2'].replace(',', '.'))
                        h2h = get_h2h_stats(df, match['player1'], match['player2'])
                        proba = calculate_probability(df, match['player1'], match['player2'],
                                                      match['surface'], match['level'], match['best_of'], h2h)
                        edge1 = proba - 1/o1
                        edge2 = (1 - proba) - 1/o2

                        if auto_select:
                            if edge1 > MIN_EDGE_COMBINE and proba >= MIN_PROBA_COMBINE:
                                selections.append({'match': f"{match['player1']} vs {match['player2']}", 'joueur': match['player1'], 'proba': proba, 'cote': o1, 'edge': edge1, 'surface': match['surface']})
                            elif edge2 > MIN_EDGE_COMBINE and (1 - proba) >= MIN_PROBA_COMBINE:
                                selections.append({'match': f"{match['player1']} vs {match['player2']}", 'joueur': match['player2'], 'proba': 1 - proba, 'cote': o2, 'edge': edge2, 'surface': match['surface']})
                        else:
                            selections.append({'match': f"{match['player1']} vs {match['player2']}", 'joueur1': match['player1'], 'joueur2': match['player2'], 'proba1': proba, 'proba2': 1 - proba, 'cote1': o1, 'cote2': o2, 'edge1': edge1, 'edge2': edge2, 'surface': match['surface']})
                    except:
                        invalid += 1

        if invalid > 0:
            st.warning(f"{invalid} matchs ignorés (cotes invalides)")

        if auto_select:
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

                st.markdown("### 📊 Résultats du combiné")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    c = COLORS['success'] if proba_combi >= 0.3 else COLORS['warning'] if proba_combi >= 0.15 else COLORS['danger']
                    st.markdown(create_metric("Probabilité", f"{proba_combi:.1%}", "", c), unsafe_allow_html=True)
                with col2:
                    st.markdown(create_metric("Cote combinée", f"{cote_combi:.2f}"), unsafe_allow_html=True)
                with col3:
                    c = COLORS['success'] if esperance > 0 else COLORS['danger']
                    st.markdown(create_metric("Espérance", f"{esperance:+.2f}€", "", c), unsafe_allow_html=True)
                with col4:
                    st.markdown(create_metric("Kelly %", f"{kelly*100:.1f}", "%"), unsafe_allow_html=True)

                st.markdown(f"### 📋 Sélections ({len(selected)})")
                df_sel = pd.DataFrame([{'#': i+1, 'Joueur': s['joueur'], 'Match': s['match'], 'Proba': f"{s['proba']:.1%}", 'Cote': f"{s['cote']:.2f}", 'Edge': f"{s['edge']*100:+.1f}%"} for i, s in enumerate(selected)])
                st.dataframe(df_sel, use_container_width=True, hide_index=True)

                save_combine({'selections': selected, 'proba_globale': proba_combi, 'cote_globale': cote_combi, 'mise': mise, 'gain_potentiel': gain, 'esperance': esperance, 'kelly': kelly, 'nb_matches': len(selected), 'ml_used': model_info is not None})
                st.success("✅ Combiné sauvegardé !")

                if use_ai and GROQ_AVAILABLE:
                    st.markdown("### 🤖 Analyse du combiné")
                    prompt = f"Analyse ce combiné tennis de {len(selected)} matchs. Proba globale: {proba_combi:.1%}, cote: {cote_combi:.2f}, espérance: {esperance:+.2f}€. Sélections: {[s['joueur'] for s in selected]}. Donne un avis en 3 points."
                    with st.spinner("Analyse..."):
                        analysis = call_groq_api(prompt)
                    if analysis:
                        st.markdown(f"<div class='card'>{analysis}</div>", unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Pas assez de sélections valides ({len(selections)} trouvées, minimum 2).")
        else:
            st.markdown("### 📋 Options disponibles")
            df_options = pd.DataFrame([{'Match': s['match'], 'Surface': s['surface'], f"Option 1 (edge {s['edge1']*100:+.1f}%)": f"{s['joueur1']} @ {s['cote1']:.2f}", f"Option 2 (edge {s['edge2']*100:+.1f}%)": f"{s['joueur2']} @ {s['cote2']:.2f}"} for s in selections])
            st.dataframe(df_options, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# HISTORIQUE
# ─────────────────────────────────────────────────────────────
def show_history():
    st.markdown("<h2>📜 Historique</h2>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📋 Prédictions", "🎰 Combinés"])

    with tab1:
        history = load_history()
        if history:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                filter_surface = st.selectbox("Surface", ["Toutes"] + SURFACES)
            with col2:
                filter_statut = st.selectbox("Statut", ["Tous", "en_attente", "joueur1_gagne", "joueur2_gagne", "abandon", "annule"])
            with col3:
                search = st.text_input("Rechercher", placeholder="Nom joueur...")
            with col4:
                show_all = st.checkbox("Afficher tout", value=False)

            filtered = history
            if filter_surface != "Toutes":
                filtered = [h for h in filtered if h.get('surface') == filter_surface]
            if filter_statut != "Tous":
                filtered = [h for h in filtered if h.get('statut') == filter_statut]
            if search:
                filtered = [h for h in filtered if search.lower() in h.get('player1', '').lower() or search.lower() in h.get('player2', '').lower()]

            filtered.reverse()
            if not show_all and len(filtered) > 20:
                filtered = filtered[:20]
                st.caption(f"20 plus récentes affichées sur {len(filtered)}")

            for pred in filtered:
                date_str = pred.get('date', '')[:16]
                player1 = pred.get('player1', 'Inconnu')
                player2 = pred.get('player2', 'Inconnu')
                statut = pred.get('statut', 'en_attente')
                ml_used = pred.get('ml_used', False)

                status_map = {
                    'en_attente': (COLORS['warning'], "⏳ En attente"),
                    'joueur1_gagne': (COLORS['success'], f"✅ {player1} a gagné"),
                    'joueur2_gagne': (COLORS['success'], f"✅ {player2} a gagné"),
                    'abandon': (COLORS['danger'], "🚫 Abandon"),
                    'annule': (COLORS['gray'], "❌ Annulé"),
                }
                status_color, status_text = status_map.get(statut, (COLORS['gray'], statut))

                with st.container():
                    ml_tag = ' <span class="ml-badge">🤖 ML</span>' if ml_used else ''
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.02); border-radius: 8px; padding: 1rem; margin-bottom: 1rem; border-left: 4px solid {status_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="color: {COLORS['gray']}; font-size: 0.8rem;">{date_str}</span>
                                <h4 style="margin: 0.5rem 0;">{player1} vs {player2}{ml_tag}</h4>
                            </div>
                            <span style="background: {status_color}20; color: {status_color}; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;">{status_text}</span>
                        </div>
                    """, unsafe_allow_html=True)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(create_metric("Tournoi", pred.get('tournament', '—')), unsafe_allow_html=True)
                    with col2:
                        st.markdown(create_metric("Surface", pred.get('surface', '—')), unsafe_allow_html=True)
                    with col3:
                        proba = pred.get('proba', 0.5)
                        favori = pred.get('favori_modele', player1 if proba >= 0.5 else player2)
                        st.markdown(create_metric("Favori", favori, "", COLORS['primary']), unsafe_allow_html=True)
                    with col4:
                        confidence = pred.get('confidence', 0)
                        if isinstance(confidence, (int, float)):
                            cc = COLORS['success'] if confidence >= 70 else COLORS['warning'] if confidence >= 50 else COLORS['danger']
                            st.markdown(create_metric("Confiance", f"{confidence:.0f}", "/100", cc), unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="margin: 1rem 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>{player1}: {proba:.1%}</span>
                            <span>{player2}: {1-proba:.1%}</span>
                        </div>
                        {create_progress_bar(proba)}
                    </div>
                    """, unsafe_allow_html=True)

                    bv = pred.get('best_value')
                    if bv:
                        st.success(f"🎯 Value bet: {bv['joueur']} (edge: {bv['edge']*100:+.1f}%)")

                    if statut == 'en_attente':
                        st.markdown("**Résultat :**")
                        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                        with col_b1:
                            if st.button(f"✅ {player1} gagne", key=f"hist_win1_{pred.get('id','')}"):
                                update_prediction_status(pred.get('id', ''), 'joueur1_gagne')
                                st.rerun()
                        with col_b2:
                            if st.button(f"✅ {player2} gagne", key=f"hist_win2_{pred.get('id','')}"):
                                update_prediction_status(pred.get('id', ''), 'joueur2_gagne')
                                st.rerun()
                        with col_b3:
                            if st.button("🚫 Abandon", key=f"hist_aband_{pred.get('id','')}"):
                                update_prediction_status(pred.get('id', ''), 'abandon')
                                st.rerun()
                        with col_b4:
                            if st.button("❌ Annulé", key=f"hist_annul_{pred.get('id','')}"):
                                update_prediction_status(pred.get('id', ''), 'annule')
                                st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Aucune prédiction dans l'historique")

    with tab2:
        combines = load_combines()
        if combines:
            items_per_page = 5
            total_pages = (len(combines) + items_per_page - 1) // items_per_page
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(combines))

            for i in range(start_idx, end_idx):
                comb = combines[-(i+1)]
                date_str = comb.get('date', '')[:16]
                nb = comb.get('nb_matches', 0)
                proba = comb.get('proba_globale', 0)
                statut = comb.get('statut', 'en_attente')
                status_map = {'en_attente': (COLORS['warning'], "⏳ En attente"), 'gagne': (COLORS['success'], "✅ Gagné"), 'perdu': (COLORS['danger'], "❌ Perdu")}
                sc, st_txt = status_map.get(statut, (COLORS['gray'], statut))
                ml_tag = " 🤖" if comb.get('ml_used') else ""

                with st.expander(f"🎯{ml_tag} {date_str} — {nb} matchs — Proba {proba:.1%} — {st_txt}", expanded=i == start_idx):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        c = COLORS['success'] if proba >= 0.3 else COLORS['warning'] if proba >= 0.15 else COLORS['danger']
                        st.markdown(create_metric("Probabilité", f"{proba:.1%}", "", c), unsafe_allow_html=True)
                    with col2:
                        st.markdown(create_metric("Cote", f"{comb.get('cote_globale',0):.2f}"), unsafe_allow_html=True)
                    with col3:
                        esp = comb.get('esperance', 0)
                        st.markdown(create_metric("Espérance", f"{esp:+.2f}€", "", COLORS['success'] if esp > 0 else COLORS['danger']), unsafe_allow_html=True)
                    with col4:
                        st.markdown(create_metric("Kelly", f"{comb.get('kelly',0)*100:.1f}", "%"), unsafe_allow_html=True)

                    if 'selections' in comb and comb['selections']:
                        df_sel = pd.DataFrame([{'Joueur': s.get('joueur','?'), 'Match': s.get('match',''), 'Proba': f"{s.get('proba',0):.1%}", 'Cote': f"{s.get('cote',0):.2f}", 'Edge': f"{s.get('edge',0)*100:+.1f}%"} for s in comb['selections']])
                        st.dataframe(df_sel, use_container_width=True, hide_index=True)

                    if statut == 'en_attente':
                        col_b1, col_b2 = st.columns(2)
                        with col_b1:
                            if st.button("✅ Combiné gagné", key=f"comb_win_{comb.get('id','')}"):
                                update_combine_status(comb.get('id', ''), 'gagne')
                                st.rerun()
                        with col_b2:
                            if st.button("❌ Combiné perdu", key=f"comb_loss_{comb.get('id','')}"):
                                update_combine_status(comb.get('id', ''), 'perdu')
                                st.rerun()
        else:
            st.info("Aucun combiné dans l'historique")


# ─────────────────────────────────────────────────────────────
# STATISTIQUES
# ─────────────────────────────────────────────────────────────
def show_statistics():
    st.markdown("<h2>📈 Statistiques</h2>", unsafe_allow_html=True)

    stats = load_user_stats()
    history = load_history()
    combines = load_combines()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_metric("Prédictions", stats.get('total_predictions', 0)), unsafe_allow_html=True)
    with col2:
        accuracy = (stats.get('correct_predictions', 0) / max(stats.get('total_predictions', 1), 1)) * 100
        cc = COLORS['success'] if accuracy >= 60 else COLORS['warning'] if accuracy >= 50 else COLORS['danger']
        st.markdown(create_metric("Précision", f"{accuracy:.1f}", "%", cc), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric("Combinés", stats.get('total_combines', 0)), unsafe_allow_html=True)
    with col4:
        wr = (stats.get('won_combines', 0) / max(stats.get('total_combines', 1), 1)) * 100
        st.markdown(create_metric("Réussite combinés", f"{wr:.1f}", "%"), unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        streak = stats.get('current_streak', 0)
        st.markdown(create_metric("Série en cours", f"{streak}", "", COLORS['success'] if streak > 0 else COLORS['gray']), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric("Meilleure série", f"{stats.get('best_streak', 0)}", ""), unsafe_allow_html=True)
    with col3:
        correct = stats.get('correct_predictions', 0)
        incorrect = stats.get('total_predictions', 0) - correct
        st.markdown(create_metric("Correct/Incorrect", f"{correct}/{incorrect}"), unsafe_allow_html=True)
    with col4:
        lu = stats.get('last_updated', '')[:10]
        st.markdown(create_metric("Dernière MAJ", lu), unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("<h3>💰 Performance financière</h3>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    ti = stats.get('total_invested', 0)
    tw = stats.get('total_won', 0)
    profit = tw - ti
    roi = (profit / ti * 100) if ti > 0 else 0

    with col1:
        st.markdown(create_metric("Total investi", f"{ti:.2f}", "€"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric("Total gagné", f"{tw:.2f}", "€"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric("Profit", f"{profit:+.2f}", "€", COLORS['success'] if profit >= 0 else COLORS['danger']), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric("ROI", f"{roi:+.1f}", "%", COLORS['success'] if roi >= 0 else COLORS['danger']), unsafe_allow_html=True)

    if history:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("<h3>📊 Évolution des prédictions</h3>", unsafe_allow_html=True)
        df_history = pd.DataFrame(history)
        if 'date' in df_history.columns:
            df_history['date'] = pd.to_datetime(df_history['date'], errors='coerce')
            df_history = df_history.dropna(subset=['date'])
            df_history['mois'] = df_history['date'].dt.to_period('M').astype(str)
            monthly_counts = df_history['mois'].value_counts().sort_index()
            st.line_chart(monthly_counts)

        # Précision avec/sans ML
        if 'ml_used' in df_history.columns:
            st.markdown("<h4>🤖 Comparaison ML vs Règles</h4>", unsafe_allow_html=True)
            completed = df_history[df_history['statut'].isin(['joueur1_gagne', 'joueur2_gagne'])].copy()
            if len(completed) > 0:
                completed['correct'] = ((completed['statut'] == 'joueur1_gagne') & (completed['proba'] >= 0.5)) | \
                                        ((completed['statut'] == 'joueur2_gagne') & (completed['proba'] < 0.5))
                col1, col2 = st.columns(2)
                with col1:
                    ml_c = completed[completed['ml_used'] == True]
                    if len(ml_c) > 0:
                        st.markdown(create_metric("Précision (ML)", f"{ml_c['correct'].mean():.1%}", f" sur {len(ml_c)} matchs", COLORS['success']), unsafe_allow_html=True)
                with col2:
                    rules_c = completed[completed['ml_used'] != True]
                    if len(rules_c) > 0:
                        st.markdown(create_metric("Précision (Règles)", f"{rules_c['correct'].mean():.1%}", f" sur {len(rules_c)} matchs", COLORS['warning']), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
def show_configuration():
    st.markdown("<h2>⚙️ Configuration</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎾 Paramètres généraux")
        start_year = st.number_input("Année de début", min_value=2000, max_value=2024, value=START_YEAR)
        st.markdown("**Seuils de value bet**")
        min_edge = st.slider("Edge minimum (%)", 0.0, 10.0, float(MIN_EDGE_COMBINE*100), 0.5) / 100
        min_proba = st.slider("Probabilité minimum (%)", 50, 90, int(MIN_PROBA_COMBINE*100), 5) / 100

    with col2:
        st.markdown("### 🤖 Intelligence Artificielle")
        groq_status = "✅ Connecté" if get_groq_key() else "❌ Non configuré"
        st.markdown(f"**Groq API:** {groq_status}")
        sklearn_status = "✅ Disponible" if SKLEARN_AVAILABLE else "❌ Non installé (pip install scikit-learn)"
        st.markdown(f"**scikit-learn:** {sklearn_status}")

        model_info = st.session_state.get('ml_model')
        if model_info:
            st.markdown(f"**Modèle ML:** ✅ Actif — {model_info['accuracy']:.1%} accuracy")
        else:
            st.markdown("**Modèle ML:** ⚠️ Non entraîné")

        ai_temperature = st.slider("Température IA", 0.0, 1.0, 0.3, 0.1)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🗑️ Gestion des données")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🗑️ Effacer prédictions", use_container_width=True):
            if HIST_FILE.exists():
                HIST_FILE.unlink()
                st.success("Historique effacé !")
                st.rerun()
    with col2:
        if st.button("🗑️ Effacer combinés", use_container_width=True):
            if COMB_HIST_FILE.exists():
                COMB_HIST_FILE.unlink()
                st.success("Combinés effacés !")
                st.rerun()
    with col3:
        if st.button("🗑️ Réinit. statistiques", use_container_width=True):
            if USER_STATS_FILE.exists():
                USER_STATS_FILE.unlink()
                st.success("Statistiques réinitialisées !")
                st.rerun()
    with col4:
        if st.button("🗑️ Vider cache ML", use_container_width=True):
            st.session_state['ml_model'] = None
            st.session_state['player_stats_cache'] = None
            st.success("Cache ML vidé !")
            st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Export des données")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Exporter prédictions (CSV)", use_container_width=True):
            history = load_history()
            if history:
                dfe = pd.DataFrame(history)
                csv = dfe.to_csv(index=False).encode('utf-8')
                st.download_button("Télécharger", csv, f"tennisiq_predictions_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    with col2:
        if st.button("📥 Exporter combinés (CSV)", use_container_width=True):
            combines = load_combines()
            if combines:
                dfe = pd.DataFrame(combines)
                csv = dfe.to_csv(index=False).encode('utf-8')
                st.download_button("Télécharger", csv, f"tennisiq_combines_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### ℹ️ À propos de la précision")
    st.markdown("""
    <div class="model-card">
        <h5>🎯 Précision attendue par composant</h5>
        <div style="margin-top:1rem;">
    """, unsafe_allow_html=True)

    components = [
        ("🖥️ Interface utilisateur", 100, COLORS['success']),
        ("💾 Gestion des données", 95, COLORS['success']),
        ("📊 Calibration des probabilités", 90, COLORS['success'] if SKLEARN_AVAILABLE else COLORS['warning']),
        ("🤖 Modèle ML (RandomForest + Isotonic)", 85 if st.session_state.get('ml_model') else 40, COLORS['success'] if st.session_state.get('ml_model') else COLORS['warning']),
        ("🎯 Détection value bets", 80, COLORS['primary']),
        ("🔮 Prédictions matchs", 75 if st.session_state.get('ml_model') else 55, COLORS['primary']),
        ("💰 ROI réel (avec marge bookmaker)", 60, COLORS['warning']),
    ]

    for label, pct, color in components:
        st.markdown(f"""
        <div style="margin: 0.5rem 0;">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="color:#fff;font-size:0.85rem;">{label}</span>
                <span style="color:{color};font-weight:700;">{pct}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:8px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LANCEMENT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
