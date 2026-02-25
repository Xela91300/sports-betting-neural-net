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

# Création des dossiers
for dir_path in [MODELS_DIR, DATA_DIR, CACHE_DIR, HIST_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Fichiers d'historique
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

SURFACES = ["Hard", "Clay", "Grass"]
TOURS = {"ATP": "atp"}
ATP_ONLY = True
START_YEAR = 2007

# Configuration des couleurs professionnelles
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

# NOUVELLES CONSTANTES POUR LES LIMITES
MAX_MATCHES_ANALYSIS = 30  # Nombre maximum de matchs pour l'analyse
MAX_MATCHES_COMBINE = 30    # Nombre maximum de matchs pour le combiné
MIN_PROBA_COMBINE = 0.55    # Probabilité minimale pour une sélection
MIN_EDGE_COMBINE = 0.02     # Edge minimum pour une sélection
MAX_SELECTIONS_COMBINE = 30 # Nombre maximum de sélections dans le combiné

# Configuration des surfaces
SURFACE_CONFIG = {
    "Hard": {
        "color": COLORS["surface_hard"],
        "icon": "🟦",
        "description": "Surface dure - Jeu rapide"
    },
    "Clay": {
        "color": COLORS["surface_clay"],
        "icon": "🟧",
        "description": "Terre battue - Jeu lent"
    },
    "Grass": {
        "color": COLORS["surface_grass"],
        "icon": "🟩",
        "description": "Gazon - Jeu très rapide"
    }
}

# Configuration des niveaux de tournoi
LEVEL_CONFIG = {
    "G": {"name": "Grand Chelem", "color": "#FFD700", "icon": "🏆"},
    "M": {"name": "Masters 1000", "color": "#C0C0C0", "icon": "🥇"},
    "500": {"name": "ATP 500", "color": "#CD7F32", "icon": "🥈"},
    "A": {"name": "ATP 250", "color": "#6C7A89", "icon": "🎾"},
    "F": {"name": "ATP Finals", "color": "#9400D3", "icon": "👑"},
}

# Liste des tournois ATP
TOURNAMENTS_ATP = [
    # Grand Chelems
    ("Australian Open", "Hard", "G", 5),
    ("Roland Garros", "Clay", "G", 5),
    ("Wimbledon", "Grass", "G", 5),
    ("US Open", "Hard", "G", 5),
    # Masters 1000
    ("Indian Wells Masters", "Hard", "M", 3),
    ("Miami Open", "Hard", "M", 3),
    ("Monte-Carlo Masters", "Clay", "M", 3),
    ("Madrid Open", "Clay", "M", 3),
    ("Italian Open", "Clay", "M", 3),
    ("Canadian Open", "Hard", "M", 3),
    ("Cincinnati Masters", "Hard", "M", 3),
    ("Shanghai Masters", "Hard", "M", 3),
    ("Paris Masters", "Hard", "M", 3),
    # ATP 500
    ("Rotterdam", "Hard", "500", 3),
    ("Dubai Tennis Champs", "Hard", "500", 3),
    ("Acapulco", "Hard", "500", 3),
    ("Barcelona Open", "Clay", "500", 3),
    ("Halle Open", "Grass", "500", 3),
    ("Queen's Club", "Grass", "500", 3),
    ("Hamburg Open", "Clay", "500", 3),
    ("Washington Open", "Hard", "500", 3),
    ("Tokyo", "Hard", "500", 3),
    ("Vienna Open", "Hard", "500", 3),
    ("Basel", "Hard", "500", 3),
    ("Beijing", "Hard", "500", 3),
    # ATP Finals
    ("Nitto ATP Finals", "Hard", "F", 3),
    # ATP 250 (sélection)
    ("Brisbane International", "Hard", "A", 3),
    ("Adelaide International", "Hard", "A", 3),
    ("Auckland Open", "Hard", "A", 3),
    ("Montpellier", "Hard", "A", 3),
    ("Marseille", "Hard", "A", 3),
    ("Buenos Aires", "Clay", "A", 3),
    ("Estoril", "Clay", "A", 3),
    ("Munich", "Clay", "A", 3),
    ("Geneva", "Clay", "A", 3),
    ("Stuttgart", "Grass", "A", 3),
    ("Eastbourne", "Grass", "A", 3),
    ("Newport", "Grass", "A", 3),
    ("Bastad", "Clay", "A", 3),
    ("Kitzbuhel", "Clay", "A", 3),
    ("Los Cabos", "Hard", "A", 3),
    ("Atlanta", "Hard", "A", 3),
    ("Stockholm", "Hard", "A", 3),
    ("Antwerp", "Hard", "A", 3),
]

TOURN_DICT = {t[0]: (t[1], t[2], t[3]) for t in TOURNAMENTS_ATP}
TOURN_NAMES = [t[0] for t in TOURNAMENTS_ATP]

# Configuration de l'API
ODDS_API_KEY = "8090906fec7338245114345194fde760"
ODDS_CACHE = {}
ODDS_TTL = 6 * 3600

# ─────────────────────────────────────────────────────────────
# CSS PROFESSIONNEL
# ─────────────────────────────────────────────────────────────
def load_css():
    """Charge le CSS personnalisé"""
    css = """
    <style>
        /* Import des fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

        /* Variables globales */
        :root {
            --primary: #00DFA2;
            --primary-dark: #00B886;
            --secondary: #0079FF;
            --secondary-dark: #0063CC;
            --success: #00DFA2;
            --warning: #FFB200;
            --danger: #FF3B3F;
            --info: #0079FF;
            --dark: #0A1E2C;
            --dark-light: #1A2E3C;
            --light: #F5F9FF;
            --gray: #6C7A89;
            --gray-light: #E1E8F0;
            --white: #FFFFFF;
            --black: #000000;
            
            --surface-hard: #0079FF;
            --surface-clay: #E67E22;
            --surface-grass: #00DFA2;
            
            --shadow-sm: 0 2px 4px rgba(0,0,0,0.05);
            --shadow-md: 0 4px 6px rgba(0,0,0,0.07);
            --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
            --shadow-xl: 0 20px 25px rgba(0,0,0,0.15);
            
            --radius-sm: 4px;
            --radius-md: 8px;
            --radius-lg: 12px;
            --radius-xl: 16px;
            
            --transition: all 0.2s ease;
        }

        /* Style de base */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: linear-gradient(135deg, #0A1E2C 0%, #1A2E3C 100%);
        }

        /* Sidebar stylisée */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F2533 0%, #0A1E2C 100%);
            border-right: 1px solid rgba(255,255,255,0.05);
            box-shadow: var(--shadow-xl);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdown"] {
            color: var(--white);
        }

        /* Cards professionnelles */
        .card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: var(--transition);
        }

        .card:hover {
            border-color: rgba(255, 255, 255, 0.1);
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .card-glass {
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.03);
        }

        /* Badges */
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.3px;
            text-transform: uppercase;
            gap: 0.25rem;
            margin: 0.25rem;
        }

        .badge-hard {
            background: rgba(0, 121, 255, 0.1);
            color: var(--surface-hard);
            border: 1px solid rgba(0, 121, 255, 0.2);
        }

        .badge-clay {
            background: rgba(230, 126, 34, 0.1);
            color: var(--surface-clay);
            border: 1px solid rgba(230, 126, 34, 0.2);
        }

        .badge-grass {
            background: rgba(0, 223, 162, 0.1);
            color: var(--surface-grass);
            border: 1px solid rgba(0, 223, 162, 0.2);
        }

        .badge-gs {
            background: rgba(255, 215, 0, 0.1);
            color: #FFD700;
            border: 1px solid rgba(255, 215, 0, 0.2);
        }

        .badge-master {
            background: rgba(192, 192, 192, 0.1);
            color: #C0C0C0;
            border: 1px solid rgba(192, 192, 192, 0.2);
        }

        .badge-atp {
            background: rgba(0, 223, 162, 0.1);
            color: var(--primary);
            border: 1px solid rgba(0, 223, 162, 0.2);
        }

        .badge-wta {
            background: rgba(255, 75, 125, 0.1);
            color: #FF4B7D;
            border: 1px solid rgba(255, 75, 125, 0.2);
        }

        /* Barres de progression */
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            border-radius: 4px;
            transition: width 0.5s ease;
        }

        /* Métriques */
        .metric-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: var(--radius-md);
            padding: 1rem;
            text-align: center;
        }

        .metric-label {
            font-size: 0.7rem;
            color: var(--gray);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--white);
            line-height: 1.2;
        }

        .metric-unit {
            font-size: 0.8rem;
            color: var(--gray);
            margin-left: 0.25rem;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: transparent;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: var(--gray) !important;
            font-size: 0.85rem;
            font-weight: 500;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            padding: 0.75rem 1.5rem;
            border-radius: 0 !important;
            border-bottom: 2px solid transparent !important;
            transition: var(--transition);
        }

        .stTabs [aria-selected="true"] {
            color: var(--primary) !important;
            border-bottom: 2px solid var(--primary) !important;
        }

        /* Boutons */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
            color: var(--white) !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            letter-spacing: 0.5px !important;
            padding: 0.75rem 2rem !important;
            transition: var(--transition) !important;
            text-transform: uppercase !important;
            box-shadow: var(--shadow-md) !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-lg) !important;
            filter: brightness(1.1) !important;
        }

        .stButton > button:disabled {
            opacity: 0.5 !important;
            cursor: not-allowed !important;
        }

        /* Inputs */
        .stTextInput > div > div {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: var(--radius-md) !important;
            color: var(--white) !important;
        }

        .stTextInput > div > div:focus-within {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 2px rgba(0, 223, 162, 0.1) !important;
        }

        /* Selectbox */
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: var(--radius-md) !important;
            color: var(--white) !important;
        }

        /* Slider */
        .stSlider [data-baseweb="slider"] {
            background: transparent !important;
        }

        .stSlider [data-baseweb="thumb"] {
            background: var(--primary) !important;
            border: 2px solid var(--white) !important;
            width: 20px !important;
            height: 20px !important;
        }

        /* Expander */
        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: var(--radius-lg) !important;
        }

        [data-testid="stExpander"] summary {
            padding: 1rem !important;
            font-weight: 600 !important;
            color: var(--white) !important;
        }

        /* Dataframe */
        [data-testid="stDataFrame"] {
            background: rgba(255, 255, 255, 0.02) !important;
            border-radius: var(--radius-lg) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
        }

        .dataframe {
            font-family: 'Inter', sans-serif !important;
        }

        .dataframe th {
            background: rgba(0, 223, 162, 0.1) !important;
            color: var(--primary) !important;
            font-weight: 600 !important;
            font-size: 0.8rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }

        .dataframe td {
            color: var(--white) !important;
            font-size: 0.9rem !important;
        }

        /* Alertes */
        .stAlert {
            background: rgba(255, 255, 255, 0.02) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-left: 4px solid var(--primary) !important;
            border-radius: var(--radius-md) !important;
            color: var(--white) !important;
        }

        /* Progress bars */
        .stProgress > div > div {
            background-color: var(--primary) !important;
        }

        /* Checkbox */
        .stCheckbox [data-baseweb="checkbox"] {
            background: rgba(255, 255, 255, 0.02) !important;
            border-color: rgba(255, 255, 255, 0.1) !important;
        }

        .stCheckbox [data-checked="true"] {
            background: var(--primary) !important;
            border-color: var(--primary) !important;
        }

        /* Radio */
        .stRadio [data-baseweb="radio"] {
            background: rgba(255, 255, 255, 0.02) !important;
            border-color: rgba(255, 255, 255, 0.1) !important;
        }

        .stRadio [data-checked="true"] {
            background: var(--primary) !important;
            border-color: var(--primary) !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: var(--gray);
            font-size: 0.8rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            margin-top: 3rem;
        }

        /* Header */
        .header {
            padding: 2rem 0 1rem 0;
            text-align: center;
        }

        .header-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }

        .header-subtitle {
            color: var(--gray);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 3px;
        }

        /* Stat rows */
        .stat-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .stat-row:last-child {
            border-bottom: none;
        }

        .stat-key {
            color: var(--gray);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .stat-value {
            color: var(--white);
            font-weight: 600;
            font-size: 0.95rem;
        }

        .stat-value-green {
            color: var(--success);
        }

        .stat-value-red {
            color: var(--danger);
        }

        .stat-value-warning {
            color: var(--warning);
        }

        /* Divider */
        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            margin: 2rem 0;
        }

        /* Grid */
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .grid-3 {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 1rem;
        }

        .grid-4 {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            gap: 1rem;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .grid-2, .grid-3, .grid-4 {
                grid-template-columns: 1fr;
            }
            
            .header-title {
                font-size: 2rem;
            }
        }

        /* Loading spinner */
        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Success/Error messages */
        .message-success {
            background: rgba(0, 223, 162, 0.1);
            border: 1px solid rgba(0, 223, 162, 0.2);
            border-left: 4px solid var(--success);
            color: var(--success);
            padding: 1rem;
            border-radius: var(--radius-md);
            margin: 1rem 0;
        }

        .message-error {
            background: rgba(255, 59, 63, 0.1);
            border: 1px solid rgba(255, 59, 63, 0.2);
            border-left: 4px solid var(--danger);
            color: var(--danger);
            padding: 1rem;
            border-radius: var(--radius-md);
            margin: 1rem 0;
        }

        .message-warning {
            background: rgba(255, 178, 0, 0.1);
            border: 1px solid rgba(255, 178, 0, 0.2);
            border-left: 4px solid var(--warning);
            color: var(--warning);
            padding: 1rem;
            border-radius: var(--radius-md);
            margin: 1rem 0;
        }

        .message-info {
            background: rgba(0, 121, 255, 0.1);
            border: 1px solid rgba(0, 121, 255, 0.2);
            border-left: 4px solid var(--info);
            color: var(--info);
            padding: 1rem;
            border-radius: var(--radius-md);
            margin: 1rem 0;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--white);
            font-weight: 600;
            margin-bottom: 1rem;
        }

        h1 { font-size: 2.5rem; }
        h2 { font-size: 2rem; }
        h3 { font-size: 1.5rem; }
        h4 { font-size: 1.25rem; }
        h5 { font-size: 1.1rem; }
        h6 { font-size: 1rem; }

        p {
            color: var(--gray);
            line-height: 1.6;
        }

        a {
            color: var(--primary);
            text-decoration: none;
            transition: var(--transition);
        }

        a:hover {
            color: var(--primary-dark);
            text-decoration: underline;
        }

        /* Code blocks */
        code {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(255, 255, 255, 0.05);
            color: var(--primary);
            padding: 0.2rem 0.4rem;
            border-radius: var(--radius-sm);
            font-size: 0.9rem;
        }

        pre {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: var(--radius-md);
            padding: 1rem;
            overflow-x: auto;
        }

        pre code {
            background: transparent;
            padding: 0;
        }

        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
        }

        th {
            text-align: left;
            padding: 0.75rem;
            background: rgba(255, 255, 255, 0.02);
            color: var(--primary);
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        td {
            padding: 0.75rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Images */
        img {
            max-width: 100%;
            border-radius: var(--radius-md);
        }

        /* Utilitaires */
        .text-center { text-align: center; }
        .text-left { text-align: left; }
        .text-right { text-align: right; }
        
        .mt-1 { margin-top: 0.5rem; }
        .mt-2 { margin-top: 1rem; }
        .mt-3 { margin-top: 1.5rem; }
        .mt-4 { margin-top: 2rem; }
        .mt-5 { margin-top: 2.5rem; }
        
        .mb-1 { margin-bottom: 0.5rem; }
        .mb-2 { margin-bottom: 1rem; }
        .mb-3 { margin-bottom: 1.5rem; }
        .mb-4 { margin-bottom: 2rem; }
        .mb-5 { margin-bottom: 2.5rem; }
        
        .p-1 { padding: 0.5rem; }
        .p-2 { padding: 1rem; }
        .p-3 { padding: 1.5rem; }
        .p-4 { padding: 2rem; }
        .p-5 { padding: 2.5rem; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Charger le CSS
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
    """Récupère la clé API Groq"""
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        import os
        return os.environ.get("GROQ_API_KEY", None)

def call_groq_api(prompt):
    """Appelle l'API Groq"""
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
            max_tokens=800,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur API: {str(e)}"

# ─────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────
def format_number(num, decimals=2):
    """Formate un nombre avec séparateur de milliers"""
    if num is None or pd.isna(num):
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
    """Formate un pourcentage"""
    if num is None or pd.isna(num):
        return "—"
    return f"{num:.{decimals}%}"

def format_date(date_str):
    """Formate une date"""
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime("%d %b %Y %H:%M")
    except:
        return date_str

def create_progress_bar(value, color=COLORS["primary"]):
    """Crée une barre de progression HTML"""
    return f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {value*100}%; background: linear-gradient(90deg, {color}, {COLORS['secondary']});"></div>
    </div>
    """

def create_badge(text, type="primary"):
    """Crée un badge HTML"""
    colors = {
        "primary": COLORS["primary"],
        "secondary": COLORS["secondary"],
        "success": COLORS["success"],
        "warning": COLORS["warning"],
        "danger": COLORS["danger"],
        "info": COLORS["info"],
        "hard": COLORS["surface_hard"],
        "clay": COLORS["surface_clay"],
        "grass": COLORS["surface_grass"],
    }
    color = colors.get(type, COLORS["primary"])
    bg_color = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)"
    return f"""
    <span class="badge" style="background: {bg_color}; color: {color}; border: 1px solid {bg_color};">
        {text}
    </span>
    """

def create_metric(label, value, unit="", color=COLORS["white"]):
    """Crée une métrique HTML"""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color};">{value}<span class="metric-unit">{unit}</span></div>
    </div>
    """

def create_stat_row(key, value, value_color=COLORS["white"]):
    """Crée une ligne de statistique"""
    return f"""
    <div class="stat-row">
        <span class="stat-key">{key}</span>
        <span class="stat-value" style="color: {value_color};">{value}</span>
    </div>
    """

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES ATP UNIQUEMENT
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_atp_data():
    """Charge uniquement les données ATP sans afficher les logs"""
    if not DATA_DIR.exists():
        return None
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        return None
    
    atp_dfs = []
    
    for f in csv_files:
        # Ne charger que les fichiers ATP (ignorer WTA)
        if 'wta' in f.name.lower():
            continue
            
        try:
            # Essayer différents délimiteurs et encodages
            df = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    # Essayer avec la détection automatique
                    df = pd.read_csv(f, encoding=encoding, on_bad_lines='skip', low_memory=False)
                    break
                except:
                    try:
                        # Essayer avec point-virgule comme délimiteur
                        df = pd.read_csv(f, sep=';', encoding=encoding, on_bad_lines='skip', low_memory=False)
                        break
                    except:
                        continue
            
            if df is not None and 'winner_name' in df.columns and 'loser_name' in df.columns:
                atp_dfs.append(df)
                
        except Exception:
            # Ignorer silencieusement les erreurs
            continue
    
    if atp_dfs:
        atp_data = pd.concat(atp_dfs, ignore_index=True)
        return atp_data
    else:
        return None

# ─────────────────────────────────────────────────────────────
# FONCTIONS DE CALCUL
# ─────────────────────────────────────────────────────────────
def get_player_stats(df, player, surface=None, n_matches=20):
    """Calcule les statistiques d'un joueur"""
    if df is None or player is None:
        return None
    
    # Nettoyer le nom du joueur
    player_clean = player.strip() if isinstance(player, str) else player
    
    # Vérifier les colonnes disponibles
    winner_col = 'winner_name' if 'winner_name' in df.columns else None
    loser_col = 'loser_name' if 'loser_name' in df.columns else None
    
    if not winner_col or not loser_col:
        return None
    
    # Nettoyer les noms dans le dataframe pour la comparaison
    df_winner_clean = df[winner_col].astype(str).str.strip()
    df_loser_clean = df[loser_col].astype(str).str.strip()
    
    # Filtrer les matchs du joueur
    matches = df[(df_winner_clean == player_clean) | (df_loser_clean == player_clean)].copy()
    if len(matches) == 0:
        return None
    
    # Statistiques de base
    stats = {
        'name': player_clean,
        'matches_played': len(matches),
        'wins': len(matches[df_winner_clean == player_clean]),
        'losses': len(matches[df_loser_clean == player_clean]),
    }
    
    # Win rate
    stats['win_rate'] = stats['wins'] / stats['matches_played'] if stats['matches_played'] > 0 else 0
    
    return stats

def get_h2h_stats(df, player1, player2):
    """Calcule les statistiques H2H"""
    if df is None or player1 is None or player2 is None:
        return None
    
    # Nettoyer les noms
    player1_clean = player1.strip() if isinstance(player1, str) else player1
    player2_clean = player2.strip() if isinstance(player2, str) else player2
    
    winner_col = 'winner_name' if 'winner_name' in df.columns else None
    loser_col = 'loser_name' if 'loser_name' in df.columns else None
    
    if not winner_col or not loser_col:
        return None
    
    # Nettoyer les noms dans le dataframe
    df_winner_clean = df[winner_col].astype(str).str.strip()
    df_loser_clean = df[loser_col].astype(str).str.strip()
    
    h2h = df[((df_winner_clean == player1_clean) & (df_loser_clean == player2_clean)) |
             ((df_winner_clean == player2_clean) & (df_loser_clean == player1_clean))].copy()
    
    if len(h2h) == 0:
        return None
    
    stats = {
        'total_matches': len(h2h),
        f'{player1_clean}_wins': len(h2h[df_winner_clean == player1_clean]),
        f'{player2_clean}_wins': len(h2h[df_winner_clean == player2_clean]),
    }
    
    return stats

def calculate_probability(stats1, stats2, h2h, surface):
    """Calcule la probabilité de victoire"""
    if stats1 is None or stats2 is None:
        return 0.5
    
    score = 0.5
    
    # Facteur forme (win rate)
    score += (stats1['win_rate'] - stats2['win_rate']) * 0.3
    
    # Facteur H2H
    if h2h and h2h['total_matches'] > 0:
        wins1 = h2h.get(f'{stats1["name"]}_wins', 0)
        total = h2h['total_matches']
        score += (wins1 / total - 0.5) * 0.2
    
    # Normalisation
    score = max(0.05, min(0.95, score))
    
    return score

def calculate_confidence(proba, stats1, stats2, h2h):
    """Calcule le score de confiance"""
    confidence = 50
    
    if stats1 and stats2:
        # Plus de matchs = plus de confiance
        confidence += min(stats1['matches_played'] / 20, 20)
        confidence += min(stats2['matches_played'] / 20, 20)
    
    if h2h and h2h['total_matches'] >= 3:
        confidence += 10
    
    # Proba extrême = plus de confiance
    confidence += abs(proba - 0.5) * 40
    
    return min(100, confidence)

# ─────────────────────────────────────────────────────────────
# GESTION DE L'HISTORIQUE
# ─────────────────────────────────────────────────────────────
def load_history():
    """Charge l'historique des prédictions"""
    if not HIST_FILE.exists():
        return []
    try:
        with open(HIST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_prediction(pred_data):
    """Sauvegarde une prédiction"""
    history = load_history()
    
    # Ajouter la date si non présente
    if 'date' not in pred_data:
        pred_data['date'] = datetime.now().isoformat()
    
    # Ajouter un ID unique
    pred_data['id'] = hashlib.md5(f"{pred_data['date']}{pred_data.get('player1', '')}{pred_data.get('player2', '')}".encode()).hexdigest()[:8]
    
    history.append(pred_data)
    
    # Limiter à 1000 entrées
    if len(history) > 1000:
        history = history[-1000:]
    
    try:
        with open(HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except:
        return False

def load_combines():
    """Charge l'historique des combinés"""
    if not COMB_HIST_FILE.exists():
        return []
    try:
        with open(COMB_HIST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_combine(combine_data):
    """Sauvegarde un combiné"""
    combines = load_combines()
    
    # Ajouter la date
    combine_data['date'] = datetime.now().isoformat()
    
    # Ajouter un ID unique
    combine_data['id'] = hashlib.md5(f"{combine_data['date']}{len(combines)}".encode()).hexdigest()[:8]
    
    combines.append(combine_data)
    
    # Limiter à 200 combinés
    if len(combines) > 200:
        combines = combines[-200:]
    
    try:
        with open(COMB_HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(combines, f, indent=2, ensure_ascii=False)
        return True
    except:
        return False

def load_user_stats():
    """Charge les statistiques utilisateur"""
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
    """Fonction principale"""
    
    # Header
    st.markdown("""
    <div class="header">
        <div class="header-title">TennisIQ Pro</div>
        <div class="header-subtitle">Intelligence Artificielle pour le Tennis</div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)
    
    # Chargement des données ATP uniquement
    with st.spinner("Chargement des données..."):
        atp_data = load_atp_data()
    
    # Sidebar - Navigation
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
        
        # Menu de navigation
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🎯 Prédictions", "📊 Multi-matchs", "🎰 Combinés", "📜 Historique", "📈 Statistiques", "⚙️ Configuration"],
            label_visibility="collapsed"
        )
        
        # Informations système
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        if atp_data is not None:
            st.markdown(create_badge(f"ATP: {len(atp_data):,} matchs", "primary"), unsafe_allow_html=True)
        else:
            st.markdown(create_badge("ATP: 0 matchs", "danger"), unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Version
        st.markdown("""
        <div style="text-align: center; color: #6C7A89; font-size: 0.7rem;">
            Version 2.0.0<br>
            © 2024 TennisIQ Pro
        </div>
        """, unsafe_allow_html=True)
    
    # Routes
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
    elif page == "⚙️ Configuration":
        show_configuration()

# ─────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────
def show_dashboard(atp_data):
    """Affiche le dashboard principal"""
    
    st.markdown("<h2>🏠 Tableau de Bord</h2>", unsafe_allow_html=True)
    
    # Statistiques rapides
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_metric("Matchs ATP", format_number(len(atp_data) if atp_data is not None else 0), ""), unsafe_allow_html=True)
    
    with col2:
        history = load_history()
        st.markdown(create_metric("Prédictions", format_number(len(history)), ""), unsafe_allow_html=True)
    
    with col3:
        stats = load_user_stats()
        accuracy = (stats.get('correct_predictions', 0) / stats.get('total_predictions', 1)) * 100 if stats.get('total_predictions', 0) > 0 else 0
        st.markdown(create_metric("Précision", f"{accuracy:.1f}", "%", COLORS['success'] if accuracy >= 60 else COLORS['warning']), unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Graphiques simples avec st.bar_chart
    if atp_data is not None and 'surface' in atp_data.columns:
        st.markdown("<h3>📊 Répartition des surfaces</h3>", unsafe_allow_html=True)
        surface_counts = atp_data['surface'].value_counts()
        st.bar_chart(surface_counts)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Dernières prédictions
    st.markdown("<h3>📋 Dernières prédictions</h3>", unsafe_allow_html=True)
    
    history = load_history()
    if history:
        df_history = pd.DataFrame(history[-10:])
        if 'date' in df_history.columns:
            df_history['date'] = pd.to_datetime(df_history['date']).dt.strftime('%d/%m/%Y %H:%M')
        df_history['match'] = df_history.get('player1', '') + " vs " + df_history.get('player2', '')
        if 'proba' in df_history.columns:
            df_history['proba'] = df_history['proba'].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else x)
        if 'confidence' in df_history.columns:
            df_history['confiance'] = df_history['confidence'].apply(lambda x: f"{x}/100" if isinstance(x, (int, float)) else x)
        
        display_cols = []
        if 'date' in df_history.columns:
            display_cols.append('date')
        if 'match' in df_history.columns:
            display_cols.append('match')
        if 'proba' in df_history.columns:
            display_cols.append('proba')
        if 'confiance' in df_history.columns:
            display_cols.append('confiance')
        if 'surface' in df_history.columns:
            display_cols.append('surface')
        
        if display_cols:
            st.dataframe(
                df_history[display_cols].rename(columns={
                    'date': 'Date',
                    'match': 'Match',
                    'proba': 'Probabilité',
                    'confiance': 'Confiance',
                    'surface': 'Surface'
                }),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Aucune prédiction pour le moment. Commence par en faire une !")

# ─────────────────────────────────────────────────────────────
# PRÉDICTIONS (CORRIGÉ AVEC STRIP)
# ─────────────────────────────────────────────────────────────
def show_predictions(atp_data):
    """Affiche l'interface de prédiction simple"""
    
    st.markdown("<h2>🎯 Prédiction Simple</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Initialisation des variables
    player1 = None
    player2 = None
    tournament = None
    surface = "Hard"
    df = atp_data
    
    with col1:
        if df is not None and not df.empty:
            # Vérifier les colonnes disponibles
            winner_col = 'winner_name' if 'winner_name' in df.columns else None
            loser_col = 'loser_name' if 'loser_name' in df.columns else None
            
            if winner_col and loser_col:
                # Liste des joueurs (avec strip pour nettoyer les noms)
                players = sorted(set(str(p).strip() for p in df[winner_col].dropna().unique() if pd.notna(p)) | 
                               set(str(p).strip() for p in df[loser_col].dropna().unique() if pd.notna(p)))
                
                if players:
                    # Sélection des joueurs
                    player1 = st.selectbox("Joueur 1", players, key="pred_p1")
                    
                    # Filtrer les joueurs pour éviter de sélectionner le même
                    players2 = [p for p in players if p != player1]
                    player2 = st.selectbox("Joueur 2", players2, key="pred_p2")
                    
                    # Sélection du tournoi
                    if 'tourney_name' in df.columns:
                        tournaments = sorted(df['tourney_name'].dropna().unique())
                        tournament = st.selectbox("Tournoi", tournaments) if tournaments else None
                        
                        # Récupérer la surface
                        if tournament and 'surface' in df.columns:
                            surface_df = df[df['tourney_name'] == tournament]['surface']
                            if not surface_df.empty:
                                surface = surface_df.iloc[0]
                    
                    # Afficher la surface
                    if surface in SURFACE_CONFIG:
                        st.markdown(create_badge(f"{SURFACE_CONFIG[surface]['icon']} {surface}", surface.lower()), unsafe_allow_html=True)
    
    with col2:
        if player1 and player2:
            # Nettoyer les noms des joueurs pour l'affichage
            player1_clean = player1.strip()
            player2_clean = player2.strip()
            
            # Calcul des statistiques
            stats1 = get_player_stats(df, player1_clean, surface)
            stats2 = get_player_stats(df, player2_clean, surface)
            h2h = get_h2h_stats(df, player1_clean, player2_clean)
            
            # Probabilité
            proba = calculate_probability(stats1, stats2, h2h, surface)
            confidence = calculate_confidence(proba, stats1, stats2, h2h)
            
            # Affichage des résultats
            st.markdown("<h3 style='text-align: center;'>Résultat</h3>", unsafe_allow_html=True)
            
            # Barre de progression avec noms nettoyés
            st.markdown(f"""
            <div style="text-align: center; margin: 2rem 0;">
                <div style="font-size: 1.2rem; color: #6C7A89; margin-bottom: 1rem;">{player1_clean}</div>
                <div style="font-size: 3rem; font-weight: 800; color: {COLORS['primary']};">{proba:.1%}</div>
                {create_progress_bar(proba)}
                <div style="font-size: 1.2rem; color: #6C7A89; margin-top: 1rem;">{player2_clean}</div>
                <div style="font-size: 1rem; color: {COLORS['gray']};">{(1-proba):.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confiance
            conf_color = COLORS['success'] if confidence >= 70 else COLORS['warning'] if confidence >= 50 else COLORS['danger']
            st.markdown(create_metric("Confiance", f"{confidence:.0f}", "/100", conf_color), unsafe_allow_html=True)
            
            # Bouton de sauvegarde
            if st.button("💾 Sauvegarder la prédiction", use_container_width=True):
                pred_data = {
                    'player1': player1_clean,
                    'player2': player2_clean,
                    'tournament': tournament if tournament else "Inconnu",
                    'surface': surface,
                    'proba': proba,
                    'confidence': confidence,
                    'circuit': "ATP"
                }
                if save_prediction(pred_data):
                    st.success("Prédiction sauvegardée !")
                else:
                    st.error("Erreur lors de la sauvegarde")
    
    if player1 and player2 and 'df' in locals():
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Nettoyer les noms pour l'affichage des stats
        player1_clean = player1.strip()
        player2_clean = player2.strip()
        
        # Détails des statistiques
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"<h4>{player1_clean}</h4>", unsafe_allow_html=True)
            if stats1:
                st.markdown(create_stat_row("Matchs joués", stats1['matches_played']), unsafe_allow_html=True)
                st.markdown(create_stat_row("Victoires", stats1['wins']), unsafe_allow_html=True)
                st.markdown(create_stat_row("Défaites", stats1['losses']), unsafe_allow_html=True)
                st.markdown(create_stat_row("Win rate", f"{stats1['win_rate']:.1%}"), unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<h4>{player2_clean}</h4>", unsafe_allow_html=True)
            if stats2:
                st.markdown(create_stat_row("Matchs joués", stats2['matches_played']), unsafe_allow_html=True)
                st.markdown(create_stat_row("Victoires", stats2['wins']), unsafe_allow_html=True)
                st.markdown(create_stat_row("Défaites", stats2['losses']), unsafe_allow_html=True)
                st.markdown(create_stat_row("Win rate", f"{stats2['win_rate']:.1%}"), unsafe_allow_html=True)
        
        with col3:
            st.markdown("<h4>Face à Face</h4>", unsafe_allow_html=True)
            if h2h:
                st.markdown(create_stat_row("Matchs", h2h['total_matches']), unsafe_allow_html=True)
                st.markdown(create_stat_row(f"{player1_clean}", h2h.get(f'{player1_clean}_wins', 0)), unsafe_allow_html=True)
                st.markdown(create_stat_row(f"{player2_clean}", h2h.get(f'{player2_clean}_wins', 0)), unsafe_allow_html=True)
            else:
                st.info("Aucun face-à-face")

# ─────────────────────────────────────────────────────────────
# MULTI-MATCHS (MODIFIÉ AVEC MAX 30)
# ─────────────────────────────────────────────────────────────
def show_multimatches(atp_data):
    """Affiche l'interface multi-matchs avec max 30 matchs"""
    
    st.markdown("<h2>📊 Multi-matchs</h2>", unsafe_allow_html=True)
    
    # Configuration avec max 30 matchs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_matches = st.number_input(
            "Nombre de matchs", 
            min_value=2, 
            max_value=MAX_MATCHES_ANALYSIS, 
            value=min(5, MAX_MATCHES_ANALYSIS),
            help=f"Maximum {MAX_MATCHES_ANALYSIS} matchs"
        )
    
    with col2:
        use_ai = st.checkbox("Activer l'analyse IA", value=True)
    
    with col3:
        show_details = st.checkbox("Afficher les détails", value=False)
    
    df = atp_data
    
    if df is not None and not df.empty:
        winner_col = 'winner_name' if 'winner_name' in df.columns else None
        loser_col = 'loser_name' if 'loser_name' in df.columns else None
        
        if winner_col and loser_col:
            # Nettoyer les noms des joueurs
            players = sorted(set(str(p).strip() for p in df[winner_col].dropna().unique() if pd.notna(p)) | 
                           set(str(p).strip() for p in df[loser_col].dropna().unique() if pd.notna(p)))
            
            tournaments = []
            if 'tourney_name' in df.columns:
                tournaments = sorted(df['tourney_name'].dropna().unique())
            
            matches = []
            
            # Interface de saisie avec scroll si beaucoup de matchs
            for i in range(n_matches):
                with st.expander(f"Match {i+1}", expanded=i==0 and n_matches <= 10):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        p1 = st.selectbox(f"Joueur 1", players, key=f"mm_p1_{i}")
                    
                    with col2:
                        players2 = [p for p in players if p != p1]
                        p2 = st.selectbox(f"Joueur 2", players2, key=f"mm_p2_{i}")
                    
                    with col3:
                        tourn = st.selectbox(f"Tournoi", tournaments if tournaments else ["Inconnu"], key=f"mm_tourn_{i}")
                    
                    # Récupérer la surface
                    surface = "Hard"
                    if tourn and tourn != "Inconnu" and 'surface' in df.columns:
                        surface_df = df[df['tourney_name'] == tourn]['surface']
                        if not surface_df.empty:
                            surface = surface_df.iloc[0]
                    
                    # Cotes
                    col1, col2 = st.columns(2)
                    with col1:
                        odds1 = st.text_input(f"Cote {p1}", key=f"mm_odds1_{i}", placeholder="1.75")
                    with col2:
                        odds2 = st.text_input(f"Cote {p2}", key=f"mm_odds2_{i}", placeholder="2.10")
                    
                    # Afficher la surface
                    if surface in SURFACE_CONFIG:
                        st.markdown(create_badge(f"{SURFACE_CONFIG[surface]['icon']} {surface}", surface.lower()), unsafe_allow_html=True)
                    
                    matches.append({
                        'player1': p1.strip() if p1 else None,
                        'player2': p2.strip() if p2 else None,
                        'tournament': tourn,
                        'surface': surface,
                        'odds1': odds1,
                        'odds2': odds2,
                        'stats1': get_player_stats(df, p1.strip() if p1 else None, surface) if p1 else None,
                        'stats2': get_player_stats(df, p2.strip() if p2 else None, surface) if p2 else None,
                        'h2h': get_h2h_stats(df, p1.strip() if p1 else None, p2.strip() if p2 else None) if p1 and p2 else None
                    })
            
            if st.button(f"🔍 Analyser {n_matches} matchs", use_container_width=True):
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, match in enumerate(matches):
                    status_text.text(f"Analyse du match {i+1}/{n_matches}...")
                    
                    if match['player1'] and match['player2']:
                        proba = calculate_probability(match['stats1'], match['stats2'], match['h2h'], match['surface'])
                        confidence = calculate_confidence(proba, match['stats1'], match['stats2'], match['h2h'])
                        
                        results.append({
                            'match': i+1,
                            'player1': match['player1'],
                            'player2': match['player2'],
                            'tournament': match['tournament'],
                            'surface': match['surface'],
                            'proba': proba,
                            'confidence': confidence,
                            'odds1': match['odds1'],
                            'odds2': match['odds2']
                        })
                    
                    progress_bar.progress((i + 1) / n_matches)
                
                status_text.empty()
                progress_bar.empty()
                
                if results:
                    # Tableau des résultats
                    df_results = pd.DataFrame(results)
                    if 'proba' in df_results.columns:
                        df_results['proba'] = df_results['proba'].apply(lambda x: f"{x:.1%}")
                    if 'confidence' in df_results.columns:
                        df_results['confidence'] = df_results['confidence'].apply(lambda x: f"{x:.0f}/100")
                    
                    st.markdown(f"**Résultats de l'analyse ({len(results)} matchs)**")
                    st.dataframe(
                        df_results[['match', 'player1', 'player2', 'tournament', 'surface', 'proba', 'confidence']].rename(columns={
                            'match': '#',
                            'player1': 'Joueur 1',
                            'player2': 'Joueur 2',
                            'tournament': 'Tournoi',
                            'surface': 'Surface',
                            'proba': 'Probabilité',
                            'confidence': 'Confiance'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Analyses IA si activé
                    if use_ai and GROQ_AVAILABLE and show_details:
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        st.markdown("<h3>🤖 Analyses IA</h3>", unsafe_allow_html=True)
                        
                        for result in results:
                            prompt = f"Analyse le match de tennis entre {result['player1']} et {result['player2']} sur surface {result['surface']}. La probabilité de victoire de {result['player1']} est de {result['proba']}. Donne une analyse concise en 3 points."
                            
                            with st.spinner(f"Analyse du match {result['match']}..."):
                                analysis = call_groq_api(prompt)
                            
                            if analysis:
                                with st.expander(f"Match {result['match']}: {result['player1']} vs {result['player2']}"):
                                    st.markdown(analysis)

# ─────────────────────────────────────────────────────────────
# COMBINÉS (MODIFIÉ POUR 30 MATCHS MAX)
# ─────────────────────────────────────────────────────────────
def show_combines(atp_data):
    """Affiche l'interface des combinés avec max 30 sélections"""
    
    st.markdown("<h2>🎰 Générateur de Combinés</h2>", unsafe_allow_html=True)
    
    # Configuration
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_matches = st.number_input(
            "Matchs à analyser", 
            min_value=2, 
            max_value=MAX_MATCHES_COMBINE, 
            value=min(5, MAX_MATCHES_COMBINE),
            help=f"Maximum {MAX_MATCHES_COMBINE} matchs"
        )
    
    with col2:
        mise = st.number_input("Mise (€)", min_value=1.0, max_value=10000.0, value=10.0, step=5.0)
    
    with col3:
        use_ai = st.checkbox("Analyses IA", value=True)
    
    with col4:
        auto_select = st.checkbox("Auto-sélection", value=True, help="Sélection automatique des meilleurs value bets")
    
    df = atp_data
    
    if df is not None and not df.empty:
        winner_col = 'winner_name' if 'winner_name' in df.columns else None
        loser_col = 'loser_name' if 'loser_name' in df.columns else None
        
        if winner_col and loser_col:
            # Nettoyer les noms des joueurs
            players = sorted(set(str(p).strip() for p in df[winner_col].dropna().unique() if pd.notna(p)) | 
                           set(str(p).strip() for p in df[loser_col].dropna().unique() if pd.notna(p)))
            
            tournaments = []
            if 'tourney_name' in df.columns:
                tournaments = sorted(df['tourney_name'].dropna().unique())
            
            matches = []
            
            # Interface de saisie avec scroll si beaucoup de matchs
            st.markdown(f"### Saisie des {n_matches} matchs")
            
            for i in range(n_matches):
                with st.container():
                    st.markdown(f"**Match {i+1}**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        p1 = st.selectbox(f"J1", players, key=f"comb_p1_{i}", label_visibility="collapsed", placeholder="Joueur 1")
                    
                    with col2:
                        players2 = [p for p in players if p != p1]
                        p2 = st.selectbox(f"J2", players2, key=f"comb_p2_{i}", label_visibility="collapsed", placeholder="Joueur 2")
                    
                    with col3:
                        tourn = st.selectbox(f"T", tournaments if tournaments else ["Inconnu"], key=f"comb_tourn_{i}", label_visibility="collapsed", placeholder="Tournoi")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        odds1 = st.text_input(f"Cote {p1 if p1 else 'J1'}", key=f"comb_odds1_{i}", placeholder="1.75")
                    with col2:
                        odds2 = st.text_input(f"Cote {p2 if p2 else 'J2'}", key=f"comb_odds2_{i}", placeholder="2.10")
                    
                    surface = "Hard"
                    if tourn and tourn != "Inconnu" and 'surface' in df.columns:
                        surface_df = df[df['tourney_name'] == tourn]['surface']
                        if not surface_df.empty:
                            surface = surface_df.iloc[0]
                    
                    if surface in SURFACE_CONFIG:
                        st.markdown(create_badge(surface, surface.lower()), unsafe_allow_html=True)
                    
                    if i < n_matches - 1:
                        st.markdown("---")
                    
                    matches.append({
                        'player1': p1.strip() if p1 else None,
                        'player2': p2.strip() if p2 else None,
                        'tournament': tourn,
                        'surface': surface,
                        'odds1': odds1,
                        'odds2': odds2,
                        'stats1': get_player_stats(df, p1.strip() if p1 else None, surface) if p1 else None,
                        'stats2': get_player_stats(df, p2.strip() if p2 else None, surface) if p2 else None,
                        'h2h': get_h2h_stats(df, p1.strip() if p1 else None, p2.strip() if p2 else None) if p1 and p2 else None
                    })
            
            if st.button("🎯 Générer le meilleur combiné", use_container_width=True):
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                # Calculer les probabilités et edges
                selections = []
                invalid_matches = 0
                
                with st.spinner("Analyse des matchs en cours..."):
                    for match in matches:
                        if match['player1'] and match['player2'] and match['odds1'] and match['odds2']:
                            try:
                                odds1 = float(match['odds1'].replace(',', '.'))
                                odds2 = float(match['odds2'].replace(',', '.'))
                                
                                proba = calculate_probability(match['stats1'], match['stats2'], match['h2h'], match['surface'])
                                
                                # Edge pour chaque joueur
                                edge1 = proba - 1/odds1
                                edge2 = (1 - proba) - 1/odds2
                                
                                if auto_select:
                                    # Sélection automatique des meilleurs edges positifs
                                    if edge1 > MIN_EDGE_COMBINE and proba >= MIN_PROBA_COMBINE:
                                        selections.append({
                                            'match': f"{match['player1']} vs {match['player2']}",
                                            'joueur': match['player1'],
                                            'proba': proba,
                                            'cote': odds1,
                                            'edge': edge1,
                                            'surface': match['surface']
                                        })
                                    elif edge2 > MIN_EDGE_COMBINE and (1 - proba) >= MIN_PROBA_COMBINE:
                                        selections.append({
                                            'match': f"{match['player1']} vs {match['player2']}",
                                            'joueur': match['player2'],
                                            'proba': 1 - proba,
                                            'cote': odds2,
                                            'edge': edge2,
                                            'surface': match['surface']
                                        })
                                else:
                                    # Mode manuel - proposer les deux options
                                    selections.append({
                                        'match': f"{match['player1']} vs {match['player2']}",
                                        'joueur1': match['player1'],
                                        'joueur2': match['player2'],
                                        'proba1': proba,
                                        'proba2': 1 - proba,
                                        'cote1': odds1,
                                        'cote2': odds2,
                                        'edge1': edge1,
                                        'edge2': edge2,
                                        'surface': match['surface']
                                    })
                            except:
                                invalid_matches += 1
                                continue
                
                if invalid_matches > 0:
                    st.warning(f"{invalid_matches} matchs ignorés (cotes invalides ou manquantes)")
                
                if auto_select:
                    # Mode automatique
                    if len(selections) >= 2:
                        # Trier par edge
                        selections.sort(key=lambda x: x['edge'], reverse=True)
                        
                        # Limiter au nombre maximum de sélections
                        max_select = min(MAX_SELECTIONS_COMBINE, len(selections))
                        selected = selections[:max_select]
                        
                        # Calculer le combiné
                        proba_combi = 1.0
                        cote_combi = 1.0
                        
                        for sel in selected:
                            proba_combi *= sel['proba']
                            cote_combi *= sel['cote']
                        
                        gain = mise * cote_combi
                        esperance = proba_combi * gain - mise
                        kelly = (proba_combi * cote_combi - 1) / (cote_combi - 1) if cote_combi > 1 else 0
                        
                        # Affichage compact des métriques
                        st.markdown("### 📊 Résultats du combiné")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        proba_color = COLORS['success'] if proba_combi >= 0.3 else COLORS['warning'] if proba_combi >= 0.15 else COLORS['danger']
                        with col1:
                            st.markdown(create_metric("Probabilité", f"{proba_combi:.1%}", "", proba_color), unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(create_metric("Cote combinée", f"{cote_combi:.2f}"), unsafe_allow_html=True)
                        
                        esp_color = COLORS['success'] if esperance > 0 else COLORS['danger']
                        with col3:
                            st.markdown(create_metric("Espérance", f"{esperance:+.2f}€", "", esp_color), unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(create_metric("Kelly %", f"{kelly*100:.1f}", "%"), unsafe_allow_html=True)
                        
                        # Détail des sélections
                        st.markdown(f"### 📋 Sélections retenues ({len(selected)}/{len(selections)})")
                        
                        # Tableau compact des sélections
                        df_selections = pd.DataFrame([{
                            '#': i+1,
                            'Joueur': sel['joueur'],
                            'Match': sel['match'],
                            'Proba': f"{sel['proba']:.1%}",
                            'Cote': f"{sel['cote']:.2f}",
                            'Edge': f"{sel['edge']*100:+.1f}%"
                        } for i, sel in enumerate(selected)])
                        
                        st.dataframe(df_selections, use_container_width=True, hide_index=True)
                        
                        # Sauvegarde
                        combine_data = {
                            'selections': selected,
                            'proba_globale': proba_combi,
                            'cote_globale': cote_combi,
                            'mise': mise,
                            'gain_potentiel': gain,
                            'esperance': esperance,
                            'kelly': kelly,
                            'nb_matches': len(selected)
                        }
                        
                        if save_combine(combine_data):
                            st.success("✅ Combiné sauvegardé dans l'historique !")
                        
                        # Analyses IA
                        if use_ai and GROQ_AVAILABLE:
                            st.markdown("### 🤖 Analyse du combiné")
                            
                            prompt = f"Analyse ce combiné de {len(selected)} matchs avec une probabilité globale de {proba_combi:.1%} et une cote de {cote_combi:.2f}. Donne un avis sur sa pertinence et les risques."
                            
                            with st.spinner("Analyse en cours..."):
                                analysis = call_groq_api(prompt)
                            
                            if analysis:
                                st.markdown(f"<div class='card'>{analysis}</div>", unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ Pas assez de sélections valides ({len(selections)} trouvées, minimum 2 required). Ajuste les seuils ou vérifie les cotes.")
                
                else:
                    # Mode manuel - afficher toutes les options
                    st.markdown("### 📋 Sélections disponibles")
                    st.markdown("Choisis manuellement tes sélections dans la liste ci-dessous")
                    
                    # Créer un dataframe des options
                    df_options = pd.DataFrame([{
                        'Match': s['match'],
                        'Surface': s['surface'],
                        'Option 1': f"{s['joueur1']} @ {s['cote1']:.2f} (edge: {s['edge1']*100:+.1f}%)",
                        'Proba 1': f"{s['proba1']:.1%}",
                        'Option 2': f"{s['joueur2']} @ {s['cote2']:.2f} (edge: {s['edge2']*100:+.1f}%)",
                        'Proba 2': f"{s['proba2']:.1%}"
                    } for s in selections])
                    
                    st.dataframe(df_options, use_container_width=True, hide_index=True)
                    
                    st.info("Mode manuel en développement - utilise le mode auto-sélection pour l'instant")

# ─────────────────────────────────────────────────────────────
# HISTORIQUE
# ─────────────────────────────────────────────────────────────
def show_history():
    """Affiche l'historique des prédictions et combinés"""
    
    st.markdown("<h2>📜 Historique</h2>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📋 Prédictions", "🎰 Combinés"])
    
    with tab1:
        history = load_history()
        
        if history:
            # Filtres
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_surface = st.selectbox("Surface", ["Toutes"] + SURFACES)
            
            with col2:
                search = st.text_input("Rechercher un joueur", placeholder="Nom...")
            
            # Appliquer les filtres
            filtered = history
            if filter_surface != "Toutes":
                filtered = [h for h in filtered if h.get('surface') == filter_surface]
            if search:
                filtered = [h for h in filtered if search.lower() in h.get('player1', '').lower() or search.lower() in h.get('player2', '').lower()]
            
            # Inverser pour avoir les plus récents en premier
            filtered.reverse()
            
            # Afficher
            for i, pred in enumerate(filtered):
                date_str = pred.get('date', 'Date inconnue')[:16]
                player1 = pred.get('player1', 'Inconnu')
                player2 = pred.get('player2', 'Inconnu')
                with st.expander(f"{date_str} - {player1} vs {player2}", expanded=i==0):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(create_metric("Tournoi", pred.get('tournament', '—')), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(create_metric("Surface", pred.get('surface', '—')), unsafe_allow_html=True)
                    
                    with col3:
                        proba = pred.get('proba', 0)
                        st.markdown(create_metric("Probabilité", f"{proba:.1%}" if isinstance(proba, (int, float)) else str(proba)), unsafe_allow_html=True)
                    
                    with col4:
                        confidence = pred.get('confidence', 0)
                        if isinstance(confidence, (int, float)):
                            conf_color = COLORS['success'] if confidence >= 70 else COLORS['warning'] if confidence >= 50 else COLORS['danger']
                            st.markdown(create_metric("Confiance", f"{confidence:.0f}", "/100", conf_color), unsafe_allow_html=True)
        else:
            st.info("Aucune prédiction dans l'historique")
    
    with tab2:
        combines = load_combines()
        
        if combines:
            # Pagination
            items_per_page = 10
            total_pages = (len(combines) + items_per_page - 1) // items_per_page
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(combines))
            
            for i in range(start_idx, end_idx):
                comb = combines[-(i+1)]  # Inverser pour avoir le plus récent en premier
                date_str = comb.get('date', 'Date inconnue')[:16]
                nb_matches = comb.get('nb_matches', 0)
                proba = comb.get('proba_globale', 0)
                
                with st.expander(f"🎯 {date_str} - {nb_matches} matchs - Proba {proba:.1%}", expanded=i==start_idx):
                    cote = comb.get('cote_globale', 0)
                    esperance = comb.get('esperance', 0)
                    kelly = comb.get('kelly', 0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    proba_color = COLORS['success'] if proba >= 0.3 else COLORS['warning'] if proba >= 0.15 else COLORS['danger']
                    with col1:
                        st.markdown(create_metric("Probabilité", f"{proba:.1%}", "", proba_color), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(create_metric("Cote", f"{cote:.2f}"), unsafe_allow_html=True)
                    
                    esp_color = COLORS['success'] if esperance > 0 else COLORS['danger']
                    with col3:
                        st.markdown(create_metric("Espérance", f"{esperance:+.2f}€", "", esp_color), unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(create_metric("Kelly", f"{kelly*100:.1f}", "%"), unsafe_allow_html=True)
                    
                    # Détail des sélections
                    if 'selections' in comb and comb['selections']:
                        st.markdown("**Sélections:**")
                        
                        # Créer un tableau pour les sélections
                        df_sel = pd.DataFrame([{
                            'Joueur': sel.get('joueur', 'Inconnu'),
                            'Match': sel.get('match', ''),
                            'Proba': f"{sel.get('proba', 0):.1%}",
                            'Cote': f"{sel.get('cote', 0):.2f}",
                            'Edge': f"{sel.get('edge', 0)*100:+.1f}%"
                        } for sel in comb['selections']])
                        
                        st.dataframe(df_sel, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun combiné dans l'historique")

# ─────────────────────────────────────────────────────────────
# STATISTIQUES
# ─────────────────────────────────────────────────────────────
def show_statistics():
    """Affiche les statistiques utilisateur"""
    
    st.markdown("<h2>📈 Statistiques</h2>", unsafe_allow_html=True)
    
    stats = load_user_stats()
    history = load_history()
    combines = load_combines()
    
    # KPIs principaux
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric("Prédictions", stats.get('total_predictions', 0)), unsafe_allow_html=True)
    
    with col2:
        accuracy = (stats.get('correct_predictions', 0) / stats.get('total_predictions', 1)) * 100 if stats.get('total_predictions', 0) > 0 else 0
        acc_color = COLORS['success'] if accuracy >= 60 else COLORS['warning'] if accuracy >= 50 else COLORS['danger']
        st.markdown(create_metric("Précision", f"{accuracy:.1f}", "%", acc_color), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric("Combinés", stats.get('total_combines', 0)), unsafe_allow_html=True)
    
    with col4:
        win_rate_comb = (stats.get('won_combines', 0) / stats.get('total_combines', 1)) * 100 if stats.get('total_combines', 0) > 0 else 0
        st.markdown(create_metric("Réussite combinés", f"{win_rate_comb:.1f}", "%"), unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Statistiques financières
    col1, col2, col3, col4 = st.columns(4)
    
    total_invested = stats.get('total_invested', 0)
    total_won = stats.get('total_won', 0)
    profit = total_won - total_invested
    roi = (profit / total_invested * 100) if total_invested > 0 else 0
    
    with col1:
        st.markdown(create_metric("Total investi", f"{total_invested:.2f}", "€"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric("Total gagné", f"{total_won:.2f}", "€"), unsafe_allow_html=True)
    
    with col3:
        profit_color = COLORS['success'] if profit >= 0 else COLORS['danger']
        st.markdown(create_metric("Profit", f"{profit:+.2f}", "€", profit_color), unsafe_allow_html=True)
    
    with col4:
        roi_color = COLORS['success'] if roi >= 0 else COLORS['danger']
        st.markdown(create_metric("ROI", f"{roi:+.1f}", "%", roi_color), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
def show_configuration():
    """Affiche la configuration"""
    
    st.markdown("<h2>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎾 Paramètres généraux")
        
        # Année de début
        start_year = st.number_input("Année de début", min_value=2000, max_value=2024, value=START_YEAR)
    
    with col2:
        st.markdown("### 🤖 Intelligence Artificielle")
        
        # Statut Groq
        groq_status = "✅ Connecté" if get_groq_key() else "❌ Non configuré"
        st.markdown(f"**Groq API:** {groq_status}")
        
        if not get_groq_key():
            st.info("Pour activer les analyses IA, ajoute ta clé API Groq dans les secrets Streamlit ou en variable d'environnement.")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### 🗑️ Gestion des données")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Effacer l'historique des prédictions", use_container_width=True):
            if HIST_FILE.exists():
                HIST_FILE.unlink()
                st.success("Historique effacé !")
                st.rerun()
    
    with col2:
        if st.button("🗑️ Effacer l'historique des combinés", use_container_width=True):
            if COMB_HIST_FILE.exists():
                COMB_HIST_FILE.unlink()
                st.success("Historique des combinés effacé !")
                st.rerun()
    
    with col3:
        if st.button("🗑️ Effacer toutes les statistiques", use_container_width=True):
            if USER_STATS_FILE.exists():
                USER_STATS_FILE.unlink()
                st.success("Statistiques effacées !")
                st.rerun()
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Export des données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Exporter les prédictions (CSV)", use_container_width=True):
            history = load_history()
            if history:
                df = pd.DataFrame(history)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Télécharger CSV",
                    csv,
                    f"tennisiq_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
    
    with col2:
        if st.button("📥 Exporter les combinés (CSV)", use_container_width=True):
            combines = load_combines()
            if combines:
                df = pd.DataFrame(combines)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Télécharger CSV",
                    csv,
                    f"tennisiq_combines_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )

# ─────────────────────────────────────────────────────────────
# LANCEMENT DE L'APPLICATION
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
