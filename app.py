import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import traceback

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "src" / "data" / "raw" / "tml-tennis"
MODELS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "rank_diff", "pts_diff", "age_diff",
    "form_diff", "fatigue_diff",
    "ace_diff", "df_diff",
    "pct_1st_in_diff", "pct_1st_won_diff", "pct_2nd_won_diff",
    "pct_bp_saved_diff",
    "pct_ret_1st_diff", "pct_ret_2nd_diff",
    "h2h_score", "best_of",
    "surface_hard", "surface_clay", "surface_grass",
    "level_gs", "level_m1000", "level_500",
]

SURFACES = ["Hard", "Clay", "Grass"]
TOURS = {"ATP": "atp", "WTA": "wta"}

# ─────────────────────────────────────────────────────────────
# CSS — Dark Luxury Tennis
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TennisIQ — Prédictions IA",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e0f;
    color: #e8e0d0;
}
.stApp { background: #0a0e0f; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1214 0%, #111a1c 100%);
    border-right: 1px solid #1e2d2f;
}
[data-testid="stSidebar"] * { color: #c8c0b0 !important; }

/* ── Header principal ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #7fff7a 0%, #3dd68c 40%, #00b894 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #6b7c7e;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ── Cards ── */
.card {
    background: linear-gradient(135deg, #111a1c 0%, #0f1719 100%);
    border: 1px solid #1e2d2f;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color 0.3s;
}
.card:hover { border-color: #3dd68c44; }

.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e8e0d0;
    margin-bottom: 4px;
    letter-spacing: 0.3px;
}
.card-sub {
    font-size: 0.78rem;
    color: #4a5e60;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* ── Badge surface ── */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.badge-hard  { background: #1a2e4a; color: #5ba3f5; border: 1px solid #2a4a6a; }
.badge-clay  { background: #3a1a0a; color: #e07840; border: 1px solid #5a2a0a; }
.badge-grass { background: #0a2a14; color: #4caf6a; border: 1px solid #0a4a1e; }
.badge-atp   { background: #1a1a3a; color: #7a8af5; border: 1px solid #2a2a5a; }
.badge-wta   { background: #3a0a2a; color: #f57ab0; border: 1px solid #5a0a3a; }
.badge-gs    { background: #2a2000; color: #f5c842; border: 1px solid #4a3800; }

/* ── Proba bar ── */
.proba-container { margin: 20px 0; }
.proba-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: #6b7c7e;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
}
.proba-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8e0d0;
}
.proba-pct {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    line-height: 1;
}
.proba-pct-green { color: #3dd68c; }
.proba-pct-dim   { color: #2a3e40; }

.bar-track {
    height: 8px;
    background: #1a2a2c;
    border-radius: 4px;
    margin: 10px 0;
    overflow: hidden;
}
.bar-fill-green {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #3dd68c, #7fff7a);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.bar-fill-dim {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #1e3a4c, #2a5060);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Confidence gauge ── */
.conf-score {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 900;
    line-height: 1;
}
.conf-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-top: 4px;
}

/* ── Stat row ── */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #141e20;
}
.stat-row:last-child { border-bottom: none; }
.stat-key {
    font-size: 0.8rem;
    color: #4a5e60;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.stat-val {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.95rem;
    color: #c8c0b0;
}
.stat-val-green { color: #3dd68c; }
.stat-val-red   { color: #e07878; }

/* ── H2H score ── */
.h2h-score {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    color: #3dd68c;
}
.h2h-vs {
    font-size: 0.7rem;
    color: #4a5e60;
    text-transform: uppercase;
    letter-spacing: 4px;
    text-align: center;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e2d2f 30%, #1e2d2f 70%, transparent);
    margin: 24px 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 8px;
    border-bottom: 1px solid #1e2d2f;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a5e60 !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 10px 20px;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #3dd68c !important;
    border-bottom: 2px solid #3dd68c !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #111a1c !important;
    border: 1px solid #1e2d2f !important;
    border-radius: 10px !important;
    color: #e8e0d0 !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #3dd68c !important;
}
.stSlider [data-testid="stSliderThumbValue"] { color: #3dd68c !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e4a3a, #2a6a4a) !important;
    color: #7fff7a !important;
    border: 1px solid #3dd68c44 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 12px 32px !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2a6a4a, #3a8a5a) !important;
    border-color: #3dd68c99 !important;
    transform: translateY(-1px) !important;
}

/* ── Metric ── */
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    color: #3dd68c !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    color: #4a5e60 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #111a1c !important;
    border: 1px solid #1e2d2f !important;
    border-radius: 12px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.dataframe { background: #111a1c !important; }

/* ── Warning / Success / Info ── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e0f; }
::-webkit-scrollbar-thumb { background: #1e2d2f; border-radius: 3px; }

/* ── Footer ── */
footer { visibility: hidden; }
.footer-custom {
    text-align: center;
    padding: 24px;
    color: #2a3e40;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-top: 1px solid #141e20;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHARGEMENT MODÈLES AVEC GESTION D'ERREURS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(tour, surface):
    """Charge le modèle pour un tour (atp/wta) et une surface avec gestion d'erreurs."""
    model_paths = [
        MODELS_DIR / f"tennis_model_{tour}_{surface.lower()}.h5",
        MODELS_DIR / f"tennis_model_{surface.lower()}.h5",
        MODELS_DIR / f"model_{tour}_{surface.lower()}.h5",
        MODELS_DIR / f"{tour}_{surface.lower()}_model.h5"
    ]
    
    for path in model_paths:
        if path.exists():
            try:
                from tensorflow.keras.models import load_model as km
                model = km(str(path))
                # Vérifier que le modèle a bien une forme d'entrée
                if hasattr(model, 'input_shape'):
                    return model
            except Exception as e:
                st.warning(f"Erreur chargement {path.name}: {str(e)[:100]}")
                continue
    return None

@st.cache_resource
def load_scaler(tour, surface):
    """Charge le scaler avec gestion d'erreurs."""
    scaler_paths = [
        MODELS_DIR / f"tennis_scaler_{tour}_{surface.lower()}.joblib",
        MODELS_DIR / f"tennis_scaler_{surface.lower()}.joblib",
        MODELS_DIR / f"scaler_{tour}_{surface.lower()}.joblib",
        MODELS_DIR / f"{tour}_{surface.lower()}_scaler.joblib"
    ]
    
    for path in scaler_paths:
        if path.exists():
            try:
                return joblib.load(str(path))
            except:
                continue
    return None

@st.cache_data
def load_model_metadata():
    """Charge les métadonnées des modèles."""
    p = MODELS_DIR / "tennis_features_meta.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {
        "features": FEATURES,
        "models": {}
    }

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DONNÉES
# ─────────────────────────────────────────────────────────────
# Mapping tennis-data.co.uk (WTA/ATP) → format standard
TENNIS_DATA_MAP = {
    "Date": "tourney_date",
    "Tournament": "tourney_name",
    "Location": "tourney_location",
    "Surface": "surface",
    "Tier": "tourney_level",   # WTA
    "Series": "tourney_level",   # ATP
    "Court": "indoor",
    "Round": "round",
    "Best of": "best_of",
    "Winner": "winner_name",
    "Loser": "loser_name",
    "WRank": "winner_rank",
    "LRank": "loser_rank",
    "WPts": "winner_rank_points",
    "LPts": "loser_rank_points",
    "Wsets": "w_sets",
    "Lsets": "l_sets",
    "WAce": "w_ace", "LAce": "l_ace",
    "WDF": "w_df", "LDF": "l_df",
    "WBpFaced": "w_bpFaced", "LBpFaced": "l_bpFaced",
    "WBpSaved": "w_bpSaved", "LBpSaved": "l_bpSaved",
    "Score": "score",
}

REQUIRED_COLS = {
    "tourney_date", "surface", "winner_name", "loser_name",
    "winner_rank", "loser_rank"
}

def normalize_surface(s):
    if pd.isna(s): return None
    s = str(s).strip().lower()
    if "hard" in s: return "Hard"
    if "clay" in s: return "Clay"
    if "grass" in s: return "Grass"
    return None

def parse_numeric_fr(series):
    """Convertit les nombres à virgule française (332,25 → 332.25)."""
    if series.dtype == object:
        series = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")

def parse_date_flexible(series):
    """Gère YYYYMMDD (Jeff Sackmann) et DD/MM/YYYY (tennis-data)."""
    s = series.astype(str).str.strip()
    result = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    mask = result.isna()
    if mask.any():
        result[mask] = pd.to_datetime(s[mask], format="%d/%m/%Y", errors="coerce")
    mask = result.isna()
    if mask.any():
        result[mask] = pd.to_datetime(s[mask], dayfirst=True, errors="coerce")
    return result

def read_csv_robust(filepath):
    """Détecte automatiquement le séparateur (, ou ;) et l'encodage."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(filepath, "r", encoding=enc) as fh:
                first = fh.readline()
            sep = ";" if first.count(";") > first.count(",") else ","
            return pd.read_csv(filepath, sep=sep, encoding=enc,
                               on_bad_lines="skip", low_memory=False)
        except Exception:
            continue
    return pd.read_csv(filepath, sep=";", encoding="latin-1",
                       on_bad_lines="skip", encoding_errors="replace",
                       low_memory=False)

def apply_column_mapping(df):
    """Normalise les colonnes tennis-data vers le format standard."""
    cols = set(df.columns)
    if "Winner" in cols or "Date" in cols:
        rename = {k: v for k, v in TENNIS_DATA_MAP.items() if k in cols}
        df = df.rename(columns=rename)
        for col in ["winner_rank_points", "loser_rank_points"]:
            if col in df.columns:
                df[col] = parse_numeric_fr(df[col])
        defaults = {
            "winner_rank_points": np.nan, "loser_rank_points": np.nan,
            "winner_age": np.nan, "loser_age": np.nan,
            "best_of": 3, "tourney_level": "A",
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val
        for col in ["w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon",
                    "w_bpSaved","w_bpFaced","l_ace","l_df","l_svpt",
                    "l_1stIn","l_1stWon","l_2ndWon","l_bpSaved","l_bpFaced"]:
            if col not in df.columns:
                df[col] = np.nan
    return df

@st.cache_data(ttl=600)
def load_all_data():
    """Charge toutes les données avec gestion d'erreurs."""
    if not DATA_DIR.exists():
        return None, None
    
    csvs = sorted(DATA_DIR.glob("*.csv"))
    atp_dfs, wta_dfs = [], []
    
    for f in csvs:
        try:
            df = read_csv_robust(f)
            df = apply_column_mapping(df)
            
            if not REQUIRED_COLS.issubset(set(df.columns)):
                continue
                
            df["surface"] = df["surface"].apply(normalize_surface)
            df["tourney_date"] = parse_date_flexible(df["tourney_date"])
            df["winner_rank"] = pd.to_numeric(df["winner_rank"], errors="coerce")
            df["loser_rank"] = pd.to_numeric(df["loser_rank"], errors="coerce")
            df = df[df["tourney_date"].notna()]
            
            if len(df) == 0:
                continue
                
            if "wta" in f.name.lower():
                wta_dfs.append(df)
            else:
                atp_dfs.append(df)
        except Exception as e:
            st.warning(f"Erreur lecture {f.name}: {str(e)[:50]}")
            continue
    
    atp = pd.concat(atp_dfs, ignore_index=True).sort_values("tourney_date").reset_index(drop=True) if atp_dfs else pd.DataFrame()
    wta = pd.concat(wta_dfs, ignore_index=True).sort_values("tourney_date").reset_index(drop=True) if wta_dfs else pd.DataFrame()
    
    return atp if len(atp) > 0 else None, wta if len(wta) > 0 else None

# ─────────────────────────────────────────────────────────────
# FEATURES ENGINEERING
# ─────────────────────────────────────────────────────────────
def get_player_stats(df, player, surface=None, n_stats=15, n_form=5, fatigue_days=7):
    """Calcule les statistiques d'un joueur avec gestion des valeurs manquantes."""
    if df is None or len(df) == 0:
        return None
        
    as_w = df[df["winner_name"] == player].copy()
    as_w["is_w"] = True
    as_w["ace"] = as_w["w_ace"]
    as_w["df_c"] = as_w["w_df"]
    as_w["svpt"] = as_w["w_svpt"]
    as_w["1stIn"] = as_w["w_1stIn"]
    as_w["1stWon"] = as_w["w_1stWon"]
    as_w["2ndWon"] = as_w["w_2ndWon"]
    as_w["bpS"] = as_w["w_bpSaved"]
    as_w["bpF"] = as_w["w_bpFaced"]
    as_w["rank"] = as_w["winner_rank"]
    as_w["rank_pts"] = as_w["winner_rank_points"]
    as_w["age"] = as_w["winner_age"]
    as_w["o1stIn"] = as_w["l_1stIn"]
    as_w["o1stWon"] = as_w["l_1stWon"]
    as_w["o2ndWon"] = as_w["l_2ndWon"]

    as_l = df[df["loser_name"] == player].copy()
    as_l["is_w"] = False
    as_l["ace"] = as_l["l_ace"]
    as_l["df_c"] = as_l["l_df"]
    as_l["svpt"] = as_l["l_svpt"]
    as_l["1stIn"] = as_l["l_1stIn"]
    as_l["1stWon"] = as_l["l_1stWon"]
    as_l["2ndWon"] = as_l["l_2ndWon"]
    as_l["bpS"] = as_l["l_bpSaved"]
    as_l["bpF"] = as_l["l_bpFaced"]
    as_l["rank"] = as_l["loser_rank"]
    as_l["rank_pts"] = as_l["loser_rank_points"]
    as_l["age"] = as_l["loser_age"]
    as_l["o1stIn"] = as_l["w_1stIn"]
    as_l["o1stWon"] = as_l["w_1stWon"]
    as_l["o2ndWon"] = as_l["w_2ndWon"]

    cols = ["tourney_date","surface","is_w","ace","df_c","svpt",
            "1stIn","1stWon","2ndWon","bpS","bpF","rank","rank_pts","age",
            "o1stIn","o1stWon","o2ndWon"]

    all_m = pd.concat([as_w[cols], as_l[cols]], ignore_index=True)
    
    if len(all_m) == 0:
        return None
        
    all_m = all_m.sort_values("tourney_date", ascending=False)

    # Forme (N derniers matchs toutes surfaces)
    form_m = all_m.head(n_form)
    form_pct = float(form_m["is_w"].sum() / len(form_m)) if len(form_m) > 0 else 0.5

    # Fatigue
    last_d = all_m["tourney_date"].iloc[0] if len(all_m) > 0 else None
    fatigue = 0
    if last_d is not None:
        cutoff = last_d - pd.Timedelta(days=fatigue_days)
        fatigue = int((all_m["tourney_date"] >= cutoff).sum())

    # Stats filtrées surface
    if surface:
        sm = all_m[all_m["surface"] == surface].head(n_stats)
        working = sm if len(sm) >= 3 else all_m.head(n_stats)
        surf_note = "surface" if len(sm) >= 3 else "all surfaces"
    else:
        working = all_m.head(n_stats)
        surf_note = "all surfaces"

    if len(working) == 0:
        working = all_m.head(1)
        if len(working) == 0:
            return None

    # Valeurs par défaut pour les statistiques manquantes
    rank = working["rank"].dropna().iloc[0] if not working["rank"].dropna().empty else 1000
    rank_pts = working["rank_pts"].dropna().iloc[0] if not working["rank_pts"].dropna().empty else 0
    age = working["age"].dropna().iloc[0] if not working["age"].dropna().empty else 25

    def safe_mean(series):
        return float(series.mean()) if not series.isna().all() else 0.0
    
    def safe_ratio(num, den):
        if den > 0 and not pd.isna(num) and not pd.isna(den):
            return float(num / den)
        return 0.0

    svpt_sum = working["svpt"].sum()
    in1_sum = working["1stIn"].sum()
    won2_sum = working["2ndWon"].sum()
    pct_2nd = safe_ratio(won2_sum, svpt_sum - in1_sum)

    oi1s = working["o1stIn"].sum()
    ow1s = working["o1stWon"].sum()
    pct_ret_1st = safe_ratio(oi1s - ow1s, oi1s)

    wins = int(working["is_w"].sum())
    played = len(working)

    return {
        "rank": rank,
        "rank_pts": rank_pts,
        "age": age,
        "form_pct": form_pct,
        "fatigue": fatigue,
        "ace_avg": safe_mean(working["ace"]),
        "df_avg": safe_mean(working["df_c"]),
        "pct_1st_in": safe_ratio(in1_sum, svpt_sum),
        "pct_1st_won": safe_ratio(working["1stWon"].sum(), in1_sum),
        "pct_2nd_won": pct_2nd,
        "pct_bp_saved": safe_ratio(working["bpS"].sum(), working["bpF"].sum()),
        "pct_ret_1st": pct_ret_1st,
        "pct_ret_2nd": safe_mean(working["o2ndWon"]),
        "wins": wins,
        "played": played,
        "win_pct": wins/played if played > 0 else 0,
        "surf_note": surf_note,
        "last_date": all_m["tourney_date"].iloc[0] if len(all_m) > 0 else None,
    }

def get_h2h(df, j1, j2, surface=None):
    """Calcule les statistiques head-to-head."""
    if df is None or len(df) == 0:
        return {"total":0, "j1_tot":0, "j2_tot":0, "j1_surf":None, "j2_surf":None,
                "surf_total":0, "h2h_score":0.5, "recent":pd.DataFrame()}
    
    mask = (
        ((df["winner_name"] == j1) & (df["loser_name"] == j2)) |
        ((df["winner_name"] == j2) & (df["loser_name"] == j1))
    )
    h2h = df[mask].copy()
    h2h_s = h2h[h2h["surface"] == surface] if surface and len(h2h) > 0 else h2h

    j1_tot = int((h2h["winner_name"] == j1).sum())
    j2_tot = int((h2h["winner_name"] == j2).sum())
    j1_surf = int((h2h_s["winner_name"] == j1).sum()) if surface and len(h2h_s) > 0 else None
    j2_surf = int((h2h_s["winner_name"] == j2).sum()) if surface and len(h2h_s) > 0 else None
    h2h_sc = j1_tot / len(h2h) if len(h2h) > 0 else 0.5

    recent = pd.DataFrame()
    if len(h2h) > 0:
        recent = h2h.sort_values("tourney_date", ascending=False).head(5)[
            ["tourney_date", "tourney_name", "surface", "round", "winner_name", "loser_name", "score"]
        ].copy()
        if len(recent) > 0:
            recent["tourney_date"] = recent["tourney_date"].dt.strftime("%Y-%m-%d")
    
    return {
        "total": j1_tot + j2_tot,
        "j1_tot": j1_tot,
        "j2_tot": j2_tot,
        "j1_surf": j1_surf,
        "j2_surf": j2_surf,
        "surf_total": len(h2h_s),
        "h2h_score": h2h_sc,
        "recent": recent
    }

def confidence_score(proba, s1, s2, h2h):
    """Calcule un score de confiance pour la prédiction."""
    signals = []
    pred_s = abs(proba - 0.5) * 2
    signals.append(("Prediction strength", pred_s * 35))
    
    dq = (min(s1["played"], 15) / 15 * 0.5 + min(s2["played"], 15) / 15 * 0.5)
    signals.append(("Data quality", dq * 25))
    
    if h2h["total"] >= 2:
        fav_h2h = h2h["h2h_score"] if proba >= 0.5 else (1 - h2h["h2h_score"])
        signals.append(("H2H consistency", fav_h2h * 25))
    else:
        signals.append(("H2H consistency", 12.5))
    
    form_ok = 1.0 if (
        (proba >= 0.5 and s1["form_pct"] >= s2["form_pct"]) or
        (proba < 0.5 and s2["form_pct"] >= s1["form_pct"])
    ) else 0.3
    signals.append(("Recent form alignment", form_ok * 15))
    
    return round(min(sum(v for _, v in signals), 100)), signals

def build_feature_vector(s1, s2, h2h_sc, surface, best_of, level):
    """Construit le vecteur de features pour la prédiction."""
    def safe_diff(k):
        a, b = s1.get(k, 0), s2.get(k, 0)
        if a is not None and b is not None and not pd.isna(a) and not pd.isna(b):
            return float(a) - float(b)
        return 0.0
    
    level_gs = int(level in ("G", "Grand Slam"))
    level_m = int(level in ("M", "Masters", "Masters 1000"))
    level_500 = int(level in ("500", "A", "ATP500"))
    
    return [
        safe_diff("rank"),
        safe_diff("rank_pts"),
        safe_diff("age"),
        safe_diff("form_pct"),
        safe_diff("fatigue"),
        safe_diff("ace_avg"),
        safe_diff("df_avg"),
        safe_diff("pct_1st_in"),
        safe_diff("pct_1st_won"),
        safe_diff("pct_2nd_won"),
        safe_diff("pct_bp_saved"),
        safe_diff("pct_ret_1st"),
        safe_diff("pct_ret_2nd"),
        float(h2h_sc),
        float(best_of),
        int(surface == "Hard"),
        int(surface == "Clay"),
        int(surface == "Grass"),
        level_gs,
        level_m,
        level_500,
    ]

# ─────────────────────────────────────────────────────────────
# HELPERS HTML
# ─────────────────────────────────────────────────────────────
def surface_badge(surface):
    s = surface.lower()
    cls = {"hard": "badge-hard", "clay": "badge-clay", "grass": "badge-grass"}.get(s, "badge-hard")
    return f'<span class="badge {cls}">{surface}</span>'

def tour_badge(tour):
    cls = "badge-atp" if tour == "ATP" else "badge-wta"
    return f'<span class="badge {cls}">{tour}</span>'

def level_badge(level):
    labels = {"G": "Grand Slam", "M": "Masters 1000", "500": "ATP 500", "A": "ATP", "F": "Finals"}
    label = labels.get(str(level), str(level))
    if level in ("G", "Grand Slam"):
        return f'<span class="badge badge-gs">{label}</span>'
    return f'<span class="badge badge-hard">{label}</span>'

def stat_html(key, val1, val2=None, higher_is_better=True):
    if val1 is None or (val2 is None and key != "LAST SEEN"):
        return ""
    
    if val2 is None:
        if isinstance(val1, float):
            if val1 < 2:
                v1_s = f"{val1:.1%}"
            else:
                v1_s = f"{val1:.1f}"
        else:
            v1_s = str(val1)
        return f"""
        <div class="stat-row">
            <span class="stat-val">{v1_s}</span>
            <span class="stat-key">{key}</span>
            <span class="stat-val">-</span>
        </div>"""
    
    if isinstance(val1, float) and val1 < 2:
        v1_s = f"{val1:.1%}"
        v2_s = f"{val2:.1%}"
    else:
        v1_s = f"{val1:.1f}" if isinstance(val1, float) else str(val1)
        v2_s = f"{val2:.1f}" if isinstance(val2, float) else str(val2)
    
    better = val1 > val2 if higher_is_better else val1 < val2
    c1 = "stat-val-green" if better else "stat-val"
    c2 = "stat-val-green" if not better else "stat-val"
    
    return f"""
    <div class="stat-row">
        <span class="stat-val {c1}">{v1_s}</span>
        <span class="stat-key">{key}</span>
        <span class="stat-val {c2}">{v2_s}</span>
    </div>"""

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0 10px 0;">
        <div style="font-family:'Playfair Display',serif; font-size:1.4rem;
                    font-weight:700; color:#3dd68c; letter-spacing:-0.5px;">
            TennisIQ
        </div>
        <div style="font-size:0.68rem; color:#2a3e40; letter-spacing:3px;
                    text-transform:uppercase; margin-top:2px;">
            AI Prediction Engine
        </div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    atp_data, wta_data = load_all_data()

    atp_ok = atp_data is not None and len(atp_data) > 0
    wta_ok = wta_data is not None and len(wta_data) > 0

    st.markdown("**DATA STATUS**")
    if atp_ok:
        n_atp = len(atp_data)
        y_min = atp_data["tourney_date"].min().year
        y_max = atp_data["tourney_date"].max().year
        st.markdown(f"""
        <div class="card" style="padding:14px; margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="badge badge-atp">ATP</span>
                <span style="color:#3dd68c; font-family:'Playfair Display',serif; font-weight:700;">✓</span>
            </div>
            <div style="margin-top:8px; font-size:0.85rem; color:#c8c0b0;">
                {n_atp:,} matchs
            </div>
            <div style="font-size:0.72rem; color:#4a5e60; margin-top:2px;">
                {y_min} — {y_max}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#e07878; font-size:0.8rem;">⚠ No ATP data found</div>', unsafe_allow_html=True)

    if wta_ok:
        n_wta = len(wta_data)
        y_min = wta_data["tourney_date"].min().year
        y_max = wta_data["tourney_date"].max().year
        st.markdown(f"""
        <div class="card" style="padding:14px; margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="badge badge-wta">WTA</span>
                <span style="color:#3dd68c; font-family:'Playfair Display',serif; font-weight:700;">✓</span>
            </div>
            <div style="margin-top:8px; font-size:0.85rem; color:#c8c0b0;">
                {n_wta:,} matchs
            </div>
            <div style="font-size:0.72rem; color:#4a5e60; margin-top:2px;">
                {y_min} — {y_max}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#e07878; font-size:0.8rem;">⚠ No WTA data found</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("**MODELS**")
    meta = load_model_metadata()
    
    for tour_key in ["atp", "wta"]:
        for surf in SURFACES:
            mp = MODELS_DIR / f"tennis_model_{tour_key}_{surf.lower()}.h5"
            ok = mp.exists()
            acc_str = ""
            if ok and meta and "models" in meta:
                res = meta["models"].get(f"{tour_key}_{surf}", {})
                if res:
                    acc_str = f" · {res.get('accuracy', 0) * 100:.0f}%"
            color = "#3dd68c" if ok else "#2a3a3c"
            icon = "●" if ok else "○"
            st.markdown(
                f'<div style="font-size:0.78rem; color:{color}; '
                f'padding:3px 0; letter-spacing:1px;">'
                f'{icon} {tour_key.upper()} {surf}{acc_str}</div>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.65rem; color:#2a3e40; letter-spacing:2px; '
        'text-transform:uppercase;">src/data/raw/tml-tennis/</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 32px 0 24px 0;">
    <p class="hero-title">TennisIQ</p>
    <p class="hero-sub">Neural Network Prediction Engine · ATP & WTA</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

if not atp_ok and not wta_ok:
    st.error("Aucune donnée trouvée dans `src/data/raw/tml-tennis/`. Vérifiez vos fichiers CSV.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_pred, tab_explore, tab_models = st.tabs([
    "⚡  PREDICT",
    "🔍  EXPLORE",
    "📊  MODELS"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab_pred:
    c1, c2 = st.columns([1, 3])

    with c1:
        tour = st.selectbox("Circuit", ["ATP", "WTA"], key="tour_sel")

    df_active = atp_data if tour == "ATP" else wta_data

    if df_active is None or len(df_active) == 0:
        st.warning(f"Données {tour} non disponibles.")
        st.stop()

    with c2:
        tournois = sorted(df_active["tourney_name"].dropna().unique())
        selected_tournoi = st.selectbox("Tournament", tournois, key="tourn_sel") if len(tournois) > 0 else None

    if selected_tournoi is None:
        st.warning("Aucun tournoi disponible.")
        st.stop()

    df_t = df_active[df_active["tourney_name"] == selected_tournoi]
    surface = df_t["surface"].iloc[0] if not df_t.empty else "Hard"
    best_of = int(df_t["best_of"].iloc[0]) if not df_t.empty and not pd.isna(df_t["best_of"].iloc[0]) else 3
    level = df_t["tourney_level"].iloc[0] if not df_t.empty and "tourney_level" in df_t.columns else "A"

    st.markdown(f"""
    <div class="card" style="padding:16px 24px; margin: 12px 0;">
        <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;">
            {tour_badge(tour)}
            {surface_badge(surface)}
            {level_badge(str(level))}
            <span style="font-size:0.78rem; color:#4a5e60; margin-left:4px;">
                Best of {best_of}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    all_players = sorted(pd.concat([
        df_active["winner_name"], df_active["loser_name"]
    ]).dropna().unique())

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    pj1, pj2 = st.columns(2)
    with pj1:
        st.markdown('<div class="card-sub" style="margin-bottom:6px;">Player 1</div>', unsafe_allow_html=True)
        joueur1 = st.selectbox("", all_players, key="j1", label_visibility="collapsed")
    with pj2:
        st.markdown('<div class="card-sub" style="margin-bottom:6px;">Player 2</div>', unsafe_allow_html=True)
        joueur2_options = [p for p in all_players if p != joueur1]
        joueur2 = st.selectbox("", joueur2_options, key="j2", label_visibility="collapsed") if joueur2_options else None

    if joueur2 is None:
        st.warning("Sélectionnez deux joueurs différents.")
        st.stop()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    pa1, pa2 = st.columns(2)
    with pa1:
        n_form = st.slider("Recent form matches", 3, 10, 5)
    with pa2:
        n_stats = st.slider("Stats window matches", 5, 30, 15)

    s1 = get_player_stats(df_active, joueur1, surface, n_stats, n_form)
    s2 = get_player_stats(df_active, joueur2, surface, n_stats, n_form)
    h2h = get_h2h(df_active, joueur1, joueur2, surface)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
        <div class="card-title">Player Stats</div>
        <div style="font-size:0.72rem; color:#4a5e60; letter-spacing:2px; text-transform:uppercase;">
            {surface} · last {n_stats} matches
        </div>
    </div>
    """, unsafe_allow_html=True)

    if s1 and s2:
        hn1, hm, hn2 = st.columns([5, 2, 5])
        with hn1:
            st.markdown(f"""
            <div class="card-title" style="font-size:1.3rem;">{joueur1}</div>
            <div class="card-sub">{s1['surf_note']} · {s1['played']} matches</div>
            """, unsafe_allow_html=True)
        with hm:
            st.markdown('<div style="text-align:center; color:#2a3e40; font-size:1.5rem; padding-top:8px;">VS</div>', unsafe_allow_html=True)
        with hn2:
            st.markdown(f"""
            <div class="card-title" style="font-size:1.3rem; text-align:right;">{joueur2}</div>
            <div class="card-sub" style="text-align:right;">{s2['surf_note']} · {s2['played']} matches</div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card" style="margin-top:12px;">
            {stat_html("RANKING", s1['rank'], s2['rank'], higher_is_better=False)}
            {stat_html("ATP PTS", s1['rank_pts'], s2['rank_pts'])}
            {stat_html("AGE", s1['age'], s2['age'], higher_is_better=False)}
            {stat_html(f"FORM {n_form} MATCHES", s1['form_pct'], s2['form_pct'])}
            {stat_html("FATIGUE (7d)", s1['fatigue'], s2['fatigue'], higher_is_better=False)}
            {stat_html("ACES / MATCH", s1['ace_avg'], s2['ace_avg'])}
            {stat_html("DBL FAULTS", s1['df_avg'], s2['df_avg'], higher_is_better=False)}
            {stat_html("1ST SERVE IN", s1['pct_1st_in'], s2['pct_1st_in'])}
            {stat_html("1ST SERVE WON", s1['pct_1st_won'], s2['pct_1st_won'])}
            {stat_html("2ND SERVE WON", s1['pct_2nd_won'], s2['pct_2nd_won'])}
            {stat_html("BP SAVED", s1['pct_bp_saved'], s2['pct_bp_saved'])}
            {stat_html("RETURN 1ST WON", s1['pct_ret_1st'], s2['pct_ret_1st'])}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Données insuffisantes pour un ou deux joueurs.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    if h2h and h2h["total"] > 0:
        st.markdown(f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="text-align:center; flex:1;">
                    <div style="font-family:'Playfair Display',serif; font-size:2.5rem;
                                font-weight:900; color:#3dd68c;">{h2h['j1_tot']}</div>
                    <div class="card-sub">{joueur1.split()[-1] if joueur1 else ''}</div>
                </div>
                <div style="text-align:center; flex:0.6;">
                    <div style="font-size:0.65rem; color:#2a3e40; letter-spacing:4px;
                                text-transform:uppercase;">HEAD TO HEAD</div>
                    <div style="font-size:0.7rem; color:#4a5e60; margin-top:4px;">
                        {h2h['total']} matches total
                    </div>
                    {f"<div style='font-size:0.68rem; color:#3a5040; margin-top:2px;'>{h2h['j1_surf']}–{h2h['j2_surf']} on {surface}</div>" if h2h['j1_surf'] is not None else ""}
                </div>
                <div style="text-align:center; flex:1;">
                    <div style="font-family:'Playfair Display',serif; font-size:2.5rem;
                                font-weight:900; color:#c8c0b0;">{h2h['j2_tot']}</div>
                    <div class="card-sub">{joueur2.split()[-1] if joueur2 else ''}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not h2h["recent"].empty:
            with st.expander("Last 5 encounters"):
                st.dataframe(
                    h2h["recent"].rename(columns={
                        "tourney_date": "Date", "tourney_name": "Tournament",
                        "surface": "Surface", "round": "Round",
                        "winner_name": "Winner", "loser_name": "Loser", "score": "Score"
                    }),
                    use_container_width=True, hide_index=True
                )
    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:20px;">
            <div style="color:#2a3e40; font-size:0.8rem; letter-spacing:2px; text-transform:uppercase;">
                No previous encounters found
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        predict_clicked = st.button("⚡  GENERATE PREDICTION", use_container_width=True)

    if predict_clicked:
        if s1 is None or s2 is None:
            st.error("Données insuffisantes pour la prédiction.")
        else:
            tour_key = tour.lower()
            model = load_model(tour_key, surface)
            scaler = load_scaler(tour_key, surface)

            if not model:
                st.warning(f"Modèle non trouvé. Vérifiez models/tennis_model_{tour_key}_{surface.lower()}.h5")
            else:
                try:
                    fv = build_feature_vector(s1, s2, h2h["h2h_score"], surface, best_of, str(level))
                    X = np.array(fv, dtype=np.float32).reshape(1, -1)
                    
                    # Vérification de la compatibilité du modèle
                    expected_features = model.input_shape[1]
                    if expected_features != X.shape[1]:
                        st.warning(f"Le modèle attend {expected_features} features, mais {X.shape[1]} fournies. Ajustement...")
                        if expected_features < X.shape[1]:
                            X = X[:, :expected_features]
                        else:
                            # Padding avec des zéros si nécessaire
                            padding = np.zeros((1, expected_features - X.shape[1]))
                            X = np.hstack([X, padding])
                    
                    # Application du scaler si disponible et compatible
                    if scaler:
                        try:
                            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ == X.shape[1]:
                                X = scaler.transform(X)
                            else:
                                st.caption("⚠ Scaler non compatible - utilisation sans normalisation")
                        except Exception as e:
                            st.caption(f"⚠ Erreur scaler: {str(e)[:50]}")

                    with st.spinner("Calcul de la prédiction..."):
                        proba = float(model.predict(X, verbose=0)[0][0])
                        proba = np.clip(proba, 0.01, 0.99)  # Éviter les extrêmes

                    conf, signals = confidence_score(proba, s1, s2, h2h)

                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                    rr1, rr2 = st.columns(2)
                    with rr1:
                        color1 = "#3dd68c" if proba >= 0.5 else "#2a3e40"
                        st.markdown(f"""
                        <div class="card" style="border-color:{color1}44; text-align:center; padding:32px 24px;">
                            <div class="card-sub" style="margin-bottom:8px;">Player 1</div>
                            <div class="card-title" style="font-size:1.5rem; margin-bottom:16px;">{joueur1}</div>
                            <div class="proba-pct" style="color:{color1};">{proba:.1%}</div>
                            <div class="bar-track" style="margin-top:16px;">
                                <div class="{'bar-fill-green' if proba>=0.5 else 'bar-fill-dim'}"
                                     style="width:{proba*100:.1f}%;"></div>
                            </div>
                            {"<div style='margin-top:12px; font-size:0.75rem; color:#3dd68c; letter-spacing:2px; text-transform:uppercase;'>FAVOURITE</div>" if proba>0.55 else ""}
                        </div>
                        """, unsafe_allow_html=True)

                    with rr2:
                        color2 = "#3dd68c" if proba < 0.5 else "#2a3e40"
                        st.markdown(f"""
                        <div class="card" style="border-color:{color2}44; text-align:center; padding:32px 24px;">
                            <div class="card-sub" style="margin-bottom:8px;">Player 2</div>
                            <div class="card-title" style="font-size:1.5rem; margin-bottom:16px;">{joueur2}</div>
                            <div class="proba-pct" style="color:{color2};">{(1-proba):.1%}</div>
                            <div class="bar-track" style="margin-top:16px;">
                                <div class="{'bar-fill-green' if proba<0.5 else 'bar-fill-dim'}"
                                     style="width:{(1-proba)*100:.1f}%;"></div>
                            </div>
                            {"<div style='margin-top:12px; font-size:0.75rem; color:#3dd68c; letter-spacing:2px; text-transform:uppercase;'>FAVOURITE</div>" if proba<0.45 else ""}
                        </div>
                        """, unsafe_allow_html=True)

                    conf_color = "#3dd68c" if conf >= 70 else "#f5c842" if conf >= 45 else "#e07878"
                    conf_label = "HIGH CONFIDENCE" if conf >= 70 else "MODERATE" if conf >= 45 else "LOW CONFIDENCE"

                    st.markdown(f"""
                    <div class="card" style="margin-top:16px; border-color:{conf_color}33;">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
                            <div>
                                <div class="card-sub">Confidence Score</div>
                                <div class="conf-score" style="color:{conf_color};">{conf}</div>
                                <div class="conf-label" style="color:{conf_color};">{conf_label}</div>
                            </div>
                            <div style="flex:1; min-width:200px;">
                    """, unsafe_allow_html=True)

                    maxs = [35, 25, 25, 15]
                    labels_fr = ["Prediction Strength", "Data Quality", "H2H Consistency", "Form Alignment"]
                    for i, (lbl, val) in enumerate(signals):
                        pct = val / maxs[i] * 100
                        st.markdown(f"""
                        <div style="margin-bottom:8px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                                <span style="font-size:0.7rem; color:#4a5e60; text-transform:uppercase; letter-spacing:1.5px;">{labels_fr[i]}</span>
                                <span style="font-size:0.7rem; color:#c8c0b0;">{val:.0f}/{maxs[i]}</span>
                            </div>
                            <div class="bar-track" style="height:4px;">
                                <div style="width:{pct:.0f}%; height:100%; border-radius:2px;
                                            background:linear-gradient(90deg,{conf_color}88,{conf_color});"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("</div></div></div>", unsafe_allow_html=True)

                    with st.expander("Feature vector details"):
                        st.dataframe(
                            pd.DataFrame({"Feature": FEATURES[:len(fv)], "Value": fv}),
                            hide_index=True, use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Erreur lors de la prédiction: {str(e)}")
                    if st.checkbox("Afficher les détails de l'erreur"):
                        st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE
# ══════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    tour_exp = st.selectbox("Circuit", ["ATP", "WTA"], key="tour_exp")
    df_exp = atp_data if tour_exp == "ATP" else wta_data

    if df_exp is None or len(df_exp) == 0:
        st.warning(f"No {tour_exp} data available.")
    else:
        all_pl = sorted(pd.concat([df_exp["winner_name"], df_exp["loser_name"]]).dropna().unique())
        player_search = st.selectbox("Search player", all_pl, key="player_exp")

        surf_filter = st.selectbox("Surface filter", ["All", "Hard", "Clay", "Grass"], key="surf_exp")
        surface_f = None if surf_filter == "All" else surf_filter

        s = get_player_stats(df_exp, player_search, surface_f, n_stats=20, n_form=10)

        if s:
            st.markdown(f"""
            <div style="margin:20px 0 12px 0;">
                <span class="card-title" style="font-size:1.6rem;">{player_search}</span>
                <span style="margin-left:12px;">{surface_badge(surf_filter) if surf_filter != "All" else ""}</span>
            </div>
            """, unsafe_allow_html=True)

            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("Ranking", f"#{int(s['rank'])}" if s['rank'] else "N/A")
            g2.metric("Points", f"{int(s['rank_pts'])}" if s['rank_pts'] else "N/A")
            g3.metric("Age", f"{s['age']:.1f}" if s['age'] else "N/A")
            g4.metric("Win rate", f"{s['win_pct']:.0%}")
            g5.metric("Fatigue", f"{s['fatigue']}")

            st.markdown(f"""
            <div class="card" style="margin-top:16px;">
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0 32px;">
                    <div>
                        <div class="card-sub" style="padding:12px 0 6px 0;">SERVICE</div>
                        {stat_html("ACES / MATCH", s['ace_avg'])}
                        {stat_html("DBL FAULTS", s['df_avg'])}
                        {stat_html("1ST SERVE IN %", s['pct_1st_in'])}
                        {stat_html("1ST SERVE WON %", s['pct_1st_won'])}
                        {stat_html("2ND SERVE WON %", s['pct_2nd_won'])}
                        {stat_html("BREAK PTS SAVED %", s['pct_bp_saved'])}
                    </div>
                    <div>
                        <div class="card-sub" style="padding:12px 0 6px 0;">RETURN</div>
                        {stat_html("RETURN 1ST WON %", s['pct_ret_1st'])}
                        {stat_html("RETURN 2ND WON %", s['pct_ret_2nd'])}
                        <div class="card-sub" style="padding:12px 0 6px 0;">RECENT</div>
                        {stat_html("FORM (last 10)", s['form_pct'])}
                        {stat_html("W/L", f"{s['wins']}/{s['played']}")}
                        {stat_html("LAST SEEN", s['last_date'].strftime('%Y-%m-%d') if s['last_date'] else "N/A")}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            mask = (df_exp["winner_name"] == player_search) | (df_exp["loser_name"] == player_search)
            recent_m = df_exp[mask].sort_values("tourney_date", ascending=False).head(10)[[
                "tourney_date", "tourney_name", "surface", "round", "winner_name", "loser_name", "score"
            ]].copy()
            
            if len(recent_m) > 0:
                recent_m["tourney_date"] = recent_m["tourney_date"].dt.strftime("%Y-%m-%d")
                recent_m["result"] = recent_m["winner_name"].apply(
                    lambda w: "✓ Win" if w == player_search else "✗ Loss"
                )

                st.markdown('<div style="margin-top:20px;" class="card-sub">LAST 10 MATCHES</div>', unsafe_allow_html=True)
                st.dataframe(
                    recent_m[["tourney_date", "tourney_name", "surface", "round", "result", "score"]].rename(columns={
                        "tourney_date": "Date", "tourney_name": "Tournament",
                        "surface": "Surface", "round": "Round", "result": "Result", "score": "Score"
                    }),
                    use_container_width=True, hide_index=True
                )
        else:
            st.info("No data found for this player.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — MODELS INFO
# ══════════════════════════════════════════════════════════════
with tab_models:
    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    meta = load_model_metadata()

    st.markdown("""
    <div class="card-title" style="font-size:1.4rem; margin-bottom:4px;">Model Architecture</div>
    <div class="card-sub" style="margin-bottom:20px;">6 specialized neural networks — ATP & WTA × Hard/Clay/Grass</div>
    """, unsafe_allow_html=True)

    for tour_k in ["atp", "wta"]:
        st.markdown(f"""
        <div style="margin:16px 0 8px 0;">
            {tour_badge(tour_k.upper())}
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(3)
        for i, surf in enumerate(SURFACES):
            mp = MODELS_DIR / f"tennis_model_{tour_k}_{surf.lower()}.h5"
            sp = MODELS_DIR / f"tennis_scaler_{tour_k}_{surf.lower()}.joblib"
            ok = mp.exists()
            res = {}
            if meta and "models" in meta:
                res = meta["models"].get(f"{tour_k}_{surf}", {})

            with cols[i]:
                status_color = "#3dd68c" if ok else "#2a3e40"
                status_text = "READY" if ok else "NOT TRAINED"
                acc_str = f"{res.get('accuracy', 0) * 100:.1f}%" if res else "—"
                auc_str = f"{res.get('auc', 0):.4f}" if res else "—"
                last_ft = res.get('last_finetuned', '—') if res else "—"

                st.markdown(f"""
                <div class="card" style="border-color:{status_color}33;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        {surface_badge(surf)}
                        <span style="font-size:0.65rem; color:{status_color}; letter-spacing:2px; text-transform:uppercase;">{status_text}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-key">ACCURACY</span>
                        <span class="stat-val" style="color:{status_color};">{acc_str}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-key">AUC</span>
                        <span class="stat-val">{auc_str}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-key">SCALER</span>
                        <span class="stat-val">{"✓" if sp.exists() else "✗"}</span>
                    </div>
                    <div class="stat-row" style="border:none;">
                        <span class="stat-key">LAST UPDATE</span>
                        <span class="stat-val" style="font-size:0.75rem;">{last_ft}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card-title" style="margin-bottom:12px;">Feature Engineering ({len(FEATURES)} features)</div>
    """, unsafe_allow_html=True)

    feat_groups = {
        "Ranking & Profile": ["rank_diff", "pts_diff", "age_diff"],
        "Recent Form": ["form_diff", "fatigue_diff"],
        "Serve": ["ace_diff", "df_diff", "pct_1st_in_diff", "pct_1st_won_diff", "pct_2nd_won_diff", "pct_bp_saved_diff"],
        "Return": ["pct_ret_1st_diff", "pct_ret_2nd_diff"],
        "H2H & Context": ["h2h_score", "best_of", "surface_hard", "surface_clay", "surface_grass", "level_gs", "level_m1000", "level_500"],
    }

    for grp, feats in feat_groups.items():
        st.markdown(f'<div class="card-sub" style="margin:10px 0 4px 0;">{grp}</div>', unsafe_allow_html=True)
        feat_html = " ".join([f'<span class="badge badge-hard" style="margin:2px;">{f}</span>' for f in feats if f in FEATURES])
        st.markdown(f'<div style="display:flex; flex-wrap:wrap; gap:4px;">{feat_html}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-custom">
    TennisIQ · Educational Project · No Guarantee of Profit · Play Responsibly
</div>
""", unsafe_allow_html=True)
