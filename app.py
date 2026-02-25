import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

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
    "surf_wr_diff", "surf_matches_diff",
    "days_since_last_diff", "p1_returning", "p2_returning",
]

SURFACES = ["Hard", "Clay", "Grass"]
TOURS = {"ATP": "atp", "WTA": "wta"}
ATP_ONLY = True
START_YEAR = 2007

TOURNAMENTS_ATP = [
    ("Australian Open", "Hard", "G", 5),
    ("Roland Garros", "Clay", "G", 5),
    ("Wimbledon", "Grass", "G", 5),
    ("US Open", "Hard", "G", 5),
    ("Indian Wells Masters", "Hard", "M", 3),
    ("Miami Open", "Hard", "M", 3),
    ("Monte-Carlo Masters", "Clay", "M", 3),
    ("Madrid Open", "Clay", "M", 3),
    ("Italian Open", "Clay", "M", 3),
    ("Canadian Open", "Hard", "M", 3),
    ("Cincinnati Masters", "Hard", "M", 3),
    ("Shanghai Masters", "Hard", "M", 3),
    ("Paris Masters", "Hard", "M", 3),
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
    ("Nitto ATP Finals", "Hard", "F", 3),
    ("Brisbane International","Hard", "A", 3),
    ("Adelaide International","Hard", "A", 3),
    ("Auckland Open", "Hard", "A", 3),
    ("Doha", "Hard", "A", 3),
    ("Montpellier", "Hard", "A", 3),
    ("Marseille", "Hard", "A", 3),
    ("Buenos Aires", "Clay", "A", 3),
    ("Delray Beach", "Hard", "A", 3),
    ("Santiago", "Clay", "A", 3),
    ("Estoril", "Clay", "A", 3),
    ("Munich", "Clay", "A", 3),
    ("Lyon", "Clay", "A", 3),
    ("Geneva", "Clay", "A", 3),
    ("Nottingham", "Grass", "A", 3),
    ("Stuttgart", "Grass", "A", 3),
    ("Eastbourne", "Grass", "A", 3),
    ("Gstaad", "Clay", "A", 3),
    ("Umag", "Clay", "A", 3),
    ("Kitzbuhel", "Clay", "A", 3),
    ("Los Cabos", "Hard", "A", 3),
    ("Atlanta", "Hard", "A", 3),
    ("Newport", "Grass", "A", 3),
    ("Bastad", "Clay", "A", 3),
    ("Metz", "Hard", "A", 3),
    ("Chengdu", "Hard", "A", 3),
    ("Hangzhou", "Hard", "A", 3),
    ("Antwerp", "Hard", "A", 3),
    ("Stockholm", "Hard", "A", 3),
    ("St. Petersburg", "Hard", "A", 3),
    ("Cordoba", "Clay", "A", 3),
    ("Dallas", "Hard", "A", 3),
    ("San Diego", "Hard", "A", 3),
    ("Florence", "Clay", "A", 3),
    ("Astana", "Hard", "A", 3),
    ("Pune", "Hard", "A", 3),
]

TOURN_DICT = {t[0]: (t[1], t[2], t[3]) for t in TOURNAMENTS_ATP}
TOURN_NAMES = [t[0] for t in TOURNAMENTS_ATP]

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
.badge-hard { background: #1a2e4a; color: #5ba3f5; border: 1px solid #2a4a6a; }
.badge-clay { background: #3a1a0a; color: #e07840; border: 1px solid #5a2a0a; }
.badge-grass { background: #0a2a14; color: #4caf6a; border: 1px solid #0a4a1e; }
.badge-atp { background: #1a1a3a; color: #7a8af5; border: 1px solid #2a2a5a; }
.badge-wta { background: #3a0a2a; color: #f57ab0; border: 1px solid #5a0a3a; }
.badge-gs { background: #2a2000; color: #f5c842; border: 1px solid #4a3800; }
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
.proba-pct-dim { color: #2a3e40; }
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
.stat-val-red { color: #e07878; }
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
# CHARGEMENT MODÈLES
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(tour, surface):
    path = MODELS_DIR / f"tennis_model_{tour}_{surface}.h5"
    if not path.exists():
        path = MODELS_DIR / f"tennis_model_{tour}_{surface.lower()}.h5"
    if not path.exists():
        path = MODELS_DIR / f"tennis_model_{surface.lower()}.h5"
    if not path.exists():
        return None
    try:
        from tensorflow.keras.models import load_model as km
        return km(str(path))
    except Exception as e:
        st.error(f"Erreur modèle {tour} {surface}: {e}")
        return None

@st.cache_resource
def load_scaler(tour, surface):
    path = MODELS_DIR / f"tennis_scaler_{tour}_{surface}.joblib"
    if not path.exists():
        path = MODELS_DIR / f"tennis_scaler_{tour}_{surface.lower()}.joblib"
    if not path.exists():
        path = MODELS_DIR / f"tennis_scaler_{surface.lower()}.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(str(path))
    except:
        return None

@st.cache_data
def load_meta():
    p = MODELS_DIR / "tennis_features_meta.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DONNÉES (inchangé)
# ─────────────────────────────────────────────────────────────
# ... (ton code de chargement des données reste identique, je ne le recopie pas ici pour ne pas alourdir)

# ─────────────────────────────────────────────────────────────
# SIDEBAR (inchangé)
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # ... (ton code sidebar reste identique)

# ─────────────────────────────────────────────────────────────
# HEADER (inchangé)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 32px 0 24px 0;">
    <p class="hero-title">TennisIQ</p>
    <p class="hero-sub">Neural Network Prediction Engine · ATP 2007–2026</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

if not atp_ok and not wta_ok:
    st.error("Aucune donnée trouvée dans `src/data/raw/tml-tennis/`. Vérifiez vos fichiers CSV.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# TABS (inchangé)
# ─────────────────────────────────────────────────────────────
tab_pred, tab_multi, tab_comb, tab_explore, tab_models, tab_hist = st.tabs([
    "⚡ PREDICT",
    "📋 MULTI-MATCH",
    "🎰 COMBINÉ",
    "🔍 EXPLORE",
    "📊 MODELS",
    "📜 HISTORIQUE",
])

# ─────────────────────────────────────────────────────────────
# TAB PREDICT (inchangé)
# ─────────────────────────────────────────────────────────────
with tab_pred:
    # ... (ton code PREDICT reste identique)

# ─────────────────────────────────────────────────────────────
# TAB MULTI-MATCH (inchangé)
# ─────────────────────────────────────────────────────────────
with tab_multi:
    # ... (ton code MULTI-MATCH reste identique)

# ══════════════════════════════════════════════════════════════
# TAB — COMBINÉ   ← ICI SEULEMENT LA MODIFICATION
# ══════════════════════════════════════════════════════════════
with tab_comb:
    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:20px;">
        <div class="card-title" style="margin-bottom:6px;">🎰 Constructeur de Combiné</div>
        <div style="font-size:0.82rem; color:#4a5e60;">
            Ajoute tes sélections · l'app calcule la proba réelle et te dit si le combiné vaut le coup
        </div>
    </div>
    """, unsafe_allow_html=True)

    if atp_data is None:
        st.error("Données ATP non disponibles.")
    else:
        # Chargement global des cotes (inchangé)
        cb_col1, cb_col2 = st.columns([2, 3])
        with cb_col1:
            if st.button("🔍 Charger toutes les cotes live", key="comb_load_odds",
                         help="Une seule requête API pour tous tes matchs — économise le quota"):
                with st.spinner("Chargement des cotes ATP en cours..."):
                    global_odds = fetch_all_atp_odds()
                st.session_state["odds_global_index"] = global_odds
                if global_odds.get("ok"):
                    n_t = global_odds.get("total", 0)
                    cr = global_odds.get("credits_remaining", "?")
                    keys = global_odds.get("sport_keys", [])
                    st.success(
                        f"✅ {n_t} matchs trouvés · {len(keys)} tournois ATP "
                        f"· {cr} crédits restants"
                    )
                else:
                    st.warning("⚠️ API inaccessible — saisis les cotes manuellement.")
        
        with cb_col2:
            gidx = st.session_state.get("odds_global_index", {})
            if gidx.get("ok"):
                st.markdown(f'<div style="color:#3dd68c; font-size:0.8rem; padding-top:10px;">✅ {gidx.get("total",0)} matchs chargés</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#e07878; font-size:0.8rem; padding-top:10px;">⚠️ API indisponible — saisie manuelle</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Paramètres
        n_comb = st.number_input("Nombre de sélections", min_value=2, max_value=20, value=4, step=1, key="comb_n")
        mise_comb = st.number_input("Mise (€)", min_value=0.10, max_value=1000.0, value=1.0, step=0.10, key="comb_mise", format="%.2f")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Saisie des sélections + cote manuelle (AJOUT ICI)
        selections = []
        for ci in range(int(n_comb)):
            st.markdown(f'<div style="font-size:0.7rem; color:#3dd68c; letter-spacing:3px; text-transform:uppercase; margin-bottom:8px;">Sélection {ci+1}</div>', unsafe_allow_html=True)
            
            sc1, sc2, sc3 = st.columns([2, 2, 3])
            with sc1:
                j1_c = st.selectbox("Joueur 1", atp_player_list, key=f"comb_j1_{ci}", index=None, placeholder="Joueur 1...")
            with sc2:
                j2_opts = [p for p in atp_player_list if p != j1_c] if j1_c else atp_player_list
                j2_c = st.selectbox("Joueur 2", j2_opts, key=f"comb_j2_{ci}", index=None, placeholder="Joueur 2...")
            with sc3:
                tourn_c = st.selectbox("Tournoi", TOURN_NAMES, key=f"comb_tourn_{ci}")

            surf_c, level_c, bo_c = TOURN_DICT.get(tourn_c, ("Hard","A",3))
            sc_color = {"Hard":"#4a90d9","Clay":"#c8703a","Grass":"#3dd68c"}.get(surf_c,"#4a5e60")

            # Sélection joueur + cote manuelle
            oc1, oc2, oc3 = st.columns([3, 2, 1])
            with oc1:
                sel_player = st.selectbox(
                    "Jouer sur",
                    [j1_c, j2_c] if j1_c and j2_c else ["—"],
                    key=f"comb_sel_{ci}",
                    label_visibility="collapsed"
                ) if j1_c and j2_c else None

            with oc2:
                cote_input = st.text_input(
                    "Cote du joueur sélectionné",
                    key=f"comb_cote_{ci}",
                    placeholder="ex: 1.45 ou 2.10",
                    help="Cote décimale du joueur que tu choisis (point ou virgule acceptée)"
                )

            with oc3:
                st.markdown(f'<div style="background:{sc_color}22; color:{sc_color}; border:1px solid {sc_color}44; padding:6px 10px; border-radius:8px; font-size:0.72rem; text-align:center; margin-top:28px;">{surf_c}</div>', unsafe_allow_html=True)

            # Conversion cote
            cote_val = None
            if cote_input:
                try:
                    cote_val = float(cote_input.replace(",", "."))
                    if cote_val <= 1.0:
                        cote_val = None
                except:
                    cote_val = None

            selections.append({
                "j1": j1_c,
                "j2": j2_c,
                "joueur": sel_player,
                "surface": surf_c,
                "level": level_c,
                "best_of": bo_c,
                "cote": cote_val,
                "tournoi": tourn_c,
            })

            if ci < int(n_comb)-1:
                st.markdown('<div style="border-top:1px solid #1a2a2c; margin:12px 0;"></div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Bouton analyser
        col_comb_btn = st.columns([1,2,1])
        with col_comb_btn[1]:
            comb_clicked = st.button("⚡ ANALYSER LE COMBINÉ", use_container_width=True, key="comb_btn")

        if comb_clicked:
            valid_sels = [s for s in selections if s["j1"] and s["j2"] and s["joueur"] and s["joueur"] != "—"]
            if len(valid_sels) < 2:
                st.warning("Renseigne au moins 2 sélections complètes.")
            else:
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                results_comb = []
                model_cache_c, scaler_cache_c = {}, {}

                for s in valid_sels:
                    if s["surface"] not in model_cache_c:
                        model_cache_c[s["surface"]] = load_model("atp", s["surface"])
                        scaler_cache_c[s["surface"]] = load_scaler("atp", s["surface"])

                    model_c = model_cache_c[s["surface"]]
                    scaler_c = scaler_cache_c[s["surface"]]

                    s1_c = get_player_stats(atp_data, s["j1"], s["surface"])
                    s2_c = get_player_stats(atp_data, s["j2"], s["surface"])
                    h2h_c = get_h2h(atp_data, s["j1"], s["j2"], s["surface"])

                    proba_c = None
                    if model_c and s1_c and s2_c:
                        try:
                            n_c = model_c.input_shape[-1]
                        except Exception:
                            n_c = 26
                        fv_c = build_feature_vector(s1_c, s2_c, h2h_c["h2h_score"],
                                                    s["surface"], float(s["best_of"]),
                                                    s["level"], n_features=n_c)
                        X_c = np.array(fv_c).reshape(1,-1)
                        if scaler_c:
                            try:
                                if getattr(scaler_c,"n_features_in_",None) == X_c.shape[1]:
                                    X_c = scaler_c.transform(X_c)
                            except:
                                pass
                        raw = float(model_c.predict(X_c, verbose=0)[0][0])
                        proba_c = raw if s["joueur"] == s["j1"] else 1 - raw
                    else:
                        proba_c = 0.55

                    results_comb.append({
                        **s,
                        "proba_model": proba_c,
                    })

                # Affichage récapitulatif
                st.markdown("""
                <div style="font-size:0.72rem; color:#4a5e60; letter-spacing:2px;
                            text-transform:uppercase; margin-bottom:12px;">Récapitulatif des sélections</div>
                """, unsafe_allow_html=True)

                for r in results_comb:
                    cote_txt = f"{r['cote']:.2f}" if r['cote'] else "— (manquant)"
                    st.markdown(
                        f'<div style="display:flex; align-items:center; gap:16px; padding:12px; background:#111a1c; border-radius:10px; margin-bottom:8px;">'
                        f'<div style="flex:1;"><strong>{r["joueur"]}</strong><br><small>{r["j1"]} vs {r["j2"]} · {r["tournoi"]}</small></div>'
                        f'<div style="min-width:80px; text-align:center;">'
                        f'<div style="font-size:0.65rem; color:#4a5e60;">PROBA</div>'
                        f'<div style="font-size:1.4rem; color:#3dd68c;">{r["proba_model"]:.0%}</div>'
                        f'</div>'
                        f'<div style="min-width:100px; text-align:center;">'
                        f'<div style="font-size:0.65rem; color:#4a5e60;">COTE</div>'
                        f'<div style="font-size:1.4rem; color:#c8c0b0;">{cote_txt}</div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # Calcul global
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                proba_globale = 1.0
                cote_globale = 1.0
                all_cotes_ok = True

                for r in results_comb:
                    proba_globale *= r["proba_model"]
                    if r["cote"]:
                        cote_globale *= r["cote"]
                    else:
                        all_cotes_ok = False

                gain_potentiel = mise_comb * cote_globale if all_cotes_ok else None
                esperance = proba_globale * gain_potentiel - mise_comb if all_cotes_ok else None

                sg1, sg2, sg3 = st.columns(3)
                with sg1:
                    st.metric("Probabilité réelle", f"{proba_globale:.1%}")
                with sg2:
                    if all_cotes_ok:
                        st.metric("Cote combinée", f"{cote_globale:.2f}")
                    else:
                        st.metric("Cote combinée", "—", delta="Saisis toutes les cotes")
                with sg3:
                    if all_cotes_ok:
                        st.metric("Espérance", f"{esperance:+.2f} €")
                    else:
                        st.metric("Espérance", "—", delta="Manque cotes")

                if not all_cotes_ok:
                    st.info("Pour voir la cote combinée et l'espérance, saisis la cote de chaque sélection ci-dessus.")

# (le reste du code : tab_explore, tab_models, tab_hist, footer reste inchangé)
