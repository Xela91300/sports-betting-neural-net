import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import joblib
import json

# ────────────────────────────────────────────────
# Chemins
# ────────────────────────────────────────────────
ROOT_DIR           = Path(__file__).parent
MODELS_DIR         = ROOT_DIR / "models"
DATA_DIR           = ROOT_DIR / "src" / "data"
DATA_RAW_DIR       = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_PATH        = ROOT_DIR / "config" / "config.yaml"

for directory in [MODELS_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

config = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

# ────────────────────────────────────────────────
# Features avancées (18 features)
# ────────────────────────────────────────────────
TENNIS_FEATURES = [
    "rank_diff", "pts_diff", "age_diff",
    "form_diff",
    "fatigue_diff",
    "ace_diff", "df_diff",
    "pct_1st_in_diff", "pct_1st_won_diff", "pct_2nd_won_diff",
    "pct_bp_saved_diff",
    "pct_ret_1st_diff", "pct_ret_2nd_diff",
    "h2h_score",
    "best_of",
    "surface_hard", "surface_clay", "surface_grass",
]

SPORT_CONFIG = {
    "Football": {
        "model_path": MODELS_DIR / "football_model.h5",
        "scaler_path": MODELS_DIR / "football_scaler.joblib",
        "features": config.get("football", {}).get("features", [
            "home_form_5", "away_form_5", "home_goals_avg", "away_goals_avg",
            "diff_classement", "odds_home", "odds_draw", "odds_away"
        ]),
        "desc": "Victoire de l'équipe à domicile",
        "type": "classification",
        "data_pattern": "*football*.csv"
    },
    "Tennis": {
        "features": TENNIS_FEATURES,
        "desc": "Victoire du joueur 1",
        "type": "classification",
        "data_pattern": "*tennis*.csv"
    },
    "Basketball": {
        "model_path": MODELS_DIR / "basketball_model.h5",
        "scaler_path": MODELS_DIR / "basketball_scaler.joblib",
        "features": config.get("basketball", {}).get("features", [
            "points_avg_home", "reb_avg_home", "eff_rating_home",
            "back_to_back", "spread", "points_avg_away"
        ]),
        "desc": "Victoire de l'équipe à domicile",
        "type": "classification",
        "data_pattern": "*basket*.csv"
    }
}

# ────────────────────────────────────────────────
# Chargement modèles tennis (un par surface)
# ────────────────────────────────────────────────
@st.cache_resource
def load_tennis_model(surface):
    """Charge le modèle spécialisé pour une surface donnée."""
    path = MODELS_DIR / f"tennis_model_{surface.lower()}.h5"
    if not path.exists():
        # Fallback sur modèle générique
        path = MODELS_DIR / "tennis_model.h5"
    if not path.exists():
        return None
    try:
        from tensorflow.keras.models import load_model
        return load_model(str(path))
    except Exception as e:
        st.error(f"Erreur chargement modèle Tennis {surface}: {e}")
        return None

@st.cache_resource
def load_tennis_scaler(surface):
    """Charge le scaler spécialisé pour une surface donnée."""
    path = MODELS_DIR / f"tennis_scaler_{surface.lower()}.joblib"
    if not path.exists():
        path = MODELS_DIR / "tennis_scaler.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(str(path))
    except:
        return None

@st.cache_resource
def load_cached_model(sport):
    if sport == "Tennis":
        return None  # Tennis utilise load_tennis_model(surface)
    cfg  = SPORT_CONFIG[sport]
    path = cfg.get("model_path")
    if not path or not path.exists():
        return None
    try:
        from tensorflow.keras.models import load_model
        return load_model(str(path))
    except Exception as e:
        st.error(f"Erreur chargement modèle {sport}: {e}")
        return None

@st.cache_resource
def load_cached_scaler(sport):
    if sport == "Tennis":
        return None
    cfg  = SPORT_CONFIG[sport]
    path = cfg.get("scaler_path")
    if path and path.exists():
        try:
            return joblib.load(str(path))
        except:
            return None
    return None

# ────────────────────────────────────────────────
# Métadonnées modèles tennis
# ────────────────────────────────────────────────
@st.cache_data
def load_tennis_meta():
    meta_path = MODELS_DIR / "tennis_features_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None

# ────────────────────────────────────────────────
# Datasets
# ────────────────────────────────────────────────
@st.cache_data(ttl=600)
def list_available_datasets(sport):
    pattern = SPORT_CONFIG[sport]["data_pattern"]
    candidates = []
    for base in [DATA_RAW_DIR, DATA_PROCESSED_DIR]:
        if base.exists():
            candidates.extend(base.rglob(pattern))
            candidates.extend(base.rglob(f"**/*{sport.lower()}*.csv"))
    if sport == "Tennis":
        tml_dir = DATA_RAW_DIR / "tml-tennis"
        if tml_dir.exists():
            candidates.extend(tml_dir.glob("*.csv"))
    seen, unique = set(), []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    datasets = []
    for p in sorted(unique, key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_file():
            datasets.append({
                "name": p.name, "path": str(p),
                "size_kb": round(p.stat().st_size / 1024, 1),
                "modified": pd.Timestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                "location": str(p.relative_to(ROOT_DIR))
            })
    return datasets

@st.cache_data
def load_dataset(path_str):
    try:
        return pd.read_csv(path_str, low_memory=False)
    except Exception as e:
        st.error(f"Impossible de lire {Path(path_str).name}: {e}")
        return None

@st.cache_data(ttl=600)
def load_all_tennis_data():
    datasets = list_available_datasets("Tennis")
    if not datasets:
        return None
    dfs = []
    for d in datasets:
        df = load_dataset(d["path"])
        if df is not None:
            dfs.append(df)
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    combined["tourney_date"] = pd.to_datetime(
        combined["tourney_date"], format="%Y%m%d", errors="coerce"
    )
    combined = combined.sort_values("tourney_date").reset_index(drop=True)
    return combined

# ────────────────────────────────────────────────
# Stats avancées d'un joueur
# ────────────────────────────────────────────────
def get_advanced_stats(df, player, surface=None, n_stats=10, n_form=5, fatigue_days=7):
    """Stats complètes : service + retour + forme + fatigue, filtrées par surface."""

    as_w = df[df["winner_name"] == player].copy()
    as_w["is_w"]    = True
    as_w["ace"]     = as_w["w_ace"]
    as_w["df_c"]    = as_w["w_df"]
    as_w["svpt"]    = as_w["w_svpt"]
    as_w["1stIn"]   = as_w["w_1stIn"]
    as_w["1stWon"]  = as_w["w_1stWon"]
    as_w["2ndWon"]  = as_w["w_2ndWon"]
    as_w["bpS"]     = as_w["w_bpSaved"]
    as_w["bpF"]     = as_w["w_bpFaced"]
    as_w["rank"]    = as_w["winner_rank"]
    as_w["rank_pts"]= as_w["winner_rank_points"]
    as_w["age"]     = as_w["winner_age"]
    as_w["opp_1stIn"]  = as_w["l_1stIn"]
    as_w["opp_1stWon"] = as_w["l_1stWon"]
    as_w["opp_2ndWon"] = as_w["l_2ndWon"]

    as_l = df[df["loser_name"] == player].copy()
    as_l["is_w"]    = False
    as_l["ace"]     = as_l["l_ace"]
    as_l["df_c"]    = as_l["l_df"]
    as_l["svpt"]    = as_l["l_svpt"]
    as_l["1stIn"]   = as_l["l_1stIn"]
    as_l["1stWon"]  = as_l["l_1stWon"]
    as_l["2ndWon"]  = as_l["l_2ndWon"]
    as_l["bpS"]     = as_l["l_bpSaved"]
    as_l["bpF"]     = as_l["l_bpFaced"]
    as_l["rank"]    = as_l["loser_rank"]
    as_l["rank_pts"]= as_l["loser_rank_points"]
    as_l["age"]     = as_l["loser_age"]
    as_l["opp_1stIn"]  = as_l["w_1stIn"]
    as_l["opp_1stWon"] = as_l["w_1stWon"]
    as_l["opp_2ndWon"] = as_l["w_2ndWon"]

    cols = ["tourney_date", "surface", "is_w", "ace", "df_c",
            "svpt", "1stIn", "1stWon", "2ndWon", "bpS", "bpF",
            "rank", "rank_pts", "age",
            "opp_1stIn", "opp_1stWon", "opp_2ndWon"]

    all_m = pd.concat([as_w[cols], as_l[cols]], ignore_index=True
                      ).sort_values("tourney_date", ascending=False)

    if all_m.empty:
        return None

    # Forme sur N derniers matchs toutes surfaces
    form_m    = all_m.head(n_form)
    form_pct  = form_m["is_w"].sum() / len(form_m) if len(form_m) > 0 else 0.5

    # Fatigue : matchs dans les X derniers jours
    last_date = all_m["tourney_date"].iloc[0]
    cutoff    = last_date - pd.Timedelta(days=fatigue_days)
    fatigue   = (all_m["tourney_date"] >= cutoff).sum()

    # Stats filtrées par surface
    if surface:
        surf_m = all_m[all_m["surface"] == surface].head(n_stats)
        working = surf_m if len(surf_m) >= 3 else all_m.head(n_stats)
        surf_note = "sur cette surface" if len(surf_m) >= 3 else "toutes surfaces (données insuffisantes)"
    else:
        working   = all_m.head(n_stats)
        surf_note = "toutes surfaces"

    # Infos de base
    rank     = all_m["rank"].dropna().iloc[0]     if not all_m["rank"].dropna().empty     else None
    rank_pts = all_m["rank_pts"].dropna().iloc[0] if not all_m["rank_pts"].dropna().empty else None
    age      = all_m["age"].dropna().iloc[0]      if not all_m["age"].dropna().empty      else None

    def sp(num, den):
        n = working[num].sum()
        d = working[den].sum()
        return float(n / d) if d > 0 else None

    ace_avg = float(working["ace"].mean()) if working["ace"].notna().any() else None
    df_avg  = float(working["df_c"].mean()) if working["df_c"].notna().any() else None

    # % 2ème balle gagnée
    svpt_sum  = working["svpt"].sum()
    in1_sum   = working["1stIn"].sum()
    won2_sum  = working["2ndWon"].sum()
    pct_2nd   = float(won2_sum / (svpt_sum - in1_sum)) if (svpt_sum - in1_sum) > 0 else None

    # Stats retour : pts gagnés quand adversaire sert
    opp_in1_sum  = working["opp_1stIn"].sum()
    opp_won1_sum = working["opp_1stWon"].sum()
    opp_won2_sum = working["opp_2ndWon"].sum()
    pct_ret_1st = float((opp_in1_sum - opp_won1_sum) / opp_in1_sum) if opp_in1_sum > 0 else None

    wins   = int(working["is_w"].sum())
    played = len(working)

    return {
        "rank": rank, "rank_pts": rank_pts, "age": age,
        "form_pct":  form_pct,
        "fatigue":   int(fatigue),
        "ace_avg":   ace_avg,
        "df_avg":    df_avg,
        "pct_1st_in":   sp("1stIn", "svpt"),
        "pct_1st_won":  sp("1stWon", "1stIn"),
        "pct_2nd_won":  pct_2nd,
        "pct_bp_saved": sp("bpS", "bpF"),
        "pct_ret_1st":  pct_ret_1st,
        "pct_ret_2nd":  float(opp_won2_sum / played) if played > 0 else None,
        "wins": wins, "played": played,
        "win_pct": wins / played if played > 0 else 0,
        "surf_note": surf_note,
    }

# ────────────────────────────────────────────────
# H2H
# ────────────────────────────────────────────────
def get_h2h(df, j1, j2, surface=None):
    mask = (
        ((df["winner_name"] == j1) & (df["loser_name"] == j2)) |
        ((df["winner_name"] == j2) & (df["loser_name"] == j1))
    )
    h2h = df[mask].copy()
    h2h_surf = h2h[h2h["surface"] == surface] if surface else h2h

    j1_tot  = int((h2h["winner_name"] == j1).sum())
    j2_tot  = int((h2h["winner_name"] == j2).sum())
    j1_surf = int((h2h_surf["winner_name"] == j1).sum()) if surface else None
    j2_surf = int((h2h_surf["winner_name"] == j2).sum()) if surface else None

    h2h_score = j1_tot / len(h2h) if len(h2h) > 0 else 0.5

    recent = h2h.sort_values("tourney_date", ascending=False).head(5)[
        ["tourney_date", "tourney_name", "surface", "round", "winner_name", "loser_name", "score"]
    ].copy()
    recent["tourney_date"] = recent["tourney_date"].dt.strftime("%Y-%m-%d")

    return {
        "total": len(h2h), "j1_tot": j1_tot, "j2_tot": j2_tot,
        "j1_surf": j1_surf, "j2_surf": j2_surf,
        "surf_total": len(h2h_surf),
        "h2h_score": h2h_score,
        "recent": recent,
    }

# ────────────────────────────────────────────────
# Score de confiance
# ────────────────────────────────────────────────
def confidence_score(proba, s1, s2, h2h):
    signals = []
    pred_strength = abs(proba - 0.5) * 2
    signals.append(("Force prédiction",    pred_strength * 35))
    data_q = min(s1["played"], 15) / 15 * 0.5 + min(s2["played"], 15) / 15 * 0.5
    signals.append(("Qualité des données", data_q * 25))
    if h2h["total"] >= 2:
        favori_h2h = h2h["h2h_score"] if proba >= 0.5 else (1 - h2h["h2h_score"])
        signals.append(("Cohérence H2H",   favori_h2h * 25))
    else:
        signals.append(("Cohérence H2H",   12.5))
    form_agree = 1.0 if (
        (proba >= 0.5 and s1["form_pct"] >= s2["form_pct"]) or
        (proba < 0.5  and s2["form_pct"] >= s1["form_pct"])
    ) else 0.3
    signals.append(("Cohérence forme récente", form_agree * 15))
    total = sum(v for _, v in signals)
    return round(min(total, 100)), signals

# ────────────────────────────────────────────────
# Interface
# ────────────────────────────────────────────────
st.set_page_config(page_title="Sports Betting NN", page_icon="🎾⚽🏀", layout="wide")
st.title("Prédictions Paris Sportifs – Réseaux de Neurones")

with st.sidebar:
    st.header("Données détectées")
    for sn in SPORT_CONFIG:
        ds = list_available_datasets(sn)
        st.write(f"{'✅' if ds else '⚠️'} **{sn}**: {len(ds)} fichier(s)")
        if ds and st.checkbox(f"Détails {sn}", key=f"chk_{sn}"):
            for d in ds[:6]:
                st.caption(f"• {d['name']} ({d['size_kb']:.1f} KB)")

    # Performances modèles tennis
    meta = load_tennis_meta()
    if meta:
        st.markdown("---")
        st.markdown("**🎾 Performances modèles**")
        for surf, res in meta.get("results", {}).items():
            st.caption(f"• {surf}: {res['accuracy']*100:.1f}% acc | AUC {res['auc']:.3f}")
    st.markdown("---")
    st.caption("src/data/raw/tml-tennis/ | models/")

col_left, _ = st.columns([1, 4])
with col_left:
    sport = st.selectbox("Sport", list(SPORT_CONFIG.keys()))

if sport:
    cfg    = SPORT_CONFIG[sport]
    tab_pred, tab_data, tab_info = st.tabs(["Prédiction", "Données", "Infos"])

    # ══════════════════════════════════════════════
    # ONGLET PRÉDICTION TENNIS
    # ══════════════════════════════════════════════
    with tab_pred:
        st.subheader(f"Prédiction – {sport}")

        if sport == "Tennis":
            df_all = load_all_tennis_data()

            if df_all is None:
                st.error("Aucune donnée tennis trouvée dans src/data/raw/tml-tennis/")
            else:
                # Tournoi
                st.markdown("### 🏆 Tournoi")
                tournois = sorted(df_all["tourney_name"].dropna().unique())
                selected_tournoi = st.selectbox("Sélectionner un tournoi", tournois)

                df_t    = df_all[df_all["tourney_name"] == selected_tournoi]
                surface = df_t["surface"].iloc[0]       if not df_t.empty else "Hard"
                best_of = int(df_t["best_of"].iloc[0])  if not df_t.empty else 3

                model  = load_tennis_model(surface)
                scaler = load_tennis_scaler(surface)

                c1, c2, c3 = st.columns(3)
                c1.metric("🎾 Surface", surface)
                c2.metric("🔢 Format",  f"Best of {best_of}")
                c3.metric("🤖 Modèle",  f"Tennis {surface}" if model else "❌ Non trouvé")

                if not model:
                    st.warning(f"Modèle tennis_{surface.lower()}.h5 introuvable dans models/")
                else:
                    # Joueurs
                    st.markdown("### 👤 Joueurs")
                    all_players = sorted(pd.concat([
                        df_all["winner_name"], df_all["loser_name"]
                    ]).dropna().unique())

                    cj1, cj2 = st.columns(2)
                    with cj1:
                        joueur1 = st.selectbox("Joueur 1", all_players, key="j1")
                    with cj2:
                        joueur2 = st.selectbox("Joueur 2",
                            [p for p in all_players if p != joueur1], key="j2")

                    n_matches = st.slider("Matchs récents à analyser", 5, 30, 10)

                    # Calcul stats + H2H
                    s1  = get_advanced_stats(df_all, joueur1, surface, n_matches)
                    s2  = get_advanced_stats(df_all, joueur2, surface, n_matches)
                    h2h = get_h2h(df_all, joueur1, joueur2, surface)

                    # ── Stats ────────────────────
                    st.markdown(f"### 📊 Stats récentes — {surface}")
                    cs1, cs2 = st.columns(2)

                    def show_stats(col, name, s):
                        with col:
                            st.markdown(f"**{name}**")
                            if s is None:
                                st.warning("Aucune donnée")
                                return
                            st.caption(f"📌 {s['surf_note']} — {s['played']} matchs")
                            st.metric("Classement",    f"#{int(s['rank'])}"    if s['rank']     else "N/A")
                            st.metric("Points ATP",    f"{int(s['rank_pts'])}" if s['rank_pts'] else "N/A")
                            st.metric("Âge",           f"{s['age']:.1f} ans"  if s['age']      else "N/A")
                            st.metric("Forme récente", f"{s['form_pct']:.0%}")
                            st.metric("Fatigue",       f"{s['fatigue']} matchs / 7j")
                            st.metric("Victoires",     f"{s['wins']}/{s['played']} ({s['win_pct']:.0%})")
                            st.markdown("**Service**")
                            st.metric("Aces / match",  f"{s['ace_avg']:.1f}"      if s['ace_avg']      else "N/A")
                            st.metric("DF / match",    f"{s['df_avg']:.1f}"       if s['df_avg']       else "N/A")
                            st.metric("% 1ère entrée", f"{s['pct_1st_in']:.1%}"   if s['pct_1st_in']   else "N/A")
                            st.metric("% 1ère gagnée", f"{s['pct_1st_won']:.1%}"  if s['pct_1st_won']  else "N/A")
                            st.metric("% 2ème gagnée", f"{s['pct_2nd_won']:.1%}"  if s['pct_2nd_won']  else "N/A")
                            st.metric("% BP sauvées",  f"{s['pct_bp_saved']:.1%}" if s['pct_bp_saved'] else "N/A")
                            st.markdown("**Retour**")
                            st.metric("% Ret 1ère",    f"{s['pct_ret_1st']:.1%}"  if s['pct_ret_1st']  else "N/A")

                    show_stats(cs1, joueur1, s1)
                    show_stats(cs2, joueur2, s2)

                    # ── H2H ──────────────────────
                    st.markdown("### ⚔️ Historique H2H")
                    if h2h["total"] == 0:
                        st.info("Aucune confrontation directe dans les données.")
                    else:
                        ch1, ch2, ch3 = st.columns(3)
                        ch1.metric("Total matchs",       h2h["total"])
                        ch2.metric(f"Victoires {joueur1}", h2h["j1_tot"])
                        ch3.metric(f"Victoires {joueur2}", h2h["j2_tot"])
                        if h2h["surf_total"] > 0:
                            st.caption(f"Sur {surface} : {joueur1} {h2h['j1_surf']}–{h2h['j2_surf']} {joueur2} ({h2h['surf_total']} matchs)")
                        if not h2h["recent"].empty:
                            with st.expander("📋 5 dernières confrontations"):
                                st.dataframe(h2h["recent"].rename(columns={
                                    "tourney_date": "Date", "tourney_name": "Tournoi",
                                    "surface": "Surface", "round": "Tour",
                                    "winner_name": "Vainqueur", "loser_name": "Perdant",
                                    "score": "Score"
                                }), use_container_width=True, hide_index=True)

                    # ── Prédiction ────────────────
                    st.markdown("---")
                    if st.button("🔮 Prédire le vainqueur", type="primary"):
                        if s1 is None or s2 is None:
                            st.error("Données insuffisantes pour un des joueurs.")
                        else:
                            def sd(a, b, k):
                                va = a.get(k) if a else None
                                vb = b.get(k) if b else None
                                if va is not None and vb is not None and pd.notna(va) and pd.notna(vb):
                                    return float(va) - float(vb)
                                return 0.0

                            fv = [
                                sd(s1, s2, "rank"),
                                sd(s1, s2, "rank_pts"),
                                sd(s1, s2, "age"),
                                sd(s1, s2, "form_pct"),
                                sd(s1, s2, "fatigue"),
                                sd(s1, s2, "ace_avg"),
                                sd(s1, s2, "df_avg"),
                                sd(s1, s2, "pct_1st_in"),
                                sd(s1, s2, "pct_1st_won"),
                                sd(s1, s2, "pct_2nd_won"),
                                sd(s1, s2, "pct_bp_saved"),
                                sd(s1, s2, "pct_ret_1st"),
                                sd(s1, s2, "pct_ret_2nd"),
                                h2h["h2h_score"],
                                float(best_of),
                                int(surface == "Hard"),
                                int(surface == "Clay"),
                                int(surface == "Grass"),
                            ]

                            X = np.array(fv).reshape(1, -1)
                            if scaler:
                                X = scaler.transform(X)

                            with st.spinner("Prédiction en cours..."):
                                proba = float(model.predict(X, verbose=0)[0][0])

                            # Résultat
                            st.markdown("## 🏆 Résultat")
                            cr1, cr2 = st.columns(2)
                            with cr1:
                                st.metric(joueur1, f"{proba:.1%}")
                                st.progress(proba)
                            with cr2:
                                st.metric(joueur2, f"{1-proba:.1%}")
                                st.progress(1 - proba)

                            if proba > 0.65:
                                st.success(f"✅ **{joueur1}** favori selon le modèle")
                            elif proba < 0.35:
                                st.success(f"✅ **{joueur2}** favori selon le modèle")
                            else:
                                st.info("⚖️ Match très serré — faible confiance")

                            # Score de confiance
                            st.markdown("### 🎯 Score de confiance")
                            conf, signals = confidence_score(proba, s1, s2, h2h)
                            conf_color = "🟢" if conf >= 70 else "🟡" if conf >= 45 else "🔴"
                            conf_label = "Élevée" if conf >= 70 else "Modérée" if conf >= 45 else "Faible"
                            st.metric(f"{conf_color} Confiance {conf_label}", f"{conf} / 100")
                            st.progress(conf / 100)

                            with st.expander("🔍 Détail score de confiance"):
                                maxs = [35, 25, 25, 15]
                                for i, (label, val) in enumerate(signals):
                                    st.write(f"**{label}** : {val:.1f} / {maxs[i]}")
                                    st.progress(val / maxs[i])

                            with st.expander("🔬 Valeurs utilisées"):
                                st.dataframe(pd.DataFrame({
                                    "Feature": TENNIS_FEATURES,
                                    "Valeur": fv
                                }), hide_index=True)

        # ── Football / Basketball ─────────────────
        else:
            model  = load_cached_model(sport)
            scaler = load_cached_scaler(sport)
            if not model:
                st.warning(f"Modèle {sport} introuvable → {cfg.get('model_path', 'N/A')}")
            else:
                st.success(f"✅ Modèle chargé")
                user_values = {}
                cols = st.columns(3)
                for idx, feat in enumerate(cfg["features"]):
                    with cols[idx % 3]:
                        label = feat.replace("_", " ").title()
                        if "odds" in feat or "cote" in feat:
                            user_values[feat] = st.number_input(label, 1.01, 50.0, 2.0, 0.1)
                        else:
                            user_values[feat] = st.number_input(label, -100.0, 100.0, 0.0, 0.1)
                if st.button("🔮 Prédire", type="primary"):
                    try:
                        X = np.array([user_values.get(f, 0.0) for f in cfg["features"]]).reshape(1, -1)
                        if scaler:
                            X = scaler.transform(X)
                        proba = float(model.predict(X, verbose=0)[0][0])
                        st.metric("Probabilité victoire", f"{proba:.1%}")
                        st.progress(proba)
                    except Exception as e:
                        st.error(f"Erreur prédiction : {e}")

    # ══════════════════════════════════════════════
    # ONGLET DONNÉES
    # ══════════════════════════════════════════════
    with tab_data:
        st.subheader(f"Données – {sport}")
        datasets = list_available_datasets(sport)
        if datasets:
            st.dataframe(pd.DataFrame(datasets)[["name", "size_kb", "modified", "location"]])
            selected = st.selectbox("Fichier à explorer", [d["name"] for d in datasets])
            if selected:
                file = next(d for d in datasets if d["name"] == selected)
                df   = load_dataset(file["path"])
                if df is not None:
                    st.markdown(f"**{selected}** — {len(df):,} lignes, {len(df.columns)} colonnes")
                    st.dataframe(df.head(15))
                    with st.expander("Statistiques descriptives"):
                        st.dataframe(df.describe())
                    with st.expander("Colonnes"):
                        st.write(list(df.columns))
        else:
            st.info("Aucune donnée trouvée. Placez vos CSV dans src/data/raw/tml-tennis/")

    # ══════════════════════════════════════════════
    # ONGLET INFOS
    # ══════════════════════════════════════════════
    with tab_info:
        st.subheader("Informations techniques")
        meta = load_tennis_meta() if sport == "Tennis" else None

        if sport == "Tennis":
            st.markdown("**3 modèles spécialisés par surface :**")
            for surf in ["Hard", "Clay", "Grass"]:
                m_path = MODELS_DIR / f"tennis_model_{surf.lower()}.h5"
                s_path = MODELS_DIR / f"tennis_scaler_{surf.lower()}.joblib"
                m_ok   = "✅" if m_path.exists() else "❌"
                s_ok   = "✅" if s_path.exists() else "❌"
                st.markdown(f"- **{surf}** : modèle {m_ok}  scaler {s_ok}")
            if meta:
                st.markdown("**Performances :**")
                for surf, res in meta.get("results", {}).items():
                    st.markdown(f"- {surf}: `{res['accuracy']*100:.1f}%` accuracy | AUC `{res['auc']:.4f}`")
            st.markdown(f"**{len(TENNIS_FEATURES)} features :**")
            for f in TENNIS_FEATURES:
                st.markdown(f"- `{f}`")
        else:
            st.markdown("**Features :**")
            for f in cfg["features"]:
                st.markdown(f"- `{f}`")

        st.markdown("**Chemins :**")
        st.code(f"RAW    : {DATA_RAW_DIR}\nMODELS : {MODELS_DIR}")

st.markdown("---")
st.caption("Projet éducatif – Pas de garantie de gain – Jouez responsablement")
