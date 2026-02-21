import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import joblib

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
else:
    st.warning("⚠️ config.yaml non trouvé → configuration par défaut utilisée")

# ────────────────────────────────────────────────
# Configuration des sports
# ────────────────────────────────────────────────
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
        "model_path": MODELS_DIR / "tennis_model.h5",
        "scaler_path": MODELS_DIR / "tennis_scaler.joblib",
        "features": config.get("tennis", {}).get("features", [
            "rank_diff", "pts_diff", "age_diff",
            "surface_hard", "surface_clay", "surface_grass",
            "best_of", "ace_diff", "df_diff", "1st_pct_diff", "bp_pct_diff"
        ]),
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
# Chargement modèle / scaler
# ────────────────────────────────────────────────
@st.cache_resource
def load_cached_model(sport):
    path = SPORT_CONFIG[sport]["model_path"]
    if not path.exists():
        return None
    try:
        from tensorflow.keras.models import load_model
        return load_model(str(path))
    except Exception as e:
        st.error(f"Erreur chargement modèle {sport}: {e}")
        return None

@st.cache_resource
def load_cached_scaler(sport):
    path = SPORT_CONFIG[sport].get("scaler_path")
    if path and path.exists():
        try:
            return joblib.load(str(path))
        except:
            return None
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
def load_dataset(path_str: str):
    try:
        return pd.read_csv(path_str)
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
    return combined

# ────────────────────────────────────────────────
# Stats joueur filtrées par surface
# ────────────────────────────────────────────────
def get_player_stats(df, player_name, n_matches=10, surface=None):
    """Stats moyennes d'un joueur sur ses N derniers matchs, filtrées par surface si précisé."""

    as_winner = df[df["winner_name"] == player_name].copy()
    as_winner["is_winner"] = True
    as_winner["ace"]      = as_winner["w_ace"]
    as_winner["df_col"]   = as_winner["w_df"]
    as_winner["1stIn"]    = as_winner["w_1stIn"]
    as_winner["1stWon"]   = as_winner["w_1stWon"]
    as_winner["bpSaved"]  = as_winner["w_bpSaved"]
    as_winner["bpFaced"]  = as_winner["w_bpFaced"]
    as_winner["rank"]     = as_winner["winner_rank"]
    as_winner["rank_pts"] = as_winner["winner_rank_points"]
    as_winner["age"]      = as_winner["winner_age"]

    as_loser = df[df["loser_name"] == player_name].copy()
    as_loser["is_winner"] = False
    as_loser["ace"]      = as_loser["l_ace"]
    as_loser["df_col"]   = as_loser["l_df"]
    as_loser["1stIn"]    = as_loser["l_1stIn"]
    as_loser["1stWon"]   = as_loser["l_1stWon"]
    as_loser["bpSaved"]  = as_loser["l_bpSaved"]
    as_loser["bpFaced"]  = as_loser["l_bpFaced"]
    as_loser["rank"]     = as_loser["loser_rank"]
    as_loser["rank_pts"] = as_loser["loser_rank_points"]
    as_loser["age"]      = as_loser["loser_age"]

    cols = ["tourney_date", "tourney_name", "surface", "is_winner",
            "ace", "df_col", "1stIn", "1stWon",
            "bpSaved", "bpFaced", "rank", "rank_pts", "age"]

    all_matches = pd.concat(
        [as_winner[cols], as_loser[cols]], ignore_index=True
    ).sort_values("tourney_date", ascending=False)

    # ── Filtrage par surface ──────────────────────
    all_matches_global  = all_matches.head(n_matches)           # tous matchs
    all_matches_surface = all_matches[
        all_matches["surface"] == surface
    ].head(n_matches) if surface else all_matches_global        # filtrés surface

    # Si pas assez de matchs sur cette surface, fallback sur global
    working = all_matches_surface if len(all_matches_surface) >= 3 else all_matches_global
    surface_note = "sur cette surface" if len(all_matches_surface) >= 3 else "toutes surfaces (manque de données)"

    if working.empty:
        return None

    rank     = all_matches_global["rank"].dropna().iloc[0]     if not all_matches_global["rank"].dropna().empty     else None
    rank_pts = all_matches_global["rank_pts"].dropna().iloc[0] if not all_matches_global["rank_pts"].dropna().empty else None
    age      = all_matches_global["age"].dropna().iloc[0]      if not all_matches_global["age"].dropna().empty      else None

    ace_avg = working["ace"].mean()
    df_avg  = working["df_col"].mean()

    total_1stIn  = working["1stIn"].sum()
    total_1stWon = working["1stWon"].sum()
    pct_1st = (total_1stWon / total_1stIn) if total_1stIn > 0 else None

    total_bpFaced = working["bpFaced"].sum()
    total_bpSaved = working["bpSaved"].sum()
    pct_bp = (total_bpSaved / total_bpFaced) if total_bpFaced > 0 else None

    wins   = int(working["is_winner"].sum())
    played = len(working)

    return {
        "rank": rank, "rank_pts": rank_pts, "age": age,
        "ace_avg": ace_avg, "df_avg": df_avg,
        "pct_1st": pct_1st, "pct_bp": pct_bp,
        "wins": wins, "played": played,
        "win_pct": wins / played if played > 0 else 0,
        "surface_note": surface_note,
        "n_surface_matches": len(all_matches_surface),
    }

# ────────────────────────────────────────────────
# Historique H2H
# ────────────────────────────────────────────────
def get_h2h(df, joueur1, joueur2, surface=None):
    """Retourne l'historique des confrontations directes entre deux joueurs."""
    mask = (
        ((df["winner_name"] == joueur1) & (df["loser_name"] == joueur2)) |
        ((df["winner_name"] == joueur2) & (df["loser_name"] == joueur1))
    )
    h2h = df[mask].copy()

    if surface:
        h2h_surface = h2h[h2h["surface"] == surface]
    else:
        h2h_surface = h2h

    h2h_sorted = h2h.sort_values("tourney_date", ascending=False)

    j1_wins_total   = (h2h["winner_name"] == joueur1).sum()
    j2_wins_total   = (h2h["winner_name"] == joueur2).sum()
    j1_wins_surface = (h2h_surface["winner_name"] == joueur1).sum() if surface else None
    j2_wins_surface = (h2h_surface["winner_name"] == joueur2).sum() if surface else None

    recent = h2h_sorted.head(5)[
        ["tourney_date", "tourney_name", "surface", "round", "winner_name", "loser_name", "score"]
    ].copy()
    recent["tourney_date"] = recent["tourney_date"].dt.strftime("%Y-%m-%d")

    return {
        "total":            len(h2h),
        "j1_wins_total":    int(j1_wins_total),
        "j2_wins_total":    int(j2_wins_total),
        "j1_wins_surface":  int(j1_wins_surface) if j1_wins_surface is not None else None,
        "j2_wins_surface":  int(j2_wins_surface) if j2_wins_surface is not None else None,
        "surface_matches":  len(h2h_surface),
        "recent":           recent,
    }

# ────────────────────────────────────────────────
# Score de confiance
# ────────────────────────────────────────────────
def confidence_score(proba, stats1, stats2, h2h):
    """
    Score de confiance 0-100 basé sur :
    - Force de la prédiction (distance à 0.5)
    - Quantité de données disponibles
    - Cohérence avec l'historique H2H
    """
    signals = []

    # 1. Force de la prédiction (0-40 pts)
    pred_strength = abs(proba - 0.5) * 2   # 0 à 1
    signals.append(("Force prédiction",  pred_strength * 40))

    # 2. Données suffisantes (0-30 pts)
    data_quality = min(stats1["played"], 15) / 15 * 0.5 + min(stats2["played"], 15) / 15 * 0.5
    signals.append(("Qualité des données", data_quality * 30))

    # 3. Cohérence H2H (0-30 pts)
    h2h_score = 0
    if h2h["total"] >= 2:
        favori_proba = joueur1 if proba >= 0.5 else joueur2
        j1_h2h_pct = h2h["j1_wins_total"] / h2h["total"] if h2h["total"] > 0 else 0.5
        h2h_favori_pct = j1_h2h_pct if proba >= 0.5 else (1 - j1_h2h_pct)
        # Plus le H2H confirme la prédiction, plus le score est élevé
        h2h_score = h2h_favori_pct * 30
    else:
        h2h_score = 15  # pas assez de H2H → score neutre
    signals.append(("Cohérence H2H", h2h_score))

    total = sum(v for _, v in signals)
    return round(total), signals

# ────────────────────────────────────────────────
# Interface principale
# ────────────────────────────────────────────────
st.set_page_config(page_title="Sports Betting NN", page_icon="🎾⚽🏀", layout="wide")
st.title("Prédictions Paris Sportifs – Réseaux de Neurones")
st.caption("Données dans src/data/raw – Modèles dans models/")

with st.sidebar:
    st.header("Données détectées")
    for sport_name in SPORT_CONFIG:
        datasets = list_available_datasets(sport_name)
        icon = "✅" if datasets else "⚠️"
        st.write(f"{icon} **{sport_name}**: {len(datasets)} fichier(s)")
        if datasets and st.checkbox(f"Détails {sport_name}", key=f"chk_{sport_name}"):
            for ds in datasets[:6]:
                st.caption(f"• {ds['name']}  ({ds['size_kb']:.1f} KB)  – {ds['modified']}")
    st.markdown("---")
    st.caption("Chemins :\n• src/data/raw/tml-tennis/\n• models/")

col_left, _ = st.columns([1, 4])
with col_left:
    sport = st.selectbox("Sport", list(SPORT_CONFIG.keys()))

if sport:
    cfg    = SPORT_CONFIG[sport]
    model  = load_cached_model(sport)
    scaler = load_cached_scaler(sport)

    tab_pred, tab_data, tab_info = st.tabs(["Prédiction", "Données", "Infos"])

    # ══════════════════════════════════════════════
    # ONGLET PRÉDICTION
    # ══════════════════════════════════════════════
    with tab_pred:
        st.subheader(f"Prédiction – {sport}")

        if not model:
            st.warning(f"Modèle {sport} introuvable → {cfg['model_path']}")

        # ── Tennis ───────────────────────────────
        elif sport == "Tennis":
            df_all = load_all_tennis_data()

            if df_all is None:
                st.error("Aucune donnée tennis trouvée dans src/data/raw/tml-tennis/")
            else:
                st.success(f"✅ Modèle chargé — {len(df_all)} matchs disponibles")

                # Tournoi
                st.markdown("### 🏆 Tournoi")
                tournois = sorted(df_all["tourney_name"].dropna().unique())
                selected_tournoi = st.selectbox("Sélectionner un tournoi", tournois)

                df_tournoi = df_all[df_all["tourney_name"] == selected_tournoi]
                surface = df_tournoi["surface"].iloc[0] if not df_tournoi.empty else "Hard"
                best_of = int(df_tournoi["best_of"].iloc[0]) if not df_tournoi.empty else 3

                c1, c2 = st.columns(2)
                c1.metric("🎾 Surface", surface)
                c2.metric("🔢 Format", f"Best of {best_of}")

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

                # Calcul stats (filtrées par surface)
                stats1 = get_player_stats(df_all, joueur1, n_matches, surface)
                stats2 = get_player_stats(df_all, joueur2, n_matches, surface)
                h2h    = get_h2h(df_all, joueur1, joueur2, surface)

                # ── Stats côte à côte ─────────────
                st.markdown("### 📊 Stats récentes sur " + surface)

                def show_stats(col, name, s):
                    with col:
                        st.markdown(f"**{name}**")
                        if s is None:
                            st.warning("Aucune donnée disponible")
                            return
                        st.caption(f"📌 Stats calculées {s['surface_note']} ({s['played']} matchs)")
                        st.metric("Classement ATP", f"#{int(s['rank'])}"     if s['rank']     else "N/A")
                        st.metric("Points ATP",     f"{int(s['rank_pts'])}"  if s['rank_pts'] else "N/A")
                        st.metric("Âge",            f"{s['age']:.1f} ans"   if s['age']      else "N/A")
                        st.metric("Victoires",      f"{s['wins']}/{s['played']} ({s['win_pct']:.0%})")
                        st.metric("Aces / match",   f"{s['ace_avg']:.1f}"   if pd.notna(s['ace_avg']) else "N/A")
                        st.metric("DF / match",     f"{s['df_avg']:.1f}"    if pd.notna(s['df_avg'])  else "N/A")
                        st.metric("% 1ère balle",   f"{s['pct_1st']:.1%}"   if s['pct_1st']  else "N/A")
                        st.metric("% BP sauvées",   f"{s['pct_bp']:.1%}"    if s['pct_bp']   else "N/A")

                cs1, cs2 = st.columns(2)
                show_stats(cs1, joueur1, stats1)
                show_stats(cs2, joueur2, stats2)

                # ── H2H ──────────────────────────
                st.markdown("### ⚔️ Historique H2H")
                if h2h["total"] == 0:
                    st.info("Aucune confrontation directe dans les données disponibles.")
                else:
                    ch1, ch2, ch3 = st.columns(3)
                    ch1.metric("Total confrontations", h2h["total"])
                    ch2.metric(f"Victoires {joueur1}", h2h["j1_wins_total"])
                    ch3.metric(f"Victoires {joueur2}", h2h["j2_wins_total"])

                    if h2h["surface_matches"] > 0 and h2h["j1_wins_surface"] is not None:
                        st.caption(f"Sur {surface} : {joueur1} {h2h['j1_wins_surface']}–{h2h['j2_wins_surface']} {joueur2} ({h2h['surface_matches']} matchs)")

                    if not h2h["recent"].empty:
                        with st.expander("📋 5 dernières confrontations"):
                            st.dataframe(
                                h2h["recent"].rename(columns={
                                    "tourney_date": "Date", "tourney_name": "Tournoi",
                                    "surface": "Surface", "round": "Tour",
                                    "winner_name": "Vainqueur", "loser_name": "Perdant",
                                    "score": "Score"
                                }),
                                use_container_width=True, hide_index=True
                            )

                # ── Prédiction ────────────────────
                st.markdown("---")
                if st.button("🔮 Prédire le vainqueur", type="primary"):
                    if stats1 is None or stats2 is None:
                        st.error("Données insuffisantes pour un des joueurs.")
                    else:
                        def sd(a, b):
                            if a is not None and b is not None and pd.notna(a) and pd.notna(b):
                                return float(a) - float(b)
                            return 0.0

                        feature_vector = [
                            sd(stats1["rank"],     stats2["rank"]),
                            sd(stats1["rank_pts"], stats2["rank_pts"]),
                            sd(stats1["age"],      stats2["age"]),
                            int(surface == "Hard"),
                            int(surface == "Clay"),
                            int(surface == "Grass"),
                            float(best_of),
                            sd(stats1["ace_avg"],  stats2["ace_avg"]),
                            sd(stats1["df_avg"],   stats2["df_avg"]),
                            sd(stats1["pct_1st"],  stats2["pct_1st"]),
                            sd(stats1["pct_bp"],   stats2["pct_bp"]),
                        ]

                        X = np.array(feature_vector).reshape(1, -1)
                        if scaler:
                            X = scaler.transform(X)

                        with st.spinner("Prédiction en cours..."):
                            proba = float(model.predict(X, verbose=0)[0][0])

                        # ── Résultat ──────────────
                        st.markdown("## 🏆 Résultat")
                        cr1, cr2 = st.columns(2)
                        with cr1:
                            st.metric(joueur1, f"{proba:.1%}")
                            st.progress(proba)
                        with cr2:
                            st.metric(joueur2, f"{1 - proba:.1%}")
                            st.progress(1 - proba)

                        if proba > 0.65:
                            st.success(f"✅ **{joueur1}** favori selon le modèle")
                        elif proba < 0.35:
                            st.success(f"✅ **{joueur2}** favori selon le modèle")
                        else:
                            st.info("⚖️ Match très serré — faible confiance du modèle")

                        # ── Score de confiance ────
                        st.markdown("### 🎯 Score de confiance")
                        conf_score, signals = confidence_score(proba, stats1, stats2, h2h)

                        # Couleur selon le score
                        if conf_score >= 70:
                            conf_color = "🟢"
                            conf_label = "Confiance élevée"
                        elif conf_score >= 45:
                            conf_color = "🟡"
                            conf_label = "Confiance modérée"
                        else:
                            conf_color = "🔴"
                            conf_label = "Confiance faible"

                        st.metric(
                            f"{conf_color} {conf_label}",
                            f"{conf_score} / 100"
                        )
                        st.progress(conf_score / 100)

                        with st.expander("🔍 Détail du score de confiance"):
                            for label, val in signals:
                                max_val = 40 if "prédiction" in label.lower() else 30
                                st.write(f"**{label}** : {val:.1f} / {max_val}")
                                st.progress(val / max_val)
                            st.caption(
                                "Force prédiction (40 pts) : distance de la probabilité à 50%\n\n"
                                "Qualité des données (30 pts) : nombre de matchs analysés\n\n"
                                "Cohérence H2H (30 pts) : l'historique confirme-t-il la prédiction ?"
                            )

                        with st.expander("🔬 Valeurs utilisées pour la prédiction"):
                            st.dataframe(pd.DataFrame({
                                "Feature": ["rank_diff", "pts_diff", "age_diff",
                                            "surface_hard", "surface_clay", "surface_grass",
                                            "best_of", "ace_diff", "df_diff",
                                            "1st_pct_diff", "bp_pct_diff"],
                                "Valeur": feature_vector
                            }), hide_index=True)

        # ── Football / Basketball ─────────────────
        else:
            st.success(f"✅ Modèle chargé ({len(cfg['features'])} features)")
            st.info(f"Objectif : {cfg['desc']}")
            st.markdown("### Caractéristiques du match")
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
                    if proba > 0.65:
                        st.success("✅ Valeur potentielle détectée")
                    elif proba > 0.5:
                        st.info("ℹ️ Légère faveur")
                    else:
                        st.warning("⚠️ Faible probabilité")
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
                    st.markdown(f"**{selected}** — {len(df)} lignes, {len(df.columns)} colonnes")
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
        st.markdown("**Features du modèle :**")
        for f in cfg["features"]:
            st.markdown(f"- `{f}`")
        st.markdown("**Modèle :**")
        st.code(cfg['model_path'].name if cfg['model_path'].exists() else '❌ Non trouvé')
        st.markdown("**Scaler :**")
        st.code("✅ Présent" if scaler else "⚠️ Absent")
        st.markdown("**Chemins :**")
        st.code(f"RAW      : {DATA_RAW_DIR}\nPROCESSED: {DATA_PROCESSED_DIR}\nMODELS   : {MODELS_DIR}")

st.markdown("---")
st.caption("Projet éducatif – Pas de garantie de gain – Jouez responsablement")
