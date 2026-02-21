"""
Script d'entraînement – Modèle Tennis
Données : format TML (tourney_id, winner_rank, loser_rank, surface, stats...)
Usage   : python train_tennis.py
Sortie  : models/tennis_model.h5 + models/tennis_scaler.joblib
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow {tf.__version__}")

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "src" / "data" / "raw" / "tml-tennis"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── 1. Chargement de tous les CSV ─────────────────────────────────────────────
csv_files = sorted(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"Aucun CSV trouvé dans {DATA_DIR}")

print(f"\n📂 {len(csv_files)} fichier(s) trouvé(s) :")
for f in csv_files:
    print(f"   • {f.name}")

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
print(f"\n✅ Dataset total : {len(df)} matchs, {df.shape[1]} colonnes")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
# Surface → encodage one-hot
df["surface_hard"]  = (df["surface"] == "Hard").astype(int)
df["surface_clay"]  = (df["surface"] == "Clay").astype(int)
df["surface_grass"] = (df["surface"] == "Grass").astype(int)

# Différence de classement (winner - loser, du point de vue joueur A)
# On va créer 2 lignes par match : une où A=winner, une où A=loser → équilibre
def build_balanced_dataset(df):
    records = []

    for _, row in df.iterrows():
        w_rank = row.get("winner_rank", np.nan)
        l_rank = row.get("loser_rank", np.nan)
        w_pts  = row.get("winner_rank_points", np.nan)
        l_pts  = row.get("loser_rank_points", np.nan)
        w_age  = row.get("winner_age", np.nan)
        l_age  = row.get("loser_age", np.nan)
        best_of = row.get("best_of", 3)

        surf_h = int(row.get("surface_hard", 0))
        surf_c = int(row.get("surface_clay", 0))
        surf_g = int(row.get("surface_grass", 0))

        # Stats de service gagnant (si disponibles)
        w_ace   = row.get("w_ace", np.nan)
        w_df    = row.get("w_df", np.nan)
        w_svpt  = row.get("w_svpt", np.nan)
        w_1stIn = row.get("w_1stIn", np.nan)
        w_1stWon= row.get("w_1stWon", np.nan)
        w_bpSaved = row.get("w_bpSaved", np.nan)
        w_bpFaced = row.get("w_bpFaced", np.nan)

        l_ace   = row.get("l_ace", np.nan)
        l_df    = row.get("l_df", np.nan)
        l_svpt  = row.get("l_svpt", np.nan)
        l_1stIn = row.get("l_1stIn", np.nan)
        l_1stWon= row.get("l_1stWon", np.nan)
        l_bpSaved = row.get("l_bpSaved", np.nan)
        l_bpFaced = row.get("l_bpFaced", np.nan)

        # % 1ère balle
        w_1st_pct = (w_1stWon / w_1stIn) if (w_1stIn and w_1stIn > 0) else np.nan
        l_1st_pct = (l_1stWon / l_1stIn) if (l_1stIn and l_1stIn > 0) else np.nan

        # % bp sauvées
        w_bp_pct = (w_bpSaved / w_bpFaced) if (w_bpFaced and w_bpFaced > 0) else np.nan
        l_bp_pct = (l_bpSaved / l_bpFaced) if (l_bpFaced and l_bpFaced > 0) else np.nan

        base = {
            "surface_hard": surf_h,
            "surface_clay": surf_c,
            "surface_grass": surf_g,
            "best_of": best_of,
        }

        # Observation A = winner (label 1)
        records.append({
            **base,
            "rank_p1": w_rank,
            "rank_p2": l_rank,
            "rank_diff": (w_rank - l_rank) if (pd.notna(w_rank) and pd.notna(l_rank)) else np.nan,
            "pts_diff": (w_pts - l_pts) if (pd.notna(w_pts) and pd.notna(l_pts)) else np.nan,
            "age_diff": (w_age - l_age) if (pd.notna(w_age) and pd.notna(l_age)) else np.nan,
            "ace_diff": (w_ace - l_ace) if (pd.notna(w_ace) and pd.notna(l_ace)) else np.nan,
            "df_diff":  (w_df  - l_df)  if (pd.notna(w_df)  and pd.notna(l_df))  else np.nan,
            "1st_pct_diff": (w_1st_pct - l_1st_pct) if (pd.notna(w_1st_pct) and pd.notna(l_1st_pct)) else np.nan,
            "bp_pct_diff":  (w_bp_pct  - l_bp_pct)  if (pd.notna(w_bp_pct)  and pd.notna(l_bp_pct))  else np.nan,
            "label": 1
        })

        # Observation B = loser (label 0) → symétrie
        records.append({
            **base,
            "rank_p1": l_rank,
            "rank_p2": w_rank,
            "rank_diff": (l_rank - w_rank) if (pd.notna(l_rank) and pd.notna(w_rank)) else np.nan,
            "pts_diff": (l_pts - w_pts) if (pd.notna(l_pts) and pd.notna(w_pts)) else np.nan,
            "age_diff": (l_age - w_age) if (pd.notna(l_age) and pd.notna(w_age)) else np.nan,
            "ace_diff": (l_ace - w_ace) if (pd.notna(l_ace) and pd.notna(w_ace)) else np.nan,
            "df_diff":  (l_df  - w_df)  if (pd.notna(l_df)  and pd.notna(w_df))  else np.nan,
            "1st_pct_diff": (l_1st_pct - w_1st_pct) if (pd.notna(l_1st_pct) and pd.notna(w_1st_pct)) else np.nan,
            "bp_pct_diff":  (l_bp_pct  - w_bp_pct)  if (pd.notna(l_bp_pct)  and pd.notna(w_bp_pct))  else np.nan,
            "label": 0
        })

    return pd.DataFrame(records)

print("\n⚙️  Construction du dataset équilibré...")
data = build_balanced_dataset(df)
print(f"   Observations : {len(data)} (dont {data['label'].sum()} victoires)")

# ── 3. Nettoyage ──────────────────────────────────────────────────────────────
FEATURES = [
    "rank_diff", "pts_diff", "age_diff",
    "surface_hard", "surface_clay", "surface_grass",
    "best_of", "ace_diff", "df_diff", "1st_pct_diff", "bp_pct_diff"
]

# Remplir les NaN par la médiane de chaque feature
for feat in FEATURES:
    median = data[feat].median()
    data[feat] = data[feat].fillna(median)

X = data[FEATURES].values
y = data["label"].values

print(f"\n📊 Features utilisées ({len(FEATURES)}) : {FEATURES}")

# ── 4. Split train / test ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n🔀 Split : {len(X_train)} train / {len(X_test)} test")

# ── 5. Normalisation ──────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

scaler_path = MODELS_DIR / "tennis_scaler.joblib"
joblib.dump(scaler, scaler_path)
print(f"💾 Scaler sauvegardé → {scaler_path}")

# ── 6. Modèle ─────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Input(shape=(len(FEATURES),)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

model.summary()

# ── 7. Entraînement ───────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_auc", patience=15, restore_best_weights=True, mode="max"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5
    )
]

print("\n🚀 Entraînement...")
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=150,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# ── 8. Évaluation ─────────────────────────────────────────────────────────────
print("\n📈 Évaluation sur le jeu de test :")
loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"   Loss     : {loss:.4f}")
print(f"   Accuracy : {acc:.4f} ({acc*100:.1f}%)")
print(f"   AUC      : {auc:.4f}")

y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
print("\n📋 Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=["Défaite", "Victoire"]))

# ── 9. Sauvegarde du modèle ───────────────────────────────────────────────────
model_path = MODELS_DIR / "tennis_model.h5"
model.save(str(model_path))
print(f"\n✅ Modèle sauvegardé → {model_path}")

# ── 10. Résumé des features pour config.yaml ─────────────────────────────────
print("\n📝 Copiez ces features dans config/config.yaml :")
print("tennis:")
print("  features:")
for f in FEATURES:
    print(f"    - {f}")
