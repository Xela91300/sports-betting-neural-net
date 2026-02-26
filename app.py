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
import asyncio
import nest_asyncio
import os
import requests
import gzip

nest_asyncio.apply()
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION DES CHEMINS
# ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "src" / "data" / "raw" / "tml-tennis"
HIST_DIR = ROOT_DIR / "history"

for dir_path in [MODELS_DIR, DATA_DIR, HIST_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

HIST_FILE = HIST_DIR / "predictions_history.json"
COMB_HIST_FILE = HIST_DIR / "combines_history.json"
USER_STATS_FILE = HIST_DIR / "user_stats.json"

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
SURFACES = ["Hard", "Clay", "Grass"]
MAX_MATCHES_ANALYSIS = 30
MAX_MATCHES_COMBINE = 30
MIN_PROBA_COMBINE = 0.55
MIN_EDGE_COMBINE = 0.02
MAX_SELECTIONS_COMBINE = 30
MAX_COMBINE_SUGGESTIONS = 5

# ── TOURNOIS ATP avec surface automatique ──────────────────
TOURNAMENTS_ATP = [
    ("Australian Open",       "Hard",  "G",   5),
    ("Roland Garros",         "Clay",  "G",   5),
    ("Wimbledon",             "Grass", "G",   5),
    ("US Open",               "Hard",  "G",   5),
    ("Indian Wells Masters",  "Hard",  "M",   3),
    ("Miami Open",            "Hard",  "M",   3),
    ("Monte-Carlo Masters",   "Clay",  "M",   3),
    ("Madrid Open",           "Clay",  "M",   3),
    ("Italian Open",          "Clay",  "M",   3),
    ("Canadian Open",         "Hard",  "M",   3),
    ("Cincinnati Masters",    "Hard",  "M",   3),
    ("Shanghai Masters",      "Hard",  "M",   3),
    ("Paris Masters",         "Hard",  "M",   3),
    ("Rotterdam",             "Hard",  "500", 3),
    ("Dubai Tennis Champs",   "Hard",  "500", 3),
    ("Acapulco",              "Hard",  "500", 3),
    ("Barcelona Open",        "Clay",  "500", 3),
    ("Halle Open",            "Grass", "500", 3),
    ("Queen's Club",          "Grass", "500", 3),
    ("Hamburg Open",          "Clay",  "500", 3),
    ("Washington Open",       "Hard",  "500", 3),
    ("Tokyo",                 "Hard",  "500", 3),
    ("Vienna Open",           "Hard",  "500", 3),
    ("Basel",                 "Hard",  "500", 3),
    ("Beijing",               "Hard",  "500", 3),
    ("Nitto ATP Finals",      "Hard",  "F",   3),
    ("Geneva Open",           "Clay",  "250", 3),
    ("Lyon Open",             "Clay",  "250", 3),
    ("Estoril",               "Clay",  "250", 3),
    ("Marrakech",             "Clay",  "250", 3),
    ("Bastad",                "Clay",  "250", 3),
    ("Gstaad",                "Clay",  "250", 3),
    ("Kitzbuhel",             "Clay",  "250", 3),
    ("Umag",                  "Clay",  "250", 3),
    ("Winston-Salem",         "Hard",  "250", 3),
    ("Metz",                  "Hard",  "250", 3),
    ("Antwerp",               "Hard",  "250", 3),
    ("Stockholm",             "Hard",  "250", 3),
    ("Gijon",                 "Hard",  "250", 3),
    ("Doha",                  "Hard",  "250", 3),
    ("Adelaide",              "Hard",  "250", 3),
    ("Auckland",              "Hard",  "250", 3),
    ("Autre tournoi",         "Hard",  "250", 3),
]

# Dict pour lookup rapide: nom → (surface, level, best_of)
TOURN_DICT = {t[0]: {"surface": t[1], "level": t[2], "best_of": t[3]} for t in TOURNAMENTS_ATP}
TOURN_NAMES = [t[0] for t in TOURNAMENTS_ATP]

SURFACE_ICONS = {"Hard": "🟦", "Clay": "🟧", "Grass": "🟩"}

BET_TYPES = {
    "winner": "🏆 Gagnant du match",
    "over_under": "📊 Over/Under Games",
    "handicap": "⚖️ Handicap",
    "set_betting": "🎯 Score exact sets",
    "both_win_set": "🔄 Les deux gagnent un set"
}

STATUS_OPTIONS = {
    "en_attente": "⏳ En attente",
    "gagne": "✅ Gagné",
    "perdu": "❌ Perdu",
    "annule": "⚠️ Annulé"
}

COLORS = {
    "primary": "#00DFA2",
    "success": "#00DFA2",
    "warning": "#FFB200",
    "danger": "#FF3B3F",
    "gray": "#6C7A89",
}

# ─────────────────────────────────────────────────────────────
# TELEGRAM INTEGRATION
# ─────────────────────────────────────────────────────────────
def get_telegram_config():
    try:
        token = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        return token, str(chat_id)
    except Exception:
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if token and chat_id:
            return token, chat_id
        return None, None

def send_telegram_message(message, parse_mode='HTML'):
    token, chat_id = get_telegram_config()
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True
        }
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code != 200:
            print(f"Telegram error {response.status_code}: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Erreur Telegram: {e}")
        return False

def format_prediction_message(pred_data, bet_suggestions=None, ai_comment=None):
    proba = pred_data.get('proba', 0.5)
    bar_length = 10
    filled = int(proba * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)
    surface_emoji = SURFACE_ICONS.get(pred_data.get('surface', ''), '🎾')
    ml_tag = "🤖 " if pred_data.get('ml_used') else ""

    message = f"""
<b>{ml_tag}🎾 PRÉDICTION TENNISIQ</b>

<b>Match:</b> {pred_data.get('player1', '?')} vs {pred_data.get('player2', '?')}
<b>Tournoi:</b> {pred_data.get('tournament', 'Inconnu')}
<b>Surface:</b> {surface_emoji} {pred_data.get('surface', '?')}

<b>📊 ANALYSE DU MATCH:</b>
{bar}  {proba:.1%} / {1-proba:.1%}

• {pred_data.get('player1', 'J1')}: <b>{proba:.1%}</b>
• {pred_data.get('player2', 'J2')}: <b>{1-proba:.1%}</b>

<b>🏆 GAGNANT PRÉDIT:</b> {pred_data.get('favori', '?')}
<b>Confiance:</b> {'🟢' if pred_data.get('confidence', 0) >= 70 else '🟡' if pred_data.get('confidence', 0) >= 50 else '🔴'} {pred_data.get('confidence', 0):.0f}/100
"""
    if bet_suggestions:
        message += f"\n<b>🎯 PARIS ALTERNATIFS:</b>\n"
        for bet in bet_suggestions[:3]:
            conf_icon = '🟢' if bet['confidence'] >= 70 else '🟡' if bet['confidence'] >= 50 else '🔴'
            message += f"\n{conf_icon} <b>{bet['type']}</b>: {bet['description']}\n"
            message += f"   Probabilité: {bet['proba']:.1%} | Cote estimée: {bet['cote']:.2f}\n"
            if bet.get('edge', 0) > 0:
                message += f"   Edge: {bet['edge']*100:+.1f}%\n"

    if pred_data.get('best_value'):
        bv = pred_data['best_value']
        edge_color = '🟢' if bv['edge'] > 0.05 else '🟡'
        message += f"""
<b>🎯 VALUE BET DÉTECTÉ!</b>
{edge_color} <b>{bv['joueur']}</b> à <b>{bv['cote']:.2f}</b>
Edge: <b>{bv['edge']*100:+.1f}%</b>
"""
    if ai_comment:
        clean_comment = ai_comment.replace('<', '&lt;').replace('>', '&gt;')
        message += f"\n\n<b>🤖 ANALYSE IA:</b>\n{clean_comment}"

    message += f"\n\n#TennisIQ #{pred_data.get('surface', 'Tennis')}"
    return message

def format_combine_message(combine_data, ai_comment=None):
    proba = combine_data.get('proba_globale', 0)
    bar_length = 10
    filled = int(proba * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)
    ml_tag = "🤖 " if combine_data.get('ml_used') else ""

    message = f"""
<b>{ml_tag}🎰 COMBINÉ TENNISIQ</b>

<b>📊 Statistiques:</b>
{bar}  {proba:.1%}
• {combine_data.get('nb_matches', 0)} sélections
• Cote combinée: <b>{combine_data.get('cote_globale', 0):.2f}</b>
• Mise: <b>{combine_data.get('mise', 0):.2f}€</b>
• Gain potentiel: <b>{combine_data.get('gain_potentiel', 0):.2f}€</b>
• Espérance: <b>{combine_data.get('esperance', 0):+.2f}€</b>

<b>📋 Sélections:</b>
"""
    for i, sel in enumerate(combine_data.get('selections', [])[:5], 1):
        edge_color = '🟢' if sel.get('edge', 0) > 0.05 else '🟡'
        message += f"\n{i}. {edge_color} {sel.get('joueur', '?')} @ {sel.get('cote', 0):.2f} (edge: {sel.get('edge', 0)*100:+.1f}%)"

    if ai_comment:
        clean_comment = ai_comment.replace('<', '&lt;').replace('>', '&gt;')
        message += f"\n\n<b>🤖 ANALYSE IA:</b>\n{clean_comment}"

    message += f"\n\n#TennisIQ #Combiné"
    return message

def format_stats_message():
    stats = load_user_stats()
    history = load_history()
    total = stats.get('total_predictions', 0)
    correct = stats.get('correct_predictions', 0)
    incorrect = stats.get('incorrect_predictions', 0)
    annules = stats.get('annules_predictions', 0)
    total_valide = correct + incorrect
    accuracy = (correct / total_valide * 100) if total_valide > 0 else 0
    bar_length = 10
    filled = int(accuracy / 10)
    bar = '█' * filled + '░' * (bar_length - filled)
    recent = [p for p in history[-20:] if p.get('statut') in ['gagne', 'perdu']]
    recent_correct = sum(1 for p in recent if p.get('statut') == 'gagne')
    recent_acc = (recent_correct / len(recent) * 100) if recent else 0
    diff = recent_acc - accuracy

    message = f"""
<b>📊 STATISTIQUES TENNISIQ</b>

<b>🎯 Performance globale:</b>
{bar}  {accuracy:.1f}%

<b>📈 Détail:</b>
• Total prédictions: <b>{total}</b>
• ✅ Gagnées: <b>{correct}</b> ({accuracy:.1f}%)
• ❌ Perdues: <b>{incorrect}</b>
• ⚠️ Annulées: <b>{annules}</b>

<b>🔥 Dernières 20:</b>
• Correctes: <b>{recent_correct}/{len(recent)}</b>
• Précision: <b>{recent_acc:.1f}%</b> ({diff:+.1f}% vs globale)

<b>🏆 Records:</b>
• Meilleure série: <b>{stats.get('best_streak', 0)}</b>
• Série actuelle: <b>{stats.get('current_streak', 0)}</b> {'🔥' if stats.get('current_streak', 0) >= 5 else ''}

📅 Mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}

#TennisIQ #Stats
"""
    return message

def send_prediction_to_telegram(pred_data, bet_suggestions=None, ai_comment=None):
    return send_telegram_message(format_prediction_message(pred_data, bet_suggestions, ai_comment))

def send_combine_to_telegram(combine_data, ai_comment=None):
    return send_telegram_message(format_combine_message(combine_data, ai_comment))

def send_stats_to_telegram():
    return send_telegram_message(format_stats_message())

def test_telegram_connection():
    token, chat_id = get_telegram_config()
    if not token:
        return False, "❌ Token manquant"
    if not chat_id:
        return False, "❌ Chat ID manquant"
    try:
        test_message = f"""
<b>✅ TEST DE CONNEXION RÉUSSI !</b>

📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}
🤖 TennisIQ Bot prêt à recevoir des prédictions

#TennisIQ #Test
"""
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {'chat_id': chat_id, 'text': test_message, 'parse_mode': 'HTML'}
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            return True, "✅ Connexion réussie ! Message de test envoyé."
        else:
            return False, f"❌ Erreur: {resp.text}"
    except Exception as e:
        return False, f"❌ Exception: {str(e)}"

# ─────────────────────────────────────────────────────────────
# GROQ API (IA)
# ─────────────────────────────────────────────────────────────
def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return os.environ.get("GROQ_API_KEY", None)

def call_groq_api(prompt):
    api_key = get_groq_key()
    if not api_key:
        return None
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "Tu es un expert en analyse de tennis et paris sportifs. Fournis des analyses concises en français avec recommandations de paris."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return None
    except Exception as e:
        print(f"Exception Groq API: {e}")
        return None

def analyze_match_with_ai(player1, player2, surface, proba, best_value=None, bet_suggestions=None):
    vb_txt = ""
    if best_value:
        vb_txt = f" Value bet détecté sur {best_value['joueur']} avec un edge de {best_value['edge']*100:+.1f}%."
    bets_txt = ""
    if bet_suggestions:
        bets_txt = "\nParis alternatifs détectés:\n"
        for bet in bet_suggestions[:3]:
            bets_txt += f"- {bet['type']}: {bet['description']} (proba {bet['proba']:.1%})\n"

    prompt = f"""Analyse ce match de tennis en détail avec recommandations de paris:

Match: {player1} vs {player2}
Surface: {surface}
Probabilités: {player1} {proba:.1%} - {player2} {1-proba:.1%}
{vb_txt}
{bets_txt}

Donne une analyse structurée en français avec:
1. Analyse du match (facteurs clés, forme des joueurs, surface)
2. Pronostic sur le gagnant avec justification
3. Recommandations de paris alternatifs (Over/Under, handicap, etc.)
4. Niveau de confiance global

Sois concis mais précis.
"""
    return call_groq_api(prompt)

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE ML
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_saved_model():
    model_path = MODELS_DIR / "tennis_ml_model_complete.pkl"
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except:
            return None
    else:
        try:
            with st.spinner("📥 Téléchargement du modèle depuis GitHub..."):
                url = "https://github.com/Xela91300/sports-betting-neural-net/releases/download/v1.0.0/tennis_ml_model_complete.pkl.gz"
                response = requests.get(url, timeout=60)
                if response.status_code == 200:
                    temp_path = MODELS_DIR / "model_temp.pkl.gz"
                    with open(temp_path, "wb") as f:
                        f.write(response.content)
                    with gzip.open(temp_path, "rb") as f:
                        model_info = joblib.load(f)
                    joblib.dump(model_info, model_path)
                    temp_path.unlink()
                    return model_info
        except:
            pass
    return None

def predict_with_ml_model(model_info, player1, player2, surface='Hard'):
    if model_info is None:
        return None
    try:
        model = model_info.get('model')
        scaler = model_info.get('scaler')
        player_stats = model_info.get('player_stats', {})
        if model is None or scaler is None:
            return None
        s1 = player_stats.get(player1, {})
        s2 = player_stats.get(player2, {})
        if not s1 or not s2:
            return None
        r1 = max(s1.get('rank', 500.0), 1.0)
        r2 = max(s2.get('rank', 500.0), 1.0)
        log_rank_ratio = np.log(r2 / r1)
        surf_wr_diff = s1.get('surface_wr', {}).get(surface, 0.5) - s2.get('surface_wr', {}).get(surface, 0.5)
        career_wr_diff = s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5)
        features = np.array([[
            log_rank_ratio, 0, 0,
            1 if surface == 'Clay' else 0,
            1 if surface == 'Grass' else 0,
            1 if surface == 'Hard' else 0,
            0, 0, 0,
            surf_wr_diff, career_wr_diff, 0, 0.5,
            0, 0, 0, 0, 0, 0, 0, 0
        ]])
        features_scaled = scaler.transform(features)
        proba = model.predict_proba(features_scaled)[0][1]
        return max(0.05, min(0.95, float(proba)))
    except:
        return None

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES ATP — liste de joueurs pour dropdown
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_atp_data():
    if not DATA_DIR.exists():
        return pd.DataFrame()
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()
    atp_dfs = []
    for f in csv_files[:5]:
        if 'wta' in f.name.lower():
            continue
        try:
            df = pd.read_csv(f, encoding='utf-8', nrows=1000, on_bad_lines='skip')
            if 'winner_name' in df.columns and 'loser_name' in df.columns:
                atp_dfs.append(df)
        except:
            continue
    if atp_dfs:
        return pd.concat(atp_dfs, ignore_index=True)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_players_list(atp_data):
    """Retourne la liste triée de tous les joueurs du dataset."""
    if atp_data.empty:
        return []
    winners = set(atp_data['winner_name'].dropna().astype(str).str.strip().unique())
    losers  = set(atp_data['loser_name'].dropna().astype(str).str.strip().unique())
    players = sorted(winners | losers)
    return players

def get_player_stats(df, player):
    if df.empty or not player:
        return None
    player_clean = player.strip()
    matches = df[(df['winner_name'] == player_clean) | (df['loser_name'] == player_clean)]
    if len(matches) == 0:
        return None
    wins = len(matches[df['winner_name'] == player_clean])
    total = len(matches)
    return {'name': player_clean, 'matches_played': total,
            'wins': wins, 'losses': total - wins,
            'win_rate': wins / total if total > 0 else 0}

def get_h2h_stats(df, player1, player2):
    if df.empty or not player1 or not player2:
        return None
    p1, p2 = player1.strip(), player2.strip()
    if 'winner_name' not in df.columns:
        return None
    h2h = df[((df['winner_name'] == p1) & (df['loser_name'] == p2)) |
             ((df['winner_name'] == p2) & (df['loser_name'] == p1))]
    if len(h2h) == 0:
        return None
    return {
        'total_matches': len(h2h),
        f'{p1}_wins': len(h2h[df['winner_name'] == p1]),
        f'{p2}_wins': len(h2h[df['winner_name'] == p2]),
    }

def calculate_probability(df, player1, player2, surface, h2h=None, model_info=None):
    if model_info:
        ml_proba = predict_with_ml_model(model_info, player1, player2, surface)
        if ml_proba is not None:
            return ml_proba, True
    stats1 = get_player_stats(df, player1)
    stats2 = get_player_stats(df, player2)
    proba = 0.5
    if stats1 and stats2:
        proba += (stats1['win_rate'] - stats2['win_rate']) * 0.3
    if h2h and h2h.get('total_matches', 0) > 0:
        wins1 = h2h.get(f'{player1}_wins', 0)
        proba += (wins1 / h2h['total_matches'] - 0.5) * 0.2
    return max(0.05, min(0.95, proba)), False

def calculate_confidence(proba, h2h=None):
    confidence = 50
    if h2h and h2h.get('total_matches', 0) >= 3:
        confidence += 10
    confidence += abs(proba - 0.5) * 40
    return min(100, confidence)

# ─────────────────────────────────────────────────────────────
# PARIS ALTERNATIFS
# ─────────────────────────────────────────────────────────────
def generate_alternative_bets(player1, player2, surface, proba, h2h=None):
    suggestions = []
    if proba > 0.6 or proba < 0.4:
        suggestions.append({'type': '📊 Under 22.5 games', 'description': "Moins de 22.5 jeux dans le match",
                            'proba': 0.65 if abs(proba - 0.5) > 0.2 else 0.55, 'cote': 1.75,
                            'confidence': 70 if abs(proba - 0.5) > 0.25 else 60, 'edge': 0.03})
    else:
        suggestions.append({'type': '📊 Over 22.5 games', 'description': "Plus de 22.5 jeux dans le match",
                            'proba': 0.62, 'cote': 1.80, 'confidence': 65, 'edge': 0.02})
    if proba > 0.65:
        suggestions.append({'type': '⚖️ Handicap -3.5', 'description': f"{player1} gagne avec au moins 4 jeux d'écart",
                            'proba': 0.58, 'cote': 2.10, 'confidence': 60, 'edge': 0.04})
    elif proba < 0.35:
        suggestions.append({'type': '⚖️ Handicap +3.5', 'description': f"{player2} perd par moins de 4 jeux ou gagne",
                            'proba': 0.62, 'cote': 1.95, 'confidence': 65, 'edge': 0.03})
    if 0.3 < proba < 0.7:
        suggestions.append({'type': '🔄 Les deux gagnent un set', 'description': "Chaque joueur remporte au moins un set",
                            'proba': 0.55, 'cote': 2.20, 'confidence': 55, 'edge': 0.01})
    if proba > 0.7:
        suggestions.append({'type': '🎯 Score 2-0', 'description': f"{player1} gagne 2-0",
                            'proba': 0.52, 'cote': 2.50, 'confidence': 50, 'edge': 0.02})
    elif proba < 0.3:
        suggestions.append({'type': '🎯 Score 0-2', 'description': f"{player2} gagne 2-0",
                            'proba': 0.51, 'cote': 2.60, 'confidence': 50, 'edge': 0.01})
    if h2h and h2h.get('total_matches', 0) >= 3:
        for bet in suggestions:
            bet['confidence'] = min(95, bet['confidence'] + 5)
    return suggestions

# ─────────────────────────────────────────────────────────────
# HISTORIQUE & STATS
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
    try:
        history = load_history()
        pred_data['id'] = hashlib.md5(f"{datetime.now()}{pred_data.get('player1','')}".encode()).hexdigest()[:8]
        pred_data['statut'] = 'en_attente'
        history.append(pred_data)
        if len(history) > 1000:
            history = history[-1000:]
        with open(HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        return True
    except:
        return False

def update_prediction_status(pred_id, new_status):
    try:
        history = load_history()
        for pred in history:
            if pred.get('id') == pred_id:
                pred['statut'] = new_status
                pred['date_maj'] = datetime.now().isoformat()
                break
        with open(HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        update_user_stats()
        return True
    except:
        return False

def load_user_stats():
    if not USER_STATS_FILE.exists():
        return {'total_predictions': 0, 'correct_predictions': 0, 'incorrect_predictions': 0,
                'annules_predictions': 0, 'total_combines': 0, 'won_combines': 0,
                'total_invested': 0, 'total_won': 0, 'best_streak': 0, 'current_streak': 0}
    try:
        with open(USER_STATS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def update_user_stats():
    history = load_history()
    total = len(history)
    correct = sum(1 for p in history if p.get('statut') == 'gagne')
    incorrect = sum(1 for p in history if p.get('statut') == 'perdu')
    annules = sum(1 for p in history if p.get('statut') == 'annule')
    current_streak = 0
    best_streak = 0
    streak = 0
    for pred in reversed(history):
        if pred.get('statut') == 'gagne':
            streak += 1
            current_streak = streak
            best_streak = max(best_streak, streak)
        elif pred.get('statut') == 'perdu':
            streak = 0
            current_streak = 0
    stats = {'total_predictions': total, 'correct_predictions': correct,
             'incorrect_predictions': incorrect, 'annules_predictions': annules,
             'total_combines': 0, 'won_combines': 0, 'total_invested': 0, 'total_won': 0,
             'current_streak': current_streak, 'best_streak': best_streak}
    try:
        with open(USER_STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
    except:
        pass
    return stats

def load_combines():
    if not COMB_HIST_FILE.exists():
        return []
    try:
        with open(COMB_HIST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []

def save_combine(combine_data):
    try:
        combines = load_combines()
        combine_data['date'] = datetime.now().isoformat()
        combine_data['id'] = hashlib.md5(f"{datetime.now()}".encode()).hexdigest()[:8]
        combine_data['statut'] = 'en_attente'
        combines.append(combine_data)
        if len(combines) > 200:
            combines = combines[-200:]
        with open(COMB_HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(combines, f, indent=2)
        return True
    except:
        return False

# ─────────────────────────────────────────────────────────────
# COMBINÉS RECOMMANDÉS
# ─────────────────────────────────────────────────────────────
def generate_recommended_combines(matches_analysis):
    if len(matches_analysis) < 2:
        return []
    matches_with_edge = [m for m in matches_analysis if m.get('best_value')]
    matches_with_edge.sort(key=lambda x: x['best_value']['edge'], reverse=True)
    suggestions = []

    if len(matches_with_edge) >= 2:
        top_edges = matches_with_edge[:min(3, len(matches_with_edge))]
        selections = [{'match': f"{m['player1']} vs {m['player2']}", 'joueur': m['best_value']['joueur'],
                       'proba': m['best_value']['proba'], 'cote': m['best_value']['cote'],
                       'edge': m['best_value']['edge']} for m in top_edges]
        suggestions.append({'name': '🔥 Top Value Bets', 'selections': selections,
                            'proba': np.prod([s['proba'] for s in selections]),
                            'cote': np.prod([s['cote'] for s in selections]),
                            'nb_matches': len(selections)})

    high_confidence = [m for m in matches_analysis if m.get('confidence', 0) >= 70]
    if len(high_confidence) >= 2:
        top_conf = high_confidence[:min(3, len(high_confidence))]
        selections = [{'match': f"{m['player1']} vs {m['player2']}", 'joueur': m['favori'],
                       'proba': m['proba'] if m['proba'] >= 0.5 else 1-m['proba'],
                       'cote': 1/m['proba'] if m['proba'] >= 0.5 else 1/(1-m['proba']),
                       'edge': 0.05} for m in top_conf]
        suggestions.append({'name': '💪 Haute Confiance', 'selections': selections,
                            'proba': np.prod([s['proba'] for s in selections]),
                            'cote': np.prod([s['cote'] for s in selections]),
                            'nb_matches': len(selections)})

    return suggestions[:MAX_COMBINE_SUGGESTIONS]

# ─────────────────────────────────────────────────────────────
# PAGE DASHBOARD
# ─────────────────────────────────────────────────────────────
def show_dashboard():
    st.markdown("<h2>🏠 Dashboard</h2>", unsafe_allow_html=True)
    stats = load_user_stats()
    history = load_history()
    total = stats.get('total_predictions', 0)
    correct = stats.get('correct_predictions', 0)
    incorrect = stats.get('incorrect_predictions', 0)
    pending = len([p for p in history if p.get('statut') == 'en_attente'])
    total_valide = correct + incorrect
    accuracy = (correct / total_valide * 100) if total_valide > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total", total)
    with col2: st.metric("✅ Gagnées", correct)
    with col3: st.metric("❌ Perdues", incorrect)
    with col4: st.metric("⏳ En attente", pending)

    st.markdown("### 🎯 Précision globale")
    st.progress(accuracy / 100)
    st.caption(f"{accuracy:.1f}% de réussite sur {total_valide} matchs résolus")

    model_info = load_saved_model()
    groq_key = get_groq_key()
    telegram_token, _ = get_telegram_config()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"✅ Modèle ML ({model_info.get('accuracy', 0):.1%})") if model_info else st.warning("⚠️ Modèle ML non chargé")
    with col2:
        st.success("✅ IA Groq") if groq_key else st.warning("⚠️ IA non configurée")
    with col3:
        st.success("✅ Telegram") if telegram_token else st.warning("⚠️ Telegram non configuré")

# ─────────────────────────────────────────────────────────────
# PAGE ANALYSE MULTI-MATCH  ←  PRINCIPALE MODIFICATION
# Menus déroulants joueurs + tournois, surface automatique
# ─────────────────────────────────────────────────────────────
def show_prediction():
    st.markdown("<h2>🎯 Analyse Multi-matchs avec Paris Alternatifs</h2>", unsafe_allow_html=True)

    model_info = load_saved_model()
    atp_data   = load_atp_data()
    players    = get_players_list(atp_data)

    # ── Options de saisie ────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_matches = st.number_input("Nombre de matchs", 1, MAX_MATCHES_ANALYSIS, 3)
    with col2:
        mise = st.number_input("Mise (€)", 1.0, 1000.0, 10.0)
    with col3:
        use_ai = st.checkbox("🤖 Analyser avec IA", True)
    with col4:
        send_all_tg = st.checkbox("📱 Envoyer tout sur Telegram", False)

    st.markdown("### 📝 Saisie des matchs")

    # ── Saisie de chaque match ────────────────────────────────
    matches = []
    for i in range(n_matches):
        with st.expander(f"Match {i+1}", expanded=(i == 0)):

            # Ligne 1 : tournoi (dropdown) + surface auto (lecture seule)
            col_t, col_s = st.columns([3, 1])
            with col_t:
                tournament = st.selectbox(
                    "🏆 Tournoi",
                    options=TOURN_NAMES,
                    index=0,
                    key=f"tourn_{i}"
                )
            with col_s:
                tourn_info = TOURN_DICT[tournament]
                surface    = tourn_info["surface"]
                surf_icon  = SURFACE_ICONS[surface]
                # Affichage surface automatique (non modifiable)
                st.markdown(f"**Surface**")
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);'
                    f'border-radius:8px;padding:0.5rem 0.75rem;font-size:1rem;margin-top:0.25rem;">'
                    f'{surf_icon} {surface}</div>',
                    unsafe_allow_html=True
                )

            # Ligne 2 : joueurs (dropdown avec recherche)
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if players:
                    p1 = st.selectbox(
                        "Joueur 1",
                        options=players,
                        index=0,
                        key=f"p1_{i}"
                    )
                else:
                    p1 = st.text_input("Joueur 1", key=f"p1_{i}", placeholder="Novak Djokovic")

            with col_p2:
                if players:
                    # Exclure p1 de la liste de p2
                    players_p2 = [p for p in players if p != p1]
                    p2 = st.selectbox(
                        "Joueur 2",
                        options=players_p2,
                        index=0,
                        key=f"p2_{i}"
                    )
                else:
                    p2 = st.text_input("Joueur 2", key=f"p2_{i}", placeholder="Carlos Alcaraz")

            # Ligne 3 : cotes bookmaker
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                odds1 = st.text_input(
                    f"Cote {p1[:20] if p1 else 'J1'}",
                    key=f"odds1_{i}",
                    placeholder="1.75"
                )
            with col_o2:
                odds2 = st.text_input(
                    f"Cote {p2[:20] if p2 else 'J2'}",
                    key=f"odds2_{i}",
                    placeholder="2.10"
                )

            matches.append({
                'player1':    p1.strip() if p1 else "",
                'player2':    p2.strip() if p2 else "",
                'surface':    surface,
                'tournament': tournament,
                'odds1':      odds1,
                'odds2':      odds2,
                'index':      i
            })

    # ── Bouton d'analyse ─────────────────────────────────────
    if not st.button("🔍 Analyser tous les matchs", type="primary", use_container_width=True):
        return

    valid_matches = [m for m in matches if m['player1'] and m['player2']]
    if not valid_matches:
        st.warning("Veuillez remplir au moins un match complet")
        return

    st.markdown("---")
    st.markdown("## 📊 Résultats de l'analyse complète")

    matches_analysis = []
    all_selections   = []

    for i, match in enumerate(valid_matches):
        p1, p2 = match['player1'], match['player2']
        surf   = match['surface']

        st.markdown(f"### Match {i+1}: {p1} vs {p2}")
        st.caption(f"{SURFACE_ICONS[surf]} {surf} · {match['tournament']}")

        h2h            = get_h2h_stats(atp_data, p1, p2)
        proba, ml_used = calculate_probability(atp_data, p1, p2, surf, h2h, model_info)
        confidence     = calculate_confidence(proba, h2h)

        # Value bet
        best_value = None
        if match['odds1'] and match['odds2']:
            try:
                o1 = float(match['odds1'].replace(',', '.'))
                o2 = float(match['odds2'].replace(',', '.'))
                edge1 = proba - 1/o1
                edge2 = (1-proba) - 1/o2
                if edge1 > edge2 and edge1 > MIN_EDGE_COMBINE:
                    best_value = {'joueur': p1, 'edge': edge1, 'cote': o1, 'proba': proba}
                elif edge2 > edge1 and edge2 > MIN_EDGE_COMBINE:
                    best_value = {'joueur': p2, 'edge': edge2, 'cote': o2, 'proba': 1-proba}
            except:
                pass

        bet_suggestions = generate_alternative_bets(p1, p2, surf, proba, h2h)

        # Affichage résultat
        favori = p1 if proba >= 0.5 else p2
        st.markdown(f"#### 🏆 GAGNANT PRÉDIT: **{favori}**")

        col1, col2 = st.columns(2)
        with col1: st.metric(p1, f"{proba:.1%}")
        with col2: st.metric(p2, f"{1-proba:.1%}")
        st.progress(float(proba))

        col1, col2, col3 = st.columns(3)
        with col1: st.caption(f"{'🤖 ML' if ml_used else '📊 Stats'}")
        with col2:
            conf_icon = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🔴"
            st.caption(f"Confiance: {conf_icon} {confidence:.0f}/100")
        with col3:
            if h2h:
                w1 = h2h.get(f"{p1}_wins", 0)
                w2 = h2h.get(f"{p2}_wins", 0)
                st.caption(f"H2H: {w1}-{w2}")

        if best_value:
            st.success(f"🎯 Value bet! {best_value['joueur']} @ {best_value['cote']:.2f} (edge: {best_value['edge']*100:+.1f}%)")
            all_selections.append({
                'match': f"{p1} vs {p2}",
                'joueur': best_value['joueur'],
                'proba': best_value['proba'],
                'cote': best_value['cote'],
                'edge': best_value['edge']
            })

        # Paris alternatifs
        if bet_suggestions:
            st.markdown("#### 🎯 Paris Alternatifs Recommandés")
            for bet in bet_suggestions:
                conf_icon = '🟢' if bet['confidence'] >= 70 else '🟡' if bet['confidence'] >= 50 else '🔴'
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.markdown(f"{conf_icon} **{bet['type']}**")
                    st.caption(bet['description'])
                with col2: st.metric("Probabilité", f"{bet['proba']:.1%}")
                with col3: st.metric("Cote", f"{bet['cote']:.2f}")
                with col4: st.metric("Edge", f"{bet.get('edge',0)*100:+.1f}%")

        # Analyse IA
        ai_comment = None
        if use_ai and get_groq_key():
            with st.spinner("🤖 Analyse IA..."):
                ai_comment = analyze_match_with_ai(p1, p2, surf, proba, best_value, bet_suggestions)
                if ai_comment:
                    with st.expander("Voir analyse IA complète"):
                        st.markdown(ai_comment)

        pred_data = {
            'player1': p1, 'player2': p2,
            'tournament': match['tournament'],
            'surface': surf,
            'proba': float(proba),
            'confidence': float(confidence),
            'odds1': match['odds1'] or None,
            'odds2': match['odds2'] or None,
            'favori': favori,
            'best_value': best_value,
            'bet_suggestions': bet_suggestions,
            'ml_used': ml_used,
            'date': datetime.now().isoformat()
        }
        matches_analysis.append(pred_data)

        # Boutons sauvegarde / Telegram par match
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button(f"💾 Sauvegarder match {i+1}", key=f"save_{i}"):
                if save_prediction(pred_data):
                    st.success("✅ Sauvegardé!")
        with col_b2:
            if st.button(f"📱 Telegram match {i+1}", key=f"tg_{i}"):
                token_check, _ = get_telegram_config()
                if not token_check:
                    st.error("❌ Telegram non configuré")
                elif send_prediction_to_telegram(pred_data, bet_suggestions, ai_comment):
                    st.success("✅ Envoyé sur Telegram!")
                else:
                    st.error("❌ Échec de l'envoi Telegram")

        st.divider()

    # Envoi groupé
    if send_all_tg and matches_analysis:
        ok = sum(1 for p in matches_analysis if send_prediction_to_telegram(p, p.get('bet_suggestions')))
        st.success(f"📱 {ok}/{len(matches_analysis)} matchs envoyés sur Telegram")

    # Combinés recommandés
    if len(all_selections) >= 2:
        st.markdown("## 🎰 Combinés recommandés")
        for idx, suggestion in enumerate(generate_recommended_combines(matches_analysis)):
            with st.expander(
                f"{suggestion['name']} — {suggestion['nb_matches']} matchs — Proba {suggestion['proba']:.1%}",
                expanded=(idx == 0)
            ):
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Probabilité", f"{suggestion['proba']:.1%}")
                with col2: st.metric("Cote", f"{suggestion['cote']:.2f}")
                with col3: st.metric("Gain potentiel", f"{mise * suggestion['cote']:.2f}€")

                st.markdown("**Sélections:**")
                for sel in suggestion['selections']:
                    st.caption(f"• {sel['joueur']} @ {sel['cote']:.2f}")

                combine_data = {
                    'selections': suggestion['selections'],
                    'proba_globale': suggestion['proba'],
                    'cote_globale': suggestion['cote'],
                    'mise': mise,
                    'gain_potentiel': mise * suggestion['cote'],
                    'esperance': suggestion['proba'] * (mise * suggestion['cote']) - mise,
                    'nb_matches': suggestion['nb_matches'],
                    'ml_used': any(m.get('ml_used', False) for m in matches_analysis)
                }
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button(f"💾 Sauvegarder combiné", key=f"save_comb_{idx}"):
                        save_combine(combine_data)
                        st.success("✅ Combiné sauvegardé!")
                with col_b2:
                    if st.button(f"📱 Envoyer combiné", key=f"tg_comb_{idx}"):
                        if send_combine_to_telegram(combine_data):
                            st.success("✅ Combiné envoyé!")
                        else:
                            st.error("❌ Échec de l'envoi")

# ─────────────────────────────────────────────────────────────
# PAGE EN ATTENTE
# ─────────────────────────────────────────────────────────────
def show_pending():
    st.markdown("<h2>⏳ Prédictions en attente</h2>", unsafe_allow_html=True)
    history = load_history()
    pending = [p for p in history if p.get('statut') == 'en_attente']
    if not pending:
        st.info("Aucune prédiction en attente")
        return
    st.caption(f"{len(pending)} prédiction(s) en attente de résultat")
    for pred in pending[::-1]:
        with st.expander(f"{pred.get('date', '')[:16]} - {pred['player1']} vs {pred['player2']}"):
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Surface", pred.get('surface', '—'))
            with col2: st.metric("Probabilité", f"{pred.get('proba', 0.5):.1%}")
            with col3: st.metric("Confiance", f"{pred.get('confidence', 0):.0f}")
            if pred.get('odds1') and pred.get('odds2'):
                st.caption(f"Cotes: {pred['player1']} @ {pred['odds1']} | {pred['player2']} @ {pred['odds2']}")
            if pred.get('best_value'):
                st.info(f"🎯 Value bet: {pred['best_value']['joueur']}")
            st.markdown("### 🎯 Résultat du match")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"✅ {pred['player1']} gagne", key=f"win1_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'gagne')
                    st.rerun()
            with col2:
                if st.button(f"✅ {pred['player2']} gagne", key=f"win2_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'gagne')
                    st.rerun()
            with col3:
                if st.button(f"❌ Perdu", key=f"loss_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'perdu')
                    st.rerun()
            if st.button(f"⚠️ Annuler le match", key=f"cancel_{pred['id']}", use_container_width=True):
                update_prediction_status(pred['id'], 'annule')
                st.rerun()

# ─────────────────────────────────────────────────────────────
# PAGE HISTORIQUE
# ─────────────────────────────────────────────────────────────
def show_history():
    st.markdown("<h2>📜 Historique complet</h2>", unsafe_allow_html=True)
    history = load_history()
    if not history:
        st.info("Aucune prédiction dans l'historique")
        return
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.multiselect(
            "Filtrer par statut",
            options=list(STATUS_OPTIONS.keys()),
            format_func=lambda x: STATUS_OPTIONS[x],
            default=list(STATUS_OPTIONS.keys())
        )
    with col2:
        search = st.text_input("🔍 Rechercher un joueur", "")
    filtered = [p for p in history if p.get('statut') in status_filter]
    if search:
        filtered = [p for p in filtered if
                   search.lower() in p.get('player1', '').lower() or
                   search.lower() in p.get('player2', '').lower()]
    st.caption(f"Affichage {len(filtered)}/{len(history)} prédictions")
    for pred in filtered[::-1]:
        status_icon = STATUS_OPTIONS.get(pred.get('statut'), "⏳")
        with st.expander(f"{status_icon} {pred.get('date', '')[:16]} - {pred['player1']} vs {pred['player2']}"):
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Surface", pred.get('surface', '—'))
            with col2: st.metric("Probabilité", f"{pred.get('proba', 0.5):.1%}")
            with col3: st.metric("Statut", STATUS_OPTIONS.get(pred.get('statut'), "Inconnu"))
            if pred.get('odds1') and pred.get('odds2'):
                st.caption(f"Cotes: {pred['player1']} @ {pred['odds1']} | {pred['player2']} @ {pred['odds2']}")
            if pred.get('best_value'):
                st.info(f"🎯 Value bet: {pred['best_value']['joueur']}")
            new_status = st.selectbox(
                "Modifier le statut",
                options=list(STATUS_OPTIONS.keys()),
                format_func=lambda x: STATUS_OPTIONS[x],
                index=list(STATUS_OPTIONS.keys()).index(pred.get('statut', 'en_attente')),
                key=f"edit_{pred['id']}"
            )
            if new_status != pred.get('statut'):
                if st.button("Mettre à jour", key=f"update_{pred['id']}"):
                    update_prediction_status(pred['id'], new_status)
                    st.rerun()

# ─────────────────────────────────────────────────────────────
# PAGE STATISTIQUES
# ─────────────────────────────────────────────────────────────
def show_statistics():
    st.markdown("<h2>📈 Statistiques détaillées</h2>", unsafe_allow_html=True)
    stats = load_user_stats()
    total = stats.get('total_predictions', 0)
    correct = stats.get('correct_predictions', 0)
    incorrect = stats.get('incorrect_predictions', 0)
    annules = stats.get('annules_predictions', 0)
    total_valide = correct + incorrect
    accuracy = (correct / total_valide * 100) if total_valide > 0 else 0
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total", total)
    with col2: st.metric("✅ Gagnées", correct, f"{accuracy:.1f}%")
    with col3: st.metric("❌ Perdues", incorrect)
    with col4: st.metric("⚠️ Annulées", annules)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Répartition")
        if total_valide > 0:
            data = pd.DataFrame({'Statut': ['Gagnées', 'Perdues'], 'Nombre': [correct, incorrect]})
            st.bar_chart(data.set_index('Statut'))
    with col2:
        st.markdown("### 🔥 Séries")
        st.metric("Série actuelle", stats.get('current_streak', 0))
        st.metric("Meilleure série", stats.get('best_streak', 0))
    if st.button("📱 Envoyer les stats sur Telegram", use_container_width=True):
        if send_stats_to_telegram():
            st.success("✅ Statistiques envoyées sur Telegram !")
        else:
            st.error("❌ Échec de l'envoi")

# ─────────────────────────────────────────────────────────────
# PAGE TELEGRAM
# ─────────────────────────────────────────────────────────────
def show_telegram():
    st.markdown("<h2>📱 Configuration Telegram</h2>", unsafe_allow_html=True)
    token, chat_id = get_telegram_config()
    if not token or not chat_id:
        st.warning("⚠️ Telegram non configuré")
        st.markdown("""
        ### Configuration requise :
        1. Va sur Telegram et cherche **@BotFather**
        2. Crée un bot avec `/newbot`
        3. Ajoute dans les secrets Streamlit :
        ```toml
        TELEGRAM_BOT_TOKEN = "ton_token_ici"
        TELEGRAM_CHAT_ID   = "ton_chat_id_ici"
        ```
        Pour obtenir ton chat_id, envoie un message à **@userinfobot**
        """)
        return
    st.success(f"✅ Telegram configuré (Chat ID: {chat_id})")
    if st.button("🔧 Tester la connexion", use_container_width=True):
        with st.spinner("Test en cours..."):
            success, msg = test_telegram_connection()
            st.success(msg) if success else st.error(msg)
    st.markdown("### 📊 Envoyer les statistiques")
    if st.button("📤 Envoyer les stats maintenant", use_container_width=True):
        with st.spinner("Envoi en cours..."):
            st.success("✅ Statistiques envoyées !") if send_stats_to_telegram() else st.error("❌ Échec de l'envoi")
    st.markdown("### 📝 Message personnalisé")
    with st.form("telegram_form"):
        message = st.text_area("Message", height=100)
        col1, col2 = st.columns(2)
        with col1: urgent = st.checkbox("🔴 Urgent")
        with col2: include_stats = st.checkbox("Inclure les stats")
        if st.form_submit_button("📤 Envoyer") and message:
            final_msg = ("🔴 URGENT\n\n" if urgent else "") + message
            if include_stats:
                final_msg += f"\n\n{format_stats_message()}"
            st.success("✅ Message envoyé !") if send_telegram_message(final_msg) else st.error("❌ Échec de l'envoi")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
def show_configuration():
    st.markdown("<h2>⚙️ Configuration</h2>", unsafe_allow_html=True)
    st.markdown("### 🤖 Modèle Machine Learning")
    model_info = load_saved_model()
    if model_info:
        st.success(f"✅ Modèle chargé · Accuracy: {model_info.get('accuracy', 0):.1%} · AUC-ROC: {model_info.get('auc', 0):.3f}")
        if st.button("🔄 Recharger le modèle"):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.warning("⚠️ Aucun modèle trouvé")
    st.markdown("### 📊 Statistiques actuelles")
    stats = load_user_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total prédictions", stats.get('total_predictions', 0))
        st.metric("✅ Gagnées", stats.get('correct_predictions', 0))
    with col2:
        total_valide = stats.get('correct_predictions', 0) + stats.get('incorrect_predictions', 0)
        accuracy = (stats.get('correct_predictions', 0) / total_valide * 100) if total_valide > 0 else 0
        st.metric("Précision", f"{accuracy:.1f}%")
        st.metric("Série actuelle", stats.get('current_streak', 0))
    st.markdown("### 🗑️ Gestion des données")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Effacer prédictions"):
            if HIST_FILE.exists():
                HIST_FILE.unlink()
                update_user_stats()
                st.rerun()
    with col2:
        if st.button("🗑️ Effacer combinés"):
            if COMB_HIST_FILE.exists():
                COMB_HIST_FILE.unlink()
                st.rerun()
    with col3:
        if st.button("🔄 Recalculer stats"):
            update_user_stats()
            st.rerun()

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="TennisIQ Pro - Paris & Analytics",
        page_icon="🎾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0A1E2C 0%, #1A2E3C 100%); }
        .stProgress > div > div > div > div { background: linear-gradient(90deg, #00DFA2, #0079FF); }
        .stButton > button { background: linear-gradient(90deg, #00DFA2, #0079FF); color: white; border: none; }
        div[data-testid="stMetricValue"] { font-size: 2rem; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;margin-bottom:2rem;">
            <div style="font-size:2rem;font-weight:800;color:#00DFA2;">TennisIQ</div>
            <div style="font-size:0.8rem;color:#6C7A89;">Paris & Analytics</div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🎯 Analyse Paris", "⏳ En Attente",
             "📜 Historique", "📈 Statistiques", "📱 Telegram", "⚙️ Configuration"],
            label_visibility="collapsed"
        )

        st.divider()
        stats   = load_user_stats()
        pending = len([p for p in load_history() if p.get('statut') == 'en_attente'])
        total_valide = stats.get('correct_predictions', 0) + stats.get('incorrect_predictions', 0)
        accuracy = (stats.get('correct_predictions', 0) / total_valide * 100) if total_valide > 0 else 0
        st.caption(f"📊 Précision: {accuracy:.1f}%")
        st.caption(f"⏳ En attente: {pending}")
        st.caption(f"🔥 Série: {stats.get('current_streak', 0)}")

    if   page == "🏠 Dashboard":     show_dashboard()
    elif page == "🎯 Analyse Paris":  show_prediction()
    elif page == "⏳ En Attente":     show_pending()
    elif page == "📜 Historique":     show_history()
    elif page == "📈 Statistiques":   show_statistics()
    elif page == "📱 Telegram":       show_telegram()
    elif page == "⚙️ Configuration": show_configuration()

if __name__ == "__main__":
    main()
