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

# Base de données des tournois avec leur surface
TOURNAMENTS_DB = {
    "Australian Open": "Hard",
    "Roland Garros": "Clay",
    "Wimbledon": "Grass",
    "US Open": "Hard",
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
    "Dubai Tennis Championships": "Hard",
    "Mexican Open": "Hard",
    "Barcelona Open": "Clay",
    "Halle Open": "Grass",
    "Queen's Club Championships": "Grass",
    "Hamburg Open": "Clay",
    "Washington Open": "Hard",
    "Japan Open": "Hard",
    "Vienna Open": "Hard",
    "Swiss Indoors": "Hard",
    "China Open": "Hard",
    "Nitto ATP Finals": "Hard",
    "Davis Cup": "Hard",
    "Laver Cup": "Hard",
    "Next Gen ATP Finals": "Hard",
    "Adelaide International": "Hard",
    "Auckland Open": "Hard",
    "Montpellier": "Hard",
    "Cordoba Open": "Clay",
    "Dallas Open": "Hard",
    "Buenos Aires": "Clay",
    "Delray Beach": "Hard",
    "Doha": "Hard",
    "Acapulco": "Hard",
    "Santiago": "Clay",
    "Houston": "Clay",
    "Marrakech": "Clay",
    "Estoril": "Clay",
    "Munich": "Clay",
    "Geneva": "Clay",
    "Lyon": "Clay",
    "Stuttgart": "Grass",
    "'s-Hertogenbosch": "Grass",
    "Mallorca": "Grass",
    "Eastbourne": "Grass",
    "Newport": "Grass",
    "Atlanta": "Hard",
    "Kitzbühel": "Clay",
    "Los Cabos": "Hard",
    "Winston-Salem": "Hard",
    "Sofia": "Hard",
    "Metz": "Hard",
    "San Diego": "Hard",
    "Seoul": "Hard",
    "Tel Aviv": "Hard",
    "Florence": "Hard",
    "Gijon": "Hard",
    "Antwerp": "Hard",
    "Stockholm": "Hard",
    "Naples": "Hard",
    "Bratislava": "Hard",
    "Helsinki": "Hard",
}

# Types de paris
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
    "surface_hard": "#0079FF",
    "surface_clay": "#E67E22",
    "surface_grass": "#00DFA2",
}

SURFACE_CONFIG = {
    "Hard": {"color": COLORS["surface_hard"], "icon": "🟦"},
    "Clay": {"color": COLORS["surface_clay"], "icon": "🟧"},
    "Grass": {"color": COLORS["surface_grass"], "icon": "🟩"}
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
        return response.status_code == 200
    except Exception as e:
        print(f"Erreur Telegram: {e}")
        return False

def format_prediction_message(pred_data, bet_suggestions=None, ai_comment=None):
    """Formate un message de prédiction pour Telegram avec suggestions de paris"""
    proba = pred_data.get('proba', 0.5)
    bar_length = 10
    filled = int(proba * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    emoji_map = {'Hard': '🟦', 'Clay': '🟧', 'Grass': '🟩'}
    surface_emoji = emoji_map.get(pred_data.get('surface', ''), '🎾')
    
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
    
    # Ajouter les suggestions de paris
    if bet_suggestions:
        message += f"\n<b>🎯 PARIS ALTERNATIFS:</b>\n"
        for bet in bet_suggestions[:3]:  # Top 3 suggestions
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
    """Formate un message de combiné pour Telegram"""
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
    """Formate un message de statistiques pour Telegram"""
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
    """Appelle l'API Groq pour générer une analyse IA"""
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
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return None
            
    except Exception as e:
        print(f"Exception Groq API: {e}")
        return None

def analyze_match_with_ai(player1, player2, surface, tournament, proba, best_value=None, bet_suggestions=None):
    """Génère une analyse IA complète avec recommandations de paris"""
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
Tournoi: {tournament}
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
    """Charge le modèle ML depuis le dossier models/"""
    model_path = MODELS_DIR / "tennis_ml_model_complete.pkl"
    
    if model_path.exists():
        try:
            model_info = joblib.load(model_path)
            return model_info
        except Exception as e:
            return None
    else:
        # Essayer de télécharger depuis GitHub
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
    """Fait une prédiction avec le modèle ML"""
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
        
        # Features simplifiées
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
        
    except Exception as e:
        return None

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES ATP (VERSION CORRIGÉE)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_atp_data():
    """Charge les données ATP depuis le dossier data/ - VERSION CORRIGÉE avec TOUS les joueurs"""
    if not DATA_DIR.exists():
        st.warning(f"📁 Dossier non trouvé: {DATA_DIR}")
        return pd.DataFrame()
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        st.warning("📁 Aucun fichier CSV trouvé")
        return pd.DataFrame()
    
    st.info(f"📊 Chargement de {len(csv_files)} fichiers...")
    
    atp_dfs = []
    progress_bar = st.progress(0)
    
    for idx, f in enumerate(csv_files):
        if 'wta' in f.name.lower():
            continue
        
        try:
            # Essayer différents encodages
            df = None
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(f, encoding=enc, on_bad_lines='skip', low_memory=False)
                    break
                except:
                    try:
                        df = pd.read_csv(f, sep=';', encoding=enc, on_bad_lines='skip', low_memory=False)
                        break
                    except:
                        continue
            
            if df is not None and 'winner_name' in df.columns and 'loser_name' in df.columns:
                # Nettoyer les noms
                df['winner_name'] = df['winner_name'].astype(str).str.strip()
                df['loser_name'] = df['loser_name'].astype(str).str.strip()
                
                # Garder seulement les colonnes essentielles pour économiser la mémoire
                keep_cols = ['winner_name', 'loser_name', 'surface', 'tourney_name', 'tourney_date']
                df = df[[c for c in keep_cols if c in df.columns]]
                
                atp_dfs.append(df)
        except Exception as e:
            print(f"Erreur avec {f.name}: {e}")
        
        # Mettre à jour la progression
        progress_bar.progress((idx + 1) / len(csv_files))
    
    progress_bar.empty()
    
    if atp_dfs:
        df_combined = pd.concat(atp_dfs, ignore_index=True)
        # Nettoyer les valeurs NaN
        df_combined = df_combined.dropna(subset=['winner_name', 'loser_name'])
        
        st.success(f"✅ {len(df_combined)} matchs chargés avec {len(df_combined['winner_name'].unique())} joueurs uniques")
        return df_combined
    
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_all_players(_df):
    """Récupère la liste de tous les joueurs (avec cache)"""
    if _df.empty:
        return []
    
    players = set()
    if 'winner_name' in _df.columns:
        players.update(_df['winner_name'].dropna().unique())
    if 'loser_name' in _df.columns:
        players.update(_df['loser_name'].dropna().unique())
    
    # Filtrer les valeurs invalides
    players = {str(p).strip() for p in players if pd.notna(p) and str(p).strip() and str(p).strip().lower() != 'nan'}
    
    return sorted(list(players))

def get_player_stats(df, player):
    """Récupère les stats basiques d'un joueur"""
    if df.empty or not player:
        return None
    
    player_clean = player.strip()
    
    matches = df[(df['winner_name'] == player_clean) | (df['loser_name'] == player_clean)]
    if len(matches) == 0:
        return None
    
    wins = len(matches[df['winner_name'] == player_clean])
    total = len(matches)
    
    return {
        'name': player_clean,
        'matches_played': total,
        'wins': wins,
        'losses': total - wins,
        'win_rate': wins / total if total > 0 else 0
    }

def get_h2h_stats(df, player1, player2):
    """Récupère les stats H2H entre deux joueurs"""
    if df.empty or not player1 or not player2:
        return None
    
    p1 = player1.strip()
    p2 = player2.strip()
    
    h2h = df[((df['winner_name'] == p1) & (df['loser_name'] == p2)) | 
             ((df['winner_name'] == p2) & (df['loser_name'] == p1))]
    
    if len(h2h) == 0:
        return None
    
    return {
        'total_matches': len(h2h),
        f'{p1}_wins': len(h2h[h2h['winner_name'] == p1]),
        f'{p2}_wins': len(h2h[h2h['winner_name'] == p2]),
    }

def calculate_probability(df, player1, player2, surface, h2h=None, model_info=None):
    """Calcule la probabilité (ML si dispo, sinon stats)"""
    
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
    """Calcule un score de confiance"""
    confidence = 50
    if h2h and h2h.get('total_matches', 0) >= 3:
        confidence += 10
    confidence += abs(proba - 0.5) * 40
    return min(100, confidence)

# ─────────────────────────────────────────────────────────────
# FONCTIONS POUR LES PARIS ALTERNATIFS
# ─────────────────────────────────────────────────────────────
def generate_alternative_bets(player1, player2, surface, proba, h2h=None):
    """Génère des suggestions de paris alternatifs"""
    suggestions = []
    
    # 1. Over/Under Games (simulé)
    if proba > 0.6 or proba < 0.4:
        # Match déséquilibré -> Under probable
        suggestions.append({
            'type': '📊 Under 22.5 games',
            'description': f"Moins de 22.5 jeux dans le match",
            'proba': 0.65 if abs(proba - 0.5) > 0.2 else 0.55,
            'cote': 1.75,
            'confidence': 70 if abs(proba - 0.5) > 0.25 else 60,
            'edge': 0.03
        })
    else:
        # Match serré -> Over probable
        suggestions.append({
            'type': '📊 Over 22.5 games',
            'description': f"Plus de 22.5 jeux dans le match",
            'proba': 0.62,
            'cote': 1.80,
            'confidence': 65,
            'edge': 0.02
        })
    
    # 2. Handicap
    if proba > 0.65:
        # Favori fort
        suggestions.append({
            'type': '⚖️ Handicap -3.5',
            'description': f"{player1} gagne avec au moins 4 jeux d'écart",
            'proba': 0.58,
            'cote': 2.10,
            'confidence': 60,
            'edge': 0.04
        })
    elif proba < 0.35:
        suggestions.append({
            'type': '⚖️ Handicap +3.5',
            'description': f"{player2} perd par moins de 4 jeux ou gagne",
            'proba': 0.62,
            'cote': 1.95,
            'confidence': 65,
            'edge': 0.03
        })
    
    # 3. Les deux gagnent un set
    if 0.3 < proba < 0.7:
        suggestions.append({
            'type': '🔄 Les deux gagnent un set',
            'description': f"Chaque joueur remporte au moins un set",
            'proba': 0.55,
            'cote': 2.20,
            'confidence': 55,
            'edge': 0.01
        })
    
    # 4. Score exact (simplifié)
    if proba > 0.7:
        suggestions.append({
            'type': '🎯 Score 2-0',
            'description': f"{player1} gagne 2-0",
            'proba': 0.52,
            'cote': 2.50,
            'confidence': 50,
            'edge': 0.02
        })
    elif proba < 0.3:
        suggestions.append({
            'type': '🎯 Score 0-2',
            'description': f"{player2} gagne 2-0",
            'proba': 0.51,
            'cote': 2.60,
            'confidence': 50,
            'edge': 0.01
        })
    
    # Ajouter des infos H2H si disponibles
    if h2h and h2h.get('total_matches', 0) >= 3:
        for bet in suggestions:
            bet['confidence'] = min(95, bet['confidence'] + 5)
    
    return suggestions

# ─────────────────────────────────────────────────────────────
# GESTION DE L'HISTORIQUE ET DES STATISTIQUES
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
    """Sauvegarde une prédiction avec statut 'en_attente'"""
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
    """Met à jour le statut d'une prédiction et recalcule les stats"""
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
        return {
            'total_predictions': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'annules_predictions': 0,
            'total_combines': 0,
            'won_combines': 0,
            'total_invested': 0,
            'total_won': 0,
            'best_streak': 0,
            'current_streak': 0,
        }
    try:
        with open(USER_STATS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def update_user_stats():
    """Calcule les statistiques à partir de l'historique"""
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
    
    stats = {
        'total_predictions': total,
        'correct_predictions': correct,
        'incorrect_predictions': incorrect,
        'annules_predictions': annules,
        'total_combines': 0,
        'won_combines': 0,
        'total_invested': 0,
        'total_won': 0,
        'current_streak': current_streak,
        'best_streak': best_streak,
    }
    
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
# FONCTIONS POUR LES COMBINÉS RECOMMANDÉS
# ─────────────────────────────────────────────────────────────
def generate_recommended_combines(matches_analysis):
    """Génère des combinés recommandés à partir des matchs analysés"""
    if len(matches_analysis) < 2:
        return []
    
    # Trier par edge (value bet)
    matches_with_edge = [m for m in matches_analysis if m.get('best_value')]
    matches_with_edge.sort(key=lambda x: x['best_value']['edge'], reverse=True)
    
    suggestions = []
    
    # Suggestion 1: Top edges (max 3 matchs)
    if len(matches_with_edge) >= 2:
        top_edges = matches_with_edge[:min(3, len(matches_with_edge))]
        selections = [{
            'match': f"{m['player1']} vs {m['player2']}",
            'joueur': m['best_value']['joueur'],
            'proba': m['best_value']['proba'],
            'cote': m['best_value']['cote'],
            'edge': m['best_value']['edge']
        } for m in top_edges]
        
        proba_combi = np.prod([s['proba'] for s in selections])
        cote_combi = np.prod([s['cote'] for s in selections])
        
        suggestions.append({
            'name': '🔥 Top Value Bets',
            'selections': selections,
            'proba': proba_combi,
            'cote': cote_combi,
            'nb_matches': len(selections)
        })
    
    # Suggestion 2: Matchs avec haute confiance
    high_confidence = [m for m in matches_analysis if m.get('confidence', 0) >= 70]
    if len(high_confidence) >= 2:
        top_confidence = high_confidence[:min(3, len(high_confidence))]
        selections = [{
            'match': f"{m['player1']} vs {m['player2']}",
            'joueur': m['favori'],
            'proba': m['proba'] if m['proba'] >= 0.5 else 1-m['proba'],
            'cote': 1/m['proba'] if m['proba'] >= 0.5 else 1/(1-m['proba']),
            'edge': 0.05
        } for m in top_confidence]
        
        proba_combi = np.prod([s['proba'] for s in selections])
        cote_combi = np.prod([s['cote'] for s in selections])
        
        suggestions.append({
            'name': '💪 Haute Confiance',
            'selections': selections,
            'proba': proba_combi,
            'cote': cote_combi,
            'nb_matches': len(selections)
        })
    
    # Suggestion 3: Combiné équilibré (mix)
    if len(matches_analysis) >= 3:
        value_bets = matches_with_edge[:2] if len(matches_with_edge) >= 2 else []
        favorites = [m for m in matches_analysis if m.get('confidence', 0) >= 60 and m not in value_bets]
        
        selections = []
        for vb in value_bets[:2]:
            selections.append({
                'match': f"{vb['player1']} vs {vb['player2']}",
                'joueur': vb['best_value']['joueur'],
                'proba': vb['best_value']['proba'],
                'cote': vb['best_value']['cote'],
                'edge': vb['best_value']['edge']
            })
        
        if favorites and len(selections) < 3:
            fav = favorites[0]
            selections.append({
                'match': f"{fav['player1']} vs {fav['player2']}",
                'joueur': fav['favori'],
                'proba': fav['proba'] if fav['proba'] >= 0.5 else 1-fav['proba'],
                'cote': 1/fav['proba'] if fav['proba'] >= 0.5 else 1/(1-fav['proba']),
                'edge': 0.03
            })
        
        if len(selections) >= 2:
            proba_combi = np.prod([s['proba'] for s in selections])
            cote_combi = np.prod([s['cote'] for s in selections])
            
            suggestions.append({
                'name': '⚖️ Combiné Équilibré',
                'selections': selections,
                'proba': proba_combi,
                'cote': cote_combi,
                'nb_matches': len(selections)
            })
    
    return suggestions[:MAX_COMBINE_SUGGESTIONS]

# ─────────────────────────────────────────────────────────────
# PAGES DE L'APPLICATION
# ─────────────────────────────────────────────────────────────

def show_dashboard():
    """Page Dashboard"""
    st.markdown("<h2>🏠 Dashboard</h2>", unsafe_allow_html=True)
    
    stats = load_user_stats()
    history = load_history()
    
    total = stats.get('total_predictions', 0)
    correct = stats.get('correct_predictions', 0)
    incorrect = stats.get('incorrect_predictions', 0)
    annules = stats.get('annules_predictions', 0)
    pending = len([p for p in history if p.get('statut') == 'en_attente'])
    
    total_valide = correct + incorrect
    accuracy = (correct / total_valide * 100) if total_valide > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", total)
    with col2:
        st.metric("✅ Gagnées", correct)
    with col3:
        st.metric("❌ Perdues", incorrect)
    with col4:
        st.metric("⏳ En attente", pending)
    
    st.markdown("### 🎯 Précision globale")
    st.progress(accuracy / 100)
    st.caption(f"{accuracy:.1f}% de réussite sur {total_valide} matchs résolus")
    
    # Stats rapides des services
    model_info = load_saved_model()
    groq_key = get_groq_key()
    telegram_token, _ = get_telegram_config()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if model_info:
            st.success(f"✅ Modèle ML ({model_info.get('accuracy', 0):.1%})")
        else:
            st.warning("⚠️ Modèle ML non chargé")
    with col2:
        st.success("✅ IA Groq" if groq_key else "⚠️ IA non configurée")
    with col3:
        st.success("✅ Telegram" if telegram_token else "⚠️ Telegram non configuré")

def show_prediction():
    """Page de prédiction avec menus déroulants et surface automatique"""
    st.markdown("<h2>🎯 Analyse Paris avec Menus Déroulants</h2>", unsafe_allow_html=True)
    
    model_info = load_saved_model()
    
    # Charger les données avec barre de progression
    with st.spinner("📊 Chargement de la base de données des joueurs..."):
        atp_data = load_atp_data()
    
    if atp_data.empty:
        st.error("❌ Impossible de charger les données des joueurs")
        st.info("Vérifie que le dossier `src/data/raw/tml-tennis` contient des fichiers CSV")
        return
    
    # Récupérer tous les joueurs
    all_players = get_all_players(atp_data)
    
    if not all_players:
        st.error("❌ Aucun joueur trouvé dans les données")
        return
    
    st.success(f"✅ {len(all_players)} joueurs disponibles dans la base")
    
    # Configuration
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_matches = st.number_input("Nombre de matchs", 1, MAX_MATCHES_ANALYSIS, 3)
    with col2:
        mise = st.number_input("Mise (€)", 1.0, 1000.0, 10.0)
    with col3:
        use_ai = st.checkbox("🤖 Analyser avec IA", True)
    with col4:
        send_tg = st.checkbox("📱 Envoyer sur Telegram", False)
    
    # Saisie des matchs avec menus déroulants
    matches = []
    st.markdown("### 📝 Sélection des matchs")
    
    # Liste des tournois triée
    tournaments_list = sorted(TOURNAMENTS_DB.keys())
    
    for i in range(n_matches):
        with st.expander(f"Match {i+1}", expanded=i==0):
            col1, col2 = st.columns(2)
            
            with col1:
                # Menu déroulant pour joueur 1
                p1 = st.selectbox(
                    f"Joueur 1", 
                    options=all_players,
                    index=0,
                    key=f"p1_{i}"
                )
                odds1 = st.text_input(f"Cote {p1}", key=f"odds1_{i}", placeholder="1.75")
            
            with col2:
                # Menu déroulant pour joueur 2 (exclure joueur 1)
                players2 = [p for p in all_players if p != p1]
                if players2:
                    p2 = st.selectbox(
                        f"Joueur 2", 
                        options=players2,
                        index=0,
                        key=f"p2_{i}"
                    )
                else:
                    p2 = st.text_input(f"Joueur 2", key=f"p2_{i}", placeholder="Carlos Alcaraz")
                
                odds2 = st.text_input(f"Cote {p2 if p2 else 'J2'}", key=f"odds2_{i}", placeholder="2.10")
            
            # Menu déroulant pour tournoi avec surface automatique
            col1, col2 = st.columns(2)
            with col1:
                tournament = st.selectbox(
                    f"Tournoi",
                    options=tournaments_list,
                    index=0,
                    key=f"tourn_{i}"
                )
                # Surface automatique basée sur le tournoi
                surface = TOURNAMENTS_DB.get(tournament, "Hard")
            
            with col2:
                # Afficher la surface avec l'icône
                surface_icon = SURFACE_CONFIG[surface]["icon"]
                st.markdown(f"### {surface_icon} Surface: **{surface}**")
            
            matches.append({
                'player1': p1.strip() if p1 else "",
                'player2': p2.strip() if p2 else "",
                'surface': surface,
                'tournament': tournament,
                'odds1': odds1,
                'odds2': odds2,
                'index': i
            })
    
    # Bouton d'analyse
    if st.button("🔍 Analyser tous les matchs", type="primary", use_container_width=True):
        valid_matches = [m for m in matches if m['player1'] and m['player2']]
        
        if not valid_matches:
            st.warning("Veuillez sélectionner au moins un match complet")
            return
        
        st.markdown("---")
        st.markdown("## 📊 Résultats de l'analyse complète")
        
        matches_analysis = []
        all_selections = []
        
        # Analyser chaque match
        for i, match in enumerate(valid_matches):
            st.markdown(f"### Match {i+1}: {match['player1']} vs {match['player2']}")
            st.caption(f"🏆 {match['tournament']} - {SURFACE_CONFIG[match['surface']]['icon']} {match['surface']}")
            
            h2h = get_h2h_stats(atp_data, match['player1'], match['player2'])
            proba, ml_used = calculate_probability(atp_data, match['player1'], match['player2'], 
                                                   match['surface'], h2h, model_info)
            confidence = calculate_confidence(proba, h2h)
            
            # Calculer value bet
            best_value = None
            if match['odds1'] and match['odds2']:
                try:
                    o1 = float(match['odds1'].replace(',', '.'))
                    o2 = float(match['odds2'].replace(',', '.'))
                    edge1 = proba - 1/o1
                    edge2 = (1-proba) - 1/o2
                    
                    if edge1 > edge2 and edge1 > MIN_EDGE_COMBINE:
                        best_value = {
                            'joueur': match['player1'],
                            'edge': edge1,
                            'cote': o1,
                            'proba': proba
                        }
                    elif edge2 > edge1 and edge2 > MIN_EDGE_COMBINE:
                        best_value = {
                            'joueur': match['player2'],
                            'edge': edge2,
                            'cote': o2,
                            'proba': 1-proba
                        }
                except:
                    pass
            
            # Générer des paris alternatifs
            bet_suggestions = generate_alternative_bets(match['player1'], match['player2'], 
                                                        match['surface'], proba, h2h)
            
            # Afficher le gagnant prédit
            st.markdown(f"#### 🏆 GAGNANT PRÉDIT: **{match['player1'] if proba >= 0.5 else match['player2']}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{match['player1']}", f"{proba:.1%}")
            with col2:
                st.metric(f"{match['player2']}", f"{1-proba:.1%}")
            
            st.progress(float(proba))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"{'🤖 ML' if ml_used else '📊 Stats'} utilisé")
            with col2:
                conf_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🔴"
                st.caption(f"Confiance: {conf_color} {confidence:.0f}/100")
            with col3:
                if h2h:
                    wins1 = h2h.get(f"{match['player1']}_wins", 0)
                    wins2 = h2h.get(f"{match['player2']}_wins", 0)
                    st.caption(f"H2H: {wins1}-{wins2}")
            
            if best_value:
                st.success(f"🎯 Value bet! {best_value['joueur']} @ {best_value['cote']:.2f} (edge: {best_value['edge']*100:+.1f}%)")
                # Ajouter aux sélections pour combiné
                all_selections.append({
                    'match': f"{match['player1']} vs {match['player2']}",
                    'joueur': best_value['joueur'],
                    'proba': best_value['proba'],
                    'cote': best_value['cote'],
                    'edge': best_value['edge']
                })
            
            # Afficher les paris alternatifs
            if bet_suggestions:
                st.markdown("#### 🎯 Paris Alternatifs Recommandés")
                for bet in bet_suggestions:
                    conf_icon = '🟢' if bet['confidence'] >= 70 else '🟡' if bet['confidence'] >= 50 else '🔴'
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        with col1:
                            st.markdown(f"{conf_icon} **{bet['type']}**")
                            st.caption(bet['description'])
                        with col2:
                            st.metric("Probabilité", f"{bet['proba']:.1%}")
                        with col3:
                            st.metric("Cote", f"{bet['cote']:.2f}")
                        with col4:
                            edge_pct = bet.get('edge', 0) * 100
                            st.metric("Edge", f"{edge_pct:+.1f}%")
            
            # Analyse IA
            ai_comment = None
            if use_ai and get_groq_key():
                with st.spinner(f"🤖 Analyse IA complète..."):
                    ai_comment = analyze_match_with_ai(match['player1'], match['player2'], 
                                                      match['surface'], match['tournament'],
                                                      proba, best_value, bet_suggestions)
                    if ai_comment:
                        with st.expander("Voir analyse IA complète"):
                            st.markdown(ai_comment)
            
            # Préparer données pour sauvegarde et envoi
            pred_data = {
                'player1': match['player1'],
                'player2': match['player2'],
                'tournament': match['tournament'],
                'surface': match['surface'],
                'proba': float(proba),
                'confidence': float(confidence),
                'odds1': match['odds1'] if match['odds1'] else None,
                'odds2': match['odds2'] if match['odds2'] else None,
                'favori': match['player1'] if proba >= 0.5 else match['player2'],
                'best_value': best_value,
                'bet_suggestions': bet_suggestions,
                'ml_used': ml_used
            }
            
            matches_analysis.append(pred_data)
            
            # Boutons pour chaque match
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"💾 Sauvegarder match {i+1}", key=f"save_{i}"):
                    if save_prediction(pred_data):
                        st.success("✅ Sauvegardé en attente!")
            
            with col2:
                if st.button(f"📱 Envoyer match {i+1} sur Telegram", key=f"tg_{i}"):
                    if send_prediction_to_telegram(pred_data, bet_suggestions, ai_comment):
                        st.success("✅ Envoyé sur Telegram!")
            
            st.divider()
        
        # Générer des combinés recommandés
        if len(all_selections) >= 2:
            st.markdown("## 🎰 Combinés recommandés")
            
            suggestions = generate_recommended_combines(matches_analysis)
            
            for idx, suggestion in enumerate(suggestions):
                with st.expander(f"{suggestion['name']} - {suggestion['nb_matches']} matchs - Proba {suggestion['proba']:.1%}", expanded=idx==0):
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Probabilité", f"{suggestion['proba']:.1%}")
                    with col2:
                        st.metric("Cote", f"{suggestion['cote']:.2f}")
                    with col3:
                        gain = mise * suggestion['cote']
                        st.metric("Gain potentiel", f"{gain:.2f}€")
                    
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
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"💾 Sauvegarder ce combiné", key=f"save_comb_{idx}"):
                            save_combine(combine_data)
                            st.success("✅ Combiné sauvegardé!")
                    
                    with col2:
                        if st.button(f"📱 Envoyer sur Telegram", key=f"tg_comb_{idx}"):
                            if send_combine_to_telegram(combine_data):
                                st.success("✅ Combiné envoyé!")
        
        # Envoi groupé si demandé
        if send_tg and matches_analysis:
            st.markdown("### 📤 Envoi groupé sur Telegram")
            if st.button("📤 Envoyer tous les matchs sur Telegram", use_container_width=True):
                success_count = 0
                for pred in matches_analysis:
                    if send_prediction_to_telegram(pred, pred.get('bet_suggestions')):
                        success_count += 1
                st.success(f"✅ {success_count}/{len(matches_analysis)} matchs envoyés sur Telegram!")

# ... (les fonctions show_pending, show_history, show_statistics, show_telegram, show_configuration, main restent identiques)
