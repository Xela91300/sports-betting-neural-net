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
import plotly.express as px
import plotly.graph_objects as go
import shutil
import hashlib
import random

# Pour le scraping réel
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    st.warning("⚠️ BeautifulSoup non installé. Le scraping ne fonctionnera pas. Installe avec: pip install beautifulsoup4")

nest_asyncio.apply()
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION DES CHEMINS
# ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "src" / "data" / "raw" / "tml-tennis"
HIST_DIR = ROOT_DIR / "history"
BACKUP_DIR = ROOT_DIR / "backups"

for dir_path in [MODELS_DIR, DATA_DIR, HIST_DIR, BACKUP_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

HIST_FILE = HIST_DIR / "predictions_history.json"
COMB_HIST_FILE = HIST_DIR / "combines_history.json"
USER_STATS_FILE = HIST_DIR / "user_stats.json"
ACHIEVEMENTS_FILE = HIST_DIR / "achievements.json"
TRENDS_FILE = HIST_DIR / "trends.json"

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

# Configuration des badges/achievements
ACHIEVEMENTS = {
    'first_win': {'name': '🎯 Première victoire', 'desc': 'Première prédiction gagnante', 'icon': '🎯'},
    'streak_5': {'name': '🔥 En forme', 'desc': '5 prédictions gagnantes consécutives', 'icon': '🔥'},
    'streak_10': {'name': '⚡ Imbattable', 'desc': '10 prédictions gagnantes consécutives', 'icon': '⚡'},
    'pred_100': {'name': '🏆 Expert', 'desc': '100 prédictions', 'icon': '🏆'},
    'value_master': {'name': '💎 Value Master', 'desc': '10 value bets gagnants', 'icon': '💎'},
    'surface_specialist': {'name': '🌍 Spécialiste surface', 'desc': 'Gagnant sur les 3 surfaces', 'icon': '🌍'},
}

# ─────────────────────────────────────────────────────────────
# BASE DE DONNÉES COMPLÈTE DES TOURNOIS (250+ tournois avec alias)
# ─────────────────────────────────────────────────────────────
TOURNAMENTS_DB = {
    # GRAND CHELEM
    "Australian Open": "Hard",
    "Roland Garros": "Clay",
    "Wimbledon": "Grass",
    "US Open": "Hard",
    
    # ATP FINALS
    "Nitto ATP Finals": "Hard",
    "Next Gen ATP Finals": "Hard",
    
    # ATP MASTERS 1000
    "Indian Wells Masters": "Hard",
    "Miami Open": "Hard",
    "Monte-Carlo Masters": "Clay",
    "Madrid Open": "Clay",
    "Italian Open": "Clay",
    "Canadian Open": "Hard",
    "Cincinnati Masters": "Hard",
    "Shanghai Masters": "Hard",
    "Paris Masters": "Hard",
    
    # ATP 500
    "Rotterdam Open": "Hard",
    "Rio Open": "Clay",
    "Dubai Tennis Championships": "Hard",
    "Mexican Open": "Hard",
    "Barcelona Open": "Clay",
    "Halle Open": "Grass",
    "Queen's Club Championships": "Grass",
    "Hamburg Open": "Clay",
    "Washington Open": "Hard",
    "China Open": "Hard",
    "Japan Open": "Hard",
    "Vienna Open": "Hard",
    "Swiss Indoors": "Hard",
    "Dallas Open": "Hard",
    "Qatar Open": "Hard",
    "St. Petersburg Open": "Hard",
    "Astana Open": "Hard",
    "European Open": "Hard",
    "Croatia Open": "Clay",
    "Swedish Open": "Clay",
    "Swiss Open": "Clay",
    "Moselle Open": "Hard",
    "Tel Aviv Open": "Hard",
    "Korea Open": "Hard",
    "Zhuhai Open": "Hard",
    "Chengdu Open": "Hard",
    
    # ATP 250 - JANVIER
    "Adelaide International": "Hard",
    "Auckland Open": "Hard",
    "Brisbane International": "Hard",
    "Hong Kong Open": "Hard",
    "Maharashtra Open": "Hard",
    
    # ATP 250 - FÉVRIER
    "Cordoba Open": "Clay",
    "Buenos Aires": "Clay",
    "Delray Beach": "Hard",
    "Marseille Open": "Hard",
    "Chile Open": "Clay",
    "Qatar ExxonMobil Open": "Hard",
    
    # ATP 250 - MARS/AVRIL
    "Santiago": "Clay",
    "Houston": "Clay",
    "Marrakech": "Clay",
    "Estoril": "Clay",
    "Munich": "Clay",
    "Geneva": "Clay",
    "Lyon": "Clay",
    "Grand Prix Hassan II": "Clay",
    "U.S. Men's Clay Court Championship": "Clay",
    "BMW Open": "Clay",
    "Millennium Estoril Open": "Clay",
    "Srpska Open": "Clay",
    "Banja Luka": "Clay",
    
    # ATP 250 - JUIN
    "Stuttgart": "Grass",
    "'s-Hertogenbosch": "Grass",
    "Mallorca": "Grass",
    "Eastbourne": "Grass",
    "BOSS Open": "Grass",
    "Libéma Open": "Grass",
    "Rothesay International": "Grass",
    "MercedesCup": "Grass",
    "Fever-Tree Championships": "Grass",
    "Gerry Weber Open": "Grass",
    "Nature Valley International": "Grass",
    
    # ATP 250 - JUILLET
    "Newport": "Grass",
    "Atlanta": "Hard",
    "Croatia Open Umag": "Clay",
    "Nordea Open": "Clay",
    "EFO Swiss Open Gstaad": "Clay",
    "Plava Laguna Croatia Open": "Clay",
    "Hall of Fame Open": "Grass",
    "Atlanta Open": "Hard",
    
    # ATP 250 - AOÛT
    "Kitzbühel": "Clay",
    "Los Cabos": "Hard",
    "Winston-Salem": "Hard",
    "Winston-Salem Open": "Hard",
    "Western & Southern Open": "Hard",
    
    # ATP 250 - SEPTEMBRE
    "Chengdu Open": "Hard",
    "Zhuhai Championships": "Hard",
    "Sofia": "Hard",
    "Metz": "Hard",
    "San Diego": "Hard",
    "Seoul": "Hard",
    "Tel Aviv": "Hard",
    "Kremlin Cup": "Hard",
    
    # ATP 250 - OCTOBRE
    "Florence": "Hard",
    "Gijon": "Hard",
    "Antwerp": "Hard",
    "Stockholm": "Hard",
    "Naples": "Hard",
    "Bratislava": "Hard",
    "Helsinki": "Hard",
    "Stockholm Open": "Hard",
    "European Open": "Hard",
    "Erste Bank Open": "Hard",
    "Swiss Indoors Basel": "Hard",
    
    # ATP 250 - NOVEMBRE
    "Moselle Open": "Hard",
    "Belgrade Open": "Hard",
    "Paris Rolex Paris Masters": "Hard",
    "Next Gen ATP Finals": "Hard",
    
    # DAVIS CUP & LA CUP
    "Davis Cup": "Hard",
    "Laver Cup": "Hard",
    "United Cup": "Hard",
    
    # AUTRES TOURNOIS HISTORIQUES
    "ASB Classic": "Hard",
    "Movistar Chile Open": "Clay",
    "Argentina Open": "Clay",
    "Ecuador Open": "Clay",
    "Hungarian Open": "Clay",
    "Portugal Open": "Clay",
    "Pacific Coast Championships": "Hard",
    "U.S. Pro Indoor": "Hard",
    "Richmond WCT": "Hard",
    "Omaha": "Hard",
    "Hampton": "Hard",
    "Birmingham WCT": "Hard",
    "Cologne WCT": "Hard",
    "Chicago": "Hard",
    "Los Angeles": "Hard",
    "Fairfield": "Hard",
    "Johannesburg WCT": "Hard",
    "Las Vegas": "Hard",
    "St. Louis WCT": "Hard",
    "Denver": "Hard",
    "Miami WCT": "Hard",
    "Fort Worth": "Hard",
    "Montreal WCT": "Hard",
    "Brussels": "Clay",
    "Båstad": "Clay",
    "Düsseldorf": "Clay",
    "Hamburg": "Clay",
    "Stuttgart Outdoor": "Clay",
    "Gstaad": "Clay",
    "Hilversum": "Clay",
    "Bournemouth": "Clay",
    "Berlin": "Hard",
    "Cologne": "Hard",
    "Copenhagen": "Hard",
    "Oslo": "Hard",
    "Brescia": "Hard",
    "Dortmund": "Hard",
    "Munich WCT": "Hard",
    "Brussels WCT": "Hard",
    "Rotterdam WCT": "Hard",
    "Salisbury": "Hard",
    "Toronto WCT": "Hard",
    "Johannesburg": "Hard",
    "Washington WCT": "Hard",
    "Houston WCT": "Clay",
    "River Oaks": "Clay",
    "Charlotte WCT": "Clay",
    "St. Louis": "Clay",
    "New Orleans": "Clay",
    "Indianapolis": "Clay",
    "Cincinnati": "Hard",
    "Louisville": "Clay",
    "Columbus": "Hard",
    "Montreal": "Hard",
    "Boston": "Hard",
    "Merion": "Hard",
    "South Orange": "Hard",
    "Orange": "Hard",
    "Newport RI": "Grass",
    "Long Island": "Hard",
    "Sydney": "Hard",
    "Melbourne": "Hard",
    "Perth": "Hard",
    "Adelaide": "Hard",
    "Wellington": "Hard",
    "Christchurch": "Hard",
    "Manila": "Hard",
    "Osaka": "Hard",
    "Tokyo Outdoor": "Hard",
    "Taipei": "Hard",
    "Kuala Lumpur": "Hard",
    "Jakarta": "Hard",
    "Calcutta": "Grass",
    "Bombay": "Grass",
    "Lagos": "Hard",
    "Cairo": "Clay",
    "Casablanca": "Clay",
    "Tunis": "Clay",
    "Nairobi": "Grass",
    "London": "Grass",
    "Manchester": "Grass",
    "Nottingham": "Grass",
    "Bristol": "Grass",
    "Leeds": "Grass",
    "Edinburgh": "Grass",
    "Dublin": "Grass",
    "Cork": "Grass",
    "Belfast": "Grass",
    "Paris Indoor": "Hard",
    "Nice": "Clay",
    "Cannes": "Clay",
    "Aix-en-Provence": "Clay",
    "Bordeaux": "Clay",
    "Toulouse": "Hard",
    "Montpellier": "Hard",
    "Strasbourg": "Clay",
    "Brest": "Hard",
    "Cherbourg": "Hard",
    "Rennes": "Hard",
    "Nantes": "Hard",
    "Lille": "Hard",
    "Grenoble": "Hard",
    "Clermont-Ferrand": "Hard",
    "Biarritz": "Clay",
    "Saint-Raphaël": "Hard",
    "Antibes": "Clay",
    "Sophia Antipolis": "Hard",
    "Mouilleron-le-Captif": "Hard",
    "Roanne": "Hard",
    "Orleans": "Hard",
    "Quimper": "Hard",
    "Bressuire": "Hard",
}

# Dictionnaire d'alias pour les tournois (pour trouver par noms alternatifs)
TOURNAMENT_ALIASES = {
    # Acapulco / Mexique
    "acapulco": "Mexican Open",
    "mexico": "Mexican Open",
    "mexican": "Mexican Open",
    "abierto mexicano": "Mexican Open",
    
    # Grand Chelem
    "australian": "Australian Open",
    "melbourne": "Australian Open",
    "roland garros": "Roland Garros",
    "french": "Roland Garros",
    "wimbledon": "Wimbledon",
    "wimby": "Wimbledon",
    "london": "Wimbledon",
    "us open": "US Open",
    "new york": "US Open",
    "flushing": "US Open",
    
    # Masters 1000
    "indian wells": "Indian Wells Masters",
    "california": "Indian Wells Masters",
    "miami": "Miami Open",
    "florida": "Miami Open",
    "monte carlo": "Monte-Carlo Masters",
    "monaco": "Monte-Carlo Masters",
    "madrid": "Madrid Open",
    "spain": "Madrid Open",
    "rome": "Italian Open",
    "italy": "Italian Open",
    "canada": "Canadian Open",
    "montreal": "Canadian Open",
    "toronto": "Canadian Open",
    "cincinnati": "Cincinnati Masters",
    "ohio": "Cincinnati Masters",
    "shanghai": "Shanghai Masters",
    "china": "Shanghai Masters",
    "paris masters": "Paris Masters",
    "bercy": "Paris Masters",
    
    # ATP 500
    "rotterdam": "Rotterdam Open",
    "netherlands": "Rotterdam Open",
    "dubai": "Dubai Tennis Championships",
    "uae": "Dubai Tennis Championships",
    "barcelona": "Barcelona Open",
    "catalunya": "Barcelona Open",
    "halle": "Halle Open",
    "germany": "Halle Open",
    "queens": "Queen's Club Championships",
    "london grass": "Queen's Club Championships",
    "hamburg": "Hamburg Open",
    "washington": "Washington Open",
    "dc": "Washington Open",
    "beijing": "China Open",
    "tokyo": "Japan Open",
    "vienna": "Vienna Open",
    "austria": "Vienna Open",
    "basel": "Swiss Indoors",
    "bâle": "Swiss Indoors",
    "switzerland": "Swiss Indoors",
    "dallas": "Dallas Open",
    "texas": "Dallas Open",
    "rio": "Rio Open",
    "brazil": "Rio Open",
    
    # ATP 250
    "adelaide": "Adelaide International",
    "auckland": "Auckland Open",
    "brisbane": "Brisbane International",
    "hong kong": "Hong Kong Open",
    "doha": "Qatar Open",
    "marseille": "Marseille Open",
    "delray": "Delray Beach",
    "buenos aires": "Buenos Aires",
    "santiago": "Santiago",
    "houston": "Houston",
    "marrakech": "Marrakech",
    "estoril": "Estoril",
    "munich": "Munich",
    "geneva": "Geneva",
    "lyon": "Lyon",
    "stuttgart": "Stuttgart",
    "s-hertogenbosch": "'s-Hertogenbosch",
    "mallorca": "Mallorca",
    "eastbourne": "Eastbourne",
    "newport": "Newport",
    "atlanta": "Atlanta",
    "umag": "Croatia Open Umag",
    "båstad": "Swedish Open",
    "gstaad": "EFO Swiss Open Gstaad",
    "kitzbühel": "Kitzbühel",
    "los cabos": "Los Cabos",
    "winston-salem": "Winston-Salem",
    "chengdu": "Chengdu Open",
    "zhuhai": "Zhuhai Championships",
    "sofia": "Sofia",
    "metz": "Metz",
    "san diego": "San Diego",
    "seoul": "Seoul",
    "tel aviv": "Tel Aviv",
    "florence": "Florence",
    "gijon": "Gijon",
    "antwerp": "Antwerp",
    "stockholm": "Stockholm",
    "naples": "Naples",
    "bratislava": "Bratislava",
    "helsinki": "Helsinki",
    "belgrade": "Belgrade Open",
}

# Fonctions utilitaires pour les tournois
def get_tournament_surface(tournament_name):
    """Récupère la surface d'un tournoi avec fallback"""
    return TOURNAMENTS_DB.get(tournament_name, "Hard")

def find_tournament(search_term):
    """Trouve un tournoi par son nom ou alias - version améliorée"""
    if not search_term:
        return None
    
    search_lower = search_term.lower().strip()
    
    # 1. Chercher d'abord dans les alias
    if search_lower in TOURNAMENT_ALIASES:
        return TOURNAMENT_ALIASES[search_lower]
    
    # 2. Chercher dans les noms exacts (case insensitive)
    for tourney in TOURNAMENTS_DB:
        if search_lower == tourney.lower():
            return tourney
    
    # 3. Chercher les correspondances partielles
    matches = []
    for tourney in TOURNAMENTS_DB:
        if search_lower in tourney.lower():
            matches.append(tourney)
    
    if matches:
        # Retourner le match le plus court (souvent le meilleur)
        return min(matches, key=len)
    
    # 4. Chercher dans les valeurs des alias (si quelqu'un tape le nom officiel)
    for alias, official in TOURNAMENT_ALIASES.items():
        if search_lower == official.lower():
            return official
    
    return None  # Aucun tournoi trouvé

def search_tournaments(search_term, limit=100):
    """Recherche des tournois par nom avec limite"""
    if not search_term:
        return sorted(TOURNAMENTS_DB.keys())[:limit]
    
    search_lower = search_term.lower().strip()
    results = set()
    
    # Chercher dans les noms officiels
    for tourney in TOURNAMENTS_DB:
        if search_lower in tourney.lower():
            results.add(tourney)
    
    # Chercher dans les alias
    for alias, official in TOURNAMENT_ALIASES.items():
        if search_lower in alias:
            results.add(official)
    
    # Si aucun résultat, retourner quelques suggestions
    if not results:
        suggestions = []
        for tourney in TOURNAMENTS_DB:
            if tourney[0].lower() == search_lower[0] if search_lower else True:
                suggestions.append(tourney)
        return sorted(suggestions)[:limit]
    
    return sorted(results)[:limit]

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
# FONCTIONS DE BACKUP AUTOMATIQUE
# ─────────────────────────────────────────────────────────────
def auto_backup():
    """Backup automatique des données"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for file in [HIST_FILE, COMB_HIST_FILE, USER_STATS_FILE, ACHIEVEMENTS_FILE, TRENDS_FILE]:
        if file.exists():
            try:
                shutil.copy(file, BACKUP_DIR / f"{file.stem}_{timestamp}{file.suffix}")
            except:
                pass
    
    # Nettoyer les vieux backups (>30 jours)
    try:
        for backup in BACKUP_DIR.glob("*"):
            if (datetime.now() - datetime.fromtimestamp(backup.stat().st_mtime)).days > 30:
                backup.unlink()
    except:
        pass

# ─────────────────────────────────────────────────────────────
# SYSTÈME DE BADGES ET ACHIEVEMENTS
# ─────────────────────────────────────────────────────────────
def load_achievements():
    if not ACHIEVEMENTS_FILE.exists():
        return {}
    try:
        with open(ACHIEVEMENTS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_achievements(achievements):
    try:
        with open(ACHIEVEMENTS_FILE, 'w') as f:
            json.dump(achievements, f)
    except:
        pass

def check_and_unlock_achievements():
    """Vérifie et débloque les achievements"""
    stats = load_user_stats()
    history = load_history()
    achievements = load_achievements()
    new_unlocks = []
    
    # Première victoire
    if stats.get('correct_predictions', 0) >= 1 and 'first_win' not in achievements:
        achievements['first_win'] = {'unlocked_at': datetime.now().isoformat()}
        new_unlocks.append(ACHIEVEMENTS['first_win'])
    
    # Série de 5
    if stats.get('best_streak', 0) >= 5 and 'streak_5' not in achievements:
        achievements['streak_5'] = {'unlocked_at': datetime.now().isoformat()}
        new_unlocks.append(ACHIEVEMENTS['streak_5'])
    
    # Série de 10
    if stats.get('best_streak', 0) >= 10 and 'streak_10' not in achievements:
        achievements['streak_10'] = {'unlocked_at': datetime.now().isoformat()}
        new_unlocks.append(ACHIEVEMENTS['streak_10'])
    
    # 100 prédictions
    if stats.get('total_predictions', 0) >= 100 and 'pred_100' not in achievements:
        achievements['pred_100'] = {'unlocked_at': datetime.now().isoformat()}
        new_unlocks.append(ACHIEVEMENTS['pred_100'])
    
    # Value bets
    value_wins = sum(1 for p in history if p.get('best_value') and p.get('statut') == 'gagne')
    if value_wins >= 10 and 'value_master' not in achievements:
        achievements['value_master'] = {'unlocked_at': datetime.now().isoformat()}
        new_unlocks.append(ACHIEVEMENTS['value_master'])
    
    # Spécialiste surface
    surfaces_won = set()
    for p in history:
        if p.get('statut') == 'gagne':
            surfaces_won.add(p.get('surface'))
    if len(surfaces_won) >= 3 and 'surface_specialist' not in achievements:
        achievements['surface_specialist'] = {'unlocked_at': datetime.now().isoformat()}
        new_unlocks.append(ACHIEVEMENTS['surface_specialist'])
    
    if new_unlocks:
        save_achievements(achievements)
        send_achievement_notification(new_unlocks)
    
    return new_unlocks

def send_achievement_notification(achievements):
    """Envoie une notification Telegram pour les nouveaux achievements"""
    message = "🏆 **NOUVEAU BADGE DÉBLOQUÉ !** 🏆\n\n"
    for ach in achievements:
        message += f"{ach['icon']} **{ach['name']}**\n{ach['desc']}\n\n"
    
    send_telegram_message(message)

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
        st.error(f"Erreur Telegram: {str(e)}")
        return False

def format_prediction_message(pred_data, bet_suggestions=None, ai_comment=None):
    """Formate un message de prédiction pour Telegram"""
    proba = pred_data.get('proba', 0.5)
    bar_length = 10
    filled = int(proba * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    emoji_map = {'Hard': '🟦', 'Clay': '🟧', 'Grass': '🟩'}
    surface_emoji = emoji_map.get(pred_data.get('surface', ''), '🎾')
    
    ml_tag = "🤖 " if pred_data.get('ml_used') else ""
    gagnant = pred_data.get('favori', '?')
    
    message = f"""
<b>{ml_tag}🎾 PRÉDICTION TENNISIQ</b>

<b>Match:</b> {pred_data.get('player1', '?')} vs {pred_data.get('player2', '?')}
<b>Tournoi:</b> {pred_data.get('tournament', 'Inconnu')}
<b>Surface:</b> {surface_emoji} {pred_data.get('surface', '?')}

<b>📊 ANALYSE DU MATCH:</b>
{bar}  {proba:.1%} / {1-proba:.1%}

• {pred_data.get('player1', 'J1')}: <b>{proba:.1%}</b>
• {pred_data.get('player2', 'J2')}: <b>{1-proba:.1%}</b>

<b>🏆 GAGNANT PRÉDIT: <u>{gagnant}</u></b>
<b>Confiance:</b> {'🟢' if pred_data.get('confidence', 0) >= 70 else '🟡' if pred_data.get('confidence', 0) >= 50 else '🔴'} {pred_data.get('confidence', 0):.0f}/100
"""
    
    if pred_data.get('odds1') and pred_data.get('odds2'):
        message += f"""
<b>Cotes:</b>
• {pred_data.get('player1', 'J1')}: <code>{pred_data.get('odds1')}</code>
• {pred_data.get('player2', 'J2')}: <code>{pred_data.get('odds2')}</code>
"""
    
    if bet_suggestions:
        message += f"\n<b>🎯 PARIS ALTERNATIFS:</b>\n"
        for bet in bet_suggestions[:3]:
            conf_icon = '🟢' if bet['confidence'] >= 70 else '🟡' if bet['confidence'] >= 50 else '🔴'
            message += f"\n{conf_icon} <b>{bet['type']}</b>: {bet['description']}\n"
            message += f"   Probabilité: {bet['proba']:.1%} | Cote: {bet['cote']:.2f}\n"
    
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

def send_stats_to_telegram():
    return send_telegram_message(format_stats_message())

def send_custom_message():
    """Envoie un message personnalisé sur Telegram"""
    st.markdown("### 📝 Message personnalisé")
    with st.form("custom_msg_form"):
        title = st.text_input("Titre", "Message TennisIQ")
        content = st.text_area("Contenu", height=100, placeholder="Écris ton message ici...")
        urgent = st.checkbox("🔴 Urgent")
        
        col1, col2 = st.columns(2)
        with col1:
            include_stats = st.checkbox("📊 Inclure les stats")
        with col2:
            include_time = st.checkbox("🕐 Inclure la date", True)
        
        if st.form_submit_button("📤 Envoyer le message") and content:
            urgent_tag = "🔴 URGENT - " if urgent else ""
            date_tag = f"\n\n📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}" if include_time else ""
            stats_tag = f"\n\n{format_stats_message()}" if include_stats else ""
            
            msg = f"<b>{urgent_tag}{title}</b>\n\n{content}{date_tag}{stats_tag}\n\n#TennisIQ"
            
            if send_telegram_message(msg):
                st.success("✅ Message envoyé avec succès sur Telegram !")
            else:
                st.error("❌ Échec de l'envoi. Vérifie la configuration Telegram.")

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
🤖 Bot TennisIQ opérationnel

📊 Statistiques actuelles:
• Prédictions: {len(load_history())}
• Précision: {calculate_global_accuracy():.1f}%

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

def setup_telegram_commands():
    """Configure les commandes du bot Telegram"""
    token, _ = get_telegram_config()
    if not token:
        return
    
    commands = [
        {"command": "start", "description": "Démarrer le bot"},
        {"command": "predict", "description": "Prédire un match (ex: /predict Djokovic Nadal Clay)"},
        {"command": "stats", "description": "Statistiques globales"},
        {"command": "today", "description": "Matchs du jour"},
        {"command": "value", "description": "Value bets du jour"},
        {"command": "badges", "description": "Mes badges"},
        {"command": "help", "description": "Aide"}
    ]
    
    url = f"https://api.telegram.org/bot{token}/setMyCommands"
    try:
        requests.post(url, json={"commands": commands})
    except:
        pass

def handle_telegram_command(text):
    """Gère les commandes Telegram"""
    parts = text.lower().split()
    cmd = parts[0] if parts else ""
    
    if cmd == "/start":
        return "👋 Bienvenue sur TennisIQ Bot ! Utilise /help pour voir les commandes."
    
    elif cmd == "/predict" and len(parts) >= 3:
        p1 = parts[1].title()
        p2 = parts[2].title()
        surface = parts[3].title() if len(parts) > 3 else "Hard"
        
        proba, _ = calculate_probability(p1, p2, surface, None, load_saved_model())
        pred_data = {
            'player1': p1, 'player2': p2,
            'surface': surface, 'proba': proba,
            'favori': p1 if proba >= 0.5 else p2
        }
        return format_prediction_message(pred_data)
    
    elif cmd == "/stats":
        return format_stats_message()
    
    elif cmd == "/today":
        matches = get_daily_matches(force_refresh=True)
        if not matches:
            return "📅 Aucun match trouvé aujourd'hui"
        
        message = "<b>📅 MATCHS DU JOUR</b>\n\n"
        for i, match in enumerate(matches[:10], 1):
            message += f"{i}. {match['p1']} vs {match['p2']} - {match['surface']}\n"
        return message
    
    elif cmd == "/value":
        matches = get_daily_matches()
        value_bets = scan_for_value_bets(matches)
        
        if not value_bets:
            return "🎯 Aucun value bet détecté aujourd'hui"
        
        message = "<b>🎯 VALUE BETS DU JOUR</b>\n\n"
        for i, vb in enumerate(value_bets[:5], 1):
            message += f"{i}. <b>{vb['joueur']}</b> @ {vb['cote']}\n"
            message += f"   Edge: +{vb['edge']:.1f}% | Proba: {vb['proba']:.1%}\n"
            message += f"   {vb['match']} - {vb['surface']}\n\n"
        return message
    
    elif cmd == "/badges":
        achievements = load_achievements()
        if not achievements:
            return "🏆 Aucun badge débloqué pour le moment"
        
        message = "<b>🏆 MES BADGES</b>\n\n"
        for ach_id, data in achievements.items():
            if ach_id in ACHIEVEMENTS:
                ach = ACHIEVEMENTS[ach_id]
                date = datetime.fromisoformat(data['unlocked_at']).strftime('%d/%m/%Y')
                message += f"{ach['icon']} <b>{ach['name']}</b>\n{ach['desc']} - {date}\n\n"
        return message
    
    elif cmd == "/help":
        return """
<b>📋 Commandes disponibles:</b>

/predict - Prédire un match
/stats - Statistiques globales
/today - Matchs du jour
/value - Value bets du jour
/badges - Mes badges
/help - Cette aide
"""
    
    return "Commande non reconnue. Tape /help pour voir les commandes."

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
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return None
    except Exception as e:
        st.error(f"Erreur IA: {str(e)}")
        return None

def analyze_match_with_ai(player1, player2, surface, tournament, proba, best_value=None, bet_suggestions=None):
    """Génère une analyse IA claire avec le gagnant en évidence"""
    gagnant = player1 if proba >= 0.5 else player2
    perdant = player2 if proba >= 0.5 else player1
    proba_gagnant = proba if proba >= 0.5 else 1-proba
    
    vb_txt = f" Value bet détecté sur {best_value['joueur']} (edge {best_value['edge']*100:+.1f}%)" if best_value else ""
    
    prompt = f"""Analyse ce match de tennis de façon claire et concise:

Match: {player1} vs {player2}
Tournoi: {tournament}
Surface: {surface}

ANALYSE DES DONNÉES:
- Probabilité {player1}: {proba:.1%}
- Probabilité {player2}: {1-proba:.1%}
- GAGNANT PRÉDIT: {gagnant} ({proba_gagnant:.1%} de chances)
{vb_txt}

Donne une analyse en 4 points:
1. Pourquoi {gagnant} est favori (facteurs clés)
2. Les points faibles de {perdant} dans ce match
3. {vb_txt if best_value else "Conseil de pari"}
4. Pronostic final clair

Sois direct et précis."""
    
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
            st.error(f"Erreur chargement modèle: {e}")
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
                    st.success("✅ Modèle téléchargé depuis GitHub!")
                    return model_info
        except Exception as e:
            st.warning(f"⚠️ Impossible de télécharger le modèle: {e}")
    
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
        
        r1 = max(s1.get('rank', 500.0), 1.0)
        r2 = max(s2.get('rank', 500.0), 1.0)
        log_rank_ratio = np.log(r2 / r1)
        
        surf_wr_diff = s1.get('surface_wr', {}).get(surface, 0.5) - s2.get('surface_wr', {}).get(surface, 0.5)
        
        features = np.array([[
            log_rank_ratio, 0, 0,
            1 if surface == 'Clay' else 0,
            1 if surface == 'Grass' else 0,
            1 if surface == 'Hard' else 0,
            0, 0, 0,
            surf_wr_diff, 0, 0, 0.5,
            0, 0, 0, 0, 0, 0, 0, 0
        ]])
        
        features_scaled = scaler.transform(features)
        proba = model.predict_proba(features_scaled)[0][1]
        
        return max(0.05, min(0.95, float(proba)))
    except Exception as e:
        st.error(f"Erreur prédiction ML: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES ATP (FICHIERS CSV) - TOUS LES JOUEURS
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_atp_data():
    """Charge TOUS les joueurs depuis les fichiers CSV sans limitation"""
    if not DATA_DIR.exists():
        st.warning(f"📁 Dossier non trouvé: {DATA_DIR}")
        return []
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        st.warning("📁 Aucun fichier CSV trouvé")
        return []
    
    all_players = set()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, f in enumerate(csv_files):
        if 'wta' in f.name.lower():
            continue
        
        status_text.text(f"Chargement: {f.name}")
        
        try:
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    # Lire tous les joueurs sans limite de lignes
                    df = pd.read_csv(f, encoding=enc, usecols=['winner_name', 'loser_name'], 
                                     on_bad_lines='skip')
                    break
                except:
                    try:
                        df = pd.read_csv(f, sep=';', encoding=enc, usecols=['winner_name', 'loser_name'],
                                         on_bad_lines='skip')
                        break
                    except:
                        continue
            else:
                continue
            
            if df is not None:
                winners = df['winner_name'].dropna().astype(str).str.strip()
                losers = df['loser_name'].dropna().astype(str).str.strip()
                
                all_players.update(winners)
                all_players.update(losers)
                
        except Exception as e:
            print(f"Erreur avec {f.name}: {e}")
        
        progress_bar.progress((idx + 1) / len(csv_files))
    
    progress_bar.empty()
    status_text.empty()
    
    # Filtrer les valeurs invalides
    valid_players = []
    for p in all_players:
        p_str = str(p).strip()
        if p_str and p_str.lower() != 'nan' and len(p_str) > 1:
            valid_players.append(p_str)
    
    # Trier tous les joueurs
    valid_players = sorted(valid_players)
    
    st.success(f"✅ {len(valid_players):,} joueurs uniques chargés")
    return valid_players

@st.cache_data(ttl=3600)
def get_h2h_stats_df():
    """Charge un DataFrame pour les stats H2H depuis les CSV"""
    if not DATA_DIR.exists():
        return pd.DataFrame()
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    dfs = []
    progress_bar = st.progress(0)
    
    for idx, f in enumerate(csv_files[:20]):  # Limiter pour la performance
        if 'wta' in f.name.lower():
            continue
        try:
            df = pd.read_csv(f, encoding='utf-8', usecols=['winner_name', 'loser_name'], 
                            on_bad_lines='skip')
            df['winner_name'] = df['winner_name'].astype(str).str.strip()
            df['loser_name'] = df['loser_name'].astype(str).str.strip()
            dfs.append(df)
        except:
            continue
        progress_bar.progress((idx + 1) / min(len(csv_files), 20))
    
    progress_bar.empty()
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def get_h2h_stats(player1, player2):
    """Récupère les stats H2H entre deux joueurs depuis les CSV"""
    df = get_h2h_stats_df()
    if df.empty:
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

def calculate_probability(player1, player2, surface, h2h=None, model_info=None):
    """Calcule la probabilité en utilisant d'abord le ML, puis les stats CSV"""
    
    # Essayer d'abord avec le modèle ML si disponible
    if model_info:
        ml_proba = predict_with_ml_model(model_info, player1, player2, surface)
        if ml_proba is not None:
            return ml_proba, True
    
    # Fallback sur les stats des CSV
    proba = 0.5
    
    # Ajustement basé sur H2H si disponible
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

def calculate_global_accuracy():
    """Calcule la précision globale"""
    stats = load_user_stats()
    total_valide = stats.get('correct_predictions', 0) + stats.get('incorrect_predictions', 0)
    return (stats.get('correct_predictions', 0) / total_valide * 100) if total_valide > 0 else 0

# ─────────────────────────────────────────────────────────────
# SCRAPING RÉEL DES MATCHS
# ─────────────────────────────────────────────────────────────
def scrape_flashscore():
    """Scrape les matchs depuis FlashScore"""
    if not BS4_AVAILABLE:
        return None
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        url = "https://www.flashscore.fr/tennis/"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        matches = []
        event_items = soup.find_all('div', class_='event__match')
        
        for item in event_items[:15]:
            try:
                home = item.find('div', class_='event__participant--home')
                away = item.find('div', class_='event__participant--away')
                
                if home and away:
                    p1 = home.text.strip()
                    p2 = away.text.strip()
                    
                    surface = 'Hard'
                    tournament_elem = item.find('div', class_='event__title')
                    if tournament_elem:
                        tournament_text = tournament_elem.text.lower()
                        if 'clay' in tournament_text or 'terre' in tournament_text:
                            surface = 'Clay'
                        elif 'grass' in tournament_text or 'gazon' in tournament_text:
                            surface = 'Grass'
                    
                    matches.append({
                        'p1': p1,
                        'p2': p2,
                        'surface': surface,
                        'tournament': tournament_elem.text if tournament_elem else "Tournoi ATP"
                    })
            except:
                continue
        
        return matches if matches else None
        
    except Exception as e:
        print(f"Erreur FlashScore: {e}")
        return None

def scrape_tennis_explorer():
    """Scrape les matchs depuis Tennis Explorer"""
    if not BS4_AVAILABLE:
        return None
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        url = "http://www.tennisexplorer.com/"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        matches = []
        today_matches = soup.find('table', class_='results')
        
        if today_matches:
            rows = today_matches.find_all('tr')[1:16]
            
            for row in rows:
                try:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        players = cells[0].text.strip().split(' - ')
                        if len(players) == 2:
                            p1, p2 = players[0], players[1]
                            
                            surface_text = cells[1].text.strip() if len(cells) > 1 else ""
                            surface = 'Clay' if 'Clay' in surface_text else 'Hard' if 'Hard' in surface_text else 'Grass'
                            
                            matches.append({
                                'p1': p1,
                                'p2': p2,
                                'surface': surface,
                                'tournament': cells[2].text.strip() if len(cells) > 2 else "Tournoi ATP"
                            })
                except:
                    continue
        
        return matches if matches else None
        
    except Exception as e:
        print(f"Erreur Tennis Explorer: {e}")
        return None

def get_mock_matches():
    """Fallback avec des matchs simulés"""
    return [
        {'p1': 'Novak Djokovic', 'p2': 'Carlos Alcaraz', 'surface': 'Clay', 'tournament': 'Roland Garros'},
        {'p1': 'Jannik Sinner', 'p2': 'Daniil Medvedev', 'surface': 'Hard', 'tournament': 'Miami Open'},
        {'p1': 'Rafael Nadal', 'p2': 'Stefanos Tsitsipas', 'surface': 'Clay', 'tournament': 'Barcelona Open'},
        {'p1': 'Alexander Zverev', 'p2': 'Andrey Rublev', 'surface': 'Hard', 'tournament': 'Madrid Open'},
        {'p1': 'Holger Rune', 'p2': 'Casper Ruud', 'surface': 'Grass', 'tournament': 'Wimbledon'},
    ]

@st.cache_data(ttl=1800)
def get_daily_matches(force_refresh=False):
    """Récupère les vrais matchs du jour avec fallback"""
    
    if force_refresh:
        st.cache_data.clear()
    
    matches = scrape_flashscore()
    if matches and len(matches) > 0:
        return matches
    
    matches = scrape_tennis_explorer()
    if matches and len(matches) > 0:
        return matches
    
    return get_mock_matches()

def auto_load_today_matches():
    """Bouton pour charger automatiquement les matchs du jour"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📅 Charger les matchs ATP du jour", use_container_width=True, key="load_today_matches"):
            with st.spinner("🌐 Récupération des matchs en direct..."):
                matches = get_daily_matches(force_refresh=True)
                if matches:
                    st.session_state['today_matches'] = matches
                    st.session_state['scrape_time'] = datetime.now().strftime('%H:%M')
                    st.success(f"✅ {len(matches)} matchs chargés!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Aucun match trouvé")
            return matches
    return None

# ─────────────────────────────────────────────────────────────
# DÉTECTION VALUE BETS
# ─────────────────────────────────────────────────────────────
def scan_for_value_bets(matches):
    """Scanne une liste de matchs pour trouver des value bets"""
    value_bets = []
    model_info = load_saved_model()
    
    for match in matches:
        proba, _ = calculate_probability(match['p1'], match['p2'], match['surface'], None, model_info)
        
        odds1 = round(1/proba * (0.9 + 0.2*np.random.random()), 2)
        odds2 = round(1/(1-proba) * (0.9 + 0.2*np.random.random()), 2)
        
        implied_prob1 = 1 / odds1
        edge1 = proba - implied_prob1
        
        implied_prob2 = 1 / odds2
        edge2 = (1-proba) - implied_prob2
        
        if edge1 > MIN_EDGE_COMBINE:
            value_bets.append({
                'match': f"{match['p1']} vs {match['p2']}",
                'joueur': match['p1'],
                'edge': edge1 * 100,
                'cote': odds1,
                'proba': proba,
                'surface': match['surface'],
                'tournament': match['tournament']
            })
        elif edge2 > MIN_EDGE_COMBINE:
            value_bets.append({
                'match': f"{match['p1']} vs {match['p2']}",
                'joueur': match['p2'],
                'edge': edge2 * 100,
                'cote': odds2,
                'proba': 1-proba,
                'surface': match['surface'],
                'tournament': match['tournament']
            })
    
    return sorted(value_bets, key=lambda x: x['edge'], reverse=True)

# ─────────────────────────────────────────────────────────────
# FONCTIONS POUR LES PARIS ALTERNATIFS
# ─────────────────────────────────────────────────────────────
def generate_alternative_bets(player1, player2, surface, proba, h2h=None):
    """Génère des suggestions de paris alternatifs"""
    suggestions = []
    
    suggestions.append({
        'type': '📊 Over 22.5 games',
        'description': f"Plus de 22.5 jeux",
        'proba': 0.62,
        'cote': 1.80,
        'confidence': 65
    })
    
    if proba > 0.65:
        suggestions.append({
            'type': '⚖️ Handicap -3.5',
            'description': f"{player1} gagne avec écart",
            'proba': 0.58,
            'cote': 2.10,
            'confidence': 60
        })
    elif proba < 0.35:
        suggestions.append({
            'type': '⚖️ Handicap +3.5',
            'description': f"{player2} perd par moins de 4 jeux",
            'proba': 0.62,
            'cote': 1.95,
            'confidence': 65
        })
    
    if 0.3 < proba < 0.7:
        suggestions.append({
            'type': '🔄 Les deux gagnent un set',
            'description': f"Chaque joueur gagne au moins un set",
            'proba': 0.55,
            'cote': 2.20,
            'confidence': 55
        })
    
    return suggestions

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
    """Sauvegarde automatiquement une prédiction"""
    try:
        history = load_history()
        pred_data['id'] = hashlib.md5(f"{datetime.now()}{pred_data.get('player1','')}".encode()).hexdigest()[:8]
        pred_data['statut'] = 'en_attente'
        history.append(pred_data)
        with open(HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(history[-1000:], f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Erreur sauvegarde: {e}")
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
            json.dump(history, f, indent=2, ensure_ascii=False)
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
            'current_streak': 0,
            'best_streak': 0
        }
    try:
        with open(USER_STATS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def update_user_stats():
    history = load_history()
    correct = sum(1 for p in history if p.get('statut') == 'gagne')
    incorrect = sum(1 for p in history if p.get('statut') == 'perdu')
    
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
        'total_predictions': len(history),
        'correct_predictions': correct,
        'incorrect_predictions': incorrect,
        'annules_predictions': sum(1 for p in history if p.get('statut') == 'annule'),
        'current_streak': current_streak,
        'best_streak': best_streak
    }
    
    with open(USER_STATS_FILE, 'w') as f:
        json.dump(stats, f)
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
        with open(COMB_HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(combines[-200:], f, indent=2, ensure_ascii=False)
        return True
    except:
        return False

# ─────────────────────────────────────────────────────────────
# SÉLECTEUR DE JOUEUR AVEC RECHERCHE - TOUS LES JOUEURS
# ─────────────────────────────────────────────────────────────
def player_selector(label, all_players, key, default=None):
    """Composant de sélection de joueur avec recherche - TOUS les joueurs"""
    
    if f"search_{key}" not in st.session_state:
        st.session_state[f"search_{key}"] = ""
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input(f"🔍 Rechercher {label}", 
                               value=st.session_state[f"search_{key}"],
                               key=f"search_input_{key}",
                               placeholder="Tapez le nom...")
        st.session_state[f"search_{key}"] = search
    
    # Filtrer tous les joueurs sans limite
    if search:
        search_lower = search.lower()
        filtered = [p for p in all_players if search_lower in p.lower()]
        if not filtered:
            st.warning(f"Aucun joueur trouvé pour '{search}'")
            # Suggestions par première lettre
            filtered = [p for p in all_players if p[0].lower() == search_lower[0]][:100]
    else:
        # Afficher les 200 premiers par défaut pour la performance
        filtered = all_players[:200]
    
    with col2:
        st.caption(f"{len(filtered)} trouvés sur {len(all_players):,} total")
    
    if filtered:
        default_idx = 0
        if default:
            # Chercher le default dans la liste filtrée
            for i, p in enumerate(filtered):
                if default.lower() in p.lower():
                    default_idx = i
                    break
        
        selected = st.selectbox(label, filtered, index=default_idx, key=key)
        return selected
    else:
        return st.text_input(label, key=key)

# ─────────────────────────────────────────────────────────────
# SÉLECTEUR DE TOURNOI AVEC RECHERCHE - TOUS LES TOURNOIS
# ─────────────────────────────────────────────────────────────
def tournament_selector(label, key, default=None):
    """Sélecteur de tournoi avec recherche - TOUS les tournois + alias"""
    if f"search_tourn_{key}" not in st.session_state:
        st.session_state[f"search_tourn_{key}"] = ""
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input(f"🔍 Rechercher {label}", 
                               value=st.session_state[f"search_tourn_{key}"],
                               key=f"search_tourn_input_{key}",
                               placeholder="Tapez le tournoi (ex: Acapulco)...")
        st.session_state[f"search_tourn_{key}"] = search
    
    # Obtenir tous les tournois
    all_tournaments = sorted(TOURNAMENTS_DB.keys())
    
    if search:
        # Recherche avec alias
        search_lower = search.lower().strip()
        results = set()
        
        # 1. Vérifier si c'est un alias direct
        if search_lower in TOURNAMENT_ALIASES:
            official = TOURNAMENT_ALIASES[search_lower]
            results.add(official)
        
        # 2. Chercher dans les noms officiels
        for tourney in all_tournaments:
            if search_lower in tourney.lower():
                results.add(tourney)
        
        # 3. Chercher dans les alias
        for alias, official in TOURNAMENT_ALIASES.items():
            if search_lower in alias:
                results.add(official)
        
        if not results:
            st.warning(f"Aucun tournoi trouvé pour '{search}'")
            # Suggestions par première lettre
            results = {t for t in all_tournaments if t[0].lower() == search_lower[0]}
        
        filtered = sorted(results)
    else:
        # Afficher les 100 premiers par défaut
        filtered = all_tournaments[:100]
    
    with col2:
        st.caption(f"{len(filtered)} tournois sur {len(all_tournaments)} total")
    
    # Trouver l'index par défaut
    default_idx = 0
    if default:
        # Chercher d'abord par alias
        found = find_tournament(default)
        if found and found in filtered:
            default_idx = filtered.index(found)
        elif default in filtered:
            default_idx = filtered.index(default)
    
    return st.selectbox(label, filtered, index=default_idx, key=key)

# ─────────────────────────────────────────────────────────────
# DASHBOARD INTERACTIF
# ─────────────────────────────────────────────────────────────
def create_interactive_dashboard():
    """Crée un dashboard interactif avec Plotly"""
    st.markdown("## 📊 Dashboard Interactif")
    
    history = load_history()
    if not history:
        st.info("Pas assez de données pour le dashboard")
        return
    
    df = pd.DataFrame(history)
    if df.empty:
        st.info("Pas assez de données pour le dashboard")
        return
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 1. Évolution de la précision
    st.markdown("### 📈 Évolution de la précision")
    
    df['correct'] = (df['statut'] == 'gagne').astype(int)
    df['cum_correct'] = df['correct'].expanding().sum()
    df['cum_total'] = (df['statut'].isin(['gagne', 'perdu'])).expanding().sum()
    df['accuracy'] = (df['cum_correct'] / df['cum_total'] * 100).fillna(0)
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df['date'],
        y=df['accuracy'],
        mode='lines+markers',
        name='Précision',
        line=dict(color=COLORS['primary'], width=3),
        fill='tozeroy',
        fillcolor=f'rgba(0,223,162,0.1)'
    ))
    
    fig1.update_layout(
        title='Évolution de la précision',
        xaxis_title='Date',
        yaxis_title='Précision (%)',
        hovermode='x',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Performance par surface
    st.markdown("### 🎾 Performance par surface")
    
    surface_stats = []
    for surface in SURFACES:
        surface_preds = df[df['surface'] == surface]
        if len(surface_preds) > 0:
            correct = len(surface_preds[surface_preds['statut'] == 'gagne'])
            total = len(surface_preds[surface_preds['statut'].isin(['gagne', 'perdu'])])
            accuracy = (correct / total * 100) if total > 0 else 0
            surface_stats.append({
                'Surface': surface,
                'Accuracy': accuracy,
                'Total': total,
                'Correct': correct
            })
    
    if surface_stats:
        df_surface = pd.DataFrame(surface_stats)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df_surface['Surface'],
            y=df_surface['Accuracy'],
            text=df_surface['Accuracy'].round(1).astype(str) + '%',
            textposition='outside',
            marker_color=[SURFACE_CONFIG[s]['color'] for s in df_surface['Surface']],
            hovertemplate='<b>%{x}</b><br>Précision: %{y:.1f}%<br>Matchs: %{customdata[0]}<br>Victoires: %{customdata[1]}<extra></extra>',
            customdata=df_surface[['Total', 'Correct']]
        ))
        
        fig2.update_layout(
            title='Précision par surface',
            xaxis_title='Surface',
            yaxis_title='Précision (%)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ANALYSE DES TENDANCES
# ─────────────────────────────────────────────────────────────
def analyze_trends():
    """Analyse les tendances des paris"""
    history = load_history()
    if len(history) < 10:
        return {}
    
    df = pd.DataFrame(history)
    df['date'] = pd.to_datetime(df['date'])
    
    trends = {}
    
    # Meilleure surface
    surface_perf = {}
    for surface in SURFACES:
        surf_preds = df[df['surface'] == surface]
        if len(surf_preds) >= 5:
            correct = len(surf_preds[surf_preds['statut'] == 'gagne'])
            total = len(surf_preds[surf_preds['statut'].isin(['gagne', 'perdu'])])
            surface_perf[surface] = correct / total if total > 0 else 0
    
    if surface_perf:
        trends['best_surface'] = max(surface_perf, key=surface_perf.get)
        trends['best_surface_acc'] = surface_perf[trends['best_surface']] * 100
    
    # Meilleure plage de confiance
    confidence_ranges = [(0,50), (50,70), (70,85), (85,100)]
    range_perf = {}
    
    for low, high in confidence_ranges:
        range_preds = df[(df['confidence'] >= low) & (df['confidence'] < high)]
        if len(range_preds) >= 5:
            correct = len(range_preds[range_preds['statut'] == 'gagne'])
            total = len(range_preds[range_preds['statut'].isin(['gagne', 'perdu'])])
            range_perf[f"{low}-{high}"] = correct / total if total > 0 else 0
    
    if range_perf:
        trends['best_confidence'] = max(range_perf, key=range_perf.get)
        trends['best_confidence_acc'] = range_perf[trends['best_confidence']] * 100
    
    # Tendance récente
    recent = df.tail(20)
    recent_correct = len(recent[recent['statut'] == 'gagne'])
    recent_total = len(recent[recent['statut'].isin(['gagne', 'perdu'])])
    trends['recent_trend'] = recent_correct / recent_total * 100 if recent_total > 0 else 0
    
    try:
        with open(TRENDS_FILE, 'w') as f:
            json.dump(trends, f, indent=2)
    except:
        pass
    
    return trends

def generate_betting_advice():
    """Génère des conseils de paris basés sur les tendances"""
    trends = analyze_trends()
    if not trends:
        return "Pas assez de données pour des conseils"
    
    advice = []
    
    if 'best_surface' in trends:
        advice.append(f"🎾 Vous êtes plus performant sur {trends['best_surface']} ({trends['best_surface_acc']:.1f}%)")
    
    if 'best_confidence' in trends:
        advice.append(f"🎯 Vos meilleurs paris sont avec une confiance {trends['best_confidence']} ({trends['best_confidence_acc']:.1f}%)")
    
    if trends.get('recent_trend', 0) > 60:
        advice.append(f"🔥 Forme récente: {trends['recent_trend']:.1f}% - vous êtes en confiance!")
    elif trends.get('recent_trend', 0) < 40:
        advice.append(f"⚠️ Forme récente: {trends['recent_trend']:.1f}% - peut-être réduire les mises")
    
    return "\n".join(advice)

# ─────────────────────────────────────────────────────────────
# PAGE DES PRÉDICTIONS EN ATTENTE
# ─────────────────────────────────────────────────────────────
def show_pending():
    """Page des prédictions en attente pour valider les résultats"""
    st.markdown("## ⏳ Prédictions en attente")
    
    history = load_history()
    pending = [p for p in history if p.get('statut') == 'en_attente']
    
    if not pending:
        st.info("🎉 Aucune prédiction en attente !")
        return
    
    st.success(f"📊 {len(pending)} prédiction(s) en attente de résultat")
    
    # Afficher les prédictions en attente
    for pred in pending[::-1]:  # Du plus récent au plus ancien
        with st.expander(f"📅 {pred.get('date', '')[:16]} - {pred['player1']} vs {pred['player2']}", expanded=True):
            
            # Informations du match
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tournoi", pred.get('tournament', 'Inconnu'))
            with col2:
                surface_icon = SURFACE_CONFIG.get(pred.get('surface'), {}).get('icon', '🎾')
                st.metric("Surface", f"{surface_icon} {pred.get('surface', '?')}")
            with col3:
                st.metric("Probabilité", f"{pred.get('proba', 0.5):.1%}")
            
            # Cotes si disponibles
            if pred.get('odds1') and pred.get('odds2'):
                st.caption(f"Cotes: {pred['player1']} @ {pred['odds1']} | {pred['player2']} @ {pred['odds2']}")
            
            # Value bet si détecté
            if pred.get('best_value'):
                bv = pred['best_value']
                st.info(f"🎯 Value bet détecté: {bv['joueur']} @ {bv['cote']} (edge: +{bv['edge']*100:.1f}%)")
            
            # Boutons pour le résultat
            st.markdown("### 🎯 Résultat du match")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button(f"✅ {pred['player1']} gagne", key=f"win1_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'gagne')
                    st.rerun()
            
            with col2:
                if st.button(f"✅ {pred['player2']} gagne", key=f"win2_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'gagne')
                    st.rerun()
            
            with col3:
                if st.button(f"❌ Match perdu", key=f"loss_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'perdu')
                    st.rerun()
            
            with col4:
                if st.button(f"⚠️ Annuler", key=f"cancel_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'annule')
                    st.rerun()

# ─────────────────────────────────────────────────────────────
# PAGES DE L'APPLICATION
# ─────────────────────────────────────────────────────────────

def show_dashboard():
    """Page Dashboard améliorée avec tendances et badges"""
    st.markdown("## 🏠 Dashboard")
    
    stats = load_user_stats()
    history = load_history()
    trends = analyze_trends()
    achievements = load_achievements()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total prédictions", stats.get('total_predictions', 0))
    with col2:
        accuracy = calculate_global_accuracy()
        delta = trends.get('recent_trend', 0) - accuracy if trends else 0
        st.metric("Précision", f"{accuracy:.1f}%", f"{delta:+.1f}%")
    with col3:
        st.metric("En attente", len([p for p in history if p.get('statut') == 'en_attente']))
    with col4:
        st.metric("Badges", len(achievements))
    
    if achievements:
        st.markdown("### 🏆 Badges débloqués")
        cols = st.columns(min(len(achievements), 4))
        for i, (ach_id, data) in enumerate(list(achievements.items())[:4]):
            with cols[i % 4]:
                ach = ACHIEVEMENTS.get(ach_id, {})
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: rgba(0,223,162,0.1); border-radius: 10px;">
                    <div style="font-size: 2rem;">{ach.get('icon', '🏆')}</div>
                    <div style="font-weight: bold;">{ach.get('name', 'Badge')}</div>
                    <div style="font-size: 0.8rem; color: {COLORS['gray']};">{ach.get('desc', '')}</div>
                </div>
                """, unsafe_allow_html=True)
    
    if trends:
        st.markdown("### 💡 Conseils personnalisés")
        advice = generate_betting_advice()
        st.info(advice)
    
    create_interactive_dashboard()
    
    model_info = load_saved_model()
    groq_key = get_groq_key()
    telegram_token, _ = get_telegram_config()
    
    st.markdown("### 🛠️ Statut des services")
    col1, col2, col3 = st.columns(3)
    with col1:
        if model_info:
            st.success(f"✅ Modèle ML ({model_info.get('accuracy', 0):.1%})")
        else:
            st.warning("⚠️ Modèle ML non chargé (mode stats CSV actif)")
    with col2:
        st.success("✅ IA Groq" if groq_key else "⚠️ IA non configurée")
    with col3:
        st.success("✅ Telegram" if telegram_token else "⚠️ Telegram non configuré")

def show_prediction():
    """Page de prédiction avec scraping automatique et value bets"""
    st.markdown("## 🎯 Analyse Multi-matchs")
    
    model_info = load_saved_model()
    
    with st.spinner("Chargement de tous les joueurs depuis les CSV..."):
        all_players = load_atp_data()
    
    st.success(f"✅ {len(all_players):,} joueurs disponibles")
    st.info(f"✅ {len(TOURNAMENTS_DB)} tournois disponibles ({len(TOURNAMENT_ALIASES)} alias)")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        auto_load = st.checkbox("📅 Auto-load matchs", value=True, key="auto_load")
    with col2:
        if auto_load:
            auto_load_today_matches()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        default_n = len(st.session_state.get('today_matches', [3]))
        n_matches = st.number_input("Nombre de matchs", 1, MAX_MATCHES_ANALYSIS, value=default_n, key="n_matches")
    with col2:
        mise = st.number_input("Mise (€)", 1.0, 1000.0, 10.0, key="mise")
    with col3:
        use_ai = st.checkbox("🤖 Analyser avec IA", True, key="use_ai")
    with col4:
        send_tg = st.checkbox("📱 Envoyer Telegram", True, key="send_tg")
    
    matches = []
    st.markdown("### 📝 Saisie des matchs")
    
    tournaments_list = sorted(TOURNAMENTS_DB.keys())
    today_matches = st.session_state.get('today_matches', [])
    
    for i in range(n_matches):
        with st.expander(f"Match {i+1}", expanded=i==0):
            default_p1 = today_matches[i]['p1'] if i < len(today_matches) else "Novak Djokovic"
            default_p2 = today_matches[i]['p2'] if i < len(today_matches) else "Carlos Alcaraz"
            default_tournament = today_matches[i]['tournament'] if i < len(today_matches) else tournaments_list[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                p1 = player_selector(f"Joueur 1", all_players, key=f"p1_{i}", default=default_p1)
                odds1 = st.text_input(f"Cote {p1}", key=f"odds1_{i}", placeholder="1.75", value="1.75" if i==0 else "")
            
            with col2:
                if p1:
                    players2 = [p for p in all_players if p != p1]
                    p2 = player_selector(f"Joueur 2", players2, key=f"p2_{i}", default=default_p2)
                else:
                    p2 = player_selector(f"Joueur 2", all_players, key=f"p2_{i}")
                odds2 = st.text_input(f"Cote {p2}", key=f"odds2_{i}", placeholder="2.10", value="2.10" if i==0 else "")
            
            col1, col2 = st.columns(2)
            with col1:
                tournament = tournament_selector(f"Tournoi", key=f"tourn_{i}", default=default_tournament)
                surface = get_tournament_surface(tournament)
            with col2:
                st.info(f"Surface: {SURFACE_CONFIG[surface]['icon']} {surface}")
            
            matches.append({
                'player1': p1 if p1 else "",
                'player2': p2 if p2 else "",
                'surface': surface,
                'tournament': tournament,
                'odds1': odds1,
                'odds2': odds2,
                'index': i
            })
    
    analyze_button = st.button("🔍 Analyser tous les matchs", type="primary", use_container_width=True, key="analyze_button")
    
    if analyze_button:
        valid_matches = [m for m in matches if m['player1'] and m['player2']]
        
        if not valid_matches:
            st.warning("Veuillez remplir au moins un match")
            return
        
        st.markdown("---")
        st.markdown("## 📊 Résultats de l'analyse")
        
        matches_analysis = []
        all_selections = []
        
        for i, match in enumerate(valid_matches):
            st.markdown(f"### Match {i+1}: {match['player1']} vs {match['player2']}")
            st.caption(f"🏆 {match['tournament']} - {SURFACE_CONFIG[match['surface']]['icon']} {match['surface']}")
            
            h2h = get_h2h_stats(match['player1'], match['player2'])
            proba, ml_used = calculate_probability(match['player1'], match['player2'], 
                                                   match['surface'], h2h, model_info)
            confidence = calculate_confidence(proba, h2h)
            gagnant = match['player1'] if proba >= 0.5 else match['player2']
            
            st.markdown(f"### 🏆 **GAGNANT PRÉDIT: {gagnant}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(match['player1'], f"{proba:.1%}")
            with col2:
                st.metric(match['player2'], f"{1-proba:.1%}")
            
            st.progress(float(proba))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"{'🤖 ML' if ml_used else '📊 CSV'}")
            with col2:
                conf_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🔴"
                st.caption(f"Confiance: {conf_color} {confidence:.0f}/100")
            with col3:
                if h2h:
                    wins1 = h2h.get(f"{match['player1']}_wins", 0)
                    wins2 = h2h.get(f"{match['player2']}_wins", 0)
                    st.caption(f"H2H: {wins1}-{wins2}")
            
            best_value = None
            if match['odds1'] and match['odds2']:
                try:
                    o1 = float(match['odds1'].replace(',', '.'))
                    o2 = float(match['odds2'].replace(',', '.'))
                    edge1 = proba - 1/o1
                    edge2 = (1-proba) - 1/o2
                    if edge1 > MIN_EDGE_COMBINE:
                        best_value = {'joueur': match['player1'], 'edge': edge1, 'cote': o1, 'proba': proba}
                        st.success(f"🎯 Value bet! {match['player1']} @ {o1} (edge: +{edge1*100:.1f}%)")
                        all_selections.append(best_value)
                    elif edge2 > MIN_EDGE_COMBINE:
                        best_value = {'joueur': match['player2'], 'edge': edge2, 'cote': o2, 'proba': 1-proba}
                        st.success(f"🎯 Value bet! {match['player2']} @ {o2} (edge: +{edge2*100:.1f}%)")
                        all_selections.append(best_value)
                except Exception as e:
                    st.warning(f"Erreur de conversion des cotes: {e}")
            
            bet_suggestions = generate_alternative_bets(match['player1'], match['player2'], 
                                                        match['surface'], proba, h2h)
            
            if bet_suggestions:
                with st.expander("🎯 Paris alternatifs"):
                    for bet in bet_suggestions:
                        st.info(f"{bet['type']}: {bet['description']} (proba: {bet['proba']:.1%})")
            
            ai_comment = None
            if use_ai and get_groq_key():
                with st.spinner("Analyse IA..."):
                    ai_comment = analyze_match_with_ai(match['player1'], match['player2'], 
                                                      match['surface'], match['tournament'],
                                                      proba, best_value, bet_suggestions)
                    if ai_comment:
                        with st.expander("🤖 Analyse IA"):
                            st.write(ai_comment)
            
            pred_data = {
                'player1': match['player1'], 'player2': match['player2'],
                'tournament': match['tournament'], 'surface': match['surface'],
                'proba': float(proba), 'confidence': float(confidence),
                'odds1': match['odds1'], 'odds2': match['odds2'],
                'favori': gagnant,
                'best_value': best_value, 'ml_used': ml_used,
                'date': datetime.now().isoformat()
            }
            
            if save_prediction(pred_data):
                st.success("✅ Sauvegardé automatiquement")
            
            if send_tg:
                if send_prediction_to_telegram(pred_data, bet_suggestions, ai_comment):
                    st.success("📱 Envoyé sur Telegram")
                else:
                    st.warning("⚠️ Échec envoi Telegram (vérifie la configuration)")
            
            matches_analysis.append(pred_data)
            st.divider()
        
        new_badges = check_and_unlock_achievements()
        if new_badges:
            st.balloons()
            st.success("🏆 Nouveaux badges débloqués!")

def show_telegram():
    """Page Telegram améliorée avec commandes"""
    st.markdown("## 📱 Bot Telegram")
    
    token, chat_id = get_telegram_config()
    
    if not token or not chat_id:
        st.warning("⚠️ Telegram non configuré")
        st.markdown("""
        ### Configuration requise :
        
        1. Va sur Telegram @BotFather
        2. Crée un bot avec `/newbot`
        3. Copie le token fourni
        4. Ajoute dans les secrets Streamlit :
        ```toml
        TELEGRAM_BOT_TOKEN = "ton_token"
        TELEGRAM_CHAT_ID = "ton_chat_id"
        ```
        """)
        return
    
    st.success(f"✅ Telegram configuré (Chat ID: {chat_id})")
    
    if st.button("🔄 Configurer les commandes", use_container_width=True):
        setup_telegram_commands()
        st.success("✅ Commandes configurées!")
    
    if st.button("🔧 Tester la connexion", use_container_width=True):
        success, msg = test_telegram_connection()
        if success:
            st.success(msg)
        else:
            st.error(msg)
    
    st.markdown("### 📋 Commandes disponibles")
    commands_df = pd.DataFrame([
        ["/start", "Démarrer le bot"],
        ["/predict Djokovic Nadal Clay", "Prédire un match"],
        ["/stats", "Statistiques globales"],
        ["/today", "Matchs du jour"],
        ["/value", "Value bets du jour"],
        ["/badges", "Mes badges"],
        ["/help", "Aide"]
    ], columns=["Commande", "Description"])
    st.dataframe(commands_df, use_container_width=True, hide_index=True)
    
    st.markdown("### 🎮 Simulateur de commande")
    cmd = st.text_input("Tape une commande", "/stats")
    if st.button("Exécuter"):
        response = handle_telegram_command(cmd)
        st.markdown("**Réponse:**")
        st.markdown(response, unsafe_allow_html=True)
    
    send_custom_message()

def show_achievements():
    """Page des badges et achievements"""
    st.markdown("## 🏆 Badges et Achievements")
    
    achievements = load_achievements()
    stats = load_user_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Badges débloqués", f"{len(achievements)}/{len(ACHIEVEMENTS)}")
    with col2:
        progress = (len(achievements) / len(ACHIEVEMENTS) * 100) if ACHIEVEMENTS else 0
        st.progress(progress / 100)
    with col3:
        st.metric("Progression", f"{progress:.0f}%")
    
    st.markdown("### 📋 Liste des badges")
    
    cols = st.columns(2)
    for i, (ach_id, ach_data) in enumerate(ACHIEVEMENTS.items()):
        with cols[i % 2]:
            unlocked = ach_id in achievements
            bg_color = "rgba(0,223,162,0.1)" if unlocked else "rgba(108,124,137,0.1)"
            border = f"2px solid {COLORS['primary']}" if unlocked else "1px solid rgba(255,255,255,0.1)"
            
            st.markdown(f"""
            <div style="background: {bg_color}; border: {border}; border-radius: 10px; padding: 1rem; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="font-size: 2rem;">{ach_data['icon']}</div>
                    <div>
                        <div style="font-weight: bold; color: {'#00DFA2' if unlocked else '#6C7A89'};">{ach_data['name']}</div>
                        <div style="font-size: 0.9rem;">{ach_data['desc']}</div>
                        {f'<div style="font-size: 0.8rem; color: #6C7A89;">Débloqué le {datetime.fromisoformat(achievements[ach_id]["unlocked_at"]).strftime("%d/%m/%Y")}</div>' if unlocked else ''}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def show_configuration():
    """Page Configuration"""
    st.markdown("## ⚙️ Configuration")
    
    st.markdown("### 🤖 Modèle Machine Learning")
    model_info = load_saved_model()
    if model_info:
        st.success(f"✅ Modèle chargé avec succès (accuracy: {model_info.get('accuracy', 0):.1%})")
        if st.button("🔄 Recharger le modèle"):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.warning("⚠️ Aucun modèle trouvé dans le dossier models/ (mode stats CSV actif)")
    
    st.markdown("### 🧠 Intelligence Artificielle")
    if get_groq_key():
        st.success("✅ Clé API Groq configurée")
    else:
        st.warning("⚠️ Clé API Groq manquante")
    
    st.markdown("### 📊 Statistiques actuelles")
    stats = load_user_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total prédictions", stats.get('total_predictions', 0))
    with col2:
        accuracy = calculate_global_accuracy()
        st.metric("Précision", f"{accuracy:.1f}%")
    
    st.markdown("### 🗑️ Gestion des données")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Effacer l'historique"):
            if HIST_FILE.exists():
                HIST_FILE.unlink()
                update_user_stats()
                st.rerun()
    with col2:
        if st.button("🗑️ Effacer les badges"):
            if ACHIEVEMENTS_FILE.exists():
                ACHIEVEMENTS_FILE.unlink()
                st.rerun()
    with col3:
        if st.button("🔄 Recalculer les stats"):
            update_user_stats()
            st.rerun()
    
    st.markdown("### 💾 Backup manuel")
    if st.button("📀 Faire un backup maintenant"):
        auto_backup()
        st.success("✅ Backup effectué!")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="TennisIQ Pro - Tous les joueurs & tournois",
        page_icon="🎾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0A1E2C 0%, #1A2E3C 100%); }
        .stProgress > div > div > div > div { background: linear-gradient(90deg, #00DFA2, #0079FF); }
        .stButton > button { background: linear-gradient(90deg, #00DFA2, #0079FF); color: white; border: none; }
        .stButton > button:hover { background: linear-gradient(90deg, #00DFA2, #0079FF); opacity: 0.9; }
        h1, h2, h3 { color: #00DFA2; }
    </style>
    """, unsafe_allow_html=True)
    
    if 'last_backup' not in st.session_state:
        st.session_state['last_backup'] = datetime.now()
    
    if (datetime.now() - st.session_state['last_backup']).seconds >= 86400:
        auto_backup()
        st.session_state['last_backup'] = datetime.now()
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 2.5rem; font-weight: 800; color: #00DFA2;">
                TennisIQ
            </div>
            <div style="font-size: 0.9rem; color: #6C7A89;">
                Tous les joueurs • Tous les tournois
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🎯 Multi-matchs", "⏳ En Attente", "💎 Value Bets", "🏆 Badges", "📱 Telegram", "⚙️ Configuration"],
            label_visibility="collapsed",
            key="navigation"
        )
        
        st.divider()
        
        stats = load_user_stats()
        pending = len([p for p in load_history() if p.get('statut') == 'en_attente'])
        accuracy = calculate_global_accuracy()
        achievements = len(load_achievements())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Précision", f"{accuracy:.1f}%")
            st.metric("Badges", achievements)
        with col2:
            st.metric("En attente", pending)
            st.metric("Série", stats.get('current_streak', 0))
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🎯 Multi-matchs":
        show_prediction()
    elif page == "⏳ En Attente":
        show_pending()
    elif page == "💎 Value Bets":
        st.markdown("## 💎 Value Bets en direct")
        matches = get_daily_matches()
        value_bets = scan_for_value_bets(matches)
        
        if value_bets:
            for vb in value_bets:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    with col1:
                        st.markdown(f"**{vb['joueur']}**")
                        st.caption(vb['match'])
                    with col2:
                        st.metric("Cote", vb['cote'])
                    with col3:
                        st.metric("Edge", f"+{vb['edge']:.1f}%")
                    with col4:
                        st.metric("Proba", f"{vb['proba']:.1%}")
                    st.divider()
        else:
            st.info("Aucun value bet détecté pour le moment")
    
    elif page == "🏆 Badges":
        show_achievements()
    
    elif page == "📱 Telegram":
        show_telegram()
    
    elif page == "⚙️ Configuration":
        show_configuration()

if __name__ == "__main__":
    main()
