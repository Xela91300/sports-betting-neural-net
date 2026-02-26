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
from streamlit_option_menu import option_menu
import hashlib

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

# Base de données des tournois
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
# FONCTIONS DE BACKUP AUTOMATIQUE
# ─────────────────────────────────────────────────────────────
def auto_backup():
    """Backup automatique des données"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for file in [HIST_FILE, COMB_HIST_FILE, USER_STATS_FILE, ACHIEVEMENTS_FILE, TRENDS_FILE]:
        if file.exists():
            shutil.copy(file, BACKUP_DIR / f"{file.stem}_{timestamp}{file.suffix}")
    
    # Nettoyer les vieux backups (>30 jours)
    for backup in BACKUP_DIR.glob("*"):
        if (datetime.now() - datetime.fromtimestamp(backup.stat().st_mtime)).days > 30:
            backup.unlink()

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
    with open(ACHIEVEMENTS_FILE, 'w') as f:
        json.dump(achievements, f)

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
        # Envoyer notification Telegram pour les nouveaux achievements
        send_achievement_notification(new_unlocks)
    
    return new_unlocks

def send_achievement_notification(achievements):
    """Envoie une notification Telegram pour les nouveaux achievements"""
    message = "🏆 **NOUVEAU BADGE DÉBLOQUÉ !** 🏆\n\n"
    for ach in achievements:
        message += f"{ach['icon']} **{ach['name']}**\n{ach['desc']}\n\n"
    
    send_telegram_message(message)

# ─────────────────────────────────────────────────────────────
# TELEGRAM INTEGRATION AMÉLIORÉE
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
    requests.post(url, json={"commands": commands})

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
        
        proba, _ = calculate_probability(p1, p2, surface)
        pred_data = {
            'player1': p1, 'player2': p2,
            'surface': surface, 'proba': proba,
            'favori': p1 if proba >= 0.5 else p2
        }
        return format_prediction_message(pred_data)
    
    elif cmd == "/stats":
        return format_stats_message()
    
    elif cmd == "/today":
        return format_today_matches()
    
    elif cmd == "/value":
        return scan_and_format_value_bets()
    
    elif cmd == "/badges":
        return format_badges_message()
    
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

def format_today_matches():
    """Formate les matchs du jour pour Telegram"""
    matches = scrape_daily_matches()
    if not matches:
        return "📅 Aucun match trouvé aujourd'hui"
    
    message = "<b>📅 MATCHS DU JOUR</b>\n\n"
    for i, match in enumerate(matches[:10], 1):
        message += f"{i}. {match['p1']} vs {match['p2']} - {match['surface']}\n"
    
    return message

def format_badges_message():
    """Formate les badges pour Telegram"""
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

# ─────────────────────────────────────────────────────────────
# SCRAPING AUTOMATIQUE DES MATCHS
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def scrape_daily_matches():
    """Simule la récupération des matchs du jour (à remplacer par vrai scraping)"""
    # Simulation pour l'exemple
    today_matches = [
        {'p1': 'Novak Djokovic', 'p2': 'Carlos Alcaraz', 'surface': 'Clay', 'tournament': 'Roland Garros'},
        {'p1': 'Jannik Sinner', 'p2': 'Daniil Medvedev', 'surface': 'Hard', 'tournament': 'Miami Open'},
        {'p1': 'Rafael Nadal', 'p2': 'Stefanos Tsitsipas', 'surface': 'Clay', 'tournament': 'Barcelona Open'},
        {'p1': 'Alexander Zverev', 'p2': 'Andrey Rublev', 'surface': 'Hard', 'tournament': 'Madrid Open'},
        {'p1': 'Holger Rune', 'p2': 'Casper Ruud', 'surface': 'Grass', 'tournament': 'Wimbledon'},
    ]
    return today_matches

def auto_load_today_matches():
    """Bouton pour charger automatiquement les matchs du jour"""
    if st.button("📅 Charger les matchs du jour", use_container_width=True):
        with st.spinner("Récupération des matchs..."):
            matches = scrape_daily_matches()
            if matches:
                st.session_state['today_matches'] = matches
                st.success(f"✅ {len(matches)} matchs chargés")
                return matches
    return None

# ─────────────────────────────────────────────────────────────
# DÉTECTION VALUE BETS EN TEMPS RÉEL
# ─────────────────────────────────────────────────────────────
def scan_for_value_bets(matches):
    """Scanne une liste de matchs pour trouver des value bets"""
    value_bets = []
    
    for match in matches:
        proba, _ = calculate_probability(match['p1'], match['p2'], match['surface'])
        
        # Simuler des cotes (à remplacer par des cotes réelles)
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
                'surface': match['surface']
            })
        elif edge2 > MIN_EDGE_COMBINE:
            value_bets.append({
                'match': f"{match['p1']} vs {match['p2']}",
                'joueur': match['p2'],
                'edge': edge2 * 100,
                'cote': odds2,
                'proba': 1-proba,
                'surface': match['surface']
            })
    
    return sorted(value_bets, key=lambda x: x['edge'], reverse=True)

def scan_and_format_value_bets():
    """Formate les value bets pour Telegram"""
    matches = scrape_daily_matches()
    value_bets = scan_for_value_bets(matches)
    
    if not value_bets:
        return "🎯 Aucun value bet détecté aujourd'hui"
    
    message = "<b>🎯 VALUE BETS DU JOUR</b>\n\n"
    for i, vb in enumerate(value_bets[:5], 1):
        message += f"{i}. <b>{vb['joueur']}</b> @ {vb['cote']}\n"
        message += f"   Edge: +{vb['edge']:.1f}% | Proba: {vb['proba']:.1%}\n"
        message += f"   {vb['match']} - {vb['surface']}\n\n"
    
    return message

# ─────────────────────────────────────────────────────────────
# DASHBOARD INTERACTIF AVEC PLOTLY
# ─────────────────────────────────────────────────────────────
def create_interactive_dashboard():
    """Crée un dashboard interactif avec Plotly"""
    st.markdown("## 📊 Dashboard Interactif")
    
    history = load_history()
    if not history:
        st.info("Pas assez de données pour le dashboard")
        return
    
    df = pd.DataFrame(history)
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
        template='plotly_dark'
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
            template='plotly_dark'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # 3. Heatmap des performances
    st.markdown("### 🔥 Heatmap des performances")
    
    # Créer une matrice de performance par mois/surface
    df['month'] = df['date'].dt.strftime('%Y-%m')
    pivot = pd.pivot_table(
        df[df['statut'].isin(['gagne', 'perdu'])],
        values='correct',
        index='month',
        columns='surface',
        aggfunc='mean'
    ).fillna(0) * 100
    
    if not pivot.empty:
        fig3 = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Viridis',
            text=np.round(pivot.values, 1),
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig3.update_layout(
            title='Performance mensuelle par surface (%)',
            xaxis_title='Surface',
            yaxis_title='Mois',
            template='plotly_dark'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # 4. Distribution des confiances
    st.markdown("### 📊 Distribution des confiances")
    
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(
        x=df['confidence'].dropna(),
        nbinsx=20,
        marker_color=COLORS['primary'],
        opacity=0.7,
        name='Confiance'
    ))
    
    fig4.update_layout(
        title='Distribution des scores de confiance',
        xaxis_title='Confiance',
        yaxis_title='Nombre de prédictions',
        template='plotly_dark'
    )
    st.plotly_chart(fig4, use_container_width=True)

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
    
    # Sauvegarder les tendances
    with open(TRENDS_FILE, 'w') as f:
        json.dump(trends, f, indent=2)
    
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
# (Le reste du code existant - load_saved_model, predict_with_ml_model, 
# load_atp_data, get_all_players, get_h2h_stats, calculate_probability,
# calculate_confidence, calculate_global_accuracy, generate_alternative_bets,
# generate_recommended_combines, load_history, save_prediction,
# update_prediction_status, load_user_stats, update_user_stats,
# load_combines, save_combine, player_selector, etc.)
# ─────────────────────────────────────────────────────────────

# [INSÉRER ICI TOUTES LES FONCTIONS EXISTANTES DE LA VERSION PRÉCÉDENTE]
# Pour garder ce message raisonnable, je n'inclus pas les ~1000 lignes existantes
# Mais elles doivent être présentes dans le fichier final

# ─────────────────────────────────────────────────────────────
# PAGE DASHBOARD AMÉLIORÉE
# ─────────────────────────────────────────────────────────────
def show_dashboard():
    """Page Dashboard améliorée avec tendances et badges"""
    st.markdown("## 🏠 Dashboard")
    
    # Métriques principales
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
    
    # Badges récents
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
    
    # Conseils de paris
    if trends:
        st.markdown("### 💡 Conseils personnalisés")
        advice = generate_betting_advice()
        st.info(advice)
    
    # Dashboard interactif
    create_interactive_dashboard()
    
    # Statut des services
    model_info = load_saved_model()
    groq_key = get_groq_key()
    telegram_token, _ = get_telegram_config()
    
    st.markdown("### 🛠️ Statut des services")
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
    """Page de prédiction avec scraping automatique et value bets"""
    st.markdown("## 🎯 Analyse Multi-matchs")
    
    model_info = load_saved_model()
    
    with st.spinner("Chargement des joueurs..."):
        all_players = load_atp_data()
    
    st.success(f"✅ {len(all_players):,} joueurs disponibles")
    
    # Option de chargement automatique
    col1, col2 = st.columns([1, 3])
    with col1:
        auto_load = st.checkbox("📅 Auto-load matchs", value=True)
    with col2:
        if auto_load:
            today_matches = auto_load_today_matches()
    
    # Configuration
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_matches = st.number_input("Nombre de matchs", 1, MAX_MATCHES_ANALYSIS, 
                                    value=len(st.session_state.get('today_matches', [3])))
    with col2:
        mise = st.number_input("Mise (€)", 1.0, 1000.0, 10.0)
    with col3:
        use_ai = st.checkbox("🤖 Analyser avec IA", True)
    with col4:
        send_tg = st.checkbox("📱 Envoyer Telegram", True)
    
    # Saisie des matchs
    matches = []
    st.markdown("### 📝 Saisie des matchs")
    
    tournaments_list = sorted(TOURNAMENTS_DB.keys())
    
    # Pré-remplir avec les matchs du jour si disponibles
    today_matches = st.session_state.get('today_matches', [])
    
    for i in range(n_matches):
        with st.expander(f"Match {i+1}", expanded=i==0):
            default_p1 = today_matches[i]['p1'] if i < len(today_matches) else None
            default_p2 = today_matches[i]['p2'] if i < len(today_matches) else None
            default_tournament = today_matches[i]['tournament'] if i < len(today_matches) else tournaments_list[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                p1 = player_selector(f"Joueur 1", all_players, key=f"p1_{i}", 
                                     default=default_p1 if default_p1 else "Novak Djokovic")
                odds1 = st.text_input(f"Cote {p1}", key=f"odds1_{i}", placeholder="1.75")
            
            with col2:
                if p1:
                    players2 = [p for p in all_players if p != p1]
                    p2 = player_selector(f"Joueur 2", players2, key=f"p2_{i}",
                                         default=default_p2 if default_p2 else "Carlos Alcaraz")
                else:
                    p2 = player_selector(f"Joueur 2", all_players, key=f"p2_{i}")
                odds2 = st.text_input(f"Cote {p2}", key=f"odds2_{i}", placeholder="2.10")
            
            col1, col2 = st.columns(2)
            with col1:
                tournament = st.selectbox(f"Tournoi", tournaments_list, 
                                         index=tournaments_list.index(default_tournament) if default_tournament in tournaments_list else 0,
                                         key=f"tourn_{i}")
                surface = TOURNAMENTS_DB[tournament]
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
    
    # Scan des value bets avant analyse
    if matches:
        valid_matches = [{'p1': m['player1'], 'p2': m['player2'], 'surface': m['surface']} 
                        for m in matches if m['player1'] and m['player2']]
        if valid_matches:
            value_bets = scan_for_value_bets(valid_matches)
            if value_bets:
                st.markdown("### 🎯 Value bets détectés")
                for vb in value_bets[:3]:
                    st.success(f"{vb['joueur']} @ {vb['cote']} (edge: +{vb['edge']:.1f}%)")
    
    if st.button("🔍 Analyser tous les matchs", type="primary", use_container_width=True):
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
                st.caption(f"{'🤖 ML' if ml_used else '📊 Stats'}")
            with col2:
                conf_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🔴"
                st.caption(f"Confiance: {conf_color} {confidence:.0f}/100")
            with col3:
                if h2h:
                    wins1 = h2h.get(f"{match['player1']}_wins", 0)
                    wins2 = h2h.get(f"{match['player2']}_wins", 0)
                    st.caption(f"H2H: {wins1}-{wins2}")
            
            # Value bet
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
                except:
                    pass
            
            # Paris alternatifs
            bet_suggestions = generate_alternative_bets(match['player1'], match['player2'], 
                                                        match['surface'], proba, h2h)
            
            if bet_suggestions:
                with st.expander("🎯 Paris alternatifs"):
                    for bet in bet_suggestions:
                        st.info(f"{bet['type']}: {bet['description']} (proba: {bet['proba']:.1%})")
            
            # Analyse IA
            ai_comment = None
            if use_ai and get_groq_key():
                with st.spinner("Analyse IA..."):
                    ai_comment = analyze_match_with_ai(match['player1'], match['player2'], 
                                                      match['surface'], match['tournament'],
                                                      proba, best_value, bet_suggestions)
                    if ai_comment:
                        with st.expander("🤖 Analyse IA"):
                            st.write(ai_comment)
            
            # Préparation et sauvegarde
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
            
            matches_analysis.append(pred_data)
            st.divider()
        
        # Vérifier les nouveaux badges
        new_badges = check_and_unlock_achievements()
        if new_badges:
            st.balloons()
            st.success("🏆 Nouveaux badges débloqués!")
        
        # Génération de combinés
        if len(all_selections) >= 2:
            st.markdown("## 🎰 Combinés recommandés")
            
            suggestions = generate_recommended_combines(matches_analysis)
            
            for idx, suggestion in enumerate(suggestions):
                with st.expander(f"{suggestion['name']} - {suggestion['nb_matches']} matchs - Proba {suggestion['proba']:.1%}"):
                    
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
                        'gain_potentiel': gain,
                        'esperance': suggestion['proba'] * gain - mise,
                        'nb_matches': suggestion['nb_matches'],
                        'ml_used': any(m.get('ml_used', False) for m in matches_analysis)
                    }
                    
                    save_combine(combine_data)
                    
                    if st.button(f"📱 Envoyer combiné", key=f"tg_comb_{idx}"):
                        if send_combine_to_telegram(combine_data):
                            st.success("✅ Combiné envoyé!")

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
        3. Ajoute dans les secrets Streamlit :
        ```toml
        TELEGRAM_BOT_TOKEN = "ton_token"
        TELEGRAM_CHAT_ID = "ton_chat_id"
        ```
        """)
        return
    
    st.success(f"✅ Telegram configuré (Chat ID: {chat_id})")
    
    # Configuration des commandes
    if st.button("🔄 Configurer les commandes", use_container_width=True):
        setup_telegram_commands()
        st.success("✅ Commandes configurées!")
    
    # Test de connexion
    if st.button("🔧 Tester la connexion", use_container_width=True):
        success, msg = test_telegram_connection()
        if success:
            st.success(msg)
        else:
            st.error(msg)
    
    # Commandes disponibles
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
    
    # Simulateur de commande
    st.markdown("### 🎮 Simulateur de commande")
    cmd = st.text_input("Tape une commande", "/stats")
    if st.button("Exécuter"):
        response = handle_telegram_command(cmd)
        st.markdown("**Réponse:**")
        st.markdown(response, unsafe_allow_html=True)
    
    # Message personnalisé
    send_custom_message()

def show_achievements():
    """Page des badges et achievements"""
    st.markdown("## 🏆 Badges et Achievements")
    
    achievements = load_achievements()
    stats = load_user_stats()
    
    # Progression globale
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Badges débloqués", f"{len(achievements)}/{len(ACHIEVEMENTS)}")
    with col2:
        progress = (len(achievements) / len(ACHIEVEMENTS) * 100)
        st.progress(progress / 100)
    with col3:
        st.metric("Progression", f"{progress:.0f}%")
    
    # Liste des badges
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

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="TennisIQ Pro - Ultimate Edition",
        page_icon="🎾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0A1E2C 0%, #1A2E3C 100%); }
        .stProgress > div > div > div > div { background: linear-gradient(90deg, #00DFA2, #0079FF); }
        .stButton > button { background: linear-gradient(90deg, #00DFA2, #0079FF); color: white; border: none; }
        .stButton > button:hover { background: linear-gradient(90deg, #00DFA2, #0079FF); opacity: 0.9; }
        h1, h2, h3 { color: #00DFA2; }
    </style>
    """, unsafe_allow_html=True)
    
    # Backup automatique (toutes les 24h)
    if 'last_backup' not in st.session_state:
        st.session_state['last_backup'] = datetime.now()
    
    if (datetime.now() - st.session_state['last_backup']).hours >= 24:
        auto_backup()
        st.session_state['last_backup'] = datetime.now()
    
    # Sidebar avec menu amélioré
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 2.5rem; font-weight: 800; color: #00DFA2;">
                TennisIQ
            </div>
            <div style="font-size: 0.9rem; color: #6C7A89;">
                Ultimate Edition
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Menu avec icônes
        page = option_menu(
            menu_title=None,
            options=["Dashboard", "Multi-matchs", "Value Bets", "Badges", "Telegram", "Configuration"],
            icons=["house", "trophy", "gem", "award", "telegram", "gear"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#00DFA2", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "#fff"},
                "nav-link-selected": {"background-color": "#00DFA2", "color": "#000"},
            }
        )
        
        st.divider()
        
        # Stats rapides
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
    
    # Navigation
    if page == "Dashboard":
        show_dashboard()
    elif page == "Multi-matchs":
        show_prediction()
    elif page == "Value Bets":
        st.markdown("## 💎 Value Bets en direct")
        matches = scrape_daily_matches()
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
    
    elif page == "Badges":
        show_achievements()
    
    elif page == "Telegram":
        show_telegram()
    
    elif page == "Configuration":
        show_configuration()

if __name__ == "__main__":
    main()
