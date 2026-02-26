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
# TELEGRAM INTEGRATION COMPLÈTE
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
    """Formate un message de prédiction pour Telegram"""
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
            "max_tokens": 300
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return None
    except:
        return None

def analyze_match_with_ai(player1, player2, surface, tournament, proba, best_value=None, bet_suggestions=None):
    """Génère une analyse IA pour un match"""
    vb_txt = f" Value bet sur {best_value['joueur']} (edge {best_value['edge']*100:+.1f}%)" if best_value else ""
    
    prompt = f"""Analyse ce match de tennis en 3 points clés:
    {player1} vs {player2}
    Tournoi: {tournament}
    Surface: {surface}
    Probabilités: {player1} {proba:.1%} - {player2} {1-proba:.1%}
    {vb_txt}
    
    Donne une analyse concise en français."""
    
    return call_groq_api(prompt)

def analyze_combine_with_ai(selections, proba_globale, cote_globale, esperance):
    """Génère une analyse IA pour un combiné"""
    selections_txt = "\n".join([f"- {s['joueur']} @ {s['cote']:.2f} (edge: {s['edge']*100:+.1f}%)" for s in selections[:5]])
    
    prompt = f"""Analyse ce combiné de tennis:
    {len(selections)} sélections:
    {selections_txt}
    
    Probabilité globale: {proba_globale:.1%}
    Cote combinée: {cote_globale:.2f}
    Espérance: {esperance:+.2f}€
    
    Donne un avis concis sur la pertinence de ce combiné."""
    
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
        except:
            return None
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
    except:
        return None

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES ATP
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_atp_data():
    """Charge les données ATP depuis le dossier data/"""
    if not DATA_DIR.exists():
        return pd.DataFrame()
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()
    
    atp_dfs = []
    for f in csv_files[:10]:
        if 'wta' in f.name.lower():
            continue
        try:
            df = pd.read_csv(f, encoding='utf-8', nrows=5000, on_bad_lines='skip')
            if 'winner_name' in df.columns and 'loser_name' in df.columns:
                df['winner_name'] = df['winner_name'].astype(str).str.strip()
                df['loser_name'] = df['loser_name'].astype(str).str.strip()
                atp_dfs.append(df[['winner_name', 'loser_name']])
        except:
            continue
    
    if atp_dfs:
        return pd.concat(atp_dfs, ignore_index=True)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_all_players(_df):
    """Récupère la liste de tous les joueurs"""
    if _df.empty:
        return ["Novak Djokovic", "Rafael Nadal", "Roger Federer", "Carlos Alcaraz"]
    
    players = set()
    players.update(_df['winner_name'].dropna().unique())
    players.update(_df['loser_name'].dropna().unique())
    
    valid_players = []
    for p in players:
        p_str = str(p).strip()
        if p_str and p_str.lower() != 'nan' and len(p_str) > 1:
            valid_players.append(p_str)
    
    return sorted(valid_players)[:1000]

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
    """Calcule la probabilité"""
    if model_info:
        ml_proba = predict_with_ml_model(model_info, player1, player2, surface)
        if ml_proba is not None:
            return ml_proba, True
    
    proba = 0.5
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
# FONCTIONS POUR LES COMBINÉS RECOMMANDÉS
# ─────────────────────────────────────────────────────────────
def generate_recommended_combines(matches_analysis):
    """Génère des combinés recommandés à partir des matchs analysés"""
    if len(matches_analysis) < 2:
        return []
    
    matches_with_edge = [m for m in matches_analysis if m.get('best_value')]
    matches_with_edge.sort(key=lambda x: x['best_value']['edge'], reverse=True)
    
    suggestions = []
    
    # Suggestion 1: Top value bets
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
    
    # Suggestion 2: Haute confiance
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
    
    # Suggestion 3: Combiné équilibré
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
    try:
        history = load_history()
        pred_data['id'] = hashlib.md5(f"{datetime.now()}{pred_data.get('player1','')}".encode()).hexdigest()[:8]
        pred_data['statut'] = 'en_attente'
        history.append(pred_data)
        with open(HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(history[-1000:], f, indent=2)
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
            json.dump(combines[-200:], f, indent=2)
        return True
    except:
        return False

# ─────────────────────────────────────────────────────────────
# PAGES DE L'APPLICATION
# ─────────────────────────────────────────────────────────────

def show_dashboard():
    """Page Dashboard"""
    st.markdown("## 🏠 Dashboard")
    
    stats = load_user_stats()
    history = load_history()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total prédictions", stats.get('total_predictions', 0))
    with col2:
        accuracy = calculate_global_accuracy()
        st.metric("Précision", f"{accuracy:.1f}%")
    with col3:
        st.metric("En attente", len([p for p in history if p.get('statut') == 'en_attente']))
    with col4:
        st.metric("Série actuelle", stats.get('current_streak', 0))
    
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
    """Page de prédiction avec analyse multi-matchs (jusqu'à 30) et génération de combinés"""
    st.markdown("## 🎯 Analyse Multi-matchs (max 30)")
    
    model_info = load_saved_model()
    
    with st.spinner("Chargement des joueurs..."):
        atp_data = load_atp_data()
        all_players = get_all_players(atp_data)
    
    st.success(f"✅ {len(all_players)} joueurs disponibles")
    
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
    
    # Saisie des matchs
    matches = []
    st.markdown("### 📝 Saisie des matchs")
    
    tournaments_list = sorted(TOURNAMENTS_DB.keys())
    
    for i in range(n_matches):
        with st.expander(f"Match {i+1}", expanded=i==0):
            col1, col2 = st.columns(2)
            
            with col1:
                p1 = st.selectbox(f"Joueur 1", all_players, key=f"p1_{i}")
                odds1 = st.text_input(f"Cote {p1}", key=f"odds1_{i}", placeholder="1.75")
            
            with col2:
                players2 = [p for p in all_players if p != p1]
                p2 = st.selectbox(f"Joueur 2", players2, key=f"p2_{i}")
                odds2 = st.text_input(f"Cote {p2}", key=f"odds2_{i}", placeholder="2.10")
            
            col1, col2 = st.columns(2)
            with col1:
                tournament = st.selectbox(f"Tournoi", tournaments_list, key=f"tourn_{i}")
                surface = TOURNAMENTS_DB[tournament]
            with col2:
                st.info(f"Surface: {SURFACE_CONFIG[surface]['icon']} {surface}")
            
            matches.append({
                'player1': p1, 'player2': p2,
                'surface': surface, 'tournament': tournament,
                'odds1': odds1, 'odds2': odds2,
                'index': i
            })
    
    if st.button("🔍 Analyser tous les matchs", type="primary", use_container_width=True):
        valid_matches = [m for m in matches if m['player1'] and m['player2']]
        
        if not valid_matches:
            st.warning("Veuillez remplir au moins un match")
            return
        
        st.markdown("---")
        st.markdown("## 📊 Résultats de l'analyse")
        
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
                    elif edge2 > MIN_EDGE_COMBINE:
                        best_value = {'joueur': match['player2'], 'edge': edge2, 'cote': o2, 'proba': 1-proba}
                except:
                    pass
            
            # Paris alternatifs
            bet_suggestions = generate_alternative_bets(match['player1'], match['player2'], 
                                                        match['surface'], proba, h2h)
            
            # Affichage
            st.markdown(f"#### 🏆 Gagnant: **{match['player1'] if proba >= 0.5 else match['player2']}**")
            
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
                    st.caption(f"H2H: {h2h.get(f'{match['player1']}_wins', 0)}-{h2h.get(f'{match['player2']}_wins', 0)}")
            
            if best_value:
                st.success(f"🎯 Value bet! {best_value['joueur']} @ {best_value['cote']:.2f} (edge: {best_value['edge']*100:+.1f}%)")
                all_selections.append(best_value)
            
            if bet_suggestions:
                st.markdown("#### 🎯 Paris alternatifs")
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
            
            # Préparation données
            pred_data = {
                'player1': match['player1'], 'player2': match['player2'],
                'tournament': match['tournament'], 'surface': match['surface'],
                'proba': float(proba), 'confidence': float(confidence),
                'odds1': match['odds1'], 'odds2': match['odds2'],
                'favori': match['player1'] if proba >= 0.5 else match['player2'],
                'best_value': best_value, 'ml_used': ml_used,
                'date': datetime.now().isoformat()
            }
            
            matches_analysis.append(pred_data)
            
            # Boutons individuels
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"💾 Sauvegarder match {i+1}", key=f"save_{i}"):
                    if save_prediction(pred_data):
                        st.success("✅ Sauvegardé!")
            with col2:
                if st.button(f"📱 Envoyer match {i+1}", key=f"tg_{i}"):
                    if send_prediction_to_telegram(pred_data, bet_suggestions, ai_comment):
                        st.success("✅ Envoyé!")
            
            st.divider()
        
        # Génération de combinés
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
                    
                    # Analyse IA du combiné
                    combine_ai = None
                    if use_ai and get_groq_key():
                        combine_ai = analyze_combine_with_ai(suggestion['selections'], 
                                                            suggestion['proba'], 
                                                            suggestion['cote'],
                                                            suggestion['proba'] * gain - mise)
                    
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
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"💾 Sauvegarder combiné", key=f"save_comb_{idx}"):
                            save_combine(combine_data)
                            st.success("✅ Combiné sauvegardé!")
                    with col2:
                        if st.button(f"📱 Envoyer combiné", key=f"tg_comb_{idx}"):
                            if send_combine_to_telegram(combine_data, combine_ai):
                                st.success("✅ Combiné envoyé!")
                    
                    if combine_ai:
                        with st.expander("🤖 Analyse IA du combiné"):
                            st.write(combine_ai)
        
        # Envoi groupé
        if send_tg and matches_analysis:
            st.markdown("### 📤 Envoi groupé")
            if st.button("📤 Envoyer tous les matchs sur Telegram", use_container_width=True):
                success = 0
                for pred in matches_analysis:
                    if send_prediction_to_telegram(pred):
                        success += 1
                st.success(f"✅ {success}/{len(matches_analysis)} matchs envoyés!")

def show_pending():
    """Page des prédictions en attente"""
    st.markdown("## ⏳ En attente")
    
    history = load_history()
    pending = [p for p in history if p.get('statut') == 'en_attente']
    
    if not pending:
        st.info("Aucune prédiction en attente")
        return
    
    for pred in pending[::-1]:
        with st.expander(f"{pred.get('date', '')[:16]} - {pred['player1']} vs {pred['player2']}"):
            st.write(f"Tournoi: {pred.get('tournament')}")
            st.write(f"Surface: {pred.get('surface')}")
            st.write(f"Probabilité: {pred.get('proba', 0.5):.1%}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"✅ {pred['player1']} gagne", key=f"w1_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'gagne')
                    st.rerun()
            with col2:
                if st.button(f"✅ {pred['player2']} gagne", key=f"w2_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'gagne')
                    st.rerun()
            with col3:
                if st.button(f"❌ Perdu", key=f"l_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'perdu')
                    st.rerun()
            
            if st.button(f"⚠️ Annuler", key=f"c_{pred['id']}", use_container_width=True):
                update_prediction_status(pred['id'], 'annule')
                st.rerun()

def show_history():
    """Page Historique"""
    st.markdown("## 📜 Historique")
    
    history = load_history()
    if not history:
        st.info("Aucune prédiction")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.multiselect(
            "Filtrer",
            options=list(STATUS_OPTIONS.keys()),
            format_func=lambda x: STATUS_OPTIONS[x],
            default=list(STATUS_OPTIONS.keys())
        )
    with col2:
        search = st.text_input("🔍 Rechercher", "")
    
    filtered = [p for p in history if p.get('statut') in status_filter]
    if search:
        filtered = [p for p in filtered if 
                   search.lower() in p.get('player1', '').lower() or 
                   search.lower() in p.get('player2', '').lower()]
    
    for pred in filtered[::-1][:50]:
        status_icon = STATUS_OPTIONS.get(pred.get('statut'), "⏳")
        with st.expander(f"{status_icon} {pred.get('date', '')[:16]} - {pred['player1']} vs {pred['player2']}"):
            st.write(f"Tournoi: {pred.get('tournament')}")
            st.write(f"Probabilité: {pred.get('proba', 0.5):.1%}")
            st.write(f"Statut: {STATUS_OPTIONS.get(pred.get('statut'), 'Inconnu')}")

def show_statistics():
    """Page Statistiques"""
    st.markdown("## 📈 Statistiques")
    
    stats = load_user_stats()
    history = load_history()
    combines = load_combines()
    
    total = stats.get('total_predictions', 0)
    correct = stats.get('correct_predictions', 0)
    incorrect = stats.get('incorrect_predictions', 0)
    annules = stats.get('annules_predictions', 0)
    
    total_valide = correct + incorrect
    accuracy = (correct / total_valide * 100) if total_valide > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", total)
    with col2:
        st.metric("✅ Gagnées", correct, f"{accuracy:.1f}%")
    with col3:
        st.metric("❌ Perdues", incorrect)
    with col4:
        st.metric("⚠️ Annulées", annules)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Série actuelle", stats.get('current_streak', 0))
    with col2:
        st.metric("Combinés", len(combines))
    
    if st.button("📊 Envoyer stats sur Telegram", use_container_width=True):
        if send_stats_to_telegram():
            st.success("✅ Stats envoyées!")

def show_telegram():
    """Page Telegram"""
    st.markdown("## 📱 Telegram")
    
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
    
    if st.button("🔧 Tester la connexion", use_container_width=True):
        success, msg = test_telegram_connection()
        if success:
            st.success(msg)
        else:
            st.error(msg)
    
    if st.button("📊 Envoyer les stats", use_container_width=True):
        if send_stats_to_telegram():
            st.success("✅ Stats envoyées!")
    
    send_custom_message()

def show_configuration():
    """Page Configuration"""
    st.markdown("## ⚙️ Configuration")
    
    st.markdown("### 🤖 Modèle ML")
    model_info = load_saved_model()
    if model_info:
        st.success(f"✅ Modèle chargé (accuracy: {model_info.get('accuracy', 0):.1%})")
    else:
        st.warning("⚠️ Aucun modèle trouvé")
    
    st.markdown("### 🧠 IA Groq")
    if get_groq_key():
        st.success("✅ Clé API configurée")
    else:
        st.warning("⚠️ Clé API manquante")
    
    st.markdown("### 🗑️ Gestion")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Effacer historique"):
            if HIST_FILE.exists():
                HIST_FILE.unlink()
                update_user_stats()
                st.rerun()
    with col2:
        if st.button("🔄 Recalculer stats"):
            update_user_stats()
            st.rerun()

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="TennisIQ Pro - Multi-matchs & Combinés",
        page_icon="🎾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0A1E2C 0%, #1A2E3C 100%); }
        .stProgress > div > div > div > div { background: linear-gradient(90deg, #00DFA2, #0079FF); }
        .stButton > button { background: linear-gradient(90deg, #00DFA2, #0079FF); color: white; border: none; }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 2rem; font-weight: 800; color: #00DFA2;">
                TennisIQ
            </div>
            <div style="font-size: 0.8rem; color: #6C7A89;">
                Multi-matchs & Combinés
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🎯 Multi-matchs", "⏳ En Attente", 
             "📜 Historique", "📈 Statistiques", "📱 Telegram", "⚙️ Configuration"],
            label_visibility="collapsed"
        )
        
        st.divider()
        stats = load_user_stats()
        pending = len([p for p in load_history() if p.get('statut') == 'en_attente'])
        accuracy = calculate_global_accuracy()
        
        st.caption(f"📊 Précision: {accuracy:.1f}%")
        st.caption(f"⏳ En attente: {pending}")
        st.caption(f"🔥 Série: {stats.get('current_streak', 0)}")
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🎯 Multi-matchs":
        show_prediction()
    elif page == "⏳ En Attente":
        show_pending()
    elif page == "📜 Historique":
        show_history()
    elif page == "📈 Statistiques":
        show_statistics()
    elif page == "📱 Telegram":
        show_telegram()
    elif page == "⚙️ Configuration":
        show_configuration()

if __name__ == "__main__":
    main()
