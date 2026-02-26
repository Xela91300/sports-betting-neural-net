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
    
    if bet_suggestions:
        message += f"\n<b>🎯 PARIS ALTERNATIFS:</b>\n"
        for bet in bet_suggestions[:3]:
            conf_icon = '🟢' if bet['confidence'] >= 70 else '🟡' if bet['confidence'] >= 50 else '🔴'
            message += f"\n{conf_icon} <b>{bet['type']}</b>: {bet['description']}\n"
            message += f"   Probabilité: {bet['proba']:.1%} | Cote: {bet['cote']:.2f}\n"
    
    if pred_data.get('best_value'):
        bv = pred_data['best_value']
        message += f"""
<b>🎯 VALUE BET DÉTECTÉ!</b>
<b>{bv['joueur']}</b> à <b>{bv['cote']:.2f}</b>
Edge: <b>{bv['edge']*100:+.1f}%</b>
"""
    
    if ai_comment:
        clean_comment = ai_comment.replace('<', '&lt;').replace('>', '&gt;')
        message += f"\n\n<b>🤖 ANALYSE IA:</b>\n{clean_comment}"
    
    message += f"\n\n#TennisIQ #{pred_data.get('surface', 'Tennis')}"
    return message

def send_prediction_to_telegram(pred_data, bet_suggestions=None, ai_comment=None):
    return send_telegram_message(format_prediction_message(pred_data, bet_suggestions, ai_comment))

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
    
    message = f"""
<b>📊 STATISTIQUES TENNISIQ</b>

<b>🎯 Performance globale:</b>
{bar}  {accuracy:.1f}%

<b>📈 Détail:</b>
• Total: <b>{total}</b>
• ✅ Gagnées: <b>{correct}</b> ({accuracy:.1f}%)
• ❌ Perdues: <b>{incorrect}</b>
• ⚠️ Annulées: <b>{annules}</b>

<b>🔥 Dernières 20:</b>
• Précision: <b>{recent_acc:.1f}%</b>

<b>🏆 Records:</b>
• Série: <b>{stats.get('current_streak', 0)}</b>
• Meilleure: <b>{stats.get('best_streak', 0)}</b>

#TennisIQ #Stats
"""
    return message

def send_stats_to_telegram():
    return send_telegram_message(format_stats_message())

def test_telegram_connection():
    token, chat_id = get_telegram_config()
    if not token:
        return False, "❌ Token manquant"
    if not chat_id:
        return False, "❌ Chat ID manquant"
    try:
        test_message = f"<b>✅ Test réussi !</b>\n\n📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {'chat_id': chat_id, 'text': test_message, 'parse_mode': 'HTML'}
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            return True, "✅ Connexion réussie !"
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
        
        # Features simplifiées
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
    for f in csv_files[:10]:  # Limiter pour la vitesse
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
    
    # Filtrer les valeurs invalides
    valid_players = []
    for p in players:
        p_str = str(p).strip()
        if p_str and p_str.lower() != 'nan' and len(p_str) > 1:
            valid_players.append(p_str)
    
    return sorted(valid_players)[:1000]  # Limiter à 1000 pour la performance

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
    try:
        history = load_history()
        pred_data['id'] = hashlib.md5(f"{datetime.now()}".encode()).hexdigest()[:8]
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
            'total_predictions': 0, 'correct_predictions': 0,
            'incorrect_predictions': 0, 'annules_predictions': 0,
            'current_streak': 0, 'best_streak': 0
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
    
    stats = {
        'total_predictions': len(history),
        'correct_predictions': correct,
        'incorrect_predictions': incorrect,
        'annules_predictions': sum(1 for p in history if p.get('statut') == 'annule'),
        'current_streak': 0,
        'best_streak': 0
    }
    
    with open(USER_STATS_FILE, 'w') as f:
        json.dump(stats, f)
    return stats

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
        st.metric("Total", stats.get('total_predictions', 0))
    with col2:
        total_valide = stats.get('correct_predictions', 0) + stats.get('incorrect_predictions', 0)
        accuracy = (stats.get('correct_predictions', 0) / total_valide * 100) if total_valide > 0 else 0
        st.metric("Précision", f"{accuracy:.1f}%")
    with col3:
        st.metric("En attente", len([p for p in history if p.get('statut') == 'en_attente']))
    with col4:
        st.metric("Série", stats.get('current_streak', 0))

def show_prediction():
    """Page de prédiction principale"""
    st.markdown("## 🎯 Prédiction de match")
    
    # Charger les données
    model_info = load_saved_model()
    
    with st.spinner("Chargement des joueurs..."):
        atp_data = load_atp_data()
        all_players = get_all_players(atp_data)
    
    st.success(f"✅ {len(all_players)} joueurs disponibles")
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Joueur 1", all_players, key="p1")
        odds1 = st.text_input("Cote Joueur 1", placeholder="1.75")
    
    with col2:
        players2 = [p for p in all_players if p != player1]
        player2 = st.selectbox("Joueur 2", players2, key="p2")
        odds2 = st.text_input("Cote Joueur 2", placeholder="2.10")
    
    col1, col2 = st.columns(2)
    with col1:
        tournament = st.selectbox("Tournoi", sorted(TOURNAMENTS_DB.keys()))
        surface = TOURNAMENTS_DB[tournament]
    with col2:
        st.info(f"Surface: {SURFACE_CONFIG[surface]['icon']} {surface}")
        use_ai = st.checkbox("🤖 Analyse IA", True)
    
    if st.button("🔍 Prédire", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            # Calculs
            h2h = get_h2h_stats(atp_data, player1, player2)
            proba, ml_used = calculate_probability(atp_data, player1, player2, surface, h2h, model_info)
            confidence = calculate_confidence(proba, h2h)
            
            # Value bet
            best_value = None
            if odds1 and odds2:
                try:
                    o1 = float(odds1.replace(',', '.'))
                    o2 = float(odds2.replace(',', '.'))
                    edge1 = proba - 1/o1
                    edge2 = (1-proba) - 1/o2
                    if edge1 > 0.02:
                        best_value = {'joueur': player1, 'edge': edge1, 'cote': o1, 'proba': proba}
                    elif edge2 > 0.02:
                        best_value = {'joueur': player2, 'edge': edge2, 'cote': o2, 'proba': 1-proba}
                except:
                    pass
            
            # Paris alternatifs
            bet_suggestions = generate_alternative_bets(player1, player2, surface, proba, h2h)
            
            # Affichage
            st.markdown("### 📊 Résultat")
            st.markdown(f"#### 🏆 Gagnant: **{player1 if proba >= 0.5 else player2}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(player1, f"{proba:.1%}")
            with col2:
                st.metric(player2, f"{1-proba:.1%}")
            
            st.progress(float(proba))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"{'🤖 ML' if ml_used else '📊 Stats'}")
            with col2:
                conf_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🔴"
                st.caption(f"Confiance: {conf_color} {confidence:.0f}/100")
            with col3:
                if h2h:
                    st.caption(f"H2H: {h2h.get(f'{player1}_wins', 0)}-{h2h.get(f'{player2}_wins', 0)}")
            
            if best_value:
                st.success(f"🎯 Value bet! {best_value['joueur']} @ {best_value['cote']:.2f}")
            
            if bet_suggestions:
                st.markdown("### 🎯 Paris alternatifs")
                for bet in bet_suggestions:
                    st.info(f"{bet['type']}: {bet['description']} (proba: {bet['proba']:.1%})")
            
            # Analyse IA
            if use_ai and get_groq_key():
                with st.spinner("Analyse IA..."):
                    ai_comment = call_groq_api(f"Analyse {player1} vs {player2} sur {surface}. Proba: {proba:.1%}. Donne 3 points clés.")
                    if ai_comment:
                        with st.expander("🤖 Analyse IA"):
                            st.write(ai_comment)
            
            # Sauvegarde
            pred_data = {
                'player1': player1, 'player2': player2,
                'tournament': tournament, 'surface': surface,
                'proba': float(proba), 'confidence': float(confidence),
                'odds1': odds1, 'odds2': odds2,
                'favori': player1 if proba >= 0.5 else player2,
                'best_value': best_value, 'ml_used': ml_used,
                'date': datetime.now().isoformat()
            }
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Sauvegarder", use_container_width=True):
                    if save_prediction(pred_data):
                        st.success("✅ Sauvegardé!")
            with col2:
                if st.button("📱 Envoyer Telegram", use_container_width=True):
                    if send_prediction_to_telegram(pred_data, bet_suggestions, ai_comment):
                        st.success("✅ Envoyé!")

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
            st.write(f"Surface: {pred.get('surface')}")
            st.write(f"Probabilité: {pred.get('proba', 0.5):.1%}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"✅ {pred['player1']} gagne", key=f"w1_{pred['id']}"):
                    update_prediction_status(pred['id'], 'gagne')
                    st.rerun()
            with col2:
                if st.button(f"✅ {pred['player2']} gagne", key=f"w2_{pred['id']}"):
                    update_prediction_status(pred['id'], 'gagne')
                    st.rerun()
            with col3:
                if st.button(f"❌ Perdu", key=f"l_{pred['id']}"):
                    update_prediction_status(pred['id'], 'perdu')
                    st.rerun()
            
            if st.button(f"⚠️ Annuler", key=f"c_{pred['id']}"):
                update_prediction_status(pred['id'], 'annule')
                st.rerun()

def show_history():
    """Page Historique"""
    st.markdown("## 📜 Historique")
    
    history = load_history()
    if not history:
        st.info("Aucune prédiction")
        return
    
    for pred in history[::-1][:50]:
        status_icon = STATUS_OPTIONS.get(pred.get('statut'), "⏳")
        with st.expander(f"{status_icon} {pred.get('date', '')[:16]} - {pred['player1']} vs {pred['player2']}"):
            st.write(f"Surface: {pred.get('surface')}")
            st.write(f"Probabilité: {pred.get('proba', 0.5):.1%}")
            st.write(f"Statut: {STATUS_OPTIONS.get(pred.get('statut'), 'Inconnu')}")

def show_statistics():
    """Page Statistiques"""
    st.markdown("## 📈 Statistiques")
    
    stats = load_user_stats()
    history = load_history()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", stats.get('total_predictions', 0))
    with col2:
        total_valide = stats.get('correct_predictions', 0) + stats.get('incorrect_predictions', 0)
        accuracy = (stats.get('correct_predictions', 0) / total_valide * 100) if total_valide > 0 else 0
        st.metric("Précision", f"{accuracy:.1f}%")
    with col3:
        st.metric("Série", stats.get('current_streak', 0))
    
    if st.button("📊 Envoyer stats Telegram"):
        if send_stats_to_telegram():
            st.success("✅ Stats envoyées!")

def show_telegram():
    """Page Telegram"""
    st.markdown("## 📱 Telegram")
    
    token, chat_id = get_telegram_config()
    if not token or not chat_id:
        st.warning("⚠️ Telegram non configuré")
        st.code("""
        Ajoute dans les secrets Streamlit:
        TELEGRAM_BOT_TOKEN = "ton_token"
        TELEGRAM_CHAT_ID = "ton_chat_id"
        """)
        return
    
    st.success(f"✅ Connecté (Chat ID: {chat_id})")
    
    if st.button("🔧 Tester connexion"):
        success, msg = test_telegram_connection()
        if success:
            st.success(msg)
        else:
            st.error(msg)

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
    
    if st.button("🗑️ Effacer historique"):
        if HIST_FILE.exists():
            HIST_FILE.unlink()
            st.rerun()

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="TennisIQ Pro",
        page_icon="🎾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #0A1E2C 0%, #1A2E3C 100%); }
        .stProgress > div > div > div > div { background: linear-gradient(90deg, #00DFA2, #0079FF); }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎾 TennisIQ")
        page = st.radio(
            "Menu",
            ["🏠 Dashboard", "🎯 Prédiction", "⏳ En Attente", 
             "📜 Historique", "📈 Statistiques", "📱 Telegram", "⚙️ Configuration"]
        )
    
    # Pages
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🎯 Prédiction":
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
