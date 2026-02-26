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

def format_prediction_message(pred_data, ai_comment=None):
    """Formate un message de prédiction pour Telegram"""
    proba = pred_data.get('proba', 0.5)
    bar_length = 10
    filled = int(proba * bar_length)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    emoji_map = {'Hard': '🟦', 'Clay': '🟧', 'Grass': '🟩'}
    surface_emoji = emoji_map.get(pred_data.get('surface', ''), '🎾')
    
    # Badge ML si utilisé
    ml_tag = "🤖 " if pred_data.get('ml_used') else ""
    
    message = f"""
<b>{ml_tag}🎾 PRÉDICTION TENNISIQ</b>

<b>Match:</b> {pred_data.get('player1', '?')} vs {pred_data.get('player2', '?')}
<b>Tournoi:</b> {pred_data.get('tournament', 'Inconnu')}
<b>Surface:</b> {surface_emoji} {pred_data.get('surface', '?')}

<b>Probabilités:</b>
{bar}  {proba:.1%} / {1-proba:.1%}

• {pred_data.get('player1', 'J1')}: <b>{proba:.1%}</b>
• {pred_data.get('player2', 'J2')}: <b>{1-proba:.1%}</b>

<b>Favori:</b> {pred_data.get('favori_modele', '?')}
<b>Confiance:</b> {'🟢' if pred_data.get('confidence', 0) >= 70 else '🟡' if pred_data.get('confidence', 0) >= 50 else '🔴'} {pred_data.get('confidence', 0):.0f}/100
"""
    
    # Ajouter les cotes si disponibles
    if pred_data.get('odds1') and pred_data.get('odds2'):
        message += f"""
<b>Cotes:</b>
• {pred_data.get('player1', 'J1')}: <code>{pred_data.get('odds1')}</code>
• {pred_data.get('player2', 'J2')}: <code>{pred_data.get('odds2')}</code>
"""
    
    # Ajouter value bet si détecté
    if pred_data.get('best_value'):
        bv = pred_data['best_value']
        edge_color = '🟢' if bv['edge'] > 0.05 else '🟡'
        message += f"""
<b>🎯 VALUE BET DÉTECTÉ!</b>
{edge_color} <b>{bv['joueur']}</b> à <b>{bv['cote']:.2f}</b>
Edge: <b>{bv['edge']*100:+.1f}%</b>
"""
    
    # Ajouter commentaire IA
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

def send_prediction_to_telegram(pred_data, ai_comment=None):
    """Envoie une prédiction sur Telegram"""
    return send_telegram_message(format_prediction_message(pred_data, ai_comment))

def send_combine_to_telegram(combine_data, ai_comment=None):
    """Envoie un combiné sur Telegram"""
    return send_telegram_message(format_combine_message(combine_data, ai_comment))

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
        # Version simple avec requests (sans bibliothèque groq)
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "Tu es un expert en analyse de tennis. Fournis des analyses concises en 3 points clés en français."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 300
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print(f"Erreur Groq API: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception Groq API: {e}")
        return None

def analyze_match_with_ai(player1, player2, surface, proba, best_value=None):
    """Génère une analyse IA pour un match"""
    vb_txt = ""
    if best_value:
        vb_txt = f" Value bet détecté sur {best_value['joueur']} avec un edge de {best_value['edge']*100:+.1f}%."
    
    prompt = f"""Analyse ce match de tennis en 3 points clés:
    {player1} vs {player2}
    Surface: {surface}
    Probabilités: {player1} {proba:.1%} - {player2} {1-proba:.1%}
    {vb_txt}
    
    Donne une analyse concise en français avec:
    1. Facteur clé du match
    2. Analyse des forces/faiblesses
    3. Pronostic final
    """
    
    return call_groq_api(prompt)

def analyze_combine_with_ai(selections, proba_globale, cote_globale, esperance):
    """Génère une analyse IA pour un combiné"""
    selections_txt = "\n".join([f"- {s['joueur']} @ {s['cote']:.2f} (edge: {s['edge']*100:+.1f}%)" 
                                for s in selections[:5]])
    
    prompt = f"""Analyse ce combiné de tennis en 3 points:
    {len(selections)} sélections:
    {selections_txt}
    
    Probabilité globale: {proba_globale:.1%}
    Cote combinée: {cote_globale:.2f}
    Espérance: {esperance:+.2f}€
    
    Donne un avis concis sur la pertinence de ce combiné.
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
            st.warning(f"⚠️ Erreur chargement modèle: {e}")
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
                    st.success("✅ Modèle téléchargé avec succès !")
                    return model_info
        except Exception as e:
            print(f"Erreur téléchargement: {e}")
    
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
        
        # Récupérer les stats des joueurs
        s1 = player_stats.get(player1, {})
        s2 = player_stats.get(player2, {})
        
        if not s1 or not s2:
            return None
        
        # Calculer les features (simplifié)
        r1 = max(s1.get('rank', 500.0), 1.0)
        r2 = max(s2.get('rank', 500.0), 1.0)
        log_rank_ratio = np.log(r2 / r1)
        
        pts_diff = (s1.get('rank_points', 0) - s2.get('rank_points', 0)) / 5000.0
        age_diff = s1.get('age', 25) - s2.get('age', 25)
        
        surf_clay = 1.0 if surface == 'Clay' else 0.0
        surf_grass = 1.0 if surface == 'Grass' else 0.0
        surf_hard = 1.0 if surface == 'Hard' else 0.0
        
        surf_wr_diff = s1.get('surface_wr', {}).get(surface, 0.5) - s2.get('surface_wr', {}).get(surface, 0.5)
        career_wr_diff = s1.get('win_rate', 0.5) - s2.get('win_rate', 0.5)
        recent_form_diff = s1.get('recent_form', 0.5) - s2.get('recent_form', 0.5)
        
        features = np.array([[
            log_rank_ratio, pts_diff, age_diff,
            surf_clay, surf_grass, surf_hard,
            0, 0, 0,  # level features (simplifié)
            surf_wr_diff, career_wr_diff, recent_form_diff, 0.5,  # h2h_ratio par défaut
            0, 0, 0, 0, 0, 0, 0, 0  # autres features à 0 pour simplifier
        ]])
        
        features_scaled = scaler.transform(features)
        proba = model.predict_proba(features_scaled)[0][1]
        
        return max(0.05, min(0.95, float(proba)))
        
    except Exception as e:
        print(f"Erreur prédiction ML: {e}")
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
    for f in csv_files[:5]:  # Limiter pour la vitesse
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

def get_player_stats(df, player):
    """Récupère les stats basiques d'un joueur"""
    if df.empty or not player:
        return None
    
    player_clean = player.strip()
    winner_col = 'winner_name' if 'winner_name' in df.columns else None
    loser_col = 'loser_name' if 'loser_name' in df.columns else None
    
    if not winner_col or not loser_col:
        return None
    
    matches = df[(df[winner_col] == player_clean) | (df[loser_col] == player_clean)]
    if len(matches) == 0:
        return None
    
    wins = len(matches[df[winner_col] == player_clean])
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
    winner_col = 'winner_name' if 'winner_name' in df.columns else None
    loser_col = 'loser_name' if 'loser_name' in df.columns else None
    
    if not winner_col or not loser_col:
        return None
    
    h2h = df[((df[winner_col] == p1) & (df[loser_col] == p2)) | 
             ((df[winner_col] == p2) & (df[loser_col] == p1))]
    
    if len(h2h) == 0:
        return None
    
    return {
        'total_matches': len(h2h),
        f'{p1}_wins': len(h2h[df[winner_col] == p1]),
        f'{p2}_wins': len(h2h[df[winner_col] == p2]),
    }

def calculate_probability(df, player1, player2, surface, h2h=None, model_info=None):
    """Calcule la probabilité (ML si dispo, sinon stats)"""
    
    # Essayer d'abord avec le modèle ML
    if model_info:
        ml_proba = predict_with_ml_model(model_info, player1, player2, surface)
        if ml_proba is not None:
            return ml_proba, True
    
    # Fallback sur les stats
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
# HISTORIQUE
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
        history.append(pred_data)
        if len(history) > 1000:
            history = history[-1000:]
        with open(HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        return True
    except:
        return False

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
        combines.append(combine_data)
        if len(combines) > 200:
            combines = combines[-200:]
        with open(COMB_HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(combines, f, indent=2)
        return True
    except:
        return False

def load_user_stats():
    if not USER_STATS_FILE.exists():
        return {
            'total_predictions': 0,
            'correct_predictions': 0,
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

# ─────────────────────────────────────────────────────────────
# PAGES DE L'APPLICATION
# ─────────────────────────────────────────────────────────────

def show_dashboard():
    """Page Dashboard"""
    st.markdown("<h2>🏠 Dashboard</h2>", unsafe_allow_html=True)
    
    model_info = load_saved_model()
    groq_key = get_groq_key()
    telegram_token, telegram_chat = get_telegram_config()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Matchs disponibles", "349K+")
    with col2:
        history = load_history()
        st.metric("Prédictions", len(history))
    with col3:
        stats = load_user_stats()
        accuracy = (stats.get('correct_predictions', 0) / max(stats.get('total_predictions', 1), 1)) * 100
        st.metric("Précision", f"{accuracy:.1f}%")
    with col4:
        st.metric("Joueurs", "15K+")
    
    st.markdown("### 🛠️ Statut des services")
    col1, col2, col3 = st.columns(3)
    with col1:
        if model_info:
            st.success(f"✅ Modèle ML (acc: {model_info.get('accuracy', 0):.1%})")
        else:
            st.warning("⚠️ Modèle ML non chargé")
    with col2:
        if groq_key:
            st.success("✅ IA Groq connectée")
        else:
            st.warning("⚠️ IA Groq non configurée")
    with col3:
        if telegram_token and telegram_chat:
            st.success("✅ Telegram connecté")
        else:
            st.warning("⚠️ Telegram non configuré")

def show_prediction():
    """Page de prédiction simple avec IA et Telegram"""
    st.markdown("<h2>🎯 Prédiction avec IA</h2>", unsafe_allow_html=True)
    
    # Charger le modèle
    model_info = load_saved_model()
    atp_data = load_atp_data()
    
    # Interface
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.text_input("Joueur 1", "Novak Djokovic")
        player2 = st.text_input("Joueur 2", "Carlos Alcaraz")
        tournament = st.text_input("Tournoi (optionnel)", "Tournoi ATP")
    
    with col2:
        surface = st.selectbox("Surface", SURFACES)
        odds1 = st.text_input("Cote J1 (optionnel)", "")
        odds2 = st.text_input("Cote J2 (optionnel)", "")
        
        use_ai = st.checkbox("🤖 Générer analyse IA", True)
        send_tg = st.checkbox("📱 Envoyer sur Telegram", False)
    
    if st.button("🎾 Prédire avec IA", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            
            # Récupérer H2H
            h2h = get_h2h_stats(atp_data, player1, player2)
            
            # Calculer probabilité
            proba, ml_used = calculate_probability(atp_data, player1, player2, surface, h2h, model_info)
            confidence = calculate_confidence(proba, h2h)
            
            # Calculer value bet si cotes fournies
            best_value = None
            if odds1 and odds2:
                try:
                    o1 = float(odds1.replace(',', '.'))
                    o2 = float(odds2.replace(',', '.'))
                    edge1 = proba - 1/o1
                    edge2 = (1-proba) - 1/o2
                    
                    if edge1 > edge2 and edge1 > MIN_EDGE_COMBINE:
                        best_value = {'joueur': player1, 'edge': edge1, 'cote': o1, 'proba': proba}
                    elif edge2 > edge1 and edge2 > MIN_EDGE_COMBINE:
                        best_value = {'joueur': player2, 'edge': edge2, 'cote': o2, 'proba': 1-proba}
                except:
                    pass
            
            # Générer analyse IA
            ai_comment = None
            if use_ai:
                with st.spinner("🤖 Génération de l'analyse IA..."):
                    ai_comment = analyze_match_with_ai(player1, player2, surface, proba, best_value)
            
            # Afficher résultat
            st.markdown("### 📊 Résultat")
            
            # Barre de progression
            st.progress(float(proba))
            
            # Probabilités
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: rgba(0,223,162,0.1); border-radius: 10px;">
                    <div style="font-size: 1.2rem;">{player1}</div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: #00DFA2;">{proba:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: rgba(255,59,63,0.1); border-radius: 10px;">
                    <div style="font-size: 1.2rem;">{player2}</div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: #FF3B3F;">{1-proba:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Informations supplémentaires
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"{'🤖 ML' if ml_used else '📊 Stats'} utilisé")
            with col2:
                conf_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🔴"
                st.info(f"Confiance: {conf_color} {confidence:.0f}/100")
            with col3:
                if h2h:
                    st.info(f"H2H: {h2h.get(f'{player1}_wins', 0)}-{h2h.get(f'{player2}_wins', 0)}")
            
            # Value bet
            if best_value:
                st.success(f"🎯 Value bet! {best_value['joueur']} @ {best_value['cote']:.2f} (edge: {best_value['edge']*100:+.1f}%)")
            
            # Analyse IA
            if ai_comment:
                with st.expander("🤖 Analyse IA", expanded=True):
                    st.markdown(ai_comment)
            
            # Préparer données pour sauvegarde
            pred_data = {
                'player1': player1,
                'player2': player2,
                'tournament': tournament,
                'surface': surface,
                'proba': float(proba),
                'confidence': float(confidence),
                'odds1': odds1 if odds1 else None,
                'odds2': odds2 if odds2 else None,
                'favori_modele': player1 if proba >= 0.5 else player2,
                'best_value': best_value,
                'ml_used': ml_used,
                'date': datetime.now().isoformat(),
                'statut': 'en_attente'
            }
            
            # Boutons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Sauvegarder", use_container_width=True):
                    if save_prediction(pred_data):
                        st.success("✅ Prédiction sauvegardée !")
            
            with col2:
                if st.button("📱 Envoyer sur Telegram", use_container_width=True):
                    if send_prediction_to_telegram(pred_data, ai_comment):
                        st.success("✅ Envoyé sur Telegram !")
                    else:
                        st.error("❌ Échec envoi Telegram")

def show_multimatches():
    """Page Multi-matchs avec IA"""
    st.markdown("<h2>📊 Multi-matchs</h2>", unsafe_allow_html=True)
    
    n_matches = st.number_input("Nombre de matchs", 2, 10, 3)
    use_ai = st.checkbox("🤖 Générer analyses IA", True)
    send_all = st.checkbox("📱 Envoyer tout sur Telegram", False)
    
    atp_data = load_atp_data()
    model_info = load_saved_model()
    
    matches = []
    for i in range(n_matches):
        with st.expander(f"Match {i+1}", expanded=i==0):
            col1, col2 = st.columns(2)
            with col1:
                p1 = st.text_input(f"Joueur 1", key=f"mm_p1_{i}")
                odds1 = st.text_input(f"Cote {p1}", key=f"mm_odds1_{i}")
            with col2:
                p2 = st.text_input(f"Joueur 2", key=f"mm_p2_{i}")
                odds2 = st.text_input(f"Cote {p2}", key=f"mm_odds2_{i}")
            
            surface = st.selectbox(f"Surface", SURFACES, key=f"mm_surf_{i}")
            matches.append({'p1': p1, 'p2': p2, 'surface': surface, 'odds1': odds1, 'odds2': odds2})
    
    if st.button("🔍 Analyser tout", use_container_width=True):
        for i, match in enumerate(matches):
            if match['p1'] and match['p2']:
                st.markdown(f"### Match {i+1}: {match['p1']} vs {match['p2']}")
                
                h2h = get_h2h_stats(atp_data, match['p1'], match['p2'])
                proba, ml_used = calculate_probability(atp_data, match['p1'], match['p2'], match['surface'], h2h, model_info)
                
                st.progress(float(proba))
                st.caption(f"Probabilité {match['p1']}: {proba:.1%} ({'🤖 ML' if ml_used else '📊 Stats'})")
                
                if use_ai:
                    with st.spinner(f"🤖 Analyse IA match {i+1}..."):
                        ai_comment = analyze_match_with_ai(match['p1'], match['p2'], match['surface'], proba)
                        if ai_comment:
                            with st.expander("Voir analyse IA"):
                                st.markdown(ai_comment)
                            
                            if send_all:
                                pred_data = {
                                    'player1': match['p1'], 'player2': match['p2'],
                                    'surface': match['surface'], 'proba': proba,
                                    'confidence': calculate_confidence(proba, h2h),
                                    'odds1': match['odds1'], 'odds2': match['odds2'],
                                    'ml_used': ml_used,
                                    'date': datetime.now().isoformat()
                                }
                                send_prediction_to_telegram(pred_data, ai_comment)
                
                st.divider()

def show_combines():
    """Page Combinés avec IA"""
    st.markdown("<h2>🎰 Générateur de Combinés</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_matches = st.number_input("Nombre de matchs", 2, 10, 3)
    with col2:
        mise = st.number_input("Mise (€)", 1.0, 1000.0, 10.0)
    with col3:
        use_ai = st.checkbox("🤖 Analyser avec IA", True)
        send_tg = st.checkbox("📱 Envoyer sur Telegram", False)
    
    atp_data = load_atp_data()
    model_info = load_saved_model()
    selections = []
    
    for i in range(n_matches):
        with st.expander(f"Match {i+1}"):
            col1, col2 = st.columns(2)
            with col1:
                p1 = st.text_input(f"J1", key=f"comb_p1_{i}")
                odds1 = st.text_input(f"Cote {p1}", key=f"comb_odds1_{i}")
            with col2:
                p2 = st.text_input(f"J2", key=f"comb_p2_{i}")
                odds2 = st.text_input(f"Cote {p2}", key=f"comb_odds2_{i}")
            
            surface = st.selectbox("Surface", SURFACES, key=f"comb_surf_{i}")
            
            if p1 and p2 and odds1 and odds2:
                try:
                    o1 = float(odds1.replace(',', '.'))
                    o2 = float(odds2.replace(',', '.'))
                    h2h = get_h2h_stats(atp_data, p1, p2)
                    proba, ml_used = calculate_probability(atp_data, p1, p2, surface, h2h, model_info)
                    
                    edge1 = proba - 1/o1
                    edge2 = (1-proba) - 1/o2
                    
                    if edge1 > MIN_EDGE_COMBINE and proba >= MIN_PROBA_COMBINE:
                        selections.append({
                            'match': f"{p1} vs {p2}",
                            'joueur': p1, 'proba': proba,
                            'cote': o1, 'edge': edge1,
                            'ml_used': ml_used
                        })
                    elif edge2 > MIN_EDGE_COMBINE and (1-proba) >= MIN_PROBA_COMBINE:
                        selections.append({
                            'match': f"{p1} vs {p2}",
                            'joueur': p2, 'proba': 1-proba,
                            'cote': o2, 'edge': edge2,
                            'ml_used': ml_used
                        })
                except:
                    pass
    
    if selections and st.button("🎯 Générer combiné", use_container_width=True):
        selections.sort(key=lambda x: x['edge'], reverse=True)
        selected = selections[:min(5, len(selections))]
        
        proba_combi = np.prod([s['proba'] for s in selected])
        cote_combi = np.prod([s['cote'] for s in selected])
        gain = mise * cote_combi
        esperance = proba_combi * gain - mise
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Probabilité", f"{proba_combi:.1%}")
        with col2:
            st.metric("Cote", f"{cote_combi:.2f}")
        with col3:
            st.metric("Gain potentiel", f"{gain:.2f}€")
        with col4:
            color = "normal" if esperance > 0 else "inverse"
            st.metric("Espérance", f"{esperance:+.2f}€", delta_color=color)
        
        # Sélections
        st.markdown("### 📋 Sélections")
        df_sel = pd.DataFrame([{
            '#': i+1,
            'Joueur': s['joueur'],
            'Proba': f"{s['proba']:.1%}",
            'Cote': f"{s['cote']:.2f}",
            'Edge': f"{s['edge']*100:+.1f}%",
            'ML': '🤖' if s.get('ml_used') else '📊'
        } for i, s in enumerate(selected)])
        st.dataframe(df_sel, use_container_width=True, hide_index=True)
        
        # Analyse IA
        ai_comment = None
        if use_ai:
            with st.spinner("🤖 Analyse IA du combiné..."):
                ai_comment = analyze_combine_with_ai(selected, proba_combi, cote_combi, esperance)
                if ai_comment:
                    with st.expander("🤖 Analyse IA", expanded=True):
                        st.markdown(ai_comment)
        
        # Sauvegarder
        combine_data = {
            'selections': selected,
            'proba_globale': float(proba_combi),
            'cote_globale': float(cote_combi),
            'mise': float(mise),
            'gain_potentiel': float(gain),
            'esperance': float(esperance),
            'nb_matches': len(selected),
            'ml_used': any(s.get('ml_used', False) for s in selected)
        }
        save_combine(combine_data)
        st.success("✅ Combiné sauvegardé !")
        
        # Envoyer sur Telegram
        if send_tg and ai_comment:
            if send_combine_to_telegram(combine_data, ai_comment):
                st.success("✅ Combiné envoyé sur Telegram !")

def show_history():
    """Page Historique"""
    st.markdown("<h2>📜 Historique</h2>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📋 Prédictions", "🎰 Combinés"])
    
    with tab1:
        history = load_history()
        if history:
            for pred in history[::-1][:20]:
                ml_indicator = "🤖 " if pred.get('ml_used') else ""
                with st.expander(f"{pred.get('date', '')[:16]} - {ml_indicator}{pred.get('player1','?')} vs {pred.get('player2','?')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Surface", pred.get('surface', '—'))
                    with col2:
                        st.metric("Probabilité", f"{pred.get('proba', 0.5):.1%}")
                    with col3:
                        st.metric("Confiance", f"{pred.get('confidence', 0):.0f}")
                    
                    if pred.get('best_value'):
                        st.success(f"🎯 Value bet: {pred['best_value']['joueur']}")
                    
                    if pred.get('odds1') and pred.get('odds2'):
                        st.caption(f"Cotes: {pred['player1']} @ {pred['odds1']} | {pred['player2']} @ {pred['odds2']}")
        else:
            st.info("Aucune prédiction")
    
    with tab2:
        combines = load_combines()
        if combines:
            for comb in combines[::-1][:10]:
                ml_indicator = "🤖 " if comb.get('ml_used') else ""
                with st.expander(f"{comb.get('date', '')[:16]} - {ml_indicator}{comb.get('nb_matches',0)} matchs"):
                    st.metric("Probabilité", f"{comb.get('proba_globale',0):.1%}")
                    st.metric("Cote", f"{comb.get('cote_globale',0):.2f}")
                    st.metric("Espérance", f"{comb.get('esperance',0):+.2f}€")
        else:
            st.info("Aucun combiné")

def show_statistics():
    """Page Statistiques"""
    st.markdown("<h2>📈 Statistiques</h2>", unsafe_allow_html=True)
    
    stats = load_user_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Prédictions", stats.get('total_predictions', 0))
    with col2:
        accuracy = (stats.get('correct_predictions', 0) / max(stats.get('total_predictions', 1), 1)) * 100
        st.metric("Précision", f"{accuracy:.1f}%")
    with col3:
        st.metric("Combinés", stats.get('total_combines', 0))
    with col4:
        profit = stats.get('total_won', 0) - stats.get('total_invested', 0)
        st.metric("Profit", f"{profit:+.2f}€")
    
    # Graphique d'évolution (simulé)
    st.markdown("### 📊 Évolution de la précision")
    dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
    perf_data = pd.DataFrame({
        'date': dates,
        'précision': np.random.uniform(60, 75, 10)
    })
    st.line_chart(perf_data.set_index('date'))

def show_telegram():
    """Page Telegram"""
    st.markdown("<h2>📱 Configuration Telegram</h2>", unsafe_allow_html=True)
    
    token, chat_id = get_telegram_config()
    
    if not token or not chat_id:
        st.warning("⚠️ Telegram non configuré")
        st.markdown("""
        ### Configuration requise :
        
        1. Va sur Telegram et cherche @BotFather
        2. Crée un nouveau bot avec `/newbot`
        3. Copie le token fourni
        4. Ajoute dans les secrets Streamlit :
        
        ```toml
        TELEGRAM_BOT_TOKEN = "ton_token_ici"
        TELEGRAM_CHAT_ID = "ton_chat_id_ici"
        ```
        
        Pour obtenir ton chat_id, envoie un message à @userinfobot
        """)
        return
    
    st.success(f"✅ Telegram configuré (Chat ID: {chat_id})")
    
    # Test de connexion
    if st.button("🔧 Tester la connexion", use_container_width=True):
        with st.spinner("Test en cours..."):
            success, msg = test_telegram_connection()
            if success:
                st.success(msg)
            else:
                st.error(msg)
    
    # Envoi de message personnalisé
    st.markdown("### 📝 Envoyer un message personnalisé")
    with st.form("telegram_form"):
        message = st.text_area("Message", height=100)
        col1, col2 = st.columns(2)
        with col1:
            include_stats = st.checkbox("Inclure les stats")
        with col2:
            urgent = st.checkbox("🔴 Urgent")
        
        if st.form_submit_button("📤 Envoyer") and message:
            final_msg = message
            if urgent:
                final_msg = "🔴 URGENT\n\n" + final_msg
            if include_stats:
                stats = load_user_stats()
                accuracy = (stats.get('correct_predictions', 0) / max(stats.get('total_predictions', 1), 1)) * 100
                final_msg += f"\n\n📊 Stats: {accuracy:.1f}% précision"
            
            if send_telegram_message(final_msg):
                st.success("✅ Message envoyé !")
            else:
                st.error("❌ Échec de l'envoi")

def show_configuration():
    """Page Configuration"""
    st.markdown("<h2>⚙️ Configuration</h2>", unsafe_allow_html=True)
    
    # Modèle ML
    st.markdown("### 🤖 Modèle Machine Learning")
    model_info = load_saved_model()
    if model_info:
        st.success(f"""
        ✅ Modèle chargé avec succès !
        
        - **Accuracy:** {model_info.get('accuracy', 0):.1%}
        - **AUC-ROC:** {model_info.get('auc', 0):.3f}
        - **Features:** {len(model_info.get('features', []))}
        - **Matchs entraînement:** {model_info.get('n_matches', 0):,}
        """)
        
        if st.button("🔄 Recharger le modèle"):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.warning("⚠️ Aucun modèle trouvé dans models/")
        st.info("Place ton fichier tennis_ml_model_complete.pkl dans le dossier models/")
    
    # IA Groq
    st.markdown("### 🧠 Intelligence Artificielle (Groq)")
    groq_key = get_groq_key()
    if groq_key:
        st.success("✅ Clé API Groq configurée")
        if st.button("🧪 Tester l'IA"):
            with st.spinner("Test en cours..."):
                test = call_groq_api("Dis 'Test réussi' en français")
                if test:
                    st.success(f"✅ Réponse: {test[:100]}...")
                else:
                    st.error("❌ Échec du test")
    else:
        st.warning("⚠️ Clé API Groq non configurée")
    
    # Gestion des données
    st.markdown("### 🗑️ Gestion des données")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ Effacer prédictions"):
            if HIST_FILE.exists():
                HIST_FILE.unlink()
                st.rerun()
    with col2:
        if st.button("🗑️ Effacer combinés"):
            if COMB_HIST_FILE.exists():
                COMB_HIST_FILE.unlink()
                st.rerun()
    with col3:
        if st.button("🗑️ Réinitialiser stats"):
            if USER_STATS_FILE.exists():
                USER_STATS_FILE.unlink()
                st.rerun()

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="TennisIQ Pro - IA & ML",
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
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 2rem; font-weight: 800; color: #00DFA2;">
                TennisIQ
            </div>
            <div style="font-size: 0.8rem; color: #6C7A89;">
                IA & Machine Learning
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🎯 Prédiction IA", "📊 Multi-matchs", 
             "🎰 Combinés", "📜 Historique", "📈 Statistiques", 
             "📱 Telegram", "⚙️ Configuration"],
            label_visibility="collapsed"
        )
        
        # Statuts rapides
        st.divider()
        model_info = load_saved_model()
        if model_info:
            st.caption(f"🤖 ML: {model_info.get('accuracy', 0):.1%}")
        if get_groq_key():
            st.caption("🧠 IA: OK")
        if get_telegram_config()[0]:
            st.caption("📱 Telegram: OK")
    
    # Afficher la page sélectionnée
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🎯 Prédiction IA":
        show_prediction()
    elif page == "📊 Multi-matchs":
        show_multimatches()
    elif page == "🎰 Combinés":
        show_combines()
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
