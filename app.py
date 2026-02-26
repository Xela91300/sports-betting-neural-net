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
🤖 TennisIQ Bot prêt à recevoir des statistiques

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

def format_stats_message():
    """Formate un message de statistiques pour Telegram"""
    stats = load_user_stats()
    history = load_history()
    
    total = stats.get('total_predictions', 0)
    correct = stats.get('correct_predictions', 0)
    incorrect = stats.get('incorrect_predictions', 0)
    annules = stats.get('annules_predictions', 0)
    
    # Calculer la précision (ignorer les annulés)
    total_valide = correct + incorrect
    accuracy = (correct / total_valide * 100) if total_valide > 0 else 0
    
    # Barre de progression
    bar_length = 10
    filled = int(accuracy / 10)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    # Calculer la tendance (comparer avec la période précédente)
    recent = [p for p in history[-20:] if p.get('statut') in ['gagne', 'perdu']]
    recent_correct = sum(1 for p in recent if p.get('statut') == 'gagne')
    recent_acc = (recent_correct / len(recent) * 100) if recent else 0
    
    # Calculer la différence avec la moyenne globale
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

def send_stats_to_telegram():
    """Envoie les statistiques sur Telegram"""
    return send_telegram_message(format_stats_message())

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
            return None
            
    except Exception as e:
        print(f"Exception Groq API: {e}")
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
    
    if model_info:
        try:
            model = model_info.get('model')
            scaler = model_info.get('scaler')
            player_stats = model_info.get('player_stats', {})
            
            if model and scaler:
                s1 = player_stats.get(player1, {})
                s2 = player_stats.get(player2, {})
                
                if s1 and s2:
                    # Version simplifiée
                    r1 = max(s1.get('rank', 500.0), 1.0)
                    r2 = max(s2.get('rank', 500.0), 1.0)
                    log_rank_ratio = np.log(r2 / r1)
                    
                    surf_wr_diff = s1.get('surface_wr', {}).get(surface, 0.5) - s2.get('surface_wr', {}).get(surface, 0.5)
                    
                    features = np.array([[log_rank_ratio, 0, 0, 0, 0, 0, 0, 0, 0, surf_wr_diff, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]])
                    features_scaled = scaler.transform(features)
                    proba = model.predict_proba(features_scaled)[0][1]
                    return max(0.05, min(0.95, float(proba))), True
        except:
            pass
    
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
        pred_data['statut'] = 'en_attente'  # Toujours en attente au départ
        history.append(pred_data)
        
        # Garder seulement les 1000 dernières
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
                old_status = pred.get('statut')
                pred['statut'] = new_status
                pred['date_maj'] = datetime.now().isoformat()
                break
        
        with open(HIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        
        # Recalculer les stats après modification
        update_user_stats()
        return True
    except:
        return False

def load_user_stats():
    """Charge les statistiques utilisateur"""
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
            'last_updated': datetime.now().isoformat()
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
    
    # Calculer la série actuelle
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
        # Annulé ne casse pas la série mais ne l'augmente pas non plus
    
    stats = {
        'total_predictions': total,
        'correct_predictions': correct,
        'incorrect_predictions': incorrect,
        'annules_predictions': annules,
        'total_combines': 0,  # À implémenter si besoin
        'won_combines': 0,
        'total_invested': 0,
        'total_won': 0,
        'current_streak': current_streak,
        'best_streak': best_streak,
        'last_updated': datetime.now().isoformat()
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
# PAGES DE L'APPLICATION
# ─────────────────────────────────────────────────────────────

def show_dashboard():
    """Page Dashboard avec stats en temps réel"""
    st.markdown("<h2>🏠 Dashboard</h2>", unsafe_allow_html=True)
    
    stats = load_user_stats()
    history = load_history()
    
    total = stats.get('total_predictions', 0)
    correct = stats.get('correct_predictions', 0)
    incorrect = stats.get('incorrect_predictions', 0)
    annules = stats.get('annules_predictions', 0)
    
    total_valide = correct + incorrect
    accuracy = (correct / total_valide * 100) if total_valide > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total prédictions", total)
    with col2:
        st.metric("✅ Gagnées", correct)
    with col3:
        st.metric("❌ Perdues", incorrect)
    with col4:
        st.metric("⚠️ Annulées", annules)
    
    # Barre de progression pour la précision
    st.markdown("### 🎯 Précision globale")
    st.progress(accuracy / 100)
    st.caption(f"{accuracy:.1f}% de réussite sur {total_valide} matchs résolus")
    
    # Graphique d'évolution
    st.markdown("### 📈 Évolution des prédictions")
    
    # Préparer les données pour le graphique
    df_history = pd.DataFrame(history)
    if not df_history.empty and 'date' in df_history.columns:
        df_history['date'] = pd.to_datetime(df_history['date'])
        df_history = df_history.sort_values('date')
        
        # Calculer la précision cumulative
        results = []
        correct_count = 0
        total_count = 0
        
        for _, row in df_history.iterrows():
            if row.get('statut') in ['gagne', 'perdu']:
                total_count += 1
                if row.get('statut') == 'gagne':
                    correct_count += 1
                results.append(correct_count / total_count * 100 if total_count > 0 else 0)
            else:
                results.append(results[-1] if results else 0)
        
        df_history['cumulative_accuracy'] = results
        
        # Afficher le graphique
        st.line_chart(df_history.set_index('date')['cumulative_accuracy'])
    
    # Dernières prédictions
    st.markdown("### 🕒 Dernières prédictions")
    recent = history[-10:][::-1] if history else []
    for pred in recent[:5]:
        status_icon = STATUS_OPTIONS.get(pred.get('statut'), "⏳")
        st.caption(f"{status_icon} {pred.get('date', '')[:16]} - {pred.get('player1')} vs {pred.get('player2')}")

def show_prediction():
    """Page de prédiction simple"""
    st.markdown("<h2>🎯 Nouvelle Prédiction</h2>", unsafe_allow_html=True)
    
    model_info = load_saved_model()
    atp_data = load_atp_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.text_input("Joueur 1", "Novak Djokovic")
        player2 = st.text_input("Joueur 2", "Carlos Alcaraz")
        tournament = st.text_input("Tournoi (optionnel)", "Tournoi ATP")
    
    with col2:
        surface = st.selectbox("Surface", SURFACES)
        odds1 = st.text_input("Cote J1 (optionnel)", "")
        odds2 = st.text_input("Cote J2 (optionnel)", "")
    
    if st.button("🎾 Prédire", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            
            h2h = get_h2h_stats(atp_data, player1, player2)
            proba, ml_used = calculate_probability(atp_data, player1, player2, surface, h2h, model_info)
            confidence = calculate_confidence(proba, h2h)
            
            # Afficher résultat
            st.markdown("### 📊 Résultat")
            st.progress(float(proba))
            
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
                'favori': player1 if proba >= 0.5 else player2,
                'ml_used': ml_used,
                'date': datetime.now().isoformat()
            }
            
            if st.button("💾 Sauvegarder la prédiction", use_container_width=True):
                if save_prediction(pred_data):
                    st.success("✅ Prédiction sauvegardée dans 'En attente' !")
                    st.balloons()
                else:
                    st.error("❌ Erreur lors de la sauvegarde")

def show_pending():
    """Page des prédictions en attente"""
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
            with col1:
                st.metric("Surface", pred.get('surface', '—'))
            with col2:
                st.metric("Probabilité", f"{pred.get('proba', 0.5):.1%}")
            with col3:
                st.metric("Confiance", f"{pred.get('confidence', 0):.0f}")
            
            if pred.get('odds1') and pred.get('odds2'):
                st.caption(f"Cotes: {pred['player1']} @ {pred['odds1']} | {pred['player2']} @ {pred['odds2']}")
            
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
                if st.button(f"❌ Match perdu", key=f"loss_{pred['id']}", use_container_width=True):
                    update_prediction_status(pred['id'], 'perdu')
                    st.rerun()
            
            if st.button(f"⚠️ Annuler", key=f"cancel_{pred['id']}", use_container_width=True):
                update_prediction_status(pred['id'], 'annule')
                st.rerun()

def show_history():
    """Page Historique complet"""
    st.markdown("<h2>📜 Historique complet</h2>", unsafe_allow_html=True)
    
    history = load_history()
    
    if not history:
        st.info("Aucune prédiction dans l'historique")
        return
    
    # Filtres
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
    
    # Appliquer les filtres
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
            with col1:
                st.metric("Surface", pred.get('surface', '—'))
            with col2:
                st.metric("Probabilité", f"{pred.get('proba', 0.5):.1%}")
            with col3:
                st.metric("Statut", STATUS_OPTIONS.get(pred.get('statut'), "Inconnu"))
            
            if pred.get('odds1') and pred.get('odds2'):
                st.caption(f"Cotes: {pred['player1']} @ {pred['odds1']} | {pred['player2']} @ {pred['odds2']}")
            
            # Permettre de modifier le statut
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

def show_statistics():
    """Page Statistiques détaillées"""
    st.markdown("<h2>📈 Statistiques détaillées</h2>", unsafe_allow_html=True)
    
    stats = load_user_stats()
    history = load_history()
    
    total = stats.get('total_predictions', 0)
    correct = stats.get('correct_predictions', 0)
    incorrect = stats.get('incorrect_predictions', 0)
    annules = stats.get('annules_predictions', 0)
    
    total_valide = correct + incorrect
    accuracy = (correct / total_valide * 100) if total_valide > 0 else 0
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", total)
    with col2:
        st.metric("✅ Gagnées", correct, f"{accuracy:.1f}%")
    with col3:
        st.metric("❌ Perdues", incorrect)
    with col4:
        st.metric("⚠️ Annulées", annules)
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Répartition des résultats")
        if total_valide > 0:
            data = pd.DataFrame({
                'Statut': ['Gagnées', 'Perdues'],
                'Nombre': [correct, incorrect]
            })
            st.bar_chart(data.set_index('Statut'))
    
    with col2:
        st.markdown("### 🔥 Séries en cours")
        st.metric("Série actuelle", f"{stats.get('current_streak', 0)}", 
                 delta="🔥" if stats.get('current_streak', 0) >= 5 else "")
        st.metric("Meilleure série", stats.get('best_streak', 0))
    
    # Bouton pour envoyer les stats sur Telegram
    if st.button("📱 Envoyer les stats sur Telegram", use_container_width=True):
        if send_stats_to_telegram():
            st.success("✅ Statistiques envoyées sur Telegram !")
        else:
            st.error("❌ Échec de l'envoi. Vérifie la configuration Telegram.")

def show_telegram():
    """Page Configuration Telegram"""
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
    
    # Envoi des stats
    st.markdown("### 📊 Envoyer les statistiques")
    if st.button("📤 Envoyer les stats maintenant", use_container_width=True):
        with st.spinner("Envoi en cours..."):
            if send_stats_to_telegram():
                st.success("✅ Statistiques envoyées !")
            else:
                st.error("❌ Échec de l'envoi")
    
    # Envoi de message personnalisé
    st.markdown("### 📝 Message personnalisé")
    with st.form("telegram_form"):
        message = st.text_area("Message", height=100)
        col1, col2 = st.columns(2)
        with col1:
            urgent = st.checkbox("🔴 Urgent")
        with col2:
            include_stats = st.checkbox("Inclure les stats")
        
        if st.form_submit_button("📤 Envoyer") and message:
            final_msg = message
            if urgent:
                final_msg = "🔴 URGENT\n\n" + final_msg
            if include_stats:
                stats_msg = format_stats_message()
                final_msg += f"\n\n{stats_msg}"
            
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
        """)
        
        if st.button("🔄 Recharger le modèle"):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.warning("⚠️ Aucun modèle trouvé")
    
    # Statistiques
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
    
    # Gestion des données
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
        page_title="TennisIQ Pro - Stats & Prédictions",
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
        div[data-testid="stMetricValue"] { font-size: 2rem; }
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
                Tracking & Analytics
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🎯 Nouvelle Prédiction", "⏳ En Attente", 
             "📜 Historique", "📈 Statistiques", "📱 Telegram", "⚙️ Configuration"],
            label_visibility="collapsed"
        )
        
        # Stats rapides dans la sidebar
        st.divider()
        stats = load_user_stats()
        pending = len([p for p in load_history() if p.get('statut') == 'en_attente'])
        
        total_valide = stats.get('correct_predictions', 0) + stats.get('incorrect_predictions', 0)
        accuracy = (stats.get('correct_predictions', 0) / total_valide * 100) if total_valide > 0 else 0
        
        st.caption(f"📊 Précision: {accuracy:.1f}%")
        st.caption(f"⏳ En attente: {pending}")
        st.caption(f"🔥 Série: {stats.get('current_streak', 0)}")
    
    # Afficher la page sélectionnée
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🎯 Nouvelle Prédiction":
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
