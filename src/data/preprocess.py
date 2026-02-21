import pandas as pd
from pathlib import Path

def calculate_form(df, team_col, result_col, n=5):
    """Calcule la forme sur n derniers matchs (points: 3 win, 1 draw, 0 loss)"""
    df['points'] = df[result_col].map({'W': 3, 'D': 1, 'L': 0})
    df[f'form_{n}'] = df.groupby(team_col)['points'].rolling(n, min_periods=1).sum().reset_index(0, drop=True)
    return df

def preprocess_football(raw_path, processed_path):
    df = pd.read_csv(raw_path)  # Assume colonnes comme FTHG (full time home goals), FTAG, etc. de football-data.co.uk
    df['home_win'] = (df['FTHG'] > df['FTAG']).astype(int)
    # Ajoute forme (exemple simplifié – adapte à ton dataset)
    df = calculate_form(df, 'HomeTeam', 'FTR', n=5).rename(columns={'form_5': 'home_form_5'})
    df = calculate_form(df, 'AwayTeam', 'FTR', n=5).rename(columns={'form_5': 'away_form_5'})
    # Ajoute averages (rolling)
    df['home_goals_avg'] = df.groupby('HomeTeam')['FTHG'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df['away_goals_avg'] = df.groupby('AwayTeam')['FTAG'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df['diff_classement'] = df['HomeTeam_rank'] - df['AwayTeam_rank']  # Assume tu as des ranks
    # Sélectionne features + target
    features = ['home_form_5', 'away_form_5', 'home_goals_avg', 'away_goals_avg', 'diff_classement', 'B365H', 'B365D', 'B365A']
    df[features + ['home_win']].rename(columns={'home_win': 'target'}).to_csv(processed_path, index=False)
    print(f"Processed football data saved to {processed_path}")

def preprocess_tennis(raw_path, processed_path):
    df = pd.read_csv(raw_path)  # Assume colonnes comme winner_id, loser_id, surface, rank_points_winner, etc. de JeffSackmann
    df['player1_win'] = 1  # Assume player1 est winner pour simplifier – adapte
    # One-hot pour surface
    df = pd.get_dummies(df, columns=['surface'])
    # Forme récente
    df = calculate_form(df, 'player1_id', 'winner', n=10).rename(columns={'form_10': 'form_10_p1'})  # Adap te 'winner' column
    df = calculate_form(df, 'player2_id', 'winner', n=10).rename(columns={'form_10': 'form_10_p2'})
    df['rank_diff'] = df['rank_points_p1'] - df['rank_points_p2']
    df['h2h_p1_wins'] = 0  # À calculer avec groupby sur historical matches
    df['fatigue_p1'] = df.groupby('player1_id')['match_num'].rolling(5).count().reset_index(0, drop=True)  # Ex fatigue
    features = ['rank_diff', 'surface_hard', 'surface_clay', 'surface_grass', 'form_10_p1', 'form_10_p2', 'h2h_p1_wins', 'fatigue_p1']
    df[features + ['player1_win']].rename(columns={'player1_win': 'target'}).to_csv(processed_path, index=False)
    print(f"Processed tennis data saved to {processed_path}")

def preprocess_basketball(raw_path, processed_path):
    df = pd.read_csv(raw_path)  # Assume colonnes comme TEAM_ID_HOME, PTS_HOME, REB_HOME, etc. de Kaggle NBA
    df['home_win'] = (df['PTS_HOME'] > df['PTS_AWAY']).astype(int)
    # Averages
    df['points_avg_home'] = df.groupby('TEAM_ID_HOME')['PTS_HOME'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df['reb_avg_home'] = df.groupby('TEAM_ID_HOME')['REB_HOME'].rolling(5).mean().reset_index(0, drop=True)
    df['eff_rating_home'] = (df['points_avg_home'] + df['reb_avg_home']) / 2  # Simplifié
    df['back_to_back'] = df.groupby('TEAM_ID_HOME')['GAME_DATE'].diff().dt.days == 1  # Assume datetime
    df['spread'] = df['SPREAD_HOME']  # Si disponible
    # Similaire pour away
    df['points_avg_away'] = df.groupby('TEAM_ID_AWAY')['PTS_AWAY'].rolling(5).mean().reset_index(0, drop=True)
    df['reb_avg_away'] = df.groupby('TEAM_ID_AWAY')['REB_AWAY'].rolling(5).mean().reset_index(0, drop=True)
    df['eff_rating_away'] = (df['points_avg_away'] + df['reb_avg_away']) / 2
    features = ['points_avg_home', 'reb_avg_home', 'eff_rating_home', 'back_to_back', 'spread', 'points_avg_away', 'reb_avg_away', 'eff_rating_away']
    df[features + ['home_win']].rename(columns={'home_win': 'target'}).to_csv(processed_path, index=False)
    print(f"Processed basketball data saved to {processed_path}")

# Exemple d'appel (à utiliser dans un script ou notebook)
if __name__ == "__main__":
    # Ex: preprocess_football('data/raw/soccer.csv', 'data/processed/football_features.csv')
    pass
