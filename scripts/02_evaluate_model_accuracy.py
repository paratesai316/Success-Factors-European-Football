import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
SEASONS = ['23_24', '24_25'] 

LEAGUES = {
    'Premier League': 'pl',
    'La Liga': 'la_liga',
    'Serie A': 'serie_a',
    'Bundesliga': 'bundesliga'
}

features_to_analyze = [
    'goals', 'assists', 'shots_total', 'shots_on_target', 'expected_goals', 'expected_assists',
    'progressive_passes', 'progressive_carries', 'touches', 'tackles_total', 'interceptions_misc',
    'ball_recoveries', 'aerial_duels_won', 'aerial_duels_lost', 'passes_completed_total',
    'passes_attempted_total', 'key_passes', 'progressive_passing_distance', 'progressive_carry_distance',
    'successful_take_ons', 'miscontrols', 'times_dispossessed', 'shot_creating_actions',
    'goal_creating_actions', 'yellow_cards', 'red_cards', 'fouls_committed', 'fouls_drawn',
    'offsides', 'clearances', 'blocks_defensive', 'passes_blocked'
]

tuning_grids = {
    'Ridge': {
        'model': Ridge(random_state=42),
        'params': {'regressor__alpha': [0.1, 1.0, 10.0, 50.0, 100.0]} 
    },
    'Lasso': { 
        'model': Lasso(max_iter=10000, random_state=42),
        'params': {'regressor__alpha': [0.01, 0.1, 1.0, 5.0, 10.0]}
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'regressor__n_estimators': [50, 100],
            'regressor__max_depth': [3, 5, 7], 
            'regressor__min_samples_leaf': [2, 4]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42),
        'params': {
            'regressor__n_estimators': [50, 100],
            'regressor__max_depth': [2, 3], 
            'regressor__learning_rate': [0.01, 0.05, 0.1]
        }
    },
    'KNN': {
        'model': KNeighborsRegressor(),
        'params': {
            'regressor__n_neighbors': [3, 5, 7],
            'regressor__weights': ['uniform', 'distance']
        }
    }
}

model_performances = []

print("--- EVALUATING MODEL ACCURACY (R²) FOR 23/24 & 24/25 ---")

for league_name, prefix in LEAGUES.items():
    league_datasets = []

    for season in SEASONS:
        stats_file = os.path.join(BASE_DIR, season, f"{prefix}_{season}.csv")
        table_file = os.path.join(BASE_DIR, season, f"{prefix}_table_{season}.csv")

        if not os.path.exists(stats_file) or not os.path.exists(table_file): 
            continue

        stats_df = pd.read_csv(stats_file, encoding='latin1')
        table_df = pd.read_csv(table_file, encoding='latin1')

        stats_df.columns = stats_df.columns.str.lower().str.strip()
        table_df.columns = table_df.columns.str.lower().str.strip()

        if 'squad' in stats_df.columns: stats_df.rename(columns={'squad': 'club'}, inplace=True)
        elif 'team' in stats_df.columns: stats_df.rename(columns={'team': 'club'}, inplace=True)

        if 'club' in table_df.columns: table_df.rename(columns={'club': 'squad'}, inplace=True)
        elif 'team' in table_df.columns: table_df.rename(columns={'team': 'squad'}, inplace=True)

        stats_df['club'] = stats_df['club'].astype(str).str.strip()
        table_df['squad'] = table_df['squad'].astype(str).str.strip()

        available_cols = [c for c in features_to_analyze if c.lower() in stats_df.columns]
        team_stats = stats_df.groupby('club')[available_cols].sum().reset_index()
        merged = pd.merge(team_stats, table_df[['squad', 'pts']], left_on='club', right_on='squad')

        if not merged.empty:
            league_datasets.append(merged)

    if not league_datasets:
        print(f"No data found for {league_name}.")
        continue

    combined_league_df = pd.concat(league_datasets, ignore_index=True)
    X = combined_league_df.drop(['club', 'squad', 'pts'], axis=1).fillna(0)
    y = combined_league_df['pts']

    print(f"\nEvaluating {league_name} (Total rows: {len(y)})...")

    for model_name, config in tuning_grids.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', config['model'])
        ])

        search = GridSearchCV(pipeline, config['params'], cv=4, scoring='r2', n_jobs=-1)
        search.fit(X, y)

        best_score = search.best_score_

        model_performances.append({
            'League': league_name, 
            'Model': model_name, 
            'Optimized_R2': best_score
        })
        print(f"  > {model_name} R²: {best_score:.3f}")

performance_df = pd.DataFrame(model_performances)
print("\n--- FINAL MODEL PERFORMANCE COMPARISON (23/24 & 24/25) ---")
pivot_perf = performance_df.pivot(index='League', columns='Model', values='Optimized_R2').round(3)
print(pivot_perf)

performance_df.to_csv('model_accuracy_23_to_25.csv', index=False)
print("\nSaved R² results to 'model_accuracy_23_to_25.csv'.")