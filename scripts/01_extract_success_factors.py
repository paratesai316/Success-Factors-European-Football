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

model_configs = {
    'Ridge': (Ridge(random_state=42), {'regressor__alpha': [0.1, 1.0, 10.0, 50.0]}),
    'Lasso': (Lasso(random_state=42, max_iter=10000), {'regressor__alpha': [0.01, 0.1, 1.0]}),
    'RandomForest': (RandomForestRegressor(random_state=42), {'regressor__max_depth': [3, 5], 'regressor__n_estimators': [50, 100]}),
    'XGBoost': (XGBRegressor(random_state=42), {'regressor__max_depth': [2, 3], 'regressor__learning_rate': [0.05, 0.1]}),
    'KNN': (KNeighborsRegressor(), {'regressor__n_neighbors': [3, 5], 'regressor__weights': ['distance']})
}

def extract_importances(best_model, X, y):
    regressor = best_model.named_steps['regressor']

    if hasattr(regressor, 'feature_importances_'):
        vals = regressor.feature_importances_
    elif hasattr(regressor, 'coef_'):
        vals = np.abs(regressor.coef_)
    else:

        r = permutation_importance(best_model, X, y, n_repeats=5, random_state=42)
        vals = r.importances_mean

    vals = np.maximum(vals, 0)
    if np.sum(vals) > 0:
        vals = vals / np.sum(vals)
    return pd.Series(vals, index=X.columns)

print("--- STARTING ALL-MODEL INDIVIDUAL SEASON ANALYSIS ---")

for league_name, prefix in LEAGUES.items():
    print(f"\nProcessing {league_name}...")

    league_results = {}

    for season in SEASONS:
        stats_file = os.path.join(BASE_DIR, season, f"{prefix}_{season}.csv")
        table_file = os.path.join(BASE_DIR, season, f"{prefix}_table_{season}.csv")

        if not (os.path.exists(stats_file) and os.path.exists(table_file)):
            print(f"  > Missing data for {season}, skipping.")
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

        if merged.empty:
            continue

        X = merged.drop(['club', 'squad', 'pts'], axis=1).fillna(0)
        y = merged['pts']

        print(f"  > Training models for {season}...")

        for model_name, (model_obj, params) in model_configs.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', model_obj)
            ])

            search = GridSearchCV(pipeline, params, cv=3, scoring='r2', n_jobs=-1)
            search.fit(X, y)

            importances = extract_importances(search.best_estimator_, X, y)

            column_name = f"{season}_{model_name}"
            league_results[column_name] = importances

    if league_results:
        league_df = pd.DataFrame(league_results).fillna(0)

        league_df['League_Overall_Average'] = league_df.mean(axis=1)
        league_df = league_df.sort_values(by='League_Overall_Average', ascending=False)

        safe_league_name = league_name.replace(" ", "_")
        output_filename = f"{safe_league_name}_all_models_success_factors.csv"

        league_df.to_csv(output_filename)
        print(f"  [SUCCESS] Saved {output_filename} with {len(league_df.columns)-1} model-season combinations.")

print("\n--- ALL ANALYSES COMPLETE ---")