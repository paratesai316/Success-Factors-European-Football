# A Regularized Regression Analysis of Tactical Success Factors Across Europe's Top Football Leagues

This repository contains the dataset and complete Python machine learning pipeline for the study: **"A Regularized Regression Analysis of Tactical Success Factors Across Europe's Top Football Leagues."**

The project aims to empirically identify the specific tactical actions (e.g., progressive passes, aerial duels, goal-creating actions) that most heavily dictate a team's success (measured in points) across Europe’s top four leagues:

1. The English Premier League
2. Spanish La Liga
3. Italian Serie A
4. German Bundesliga

By evaluating five distinct machine learning algorithms (Ridge, Lasso, Random Forest, XGBoost, and KNN), this study determines which mathematical model best captures the tactical identity of each league and extracts the corresponding success factors.

## Repository Structure

The repository is organized to ensure full reproducibility:

* **`/data`**: Contains the raw, aggregated player and squad statistics sourced from FBref.com, organized by season (2024/2025, 2023/2024, and 2022/2023).
* **`/scripts`**: Contains the Python pipeline used to clean the data, train the models, evaluate accuracy, and extract feature importances.
* **`/outputs`**: Stores the generated tactical attribute charts (.png) and the exported results (.csv).
* **`requirements.txt`**: Lists the necessary Python packages to run the environment.

## Installation & Setup

To reproduce the findings, please ensure you have Python 3.8+ installed.

1. Clone this repository to your local machine
2. Install the required dependencies using the requirements.txt file:

    ```pip install -r requirements.txt```

## How to Run the Pipeline

The codebase is split into sequential scripts to demonstrate the methodology clearly. Navigate to the /scripts folder and run them in the following order:

1. **Extract Success Factors**

    Run ```01_extract_success_factors.py``` to train the optimized models and extract the normalized success coefficients (feature importances) for all 32 tactical metrics.
    * Purpose: Generates the precise weights mapping on-pitch actions to league success.

2. **Evaluate Model Accuracy**

    Run ```02_evaluate_model_accuracy.py``` to calculate the cross-validated $R^2$ scores for all five models across the specified seasons.
    * Purpose: This script proves mathematically which algorithm is best suited to predict point totals for each specific league (e.g., identifying that XGBoost best models Serie A, while Lasso best models the Premier League).

## Key Findings

* **Premier League** (evaluated using **Lasso Regression**): focuses on fast-paced, physicality and chaos in the game.

  * Factors like expected goals (0.1651) and assists (0.2405) and progressive passing (0.1239) shows direct translation of high attack and passing leads to victories.
  * Tactical fouls indicated by red (0.0716) and yellow (0.0698) cards show chaos.

* **Serie A** (evaluated using **XGBoost**): focuses on tactical complexity and defensive structures.

  * Progressive passes (0.3518), successful take-ons (0.0121) and total shots (0.0189) demonstrate tactical complexity.
  * Fouls drawn (0.0853), times dispossessed (0.0421), blocks (0.0294) and tackles (0.0097) give insight into defense solidarity.

* **La Liga** (evaluated using **Ridge Regression**): focuses on control, rhythm and tempo as well as high fouling and aggressive culture.

  * Passing distance (0.0387), passes attempted (0.0307) and completed (0.0302) and carry distance (0.0225) show dictation of control.
  * Yellow cards (0.0567), fouls committed (0.0512) and fouls drawn (0.0346) give indication of aggressive behavior and tactical fouling.

* **Bundesliga** (evaluated using **Ridge Regression**): focuses on pressing and ball recovery along with transitional-play.

  * Tackles (0.0414) and fouls committed (0.0394) give the view of pressing.
  * Passing distance (0.0340), progressive passes (0.0469) and goal creating actions (0.0358) describe the transitional-play.

## Data Source

All raw datasets provided in the /data directory were sourced from [FBref.com](https://www.fbref.com "Football Statistics and History | FBref.com").
