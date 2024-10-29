# Chess Outcome Prediction Project

## Project Overview

The goal of this project is to develop predictive models for chess outcomes, focusing on three key aspects:

1. **Chess Move Prediction Model**: Predicts the next best move in a chess game based on the current board position using classification techniques.
2. **Endgame Outcome Prediction**: Predicts the final outcome of a chess game (win, loss, or draw) based on the state of the board during the endgame phase.
3. **Win Probability Prediction**: Estimates the probability of winning as the game progresses, framed as a regression problem.

The models utilize various machine learning techniques, including Random Forest and Gradient Boosting, with hyperparameter tuning and cross-validation to enhance performance and robustness.

## Data Source

The dataset used in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/chess%3A+King+%26+Rook+vs.+King+%26+Pawn), specifically the chess dataset titled "Chess: King & Rook vs. King & Pawn". 

### Key Attributes of the Dataset:

- **Features**: Attributes of a chess position, such as the arrangement of pieces and the player's turn.
- **Target Variable**: The outcome of the game, classified into categories like "win," "loss," or "draw."

## Requirements

To run this project, you will need the following Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `joblib`
- `imbalanced-learn`

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib joblib imbalanced-learn
