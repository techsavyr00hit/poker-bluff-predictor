# Opponent Bluff Probability Engine

A small machine learning project that estimates how likely a poker opponent is to be bluffing based on the current hand and their previous betting behaviour.

This is meant to be a practical college-level project, not a full poker solver. The model learns from hand-history style data and returns a bluff probability for a new situation.

## What it does

The project uses features such as:

- opponent aggression
- previous bluff rate
- bet size compared with the pot
- street (flop, turn, river)
- opponent position
- number of previous raises
- board texture
- estimated hand strength
- whether the opponent has recently lost hands

A Random Forest classifier is used because it is fairly easy to train, works well with mixed numeric/categorical features after encoding, and is easy to explain.

## Tech stack

- Python 3.10+
- pandas
- NumPy
- scikit-learn
- joblib

No web app, database, Docker, or deep learning framework is needed.

## Project structure

```text
opponent_bluff_probability_engine/
├── data/
│   └── poker_hands.csv
├── models/
├── src/
│   ├── generate_data.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── tests/
│   └── test_features.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```text
.venv\Scripts\activate
```

## 1. Generate a dataset

The repository includes a sample dataset, but you can create a larger one with:

```bash
python src/generate_data.py --rows 20000
```

The generated data is synthetic. That is intentional: real poker hand histories are harder to obtain and often come in different formats.

## 2. Train the model

```bash
python src/train.py
```

The trained model is saved to `models/bluff_model.joblib`.

## 3. Evaluate it

```bash
python src/evaluate.py
```

This prints accuracy, precision, recall, F1 score and a confusion matrix.

## 4. Predict bluff probability

Run:

```bash
python src/predict.py
```

The program asks for a few details about the current situation and prints something like:

```text
Estimated bluff probability: 68.4%

Interpretation: fairly high bluff probability
```

## Important limitation

The model does **not** know the opponent's actual cards. The target is based on the labelled training data. In a real poker system, the quality of the result would depend heavily on the quality and size of the hand-history data.

Also, a bluff probability is not the same thing as a recommendation to call. A real decision would also need pot odds, your own hand equity, stack sizes, and other information.

## Possible improvements

Some things I would add in a bigger version:

1. Train on real hand histories.
2. Add player-specific models when enough hands are available.
3. Calibrate probabilities more carefully.
4. Add SHAP explanations.
5. Add opponent clustering for playing style.
6. Build a small Streamlit interface.

For the current version, the goal is to keep the project understandable and easy to explain in a viva.
