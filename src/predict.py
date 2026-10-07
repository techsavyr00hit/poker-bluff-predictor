"""Interactive prediction script."""

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/bluff_model.joblib")


def ask_float(message, minimum, maximum):
    while True:
        try:
            value = float(input(message))
            if minimum <= value <= maximum:
                return value
        except ValueError:
            pass
        print(f"Please enter a number between {minimum} and {maximum}.")


def ask_int(message, minimum, maximum):
    while True:
        try:
            value = int(input(message))
            if minimum <= value <= maximum:
                return value
        except ValueError:
            pass
        print(f"Please enter an integer between {minimum} and {maximum}.")


def ask_choice(message, choices):
    while True:
        value = input(f"{message} ({'/'.join(choices)}): ").strip().lower()
        if value in choices:
            return value
        print("Please choose one of the listed options.")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python src/train.py` first.")

    model = joblib.load(MODEL_PATH)

    print("\nOpponent Bluff Probability Engine")
    print("Enter the current opponent situation.\n")

    row = {
        "street": ask_choice("Street", ["flop", "turn", "river"]),
        "position": ask_choice("Opponent position", ["early", "middle", "late", "blind"]),
        "board_texture": ask_choice("Board texture", ["dry", "semi_wet", "wet"]),
        "bet_to_pot": ask_float("Bet as a multiple of the pot (0.15-3.0): ", 0.15, 3.0),
        "opponent_aggression": ask_float("Opponent aggression (0-1): ", 0, 1),
        "historical_bluff_rate": ask_float("Historical bluff rate (0-1): ", 0, 1),
        "previous_raises": ask_int("Previous raises this hand (0-5): ", 0, 5),
        "recent_losses": ask_int("Recent losses (0-5): ", 0, 5),
        "estimated_hand_strength": ask_float("Estimated hand strength (0-1): ", 0, 1),
    }

    x = pd.DataFrame([row])
    probability = model.predict_proba(x)[0, 1]

    print(f"\nEstimated bluff probability: {probability:.1%}")
    if probability >= 0.70:
        print("Interpretation: fairly high bluff probability")
    elif probability >= 0.40:
        print("Interpretation: uncertain / mixed situation")
    else:
        print("Interpretation: relatively low bluff probability")

    print("\nThis is a model estimate, not a guaranteed poker decision.")


if __name__ == "__main__":
    main()
