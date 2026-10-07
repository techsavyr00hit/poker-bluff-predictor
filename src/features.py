"""Feature preparation for the bluff prediction model."""

import pandas as pd

NUMERIC_COLUMNS = [
    "bet_to_pot",
    "opponent_aggression",
    "historical_bluff_rate",
    "previous_raises",
    "recent_losses",
    "estimated_hand_strength",
]

CATEGORICAL_COLUMNS = [
    "street",
    "position",
    "board_texture",
]

TARGET_COLUMN = "is_bluff"


def split_features_target(df: pd.DataFrame):
    """Return X and y after doing basic validation."""
    required = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    x = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)
    return x, y
