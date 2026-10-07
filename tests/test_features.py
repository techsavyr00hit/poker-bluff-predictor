import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import split_features_target


def test_feature_split():
    df = pd.DataFrame(
        {
            "street": ["river"],
            "position": ["late"],
            "board_texture": ["dry"],
            "bet_to_pot": [1.0],
            "opponent_aggression": [0.7],
            "historical_bluff_rate": [0.4],
            "previous_raises": [1],
            "recent_losses": [2],
            "estimated_hand_strength": [0.3],
            "is_bluff": [1],
        }
    )

    x, y = split_features_target(df)
    assert x.shape == (1, 9)
    assert y.iloc[0] == 1
