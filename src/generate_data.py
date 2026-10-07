"""Generate a simple synthetic poker hand-history dataset."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def generate(rows: int) -> pd.DataFrame:
    streets = RNG.choice(["flop", "turn", "river"], rows, p=[0.35, 0.35, 0.30])
    positions = RNG.choice(["early", "middle", "late", "blind"], rows)
    textures = RNG.choice(["dry", "semi_wet", "wet"], rows, p=[0.4, 0.35, 0.25])

    bet_to_pot = np.clip(RNG.lognormal(mean=-0.05, sigma=0.55, size=rows), 0.15, 3.0)
    aggression = RNG.beta(3, 2, rows)
    bluff_rate = RNG.beta(2.2, 4.5, rows)
    previous_raises = RNG.poisson(1.1, rows).clip(0, 5)
    recent_losses = RNG.poisson(1.0, rows).clip(0, 5)
    hand_strength = RNG.beta(3.0, 2.0, rows)

    # This is only a rough simulation of human betting behaviour.
    # It gives the model a learnable pattern without pretending that this
    # synthetic data represents a real poker population.
    street_bonus = np.where(streets == "river", 0.20, np.where(streets == "turn", 0.08, -0.05))
    texture_bonus = np.where(textures == "dry", 0.12, np.where(textures == "wet", -0.10, 0.0))
    position_bonus = np.where(positions == "late", 0.10, np.where(positions == "early", -0.08, 0.0))

    score = (
        -1.4
        + 1.8 * bluff_rate
        + 0.75 * aggression
        + 0.42 * np.log1p(bet_to_pot)
        + 0.15 * previous_raises
        + 0.12 * recent_losses
        - 2.1 * hand_strength
        + street_bonus
        + texture_bonus
        + position_bonus
    )

    probability = sigmoid(score)
    is_bluff = RNG.binomial(1, probability)

    return pd.DataFrame(
        {
            "street": streets,
            "position": positions,
            "board_texture": textures,
            "bet_to_pot": np.round(bet_to_pot, 3),
            "opponent_aggression": np.round(aggression, 3),
            "historical_bluff_rate": np.round(bluff_rate, 3),
            "previous_raises": previous_raises,
            "recent_losses": recent_losses,
            "estimated_hand_strength": np.round(hand_strength, 3),
            "is_bluff": is_bluff,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--output", default="data/poker_hands.csv")
    args = parser.parse_args()

    if args.rows < 100:
        raise ValueError("Please generate at least 100 rows.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} rows -> {output}")
    print(f"Bluff rate in dataset: {df['is_bluff'].mean():.2%}")


if __name__ == "__main__":
    main()
