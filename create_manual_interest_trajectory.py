import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a trajectory-style version of annotations_manual_interest.csv.\n"
            "For each sample, generate 4 preceding CUR tile positions around the current CUR\n"
            "to simulate an approximate UAV flight path (they do not have to lie exactly\n"
            "on the vector from CUR to TAR). The new CSV will contain extra columns:\n"
            "prev1_tile_x/y, ..., prev4_tile_x/y."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/annotations_manual_interest.csv",
        help="Path to the original manual_interest annotations CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/annotations_manual_interest_traj.csv",
        help="Path to save the augmented trajectory CSV.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=2000,
        help="Map crop size in pixels (must match map_size used when annotations were created).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=200,
        help="Tile size in pixels (for clamping prev tiles inside the map).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=300.0,
        help="Base distance (in pixels) between consecutive previous tiles along the approximate path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for small perturbations of the path.",
    )
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def main() -> None:
    args = parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input annotations CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    if df.empty:
        raise RuntimeError(f"No rows in {in_path}")

    # Filter manual_interest rows if split column exists
    if "split" in df.columns:
        mask = df["split"] == "manual_interest"
        if mask.any():
            df = df[mask].reset_index(drop=True)

    map_size = float(args.map_size)
    tile_size = float(args.tile_size)
    half_tile = tile_size / 2.0

    rng = np.random.default_rng(args.seed)

    prev_cols_x = [f"prev{i}_tile_x" for i in range(1, 5)]
    prev_cols_y = [f"prev{i}_tile_y" for i in range(1, 5)]

    # Initialize new columns with NaNs
    for c in prev_cols_x + prev_cols_y:
        df[c] = np.nan

    for idx, row in df.iterrows():
        cur_x = float(row["cur_tile_x"])
        cur_y = float(row["cur_tile_y"])
        tar_x = float(row["tar_tile_x"])
        tar_y = float(row["tar_tile_y"])

        dx = tar_x - cur_x
        dy = tar_y - cur_y
        norm = np.hypot(dx, dy)

        if norm < 1e-6:
            # If vector is degenerate, use a default direction (to the right)
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / norm, dy / norm

        # We will go "backwards" from cur along -direction, with small perpendicular jitter
        # to make path not lie exactly on the vector.
        # Perpendicular direction:
        px, py = -uy, ux

        for i in range(4):
            dist = args.step * (i + 1)
            # Base position: behind current along -u
            bx = cur_x - ux * dist
            by = cur_y - uy * dist

            # Small sideways jitter (± step/4 along perpendicular)
            jitter = args.step / 4.0
            side = rng.uniform(-1.0, 1.0)
            bx += px * side * jitter
            by += py * side * jitter

            # Clamp so full tile fits inside map
            bx = clamp(bx, half_tile, map_size - half_tile)
            by = clamp(by, half_tile, map_size - half_tile)

            df.at[idx, prev_cols_x[i]] = bx
            df.at[idx, prev_cols_y[i]] = by

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(
        f"Augmented trajectory annotations written to {out_path}\n"
        f"Rows: {len(df)}; added columns: {', '.join(prev_cols_x + prev_cols_y)}"
    )


if __name__ == "__main__":
    main()


