from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "SPI",
    "ISR",
    "Ecc_X_1",
    "Ecc_Y_1",
    "Ecc_X_2",
    "Ecc_Y_2",
    "PGA",
]

TARGET_COLUMNS = ["IDR_frame", "IDR_masonry"]

COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS

DAMAGE_NAMES = ["intact", "slight", "moderate", "severe", "destructive"]
DAMAGE_LEVELS = list(range(len(DAMAGE_NAMES)))

FRAME_BOUNDS = np.array([1 / 800, 1 / 400, 1 / 150, 1 / 100], dtype=float)
MASONRY_BOUNDS = np.array([1 / 2500, 1 / 900, 1 / 200, 1 / 150], dtype=float)


PGA_BY_SPI = {
    6: [18.0, 50.0, 125.0],
    7: [35.0, 100.0, 220.0],
    8: [70.0, 200.0, 400.0],
}


def assign_damage(values: np.ndarray | pd.Series, bounds: np.ndarray) -> np.ndarray:
    """Map drift ratios to integer damage labels.

    The lower bound is included in the more severe class. For example,
    IDR == 1/800 is slight damage for the bottom frame story.
    """

    arr = np.asarray(values, dtype=float)
    return np.searchsorted(bounds, arr, side="right").astype(int)


def add_damage_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["damage_frame"] = assign_damage(out["IDR_frame"], FRAME_BOUNDS)
    out["damage_masonry"] = assign_damage(out["IDR_masonry"], MASONRY_BOUNDS)
    out["damage_global"] = np.maximum(out["damage_frame"], out["damage_masonry"])
    return out


def add_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = FEATURE_COLUMNS[:6]
    keys = out[group_cols].apply(lambda row: tuple(row.to_list()), axis=1)
    out["group_key"] = keys
    out["group_id"] = pd.factorize(keys, sort=True)[0]
    return out


def load_dataset(path: str | Path = "data/raw_fe_records_339.csv") -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    if list(df.columns) != COLUMNS:
        df = pd.read_csv(path, header=None, names=COLUMNS)
    df = add_damage_columns(df)
    df = add_group_columns(df)
    return df


def damage_count_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in [
        ("damage_frame", "bottom_frame"),
        ("damage_masonry", "masonry"),
        ("damage_global", "global"),
    ]:
        counts = df[col].value_counts().reindex(DAMAGE_LEVELS, fill_value=0)
        row = {"target": label}
        row.update({DAMAGE_NAMES[i]: int(counts.loc[i]) for i in DAMAGE_LEVELS})
        rows.append(row)
    return pd.DataFrame(rows)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].to_numpy(dtype=float)


def damage_labels_from_idr(
    frame_idr: np.ndarray,
    masonry_idr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = assign_damage(frame_idr, FRAME_BOUNDS)
    masonry = assign_damage(masonry_idr, MASONRY_BOUNDS)
    global_damage = np.maximum(frame, masonry)
    return frame, masonry, global_damage


def probability_from_labels(labels: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Convert labels shaped (n_members, n_samples) into class probabilities."""

    labels = np.asarray(labels, dtype=int)
    probs = np.zeros((labels.shape[1], n_classes), dtype=float)
    for cls in range(n_classes):
        probs[:, cls] = np.mean(labels == cls, axis=0)
    return probs


def exceedance_probabilities(probs: np.ndarray) -> np.ndarray:
    """Return P(Damage >= level) for levels 1..4."""

    probs = np.asarray(probs, dtype=float)
    return np.column_stack([probs[:, level:].sum(axis=1) for level in range(1, 5)])
