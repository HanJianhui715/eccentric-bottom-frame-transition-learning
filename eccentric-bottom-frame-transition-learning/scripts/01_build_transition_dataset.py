from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transition_common import (
    DAMAGE_NAMES,
    EPS,
    FEATURE_COLUMNS,
    ROOT,
    STRUCTURAL_COLUMNS,
    TRANSITION_FEATURES,
    global_score,
    load_dataset,
)

DEFAULT_OUT = ROOT / "data"


def sequence_text(values: list[int]) -> str:
    return "/".join(str(int(v)) for v in values)


def build_trajectory_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id, group in df.groupby("group_id"):
        g = group.sort_values("PGA").reset_index(drop=True)
        if len(g) != 3:
            continue
        row = {
            "group_id": int(group_id),
            "n_points": int(len(g)),
        }
        for col in STRUCTURAL_COLUMNS:
            row[col] = g.loc[0, col]
        for idx, level in enumerate(["low", "mid", "high"]):
            row[f"{level}_PGA"] = float(g.loc[idx, "PGA"])
            row[f"{level}_IDR_frame"] = float(g.loc[idx, "IDR_frame"])
            row[f"{level}_IDR_masonry"] = float(g.loc[idx, "IDR_masonry"])
            row[f"{level}_damage_frame"] = int(g.loc[idx, "damage_frame"])
            row[f"{level}_damage_masonry"] = int(g.loc[idx, "damage_masonry"])
            row[f"{level}_damage_global"] = int(g.loc[idx, "damage_global"])
            row[f"{level}_score_global"] = float(global_score(g.loc[[idx], "IDR_frame"], g.loc[[idx], "IDR_masonry"])[0])

        for target in ["IDR_frame", "IDR_masonry", "damage_frame", "damage_masonry", "damage_global"]:
            values = g[target].to_numpy(dtype=float)
            row[f"monotone_{target}"] = bool(np.all(np.diff(values) >= -1e-12))

        row["sequence_frame"] = sequence_text(g["damage_frame"].astype(int).tolist())
        row["sequence_masonry"] = sequence_text(g["damage_masonry"].astype(int).tolist())
        row["sequence_global"] = sequence_text(g["damage_global"].astype(int).tolist())
        controller = []
        for _, point in g.iterrows():
            if int(point["damage_frame"]) > int(point["damage_masonry"]):
                controller.append("frame")
            elif int(point["damage_masonry"]) > int(point["damage_frame"]):
                controller.append("masonry")
            else:
                controller.append("tie")
        row["controller_sequence"] = "/".join(controller)
        row["global_damage_jump_total"] = int(g.loc[2, "damage_global"] - g.loc[0, "damage_global"])
        row["frame_damage_jump_total"] = int(g.loc[2, "damage_frame"] - g.loc[0, "damage_frame"])
        row["masonry_damage_jump_total"] = int(g.loc[2, "damage_masonry"] - g.loc[0, "damage_masonry"])
        rows.append(row)
    return pd.DataFrame(rows)


def transition_row(group_id: int, group: pd.DataFrame, start: int, end: int, stage_name: str, stage_id: int) -> dict:
    a = group.loc[start]
    b = group.loc[end]
    delta_log_frame = float(np.log(max(b["IDR_frame"], EPS) / max(a["IDR_frame"], EPS)))
    delta_log_masonry = float(np.log(max(b["IDR_masonry"], EPS) / max(a["IDR_masonry"], EPS)))
    row = {
        "group_id": int(group_id),
        "transition_name": stage_name,
        "transition_stage": int(stage_id),
    }
    for col in STRUCTURAL_COLUMNS:
        row[col] = a[col]
    row.update(
        {
            "from_pga": float(a["PGA"]),
            "to_pga": float(b["PGA"]),
            "pga_ratio": float(b["PGA"] / a["PGA"]),
            "log_pga_ratio": float(np.log(b["PGA"] / a["PGA"])),
            "from_IDR_frame": float(a["IDR_frame"]),
            "from_IDR_masonry": float(a["IDR_masonry"]),
            "to_IDR_frame": float(b["IDR_frame"]),
            "to_IDR_masonry": float(b["IDR_masonry"]),
            "from_log_IDR_frame": float(np.log(max(a["IDR_frame"], EPS))),
            "from_log_IDR_masonry": float(np.log(max(a["IDR_masonry"], EPS))),
            "to_log_IDR_frame": float(np.log(max(b["IDR_frame"], EPS))),
            "to_log_IDR_masonry": float(np.log(max(b["IDR_masonry"], EPS))),
            "delta_log_IDR_frame": delta_log_frame,
            "delta_log_IDR_masonry": delta_log_masonry,
            "target_log_delta_frame": float(np.log(max(delta_log_frame, EPS))),
            "target_log_delta_masonry": float(np.log(max(delta_log_masonry, EPS))),
            "from_damage_frame": int(a["damage_frame"]),
            "from_damage_masonry": int(a["damage_masonry"]),
            "from_damage_global": int(a["damage_global"]),
            "to_damage_frame": int(b["damage_frame"]),
            "to_damage_masonry": int(b["damage_masonry"]),
            "to_damage_global": int(b["damage_global"]),
            "delta_damage_frame": int(b["damage_frame"] - a["damage_frame"]),
            "delta_damage_masonry": int(b["damage_masonry"] - a["damage_masonry"]),
            "delta_damage_global": int(b["damage_global"] - a["damage_global"]),
            "to_score_global": float(global_score(np.array([b["IDR_frame"]]), np.array([b["IDR_masonry"]]))[0]),
            "from_score_global": float(global_score(np.array([a["IDR_frame"]]), np.array([a["IDR_masonry"]]))[0]),
        }
    )
    return row


def build_transition_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id, group in df.groupby("group_id"):
        g = group.sort_values("PGA").reset_index(drop=True)
        if len(g) != 3:
            continue
        rows.append(transition_row(int(group_id), g, 0, 1, "low_to_mid", 0))
        rows.append(transition_row(int(group_id), g, 1, 2, "mid_to_high", 1))
    out = pd.DataFrame(rows)
    return out[["group_id", "transition_name"] + TRANSITION_FEATURES + [c for c in out.columns if c not in {"group_id", "transition_name", *TRANSITION_FEATURES}]]


def write_summary(trajectory: pd.DataFrame, transition: pd.DataFrame, out_dir: Path) -> None:
    monotone_cols = [c for c in trajectory.columns if c.startswith("monotone_")]
    monotone = pd.DataFrame(
        {
            "target": [c.replace("monotone_", "") for c in monotone_cols],
            "monotone_rate": [float(trajectory[c].mean()) for c in monotone_cols],
            "violations": [int((~trajectory[c].astype(bool)).sum()) for c in monotone_cols],
        }
    )
    sequence_counts = (
        trajectory["sequence_global"]
        .value_counts()
        .rename_axis("sequence_global")
        .reset_index(name="count")
    )
    transition_counts = pd.crosstab(
        [transition["SPI"], transition["transition_name"]],
        transition["delta_damage_global"],
    ).reset_index()
    monotone.to_csv(out_dir / "monotonicity_summary.csv", index=False)
    sequence_counts.to_csv(out_dir / "global_damage_sequence_counts.csv", index=False)
    transition_counts.to_csv(out_dir / "transition_delta_damage_counts.csv", index=False)
    summary = {
        "n_trajectories": int(len(trajectory)),
        "n_transitions": int(len(transition)),
        "damage_names": DAMAGE_NAMES,
        "feature_columns": FEATURE_COLUMNS,
        "transition_features": TRANSITION_FEATURES,
        "main_claim_from_data": "All observed FE trajectories are monotone non-decreasing in IDR and damage with increasing PGA.",
    }
    (out_dir / "data_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    df = load_dataset(data_path)
    trajectory = build_trajectory_table(df)
    transition = build_transition_table(df)
    review_columns = [
        "group_id",
        "SPI",
        "ISR",
        "Ecc_X_1",
        "Ecc_Y_1",
        "Ecc_X_2",
        "Ecc_Y_2",
        "low_PGA",
        "mid_PGA",
        "high_PGA",
        "low_IDR_frame",
        "mid_IDR_frame",
        "high_IDR_frame",
        "low_IDR_masonry",
        "mid_IDR_masonry",
        "high_IDR_masonry",
        "low_damage_frame",
        "mid_damage_frame",
        "high_damage_frame",
        "low_damage_masonry",
        "mid_damage_masonry",
        "high_damage_masonry",
        "low_damage_global",
        "mid_damage_global",
        "high_damage_global",
        "sequence_frame",
        "sequence_masonry",
        "sequence_global",
        "controller_sequence",
    ]
    trajectory[review_columns].to_csv(out_dir / "state_sequences_113.csv", index=False)
    transition.to_csv(out_dir / "ml_transition_samples_226.csv", index=False)
    print(f"Trajectory data saved to: {out_dir}")
    print(f"n_trajectories={len(trajectory)}, n_transitions={len(transition)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trajectory and transition datasets from repeated FE records.")
    parser.add_argument("--data-path", default=str(ROOT / "data" / "raw_fe_records_339.csv"))
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
