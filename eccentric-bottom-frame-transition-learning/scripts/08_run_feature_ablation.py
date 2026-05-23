from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from transition_common import (
    DAMAGE_NAMES,
    EPS,
    ROOT,
    TRANSITION_FEATURES,
    damage_labels_from_idr,
    dependency_status,
    global_score,
    gpu_status,
    import_tabpfn,
    make_group_splits,
    make_regressor,
    parse_csv_list,
    point_damage_metrics,
    rmse,
    safe_r2,
    stable_seed,
)

DATA_DIR = ROOT / "data"
DEFAULT_OUT = ROOT / "results" / "feature_ablation"

STRUCTURAL = ["SPI", "ISR", "Ecc_X_1", "Ecc_Y_1", "Ecc_X_2", "Ecc_Y_2"]
PGA_TRANSFER = ["from_pga", "to_pga", "pga_ratio", "log_pga_ratio", "transition_stage"]
CURRENT_IDR = ["from_log_IDR_frame", "from_log_IDR_masonry"]
CURRENT_DAMAGE = ["from_damage_frame", "from_damage_masonry", "from_damage_global"]

FEATURE_SETS = {
    "full_16": TRANSITION_FEATURES,
    "reduced_no_redundant": [
        "SPI",
        "ISR",
        "Ecc_X_1",
        "Ecc_Y_1",
        "Ecc_X_2",
        "Ecc_Y_2",
        "from_pga",
        "to_pga",
        "from_log_IDR_frame",
        "from_log_IDR_masonry",
        "from_damage_frame",
        "from_damage_masonry",
        "transition_stage",
    ],
    "no_current_state": STRUCTURAL + PGA_TRANSFER,
    "no_current_idr": [c for c in TRANSITION_FEATURES if c not in CURRENT_IDR],
    "no_current_damage": [c for c in TRANSITION_FEATURES if c not in CURRENT_DAMAGE],
    "no_structural": [c for c in TRANSITION_FEATURES if c not in STRUCTURAL],
    "no_pga_transfer": [c for c in TRANSITION_FEATURES if c not in PGA_TRANSFER],
}

MODEL_LABELS = {"tabpfn": "TabPFN", "rf": "RF", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
COLORS = {
    "tabpfn": "#2F6FA3",
    "rf": "#4F8A45",
    "lightgbm": "#D9822B",
    "xgboost": "#6F5AA6",
}


def ensure_data_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    transition_path = DATA_DIR / "ml_transition_samples_226.csv"
    trajectory_path = DATA_DIR / "state_sequences_113.csv"
    if not transition_path.exists() or not trajectory_path.exists():
        raise FileNotFoundError("Run 01_build_transition_dataset.py before feature ablation.")
    return pd.read_csv(trajectory_path), pd.read_csv(transition_path)


def selected_feature_sets(text: str) -> dict[str, list[str]]:
    names = parse_csv_list(text)
    unknown = [name for name in names if name not in FEATURE_SETS]
    if unknown:
        raise ValueError(f"Unknown feature sets: {unknown}. Available: {sorted(FEATURE_SETS)}")
    return {name: FEATURE_SETS[name] for name in names}


def fit_delta_models(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    seed_tag: str,
    TabPFNRegressor=None,
) -> list:
    models = []
    for member in range(args.members):
        seed = stable_seed(model_name, seed_tag, member, base=args.random_state)
        model = make_regressor(model_name, seed, args, TabPFNRegressor=TabPFNRegressor)
        model.fit(X_train, y_train)
        setattr(model, "_codex_predict_batch_size", int(args.predict_batch_size or 8))
        models.append(model)
    return models


def predict_with_delta_models(models: list, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    preds = []
    batch_size = int(getattr(models[0], "_codex_predict_batch_size", 8) or 8)
    for model in models:
        batches = []
        for start in range(0, len(X), batch_size):
            batches.append(np.asarray(model.predict(X[start : start + batch_size]), dtype=float).reshape(-1))
        preds.append(np.concatenate(batches))
    mat = np.vstack(preds)
    return mat.mean(axis=0), mat.std(axis=0)


def transition_metrics(pred: pd.DataFrame, model: str, protocol: str, split: str, feature_set: str, features: list[str]) -> dict:
    row: dict[str, float | str | int] = {
        "feature_set": feature_set,
        "n_features": len(features),
        "model": model,
        "protocol": protocol,
        "split": split,
        "n": int(len(pred)),
        "n_groups": int(pred["group_id"].nunique()),
    }
    row.update(
        {
            "to_global_score_r2": safe_r2(pred["to_score_global"].to_numpy(), pred["pred_to_score_global"].to_numpy()),
            "to_global_score_mae": float(mean_absolute_error(pred["to_score_global"], pred["pred_to_score_global"])),
            "to_frame_r2": safe_r2(pred["to_IDR_frame"].to_numpy(), pred["pred_to_IDR_frame"].to_numpy()),
            "to_frame_mae": float(mean_absolute_error(pred["to_IDR_frame"], pred["pred_to_IDR_frame"])),
            "to_masonry_r2": safe_r2(pred["to_IDR_masonry"].to_numpy(), pred["pred_to_IDR_masonry"].to_numpy()),
            "to_masonry_mae": float(mean_absolute_error(pred["to_IDR_masonry"], pred["pred_to_IDR_masonry"])),
            "delta_log_frame_r2": safe_r2(pred["delta_log_IDR_frame"].to_numpy(), pred["pred_delta_log_IDR_frame"].to_numpy()),
            "delta_log_masonry_r2": safe_r2(
                pred["delta_log_IDR_masonry"].to_numpy(),
                pred["pred_delta_log_IDR_masonry"].to_numpy(),
            ),
            "global_damage_monotonicity_violation_rate": float(
                (pred["pred_to_damage_global"] < pred["from_damage_global"]).mean()
            ),
            "frame_monotonicity_violation_rate": float((pred["pred_to_IDR_frame"] < pred["from_IDR_frame"]).mean()),
            "masonry_monotonicity_violation_rate": float((pred["pred_to_IDR_masonry"] < pred["from_IDR_masonry"]).mean()),
        }
    )
    row.update(
        point_damage_metrics(
            pred["to_damage_global"].to_numpy(dtype=int),
            pred["pred_to_damage_global"].to_numpy(dtype=int),
            "to_global_damage",
        )
    )
    return row


def fit_predict_split(
    transition: pd.DataFrame,
    split: dict[str, object],
    model_name: str,
    feature_set: str,
    features: list[str],
    args: argparse.Namespace,
    TabPFNRegressor=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx = np.asarray(split["train_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)
    train = transition.iloc[train_idx].copy()
    test = transition.iloc[test_idx].copy()

    X_train = train[features].to_numpy(dtype=float)
    X_test = test[features].to_numpy(dtype=float)

    frame_models = fit_delta_models(
        model_name,
        X_train,
        train["target_log_delta_frame"].to_numpy(dtype=float),
        args,
        f"{feature_set}-{split['protocol']}-{split['split']}-frame",
        TabPFNRegressor=TabPFNRegressor,
    )
    masonry_models = fit_delta_models(
        model_name,
        X_train,
        train["target_log_delta_masonry"].to_numpy(dtype=float),
        args,
        f"{feature_set}-{split['protocol']}-{split['split']}-masonry",
        TabPFNRegressor=TabPFNRegressor,
    )

    frame_log_delta_mu, frame_log_delta_sd = predict_with_delta_models(frame_models, X_test)
    masonry_log_delta_mu, masonry_log_delta_sd = predict_with_delta_models(masonry_models, X_test)
    pred_delta_frame = np.exp(frame_log_delta_mu)
    pred_delta_masonry = np.exp(masonry_log_delta_mu)
    pred_to_frame = test["from_IDR_frame"].to_numpy(dtype=float) * np.exp(pred_delta_frame)
    pred_to_masonry = test["from_IDR_masonry"].to_numpy(dtype=float) * np.exp(pred_delta_masonry)
    pred_frame_damage, pred_masonry_damage, pred_global_damage = damage_labels_from_idr(pred_to_frame, pred_to_masonry)
    pred_score = global_score(pred_to_frame, pred_to_masonry)

    out = test.reset_index().rename(columns={"index": "transition_index"})
    out.insert(0, "feature_set", feature_set)
    out.insert(1, "n_features", len(features))
    out.insert(2, "model", model_name)
    out.insert(3, "protocol", str(split["protocol"]))
    out.insert(4, "split", str(split["split"]))
    out["pred_log_delta_frame_mu"] = frame_log_delta_mu
    out["pred_log_delta_frame_sd"] = frame_log_delta_sd
    out["pred_log_delta_masonry_mu"] = masonry_log_delta_mu
    out["pred_log_delta_masonry_sd"] = masonry_log_delta_sd
    out["pred_delta_log_IDR_frame"] = pred_delta_frame
    out["pred_delta_log_IDR_masonry"] = pred_delta_masonry
    out["pred_to_IDR_frame"] = pred_to_frame
    out["pred_to_IDR_masonry"] = pred_to_masonry
    out["pred_to_damage_frame"] = pred_frame_damage
    out["pred_to_damage_masonry"] = pred_masonry_damage
    out["pred_to_damage_global"] = pred_global_damage
    out["pred_to_score_global"] = pred_score

    rollout = continuous_two_step_prediction(
        test,
        model_name,
        str(split["protocol"]),
        str(split["split"]),
        feature_set,
        features,
        frame_models,
        masonry_models,
    )
    return out, rollout


def continuous_two_step_prediction(
    test: pd.DataFrame,
    model_name: str,
    protocol: str,
    split_name: str,
    feature_set: str,
    features: list[str],
    frame_models: list,
    masonry_models: list,
) -> pd.DataFrame:
    rows = []
    for group_id, group in test.groupby("group_id"):
        if set(group["transition_name"]) != {"low_to_mid", "mid_to_high"}:
            continue
        low_mid = group[group["transition_name"] == "low_to_mid"].iloc[0].copy()
        mid_high = group[group["transition_name"] == "mid_to_high"].iloc[0].copy()

        low_features = low_mid[features].to_numpy(dtype=float).reshape(1, -1)
        mid_frame_log_delta, _ = predict_with_delta_models(frame_models, low_features)
        mid_masonry_log_delta, _ = predict_with_delta_models(masonry_models, low_features)
        pred_mid_delta_frame = float(np.exp(mid_frame_log_delta[0]))
        pred_mid_delta_masonry = float(np.exp(mid_masonry_log_delta[0]))
        pred_mid_frame = float(low_mid["from_IDR_frame"] * np.exp(pred_mid_delta_frame))
        pred_mid_masonry = float(low_mid["from_IDR_masonry"] * np.exp(pred_mid_delta_masonry))
        pred_mid_frame_damage, pred_mid_masonry_damage, pred_mid_global_damage = damage_labels_from_idr(
            np.array([pred_mid_frame]),
            np.array([pred_mid_masonry]),
        )

        high_input = mid_high.copy()
        high_input["from_IDR_frame"] = pred_mid_frame
        high_input["from_IDR_masonry"] = pred_mid_masonry
        high_input["from_log_IDR_frame"] = np.log(max(pred_mid_frame, EPS))
        high_input["from_log_IDR_masonry"] = np.log(max(pred_mid_masonry, EPS))
        high_input["from_damage_frame"] = int(pred_mid_frame_damage[0])
        high_input["from_damage_masonry"] = int(pred_mid_masonry_damage[0])
        high_input["from_damage_global"] = int(pred_mid_global_damage[0])

        high_features = high_input[features].to_numpy(dtype=float).reshape(1, -1)
        high_frame_log_delta, _ = predict_with_delta_models(frame_models, high_features)
        high_masonry_log_delta, _ = predict_with_delta_models(masonry_models, high_features)
        pred_high_delta_frame = float(np.exp(high_frame_log_delta[0]))
        pred_high_delta_masonry = float(np.exp(high_masonry_log_delta[0]))
        pred_high_frame = float(pred_mid_frame * np.exp(pred_high_delta_frame))
        pred_high_masonry = float(pred_mid_masonry * np.exp(pred_high_delta_masonry))
        _, _, pred_high_global_damage = damage_labels_from_idr(
            np.array([pred_high_frame]),
            np.array([pred_high_masonry]),
        )

        true_sequence = [
            int(low_mid["from_damage_global"]),
            int(low_mid["to_damage_global"]),
            int(mid_high["to_damage_global"]),
        ]
        pred_sequence = [
            int(low_mid["from_damage_global"]),
            int(pred_mid_global_damage[0]),
            int(pred_high_global_damage[0]),
        ]
        rows.append(
            {
                "feature_set": feature_set,
                "n_features": len(features),
                "model": model_name,
                "protocol": protocol,
                "split": split_name,
                "group_id": int(group_id),
                "SPI": int(low_mid["SPI"]),
                "true_sequence_global": "/".join(map(str, true_sequence)),
                "pred_sequence_global": "/".join(map(str, pred_sequence)),
                "trajectory_exact_match": bool(true_sequence == pred_sequence),
                "trajectory_within_one_each_step": bool(
                    all(abs(a - b) <= 1 for a, b in zip(true_sequence, pred_sequence))
                ),
                "rollout_monotone": bool(pred_sequence[0] <= pred_sequence[1] <= pred_sequence[2]),
                "true_high_IDR_frame": float(mid_high["to_IDR_frame"]),
                "pred_high_IDR_frame": pred_high_frame,
                "true_high_IDR_masonry": float(mid_high["to_IDR_masonry"]),
                "pred_high_IDR_masonry": pred_high_masonry,
                "true_high_score_global": float(mid_high["to_score_global"]),
                "pred_high_score_global": float(global_score(np.array([pred_high_frame]), np.array([pred_high_masonry]))[0]),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["high_score_abs_error"] = np.abs(out["true_high_score_global"] - out["pred_high_score_global"])
    return out


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in metrics.select_dtypes(include=[np.number]).columns if c not in {"n", "n_groups"}]
    out = metrics.groupby(["feature_set", "model", "protocol"])[numeric_cols].agg(["mean", "std", "min"]).reset_index()
    out.columns = ["_".join([str(part) for part in c if str(part)]) if isinstance(c, tuple) else str(c) for c in out.columns]
    return out


def summarize_rollout(rollout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (feature_set, model, protocol), sub in rollout.groupby(["feature_set", "model", "protocol"]):
        rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "protocol": protocol,
                "n_groups": int(len(sub)),
                "trajectory_exact_match_rate": float(sub["trajectory_exact_match"].mean()),
                "trajectory_within_one_rate": float(sub["trajectory_within_one_each_step"].mean()),
                "rollout_monotone_rate": float(sub["rollout_monotone"].mean()),
                "high_score_r2": safe_r2(sub["true_high_score_global"].to_numpy(), sub["pred_high_score_global"].to_numpy()),
                "high_score_mae": float(mean_absolute_error(sub["true_high_score_global"], sub["pred_high_score_global"])),
                "high_frame_r2": safe_r2(sub["true_high_IDR_frame"].to_numpy(), sub["pred_high_IDR_frame"].to_numpy()),
                "high_masonry_r2": safe_r2(sub["true_high_IDR_masonry"].to_numpy(), sub["pred_high_IDR_masonry"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def make_effect_table(transition_summary: pd.DataFrame, rollout_summary: pd.DataFrame) -> pd.DataFrame:
    one = transition_summary[
        [
            "feature_set",
            "model",
            "protocol",
            "n_features_mean",
            "to_global_score_r2_mean",
            "to_global_damage_macro_f1_mean",
            "to_global_score_mae_mean",
        ]
    ].copy()
    combined = one.merge(
        rollout_summary[
            [
                "feature_set",
                "model",
                "protocol",
                "trajectory_exact_match_rate",
                "high_score_r2",
                "high_score_mae",
            ]
        ],
        on=["feature_set", "model", "protocol"],
        how="left",
    )
    full = combined[combined["feature_set"] == "full_16"].copy()
    full = full.rename(
        columns={
            "to_global_score_r2_mean": "full_one_step_r2",
            "to_global_damage_macro_f1_mean": "full_macro_f1",
            "trajectory_exact_match_rate": "full_path_exact",
            "high_score_r2": "full_high_score_r2",
        }
    )
    full = full[["model", "protocol", "full_one_step_r2", "full_macro_f1", "full_path_exact", "full_high_score_r2"]]
    out = combined.merge(full, on=["model", "protocol"], how="left")
    out["delta_one_step_r2_vs_full"] = out["to_global_score_r2_mean"] - out["full_one_step_r2"]
    out["delta_macro_f1_vs_full"] = out["to_global_damage_macro_f1_mean"] - out["full_macro_f1"]
    out["delta_path_exact_vs_full"] = out["trajectory_exact_match_rate"] - out["full_path_exact"]
    out["delta_high_score_r2_vs_full"] = out["high_score_r2"] - out["full_high_score_r2"]
    return out


def save_figures(effect: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "figure.dpi": 160,
            "savefig.dpi": 500,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    key = effect[effect["protocol"] == "leave_spi_out"].copy()
    figure_protocol = "leave_spi_out"
    if key.empty:
        key = effect.copy()
        figure_protocol = "all_protocols"
    order = [
        "full_16",
        "reduced_no_redundant",
        "no_current_state",
        "no_current_idr",
        "no_current_damage",
        "no_structural",
        "no_pga_transfer",
    ]
    key["feature_set"] = pd.Categorical(key["feature_set"], categories=order, ordered=True)
    key = key.sort_values(["model", "feature_set"])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    for model, sub in key.groupby("model"):
        axes[0].plot(
            sub["feature_set"].astype(str),
            sub["delta_high_score_r2_vs_full"],
            marker="o",
            lw=1.1,
            color=COLORS.get(model, "0.35"),
            label=MODEL_LABELS.get(model, model),
        )
        axes[1].plot(
            sub["feature_set"].astype(str),
            sub["delta_path_exact_vs_full"],
            marker="o",
            lw=1.1,
            color=COLORS.get(model, "0.35"),
            label=MODEL_LABELS.get(model, model),
        )
    for ax, title, ylabel in [
        (axes[0], "Terminal continuous-score change", "Delta high-score R2 vs full"),
        (axes[1], "Damage-path exact-match change", "Delta exact-match rate vs full"),
    ]:
        ax.axhline(0, color="0.25", lw=0.7)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.20, lw=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(frameon=False, loc="best")
    fig.savefig(fig_dir / f"feature_ablation_{figure_protocol}_delta.png", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(fig_dir / f"feature_ablation_{figure_protocol}_delta.pdf", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)

    tab = key.pivot_table(
        index="feature_set",
        columns="model",
        values="delta_high_score_r2_vs_full",
        aggfunc="mean",
        observed=False,
    ).reindex(order)
    fig, ax = plt.subplots(figsize=(4.8, 3.7), constrained_layout=True)
    im = ax.imshow(tab.to_numpy(dtype=float), cmap="RdBu_r", vmin=-0.35, vmax=0.35, aspect="auto")
    ax.set_yticks(np.arange(len(tab.index)), tab.index)
    ax.set_xticks(np.arange(len(tab.columns)), [MODEL_LABELS.get(c, c) for c in tab.columns], rotation=25, ha="right")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            val = tab.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=6)
    ax.set_title("Leave-SPI feature-block effect on terminal R2", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Delta R2 vs full")
    fig.savefig(fig_dir / f"feature_ablation_{figure_protocol}_heatmap.png", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(fig_dir / f"feature_ablation_{figure_protocol}_heatmap.pdf", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def write_notes(out_dir: Path, feature_sets: dict[str, list[str]]) -> None:
    rows = ["# Feature Ablation Notes", ""]
    rows.append("This experiment tests whether the transition features are necessary or redundant.")
    rows.append("")
    rows.append("## Feature Sets")
    for name, features in feature_sets.items():
        rows.append(f"- `{name}` ({len(features)} features): {', '.join(features)}")
    rows.append("")
    rows.append("## Interpretation")
    rows.append("- If `reduced_no_redundant` is close to `full_16`, the paper can use it as a compact main feature set.")
    rows.append("- If `no_current_state` drops strongly, it supports the transition-state formulation rather than static point prediction.")
    rows.append("- If `no_current_idr` drops strongly, continuous current response is essential.")
    rows.append("- If `no_current_damage` drops mainly in path exact-match, damage-state information helps threshold-level pathway recovery.")
    rows.append("- If `no_pga_transfer` drops strongly, the model needs explicit intensity-transition information.")
    rows.append("- If `no_structural` remains strong, the previous damage state dominates short-step prediction; this should be discussed honestly.")
    (out_dir / "FEATURE_ABLATION_NOTES.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def dry_run(args: argparse.Namespace, out_dir: Path) -> None:
    _, transition = ensure_data_tables()
    models = parse_csv_list(args.models)
    feature_sets = selected_feature_sets(args.feature_sets)
    info = {
        "n_transitions": int(len(transition)),
        "n_groups": int(transition["group_id"].nunique()),
        "models": models,
        "protocols": parse_csv_list(args.protocols),
        "feature_sets": feature_sets,
        "n_splits": len(make_group_splits(transition, parse_csv_list(args.protocols), args.n_splits, args.random_state)),
        "dependency_status": dependency_status(models),
        "gpu_status": gpu_status(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dry_run_summary.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        dry_run(args, out_dir)
        return

    _, transition = ensure_data_tables()
    models = parse_csv_list(args.models)
    feature_sets = selected_feature_sets(args.feature_sets)
    splits = make_group_splits(transition, parse_csv_list(args.protocols), args.n_splits, args.random_state)

    tabpfn_regressor = None
    if "tabpfn" in [m.lower() for m in models]:
        _, tabpfn_regressor = import_tabpfn()

    pred_frames = []
    rollout_frames = []
    metric_rows = []
    total = len(feature_sets) * len(splits) * len(models)
    step = 0
    for feature_set, features in feature_sets.items():
        for split in splits:
            for model in models:
                step += 1
                print(f"[{step}/{total}] {feature_set} | {split['protocol']} {split['split']} | {model}")
                pred, rollout = fit_predict_split(
                    transition,
                    split,
                    model,
                    feature_set,
                    features,
                    args,
                    TabPFNRegressor=tabpfn_regressor,
                )
                pred_frames.append(pred)
                rollout_frames.append(rollout)
                metric_rows.append(
                    transition_metrics(
                        pred,
                        model,
                        str(split["protocol"]),
                        str(split["split"]),
                        feature_set,
                        features,
                    )
                )

    predictions = pd.concat(pred_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    transition_summary = summarize_metrics(metrics)
    rollout = pd.concat(rollout_frames, ignore_index=True) if rollout_frames else pd.DataFrame()
    rollout_summary = summarize_rollout(rollout)
    effect = make_effect_table(transition_summary, rollout_summary)

    predictions.to_csv(out_dir / "feature_ablation_transition_predictions.csv", index=False)
    metrics.to_csv(out_dir / "feature_ablation_metrics_by_split.csv", index=False)
    transition_summary.to_csv(out_dir / "feature_ablation_transition_summary.csv", index=False)
    rollout.to_csv(out_dir / "feature_ablation_rollout_predictions.csv", index=False)
    rollout_summary.to_csv(out_dir / "feature_ablation_rollout_summary.csv", index=False)
    effect.to_csv(out_dir / "feature_ablation_effect_vs_full.csv", index=False)

    config = vars(args).copy()
    config["damage_names"] = DAMAGE_NAMES
    config["feature_sets"] = feature_sets
    config["target_transform"] = "log(max(delta_log_IDR, EPS)); IDR is updated as theta_i * exp(exp(predicted_target))"
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    write_notes(out_dir, feature_sets)
    save_figures(effect, out_dir)
    print(f"Feature ablation outputs saved to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature-block ablation for the damage-transition emulator.")
    parser.add_argument("--models", default="tabpfn,rf,lightgbm,xgboost")
    parser.add_argument("--protocols", default="groupkfold,leave_spi")
    parser.add_argument("--feature-sets", default="full_16,reduced_no_redundant,no_current_state,no_current_idr,no_current_damage,no_structural,no_pga_transfer")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--members", type=int, default=1)
    parser.add_argument("--tabpfn-estimators", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--rf-min-leaf", type=int, default=1)
    parser.add_argument("--gbdt-trees", type=int, default=450)
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.03)
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
