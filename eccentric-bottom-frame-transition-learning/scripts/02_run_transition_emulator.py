from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from transition_common import (
    DAMAGE_NAMES,
    EPS,
    ROOT,
    TRANSITION_FEATURES,
    binary_predict_proba,
    binary_probability_metrics,
    damage_labels_from_idr,
    dependency_status,
    global_score,
    gpu_status,
    import_tabpfn,
    knn_distance_score,
    make_regressor,
    make_group_splits,
    parse_csv_list,
    point_damage_metrics,
    rmse,
    safe_r2,
    stable_seed,
    zscore,
)

DATA_DIR = ROOT / "data"
DEFAULT_OUT = ROOT / "results" / "emulator"


def ensure_data_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    transition_path = DATA_DIR / "ml_transition_samples_226.csv"
    trajectory_path = DATA_DIR / "state_sequences_113.csv"
    if not transition_path.exists() or not trajectory_path.exists():
        raise FileNotFoundError(
            "Transition data not found. Run 01_build_transition_dataset.py first."
        )
    return pd.read_csv(trajectory_path), pd.read_csv(transition_path)


def predict_exceedance_probabilities(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
    TabPFNClassifier=None,
) -> pd.DataFrame:
    X_train = train[TRANSITION_FEATURES].to_numpy(dtype=float)
    X_test = test[TRANSITION_FEATURES].to_numpy(dtype=float)
    out = pd.DataFrame(index=test.index)
    for level in [1, 2, 3, 4]:
        y_train = (train["to_damage_global"].to_numpy(dtype=int) >= level).astype(int)
        seed = stable_seed("exceedance", model_name, test["protocol"].iloc[0], test["split"].iloc[0], level)
        prob = binary_predict_proba(
            model_name,
            X_train,
            y_train,
            X_test,
            args,
            seed,
            TabPFNClassifier=TabPFNClassifier,
        )
        out[f"prob_to_global_ge_{level}"] = prob
    probs = out[[f"prob_to_global_ge_{level}" for level in [1, 2, 3, 4]]].to_numpy(dtype=float)
    for col in range(2, -1, -1):
        probs[:, col] = np.maximum(probs[:, col], probs[:, col + 1])
    probs = np.clip(probs, 0.0, 1.0)
    for idx, level in enumerate([1, 2, 3, 4]):
        out[f"prob_to_global_ge_{level}"] = probs[:, idx]
    return out


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
        setattr(model, "_codex_predict_batch_size", int(getattr(args, "predict_batch_size", 8) or 8))
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


def transition_metrics(pred: pd.DataFrame, model: str, protocol: str, split: str) -> dict[str, float | str | int]:
    row: dict[str, float | str | int] = {
        "model": model,
        "protocol": protocol,
        "split": split,
        "n": int(len(pred)),
        "n_groups": int(pred["group_id"].nunique()),
    }
    row.update(
        {
            "delta_log_frame_r2": safe_r2(pred["delta_log_IDR_frame"].to_numpy(), pred["pred_delta_log_IDR_frame"].to_numpy()),
            "delta_log_frame_mae": float(mean_absolute_error(pred["delta_log_IDR_frame"], pred["pred_delta_log_IDR_frame"])),
            "delta_log_masonry_r2": safe_r2(
                pred["delta_log_IDR_masonry"].to_numpy(),
                pred["pred_delta_log_IDR_masonry"].to_numpy(),
            ),
            "delta_log_masonry_mae": float(
                mean_absolute_error(pred["delta_log_IDR_masonry"], pred["pred_delta_log_IDR_masonry"])
            ),
            "to_frame_r2": safe_r2(pred["to_IDR_frame"].to_numpy(), pred["pred_to_IDR_frame"].to_numpy()),
            "to_frame_rmse": rmse(pred["to_IDR_frame"].to_numpy(), pred["pred_to_IDR_frame"].to_numpy()),
            "to_frame_mae": float(mean_absolute_error(pred["to_IDR_frame"], pred["pred_to_IDR_frame"])),
            "to_masonry_r2": safe_r2(pred["to_IDR_masonry"].to_numpy(), pred["pred_to_IDR_masonry"].to_numpy()),
            "to_masonry_rmse": rmse(pred["to_IDR_masonry"].to_numpy(), pred["pred_to_IDR_masonry"].to_numpy()),
            "to_masonry_mae": float(mean_absolute_error(pred["to_IDR_masonry"], pred["pred_to_IDR_masonry"])),
            "to_global_score_r2": safe_r2(pred["to_score_global"].to_numpy(), pred["pred_to_score_global"].to_numpy()),
            "to_global_score_mae": float(mean_absolute_error(pred["to_score_global"], pred["pred_to_score_global"])),
            "frame_monotonicity_violation_rate": float((pred["pred_to_IDR_frame"] < pred["from_IDR_frame"]).mean()),
            "masonry_monotonicity_violation_rate": float((pred["pred_to_IDR_masonry"] < pred["from_IDR_masonry"]).mean()),
            "global_damage_monotonicity_violation_rate": float(
                (pred["pred_to_damage_global"] < pred["from_damage_global"]).mean()
            ),
            "ood_error_spearman": float(
                pred["applicability_risk"].rank().corr(pred["to_global_score_abs_error"].rank())
            ),
        }
    )
    row.update(
        point_damage_metrics(
            pred["to_damage_global"].to_numpy(dtype=int),
            pred["pred_to_damage_global"].to_numpy(dtype=int),
            "to_global_damage",
        )
    )
    prob_cols = [f"prob_to_global_ge_{level}" for level in [1, 2, 3, 4]]
    if all(c in pred.columns for c in prob_cols):
        briers = []
        logs = []
        for level in [1, 2, 3, 4]:
            y = (pred["to_damage_global"].to_numpy(dtype=int) >= level).astype(int)
            metrics = binary_probability_metrics(y, pred[f"prob_to_global_ge_{level}"].to_numpy(), f"ge_{level}")
            row.update(metrics)
            briers.append(metrics[f"ge_{level}_brier"])
            logs.append(metrics[f"ge_{level}_log_loss"])
        row["mean_exceedance_brier"] = float(np.mean(briers))
        row["mean_exceedance_log_loss"] = float(np.mean(logs))
    return row


def fit_predict_split(
    transition: pd.DataFrame,
    split: dict[str, object],
    model_name: str,
    args: argparse.Namespace,
    TabPFNRegressor=None,
    TabPFNClassifier=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx = np.asarray(split["train_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)
    train = transition.iloc[train_idx].copy()
    test = transition.iloc[test_idx].copy()
    X_train = train[TRANSITION_FEATURES].to_numpy(dtype=float)
    X_test = test[TRANSITION_FEATURES].to_numpy(dtype=float)

    frame_models = fit_delta_models(
        model_name,
        X_train,
        train["target_log_delta_frame"].to_numpy(dtype=float),
        args,
        f"{split['protocol']}-{split['split']}-frame",
        TabPFNRegressor=TabPFNRegressor,
    )
    masonry_models = fit_delta_models(
        model_name,
        X_train,
        train["target_log_delta_masonry"].to_numpy(dtype=float),
        args,
        f"{split['protocol']}-{split['split']}-masonry",
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
    out.insert(0, "model", model_name)
    out.insert(1, "protocol", str(split["protocol"]))
    out.insert(2, "split", str(split["split"]))
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
    out["to_global_score_abs_error"] = np.abs(out["to_score_global"].to_numpy(dtype=float) - pred_score)

    kdist = knn_distance_score(X_train, X_test, n_neighbors=args.knn)
    uncertainty = 0.5 * (frame_log_delta_sd + masonry_log_delta_sd)
    out["knn_distance"] = kdist
    out["transition_uncertainty"] = uncertainty
    out["threshold_proximity"] = -np.log(np.maximum(np.abs(out["to_score_global"].to_numpy(dtype=float) - np.round(out["to_score_global"].to_numpy(dtype=float))), 1e-4))
    out["applicability_risk"] = zscore(kdist) + zscore(uncertainty) + 0.5 * zscore(out["threshold_proximity"].to_numpy(dtype=float))

    if args.with_probabilities:
        probs = predict_exceedance_probabilities(
            model_name,
            train.assign(protocol=str(split["protocol"]), split=str(split["split"])),
            test.assign(protocol=str(split["protocol"]), split=str(split["split"])),
            args,
            TabPFNClassifier=TabPFNClassifier,
        )
        out = pd.concat([out.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)
    rollout = autonomous_rollout_for_split(
        test,
        model_name,
        str(split["protocol"]),
        str(split["split"]),
        frame_models,
        masonry_models,
    )
    return out, rollout


def autonomous_rollout_for_split(
    test: pd.DataFrame,
    model_name: str,
    protocol: str,
    split_name: str,
    frame_models: list,
    masonry_models: list,
) -> pd.DataFrame:
    rows = []
    for group_id, group in test.groupby("group_id"):
        if set(group["transition_name"]) != {"low_to_mid", "mid_to_high"}:
            continue
        low_mid = group[group["transition_name"] == "low_to_mid"].iloc[0].copy()
        mid_high = group[group["transition_name"] == "mid_to_high"].iloc[0].copy()
        low_features = low_mid[TRANSITION_FEATURES].to_numpy(dtype=float).reshape(1, -1)
        mid_frame_log_delta, mid_frame_log_sd = predict_with_delta_models(frame_models, low_features)
        mid_masonry_log_delta, mid_masonry_log_sd = predict_with_delta_models(masonry_models, low_features)
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
        high_features = high_input[TRANSITION_FEATURES].to_numpy(dtype=float).reshape(1, -1)
        high_frame_log_delta, high_frame_log_sd = predict_with_delta_models(frame_models, high_features)
        high_masonry_log_delta, high_masonry_log_sd = predict_with_delta_models(masonry_models, high_features)
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
                "true_mid_IDR_frame": float(low_mid["to_IDR_frame"]),
                "pred_mid_IDR_frame": pred_mid_frame,
                "true_mid_IDR_masonry": float(low_mid["to_IDR_masonry"]),
                "pred_mid_IDR_masonry": pred_mid_masonry,
                "true_high_IDR_frame": float(mid_high["to_IDR_frame"]),
                "pred_high_IDR_frame": pred_high_frame,
                "true_high_IDR_masonry": float(mid_high["to_IDR_masonry"]),
                "pred_high_IDR_masonry": pred_high_masonry,
                "true_high_score_global": float(mid_high["to_score_global"]),
                "pred_high_score_global": float(global_score(np.array([pred_high_frame]), np.array([pred_high_masonry]))[0]),
                "mid_transition_uncertainty": float(0.5 * (mid_frame_log_sd[0] + mid_masonry_log_sd[0])),
                "high_transition_uncertainty": float(0.5 * (high_frame_log_sd[0] + high_masonry_log_sd[0])),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["high_score_abs_error"] = np.abs(out["true_high_score_global"] - out["pred_high_score_global"])
    return out


def summarize_rollout(rollout: pd.DataFrame) -> pd.DataFrame:
    if rollout.empty:
        return pd.DataFrame()
    return (
        rollout.groupby(["model", "protocol"])
        .agg(
            n_groups=("group_id", "count"),
            trajectory_exact_match_rate=("trajectory_exact_match", "mean"),
            trajectory_within_one_rate=("trajectory_within_one_each_step", "mean"),
            rollout_monotone_rate=("rollout_monotone", "mean"),
            high_score_r2=("true_high_score_global", lambda s: np.nan),
            high_score_mae=("high_score_abs_error", "mean"),
        )
        .reset_index()
    )


def add_rollout_r2(summary: pd.DataFrame, rollout: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    rows = []
    for _, row in summary.iterrows():
        sub = rollout[(rollout["model"] == row["model"]) & (rollout["protocol"] == row["protocol"])]
        row = row.to_dict()
        row["high_score_r2"] = safe_r2(sub["true_high_score_global"].to_numpy(), sub["pred_high_score_global"].to_numpy())
        row["high_frame_r2"] = safe_r2(sub["true_high_IDR_frame"].to_numpy(), sub["pred_high_IDR_frame"].to_numpy())
        row["high_masonry_r2"] = safe_r2(sub["true_high_IDR_masonry"].to_numpy(), sub["pred_high_IDR_masonry"].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in metrics.select_dtypes(include=[np.number]).columns if c not in {"n", "n_groups"}]
    out = metrics.groupby(["model", "protocol"])[numeric_cols].agg(["mean", "std", "min"]).reset_index()
    out.columns = ["_".join([str(part) for part in c if str(part)]) if isinstance(c, tuple) else str(c) for c in out.columns]
    return out


def dry_run(args: argparse.Namespace, out_dir: Path) -> None:
    _, transition = ensure_data_tables()
    models = parse_csv_list(args.models)
    splits = make_group_splits(transition, parse_csv_list(args.protocols), args.n_splits, args.random_state)
    info = {
        "n_transitions": int(len(transition)),
        "n_groups": int(transition["group_id"].nunique()),
        "transition_features": TRANSITION_FEATURES,
        "models": models,
        "protocols": parse_csv_list(args.protocols),
        "n_splits": len(splits),
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
    splits = make_group_splits(transition, parse_csv_list(args.protocols), args.n_splits, args.random_state)
    tabpfn_regressor = None
    tabpfn_classifier = None
    if "tabpfn" in [m.lower() for m in models]:
        tabpfn_classifier, tabpfn_regressor = import_tabpfn()

    pred_frames = []
    rollout_frames = []
    metric_rows = []
    for split_no, split in enumerate(splits, start=1):
        for model in models:
            print(f"[{split_no}/{len(splits)}] {split['protocol']} {split['split']} - {model}")
            pred, rollout = fit_predict_split(
                transition,
                split,
                model,
                args,
                TabPFNRegressor=tabpfn_regressor,
                TabPFNClassifier=tabpfn_classifier,
            )
            pred_frames.append(pred)
            rollout_frames.append(rollout)
            metric_rows.append(transition_metrics(pred, model, str(split["protocol"]), str(split["split"])))

    predictions = pd.concat(pred_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = summarize_metrics(metrics)
    rollout = pd.concat(rollout_frames, ignore_index=True) if rollout_frames else pd.DataFrame()
    rollout_summary = add_rollout_r2(summarize_rollout(rollout), rollout)

    predictions.to_csv(out_dir / "transition_predictions.csv", index=False)
    metrics.to_csv(out_dir / "transition_metrics_by_split.csv", index=False)
    summary.to_csv(out_dir / "transition_metric_summary.csv", index=False)
    rollout.to_csv(out_dir / "rollout_predictions.csv", index=False)
    rollout_summary.to_csv(out_dir / "rollout_summary.csv", index=False)
    config = vars(args).copy()
    config["transition_features"] = TRANSITION_FEATURES
    config["damage_names"] = DAMAGE_NAMES
    config["target_transform"] = "log(max(delta_log_IDR, EPS)); IDR is updated as theta_i * exp(exp(predicted_target))"
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Transition emulator outputs saved to: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monotone damage-transition emulator.")
    parser.add_argument("--models", default="tabpfn,rf,lightgbm,xgboost")
    parser.add_argument("--protocols", default="groupkfold,leave_spi")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--tabpfn-estimators", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--rf-min-leaf", type=int, default=1)
    parser.add_argument("--gbdt-trees", type=int, default=450)
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.03)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--predict-batch-size", type=int, default=8, help="Batch size for model.predict on small GPUs.")
    parser.add_argument("--with-probabilities", action="store_true")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
