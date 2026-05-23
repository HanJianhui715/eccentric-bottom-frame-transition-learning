from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from damage_assessment.data import (  # noqa: E402
    DAMAGE_NAMES,
    FEATURE_COLUMNS,
    FRAME_BOUNDS,
    MASONRY_BOUNDS,
    assign_damage,
    damage_labels_from_idr,
    load_dataset,
)

EPS = 1e-12
STRUCTURAL_COLUMNS = FEATURE_COLUMNS[:6]
TRANSITION_FEATURES = [
    "SPI",
    "ISR",
    "Ecc_X_1",
    "Ecc_Y_1",
    "Ecc_X_2",
    "Ecc_Y_2",
    "from_pga",
    "to_pga",
    "pga_ratio",
    "log_pga_ratio",
    "from_log_IDR_frame",
    "from_log_IDR_masonry",
    "from_damage_frame",
    "from_damage_masonry",
    "from_damage_global",
    "transition_stage",
]


def stable_seed(*parts: object, base: int = 0) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return (int(digest[:12], 16) + int(base)) % (2**31 - 1)


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def safe_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return float("nan")


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = float(np.nanstd(values))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(values)
    return (values - float(np.nanmean(values))) / sd


def continuous_damage_score(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    score = np.zeros_like(values, dtype=float)
    below = values < bounds[0]
    score[below] = np.clip(values[below] / bounds[0], 0.0, 1.0)
    for level in range(1, len(bounds)):
        low = bounds[level - 1]
        high = bounds[level]
        mask = (values >= low) & (values < high)
        score[mask] = level + np.clip((values[mask] - low) / (high - low), 0.0, 1.0)
    score[values >= bounds[-1]] = 4.0
    return score


def global_score(frame_idr: np.ndarray, masonry_idr: np.ndarray) -> np.ndarray:
    return np.maximum(
        continuous_damage_score(frame_idr, FRAME_BOUNDS),
        continuous_damage_score(masonry_idr, MASONRY_BOUNDS),
    )


def nearest_threshold_margin(frame_idr: np.ndarray, masonry_idr: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame_idr, dtype=float)[:, None]
    masonry = np.asarray(masonry_idr, dtype=float)[:, None]
    frame_margin = np.min(np.abs(frame - FRAME_BOUNDS[None, :]) / np.maximum(FRAME_BOUNDS[None, :], EPS), axis=1)
    masonry_margin = np.min(
        np.abs(masonry - MASONRY_BOUNDS[None, :]) / np.maximum(MASONRY_BOUNDS[None, :], EPS),
        axis=1,
    )
    return np.minimum(frame_margin, masonry_margin)


def import_tabpfn():
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
    except Exception as exc:
        raise RuntimeError("TabPFN is not available. Install it with: python -m pip install tabpfn") from exc
    return TabPFNClassifier, TabPFNRegressor


def optional_import_lightgbm():
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except Exception as exc:
        raise RuntimeError("LightGBM is not installed. Run: python -m pip install lightgbm") from exc
    return LGBMClassifier, LGBMRegressor


def optional_import_xgboost():
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception as exc:
        raise RuntimeError("XGBoost is not installed. Run: python -m pip install xgboost") from exc
    return XGBClassifier, XGBRegressor


def make_tabpfn_estimator(cls, n_estimators: int, seed: int, device: str):
    attempts = [
        {
            "n_estimators": n_estimators,
            "random_state": seed,
            "device": device,
            "categorical_features_indices": [0, 14, 15] if len(TRANSITION_FEATURES) > 15 else [0],
            "fit_mode": "fit_preprocessors",
        },
        {"n_estimators": n_estimators, "random_state": seed, "device": device, "categorical_features_indices": [0]},
        {"n_estimators": n_estimators, "random_state": seed, "device": device},
        {"n_estimators": n_estimators, "device": device},
        {"device": device},
        {},
    ]
    last_error = None
    for params in attempts:
        try:
            return cls(**params)
        except TypeError as exc:
            last_error = exc
    raise last_error


def make_regressor(model_name: str, seed: int, args, TabPFNRegressor=None):
    name = model_name.lower()
    if name == "tabpfn":
        if TabPFNRegressor is None:
            _, TabPFNRegressor = import_tabpfn()
        return make_tabpfn_estimator(TabPFNRegressor, args.tabpfn_estimators, seed, args.device)
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=args.rf_trees,
            min_samples_leaf=args.rf_min_leaf,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "lightgbm":
        _, LGBMRegressor = optional_import_lightgbm()
        return LGBMRegressor(
            n_estimators=args.gbdt_trees,
            learning_rate=args.gbdt_learning_rate,
            num_leaves=15,
            min_child_samples=3,
            subsample=0.85,
            colsample_bytree=0.90,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "xgboost":
        _, XGBRegressor = optional_import_xgboost()
        return XGBRegressor(
            n_estimators=args.gbdt_trees,
            max_depth=3,
            learning_rate=args.gbdt_learning_rate,
            subsample=0.85,
            colsample_bytree=0.90,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown regressor: {model_name}")


def make_classifier(model_name: str, seed: int, args, TabPFNClassifier=None):
    name = model_name.lower()
    if name == "tabpfn":
        if TabPFNClassifier is None:
            TabPFNClassifier, _ = import_tabpfn()
        return make_tabpfn_estimator(TabPFNClassifier, args.tabpfn_estimators, seed, args.device)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=args.rf_trees,
            min_samples_leaf=args.rf_min_leaf,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "lightgbm":
        LGBMClassifier, _ = optional_import_lightgbm()
        return LGBMClassifier(
            n_estimators=args.gbdt_trees,
            learning_rate=args.gbdt_learning_rate,
            num_leaves=15,
            min_child_samples=3,
            subsample=0.85,
            colsample_bytree=0.90,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "xgboost":
        XGBClassifier, _ = optional_import_xgboost()
        return XGBClassifier(
            n_estimators=args.gbdt_trees,
            max_depth=3,
            learning_rate=args.gbdt_learning_rate,
            subsample=0.85,
            colsample_bytree=0.90,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier: {model_name}")


def fit_predict_members(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    args,
    seed_tag: str,
    TabPFNRegressor=None,
) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    for member in range(args.members):
        seed = stable_seed(model_name, seed_tag, member, base=args.random_state)
        model = make_regressor(model_name, seed, args, TabPFNRegressor=TabPFNRegressor)
        model.fit(X_train, y_train)
        preds.append(np.asarray(model.predict(X_test), dtype=float).reshape(-1))
    mat = np.vstack(preds)
    return mat.mean(axis=0), mat.std(axis=0)


def binary_predict_proba(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    args,
    seed: int,
    TabPFNClassifier=None,
) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=int)
    unique = np.unique(y_train)
    if len(unique) == 1:
        return np.full(X_test.shape[0], float(unique[0]), dtype=float)
    model = make_classifier(model_name, seed, args, TabPFNClassifier=TabPFNClassifier)
    model.fit(X_train, y_train)
    batch_size = int(getattr(args, "predict_batch_size", 16) or 16)
    prob_batches = []
    for start in range(0, len(X_test), batch_size):
        prob_batches.append(np.asarray(model.predict_proba(X_test[start : start + batch_size]), dtype=float))
    probs = np.vstack(prob_batches)
    classes = np.asarray(getattr(model, "classes_", np.array([0, 1])), dtype=int)
    pos = np.where(classes == 1)[0]
    if len(pos) == 0:
        return np.zeros(X_test.shape[0], dtype=float)
    return probs[:, int(pos[0])]


def make_group_splits(df: pd.DataFrame, protocols: Iterable[str], n_splits: int, random_state: int) -> list[dict[str, object]]:
    protocols = set(protocols)
    X = df[TRANSITION_FEATURES].to_numpy(dtype=float)
    y = df["to_damage_global"].to_numpy(dtype=int)
    groups = df["group_id"].to_numpy(dtype=int)
    specs: list[dict[str, object]] = []

    if "groupkfold" in protocols:
        try:
            splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        except TypeError:
            splitter = GroupKFold(n_splits=n_splits)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            specs.append(
                {
                    "protocol": "groupkfold",
                    "split": f"fold_{fold}",
                    "train_idx": np.asarray(train_idx, dtype=int),
                    "test_idx": np.asarray(test_idx, dtype=int),
                }
            )

    if "leave_spi" in protocols:
        for spi in sorted(df["SPI"].unique()):
            mask = df["SPI"].to_numpy(dtype=float) == float(spi)
            specs.append(
                {
                    "protocol": "leave_spi_out",
                    "split": f"SPI_{int(spi)}",
                    "train_idx": np.where(~mask)[0].astype(int),
                    "test_idx": np.where(mask)[0].astype(int),
                }
            )

    if "leave_stage" in protocols:
        for stage_name in sorted(df["transition_name"].unique()):
            mask = df["transition_name"].to_numpy() == stage_name
            specs.append(
                {
                    "protocol": "leave_stage_out",
                    "split": str(stage_name),
                    "train_idx": np.where(~mask)[0].astype(int),
                    "test_idx": np.where(mask)[0].astype(int),
                }
            )
    return specs


def dependency_status(models: Iterable[str]) -> dict[str, str]:
    status: dict[str, str] = {}
    for model in sorted(set(m.lower() for m in models)):
        try:
            if model == "tabpfn":
                import_tabpfn()
            elif model == "lightgbm":
                optional_import_lightgbm()
            elif model == "xgboost":
                optional_import_xgboost()
            elif model == "rf":
                pass
            else:
                raise ValueError(f"unknown model: {model}")
            status[model] = "ok"
        except Exception as exc:
            status[model] = f"missing_or_error: {exc}"
    return status


def gpu_status():
    try:
        import torch

        return {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        }
    except Exception as exc:
        return f"torch_check_failed: {exc}"


def knn_distance_score(X_train: np.ndarray, X_test: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    n_neighbors = min(n_neighbors, max(1, len(X_train)))
    train_mean = np.nanmean(X_train, axis=0)
    train_sd = np.nanstd(X_train, axis=0)
    train_sd[train_sd <= 1e-12] = 1.0
    Xtr = (X_train - train_mean) / train_sd
    Xte = (X_test - train_mean) / train_sd
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(Xtr)
    dist, _ = nn.kneighbors(Xte)
    return dist.mean(axis=1)


def point_damage_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}_qwk": safe_qwk(y_true, y_pred),
        f"{prefix}_ordinal_mae": float(mean_absolute_error(y_true, y_pred)),
    }


def binary_probability_metrics(y_true: np.ndarray, prob: np.ndarray, prefix: str) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-8, 1.0 - 1e-8)
    return {
        f"{prefix}_brier": float(brier_score_loss(y_true, prob)),
        f"{prefix}_log_loss": float(log_loss(y_true, np.column_stack([1.0 - prob, prob]), labels=[0, 1])),
    }
