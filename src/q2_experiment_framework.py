#!/usr/bin/env python3
"""Shared data, split, scoring, bootstrap, and output helpers for Q2 experiments.

The framework deliberately does not implement the Q2 model families. Callers
provide a fit/predict function, so B1, p-only, and p+Q experiments can share
the same evaluation contract and persisted row splits.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


MODEL_FEATURES = {
    "M0": ("N", "D"),
    "M1": ("N", "D", "p"),
    "M2": ("N", "D", "p", "Q"),
}
Q_STATUS = {"absent", "candidate", "final"}
METRIC_NAMES = ("rmse", "mae", "r2")
FitPredict = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], Sequence[float]]


@dataclass(frozen=True)
class Q2Dataset:
    """A normalized table plus provenance needed to validate model inputs."""

    frame: pd.DataFrame
    p_columns: tuple[str, ...]
    sources: tuple[dict[str, Any], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")


def _p_name(name: str) -> str:
    return f"p::{name}"


def load_q2_data(source_specs: Sequence[Mapping[str, Any]]) -> Q2Dataset:
    """Load one or more CSV sources into the canonical Q2 data structure.

    Each source spec has ``path``, ``source``, ``columns`` (canonical-to-raw
    mappings for N, D, loss, and optionally Q/group/id), and an optional
    ``p_columns`` mapping from semantic domain name to raw column. A source
    carrying Q must declare ``q_status`` as ``candidate`` or ``final``;
    final Q additionally requires a non-empty ``q_version``.
    """
    if not source_specs:
        raise ValueError("source_specs must contain at least one input source")

    parts: list[pd.DataFrame] = []
    source_metadata: list[dict[str, Any]] = []
    p_names: set[str] = set()

    for spec in source_specs:
        path = Path(spec["path"]).expanduser().resolve()
        source = str(spec["source"])
        column_map = dict(spec["columns"])
        p_map = dict(spec.get("p_columns", {}))
        q_status = str(spec.get("q_status", "absent"))
        q_version = spec.get("q_version")
        if q_status not in Q_STATUS:
            raise ValueError(f"{source}: q_status must be one of {sorted(Q_STATUS)}")
        if not {"N", "D", "loss"}.issubset(column_map):
            raise ValueError(f"{source}: columns must map N, D, and loss")
        if ("Q" in column_map) != (q_status != "absent"):
            raise ValueError(f"{source}: map Q exactly when q_status is candidate or final")
        if q_status == "final" and not str(q_version or "").strip():
            raise ValueError(f"{source}: final Q requires a frozen q_version identifier")
        if q_status != "final" and q_version:
            raise ValueError(f"{source}: q_version may only identify frozen final Q")
        if not path.is_file():
            raise FileNotFoundError(path)

        raw = pd.read_csv(path)
        required_raw = set(column_map.values()) | set(p_map.values())
        missing = sorted(required_raw - set(raw.columns))
        if missing:
            raise ValueError(f"{source}: missing configured columns: {missing}")

        frame = pd.DataFrame(index=raw.index)
        for canonical in ("N", "D", "loss", "Q", "group", "id"):
            raw_name = column_map.get(canonical)
            if raw_name:
                frame[canonical] = raw[raw_name]
        for domain, raw_name in p_map.items():
            cname = _p_name(str(domain))
            frame[cname] = raw[raw_name]
            p_names.add(cname)
        frame["source"] = source
        if "group" not in frame:
            frame["group"] = pd.NA
        raw_ids = frame["id"].astype(str) if "id" in frame else pd.Series(raw.index.astype(str), index=raw.index)
        frame["row_id"] = [f"{source}:{path.name}:{value}" for value in raw_ids]
        frame["q_status"] = q_status
        frame["q_version"] = str(q_version) if q_version else ""
        frame = frame.drop(columns=["id"], errors="ignore")

        for col in ("N", "D", "loss", "Q", *(_p_name(str(name)) for name in p_map)):
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if not np.isfinite(frame[["N", "D", "loss"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{source}: N, D, and loss must be finite numeric values")
        if (frame[["N", "D"]] <= 0).any().any():
            raise ValueError(f"{source}: N and D must be strictly positive")
        if "Q" in frame and not np.isfinite(frame["Q"].to_numpy(dtype=float)).all():
            raise ValueError(f"{source}: mapped Q contains missing or non-finite values")
        source_p_cols = [_p_name(str(name)) for name in p_map]
        if source_p_cols:
            values = frame[source_p_cols].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{source}: p components must be finite and non-negative")
            if (values.sum(axis=1) <= 0).any():
                raise ValueError(f"{source}: each p vector must have positive total mass")
        if frame["row_id"].duplicated().any():
            raise ValueError(f"{source}: row IDs are not unique")

        parts.append(frame)
        source_metadata.append({
            "source": source,
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "rows": int(len(frame)),
            "q_status": q_status,
            "q_version": str(q_version) if q_version else None,
            "group_column": column_map.get("group"),
        })

    all_p_columns = tuple(sorted(p_names))
    aligned_parts: list[pd.DataFrame] = []
    for part in parts:
        for col in all_p_columns:
            if col not in part:
                part[col] = np.nan
        aligned_parts.append(part)
    combined = pd.concat(aligned_parts, ignore_index=True, sort=False)
    if combined["row_id"].duplicated().any():
        raise ValueError("row_id collision across sources; use unique source labels or IDs")
    return Q2Dataset(combined, all_p_columns, tuple(source_metadata))


def model_feature_columns(dataset: Q2Dataset, model: str) -> tuple[str, ...]:
    """Resolve features and enforce the final-Q gate for the p+Q family."""
    if model not in MODEL_FEATURES:
        raise ValueError(f"model must be one of {sorted(MODEL_FEATURES)}")
    needed = list(MODEL_FEATURES[model])
    features = ["N", "D"]
    if "p" in needed:
        if not dataset.p_columns:
            raise ValueError(f"{model} requires p columns, but none were loaded")
        features.extend(dataset.p_columns)
    if "Q" in needed:
        q_sources = [s["source"] for s in dataset.sources if s["q_status"] != "final"]
        if q_sources:
            raise ValueError(f"M2 is locked until final Q is frozen; non-final sources: {q_sources}")
        if "Q" not in dataset.frame or not dataset.frame["Q"].notna().all():
            raise ValueError("M2 requires a complete final Q column")
        features.append("Q")
    return tuple(features)


def select_complete_cases(dataset: Q2Dataset, models: Sequence[str]) -> Q2Dataset:
    """Explicitly select one common row set complete for every compared model."""
    if not models:
        raise ValueError("models must not be empty")
    columns: set[str] = {"row_id", "N", "D", "loss", "group"}
    for model in models:
        columns.update(model_feature_columns(dataset, model))
    frame = dataset.frame.copy()
    used = sorted(columns - {"row_id", "group"})
    eligible = frame.dropna(subset=used + ["group"]).copy()
    if eligible.empty:
        raise ValueError("no complete cases remain for the requested model comparison")
    return Q2Dataset(eligible.reset_index(drop=True), dataset.p_columns, dataset.sources)


def make_cv_splits(
    dataset: Q2Dataset,
    *,
    n_splits: int = 5,
    seed: int = 20260924,
    group_col: str = "group",
) -> dict[str, Any]:
    """Create deterministic K-fold splits, keeping each group in one fold.

    The returned manifest uses row IDs rather than row positions, so each
    model run can reload and validate the exact same assignment.
    """
    frame = dataset.frame
    if "row_id" not in frame or frame["row_id"].duplicated().any():
        raise ValueError("dataset must have unique row_id values")
    if group_col not in frame:
        raise ValueError(f"group column {group_col!r} is missing")
    if frame[group_col].isna().any():
        raise ValueError("group values are required for grouped CV and bootstrap")
    unique_groups = sorted(frame[group_col].astype(str).unique().tolist())
    if not 2 <= n_splits <= len(unique_groups):
        raise ValueError(f"n_splits must be between 2 and {len(unique_groups)} groups")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(unique_groups, dtype=object)
    rng.shuffle(shuffled)
    fold_groups = np.array_split(shuffled, n_splits)
    all_rows = set(frame["row_id"].astype(str))
    split_rows: list[dict[str, Any]] = []
    for fold_id, held_groups in enumerate(fold_groups, start=1):
        held = set(str(value) for value in held_groups)
        test_ids = sorted(frame.loc[frame[group_col].astype(str).isin(held), "row_id"].astype(str).tolist())
        train_ids = sorted(all_rows - set(test_ids))
        if not train_ids or not test_ids or set(train_ids) & set(test_ids):
            raise RuntimeError(f"invalid fold {fold_id}: empty or overlapping train/test IDs")
        if set(frame.loc[frame["row_id"].astype(str).isin(train_ids), group_col].astype(str)) & held:
            raise RuntimeError(f"group leakage detected in fold {fold_id}")
        split_rows.append({"fold": fold_id, "train_ids": train_ids, "test_ids": test_ids, "test_groups": sorted(held)})

    return {
        "schema_version": 1,
        "seed": int(seed),
        "n_splits": int(n_splits),
        "split_unit": group_col,
        "row_count": int(len(frame)),
        "row_ids": sorted(all_rows),
        "group_count": len(unique_groups),
        "folds": split_rows,
    }


def save_cv_splits(splits: Mapping[str, Any], path: str | Path) -> None:
    _write_json(Path(path), dict(splits))


def load_cv_splits(path: str | Path) -> dict[str, Any]:
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if result.get("schema_version") != 1 or not result.get("folds"):
        raise ValueError("unsupported or empty CV split manifest")
    return result


def score_predictions(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, Any]:
    observed = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if observed.shape != predicted.shape or observed.size == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays of the same shape")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("y_true and y_pred must be finite")
    residual = predicted - observed
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else None
    return {
        "n": int(observed.size),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(r2) if r2 is not None and math.isfinite(r2) else None,
    }


def evaluate_model(
    fit_predict: FitPredict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str = "loss",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit one model and score it on one pre-defined fold."""
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    missing = (set(feature_columns) | {target_column}) - set(train.columns)
    if missing or (set(feature_columns) | {target_column}) - set(test.columns):
        raise ValueError(f"train/test frame missing columns: {sorted(missing | ((set(feature_columns) | {target_column}) - set(test.columns)))}")
    x_train = train[list(feature_columns)].astype(float)
    x_test = test[list(feature_columns)].astype(float)
    y_train = train[target_column].to_numpy(dtype=float)
    y_test = test[target_column].to_numpy(dtype=float)
    if not np.isfinite(x_train.to_numpy()).all() or not np.isfinite(x_test.to_numpy()).all():
        raise ValueError("features contain missing or non-finite values; select complete cases explicitly")
    prediction = np.asarray(fit_predict(x_train, y_train, x_test), dtype=float).reshape(-1)
    metrics = score_predictions(y_test, prediction)
    output = pd.DataFrame({
        "row_id": test["row_id"].astype(str).to_numpy(),
        "source": test["source"].astype(str).to_numpy(),
        "group": test["group"].astype(str).to_numpy(),
        "actual": y_test,
        "predicted": prediction,
    })
    output["residual_pred_minus_actual"] = output["predicted"] - output["actual"]
    output["absolute_error"] = output["residual_pred_minus_actual"].abs()
    return metrics, output


def bootstrap_eval(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    groups: Sequence[Any],
    *,
    n_resamples: int = 2000,
    seed: int = 20260924,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cluster-bootstrap RMSE, MAE, and R2 over already-held-out predictions."""
    observed = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    group_values = np.asarray(groups, dtype=object).reshape(-1)
    if not (len(observed) == len(predicted) == len(group_values)) or not len(observed):
        raise ValueError("y_true, y_pred, and groups must have equal non-zero lengths")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all() or pd.isna(group_values).any():
        raise ValueError("bootstrap inputs must be finite and groups non-missing")
    if n_resamples < 1 or not 0 < confidence < 1:
        raise ValueError("n_resamples must be positive and confidence must be in (0, 1)")
    group_strings = group_values.astype(str)
    unique_groups = np.unique(group_strings)
    if len(unique_groups) < 2:
        raise ValueError("cluster bootstrap needs at least two independent groups")
    group_indices = {group: np.flatnonzero(group_strings == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []
    for replicate in range(1, n_resamples + 1):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        rows = np.concatenate([group_indices[str(group)] for group in picked])
        metrics = score_predictions(observed[rows], predicted[rows])
        samples.append({"replicate": replicate, **{key: metrics[key] for key in METRIC_NAMES}})
    table = pd.DataFrame(samples)
    alpha = (1.0 - confidence) / 2.0
    summary = {
        "method": "cluster bootstrap of out-of-fold prediction rows; sample independent groups with replacement and retain all rows in each selected group",
        "group_count": int(len(unique_groups)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "confidence": float(confidence),
        "intervals": {},
        "limitation": "Intervals describe held-out prediction metrics conditional on the fixed CV assignments; a small number of independent groups limits precision.",
    }
    for metric in METRIC_NAMES:
        values = table[metric].to_numpy(dtype=float)
        summary["intervals"][metric] = {
            "lower": float(np.quantile(values, alpha, method="linear")),
            "upper": float(np.quantile(values, 1.0 - alpha, method="linear")),
        }
    return table, summary


def run_experiment(
    dataset: Q2Dataset,
    *,
    experiment: str,
    model: str,
    fit_predict: FitPredict,
    splits: Mapping[str, Any],
    output_dir: str | Path,
    seed: int = 20260924,
    bootstrap_resamples: int = 2000,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one model over a shared persisted split manifest and write outputs."""
    features = model_feature_columns(dataset, model)
    frame = dataset.frame.copy()
    if splits.get("split_unit") != "group":
        raise ValueError("experiment evaluation requires the canonical group-based split manifest")
    expected_ids = set(str(v) for v in splits["row_ids"])
    frame_ids = set(frame["row_id"].astype(str))
    if frame_ids != expected_ids:
        raise ValueError("dataset row IDs differ from the persisted common split; rebuild only before any model comparison starts")
    if frame[list(features) + ["loss", "group"]].isna().any().any():
        raise ValueError("dataset has incomplete model inputs; call select_complete_cases for every compared model before splitting")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    by_id = frame.set_index(frame["row_id"].astype(str), drop=False)
    fold_metrics: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    seen_test_ids: list[str] = []
    for split in splits["folds"]:
        train_ids = [str(value) for value in split["train_ids"]]
        test_ids = [str(value) for value in split["test_ids"]]
        if set(train_ids) | set(test_ids) != expected_ids or set(train_ids) & set(test_ids):
            raise ValueError(f"fold {split['fold']} does not partition the common row set")
        train = by_id.loc[train_ids].reset_index(drop=True)
        test = by_id.loc[test_ids].reset_index(drop=True)
        train_groups = set(train["group"].astype(str))
        test_groups = set(test["group"].astype(str))
        if train_groups & test_groups:
            raise ValueError(f"group leakage detected in fold {split['fold']}")
        metrics, predictions = evaluate_model(fit_predict, train, test, feature_columns=features)
        metrics["fold"] = int(split["fold"])
        fold_metrics.append(metrics)
        predictions["fold"] = int(split["fold"])
        prediction_parts.append(predictions)
        seen_test_ids.extend(test_ids)
    if sorted(seen_test_ids) != sorted(expected_ids):
        raise ValueError("CV folds must predict every common row exactly once")

    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(["fold", "row_id"]).reset_index(drop=True)
    overall = score_predictions(predictions["actual"], predictions["predicted"])
    bootstrap_table, bootstrap_summary = bootstrap_eval(
        predictions["actual"], predictions["predicted"], predictions["group"],
        n_resamples=bootstrap_resamples, seed=seed,
    )
    config_payload = {
        "schema_version": 1,
        "experiment": experiment,
        "model": model,
        "features": list(features),
        "target": "loss",
        "seed": int(seed),
        "cv": {key: splits[key] for key in ("seed", "n_splits", "split_unit", "row_count", "group_count")},
        "bootstrap_resamples": int(bootstrap_resamples),
        "data_sources": list(dataset.sources),
        "q_status_by_source": {item["source"]: item["q_status"] for item in dataset.sources},
        "q_version_by_source": {item["source"]: item["q_version"] for item in dataset.sources},
        "p_columns": list(dataset.p_columns),
        "created_local_date": pd.Timestamp.now(tz="Asia/Shanghai").date().isoformat(),
        "additional": dict(config or {}),
    }
    metrics_payload = {
        "overall_oof": overall,
        "folds": fold_metrics,
        "bootstrap": bootstrap_summary,
    }
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    bootstrap_table.to_csv(output / "bootstrap.csv", index=False, encoding="utf-8-sig")
    _write_json(output / "config.json", config_payload)
    _write_json(output / "metrics.json", metrics_payload)
    return {"config": config_payload, "metrics": metrics_payload, "output_dir": output.as_posix()}
