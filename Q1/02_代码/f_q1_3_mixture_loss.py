"""Q1.3 p-only mixture-to-loss modeling and read-only source audit.

The audit mode uses only the Python standard library. Model modes additionally
require NumPy. Run from the project root:

    python src/f_q1_3_mixture_loss.py --mode audit --data-root <regmix_tables>
    python src/f_q1_3_mixture_loss.py --mode p1
    python src/f_q1_3_mixture_loss.py --mode full
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # Audit mode deliberately remains stdlib-only.
    np = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "中文题目" / "F题" / "real_attachments" / "A_data_value" / "regmix_tables"
DATA_ROOT = DEFAULT_DATA_ROOT
SEED = 20260923
LAMBDA_GRID = np.concatenate(([0.0], np.logspace(-6, 1, 15))) if np is not None else ()
QUADRATIC_LAMBDA_GRID = np.logspace(-5, 2, 12) if np is not None else ()
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4
REFERENCE_DOMAIN_INDEX = -1
OLS_CONDITION_WARNING_THRESHOLD = 1e7

SPLITS = [
    ("train_1m", "train_mixture_1m.csv", "train_pile_loss_1m.csv", "fit"),
    ("test_1m", "test_mixture_1m.csv", "test_pile_loss_1m.csv", "heldout_same_scale"),
    ("test_60m", "test_mixture_60m.csv", "test_pile_loss_60m.csv", "heldout_cross_scale"),
    ("test_1b", "test_mixture_1B.csv", "test_pile_loss_1B.csv", "heldout_cross_scale"),
    ("estimate_10b", "est_mixture_10b.csv", "est_pile_loss_10b.csv", "estimate_consistency_only"),
    ("estimate_70b", "est_mixture_70b.csv", "est_pile_loss_70b.csv", "estimate_consistency_only"),
]

# Frozen field contract from the visible A4-A15 table headers/data description.
# Audit mode compares every file against this literal contract; it never learns
# the global schema from whichever file happens to be read first.
CANONICAL_MIXTURE_COLUMNS = (
    "index",
    "train_the_pile_arxiv",
    "train_the_pile_freelaw",
    "train_the_pile_nih_exporter",
    "train_the_pile_pubmed_central",
    "train_the_pile_wikipedia_en",
    "train_the_pile_dm_mathematics",
    "train_the_pile_github",
    "train_the_pile_philpapers",
    "train_the_pile_stackexchange",
    "train_the_pile_enron_emails",
    "train_the_pile_gutenberg_pg_19",
    "train_the_pile_pile_cc",
    "train_the_pile_ubuntu_irc",
    "train_the_pile_europarl",
    "train_the_pile_hackernews",
    "train_the_pile_pubmed_abstracts",
    "train_the_pile_uspto_backgrounds",
)
CANONICAL_LOSS_COLUMNS = (
    "index",
    "metric/the_pile_arxiv_val_loss",
    "metric/the_pile_freelaw_val_loss",
    "metric/the_pile_pubmed_central_val_loss",
    "metric/the_pile_wikipedia_en_val_loss",
    "metric/the_pile_dm_mathematics_val_loss",
    "metric/the_pile_github_val_loss",
    "metric/the_pile_stackexchange_val_loss",
    "metric/the_pile_gutenberg_pg_19_val_loss",
    "metric/the_pile_pile_cc_val_loss",
    "metric/the_pile_ubuntu_irc_val_loss",
    "metric/the_pile_hackernews_val_loss",
    "metric/the_pile_pubmed_abstracts_val_loss",
    "metric/the_pile_uspto_backgrounds_val_loss",
)
AUDIT_SPLITS = (
    ("train_1m", "A4", "train_mixture_1m.csv", "A5", "train_pile_loss_1m.csv", "training"),
    ("test_1m", "A6", "test_mixture_1m.csv", "A7", "test_pile_loss_1m.csv", "validation_same_scale"),
    ("test_60m", "A8", "test_mixture_60m.csv", "A9", "test_pile_loss_60m.csv", "validation_cross_scale"),
    ("test_1b", "A10", "test_mixture_1B.csv", "A11", "test_pile_loss_1B.csv", "validation_cross_scale"),
    ("estimate_10b", "A12", "est_mixture_10b.csv", "A13", "est_pile_loss_10b.csv", "extrapolation_estimate_only"),
    ("estimate_70b", "A14", "est_mixture_70b.csv", "A15", "est_pile_loss_70b.csv", "extrapolation_estimate_only"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    """Prefer a project-relative path, but retain traceability for external data."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def require_numpy(mode: str) -> None:
    if np is None:
        raise RuntimeError(
            f"--mode {mode} requires NumPy. Audit mode remains available without it; "
            "install the project environment before running model code."
        )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def load_pair(mixture_name: str, loss_name: str, expected_p: list[str] | None = None,
              expected_y: list[str] | None = None) -> dict[str, Any]:
    require_numpy("p1/full")
    mixture_path = DATA_ROOT / mixture_name
    loss_path = DATA_ROOT / loss_name
    mcols, mrows = read_csv(mixture_path)
    ycols, yrows = read_csv(loss_path)
    if not mrows or not yrows:
        raise ValueError(f"Empty input pair: {mixture_name}, {loss_name}")
    if "index" not in mcols or "index" not in ycols:
        raise ValueError(f"Missing index column in {mixture_name} or {loss_name}")
    pcols = [c for c in mcols if c != "index"]
    losscols = [c for c in ycols if c != "index"]
    if expected_p is not None and pcols != expected_p:
        raise ValueError(f"Mixture columns/order differ in {mixture_name}")
    if expected_y is not None and losscols != expected_y:
        raise ValueError(f"Loss columns/order differ in {loss_name}")
    mids = [row["index"] for row in mrows]
    yids = [row["index"] for row in yrows]
    if len(mids) != len(set(mids)) or len(yids) != len(set(yids)):
        raise ValueError(f"Duplicate index within pair: {mixture_name}, {loss_name}")
    if mids != yids:
        raise ValueError(f"Index order or membership differs: {mixture_name}, {loss_name}")
    if len(pcols) != 17 or len(losscols) != 13:
        raise ValueError(f"Expected 17 mixture and 13 Loss columns; got {len(pcols)} and {len(losscols)}")
    try:
        x = np.asarray([[float(row[c]) for c in pcols] for row in mrows], dtype=float)
        y = np.asarray([[float(row[c]) for c in losscols] for row in yrows], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric value in {mixture_name} or {loss_name}: {exc}") from exc
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite value in {mixture_name} or {loss_name}")
    if np.any(x < 0):
        raise ValueError(f"Negative mixture share in {mixture_name}")
    row_sums = x.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError(f"Zero-sum mixture row in {mixture_name}")
    return {
        "mixture_path": mixture_path,
        "loss_path": loss_path,
        "ids": mids,
        "pcols": pcols,
        "ycols": losscols,
        "x": x,
        "p": x / row_sums[:, None],
        "y": y,
        "row_sums": row_sums,
    }


def recipe_groups(p: np.ndarray) -> list[np.ndarray]:
    """Group exactly equal closed compositions (12 decimal places only absorbs float noise)."""
    buckets: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for i, row in enumerate(p):
        buckets[tuple(np.round(row, 12))].append(i)
    return [np.asarray(v, dtype=int) for v in buckets.values()]


def make_folds(p: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = recipe_groups(p)
    if len(groups) < n_splits:
        raise ValueError(f"Need at least {n_splits} unique recipes, found {len(groups)}")
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(groups)))
    # Largest groups first, with seeded order as deterministic tie-breaker.
    order.sort(key=lambda idx: -len(groups[idx]))
    fold_groups: list[list[np.ndarray]] = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    for group_idx in order:
        fold = int(np.argmin(fold_sizes))
        fold_groups[fold].append(groups[group_idx])
        fold_sizes[fold] += len(groups[group_idx])
    all_indices = np.arange(len(p))
    result = []
    for pieces in fold_groups:
        valid = np.concatenate(pieces)
        train = np.setdiff1d(all_indices, valid, assume_unique=True)
        result.append((train, valid))
    return result


def helmert_basis(dimension: int) -> np.ndarray:
    """Return an orthonormal basis for vectors whose components sum to zero."""
    basis = np.zeros((dimension, dimension - 1), dtype=float)
    for j in range(dimension - 1):
        denom = math.sqrt((j + 1) * (j + 2))
        basis[:j + 1, j] = 1.0 / denom
        basis[j + 1, j] = -(j + 1) / denom
    if not np.allclose(basis.T @ basis, np.eye(dimension - 1), atol=1e-12):
        raise ArithmeticError("Contrast basis is not orthonormal")
    if not np.allclose(np.ones(dimension) @ basis, 0.0, atol=1e-12):
        raise ArithmeticError("Contrast basis does not sum to zero")
    return basis


def fit_mean(y: np.ndarray) -> dict[str, Any]:
    return {"kind": "mean", "intercept": y.mean(axis=0), "coef": np.zeros((17, y.shape[1])), "lambda": 0.0}


def predict_mean(model: dict[str, Any], p: np.ndarray) -> np.ndarray:
    return np.tile(model["intercept"], (len(p), 1))


def fit_reference_ols(p: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit an unregularized linear comparator identified on the simplex.

    Fixing one reference share coefficient to zero removes the intercept/share
    alias created by sum(p)=1. This candidate is assessed by grouped CV; its
    presence in hidden text is neither a reason to include nor exclude it.
    """
    x = np.column_stack([np.ones(len(p)), p[:, :REFERENCE_DOMAIN_INDEX]])
    coef, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise ValueError(f"Reference baseline design rank deficient: {rank}/{x.shape[1]}")
    condition_number = float(np.linalg.cond(x))
    b = np.zeros((p.shape[1], y.shape[1]), dtype=float)
    b[:REFERENCE_DOMAIN_INDEX, :] = coef[1:, :]
    return {
        "kind": "reference_ols",
        "intercept": coef[0, :],
        "coef": b,
        "lambda": 0.0,
        "rank": int(rank),
        "condition_number": condition_number,
        "condition_warning": bool(condition_number > OLS_CONDITION_WARNING_THRESHOLD),
        "singular_values": singular,
    }


def fit_simplex_ridge(p: np.ndarray, y: np.ndarray, penalty: float,
                      basis: np.ndarray | None = None) -> dict[str, Any]:
    """Fit the sum-to-zero ridge model using fold-local target scales.

    Minimizes mean squared standardized residuals plus penalty * ||B||_F^2.
    The output-scale term therefore enters each contrast-space ridge penalty.
    """
    basis = basis if basis is not None else helmert_basis(p.shape[1])
    z = p @ basis
    z_mean = z.mean(axis=0)
    zc = z - z_mean
    y_mean = y.mean(axis=0)
    y_scale = y.std(axis=0, ddof=1) if len(y) > 1 else np.ones(y.shape[1])
    zero_scale = y_scale <= np.finfo(float).eps
    y_scale[zero_scale] = 1.0
    ys = (y - y_mean) / y_scale
    gram = zc.T @ zc / len(z)
    rhs = zc.T @ ys / len(z)
    theta = np.zeros((basis.shape[1], y.shape[1]), dtype=float)
    identity = np.eye(basis.shape[1])
    for k in range(y.shape[1]):
        # B[:,k] = basis @ theta[:,k] * y_scale[k].
        # This diagonal term implements penalty * ||B||^2 exactly.
        penalty_k = penalty * (y_scale[k] ** 2)
        system = gram + penalty_k * identity
        theta[:, k] = np.linalg.solve(system, rhs[:, k])
        if zero_scale[k]:
            theta[:, k] = 0.0
    b = basis @ (theta * y_scale[None, :])
    intercept = y_mean - z_mean @ (theta * y_scale[None, :])
    if not np.allclose(b.sum(axis=0), 0.0, atol=1e-9):
        raise ArithmeticError("Ridge coefficients violate the sum-to-zero constraint")
    return {
        "kind": "simplex_ridge",
        "intercept": intercept,
        "coef": b,
        "lambda": float(penalty),
        "target_scale": y_scale,
        "zero_scale_targets": np.flatnonzero(zero_scale).tolist(),
    }


def fit_simplex_huber_ridge(p: np.ndarray, y: np.ndarray, penalty: float,
                            delta: float = 1.345, max_iter: int = 200,
                            tol: float = 1e-9,
                            basis: np.ndarray | None = None) -> dict[str, Any]:
    """Fit a zero-sum simplex ridge model with Huber residual loss.

    Huber weights are target-specific and training-fold scales are used. No row
    is removed; this is a response-outlier sensitivity model, not a detector of
    malicious data or high-leverage mixture contamination.
    """
    basis = basis if basis is not None else helmert_basis(p.shape[1])
    z = p @ basis
    z_mean = z.mean(axis=0)
    zc = z - z_mean
    y_mean = y.mean(axis=0)
    y_scale = y.std(axis=0, ddof=1) if len(y) > 1 else np.ones(y.shape[1])
    zero_scale = y_scale <= np.finfo(float).eps
    y_scale[zero_scale] = 1.0
    ys = (y - y_mean) / y_scale
    design = np.column_stack([np.ones(len(p)), zc])
    n_parameters = design.shape[1]
    theta = np.zeros((n_parameters, y.shape[1]), dtype=float)
    theta[0, :] = np.median(ys, axis=0)
    coefficients: list[np.ndarray] = []
    intercept_std = np.zeros(y.shape[1], dtype=float)
    weights = np.ones_like(y, dtype=float)
    iterations = np.zeros(y.shape[1], dtype=int)
    converged = np.ones(y.shape[1], dtype=bool)
    identity = np.eye(n_parameters)
    for k in range(y.shape[1]):
        if zero_scale[k]:
            weights[:, k] = 1.0
            continue
        # Objective is mean Huber loss + lambda * ||B_k||^2. Since its
        # derivative contributes 2*lambda*B, standardized contrast coordinates
        # use the per-target penalty 2*lambda*s_k^2.
        penalty_matrix = np.diag(np.r_[0.0, np.full(basis.shape[1], 2.0 * penalty * y_scale[k] ** 2)])
        beta = theta[:, k].copy()
        for iteration in range(1, max_iter + 1):
            residual = ys[:, k] - design @ beta
            absolute = np.abs(residual)
            w = np.ones_like(absolute)
            tail = absolute > delta
            w[tail] = delta / absolute[tail]
            lhs = (design.T * w) @ design / len(p) + penalty_matrix
            rhs = (design.T * w) @ ys[:, k] / len(p)
            try:
                updated = np.linalg.solve(lhs + np.finfo(float).eps * identity, rhs)
            except np.linalg.LinAlgError:
                updated = np.linalg.lstsq(lhs + np.finfo(float).eps * identity, rhs, rcond=None)[0]
            change = np.max(np.abs(updated - beta))
            beta = updated
            if change <= tol * (1.0 + np.max(np.abs(beta))):
                break
        else:
            converged[k] = False
        residual = ys[:, k] - design @ beta
        absolute = np.abs(residual)
        w = np.ones_like(absolute)
        tail = absolute > delta
        w[tail] = delta / absolute[tail]
        weights[:, k] = w
        intercept_std[k] = beta[0]
        coefficients.append(beta[1:])
        iterations[k] = iteration
    theta_contrast = np.column_stack(coefficients) if coefficients else np.zeros((basis.shape[1], 0))
    # Keep output width stable even when one or more targets have zero scale.
    theta_full = np.zeros((basis.shape[1], y.shape[1]), dtype=float)
    nonzero_targets = [k for k in range(y.shape[1]) if not zero_scale[k]]
    if nonzero_targets:
        theta_full[:, nonzero_targets] = theta_contrast
    b = basis @ (theta_full * y_scale[None, :])
    intercept = y_mean + intercept_std * y_scale - z_mean @ (theta_full * y_scale[None, :])
    if not np.allclose(b.sum(axis=0), 0.0, atol=1e-9):
        raise ArithmeticError("Huber ridge coefficients violate the sum-to-zero constraint")
    return {
        "kind": "simplex_huber_ridge",
        "intercept": intercept,
        "coef": b,
        "lambda": float(penalty),
        "delta": float(delta),
        "target_scale": y_scale,
        "weights": weights,
        "iterations": iterations,
        "converged": converged,
        "zero_scale_targets": np.flatnonzero(zero_scale).tolist(),
    }


def predict_huber(model: dict[str, Any], p: np.ndarray) -> np.ndarray:
    return model["intercept"] + p @ model["coef"]


def quadratic_feature_matrix(p: np.ndarray) -> np.ndarray:
    """Canonical zero-friendly quadratic mixture terms: p_j and p_j*p_l."""
    pair_terms = [p[:, j] * p[:, ell]
                  for j in range(p.shape[1]) for ell in range(j + 1, p.shape[1])]
    return np.column_stack([p, *pair_terms])


def quadratic_feature_names(pcols: list[str]) -> list[str]:
    return pcols + [f"{pcols[j]}*{pcols[ell]}"
                    for j in range(len(pcols)) for ell in range(j + 1, len(pcols))]


def fit_quadratic_ridge(p: np.ndarray, y: np.ndarray, penalty: float) -> dict[str, Any]:
    """Fit a second-order Scheffe mixture surface with an unpenalized intercept.

    The 17 linear mixture terms can represent a constant on the simplex, so an
    explicit intercept would be aliased in the uncentered basis. Centering the
    feature matrix and responses within the training fold gives an unpenalized
    intercept equal to the fold response mean. Positive ridge shrinkage handles
    the 153-term design without changing zero shares in the raw feature map.
    """
    x = quadratic_feature_matrix(p)
    feature_center = x.mean(axis=0)
    x_centered = x - feature_center[None, :]
    feature_scale = np.sqrt(np.mean(x_centered ** 2, axis=0))
    feature_scale[feature_scale <= np.finfo(float).eps] = 1.0
    z = x_centered / feature_scale[None, :]
    y_mean = y.mean(axis=0)
    y_centered = y - y_mean[None, :]
    y_scale = y_centered.std(axis=0, ddof=1) if len(y) > 1 else np.ones(y.shape[1])
    zero_scale = y_scale <= np.finfo(float).eps
    y_scale[zero_scale] = 1.0
    ys = y_centered / y_scale[None, :]
    gram = z.T @ z / len(z)
    rhs = z.T @ ys / len(z)
    theta = np.linalg.solve(gram + float(penalty) * np.eye(gram.shape[0]), rhs)
    theta[:, zero_scale] = 0.0
    coef = theta * y_scale[None, :] / feature_scale[:, None]
    return {
        "kind": "quadratic_ridge",
        "coef": coef,
        "intercept": y_mean,
        "lambda": float(penalty),
        "feature_center": feature_center,
        "feature_scale": feature_scale,
        "target_scale": y_scale,
        "zero_scale_targets": np.flatnonzero(zero_scale).tolist(),
    }


def predict_quadratic(model: dict[str, Any], p: np.ndarray) -> np.ndarray:
    x = quadratic_feature_matrix(p)
    return model["intercept"] + (x - model["feature_center"][None, :]) @ model["coef"]


def predict_linear(model: dict[str, Any], p: np.ndarray) -> np.ndarray:
    return model["intercept"] + p @ model["coef"]


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float(rx @ rx) * float(ry @ ry))
    if denom <= np.finfo(float).eps:
        return None
    return float((rx @ ry) / denom)


def standardized_mse(y_true: np.ndarray, y_pred: np.ndarray, train_y: np.ndarray) -> float:
    scale = train_y.std(axis=0, ddof=1) if len(train_y) > 1 else np.ones(y_true.shape[1])
    scale[scale <= np.finfo(float).eps] = 1.0
    return float(np.mean(((y_true - y_pred) / scale[None, :]) ** 2))


def choose_lambda(p: np.ndarray, y: np.ndarray, seed: int,
                  n_splits: int = N_INNER_FOLDS,
                  candidates: np.ndarray = LAMBDA_GRID) -> tuple[float, list[dict[str, float]]]:
    folds = make_folds(p, min(n_splits, len(recipe_groups(p))), seed)
    scores: list[dict[str, float]] = []
    basis = helmert_basis(p.shape[1])
    for candidate in candidates:
        total_sq = 0.0
        total_values = 0
        for train_idx, valid_idx in folds:
            model = fit_simplex_ridge(p[train_idx], y[train_idx], float(candidate), basis)
            pred = predict_linear(model, p[valid_idx])
            scale = y[train_idx].std(axis=0, ddof=1)
            scale[scale <= np.finfo(float).eps] = 1.0
            residual = (y[valid_idx] - pred) / scale[None, :]
            total_sq += float(np.sum(residual ** 2))
            total_values += residual.size
        scores.append({"lambda": float(candidate), "standardized_mse": total_sq / total_values})
    best = min(scores, key=lambda row: (row["standardized_mse"], -row["lambda"]))
    return best["lambda"], scores


def choose_huber_lambda(p: np.ndarray, y: np.ndarray, seed: int,
                        n_splits: int = N_INNER_FOLDS,
                        candidates: np.ndarray = LAMBDA_GRID) -> tuple[float, list[dict[str, float]]]:
    """Select Huber-ridge shrinkage by grouped CV on the supplied train fold."""
    folds = make_folds(p, min(n_splits, len(recipe_groups(p))), seed)
    scores: list[dict[str, float]] = []
    for candidate in candidates:
        total_sq = 0.0
        total_values = 0
        for train_idx, valid_idx in folds:
            model = fit_simplex_huber_ridge(p[train_idx], y[train_idx], float(candidate))
            if not np.all(model["converged"]):
                raise ArithmeticError("Huber IRLS failed to converge during lambda selection")
            pred = predict_huber(model, p[valid_idx])
            scale = y[train_idx].std(axis=0, ddof=1)
            scale[scale <= np.finfo(float).eps] = 1.0
            residual = (y[valid_idx] - pred) / scale[None, :]
            total_sq += float(np.sum(residual ** 2))
            total_values += residual.size
        scores.append({"lambda": float(candidate), "standardized_mse": total_sq / total_values})
    best = min(scores, key=lambda row: (row["standardized_mse"], -row["lambda"]))
    return best["lambda"], scores


def nested_huber_cv(p: np.ndarray, y: np.ndarray, seed: int) -> dict[str, Any]:
    """Nested grouped-CV sensitivity run; each outer fold selects lambda internally."""
    outer_folds = make_folds(p, N_OUTER_FOLDS, seed)
    prediction = np.full_like(y, np.nan, dtype=float)
    fold_id = np.full(len(p), -1, dtype=int)
    selected_lambdas: list[float] = []
    fold_scores: list[float] = []
    for outer_id, (train_idx, valid_idx) in enumerate(outer_folds):
        penalty, _ = choose_huber_lambda(
            p[train_idx], y[train_idx], seed + 2000 + outer_id, N_INNER_FOLDS
        )
        model = fit_simplex_huber_ridge(p[train_idx], y[train_idx], penalty)
        if not np.all(model["converged"]):
            raise ArithmeticError(f"Huber IRLS failed to converge in outer fold {outer_id}")
        pred = predict_huber(model, p[valid_idx])
        prediction[valid_idx] = pred
        fold_scores.append(standardized_mse(y[valid_idx], pred, y[train_idx]))
        selected_lambdas.append(penalty)
        fold_id[valid_idx] = outer_id
    if np.any(fold_id < 0) or not np.isfinite(prediction).all():
        raise ArithmeticError("Huber nested CV did not predict every training row")
    return {
        "predictions": prediction,
        "fold_id": fold_id,
        "fold_scores": fold_scores,
        "normalized_rmse": math.sqrt(float(np.mean(fold_scores))),
        "outer_selected_lambdas": selected_lambdas,
    }


def choose_quadratic_lambda(p: np.ndarray, y: np.ndarray, seed: int,
                            n_splits: int = N_INNER_FOLDS) -> tuple[float, list[dict[str, float]]]:
    """Select quadratic ridge shrinkage using grouped CV on the supplied data."""
    folds = make_folds(p, min(n_splits, len(recipe_groups(p))), seed)
    scores: list[dict[str, float]] = []
    for candidate in QUADRATIC_LAMBDA_GRID:
        total_sq = 0.0
        total_values = 0
        for train_idx, valid_idx in folds:
            model = fit_quadratic_ridge(p[train_idx], y[train_idx], float(candidate))
            pred = predict_quadratic(model, p[valid_idx])
            scale = y[train_idx].std(axis=0, ddof=1)
            scale[scale <= np.finfo(float).eps] = 1.0
            residual = (y[valid_idx] - pred) / scale[None, :]
            total_sq += float(np.sum(residual ** 2))
            total_values += residual.size
        scores.append({"lambda": float(candidate), "standardized_mse": total_sq / total_values})
    best = min(scores, key=lambda row: (row["standardized_mse"], -row["lambda"]))
    return best["lambda"], scores


def nested_quadratic_cv(p: np.ndarray, y: np.ndarray, seed: int) -> dict[str, Any]:
    """Nested grouped-CV evaluation of the prespecified quadratic sensitivity."""
    outer_folds = make_folds(p, N_OUTER_FOLDS, seed)
    prediction = np.full_like(y, np.nan, dtype=float)
    fold_id = np.full(len(p), -1, dtype=int)
    selected_lambdas: list[float] = []
    fold_scores: list[float] = []
    for outer_id, (train_idx, valid_idx) in enumerate(outer_folds):
        penalty, _ = choose_quadratic_lambda(
            p[train_idx], y[train_idx], seed + 3000 + outer_id, N_INNER_FOLDS
        )
        model = fit_quadratic_ridge(p[train_idx], y[train_idx], penalty)
        pred = predict_quadratic(model, p[valid_idx])
        prediction[valid_idx] = pred
        fold_scores.append(standardized_mse(y[valid_idx], pred, y[train_idx]))
        selected_lambdas.append(penalty)
        fold_id[valid_idx] = outer_id
    if np.any(fold_id < 0) or not np.isfinite(prediction).all():
        raise ArithmeticError("Quadratic nested CV did not predict every training row")
    return {
        "predictions": prediction,
        "fold_id": fold_id,
        "fold_scores": fold_scores,
        "normalized_rmse": math.sqrt(float(np.mean(fold_scores))),
        "outer_selected_lambdas": selected_lambdas,
    }


def fixed_model_cv_score(p: np.ndarray, y: np.ndarray, model_name: str,
                         seed: int, n_splits: int) -> float:
    """Grouped-CV standardized MSE for mean or reference-component OLS."""
    folds = make_folds(p, min(n_splits, len(recipe_groups(p))), seed)
    total_sq = 0.0
    total_values = 0
    for train_idx, valid_idx in folds:
        pt, pv = p[train_idx], p[valid_idx]
        yt, yv = y[train_idx], y[valid_idx]
        if model_name == "mean":
            model = fit_mean(yt)
            pred = predict_mean(model, pv)
        elif model_name == "reference_ols":
            model = fit_reference_ols(pt, yt)
            pred = predict_linear(model, pv)
        else:
            raise ValueError(f"Unsupported fixed model: {model_name}")
        scale = yt.std(axis=0, ddof=1)
        scale[scale <= np.finfo(float).eps] = 1.0
        residual = (yv - pred) / scale[None, :]
        total_sq += float(np.sum(residual ** 2))
        total_values += residual.size
    return total_sq / total_values


def choose_model_family(p: np.ndarray, y: np.ndarray, seed: int,
                        n_splits: int) -> tuple[str, dict[str, float], float, list[dict[str, float]]]:
    """Select mean, OLS, or ridge using only the supplied training partition."""
    scores = {
        "mean": fixed_model_cv_score(p, y, "mean", seed, n_splits),
        "reference_ols": fixed_model_cv_score(p, y, "reference_ols", seed, n_splits),
    }
    ridge_lambda, lambda_scores = choose_lambda(p, y, seed, n_splits)
    scores["simplex_ridge"] = min(row["standardized_mse"] for row in lambda_scores)
    simplicity = {"mean": 0, "reference_ols": 1, "simplex_ridge": 2}
    selected = min(scores, key=lambda name: (round(scores[name], 12), simplicity[name]))
    return selected, scores, ridge_lambda, lambda_scores


def nested_cv(p: np.ndarray, y: np.ndarray, seed: int) -> dict[str, Any]:
    outer_folds = make_folds(p, N_OUTER_FOLDS, seed)
    predictions = {
        "mean": np.full_like(y, np.nan, dtype=float),
        "reference_ols": np.full_like(y, np.nan, dtype=float),
        "simplex_ridge": np.full_like(y, np.nan, dtype=float),
        "nested_selected": np.full_like(y, np.nan, dtype=float),
    }
    fold_id = np.full(len(p), -1, dtype=int)
    selected_lambdas: list[float] = []
    ols_conditions: list[float] = []
    outer_selected_models: list[str] = []
    outer_inner_model_scores: list[dict[str, float]] = []
    fold_scores = {name: [] for name in predictions}
    for outer_id, (train_idx, valid_idx) in enumerate(outer_folds):
        pt, pv = p[train_idx], p[valid_idx]
        yt, yv = y[train_idx], y[valid_idx]
        ols_model = fit_reference_ols(pt, yt)
        ols_conditions.append(ols_model["condition_number"])
        inner_seed = seed + 1000 + outer_id
        selected_family, inner_scores, penalty, _ = choose_model_family(
            pt, yt, inner_seed, N_INNER_FOLDS
        )
        mean_model = fit_mean(yt)
        ridge_model = fit_simplex_ridge(pt, yt, penalty)
        preds = {
            "mean": predict_mean(mean_model, pv),
            "reference_ols": predict_linear(ols_model, pv),
            "simplex_ridge": predict_linear(ridge_model, pv),
        }
        preds["nested_selected"] = preds[selected_family]
        for name, pred in preds.items():
            predictions[name][valid_idx] = pred
            fold_scores[name].append(standardized_mse(yv, pred, yt))
        selected_lambdas.append(penalty)
        outer_selected_models.append(selected_family)
        outer_inner_model_scores.append(inner_scores)
        fold_id[valid_idx] = outer_id
    if np.any(fold_id < 0) or any(not np.isfinite(value).all() for value in predictions.values()):
        raise ArithmeticError("Nested CV did not produce finite out-of-fold predictions for every row")
    normalized_rmse = {name: math.sqrt(float(np.mean(scores))) for name, scores in fold_scores.items()}
    return {
        "predictions": predictions,
        "fold_id": fold_id,
        "fold_scores": fold_scores,
        "normalized_rmse": normalized_rmse,
        "outer_selected_lambdas": selected_lambdas,
        "outer_selected_models": outer_selected_models,
        "outer_inner_model_scores": outer_inner_model_scores,
        "reference_ols_condition_numbers": ols_conditions,
    }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str],
                      split: str, model_name: str, role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for j, name in enumerate(names):
        error = y_pred[:, j] - y_true[:, j]
        rmse = float(np.sqrt(np.mean(error ** 2)))
        mae = float(np.mean(np.abs(error)))
        denominator = float(np.sum((y_true[:, j] - np.mean(y_true[:, j])) ** 2))
        r2 = None if denominator <= np.finfo(float).eps else float(1.0 - np.sum(error ** 2) / denominator)
        rho = spearman_rho(y_true[:, j], y_pred[:, j])
        rows.append({
            "split": split,
            "role": role,
            "model": model_name,
            "target": name,
            "n": len(y_true),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "spearman_rho": rho,
        })
    return rows


def residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str],
                         n_bins: int = 5) -> list[dict[str, Any]]:
    """Summarize OOF residual bias and error across prediction-rank bins."""
    rows: list[dict[str, Any]] = []
    for k, target in enumerate(names):
        pred = y_pred[:, k]
        observed = y_true[:, k]
        residual = pred - observed
        order = np.argsort(pred, kind="mergesort")
        bins = np.array_split(order, n_bins)
        groups: list[tuple[int, np.ndarray]] = [(0, np.arange(len(pred)))]
        groups.extend((i + 1, idx) for i, idx in enumerate(bins))
        for bin_id, idx in groups:
            e = residual[idx]
            rows.append({
                "target": target,
                "prediction_rank_bin": bin_id,
                "n": int(len(idx)),
                "mean_predicted": float(np.mean(pred[idx])),
                "mean_observed": float(np.mean(observed[idx])),
                "mean_error_pred_minus_observed": float(np.mean(e)),
                "mae": float(np.mean(np.abs(e))),
                "rmse": float(np.sqrt(np.mean(e ** 2))),
                "spearman_abs_error_vs_prediction": spearman_rho(np.abs(e), pred[idx]),
            })
    return rows


def bootstrap_substitution_effects(p: np.ndarray, y: np.ndarray,
                                   fitted_model: dict[str, Any],
                                   model_name: str, penalty: float | None,
                                   pcols: list[str], ycols: list[str],
                                   n_boot: int = 500,
                                   seed: int = SEED + 50000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pairs-bootstrap 10-pp substitution contrasts, conditional on model/lambda."""
    if model_name == "mean":
        return [], {"status": "not_applicable_mean_model", "requested": n_boot, "valid": 0}
    rng = np.random.default_rng(seed)
    coefficients: list[np.ndarray] = []
    failed_rank = 0
    for _ in range(n_boot):
        sample = rng.integers(0, len(p), size=len(p))
        try:
            if model_name == "simplex_ridge":
                model = fit_simplex_ridge(p[sample], y[sample], float(penalty))
            elif model_name == "reference_ols":
                model = fit_reference_ols(p[sample], y[sample])
            else:
                raise ValueError(f"No coefficient bootstrap for selected model {model_name}")
            coefficients.append(model["coef"])
        except ValueError as exc:
            if model_name == "reference_ols" and "rank deficient" in str(exc):
                failed_rank += 1
                continue
            raise
    if not coefficients:
        return [], {"status": "no_valid_bootstrap_fits", "requested": n_boot, "valid": 0,
                    "rank_deficient": failed_rank}
    beta = np.stack(coefficients, axis=0)
    fitted_beta = fitted_model["coef"]
    result: list[dict[str, Any]] = []
    for to_idx in range(len(pcols)):
        for from_idx in range(to_idx):
            contrast = 0.10 * (beta[:, to_idx, :] - beta[:, from_idx, :])
            point = 0.10 * (fitted_beta[to_idx, :] - fitted_beta[from_idx, :])
            lower, upper = np.quantile(contrast, [0.025, 0.975], axis=0)
            support_n = int(np.sum(p[:, from_idx] >= 0.10 - 1e-12))
            for k, target in enumerate(ycols):
                result.append({
                    "model": model_name,
                    "target": target,
                    "from_domain": pcols[from_idx],
                    "to_domain": pcols[to_idx],
                    "share_transfer": 0.10,
                    "effect_per_10pp_transfer": float(point[k]),
                    "bootstrap_ci95_lower": float(lower[k]),
                    "bootstrap_ci95_upper": float(upper[k]),
                    "bootstrap_probability_effect_positive": float(np.mean(contrast[:, k] > 0.0)),
                    "training_recipes_with_from_share_at_least_10pp": support_n,
                    "training_support_fraction": support_n / len(p),
                    "interpretation": "association conditional on the fitted linear model; feasible only where the source share is at least 10pp",
                })
    supported_rows = [row for row in result if row["training_recipes_with_from_share_at_least_10pp"] > 0]
    positive_ci = sum(row["bootstrap_ci95_lower"] > 0.0 for row in supported_rows)
    negative_ci = sum(row["bootstrap_ci95_upper"] < 0.0 for row in supported_rows)
    return result, {
        "status": "completed",
        "requested": n_boot,
        "valid": len(coefficients),
        "rank_deficient": failed_rank,
        "pair_count": len(pcols) * (len(pcols) - 1) // 2,
        "unsupported_10pp_pairs": sum(
            int(np.sum(p[:, i] >= 0.10 - 1e-12) == 0)
            for j in range(len(pcols)) for i in range(j)
        ),
        "supported_10pp_pairs": sum(
            int(np.sum(p[:, i] >= 0.10 - 1e-12) > 0)
            for j in range(len(pcols)) for i in range(j)
        ),
        "interval": "95% percentile pairs bootstrap",
        "interval_scope": "pointwise across target and substitution pairs; no multiplicity adjustment",
        "supported_pair_target_rows": len(supported_rows),
        "supported_intervals_entirely_positive": positive_ci,
        "supported_intervals_entirely_negative": negative_ci,
        "supported_intervals_cross_zero": len(supported_rows) - positive_ci - negative_ci,
        "conditional_on": ["selected model family", "selected ridge lambda", "observed training distribution"],
        "seed": seed,
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["role"], row["model"], row.get("variant", ""))].append(row)
    summary = []
    for (split, role, model, variant), items in grouped.items():
        r2s = [item["r2"] for item in items if item["r2"] is not None]
        summary.append({
            "split": split,
            "role": role,
            "model": model,
            "variant": variant,
            "n_targets": len(items),
            "nested_macro_normalized_rmse": items[0].get("nested_macro_normalized_rmse"),
            "macro_mae": float(np.mean([item["mae"] for item in items])),
            "macro_rmse": float(np.mean([item["rmse"] for item in items])),
            "macro_r2": float(np.mean(r2s)) if r2s else None,
            "macro_spearman_rho": float(np.mean([item["spearman_rho"] for item in items
                                                   if item["spearman_rho"] is not None]))
            if any(item["spearman_rho"] is not None for item in items) else None,
            "worst_rmse_target": max(items, key=lambda item: item["rmse"])["target"],
            "worst_rmse": max(item["rmse"] for item in items),
        })
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Need fieldnames for empty CSV: {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inspect_csv_structure(path: Path, expected_columns: tuple[str, ...]) -> dict[str, Any]:
    """Inspect schema and index only; numeric payloads remain untouched here."""
    result: dict[str, Any] = {
        "path": path,
        "exists": path.is_file(),
        "size_bytes": None,
        "sha256": None,
        "header": [],
        "rows": [],
        "row_count": 0,
        "schema_exact": False,
        "header_order_exact": False,
        "duplicate_header_names": [],
        "missing_columns": list(expected_columns),
        "unexpected_columns": [],
        "malformed_row_count": 0,
        "index_values": [],
        "blank_index_count": 0,
        "duplicate_index_count": 0,
        "duplicate_index_values": [],
        "read_error": None,
    }
    if not result["exists"]:
        result["read_error"] = "file_not_found"
        return result
    result["size_bytes"] = path.stat().st_size
    result["sha256"] = sha256(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            try:
                header = next(reader)
            except StopIteration:
                result["read_error"] = "empty_file"
                return result
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        result["read_error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["header"] = header
    result["rows"] = rows
    result["row_count"] = len(rows)
    header_counts = Counter(header)
    result["duplicate_header_names"] = sorted(
        name for name, count in header_counts.items() if count > 1
    )
    result["missing_columns"] = [name for name in expected_columns if header_counts[name] == 0]
    expected_counts = Counter(expected_columns)
    result["unexpected_columns"] = sorted(
        name for name, count in header_counts.items()
        for _ in range(max(0, count - expected_counts[name]))
    )
    result["header_order_exact"] = header == list(expected_columns)
    result["schema_exact"] = bool(
        result["header_order_exact"]
        and not result["duplicate_header_names"]
        and not result["missing_columns"]
        and not result["unexpected_columns"]
    )
    result["malformed_row_count"] = sum(len(row) != len(header) for row in rows)

    if header_counts["index"] == 1:
        index_position = header.index("index")
        indices = [row[index_position].strip() for row in rows if len(row) > index_position]
        result["index_values"] = indices
        result["blank_index_count"] = sum(not value for value in indices)
        index_counts = Counter(value for value in indices if value)
        duplicates = sorted(value for value, count in index_counts.items() if count > 1)
        result["duplicate_index_values"] = duplicates
        result["duplicate_index_count"] = sum(index_counts[value] - 1 for value in duplicates)
    return result


def public_file_role_row(document_id: str, split: str, role: str, table_type: str,
                         inspection: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "split": split,
        "role": role,
        "table_type": table_type,
        "file": portable_path(inspection["path"]),
        "exists": inspection["exists"],
        "size_bytes": inspection["size_bytes"],
        "sha256": inspection["sha256"],
        "row_count": inspection["row_count"],
        "schema_exact": inspection["schema_exact"],
        "header_order_exact": inspection["header_order_exact"],
        "duplicate_header_names": json.dumps(inspection["duplicate_header_names"], ensure_ascii=False),
        "missing_columns": json.dumps(inspection["missing_columns"], ensure_ascii=False),
        "unexpected_columns": json.dumps(inspection["unexpected_columns"], ensure_ascii=False),
        "malformed_row_count": inspection["malformed_row_count"],
        "blank_index_count": inspection["blank_index_count"],
        "duplicate_index_count": inspection["duplicate_index_count"],
        "read_error": inspection["read_error"] or "",
    }


def pair_integrity_row(split: str, role: str, mixture: dict[str, Any],
                       loss: dict[str, Any]) -> dict[str, Any]:
    mixture_ids = mixture["index_values"]
    loss_ids = loss["index_values"]
    mixture_set = {value for value in mixture_ids if value}
    loss_set = {value for value in loss_ids if value}
    missing_in_loss = sorted(mixture_set - loss_set)
    extra_in_loss = sorted(loss_set - mixture_set)
    valid = bool(
        mixture["schema_exact"]
        and loss["schema_exact"]
        and mixture["read_error"] is None
        and loss["read_error"] is None
        and mixture["malformed_row_count"] == 0
        and loss["malformed_row_count"] == 0
        and mixture["blank_index_count"] == 0
        and loss["blank_index_count"] == 0
        and mixture["duplicate_index_count"] == 0
        and loss["duplicate_index_count"] == 0
        and len(mixture_ids) == len(loss_ids)
        and not missing_in_loss
        and not extra_in_loss
        and mixture_ids == loss_ids
    )
    return {
        "split": split,
        "role": role,
        "mixture_rows": len(mixture_ids),
        "loss_rows": len(loss_ids),
        "mixture_duplicate_index_count": mixture["duplicate_index_count"],
        "loss_duplicate_index_count": loss["duplicate_index_count"],
        "mixture_blank_index_count": mixture["blank_index_count"],
        "loss_blank_index_count": loss["blank_index_count"],
        "missing_in_loss_count": len(missing_in_loss),
        "extra_in_loss_count": len(extra_in_loss),
        "missing_in_loss_sample": json.dumps(missing_in_loss[:10], ensure_ascii=False),
        "extra_in_loss_sample": json.dumps(extra_in_loss[:10], ensure_ascii=False),
        "index_order_exact": mixture_ids == loss_ids,
        "pair_status": "PASS" if valid else "FAIL",
    }


def parse_finite_decimal(value: str) -> tuple[Decimal | None, str]:
    if value.strip() == "":
        return None, "missing"
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation:
        return None, "invalid"
    if not parsed.is_finite():
        return None, "nonfinite"
    return parsed, "ok"


def build_training_qc(mixture: dict[str, Any], loss: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit A4/A5 values without closure, repair, deletion, or feature creation."""
    qc_rows: list[dict[str, Any]] = []
    if not mixture["schema_exact"] or not loss["schema_exact"]:
        return qc_rows, {"status": "not_run_schema_failure", "rows_audited": 0}

    mixture_header = mixture["header"]
    loss_header = loss["header"]
    mix_positions = {name: mixture_header.index(name) for name in CANONICAL_MIXTURE_COLUMNS}
    loss_positions = {name: loss_header.index(name) for name in CANONICAL_LOSS_COLUMNS}
    loss_rows_by_id: dict[str, list[list[str]]] = defaultdict(list)
    for row in loss["rows"]:
        if len(row) == len(loss_header):
            loss_rows_by_id[row[loss_positions["index"]].strip()].append(row)

    recipe_groups_raw: dict[tuple[Decimal, ...], list[int]] = defaultdict(list)
    parsed_losses: list[dict[str, Decimal] | None] = []
    total_loss_missing = total_loss_invalid = total_loss_nonfinite = 0
    for source_row, row in enumerate(mixture["rows"], start=2):
        index = row[mix_positions["index"]].strip() if len(row) == len(mixture_header) else ""
        mix_values: list[Decimal] = []
        mix_missing = mix_invalid = mix_nonfinite = mix_negative = mix_zero = 0
        if len(row) != len(mixture_header):
            mix_invalid = len(CANONICAL_MIXTURE_COLUMNS) - 1
        else:
            for name in CANONICAL_MIXTURE_COLUMNS[1:]:
                value, status = parse_finite_decimal(row[mix_positions[name]])
                if status == "missing":
                    mix_missing += 1
                elif status == "invalid":
                    mix_invalid += 1
                elif status == "nonfinite":
                    mix_nonfinite += 1
                else:
                    assert value is not None
                    mix_values.append(value)
                    mix_negative += int(value < 0)
                    mix_zero += int(value == 0)

        raw_sum = sum(mix_values, Decimal(0)) if len(mix_values) == 17 else None
        zero_sum = raw_sum == 0 if raw_sum is not None else False
        recipe_key = tuple(mix_values) if len(mix_values) == 17 else None
        if recipe_key is not None:
            recipe_groups_raw[recipe_key].append(len(qc_rows))

        loss_matches = loss_rows_by_id.get(index, [])
        parsed_loss: dict[str, Decimal] | None = None
        loss_missing = loss_invalid = loss_nonfinite = 0
        if len(loss_matches) == 1:
            loss_row = loss_matches[0]
            parsed_loss = {}
            for name in CANONICAL_LOSS_COLUMNS[1:]:
                value, status = parse_finite_decimal(loss_row[loss_positions[name]])
                if status == "missing":
                    loss_missing += 1
                elif status == "invalid":
                    loss_invalid += 1
                elif status == "nonfinite":
                    loss_nonfinite += 1
                else:
                    assert value is not None
                    parsed_loss[name] = value
        else:
            loss_invalid = len(CANONICAL_LOSS_COLUMNS) - 1
        total_loss_missing += loss_missing
        total_loss_invalid += loss_invalid
        total_loss_nonfinite += loss_nonfinite
        parsed_losses.append(parsed_loss)

        row_ok = bool(
            index
            and mix_missing == mix_invalid == mix_nonfinite == mix_negative == 0
            and not zero_sum
            and len(loss_matches) == 1
            and loss_missing == loss_invalid == loss_nonfinite == 0
        )
        qc_rows.append({
            "source_row": source_row,
            "index": index,
            "mixture_missing_count": mix_missing,
            "mixture_invalid_count": mix_invalid,
            "mixture_nonfinite_count": mix_nonfinite,
            "negative_share_count": mix_negative,
            "zero_share_count": mix_zero,
            "raw_share_sum": format(raw_sum, "f") if raw_sum is not None else "",
            "absolute_raw_sum_deviation_from_one": format(abs(raw_sum - Decimal(1)), "f") if raw_sum is not None else "",
            "zero_sum_row": zero_sum,
            "loss_match_count": len(loss_matches),
            "loss_missing_count": loss_missing,
            "loss_invalid_count": loss_invalid,
            "loss_nonfinite_count": loss_nonfinite,
            "duplicate_raw_recipe_group_size": 1,
            "duplicate_recipe_loss_disagreement": False,
            "max_loss_range_within_duplicate_recipe": "",
            "loss_target_ranges_json": "{}",
            "row_status": "PASS" if row_ok else "FAIL",
        })

    disagreement_groups = 0
    target_disagreement_counts: Counter[str] = Counter()
    duplicate_rows = 0
    duplicate_groups = 0
    for indices in recipe_groups_raw.values():
        if len(indices) <= 1:
            continue
        duplicate_groups += 1
        duplicate_rows += len(indices)
        target_ranges: dict[str, str] = {}
        for target in CANONICAL_LOSS_COLUMNS[1:]:
            values = [parsed_losses[i][target] for i in indices
                      if parsed_losses[i] is not None and target in parsed_losses[i]]
            if len(values) == len(indices):
                value_range = max(values) - min(values)
                if value_range != 0:
                    target_ranges[target] = format(value_range, "f")
                    target_disagreement_counts[target] += 1
        if target_ranges:
            disagreement_groups += 1
        max_range = max((Decimal(value) for value in target_ranges.values()), default=None)
        group_json = json.dumps(target_ranges, ensure_ascii=False, sort_keys=True)
        for row_index in indices:
            qc_rows[row_index]["duplicate_raw_recipe_group_size"] = len(indices)
            qc_rows[row_index]["duplicate_recipe_loss_disagreement"] = bool(target_ranges)
            qc_rows[row_index]["max_loss_range_within_duplicate_recipe"] = (
                format(max_range, "f") if max_range is not None else "0"
            )
            qc_rows[row_index]["loss_target_ranges_json"] = group_json

    raw_sums = [Decimal(row["raw_share_sum"]) for row in qc_rows if row["raw_share_sum"] != ""]
    summary = {
        "status": "completed",
        "rows_audited": len(qc_rows),
        "failed_rows": sum(row["row_status"] == "FAIL" for row in qc_rows),
        "mixture_missing_values": sum(row["mixture_missing_count"] for row in qc_rows),
        "mixture_invalid_values": sum(row["mixture_invalid_count"] for row in qc_rows),
        "mixture_nonfinite_values": sum(row["mixture_nonfinite_count"] for row in qc_rows),
        "negative_share_values": sum(row["negative_share_count"] for row in qc_rows),
        "zero_sum_rows": sum(bool(row["zero_sum_row"]) for row in qc_rows),
        "zero_share_values": sum(row["zero_share_count"] for row in qc_rows),
        "raw_share_sum_min": format(min(raw_sums), "f") if raw_sums else None,
        "raw_share_sum_max": format(max(raw_sums), "f") if raw_sums else None,
        "duplicate_raw_recipe_groups": duplicate_groups,
        "rows_in_duplicate_raw_recipe_groups": duplicate_rows,
        "duplicate_recipe_groups_with_loss_disagreement": disagreement_groups,
        "loss_target_disagreement_group_counts": dict(sorted(target_disagreement_counts.items())),
        "loss_missing_values": total_loss_missing,
        "loss_invalid_values": total_loss_invalid,
        "loss_nonfinite_values": total_loss_nonfinite,
        "definition_note": "Duplicate recipes use exact parsed raw 17-share vectors; no closure or pseudocount is applied.",
    }
    return qc_rows, summary


def run_audit(output_dir: Path) -> dict[str, Any]:
    """Run the isolated A4-A15 audit without importing data into any model."""
    started_at = datetime.now(timezone.utc).isoformat()
    file_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    inspections: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    issues: list[dict[str, str]] = []

    for split, mixture_id, mixture_name, loss_id, loss_name, role in AUDIT_SPLITS:
        mixture = inspect_csv_structure(DATA_ROOT / mixture_name, CANONICAL_MIXTURE_COLUMNS)
        loss = inspect_csv_structure(DATA_ROOT / loss_name, CANONICAL_LOSS_COLUMNS)
        inspections[split] = (mixture, loss)
        file_rows.append(public_file_role_row(mixture_id, split, role, "mixture_17", mixture))
        file_rows.append(public_file_role_row(loss_id, split, role, "loss_13", loss))
        pair = pair_integrity_row(split, role, mixture, loss)
        pair_rows.append(pair)
        for document_id, table_type, inspected in (
            (mixture_id, "mixture_17", mixture), (loss_id, "loss_13", loss)
        ):
            if inspected["read_error"]:
                issues.append({"severity": "ERROR", "code": "READ_FAILURE", "scope": document_id,
                               "message": inspected["read_error"]})
            elif not inspected["schema_exact"]:
                issues.append({"severity": "ERROR", "code": "SCHEMA_MISMATCH", "scope": document_id,
                               "message": f"{table_type} header differs from frozen contract"})
            if inspected["malformed_row_count"]:
                issues.append({"severity": "ERROR", "code": "MALFORMED_ROWS", "scope": document_id,
                               "message": str(inspected["malformed_row_count"])})
            if inspected["blank_index_count"] or inspected["duplicate_index_count"]:
                issues.append({"severity": "ERROR", "code": "INDEX_INVALID", "scope": document_id,
                               "message": f"blank={inspected['blank_index_count']}, duplicate={inspected['duplicate_index_count']}"})
        if pair["pair_status"] != "PASS":
            issues.append({"severity": "ERROR", "code": "PAIR_KEY_MISMATCH", "scope": split,
                           "message": "row count, key membership, uniqueness, or order differs"})

    train_qc, train_summary = build_training_qc(*inspections["train_1m"])
    if train_summary.get("failed_rows", 0):
        issues.append({"severity": "ERROR", "code": "TRAIN_VALUE_INVALID", "scope": "A4/A5",
                       "message": f"failed_rows={train_summary['failed_rows']}"})

    status = "PASS" if not any(item["severity"] == "ERROR" for item in issues) else "FAIL"
    conclusion_payload = {
        "status": status,
        "input_hashes": [(row["document_id"], row["sha256"]) for row in file_rows],
        "file_schema_status": [(row["document_id"], row["schema_exact"]) for row in file_rows],
        "pair_status": [(row["split"], row["pair_status"]) for row in pair_rows],
        "train_summary": train_summary,
        "issues": issues,
    }
    conclusion_hash = hashlib.sha256(
        json.dumps(conclusion_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    summary = {
        "status": status,
        "mode": "audit",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code": {"file": portable_path(Path(__file__)), "sha256": sha256(Path(__file__))},
        "data_root": str(DATA_ROOT.resolve()),
        "data_role_policy": {
            "A4_A5": "training; schema, keys, and full value audit",
            "A6_A11": "validation only; schema, keys, size, and hashes only",
            "A12_A15": "extrapolation/estimate only; schema, keys, size, and hashes only",
        },
        "counts": {"files": len(file_rows), "pairs": len(pair_rows), "errors": len(issues)},
        "training_value_audit": train_summary,
        "issues": issues,
        "audit_conclusion_sha256": conclusion_hash,
        "isolation_guards": {
            "model_functions_called": False,
            "quality_or_conflict_outputs_read": False,
            "a16_mapping_read": False,
            "closure_computed_or_saved": False,
            "pseudocount_or_imputation_applied": False,
            "rows_deleted_or_repaired": False,
            "external_loss_distributions_or_metrics_summarized": False,
            "pdf_or_report_text_parsed_as_instructions": False,
        },
        "outputs": ["audit_summary.json", "file_roles.csv", "pair_integrity.csv", "train_composition_qc.csv"],
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "command": f'"{sys.executable}" -B src/f_q1_3_mixture_loss.py --mode audit --data-root "{DATA_ROOT.resolve()}"',
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    file_fields = [
        "document_id", "split", "role", "table_type", "file", "exists", "size_bytes", "sha256",
        "row_count", "schema_exact", "header_order_exact", "duplicate_header_names", "missing_columns",
        "unexpected_columns", "malformed_row_count", "blank_index_count", "duplicate_index_count", "read_error",
    ]
    pair_fields = [
        "split", "role", "mixture_rows", "loss_rows", "mixture_duplicate_index_count",
        "loss_duplicate_index_count", "mixture_blank_index_count", "loss_blank_index_count",
        "missing_in_loss_count", "extra_in_loss_count", "missing_in_loss_sample", "extra_in_loss_sample",
        "index_order_exact", "pair_status",
    ]
    qc_fields = [
        "source_row", "index", "mixture_missing_count", "mixture_invalid_count", "mixture_nonfinite_count",
        "negative_share_count", "zero_share_count", "raw_share_sum", "absolute_raw_sum_deviation_from_one",
        "zero_sum_row", "loss_match_count", "loss_missing_count", "loss_invalid_count", "loss_nonfinite_count",
        "duplicate_raw_recipe_group_size", "duplicate_recipe_loss_disagreement",
        "max_loss_range_within_duplicate_recipe", "loss_target_ranges_json", "row_status",
    ]
    write_csv(output_dir / "file_roles.csv", file_rows, file_fields)
    write_csv(output_dir / "pair_integrity.csv", pair_rows, pair_fields)
    write_csv(output_dir / "train_composition_qc.csv", train_qc, qc_fields)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def data_audit_row(name: str, data: dict[str, Any], role: str) -> dict[str, Any]:
    x, p, y, sums = data["x"], data["p"], data["y"], data["row_sums"]
    groups = recipe_groups(p)
    key_matrix = np.round(p, 12)
    if len(p) > 1:
        nearest_l1 = min(
            float(np.abs(key_matrix[i] - key_matrix[j]).sum())
            for i in range(len(p)) for j in range(i + 1, len(p))
        )
    else:
        nearest_l1 = None
    return {
        "split": name,
        "role": role,
        "n_recipes": len(x),
        "n_mixture_domains": x.shape[1],
        "n_loss_targets": y.shape[1],
        "index_unique_and_aligned": True,
        "row_sum_min": float(sums.min()),
        "row_sum_max": float(sums.max()),
        "rows_abs_sum_deviation_gt_0_001": int(np.sum(np.abs(sums - 1.0) > 0.001 + 1e-12)),
        "zero_cells": int(np.sum(x == 0)),
        "rows_with_zero": int(np.sum(np.any(x == 0, axis=1))),
        "negative_share_cells": int(np.sum(x < 0)),
        "exact_recipe_groups": len(groups),
        "exact_duplicate_recipes": int(len(p) - len(groups)),
        "nearest_pair_l1_after_closure": nearest_l1,
        "loss_min": float(y.min()),
        "loss_max": float(y.max()),
    }


def write_parameter_table(path: Path, model: dict[str, Any], pcols: list[str],
                          ycols: list[str], method: str) -> None:
    rows = []
    for k, target in enumerate(ycols):
        row: dict[str, Any] = {
            "model": method,
            "target": target,
            "intercept": float(model["intercept"][k]),
            "lambda": float(model.get("lambda", 0.0)),
            "reference_domain": pcols[REFERENCE_DOMAIN_INDEX] if method == "reference_ols" else "",
        }
        for j, domain in enumerate(pcols):
            row[f"coef::{domain}"] = float(model["coef"][j, k])
        rows.append(row)
    write_csv(path, rows)


def write_quadratic_parameter_table(path: Path, model: dict[str, Any],
                                    pcols: list[str], ycols: list[str]) -> None:
    terms = quadratic_feature_names(pcols)
    rows = []
    for j, term in enumerate(terms):
        for k, target in enumerate(ycols):
            rows.append({
                "model": "quadratic_ridge_sensitivity",
                "target": target,
                "term": term,
                "coefficient": float(model["coef"][j, k]),
                "feature_center": float(model["feature_center"][j]),
                "feature_scale_rms": float(model["feature_scale"][j]),
                "intercept_target_mean": float(model["intercept"][k]),
                "lambda": float(model["lambda"]),
            })
    write_csv(path, rows)


def run_p1(output_dir: Path) -> dict[str, Any]:
    train = load_pair(SPLITS[0][1], SPLITS[0][2])
    n = min(96, len(train["p"]))
    p, y = train["p"][:n], train["y"][:n]
    raw_p = train["x"][:n]
    mean_model = fit_mean(y)
    ols_model = fit_reference_ols(p, y)
    ridge_model = fit_simplex_ridge(p, y, 0.01)
    huber_model = fit_simplex_huber_ridge(p, y, 0.01)
    quadratic_model = fit_quadratic_ridge(p, y, 0.1)
    if not np.all(huber_model["converged"]):
        raise ArithmeticError("Huber IRLS failed to converge in the P1 smoke fit")
    models = {
        "mean": mean_model, "reference_ols": ols_model,
        "simplex_ridge": ridge_model,
        "simplex_huber_ridge_sensitivity": huber_model,
        "quadratic_ridge_sensitivity": quadratic_model,
    }
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for model_name, model in models.items():
        if model_name == "mean":
            pred = predict_mean(model, p)
        elif model_name == "simplex_huber_ridge_sensitivity":
            pred = predict_huber(model, p)
        elif model_name == "quadratic_ridge_sensitivity":
            pred = predict_quadratic(model, p)
        else:
            pred = predict_linear(model, p)
        metrics.extend(calculate_metrics(y, pred, train["ycols"], "p1_train_slice", model_name, "smoke_in_sample"))
        for i in range(n):
            for k, target in enumerate(train["ycols"]):
                predictions.append({
                    "index": train["ids"][i], "model": model_name, "target": target,
                    "observed_loss": float(y[i, k]), "predicted_loss": float(pred[i, k]),
                })
    if not np.allclose(ridge_model["coef"].sum(axis=0), 0.0, atol=1e-9):
        raise ArithmeticError("P1 ridge result violates sum-to-zero coefficient constraint")
    if not np.allclose(raw_p.sum(axis=1), train["row_sums"][:n]):
        raise ArithmeticError("P1 raw row-sum audit does not match loaded data")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "p1_metrics.csv", metrics)
    write_csv(output_dir / "p1_predictions.csv", predictions)
    summary = {
        "status": "MINIMAL_RUN_COMPLETED",
        "mode": "p1",
        "source_code": {"file": portable_path(Path(__file__)),
                        "sha256": sha256(Path(__file__))},
        "input": {
            "mixture_file": portable_path(train["mixture_path"]),
            "loss_file": portable_path(train["loss_path"]),
            "mixture_sha256": sha256(train["mixture_path"]),
            "loss_sha256": sha256(train["loss_path"]),
        },
        "rows_used": n,
        "mixture_dimensions": p.shape[1],
        "loss_dimensions": y.shape[1],
        "models_run": list(models),
        "contamination_guard": {
            "data_inputs": "A4/A5 for fitting and selection; A6-A15 for their declared frozen evaluation roles; no PDF or report text is parsed",
            "method_basis": "simplex identifiability and grouped internal validation; hidden text does not decide inclusion or exclusion",
            "candidate_models": ["training-mean baseline", "reference-component OLS", "simplex ridge"],
            "robustness_model": "simplex_huber_ridge_sensitivity (smoke fit only; not a primary candidate)",
            "additional_sensitivity_model": "quadratic_ridge_sensitivity (smoke fit only; not a primary candidate)",
        },
        "ridge_lambda": 0.01,
        "ridge_constraint_max_abs_sum": float(np.max(np.abs(ridge_model["coef"].sum(axis=0)))),
        "reference_ols_rank": ols_model["rank"],
        "reference_ols_condition_number": ols_model["condition_number"],
        "reference_ols_condition_warning": ols_model["condition_warning"],
        "huber_sensitivity_diagnostics": {
            "all_targets_converged": bool(np.all(huber_model["converged"])),
            "max_iterations": int(np.max(huber_model["iterations"])),
            "minimum_weight": float(np.min(huber_model["weights"])),
            "coefficient_sum_max_abs": float(np.max(np.abs(huber_model["coef"].sum(axis=0)))),
        },
        "quadratic_sensitivity_diagnostics": {
            "feature_count": int(quadratic_model["coef"].shape[0]),
            "coefficient_matrix_finite": bool(np.isfinite(quadratic_model["coef"]).all()),
            "smoke_lambda": 0.1,
        },
        "numpy": np.__version__,
        "python": platform.python_version(),
        "command": "python src/f_q1_3_mixture_loss.py --mode p1",
        "note": "P1 slice uses real A4/A5 records; metrics are in-sample smoke diagnostics, not model validation.",
    }
    (output_dir / "p1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_full(output_dir: Path) -> dict[str, Any]:
    loaded: dict[str, dict[str, Any]] = {}
    expected_p = expected_y = None
    roles = {}
    for split_name, mixture_name, loss_name, role in SPLITS:
        loaded[split_name] = load_pair(mixture_name, loss_name, expected_p, expected_y)
        expected_p, expected_y = loaded[split_name]["pcols"], loaded[split_name]["ycols"]
        roles[split_name] = role
    train = loaded["train_1m"]
    audit_rows = [data_audit_row(name, d, roles[name]) for name, d in loaded.items()]
    p_closed, p_raw, y = train["p"], train["x"], train["y"]

    cv_by_variant = {}
    cv_metric_rows: list[dict[str, Any]] = []
    cv_prediction_rows: list[dict[str, Any]] = []
    outer_selection_rows: list[dict[str, Any]] = []
    for variant, p in (("closed", p_closed), ("raw", p_raw)):
        cv = nested_cv(p, y, SEED)
        cv_by_variant[variant] = cv
        for model_name, prediction in cv["predictions"].items():
            rows = calculate_metrics(y, prediction, train["ycols"], f"nested_cv_{variant}", model_name, "out_of_fold")
            for row in rows:
                row["variant"] = variant
                row["nested_macro_normalized_rmse"] = cv["normalized_rmse"][model_name]
                row["outer_selected_lambdas"] = json.dumps(cv["outer_selected_lambdas"])
                row["outer_selected_models"] = json.dumps(cv["outer_selected_models"])
            cv_metric_rows.extend(rows)
            for i in range(len(y)):
                for k, target in enumerate(train["ycols"]):
                    cv_prediction_rows.append({
                        "index": train["ids"][i], "variant": variant, "model": model_name,
                        "fold": int(cv["fold_id"][i]), "target": target,
                        "selected_model_for_fold": cv["outer_selected_models"][int(cv["fold_id"][i])],
                        "observed_loss": float(y[i, k]), "predicted_loss": float(prediction[i, k]),
                    })
        for fold, (family, scores, penalty) in enumerate(zip(
                cv["outer_selected_models"], cv["outer_inner_model_scores"], cv["outer_selected_lambdas"])):
            outer_selection_rows.append({
                "variant": variant,
                "outer_fold": fold,
                "selected_family_from_outer_training_only": family,
                "ridge_lambda_selected_inside_outer_training": penalty,
                "inner_cv_standardized_mse_by_family": json.dumps(scores, sort_keys=True),
            })

    # A response-robust sensitivity model is evaluated separately from primary
    # model-family selection. It cannot displace the primary model post hoc.
    huber_cv = nested_huber_cv(p_closed, y, SEED + 20000)
    huber_cv_rows = calculate_metrics(
        y, huber_cv["predictions"], train["ycols"], "nested_cv_closed",
        "simplex_huber_ridge_sensitivity", "out_of_fold_robustness_sensitivity"
    )
    for row in huber_cv_rows:
        row["variant"] = "closed"
        row["nested_macro_normalized_rmse"] = huber_cv["normalized_rmse"]
        row["outer_selected_lambdas"] = json.dumps(huber_cv["outer_selected_lambdas"])
        row["outer_selected_models"] = "simplex_huber_ridge_sensitivity"
    cv_metric_rows.extend(huber_cv_rows)
    for i in range(len(y)):
        for k, target in enumerate(train["ycols"]):
            cv_prediction_rows.append({
                "index": train["ids"][i], "variant": "closed",
                "model": "simplex_huber_ridge_sensitivity",
                "fold": int(huber_cv["fold_id"][i]), "target": target,
                "selected_model_for_fold": "simplex_huber_ridge_sensitivity",
                "observed_loss": float(y[i, k]),
                "predicted_loss": float(huber_cv["predictions"][i, k]),
            })
    for fold, penalty in enumerate(huber_cv["outer_selected_lambdas"]):
        outer_selection_rows.append({
            "variant": "closed_robustness_sensitivity",
            "outer_fold": fold,
            "selected_family_from_outer_training_only": "simplex_huber_ridge_sensitivity",
            "ridge_lambda_selected_inside_outer_training": penalty,
            "inner_cv_standardized_mse_by_family": "Huber ridge only; lambda selected inside outer training",
        })

    # Zero-friendly native-share quadratic model, assessed as a sensitivity only.
    quadratic_cv = nested_quadratic_cv(p_closed, y, SEED + 30000)
    quadratic_cv_rows = calculate_metrics(
        y, quadratic_cv["predictions"], train["ycols"], "nested_cv_closed",
        "quadratic_ridge_sensitivity", "nonlinear_sensitivity_only"
    )
    for row in quadratic_cv_rows:
        row["variant"] = "closed"
        row["nested_macro_normalized_rmse"] = quadratic_cv["normalized_rmse"]
        row["outer_selected_lambdas"] = json.dumps(quadratic_cv["outer_selected_lambdas"])
        row["outer_selected_models"] = "quadratic_ridge_sensitivity"
    cv_metric_rows.extend(quadratic_cv_rows)
    for i in range(len(y)):
        for k, target in enumerate(train["ycols"]):
            cv_prediction_rows.append({
                "index": train["ids"][i], "variant": "closed",
                "model": "quadratic_ridge_sensitivity",
                "fold": int(quadratic_cv["fold_id"][i]), "target": target,
                "selected_model_for_fold": "quadratic_ridge_sensitivity",
                "observed_loss": float(y[i, k]),
                "predicted_loss": float(quadratic_cv["predictions"][i, k]),
            })
    for fold, penalty in enumerate(quadratic_cv["outer_selected_lambdas"]):
        outer_selection_rows.append({
            "variant": "closed_nonlinear_sensitivity",
            "outer_fold": fold,
            "selected_family_from_outer_training_only": "quadratic_ridge_sensitivity",
            "ridge_lambda_selected_inside_outer_training": penalty,
            "inner_cv_standardized_mse_by_family": "quadratic ridge only; lambda selected inside outer training",
        })

    selected_model, full_candidate_scores, selected_candidate_lambda, lambda_scores = choose_model_family(
        p_closed, y, SEED + 9000, N_OUTER_FOLDS
    )
    raw_candidate_lambda, raw_lambda_scores = choose_lambda(p_raw, y, SEED + 10000, N_OUTER_FOLDS)
    huber_lambda, huber_lambda_scores = choose_huber_lambda(p_closed, y, SEED + 11000, N_OUTER_FOLDS)
    huber_model = fit_simplex_huber_ridge(p_closed, y, huber_lambda)
    if not np.all(huber_model["converged"]):
        raise ArithmeticError("Huber IRLS failed to converge on the full training data")
    quadratic_lambda, quadratic_lambda_scores = choose_quadratic_lambda(
        p_closed, y, SEED + 12000, N_OUTER_FOLDS
    )
    quadratic_model = fit_quadratic_ridge(p_closed, y, quadratic_lambda)
    final_lambda = selected_candidate_lambda if selected_model == "simplex_ridge" else None
    final_raw_lambda = raw_candidate_lambda if selected_model == "simplex_ridge" else None
    full_selection_rows = [
        {"candidate_model": name, "role": "primary_candidate",
         "grouped_cv_standardized_mse": score,
         "selected_on_full_training_cv": name == selected_model}
        for name, score in full_candidate_scores.items()
    ]
    full_selection_rows.append({
        "candidate_model": "simplex_huber_ridge_sensitivity",
        "role": "robustness_sensitivity_only",
        "grouped_cv_standardized_mse": min(row["standardized_mse"] for row in huber_lambda_scores),
        "selected_on_full_training_cv": False,
    })
    full_selection_rows.append({
        "candidate_model": "quadratic_ridge_sensitivity",
        "role": "nonlinear_sensitivity_only",
        "grouped_cv_standardized_mse": min(row["standardized_mse"] for row in quadratic_lambda_scores),
        "selected_on_full_training_cv": False,
    })

    final_models: dict[str, dict[str, Any]] = {}
    for variant, p, penalty in (("closed", p_closed, final_lambda), ("raw", p_raw, final_raw_lambda)):
        if selected_model == "mean":
            final_models[(variant, selected_model)] = fit_mean(y)
        elif selected_model == "reference_ols":
            final_models[(variant, selected_model)] = fit_reference_ols(p, y)
        else:
            final_models[(variant, selected_model)] = fit_simplex_ridge(p, y, float(penalty))

    final_model = final_models[("closed", selected_model)]
    oof_selection_predictions = cv_by_variant["closed"]["predictions"]["nested_selected"]
    residual_rows = residual_diagnostics(y, oof_selection_predictions, train["ycols"])
    overall_residual_rows = [row for row in residual_rows if row["prediction_rank_bin"] == 0]
    binned_residual_rows = [row for row in residual_rows if row["prediction_rank_bin"] > 0]
    worst_oof_target = max(overall_residual_rows, key=lambda row: row["rmse"])
    largest_binned_bias = max(binned_residual_rows,
                              key=lambda row: abs(row["mean_error_pred_minus_observed"]))
    substitution_rows, substitution_bootstrap_summary = bootstrap_substitution_effects(
        p_closed, y, final_model, selected_model, final_lambda,
        train["pcols"], train["ycols"], n_boot=500
    )
    holdout_rows: list[dict[str, Any]] = []
    holdout_prediction_rows: list[dict[str, Any]] = []
    estimate_rows: list[dict[str, Any]] = []
    for split_name in ("test_1m", "test_60m", "test_1b", "estimate_10b", "estimate_70b"):
        data = loaded[split_name]
        role = roles[split_name]
        for variant, p_eval in (("closed", data["p"]), ("raw", data["x"])):
            model = final_models[(variant, selected_model)]
            pred = predict_mean(model, p_eval) if selected_model == "mean" else predict_linear(model, p_eval)
            metrics = calculate_metrics(data["y"], pred, data["ycols"], split_name, selected_model, role)
            target_rows = []
            for row in metrics:
                row["variant"] = variant
                target_rows.append(row)
            if role == "estimate_consistency_only":
                estimate_rows.extend(target_rows)
            else:
                holdout_rows.extend(target_rows)
            for i in range(len(data["y"])):
                for k, target in enumerate(data["ycols"]):
                    holdout_prediction_rows.append({
                        "split": split_name, "role": role, "variant": variant,
                        "model": selected_model,
                        "index": data["ids"][i], "target": target,
                        "observed_or_estimated_loss": float(data["y"][i, k]),
                        "predicted_loss": float(pred[i, k]),
                        "error_pred_minus_observed_or_estimated": float(pred[i, k] - data["y"][i, k]),
                    })
        # Frozen out-of-sample robustness evaluation uses the same closed recipe
        # and is never used to alter the primary model or Huber lambda.
        huber_pred = predict_huber(huber_model, data["p"])
        huber_metrics = calculate_metrics(
            data["y"], huber_pred, data["ycols"], split_name,
            "simplex_huber_ridge_sensitivity", role
        )
        for row in huber_metrics:
            row["variant"] = "closed"
        if role == "estimate_consistency_only":
            estimate_rows.extend(huber_metrics)
        else:
            holdout_rows.extend(huber_metrics)
        for i in range(len(data["y"])):
            for k, target in enumerate(data["ycols"]):
                holdout_prediction_rows.append({
                    "split": split_name, "role": role, "variant": "closed",
                    "model": "simplex_huber_ridge_sensitivity",
                    "index": data["ids"][i], "target": target,
                    "observed_or_estimated_loss": float(data["y"][i, k]),
                    "predicted_loss": float(huber_pred[i, k]),
                    "error_pred_minus_observed_or_estimated": float(huber_pred[i, k] - data["y"][i, k]),
                })
        quadratic_pred = predict_quadratic(quadratic_model, data["p"])
        quadratic_metrics = calculate_metrics(
            data["y"], quadratic_pred, data["ycols"], split_name,
            "quadratic_ridge_sensitivity", role
        )
        for row in quadratic_metrics:
            row["variant"] = "closed"
        if role == "estimate_consistency_only":
            estimate_rows.extend(quadratic_metrics)
        else:
            holdout_rows.extend(quadratic_metrics)
        for i in range(len(data["y"])):
            for k, target in enumerate(data["ycols"]):
                holdout_prediction_rows.append({
                    "split": split_name, "role": role, "variant": "closed",
                    "model": "quadratic_ridge_sensitivity",
                    "index": data["ids"][i], "target": target,
                    "observed_or_estimated_loss": float(data["y"][i, k]),
                    "predicted_loss": float(quadratic_pred[i, k]),
                    "error_pred_minus_observed_or_estimated": float(quadratic_pred[i, k] - data["y"][i, k]),
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "data_audit.csv", audit_rows)
    write_csv(output_dir / "nested_cv_metrics.csv", cv_metric_rows)
    write_csv(output_dir / "nested_cv_predictions.csv", cv_prediction_rows)
    write_csv(output_dir / "nested_model_selection.csv", outer_selection_rows)
    write_csv(output_dir / "full_training_model_selection.csv", full_selection_rows)
    write_csv(output_dir / "residual_diagnostics_nested_selection.csv", residual_rows)
    write_csv(output_dir / "substitution_effects_10pp_bootstrap.csv", substitution_rows,
              fieldnames=(list(substitution_rows[0]) if substitution_rows else [
                  "model", "target", "from_domain", "to_domain", "share_transfer",
                  "effect_per_10pp_transfer", "bootstrap_ci95_lower", "bootstrap_ci95_upper",
                  "bootstrap_probability_effect_positive",
                  "training_recipes_with_from_share_at_least_10pp", "training_support_fraction",
                  "interpretation",
              ]))
    write_csv(output_dir / "holdout_metrics.csv", holdout_rows)
    write_csv(output_dir / "estimate_consistency_metrics.csv", estimate_rows)
    write_csv(output_dir / "holdout_predictions.csv", holdout_prediction_rows)
    write_csv(output_dir / "lambda_selection_closed.csv", lambda_scores)
    write_csv(output_dir / "lambda_selection_raw_sensitivity.csv", raw_lambda_scores)
    write_csv(output_dir / "lambda_selection_huber_sensitivity.csv", huber_lambda_scores)
    write_csv(output_dir / "lambda_selection_quadratic_sensitivity.csv", quadratic_lambda_scores)
    huber_weight_rows = []
    for i, index in enumerate(train["ids"]):
        for k, target in enumerate(train["ycols"]):
            huber_weight_rows.append({
                "index": index, "target": target,
                "irls_weight": float(huber_model["weights"][i, k]),
                "downweighted": bool(huber_model["weights"][i, k] < 1.0 - 1e-12),
            })
    write_csv(output_dir / "huber_training_residual_weights.csv", huber_weight_rows)
    write_parameter_table(output_dir / "parameters_selected_closed.csv", final_model,
                          train["pcols"], train["ycols"], selected_model)
    write_parameter_table(output_dir / "parameters_huber_sensitivity.csv", huber_model,
                          train["pcols"], train["ycols"], "simplex_huber_ridge_sensitivity")
    write_quadratic_parameter_table(
        output_dir / "parameters_quadratic_sensitivity.csv", quadratic_model,
        train["pcols"], train["ycols"]
    )
    if selected_model != "mean":
        raw_selected = final_models[("raw", selected_model)]
        write_parameter_table(output_dir / "parameters_selected_raw_sensitivity.csv", raw_selected,
                              train["pcols"], train["ycols"], selected_model)

    summary = {
        "status": "MODEL_FIT_AND_FROZEN_EVALUATION_COMPLETED",
        "mode": "full",
        "source_code": {"file": portable_path(Path(__file__)),
                        "sha256": sha256(Path(__file__))},
        "selected_model_by_full_training_cv_on_closed_p": selected_model,
        "full_training_candidate_standardized_mse_on_closed_p": full_candidate_scores,
        "huber_robustness_sensitivity": {
            "status": "completed",
            "selection_role": "prespecified robustness sensitivity; not part of primary family selection",
            "outer_nested_macro_normalized_rmse": huber_cv["normalized_rmse"],
            "outer_selected_lambdas": huber_cv["outer_selected_lambdas"],
            "full_training_grouped_cv_lambda": huber_lambda,
            "full_training_grouped_cv_standardized_mse": min(
                row["standardized_mse"] for row in huber_lambda_scores
            ),
            "delta_standardized_residual": float(huber_model["delta"]),
            "irls_max_iterations": int(np.max(huber_model["iterations"])),
            "irls_all_targets_converged": bool(np.all(huber_model["converged"])),
            "per_target_downweight_counts": {
                target: {
                    "n_weight_below_1": int(np.sum(huber_model["weights"][:, k] < 1.0 - 1e-12)),
                    "n_weight_below_0_5": int(np.sum(huber_model["weights"][:, k] < 0.5)),
                    "minimum_weight": float(np.min(huber_model["weights"][:, k])),
                }
                for k, target in enumerate(train["ycols"])
            },
            "interpretation_limit": "Huber weights indicate sensitivity to large conditional residuals; they do not prove data corruption and do not protect against high-leverage composition errors.",
        },
        "quadratic_zero_friendly_sensitivity": {
            "status": "completed",
            "selection_role": "prespecified nonlinear sensitivity; not part of primary family selection",
            "feature_count": int(quadratic_model["coef"].shape[0]),
            "outer_nested_macro_normalized_rmse": quadratic_cv["normalized_rmse"],
            "outer_selected_lambdas": quadratic_cv["outer_selected_lambdas"],
            "full_training_grouped_cv_lambda": quadratic_lambda,
            "full_training_grouped_cv_standardized_mse": min(
                row["standardized_mse"] for row in quadratic_lambda_scores
            ),
            "coefficient_matrix_finite": bool(np.isfinite(quadratic_model["coef"]).all()),
            "intercept_finite": bool(np.isfinite(quadratic_model["intercept"]).all()),
            "zero_preservation": "Uses native closed shares and pairwise products; zero shares stay zero when the raw feature map is formed.",
            "raw_feature_map_zero_handling": "No log or pseudocount is applied; when a share is zero its raw main effect and all raw pairwise products involving that share are zero. Fold centering is applied only for regression fitting.",
            "interpretation_limit": "153 features are estimated from 512 recipes; use only as a sensitivity comparison. It is not used to select the primary model and gives no assurance of cross-scale extrapolation.",
        },
        "outer_nested_selection_pipeline_macro_normalized_rmse_closed_p": cv_by_variant["closed"]["normalized_rmse"]["nested_selected"],
        "outer_nested_selection_pipeline_macro_normalized_rmse_raw_p": cv_by_variant["raw"]["normalized_rmse"]["nested_selected"],
        "outer_selected_model_families_closed_p": cv_by_variant["closed"]["outer_selected_models"],
        "outer_selected_model_families_raw_p": cv_by_variant["raw"]["outer_selected_models"],
        "substitution_effect_bootstrap": substitution_bootstrap_summary,
        "residual_diagnostics": {
            "prediction_source": "outer-fold predictions from nested model-family selection on closed p",
            "prediction_rank_bins": 5,
            "out_of_fold_rows": int(len(y)),
            "largest_target_rmse": {
                "target": worst_oof_target["target"],
                "rmse": worst_oof_target["rmse"],
            },
            "largest_absolute_binned_bias": {
                "target": largest_binned_bias["target"],
                "prediction_rank_bin": largest_binned_bias["prediction_rank_bin"],
                "mean_error_pred_minus_observed": largest_binned_bias["mean_error_pred_minus_observed"],
            },
        },
        "contamination_guard": {
            "data_inputs": "A4/A5 only for fitting and model selection; A6-A11 read only for frozen evaluation; A12-A15 read only for estimate consistency; no PDF or report text is parsed as instructions",
            "method_basis": "simplex identifiability, input hash and row alignment checks, raw-vs-closed composition sensitivity, Huber residual sensitivity, and native-share quadratic sensitivity; hidden text does not decide inclusion or exclusion",
            "candidate_models": ["training-mean baseline", "reference-component OLS", "simplex ridge"],
            "robustness_model": "simplex_huber_ridge_sensitivity",
            "additional_sensitivity_model": "quadratic_ridge_sensitivity",
            "no_rows_removed_or_repaired": True,
            "report_row_sum_count_gt_0_001_after_reconciliation": 47,
            "historical_report_row_sum_count_gt_0_001": 189,
            "recomputed_train_row_sum_count_gt_0_001": int(audit_rows[0]["rows_abs_sum_deviation_gt_0_001"]),
            "row_sum_discrepancy_note": "An earlier version of 题目分析报告.md recorded 189; the report now notes the direct strict-Decimal recount of the current A4 source is 47. The hash-bound code audit also computes 47.",
        },
        "reference_ols_condition_numbers_nested_cv": cv_by_variant["closed"]["reference_ols_condition_numbers"],
        "ols_condition_warning_threshold": OLS_CONDITION_WARNING_THRESHOLD,
        "closed_fit_lambda": final_lambda,
        "huber_sensitivity_lambda": huber_lambda,
        "closed_candidate_ridge_lambda": selected_candidate_lambda,
        "raw_sensitivity_lambda": final_raw_lambda,
        "raw_candidate_ridge_lambda": raw_candidate_lambda,
        "seed": SEED,
        "outer_folds": N_OUTER_FOLDS,
        "inner_folds": N_INNER_FOLDS,
        "lambda_grid": LAMBDA_GRID.tolist(),
        "reference_domain": train["pcols"][REFERENCE_DOMAIN_INDEX],
        "inputs": [
            {"file": portable_path(data[key]), "sha256": sha256(data[key])}
            for data in loaded.values() for key in ("mixture_path", "loss_path")
        ],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "command": "python src/f_q1_3_mixture_loss.py --mode full",
        "validation_policy": {
            "A6_A7": "frozen same-scale validation; prior review notes these were previously viewed",
            "A8_A11": "frozen cross-scale transfer evaluation; no recalibration",
            "A12_A15": "estimate-table consistency only, never interpreted as real test accuracy",
        },
        "limitations": [
            "Cross-scale absolute errors are descriptive because fitting uses only 1M training data.",
            "Regression associations are not causal effects.",
            "Raw proportions are reported only as a closure sensitivity; closed proportions define primary selection.",
            "The model uses gamma=0 (no shared group sparsity); this is the planned ridge fallback.",
            "The Huber sensitivity only reduces the influence of large conditional Loss residuals; it does not establish malicious corruption or handle high-leverage mixture contamination.",
            "The quadratic native-share sensitivity uses 153 terms for 512 recipes; it is reported only as a nonlinear sensitivity and is not used to select the primary model. Its zero-friendly basis does not identify whether observed zeros are structural or rounded.",
            "An earlier analysis report version recorded 189 A4 row-sum deviations above 0.001; direct strict-Decimal recount of the current hash-bound A4 source gives 47 and the report now records 47 with the historical value noted.",
        ],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "q1_3_run_report.md", summary, aggregate_metrics(cv_metric_rows),
                 aggregate_metrics(holdout_rows), aggregate_metrics(estimate_rows), audit_rows)
    return summary


def write_report(path: Path, summary: dict[str, Any], cv_summary: list[dict[str, Any]],
                 holdout_summary: list[dict[str, Any]], estimate_summary: list[dict[str, Any]],
                 audit_rows: list[dict[str, Any]]) -> None:
    def table(rows: list[dict[str, Any]], keys: list[str]) -> list[str]:
        lines = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
        for row in rows:
            vals = ["" if row.get(k) is None else str(row.get(k)) for k in keys]
            lines.append("| " + " | ".join(vals) + " |")
        return lines

    lines = [
        "# Q1.3 p-only 配比—Loss：本轮运行摘要", "",
        "> 该结果由脚本对项目 A4/A5 及冻结评估文件实际计算。它是建模进展稿，不等于 Q1.3 全部敏感性、制图和最终验收完成。", "",
        "## 数据审计", "",
    ]
    lines += table(audit_rows, ["split", "role", "n_recipes", "n_mixture_domains", "n_loss_targets",
                                "row_sum_min", "row_sum_max", "rows_abs_sum_deviation_gt_0_001",
                                "zero_cells", "rows_with_zero", "exact_duplicate_recipes", "nearest_pair_l1_after_closure"])
    lines += ["", "## 训练内嵌套交叉验证", "",
              f"在完整 A4/A5 上按分组 CV 选出的最终模型族：`{summary['selected_model_by_full_training_cv_on_closed_p']}`。",
              f"外层分组 CV 对‘仅在各外层训练折内重新选择模型族’的整套流程给出宏平均标准化 RMSE：`{summary['outer_nested_selection_pipeline_macro_normalized_rmse_closed_p']}`。",
              "候选模型单独列出的外层分数用于描述；最终家族选择不使用这些外层验证标签，而在完整 A4/A5 的内层 CV 上完成。A6–A11 和 A12–A15 未用于调参或选模。", ""]
    lines += table(cv_summary, ["split", "role", "model", "macro_mae", "macro_rmse", "macro_r2",
                                "macro_spearman_rho", "nested_macro_normalized_rmse", "variant",
                                "worst_rmse_target", "worst_rmse"])
    lines += ["", "## 冻结的真实验证集", "",
              "以下 A6–A11 结果仅评估已在 A4/A5 内部选定并冻结的模型；跨尺度不作事后校准。闭合配比为主结果，原始配比为闭合敏感性。", ""]
    lines += table(holdout_summary, ["split", "role", "model", "macro_mae", "macro_rmse", "macro_r2",
                                     "macro_spearman_rho", "variant", "worst_rmse_target", "worst_rmse"])
    lines += ["", "## 估计表一致性检查", "",
              "A12–A15 对应的 10B/70B 表为估计结果。其误差只称为相对估计表的偏差，不解释为真实测试精度。", ""]
    lines += table(estimate_summary, ["split", "role", "model", "macro_mae", "macro_rmse", "macro_r2",
                                       "macro_spearman_rho", "variant", "worst_rmse_target", "worst_rmse"])
    boot = summary["substitution_effect_bootstrap"]
    lines += ["", "## OOF 残差与替代效应不确定性", "",
              "残差诊断基于外层折中、仅由各折训练数据选择模型族后产生的 OOF 预测；分目标总体及预测值五分位诊断见 `residual_diagnostics_nested_selection.csv`。"]
    residual = summary["residual_diagnostics"]
    lines.append(
        f"最大 OOF 逐目标 RMSE 为 `{residual['largest_target_rmse']['target']}`：{residual['largest_target_rmse']['rmse']:.4f}；最大预测五分位平均偏差出现在 `{residual['largest_absolute_binned_bias']['target']}` 第 {residual['largest_absolute_binned_bias']['prediction_rank_bin']} 组，预测减观测为 {residual['largest_absolute_binned_bias']['mean_error_pred_minus_observed']:.4f}。"
    )
    if boot["status"] == "completed":
        lines.append(
            f"配比替代效应使用 {boot['valid']}/{boot['requested']} 次有效 pairs bootstrap；95% percentile 区间为逐点区间、未作多重比较调整，并以最终模型族和正则强度固定为条件。136 个域对中 {boot['supported_10pp_pairs']} 个至少有一条配方可支持10个百分点替代，{boot['unsupported_10pp_pairs']} 个没有该份额支持；受支持的 {boot['supported_pair_target_rows']} 个目标×域对区间中，{boot['supported_intervals_entirely_positive']} 个全为正、{boot['supported_intervals_entirely_negative']} 个全为负、{boot['supported_intervals_cross_zero']} 个跨零。每项支持量见 `substitution_effects_10pp_bootstrap.csv`，这些区间不作显著性结论。"
        )
    else:
        lines.append(f"替代效应 bootstrap 状态：`{boot['status']}`；适用边界见 run_summary。")
    lines += ["", "## 选择与限制", "",
              f"- 完整 A4/A5 分组CV选择模型：`{summary['selected_model_by_full_training_cv_on_closed_p']}`；外层模型选择流程标准化 RMSE：`{summary['outer_nested_selection_pipeline_macro_normalized_rmse_closed_p']}`。",
              f"- 闭合配比最终正则强度：`{summary['closed_fit_lambda']}`；原始配比敏感性正则强度：`{summary['raw_sensitivity_lambda']}`。",
              "- 主候选包含训练均值空模型、固定参考配比系数为 0 的无正则线性对照，以及原单纯形 ridge；Huber ridge 单列为污染敏感性，不参与主模型选族。两种单纯形回归均对每个目标约束17个配比系数和为0。线性系数按配比替代效应解释。",
              f"- 参考分量线性对照的固定参考域为 `{summary['reference_domain']}`；训练设计矩阵条件数超过 {OLS_CONDITION_WARNING_THRESHOLD:.0e} 时标记病态。",
              "- 算法依据为闭合单纯形的可识别参数化和 A4/A5 内部验证；隐藏文本建议既不授权也不否决任何方法。",
              "- 超参数与候选模型选择仅基于 A4/A5；跨尺度结果不能当作训练尺度校准数据。",
              f"- Huber 稳健敏感性 nested-CV 宏平均标准化 RMSE：`{summary['huber_robustness_sensitivity']['outer_nested_macro_normalized_rmse']}`；完整训练集 CV 选得 λ=`{summary['huber_sensitivity_lambda']}`。它降低大条件残差的影响，不证明数据被污染，也不能抵抗高杠杆配比错误；权重明细见 `huber_training_residual_weights.csv`。",
              f"- 零友好的二次配比敏感性 nested-CV 宏平均标准化 RMSE：`{summary['quadratic_zero_friendly_sensitivity']['outer_nested_macro_normalized_rmse']}`；A4/A5 分组 CV 选得 λ=`{summary['quadratic_zero_friendly_sensitivity']['full_training_grouped_cv_lambda']}`，153 个原配比/两两乘积项仅作为敏感性比较，不参与主模型选族。逐项系数见 `parameters_quadratic_sensitivity.csv`。",
              f"- 行和审计更正：分析报告早期版本记录 189 行偏差超过 0.001；已按当前 A4 原表十进制逐行重算更正为 47 行，哈希绑定脚本复算为 `{summary['contamination_guard']['recomputed_train_row_sum_count_gt_0_001']}` 行。旧值留作历史差异记录。",
              "- 当前未运行 ILR：零配比来源尚未确认。二次面在原配比空间计算，对数变换没有必要；它不区分结构零和舍入零。正式图表和最终 P2 编程终检仍待完成。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit or model Q1.3 p-only mixture-to-Loss data")
    parser.add_argument("--mode", choices=("audit", "p1", "full"), default="p1",
                        help="audit is read-only/no-fit; p1 is a minimal fit; full runs nested CV and frozen evaluations")
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Directory containing the twelve A4-A15 regmix CSV files",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    global DATA_ROOT
    DATA_ROOT = args.data_root if args.data_root.is_absolute() else PROJECT_ROOT / args.data_root
    if args.output_dir is None:
        if args.mode == "audit":
            out = (PROJECT_ROOT / "Q1" / "归档_20260925" / "F题_Q1_当前编程任务"
                   / "results" / "q1_3" / "audit")
        else:
            subdir = {"p1": "p1", "full": "v1"}[args.mode]
            out = PROJECT_ROOT / "results" / "q1_3" / subdir
    else:
        out = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    if args.mode == "audit":
        summary = run_audit(out)
    elif args.mode == "p1":
        summary = run_p1(out)
    else:
        summary = run_full(out)
    print(json.dumps({"status": summary["status"], "output_dir": str(out),
                      "mode": args.mode,
                      "selected_model": summary.get("selected_model_by_full_training_cv_on_closed_p")}, ensure_ascii=False))
    return 0 if summary["status"] not in {"FAIL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
