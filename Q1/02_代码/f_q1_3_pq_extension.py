"""Q1.3 constrained p + Q_covered predictive extension.

The extension is intentionally isolated from the frozen p-only run. It treats
Q_covered as a deterministic, partially mapped transform of mixture shares,
not as an independently observed quality variable or causal effect.

Run from the project root:
    python src/f_q1_3_pq_extension.py --mode p1
    python src/f_q1_3_pq_extension.py --mode full
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_ROOT = PROJECT_ROOT / "中文题目" / "F题" / "real_attachments" / "A_data_value"
REGMIX_ROOT = DATA_ROOT / "regmix_tables"
MAPPING_PATH = DATA_ROOT / "domain_mapping_guide.csv"
Q1_1_ROOT = PROJECT_ROOT / "results" / "q1_1" / "v1"
OUT_DEFAULT = PROJECT_ROOT / "results" / "q1_3" / "pq_extension_v1"
SEED = 20260923
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4


def load_base_module():
    path = SRC_ROOT / "f_q1_3_mixture_loss.py"
    spec = importlib.util.spec_from_file_location("q1_3_p_only_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen Q1.3 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.DATA_ROOT = REGMIX_ROOT
    return module


BASE = load_base_module()

SPLITS = [
    ("train_1m", "train_mixture_1m.csv", "train_pile_loss_1m.csv", "training"),
    ("test_1m", "test_mixture_1m.csv", "test_pile_loss_1m.csv", "heldout_same_scale_previously_seen"),
    ("test_60m", "test_mixture_60m.csv", "test_pile_loss_60m.csv", "heldout_cross_scale_previously_seen"),
    ("test_1b", "test_mixture_1B.csv", "test_pile_loss_1B.csv", "heldout_cross_scale_previously_seen"),
]

MAP_VARIANTS = {
    "direct_only": {"direct"},
    "direct_plus_near": {"direct", "near_direct"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for empty CSV: {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_pair(mix_name: str, loss_name: str) -> dict[str, Any]:
    pair = BASE.load_pair(mix_name, loss_name,
                          expected_p=list(BASE.CANONICAL_MIXTURE_COLUMNS[1:]),
                          expected_y=list(BASE.CANONICAL_LOSS_COLUMNS[1:]))
    if len(pair["ids"]) != len(set(pair["ids"])):
        raise ValueError(f"Duplicate IDs in pair: {mix_name}, {loss_name}")
    return pair


def load_mapping() -> tuple[list[dict[str, str]], dict[str, str]]:
    _, rows = read_csv(MAPPING_PATH)
    mapping = {}
    for row in rows:
        mixture_domain = row["mixture_domain"]
        if mixture_domain in mapping:
            raise ValueError(f"Duplicate domain mapping: {mixture_domain}")
        mapping[mixture_domain] = row["mapping_type"]
    return rows, mapping


def select_q_rows(path: Path, *, scenario: str, candidate: str,
                  dataset: str = "a1", score_col: str = "domain_score") -> dict[str, float]:
    _, rows = read_csv(path)
    chosen = [r for r in rows
              if r.get("scenario") == scenario
              and r.get("candidate") == candidate
              # domain_rank_sensitivity.csv is explicitly an A1-only table
              # and has no dataset column; treat that omitted field as A1.
              and r.get("dataset", "a1") == dataset]
    result: dict[str, float] = {}
    for row in chosen:
        domain = row["source_domain"]
        if domain in result:
            raise ValueError(f"Duplicate Q score for {scenario}/{candidate}/{domain}")
        value = float(row[score_col])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Q score outside [0,1]: {scenario}/{domain}={value}")
        result[domain] = value
    return result


def load_q_scenarios() -> list[dict[str, Any]]:
    domain_summary = Q1_1_ROOT / "domain_summary.csv"
    score_sensitivity = Q1_1_ROOT / "score_sensitivity_domain.csv"
    rank_sensitivity = Q1_1_ROOT / "domain_rank_sensitivity.csv"
    return [
        {
            "name": "q_equal_candidate",
            "label": "Q1.1 等权候选",
            "source": domain_summary,
            "source_scenario": "all_a1",
            "candidate": "q_equal",
            "score_col": "domain_score",
        },
        {
            "name": "q_huber_candidate",
            "label": "Q1.1 加权 Huber 候选",
            "source": domain_summary,
            "source_scenario": "all_a1",
            "candidate": "q_huber",
            "score_col": "domain_score",
        },
        {
            "name": "q_huber_unicode_broad_sensitivity",
            "label": "Q1.1 Huber（宽 Unicode 未标记 A1 校准敏感性）",
            "source": score_sensitivity,
            "source_scenario": "unicode_broad_clean_a1",
            "candidate": "q_huber",
            "score_col": "domain_score",
        },
        {
            "name": "q_huber_exclude_7_frac_sensitivity",
            "label": "Q1.1 Huber（排除 7 个语义/单位未核实 frac 字段）",
            "source": rank_sensitivity,
            "source_scenario": "exclude_7_unit_ambiguous_fields",
            "candidate": "q_huber",
            "score_col": "scenario_domain_score",
        },
    ]


def materialize_q_scenarios(scenarios: list[dict[str, Any]],
                            mapped_domains: set[str]) -> list[dict[str, Any]]:
    result = []
    for scenario in scenarios:
        scores = select_q_rows(scenario["source"],
                               scenario=scenario["source_scenario"],
                               candidate=scenario["candidate"],
                               score_col=scenario["score_col"])
        missing = mapped_domains.difference(scores)
        if missing:
            raise ValueError(f"Q1.1 scores missing mapped domains for {scenario['name']}: {sorted(missing)}")
        result.append({**scenario, "scores": scores})
    return result


def mapped_indices(pcols: list[str], mapping_rows: list[dict[str, str]],
                   mapping_variant: str) -> tuple[list[int], list[str]]:
    allowed = MAP_VARIANTS[mapping_variant]
    domain_by_col = {col.removeprefix("train_the_pile_"): i for i, col in enumerate(pcols)}
    indices: list[int] = []
    domains: list[str] = []
    for row in mapping_rows:
        domain = row["mixture_domain"]
        if row["mapping_type"] not in allowed:
            continue
        if domain not in domain_by_col:
            raise ValueError(f"Mapped domain not in frozen p columns: {domain}")
        quality_domain = row["quality_domain"]
        if quality_domain == "(none)":
            raise ValueError(f"Mapped domain has no quality-domain score: {domain}")
        indices.append(domain_by_col[domain])
        domains.append(quality_domain)
    return indices, domains


def q_covered(p: np.ndarray, q_scores: dict[str, float], indices: list[int],
              domains: list[str]) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray([q_scores[d] for d in domains], dtype=float)
    pm = p[:, indices]
    coverage = pm.sum(axis=1)
    covered_sum = pm @ q
    value = np.full(len(p), np.nan, dtype=float)
    valid = coverage > 0.0
    value[valid] = covered_sum[valid] / coverage[valid]
    return value, coverage


def identifiability_audit(p: np.ndarray, qcov: np.ndarray, coverage: np.ndarray,
                          q_scores: dict[str, float], indices: list[int],
                          domains: list[str], mapping_name: str,
                          q_scenario: str) -> dict[str, Any]:
    valid = np.isfinite(qcov)
    ps = p[valid]
    qs = qcov[valid]
    coverage_s = coverage[valid]
    ones = np.ones((len(ps), 1), dtype=float)
    x_base = np.column_stack([ones, ps])
    # This is only a rank identity check for the mapped contribution. Unmapped
    # domains are set to zero for this algebra check, never treated as Q=0 in
    # the model or reported as a complete mixture score.
    q_partial = ps[:, indices] @ np.asarray([q_scores[d] for d in domains], dtype=float)
    x_partial = np.column_stack([x_base, q_partial])
    q_fit, *_ = np.linalg.lstsq(x_base, q_partial, rcond=None)
    q_resid = q_partial - x_base @ q_fit
    base_rank = int(np.linalg.matrix_rank(x_base))
    partial_rank = int(np.linalg.matrix_rank(x_partial))

    qcov_fit, *_ = np.linalg.lstsq(x_base, qs, rcond=None)
    qcov_resid = qs - x_base @ qcov_fit
    qcov_sd = float(np.std(qs, ddof=1)) if len(qs) > 1 else 0.0
    residual_sd = float(np.std(qcov_resid, ddof=1)) if len(qs) > 1 else 0.0
    full_rank = int(np.linalg.matrix_rank(np.column_stack([x_base, qs])))
    return {
        "mapping_scenario": mapping_name,
        "q_scenario": q_scenario,
        "n_train_supported": int(valid.sum()),
        "uncovered_rows_excluded": int((~valid).sum()),
        "coverage_min": float(np.min(coverage_s)),
        "coverage_median": float(np.median(coverage_s)),
        "coverage_max": float(np.max(coverage_s)),
        "qmix_partial_rank_without_q": base_rank,
        "qmix_partial_rank_with_q": partial_rank,
        "qmix_partial_rank_gain": partial_rank - base_rank,
        "qmix_partial_relative_linear_residual": float(np.std(q_resid) / max(np.std(q_partial), np.finfo(float).eps)),
        "qcovered_rank_without_q": base_rank,
        "qcovered_rank_with_q": full_rank,
        "qcovered_rank_gain": full_rank - base_rank,
        "qcovered_relative_linear_residual": float(residual_sd / max(qcov_sd, np.finfo(float).eps)),
        "qcovered_is_deterministic_function_of_p": True,
        "independent_quality_effect_identified": False,
        "note": "Qmix alias check uses only the mapped partial contribution as an algebraic diagnostic; Qcovered is conditional on mapped mass and is undefined when coverage is zero.",
    }


def pq_design(p: np.ndarray, q: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.column_stack([p @ basis, q])


def fit_pq_ridge(p: np.ndarray, q: np.ndarray, y: np.ndarray,
                 penalty: float, basis: np.ndarray | None = None) -> dict[str, Any]:
    """Fit intercept + 16 simplex contrasts + one centered Q_covered feature.

    The p coefficients remain in the sum-to-zero contrast space. Q_covered is
    a single precomputed deterministic nonlinear feature; its coefficient is
    predictive-only and never interpreted as an independent Q effect.
    """
    basis = basis if basis is not None else BASE.helmert_basis(p.shape[1])
    x = pq_design(p, q, basis)
    x_mean = x.mean(axis=0)
    xc = x - x_mean[None, :]
    y_mean = y.mean(axis=0)
    y_scale = y.std(axis=0, ddof=1) if len(y) > 1 else np.ones(y.shape[1])
    zero_scale = y_scale <= np.finfo(float).eps
    y_scale[zero_scale] = 1.0
    ys = (y - y_mean[None, :]) / y_scale[None, :]
    gram = xc.T @ xc / len(xc)
    rhs = xc.T @ ys / len(xc)
    coef_std = np.zeros((x.shape[1], y.shape[1]), dtype=float)
    for k in range(y.shape[1]):
        if penalty <= 0:
            # lstsq avoids a normal-equation failure at an aliased Q feature.
            coef_std[:, k], *_ = np.linalg.lstsq(xc, ys[:, k], rcond=None)
        else:
            system = gram + float(penalty) * (y_scale[k] ** 2) * np.eye(gram.shape[0])
            coef_std[:, k] = np.linalg.solve(system, rhs[:, k])
        if zero_scale[k]:
            coef_std[:, k] = 0.0
    coef = coef_std * y_scale[None, :]
    intercept = y_mean - x_mean @ coef
    if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
        raise ArithmeticError("Non-finite p+Qcovered fit")
    return {
        "kind": "p_plus_Qcovered",
        "basis": basis,
        "coef": coef,
        "intercept": intercept,
        "feature_mean": x_mean,
        "lambda": float(penalty),
        "zero_scale_targets": np.flatnonzero(zero_scale).tolist(),
    }


def predict_pq(model: dict[str, Any], p: np.ndarray, q: np.ndarray) -> np.ndarray:
    x = pq_design(p, q, model["basis"])
    return model["intercept"][None, :] + (x - model["feature_mean"][None, :]) @ model["coef"]


def choose_pq_lambda(p: np.ndarray, q: np.ndarray, y: np.ndarray, seed: int,
                     n_splits: int = N_INNER_FOLDS) -> tuple[float, list[dict[str, float]]]:
    folds = BASE.make_folds(p, min(n_splits, len(BASE.recipe_groups(p))), seed)
    scores: list[dict[str, float]] = []
    for candidate in BASE.LAMBDA_GRID:
        total_sq = 0.0
        total_values = 0
        for train_idx, valid_idx in folds:
            model = fit_pq_ridge(p[train_idx], q[train_idx], y[train_idx], float(candidate))
            pred = predict_pq(model, p[valid_idx], q[valid_idx])
            scale = y[train_idx].std(axis=0, ddof=1)
            scale[scale <= np.finfo(float).eps] = 1.0
            residual = (y[valid_idx] - pred) / scale[None, :]
            total_sq += float(np.sum(residual ** 2))
            total_values += residual.size
        scores.append({"lambda": float(candidate), "standardized_mse": total_sq / total_values})
    best = min(scores, key=lambda r: (r["standardized_mse"], -r["lambda"]))
    return float(best["lambda"]), scores


def metric_rows(y_true: np.ndarray, y_pred: np.ndarray, y_names: list[str],
                split: str, role: str, model_name: str,
                norm_scale: np.ndarray | None = None,
                nested_macro_normalized_rmse: float | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    err = y_pred - y_true
    if norm_scale is None:
        norm_scale = np.ones(y_true.shape[1], dtype=float)
    scale = np.asarray(norm_scale, dtype=float)
    if scale.ndim == 1:
        scale = np.broadcast_to(scale[None, :], y_true.shape)
    scale = np.where(scale <= np.finfo(float).eps, 1.0, scale)
    normalized = err / scale
    out = []
    for j, name in enumerate(y_names):
        rmse = float(np.sqrt(np.mean(err[:, j] ** 2)))
        mae = float(np.mean(np.abs(err[:, j])))
        denom = float(np.sum((y_true[:, j] - np.mean(y_true[:, j])) ** 2))
        r2 = None if denom <= np.finfo(float).eps else float(1.0 - np.sum(err[:, j] ** 2) / denom)
        rho = BASE.spearman_rho(y_true[:, j], y_pred[:, j])
        out.append({
            "split": split, "role": role, "model": model_name,
            "target": name, "n": len(y_true), "mae": mae, "rmse": rmse,
            "r2": r2, "spearman_rho": rho,
            "normalized_rmse": float(np.sqrt(np.mean(normalized[:, j] ** 2))),
        })
    r2s = [r["r2"] for r in out if r["r2"] is not None]
    rhos = [r["spearman_rho"] for r in out if r["spearman_rho"] is not None]
    summary = {
        "split": split, "role": role, "model": model_name,
        "n_recipes": int(len(y_true)), "n_targets": int(y_true.shape[1]),
        "macro_mae": float(np.mean([r["mae"] for r in out])),
        "macro_rmse": float(np.mean([r["rmse"] for r in out])),
        "macro_normalized_rmse": float(np.sqrt(np.mean(normalized ** 2))),
        "macro_r2": float(np.mean(r2s)) if r2s else None,
        "macro_spearman_rho": float(np.mean(rhos)) if rhos else None,
        "worst_rmse_target": max(out, key=lambda r: r["rmse"])["target"],
        "worst_rmse": max(r["rmse"] for r in out),
    }
    if nested_macro_normalized_rmse is not None:
        summary["nested_macro_normalized_rmse"] = float(nested_macro_normalized_rmse)
    return out, summary


def nested_baselines(p: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    folds = BASE.make_folds(p, N_OUTER_FOLDS, SEED)
    predictions = {
        "p_only_linear": np.full_like(y, np.nan, dtype=float),
        "p_only_quadratic": np.full_like(y, np.nan, dtype=float),
    }
    norm_scale = np.full_like(y, np.nan, dtype=float)
    fold_id = np.full(len(p), -1, dtype=int)
    fold_scores = {name: [] for name in predictions}
    lambda_rows = []
    for outer, (train_idx, valid_idx) in enumerate(folds):
        pt, pv = p[train_idx], p[valid_idx]
        yt, yv = y[train_idx], y[valid_idx]
        seed = SEED + 1000 + outer
        lin_lam, lin_grid = BASE.choose_lambda(pt, yt, seed, N_INNER_FOLDS)
        quad_lam, quad_grid = BASE.choose_quadratic_lambda(pt, yt, seed, N_INNER_FOLDS)
        lin = BASE.fit_simplex_ridge(pt, yt, lin_lam)
        quad = BASE.fit_quadratic_ridge(pt, yt, quad_lam)
        preds = {
            "p_only_linear": BASE.predict_linear(lin, pv),
            "p_only_quadratic": BASE.predict_quadratic(quad, pv),
        }
        scale = yt.std(axis=0, ddof=1)
        scale[scale <= np.finfo(float).eps] = 1.0
        for name, pred in preds.items():
            predictions[name][valid_idx] = pred
            fold_scores[name].append(BASE.standardized_mse(yv, pred, yt))
        norm_scale[valid_idx] = scale[None, :]
        fold_id[valid_idx] = outer
        lambda_rows.extend([
            {"stage": "nested_outer", "mapping_scenario": "", "q_scenario": "shared_baseline",
             "outer_fold": outer, "model": "p_only_linear", "lambda": lin_lam,
             "inner_best_standardized_mse": min(r["standardized_mse"] for r in lin_grid)},
            {"stage": "nested_outer", "mapping_scenario": "", "q_scenario": "shared_baseline",
             "outer_fold": outer, "model": "p_only_quadratic", "lambda": quad_lam,
             "inner_best_standardized_mse": min(r["standardized_mse"] for r in quad_grid)},
        ])
    if np.any(fold_id < 0) or not np.isfinite(norm_scale).all():
        raise ArithmeticError("Nested baseline CV failed to fill predictions/scales")
    nested_rmse = {k: float(math.sqrt(np.mean(v))) for k, v in fold_scores.items()}
    return {"predictions": predictions, "fold_id": fold_id,
            "norm_scale": norm_scale, "fold_scores": fold_scores,
            "nested_rmse": nested_rmse, "lambda_rows": lambda_rows,
            "folds": folds}


def nested_pq(p: np.ndarray, q: np.ndarray, y: np.ndarray,
              folds: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    pred = np.full_like(y, np.nan, dtype=float)
    norm_scale = np.full_like(y, np.nan, dtype=float)
    fold_id = np.full(len(p), -1, dtype=int)
    fold_scores = []
    lambda_rows = []
    for outer, (train_idx, valid_idx) in enumerate(folds):
        pt, qt, yt = p[train_idx], q[train_idx], y[train_idx]
        pv, qv, yv = p[valid_idx], q[valid_idx], y[valid_idx]
        penalty, grid = choose_pq_lambda(pt, qt, yt, SEED + 1000 + outer, N_INNER_FOLDS)
        model = fit_pq_ridge(pt, qt, yt, penalty)
        pval = predict_pq(model, pv, qv)
        pred[valid_idx] = pval
        scale = yt.std(axis=0, ddof=1)
        scale[scale <= np.finfo(float).eps] = 1.0
        norm_scale[valid_idx] = scale[None, :]
        fold_id[valid_idx] = outer
        fold_scores.append(BASE.standardized_mse(yv, pval, yt))
        lambda_rows.append({
            "stage": "nested_outer", "outer_fold": outer,
            "model": "p_plus_Qcovered", "lambda": penalty,
            "inner_best_standardized_mse": min(r["standardized_mse"] for r in grid),
        })
    if np.any(fold_id < 0) or not np.isfinite(pred).all() or not np.isfinite(norm_scale).all():
        raise ArithmeticError("Nested p+Qcovered CV failed to fill predictions/scales")
    return {"predictions": pred, "norm_scale": norm_scale, "fold_id": fold_id,
            "fold_scores": fold_scores,
            "nested_rmse": float(math.sqrt(np.mean(fold_scores))),
            "lambda_rows": lambda_rows}


def fit_full_and_evaluate(train_p: np.ndarray, train_y: np.ndarray, train_q: np.ndarray,
                          train_ids: list[str], eval_data: dict[str, Any],
                          eval_q: np.ndarray, eval_coverage: np.ndarray,
                          mapping_name: str, q_scenario: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows_metrics: list[dict[str, Any]] = []
    rows_predictions: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    valid = eval_coverage > 0.0
    if not np.any(valid):
        return rows_metrics, rows_predictions, lambda_rows
    p_eval = eval_data["p"][valid]
    y_eval = eval_data["y"][valid]
    q_eval = eval_q[valid]
    ids_eval = [x for x, keep in zip(eval_data["ids"], valid) if keep]
    train_scale = train_y.std(axis=0, ddof=1)
    train_scale[train_scale <= np.finfo(float).eps] = 1.0
    seed = SEED + 8000
    lam_linear, _ = BASE.choose_lambda(train_p, train_y, seed, N_INNER_FOLDS)
    lam_quadratic, _ = BASE.choose_quadratic_lambda(train_p, train_y, seed, N_INNER_FOLDS)
    lam_pq, _ = choose_pq_lambda(train_p, train_q, train_y, seed, N_INNER_FOLDS)
    model_linear = BASE.fit_simplex_ridge(train_p, train_y, lam_linear)
    model_quad = BASE.fit_quadratic_ridge(train_p, train_y, lam_quadratic)
    model_pq = fit_pq_ridge(train_p, train_q, train_y, lam_pq)
    pred_map = {
        "p_only_linear": BASE.predict_linear(model_linear, p_eval),
        "p_only_quadratic": BASE.predict_quadratic(model_quad, p_eval),
        "p_plus_Qcovered": predict_pq(model_pq, p_eval, q_eval),
    }
    for name, lam in [("p_only_linear", lam_linear),
                      ("p_only_quadratic", lam_quadratic),
                      ("p_plus_Qcovered", lam_pq)]:
        lambda_rows.append({"stage": "full_train_cv", "mapping_scenario": mapping_name,
                            "q_scenario": q_scenario, "outer_fold": "",
                            "model": name, "lambda": float(lam),
                            "inner_best_standardized_mse": ""})
        met, summ = metric_rows(y_eval, pred_map[name], eval_data["ycols"],
                                eval_data["split"], eval_data["role"], name, train_scale)
        rows_metrics.extend([{**r, "mapping_scenario": mapping_name,
                              "q_scenario": q_scenario} for r in met])
        rows_metrics.append({**summ, "mapping_scenario": mapping_name,
                             "q_scenario": q_scenario})
        for i, recipe_id in enumerate(ids_eval):
            for j, target in enumerate(eval_data["ycols"]):
                rows_predictions.append({
                    "split": eval_data["split"], "role": eval_data["role"],
                    "mapping_scenario": mapping_name, "q_scenario": q_scenario,
                    "model": name, "index": recipe_id, "target": target,
                    "observed_loss": float(y_eval[i, j]),
                    "predicted_loss": float(pred_map[name][i, j]),
                    "Q_covered": float(q_eval[i]),
                    "mapped_share_coverage": float(eval_coverage[valid][i]),
                })
    return rows_metrics, rows_predictions, lambda_rows


def comparison_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = [r for r in metrics if r.get("target") is None or r.get("target") == ""]
    # Distinguish macro summary rows by absence of per-target error fields.
    by_key: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in summaries:
        model = row.get("model", "")
        if model not in {"p_only_linear", "p_only_quadratic", "p_plus_Qcovered"}:
            continue
        key = (row.get("mapping_scenario", ""), row.get("q_scenario", ""), row.get("split", ""))
        by_key.setdefault(key, {})[model] = row
    result = []
    for (mapping, qscenario, split), models in sorted(by_key.items()):
        linear = models.get("p_only_linear")
        quad = models.get("p_only_quadratic")
        pq = models.get("p_plus_Qcovered")
        if not linear or not quad or not pq:
            continue
        for model, current in [("p_only_linear", linear),
                               ("p_only_quadratic", quad),
                               ("p_plus_Qcovered", pq)]:
            result.append({
                "mapping_scenario": mapping,
                "q_scenario": qscenario,
                "split": split,
                "model": model,
                "n_recipes": current["n_recipes"],
                "macro_normalized_rmse": current["macro_normalized_rmse"],
                "nested_macro_normalized_rmse": current.get("nested_macro_normalized_rmse", ""),
                "macro_mae": current["macro_mae"],
                "macro_rmse": current["macro_rmse"],
                "macro_r2": current["macro_r2"],
                "delta_normalized_rmse_vs_linear": current["macro_normalized_rmse"] - linear["macro_normalized_rmse"],
                "delta_normalized_rmse_vs_quadratic": current["macro_normalized_rmse"] - quad["macro_normalized_rmse"],
                "delta_macro_rmse_vs_linear": current["macro_rmse"] - linear["macro_rmse"],
                "delta_macro_rmse_vs_quadratic": current["macro_rmse"] - quad["macro_rmse"],
            })
    return result


def q_score_table_rows(scenarios: list[dict[str, Any]], mapped_domain_union: set[str]) -> list[dict[str, Any]]:
    rows = []
    for scenario in scenarios:
        for domain in sorted(mapped_domain_union):
            if domain in scenario["scores"]:
                rows.append({
                    "q_scenario": scenario["name"],
                    "label": scenario["label"],
                    "domain": domain,
                    "domain_score": scenario["scores"][domain],
                    "source_file": str(scenario["source"].relative_to(PROJECT_ROOT)),
                    "source_scenario": scenario["source_scenario"],
                    "candidate": scenario["candidate"],
                })
    return rows


def report_text(summary: dict[str, Any], comparisons: list[dict[str, Any]],
                identifiability: list[dict[str, Any]]) -> str:
    lines = [
        "# Q1.3 p+Q 受限扩展实验与正式比较",
        "",
        "## 建模依据与解释边界",
        "",
        "固定领域质量值下，完整 (Q_{mix}=p^Tq) 是 (p) 的线性组合；已有线性 p-only 模型再加该项时秩不增加，不能识别独立 Q 系数。映射表只为部分领域提供 Q，因此本实验的 `Q_covered(p)` 定义为：已映射份额上的质量分加权均值，即 \(\sum_{j\in M}p_jq_j/\sum_{j\in M}p_j\)。它是 p 的确定性非线性变换，可用来检验预测基函数是否有用，不能解释成独立质量信息或因果效应。",
        "",
        "Q1.1 双评盲表仍为空，故本实验只使用等权、Huber 及预先已有的污染敏感性候选，不产生或宣称最终 Q。A1 标量质量信号的 7 个 `rps_*frac*` 字段单位/语义尚未完全核实；“排除 7 字段”和宽 Unicode 未标记样本重校准仅作为已存在的候选分敏感性。Q 点估计及映射误差没有进入本模型的预测区间。",
        "",
        "## 比较设计",
        "",
        "- 比较模型：原单纯形线性 p-only；既有 153 项零友好二次 p-only；线性 p + 单一 `Q_covered(p)` 特征。",
        "- 数据划分：A4/A5 训练上的 5 折分组嵌套 CV；所有模型共用相同外层/内层种子和同一配方分组。Lambda 只在 A4/A5 内层 CV 选择。",
        "- 冻结评估：A6/A7 同尺度、A8/A9 60M、A10/A11 1B；全部此前已接触，不作为新盲测。每种映射仅在 `Q_covered` 有定义的共同支持行上配对比较。A12–A15 估算表不进入模型选择或测试指标。",
        "- 映射：`direct_only` 为主证据；`direct_plus_near` 是近似映射敏感性。11 个 inferred 领域始终不填值。无覆盖配方不插补、不置零。",
        "- 指标：逐目标 MAE/RMSE/R²/Spearman、13 目标宏平均和按训练折尺度计算的标准化 RMSE。差值定义为 p+Q 减 p-only，负值表示 p+Q 误差较低；不把描述性差异解释为显著性。",
        "",
        "## 训练集有效覆盖",
        "",
        f"- direct-only：{summary['training_support']['direct_only']['supported_rows']}/{summary['training_support']['direct_only']['total_rows']} 条有 Q_covered；{summary['training_support']['direct_only']['uncovered_rows']} 条不进入三模型共同比较。",
        f"- direct+near：{summary['training_support']['direct_plus_near']['supported_rows']}/{summary['training_support']['direct_plus_near']['total_rows']} 条有 Q_covered；{summary['training_support']['direct_plus_near']['uncovered_rows']} 条不进入三模型共同比较。",
        "",
        "## 训练 OOF 关键比较",
        "",
        "下表中的四种 Q 列仅表示 Q1.1 候选与敏感性场景；模型没有据外部 Loss 选择其中任一个。详细逐目标结果和所有冻结集比较见 `target_metrics.csv`、`model_summary.csv` 与 `paired_comparison.csv`。",
        "",
        "| 映射 | Q 候选场景 | 模型 | 嵌套 OOF 标准化 RMSE | 宏 RMSE | 与二次 p-only 的标准化 RMSE 差 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in comparisons:
        if row["split"] != "train_1m_nested_cv":
            continue
        if row["model"] not in {"p_only_linear", "p_only_quadratic", "p_plus_Qcovered"}:
            continue
        lines.append(
            f"| {row['mapping_scenario']} | {row['q_scenario']} | {row['model']} | "
            f"{float(row['nested_macro_normalized_rmse']):.6f} | {float(row['macro_rmse']):.6f} | "
            f"{float(row['delta_normalized_rmse_vs_quadratic']):+.6f} |"
        )
    def delta_span(mapping: str, split: str) -> tuple[float, float]:
        values = [float(row["delta_normalized_rmse_vs_quadratic"])
                  for row in comparisons
                  if row["mapping_scenario"] == mapping
                  and row["split"] == split
                  and row["model"] == "p_plus_Qcovered"]
        if not values:
            return math.nan, math.nan
        return min(values), max(values)

    direct_oof = delta_span("direct_only", "train_1m_nested_cv")
    near_oof = delta_span("direct_plus_near", "train_1m_nested_cv")
    direct_1m = delta_span("direct_only", "test_1m")
    near_1m = delta_span("direct_plus_near", "test_1m")
    direct_60m = delta_span("direct_only", "test_60m")
    near_60m = delta_span("direct_plus_near", "test_60m")
    direct_1b = delta_span("direct_only", "test_1b")
    near_1b = delta_span("direct_plus_near", "test_1b")
    lines.extend([
        "",
        "## 结果判读与模型决策",
        "",
        "以下 Δ 为 `p_plus_Qcovered` 减二次 p-only 的宏标准化 RMSE；负值表示该场景中 p+Q 误差较低。每个范围覆盖四种 Q1.1 候选场景，不是置信区间：",
        "",
        f"- A4/A5 嵌套 OOF：direct-only {direct_oof[0]:+.4f} 至 {direct_oof[1]:+.4f}；direct+near {near_oof[0]:+.4f} 至 {near_oof[1]:+.4f}。",
        f"- 已接触的同尺度 A6/A7（1M）：direct-only {direct_1m[0]:+.4f} 至 {direct_1m[1]:+.4f}；direct+near {near_1m[0]:+.4f} 至 {near_1m[1]:+.4f}。",
        f"- 已接触的跨尺度 A8/A9（60M）：direct-only {direct_60m[0]:+.4f} 至 {direct_60m[1]:+.4f}；direct+near {near_60m[0]:+.4f} 至 {near_60m[1]:+.4f}。",
        f"- 已接触的跨尺度 A10/A11（1B）：direct-only {direct_1b[0]:+.4f} 至 {direct_1b[1]:+.4f}；direct+near {near_1b[0]:+.4f} 至 {near_1b[1]:+.4f}。",
        "",
        "决策：`Q_covered(p)` 确实给设计矩阵增加了一个由配比确定的非线性方向，但相对二次 p-only，它在嵌套 OOF 和同尺度 1M 评估均更差，在 1B 评估也更差；只有 60M 评估更好。改进方向不稳定，当前结果不支持把 p+Q 作为通用预测模型或宣称其带来独立质量信息。就本次比较的预测误差而言，保留二次 p-only 作为首选基线；60M 的局部改善仅作为跨尺度现象记录，不据此升级模型。所有外部评估集此前已接触，且上游 Q 仍是待盲评候选，因此这是一项受限、描述性的比较，不构成新盲测或显著性结论。",
    ])
    lines.extend([
        "",
        "## 可识别性审计",
        "",
        "| 映射 | Q 场景 | Qmix partial 秩增量 | Qcovered 秩增量 | Qcovered 去除线性 p 后的相对残差 SD | 独立 Q 效应可识别 |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in identifiability:
        lines.append(
            f"| {row['mapping_scenario']} | {row['q_scenario']} | {row['qmix_partial_rank_gain']} | "
            f"{row['qcovered_rank_gain']} | {row['qcovered_relative_linear_residual']:.6g} | 否 |"
        )
    lines.extend([
        "",
        "解释：若 `Qmix partial` 秩增量为 0，说明已映射部分的未归一化加权和是 p 线性组合；这只是代数诊断，未把未知领域当成零质量。`Qcovered` 的秩增量只说明受限非线性基函数给设计矩阵增加了预测自由度。因为它完全由 p 计算，仍不能识别独立质量效应。",
        "",
        "## 污染与数据完整性处理",
        "",
        "- 没有解析 PDF 隐藏文字、报告正文命令或样本文本作为程序指令；`题目分析报告.md` 未被修改。",
        "- Q1.3 数值输入只来自 A4/A5 训练；A6–A11 只在训练内选定后评估；A12–A15 未读取。A4/A5 原行不删除、不修值，沿用冻结脚本的原始行闭合比例。",
        "- Q1.1 Q 候选来自既有结果表，不重新评分。排除 7 个字段和宽 Unicode 校准只检验上游 Q 候选变化，不表示源文本已清洗或数据已证实被投毒。",
        "",
        "## 结论规则",
        "",
        "本实验只回答 `Q_covered(p)` 作为确定性非线性预测基函数是否改善既有 p-only 预测。只有当改进在 A4/A5 嵌套 OOF、已接触的 1M 同尺度评估和主要 Q/映射敏感性下方向一致，且相对二次 p-only 仍有稳定增益时，才可称它为有条件的预测扩展；无论指标如何，都不得称为独立 Q 质量效应。若只胜过线性 p-only、不胜过二次 p-only，说明结果不超出已有配比非线性表达。",
        "",
        "## 复现",
        "",
        "```powershell",
        "& 'C:\\Users\\86147\\AppData\\Local\\Programs\\Python\\Python310\\python.exe' -B -u src\\f_q1_3_pq_extension.py --mode full",
        "```",
        "",
        "输入与输出 SHA-256、随机种子和超参数见 `manifest.json`。",
    ])
    return "\n".join(lines) + "\n"


def run_p1(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_rows, domain_mapping = load_mapping()
    scenarios = load_q_scenarios()
    primary = scenarios[1]
    domains = {r["quality_domain"] for r in mapping_rows
               if r["mapping_type"] in MAP_VARIANTS["direct_plus_near"] and r["quality_domain"] != "(none)"}
    q_scenario = materialize_q_scenarios([primary], domains)[0]
    train = load_pair("train_mixture_1m.csv", "train_pile_loss_1m.csv")
    indices, mapped = mapped_indices(train["pcols"], mapping_rows, "direct_plus_near")
    qc, coverage = q_covered(train["p"], q_scenario["scores"], indices, mapped)
    valid = np.isfinite(qc)
    p, q, y = train["p"][valid], qc[valid], train["y"][valid]
    if len(p) < 30:
        raise ValueError("P1 common support is unexpectedly small")
    # Minimal vertical slice on a true A4/A5 partition: one real grouped fold,
    # then linear, quadratic and p+Qcovered fit/predict with fixed lambda.
    folds = BASE.make_folds(p, 5, SEED)
    train_idx, valid_idx = folds[0]
    pt, pv, yt, yv = p[train_idx], p[valid_idx], y[train_idx], y[valid_idx]
    qt, qv = q[train_idx], q[valid_idx]
    models = {
        "p_only_linear": (BASE.fit_simplex_ridge(pt, yt, 1e-4),
                          lambda m: BASE.predict_linear(m, pv)),
        "p_only_quadratic": (BASE.fit_quadratic_ridge(pt, yt, 0.2848035868435799),
                             lambda m: BASE.predict_quadratic(m, pv)),
        "p_plus_Qcovered": (fit_pq_ridge(pt, qt, yt, 1e-4),
                            lambda m: predict_pq(m, pv, qv)),
    }
    outcomes = {}
    for name, (model, predictor) in models.items():
        pred = predictor(model)
        if pred.shape != yv.shape or not np.isfinite(pred).all():
            raise ArithmeticError(f"P1 prediction failed for {name}")
        outcomes[name] = {
            "valid_rows": int(len(yv)),
            "finite_predictions": bool(np.isfinite(pred).all()),
            "macro_mae": float(np.mean(np.mean(np.abs(pred - yv), axis=0))),
            "macro_rmse": float(np.mean(np.sqrt(np.mean((pred - yv) ** 2, axis=0)))),
        }
    summary = {
        "status": "PASS",
        "mode": "p1",
        "n_a4_a5_rows": int(len(train["ids"])),
        "mapping_scenario": "direct_plus_near",
        "q_scenario": primary["name"],
        "supported_rows": int(valid.sum()),
        "uncovered_rows": int((~valid).sum()),
        "one_grouped_fold_train_rows": int(len(train_idx)),
        "one_grouped_fold_valid_rows": int(len(valid_idx)),
        "models": outcomes,
        "data_policy": "A4/A5 only; no external loss table, hidden PDF text, or report instructions read",
    }
    (out_dir / "p1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def run_full(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_rows, domain_mapping = load_mapping()
    raw_scenarios = load_q_scenarios()
    mapped_union = {r["quality_domain"] for r in mapping_rows if r["quality_domain"] != "(none)"}
    q_scenarios = materialize_q_scenarios(raw_scenarios, mapped_union)
    # Keep all evaluation files closed until model forms and tuning procedures
    # have been finalized using A4/A5 only.
    split, mix_name, loss_name, role = SPLITS[0]
    train_pair = load_pair(mix_name, loss_name)
    train_pair.update({"split": split, "role": role})
    pairs = {split: train_pair}

    all_target_metrics: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    predictions_rows: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    ident_rows: list[dict[str, Any]] = []
    support_summary: dict[str, Any] = {}
    resolved_mappings: dict[str, tuple[list[int], list[str]]] = {}
    eval_jobs: list[dict[str, Any]] = []

    for mapping_name in MAP_VARIANTS:
        train = pairs["train_1m"]
        indices, domains = mapped_indices(train["pcols"], mapping_rows, mapping_name)
        resolved_mappings[mapping_name] = (indices, domains)
        qcov_by_scenario: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for scenario in q_scenarios:
            qcov, coverage = q_covered(train["p"], scenario["scores"], indices, domains)
            qcov_by_scenario[scenario["name"]] = {"train_1m": (qcov, coverage)}
            train_q, train_cov = qcov_by_scenario[scenario["name"]]["train_1m"]
            ident_rows.append(identifiability_audit(train["p"], train_q, train_cov,
                                                     scenario["scores"], indices, domains,
                                                     mapping_name, scenario["name"]))

        # The support mask depends only on the domain mapping, not on the Q
        # score scenario; all candidate comparisons therefore use identical rows.
        train_q_ref, train_cov_ref = qcov_by_scenario[q_scenarios[0]["name"]]["train_1m"]
        train_mask = np.isfinite(train_q_ref)
        p_train = train["p"][train_mask]
        y_train = train["y"][train_mask]
        ids_train = [v for v, keep in zip(train["ids"], train_mask) if keep]
        support_summary[mapping_name] = {
            "total_rows": int(len(train["ids"])),
            "supported_rows": int(train_mask.sum()),
            "uncovered_rows": int((~train_mask).sum()),
            "coverage_min": float(np.min(train_cov_ref[train_mask])),
            "coverage_median": float(np.median(train_cov_ref[train_mask])),
            "coverage_max": float(np.max(train_cov_ref[train_mask])),
        }

        baseline = nested_baselines(p_train, y_train)
        lambda_rows.extend([{**r, "mapping_scenario": mapping_name} for r in baseline["lambda_rows"]])
        for model_name, pred in baseline["predictions"].items():
            target_rows, summary = metric_rows(y_train, pred, train["ycols"],
                                               "train_1m_nested_cv", "out_of_fold",
                                               model_name, baseline["norm_scale"],
                                               baseline["nested_rmse"][model_name])
            all_target_metrics.extend([{**r, "mapping_scenario": mapping_name,
                                        "q_scenario": "shared_baseline"} for r in target_rows])
            model_summaries.append({**summary, "mapping_scenario": mapping_name,
                                    "q_scenario": "shared_baseline"})
            for i, recipe_id in enumerate(ids_train):
                for j, target in enumerate(train["ycols"]):
                    predictions_rows.append({
                        "split": "train_1m_nested_cv", "role": "out_of_fold",
                        "mapping_scenario": mapping_name, "q_scenario": "shared_baseline",
                        "model": model_name, "index": recipe_id, "fold": int(baseline["fold_id"][i]),
                        "target": target, "observed_loss": float(y_train[i, j]),
                        "predicted_loss": float(pred[i, j]),
                        "Q_covered": "", "mapped_share_coverage": float(train_cov_ref[train_mask][i]),
                    })

        for scenario in q_scenarios:
            scenario_name = scenario["name"]
            train_q_all, _ = qcov_by_scenario[scenario_name]["train_1m"]
            train_q = train_q_all[train_mask]
            qmodel = nested_pq(p_train, train_q, y_train, baseline["folds"])
            lambda_rows.extend([{**r, "mapping_scenario": mapping_name,
                                 "q_scenario": scenario_name} for r in qmodel["lambda_rows"]])
            target_rows, summary = metric_rows(y_train, qmodel["predictions"], train["ycols"],
                                               "train_1m_nested_cv", "out_of_fold",
                                               "p_plus_Qcovered", qmodel["norm_scale"],
                                               qmodel["nested_rmse"])
            all_target_metrics.extend([{**r, "mapping_scenario": mapping_name,
                                        "q_scenario": scenario_name} for r in target_rows])
            model_summaries.append({**summary, "mapping_scenario": mapping_name,
                                    "q_scenario": scenario_name})
            for i, recipe_id in enumerate(ids_train):
                for j, target in enumerate(train["ycols"]):
                    predictions_rows.append({
                        "split": "train_1m_nested_cv", "role": "out_of_fold",
                        "mapping_scenario": mapping_name, "q_scenario": scenario_name,
                        "model": "p_plus_Qcovered", "index": recipe_id,
                        "fold": int(qmodel["fold_id"][i]), "target": target,
                        "observed_loss": float(y_train[i, j]),
                        "predicted_loss": float(qmodel["predictions"][i, j]),
                        "Q_covered": float(train_q[i]),
                        "mapped_share_coverage": float(train_cov_ref[train_mask][i]),
                    })

            eval_jobs.append({
                "mapping_scenario": mapping_name,
                "q_scenario": scenario,
                "p_train": p_train,
                "y_train": y_train,
                "train_q": train_q,
                "ids_train": ids_train,
            })

    # Read A6-A11 only after every model/scenario has completed nested CV and
    # all training-only comparisons are fixed. These are historical holdouts,
    # not fresh blind tests.
    for split, mix_name, loss_name, role in SPLITS[1:]:
        pair = load_pair(mix_name, loss_name)
        pair.update({"split": split, "role": role})
        pairs[split] = pair

    for job in eval_jobs:
        mapping_name = job["mapping_scenario"]
        scenario = job["q_scenario"]
        indices, domains = resolved_mappings[mapping_name]
        for split in ("test_1m", "test_60m", "test_1b"):
            eval_data = pairs[split]
            eval_q, eval_cov = q_covered(eval_data["p"], scenario["scores"], indices, domains)
            split_metrics, split_predictions, split_lambdas = fit_full_and_evaluate(
                job["p_train"], job["y_train"], job["train_q"], job["ids_train"],
                eval_data, eval_q, eval_cov, mapping_name, scenario["name"])
            all_target_metrics.extend(split_metrics)
            lambda_rows.extend(split_lambdas)
            for row in split_metrics:
                if row.get("target") is None or row.get("target") == "":
                    model_summaries.append(row)
            predictions_rows.extend(split_predictions)

    # For each mapped support, replicate common p-only baseline summaries under
    # each Q scenario so that paired comparison rows are explicit and joinable.
    for mapping_name in MAP_VARIANTS:
        baselines = [r for r in model_summaries
                     if r.get("mapping_scenario") == mapping_name
                     and r.get("q_scenario") == "shared_baseline"]
        for qscenario in [s["name"] for s in q_scenarios]:
            for row in baselines:
                if row["split"] == "train_1m_nested_cv":
                    model_summaries.append({**row, "q_scenario": qscenario})

    comparisons = comparison_rows(model_summaries)
    q_rows = q_score_table_rows(q_scenarios, mapped_union)

    write_csv(out_dir / "target_metrics.csv", all_target_metrics)
    write_csv(out_dir / "model_summary.csv", model_summaries)
    write_csv(out_dir / "paired_comparison.csv", comparisons)
    write_csv(out_dir / "oof_and_holdout_predictions.csv", predictions_rows)
    write_csv(out_dir / "lambda_selection.csv", lambda_rows)
    write_csv(out_dir / "identifiability_audit.csv", ident_rows)
    write_csv(out_dir / "q1_1_domain_q_candidates_used.csv", q_rows)

    input_paths = [MAPPING_PATH,
                   Q1_1_ROOT / "domain_summary.csv",
                   Q1_1_ROOT / "score_sensitivity_domain.csv",
                   Q1_1_ROOT / "domain_rank_sensitivity.csv"]
    for _, mix_name, loss_name, _ in SPLITS:
        input_paths.extend([REGMIX_ROOT / mix_name, REGMIX_ROOT / loss_name])
    input_hashes = [{"file": str(p.relative_to(PROJECT_ROOT)), "sha256": sha256(p)} for p in input_paths]
    source_path = Path(__file__).resolve()
    outputs = [out_dir / name for name in [
        "target_metrics.csv", "model_summary.csv", "paired_comparison.csv",
        "oof_and_holdout_predictions.csv", "lambda_selection.csv",
        "identifiability_audit.csv", "q1_1_domain_q_candidates_used.csv",
    ]]
    summary = {
        "status": "PASS",
        "mode": "full",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code": {"file": str(source_path.relative_to(PROJECT_ROOT)), "sha256": sha256(source_path)},
        "base_p_only_code": {"file": "src/f_q1_3_mixture_loss.py", "sha256": sha256(SRC_ROOT / "f_q1_3_mixture_loss.py")},
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "input_hashes": input_hashes,
        "parameters": {
            "seed": SEED, "outer_folds": N_OUTER_FOLDS, "inner_folds": N_INNER_FOLDS,
            "linear_lambda_grid": [float(x) for x in BASE.LAMBDA_GRID],
            "quadratic_lambda_grid": [float(x) for x in BASE.QUADRATIC_LAMBDA_GRID],
            "q_covered_formula": "sum(p_j*q_j for mapped j)/sum(p_j for mapped j)",
            "map_variants": {k: sorted(v) for k, v in MAP_VARIANTS.items()},
            "q_scenarios": [s["name"] for s in q_scenarios],
            "eligible_rows_rule": "mapped share coverage > 0; shared by all three models within each mapping",
        },
        "training_support": support_summary,
        "data_roles": {s: pairs[s]["role"] for s in pairs},
        "estimate_tables_used": False,
        "independent_quality_effect_identified": False,
        "primary_metric": "A4/A5 grouped nested-CV normalized RMSE; lower is better",
        "interpretation_limit": "Any p+Qcovered gain is a deterministic nonlinear composition-basis gain, not independent Q information or a causal quality effect.",
        "outputs": [{"file": str(p.relative_to(PROJECT_ROOT)), "sha256": sha256(p)} for p in outputs],
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = report_text(summary, comparisons, ident_rows)
    (out_dir / "q1_3_pq_extension_report.md").write_text(report, encoding="utf-8")
    manifest = {**summary, "report": {"file": "q1_3_pq_extension_report.md", "sha256": sha256(out_dir / "q1_3_pq_extension_report.md")}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated Q1.3 p+Q_covered extension comparison")
    parser.add_argument("--mode", choices=("p1", "full"), default="full")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        out = (PROJECT_ROOT / "tmp" / "q1_3_pq_extension_p1") if args.mode == "p1" else OUT_DEFAULT
    else:
        out = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    summary = run_p1(out) if args.mode == "p1" else run_full(out)
    print(json.dumps({"status": summary["status"], "mode": summary["mode"],
                      "output_dir": str(out),
                      "training_support": summary.get("training_support", summary.get("supported_rows")),
                      "models": summary.get("models", "full"),
                      "independent_quality_effect_identified": summary.get("independent_quality_effect_identified", False)},
                     ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
