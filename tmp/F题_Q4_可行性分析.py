"""F 题 Q4 小型可行性分析。

只读附件 C，输出数据可用性、Loss--Benchmark 桥接和一个透明的基线趋势/分解试算。
该脚本不是最终 Q4 模型：不处理因果识别、重复提交去重和完整不确定性传播。
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution"


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        value = str(value).replace(",", "")
        x = float(value)
        return x if math.isfinite(x) else None
    except ValueError:
        return None


def year(value: str | None) -> int | None:
    if not value or not value.strip():
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S%z"):
        try:
            return datetime.strptime(value.strip(), fmt).year
        except ValueError:
            pass
    try:
        return int(value[:4])
    except (TypeError, ValueError):
        return None


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def rank(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float), kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(rank(x).tolist(), rank(y).tolist())


def ols(x: list[list[float]], y: list[float]) -> dict[str, object]:
    X, Y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    beta, _, rank_x, singular = np.linalg.lstsq(X, Y, rcond=None)
    fitted = X @ beta
    residual = Y - fitted
    sst = float(np.sum((Y - np.mean(Y)) ** 2))
    r2 = None if sst == 0 else float(1 - np.sum(residual**2) / sst)
    return {
        "n": int(len(Y)),
        "rank": int(rank_x),
        "condition_number": float(singular[0] / singular[-1]) if len(singular) and singular[-1] else None,
        "beta": [float(v) for v in beta],
        "r2": r2,
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def data_audit(cleaned: list[dict[str, str]], enhanced: list[dict[str, str]], timeseries: list[dict[str, str]], bridge: list[dict[str, str]]) -> dict[str, object]:
    dates = [year(r.get("Submission Date")) for r in cleaned]
    dates = [v for v in dates if v is not None]
    enhanced_dates = [year(r.get("Submission Date")) for r in enhanced]
    enhanced_dates = [v for v in enhanced_dates if v is not None]
    type_counts = Counter(r.get("Type", "") for r in cleaned)
    bridge_levels = Counter(r.get("Loss_Comparability", "") for r in bridge)
    return {
        "leaderboard_cleaned": {
            "rows": len(cleaned),
            "unique_models": len({r.get("Model") for r in cleaned}),
            "duplicate_model_rows": len(cleaned) - len({r.get("Model") for r in cleaned}),
            "submission_year_range": [min(dates), max(dates)] if dates else None,
            "type_counts": dict(type_counts),
        },
        "leaderboard_enhanced": {
            "rows": len(enhanced),
            "publication_date_missing": sum(not r.get("Epoch_AI_Publication_Date", "").strip() for r in enhanced),
            "open_weights_missing": sum(not r.get("Epoch_AI_Open_Weights", "").strip() for r in enhanced),
            "open_weights_yes": sum(r.get("Epoch_AI_Open_Weights", "").strip().lower() == "yes" for r in enhanced),
            "submission_year_range": [min(enhanced_dates), max(enhanced_dates)] if enhanced_dates else None,
        },
        "leaderboard_extended_timeseries": {
            "rows": len(timeseries),
            "year_counts": dict(sorted(Counter(int(r["Year"]) for r in timeseries).items())),
            "source_counts": dict(Counter(r.get("Source", "") for r in timeseries)),
        },
        "bridge": {
            "rows": len(bridge),
            "comparability_counts": dict(bridge_levels),
            "d_tokens_present": sum(num(r.get("D_tokens_B")) is not None for r in bridge),
            "loss_present": sum(num(r.get("Val_Loss")) is not None for r in bridge),
            "benchmark_present": sum(num(r.get("LB_Average")) is not None for r in bridge),
        },
    }


def bridge_check(bridge: list[dict[str, str]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for label, rows in {
        "all": bridge,
        "high_only": [r for r in bridge if r.get("Loss_Comparability", "").startswith("High")],
        "medium_only": [r for r in bridge if r.get("Loss_Comparability", "").startswith("Medium")],
    }.items():
        rows = [r for r in rows if num(r.get("Val_Loss")) is not None and num(r.get("LB_Average")) is not None]
        losses = [num(r["Val_Loss"]) for r in rows]
        scores = [num(r["LB_Average"]) for r in rows]
        fit = ols([[1.0, loss] for loss in losses], scores) if len(rows) >= 2 else None
        result[label] = {
            "n": len(rows),
            "pearson_loss_vs_average": pearson(losses, scores),
            "spearman_loss_vs_average": spearman(losses, scores),
            "linear_score_on_loss": fit,
        }
    return result


def frontier_check(timeseries: list[dict[str, str]]) -> dict[str, object]:
    by_year: dict[int, list[float]] = defaultdict(list)
    for row in timeseries:
        y, score = num(row.get("Year")), num(row.get("Average"))
        if y is not None and score is not None:
            by_year[int(y)].append(score)
    annual: list[dict[str, float | int]] = []
    for y in sorted(by_year):
        scores = np.asarray(by_year[y], dtype=float)
        k = min(5, len(scores))
        annual.append({
            "year": y,
            "n": int(len(scores)),
            "max": float(np.max(scores)),
            "p95": float(np.percentile(scores, 95)),
            "top5_mean": float(np.mean(np.sort(scores)[-k:])),
        })
    x = [[1.0, float(row["year"])] for row in annual]
    y = [float(row["top5_mean"]) for row in annual]
    fit = ols(x, y) if len(annual) >= 3 else None
    forecasts = None
    backtest = []
    if fit is not None:
        beta = fit["beta"]
        last_year = int(annual[-1]["year"])
        forecasts = {
            str(last_year + h): float(beta[0] + beta[1] * (last_year + h))
            for h in (1, 2)
        }
        for i in range(2, len(annual)):
            train = annual[:i]
            train_fit = ols([[1.0, float(row["year"])] for row in train], [float(row["top5_mean"]) for row in train])
            pred = train_fit["beta"][0] + train_fit["beta"][1] * int(annual[i]["year"])
            actual = float(annual[i]["top5_mean"])
            backtest.append({"forecast_year": int(annual[i]["year"]), "actual": actual, "predicted": float(pred), "abs_error": abs(actual - pred)})
    return {"annual_frontier": annual, "linear_top5_fit": fit, "forecasts": forecasts, "rolling_backtest": backtest}


def scale_technology_check(enhanced: list[dict[str, str]]) -> dict[str, object]:
    rows = []
    for r in enhanced:
        model_type = r.get("Type", "").lower()
        if "pretrained" not in model_type:
            continue
        n, score, y = num(r.get("#Params (B)")), num(r.get("Average ⬆️")), year(r.get("Submission Date"))
        if n is not None and n > 0 and score is not None and y is not None:
            rows.append({"log_params": math.log10(n), "year_centered": y - 2024, "score": score})
    fit = None
    if len(rows) >= 6:
        X = [[1.0, r["log_params"], r["year_centered"]] for r in rows]
        fit = ols(X, [r["score"] for r in rows])
    return {
        "scope": "Type contains pretrained; continuously pretrained is included, chat/fine-tuned is excluded",
        "n": len(rows),
        "year_counts": dict(sorted(Counter(r["year_centered"] + 2024 for r in rows).items())),
        "log_params_year_correlation": pearson([r["log_params"] for r in rows], [r["year_centered"] for r in rows]) if rows else None,
        "ols_average_on_log10_params_and_year": fit,
    }


def main() -> None:
    cleaned = read_csv("leaderboard_cleaned.csv")
    enhanced = read_csv("leaderboard_enhanced.csv")
    timeseries = read_csv("leaderboard_extended_timeseries.csv")
    bridge = read_csv("loss_benchmark_bridge_expanded.csv")
    output = {
        "data_audit": data_audit(cleaned, enhanced, timeseries, bridge),
        "bridge_check": bridge_check(bridge),
        "frontier_check": frontier_check(timeseries),
        "scale_technology_check": scale_technology_check(enhanced),
        "interpretation": {
            "status": "pilot_feasible_with_strong_caveats",
            "main_caveats": [
                "bridge mostly medium comparability; high-comparability rows are only the Pythia family",
                "enhanced publication/open-weight metadata are mostly missing",
                "leaderboard contains duplicate model names and mixed model types",
                "frontier forecast is an exploratory linear extrapolation, not a causal prediction",
            ],
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
