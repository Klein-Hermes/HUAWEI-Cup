from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "C5": ROOT / "中文题目/F题/real_attachments/C_efficiency_evolution/loss_benchmark_bridge.csv",
    "C6": ROOT / "中文题目/F题/real_attachments/C_efficiency_evolution/loss_benchmark_bridge_expanded.csv",
    "license_audit": ROOT / "Q4/03_结果/results/q4_license_source_audit.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def publisher_group(model: str) -> str:
    return model.split("/", 1)[0].strip().casefold() if "/" in model else model.strip().casefold()


def linear_fit(rows: list[dict[str, str]]) -> tuple[float, float, float]:
    xs = [float(row["Val_Loss"]) for row in rows]
    ys = [float(row["LB_Average"]) for row in rows]
    x_bar = statistics.mean(xs)
    y_bar = statistics.mean(ys)
    sxx = sum((x - x_bar) ** 2 for x in xs)
    if len(rows) < 3 or sxx <= 0:
        raise ValueError("OLS requires at least three rows with varying Val_Loss")
    slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / sxx
    intercept = y_bar - slope * x_bar
    return intercept, slope, y_bar


def regression_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    intercept, slope, y_bar = linear_fit(rows)
    xs = [float(row["Val_Loss"]) for row in rows]
    ys = [float(row["LB_Average"]) for row in rows]
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    total = sum((y - y_bar) ** 2 for y in ys)
    sse = sum(error**2 for error in residuals)
    covariance = sum((x - statistics.mean(xs)) * (y - y_bar) for x, y in zip(xs, ys))
    sxx = sum((x - statistics.mean(xs)) ** 2 for x in xs)
    syy = total
    pearson = covariance / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")
    return {
        "intercept": intercept,
        "slope_per_nats_token": slope,
        "pearson_r": pearson,
        "r_squared": 1 - sse / total if total > 0 else float("nan"),
        "in_sample_mae": statistics.mean(abs(error) for error in residuals),
        "in_sample_rmse": math.sqrt(statistics.mean(error**2 for error in residuals)),
        "in_sample_mean_baseline_mae": statistics.mean(abs(y - y_bar) for y in ys),
        "in_sample_mean_baseline_rmse": math.sqrt(statistics.mean((y - y_bar) ** 2 for y in ys)),
    }


def holdout_metrics(
    rows: list[dict[str, str]], method: str
) -> tuple[dict[str, float | int | str], list[dict[str, object]]]:
    if method == "leave_one_size_out":
        fold_keys = {row["Model"] for row in rows}
        get_key = lambda row: row["Model"]
    elif method == "leave_one_publisher_out":
        fold_keys = {publisher_group(row["Model"]) for row in rows}
        get_key = lambda row: publisher_group(row["Model"])
    else:
        raise ValueError(method)

    errors: list[float] = []
    baseline_errors: list[float] = []
    fold_rows: list[dict[str, object]] = []
    for key in sorted(fold_keys):
        test = [row for row in rows if get_key(row) == key]
        train = [row for row in rows if get_key(row) != key]
        if len(train) < 3 or len({float(row["Val_Loss"]) for row in train}) < 2:
            continue
        intercept, slope, train_mean = linear_fit(train)
        fold_errors: list[float] = []
        fold_baseline_errors: list[float] = []
        for row in test:
            x = float(row["Val_Loss"])
            y = float(row["LB_Average"])
            error = y - (intercept + slope * x)
            baseline_error = y - train_mean
            errors.append(error)
            baseline_errors.append(baseline_error)
            fold_errors.append(error)
            fold_baseline_errors.append(baseline_error)
        fold_rows.append(
            {
                "holdout_method": method,
                "held_out_group": key,
                "n_train": len(train),
                "n_test": len(test),
                "intercept": intercept,
                "slope_per_nats_token": slope,
                "holdout_mae": statistics.mean(abs(error) for error in fold_errors),
                "holdout_rmse": math.sqrt(statistics.mean(error**2 for error in fold_errors)),
                "mean_baseline_mae": statistics.mean(abs(error) for error in fold_baseline_errors),
                "mean_baseline_rmse": math.sqrt(
                    statistics.mean(error**2 for error in fold_baseline_errors)
                ),
            }
        )

    if not errors:
        raise ValueError(f"No valid folds for {method}")
    return (
        {
            "holdout_method": method,
            "holdout_n": len(errors),
            "holdout_fold_n": len(fold_keys),
            "holdout_mae": statistics.mean(abs(error) for error in errors),
            "holdout_rmse": math.sqrt(statistics.mean(error**2 for error in errors)),
            "holdout_mean_baseline_mae": statistics.mean(abs(error) for error in baseline_errors),
            "holdout_mean_baseline_rmse": math.sqrt(
                statistics.mean(error**2 for error in baseline_errors)
            ),
        },
        fold_rows,
    )


bridge_summary: list[dict[str, object]] = []
bridge_fold_details: list[dict[str, object]] = []
for source, path in (("C5", INPUTS["C5"]), ("C6_expanded", INPUTS["C6"])):
    rows = read_csv(path)
    by_tier: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_tier[row["Loss_Comparability"].strip()].append(row)
    for tier, tier_rows in sorted(by_tier.items()):
        fit = regression_metrics(tier_rows)
        cv_method = "leave_one_size_out" if tier.startswith("High") else "leave_one_publisher_out"
        cv, folds = holdout_metrics(tier_rows, cv_method)
        for fold in folds:
            bridge_fold_details.append(
                {
                    "source": source,
                    "comparability_tier": tier,
                    "n_publisher_groups": len({publisher_group(row["Model"]) for row in tier_rows}),
                    **fold,
                }
            )
        bridge_summary.append(
            {
                "source": source,
                "comparability_tier": tier,
                "n_rows": len(tier_rows),
                "n_publisher_groups": len({publisher_group(row["Model"]) for row in tier_rows}),
                "loss_min": min(float(row["Val_Loss"]) for row in tier_rows),
                "loss_max": max(float(row["Val_Loss"]) for row in tier_rows),
                **fit,
                **cv,
                "beats_holdout_mean_baseline_mae": cv["holdout_mae"] < cv["holdout_mean_baseline_mae"],
                "beats_holdout_mean_baseline_rmse": cv["holdout_rmse"] < cv["holdout_mean_baseline_rmse"],
            }
        )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty result: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


bridge_path = OUT / "q4_bridge_tier_validation.csv"
write_csv(bridge_path, bridge_summary)
bridge_folds_path = OUT / "q4_bridge_holdout_folds.csv"
write_csv(bridge_folds_path, bridge_fold_details)

license_rows = read_csv(INPUTS["license_audit"])
open_yes = [
    row for row in license_rows
    if (row.get("c2_epoch_open_weights_raw") or "").strip().casefold() == "yes"
]
license_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in open_yes:
    label = (row.get("c1_hub_license_raw") or "").strip() or "<blank>"
    license_groups[label].append(row)

license_summary = []
for label, rows in sorted(license_groups.items(), key=lambda item: (-len(item[1]), item[0].casefold())):
    license_summary.append(
        {
            "open_weights_filter": "C2 Open Weights=Yes",
            "c1_hub_license_raw": label,
            "n_models": len(rows),
            "strict_version_confirmed_n": sum(
                (row.get("strict_version_confirmed") or "").strip().casefold() == "true"
                for row in rows
            ),
            "permission_status_counts": json.dumps(
                dict(Counter(row.get("research_reproduction_permission_status", "") for row in rows)),
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
    )
license_path = OUT / "q4_open_weight_license_coverage.csv"
write_csv(license_path, license_summary)

script_path = Path(__file__).resolve()
outputs = [bridge_path, bridge_folds_path, license_path]
manifest = {
    "schema_version": 1,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "script": str(script_path.relative_to(ROOT)).replace("\\", "/"),
    "python_version": platform.python_version(),
    "scope": "stratified descriptive Loss-Benchmark diagnostics and license-field coverage; no calibration or legal determination",
    "inputs": {
        key: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)}
        for key, path in INPUTS.items()
    },
    "script_sha256": sha256(script_path),
    "outputs": {
        path.name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)}
        for path in outputs
    },
    "bridge_validation": {
        "C5_primary_medium_validation": "leave-one-publisher-out",
        "C5_high_validation": "leave-one-size-out within one Pythia family",
        "C6_expanded_used_as_sensitivity_only": True,
        "no_rows_pooled_across_comparability_tiers": True,
        "publisher_group": "Model namespace prefix before the first slash; full model identifier when no slash exists",
    },
    "license_validation": {
        "open_weight_row_count": len(open_yes),
        "permission_statuses": dict(Counter(row.get("research_reproduction_permission_status", "") for row in open_yes)),
        "strict_version_confirmed_count": sum(
            (row.get("strict_version_confirmed") or "").strip().casefold() == "true"
            for row in open_yes
        ),
        "no_license_terms_interpreted": True,
    },
}
manifest_path = OUT / "q4_evidence_gap_reconciliation_manifest.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"bridge_rows": len(bridge_summary), "license_groups": len(license_summary), "open_yes": len(open_yes), "manifest": str(manifest_path)}, ensure_ascii=False))
