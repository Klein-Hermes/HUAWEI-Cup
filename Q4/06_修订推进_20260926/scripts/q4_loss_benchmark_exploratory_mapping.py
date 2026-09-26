#!/usr/bin/env python3
"""Build a provenance-checked, within-Pythia exploratory Loss–Benchmark map.

This is a supplemental limited-exploration branch. It does not overwrite the
strict v0.6 BLOCKED_NO_FIT audit or feed the separate C1 frontier forecast.
Only Python's standard library is required.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "Q4" / "06_修订推进_20260926"
OUT = PACKAGE / "results" / "bridge_exploratory"

PATHS = {
    "b1_training_log": ROOT / "中文题目" / "F题" / "real_attachments" / "B_scaling_laws" / "pythia_training_log_existing.csv",
    "c1_leaderboard": ROOT / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution" / "leaderboard_cleaned.csv",
    "c5_bridge": ROOT / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution" / "loss_benchmark_bridge.csv",
    "c6_bridge": ROOT / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution" / "loss_benchmark_bridge_expanded.csv",
    "q4_pair_audit": ROOT / "Q4" / "03_结果" / "results" / "q4_loss_bridge_pair_audit.csv",
    "q4_target_audit": ROOT / "Q4" / "03_结果" / "results" / "q4_loss_bridge_target_audit.csv",
    "v0_9_contract": PACKAGE / "q4_loss_bridge_exploratory_contract_v0.9.md",
}

TASK_TO_C1 = {
    "IFEval": "IFEval",
    "BBH": "BBH",
    "MATH Lvl 5": "MATH Lvl 5",
    "GPQA": "GPQA",
    "MUSR": "MUSR",
    "MMLU-PRO": "MMLU-PRO",
}
TASK_TO_BRIDGE = {
    "IFEval": "LB_IFEval",
    "BBH": "LB_BBH",
    "MATH Lvl 5": "LB_MATH",
    "GPQA": "LB_GPQA",
    "MUSR": "LB_MUSR",
    "MMLU-PRO": "LB_MMLU_PRO",
}
BBH_CHOICE_FALLBACK = {
    "boolean_expressions": 2,
    "causal_judgement": 2,
    "date_understanding": 6,
    "disambiguation_qa": 3,
    "formal_fallacies": 2,
    "geometric_shapes": 11,
    "hyperbaton": 2,
    "logical_deduction_five_objects": 5,
    "logical_deduction_seven_objects": 7,
    "logical_deduction_three_objects": 3,
    "movie_recommendation": 6,
    "navigate": 2,
    "object_counting": 19,
    "penguins_in_a_table": 5,
    "reasoning_about_colored_objects": 18,
    "ruin_names": 6,
    "salient_translation_error_detection": 6,
    "snarks": 2,
    "sports_understanding": 2,
    "temporal_sequences": 4,
    "tracking_shuffled_objects_five_objects": 5,
    "tracking_shuffled_objects_seven_objects": 7,
    "tracking_shuffled_objects_three_objects": 3,
    "web_of_lies": 2,
}
MUSR_CHOICE_COUNTS = {
    "leaderboard_musr_murder_mysteries": 2,
    "leaderboard_musr_object_placements": 5,
    "leaderboard_musr_team_allocation": 3,
}
OBSERVATION_TOL = 1e-10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def displayed_precision(value: str) -> float:
    """Half-width of the rounding interval implied by a displayed decimal."""
    text = str(value).strip()
    decimals = len(text.partition(".")[2]) if "." in text else 0
    return 0.5 * (10.0 ** -decimals)


def exp_loss_ppl_rounding_intervals_overlap(loss_text: str, ppl_text: str) -> bool:
    loss = float(loss_text)
    ppl = float(ppl_text)
    loss_half_width = displayed_precision(loss_text)
    ppl_half_width = displayed_precision(ppl_text)
    exp_low = math.exp(loss - loss_half_width)
    exp_high = math.exp(loss + loss_half_width)
    ppl_low = max(0.0, ppl - ppl_half_width)
    ppl_high = ppl + ppl_half_width
    return max(exp_low, ppl_low) <= min(exp_high, ppl_high)


def norm_chance(raw: float, choices: int) -> float:
    lower = 1.0 / choices
    return max(0.0, (raw - lower) / (1.0 - lower)) * 100.0


def detail_scores(payload: dict[str, Any]) -> tuple[dict[str, float], int]:
    results = payload["results"]
    configs = payload.get("configs", {})

    ifeval = results["leaderboard_ifeval"]
    scores: dict[str, float] = {
        "IFEval": 50.0
        * (
            float(ifeval["inst_level_strict_acc,none"])
            + float(ifeval["prompt_level_strict_acc,none"])
        )
    }

    bbh_parts: list[float] = []
    for task_name, result in results.items():
        if not task_name.startswith("leaderboard_bbh_"):
            continue
        suffix = task_name.removeprefix("leaderboard_bbh_")
        config = configs.get(task_name, {})
        choices = config.get("doc_to_choice")
        n_choices = len(choices) if isinstance(choices, list) else BBH_CHOICE_FALLBACK.get(suffix)
        if not n_choices or "acc_norm,none" not in result:
            raise ValueError(f"missing choice count or BBH metric for {task_name}")
        bbh_parts.append(norm_chance(float(result["acc_norm,none"]), int(n_choices)))
    if len(bbh_parts) != 24:
        raise ValueError(f"expected 24 BBH leaf results, got {len(bbh_parts)}")
    scores["BBH"] = statistics.mean(bbh_parts)

    # The C1 target is its published MATH Lvl 5 field. This diagnostic uses
    # the matching parent task metric from the detail JSON; known differences
    # are preserved as version/provenance warnings, never silently corrected.
    scores["MATH Lvl 5"] = 100.0 * float(
        results["leaderboard_math_hard"]["exact_match,none"]
    )
    scores["GPQA"] = norm_chance(
        float(results["leaderboard_gpqa"]["acc_norm,none"]), 4
    )
    musr_parts = [
        norm_chance(float(results[task]["acc_norm,none"]), n_choices)
        for task, n_choices in MUSR_CHOICE_COUNTS.items()
    ]
    scores["MUSR"] = statistics.mean(musr_parts)
    scores["MMLU-PRO"] = norm_chance(
        float(results["leaderboard_mmlu_pro"]["acc,none"]), 10
    )
    return scores, len(bbh_parts)


def ols(x: list[float], y: list[float]) -> tuple[float, float, float, float, float]:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("OLS requires at least three paired observations")
    mx, my = statistics.mean(x), statistics.mean(y)
    sxx = sum((value - mx) ** 2 for value in x)
    syy = sum((value - my) ** 2 for value in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    if sxx <= 0 or syy <= 0:
        raise ValueError("OLS design or response has zero variance")
    slope = sxy / sxx
    intercept = my - slope * mx
    pearson = sxy / math.sqrt(sxx * syy)
    fitted = [intercept + slope * value for value in x]
    rmse = math.sqrt(statistics.mean((actual - pred) ** 2 for actual, pred in zip(y, fitted)))
    return intercept, slope, pearson, pearson * pearson, rmse


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, path in PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")

    b1_rows = read_csv(PATHS["b1_training_log"])
    c1_rows: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(PATHS["c1_leaderboard"]):
        c1_rows.setdefault(row["Model"], []).append(row)
    c5_rows = read_csv(PATHS["c5_bridge"])
    c6_rows = read_csv(PATHS["c6_bridge"])
    c5_by_model = {row["Model"]: row for row in c5_rows}
    c6_by_model = {row["Model"]: row for row in c6_rows}
    if len(c5_by_model) != len(c5_rows) or len(c6_by_model) != len(c6_rows):
        raise ValueError("C5/C6 bridge tables must have unique Model keys for the source crosswalk")
    for model, row in c5_by_model.items():
        if c6_by_model.get(model) != row:
            raise ValueError(f"C6 expanded bridge row disagrees with C5 for {model}")
    raw_bridge_rows = c5_rows
    pair_rows = read_csv(PATHS["q4_pair_audit"])
    target_rows = read_csv(PATHS["q4_target_audit"])

    candidates = [
        row
        for row in pair_rows
        if row["source_table"] == "C5"
        and row["Loss_Comparability"].lower().startswith("high")
        and truth(row["kept_in_pooled_unique_table"])
    ]
    if len(candidates) != 7:
        raise ValueError(f"expected 7 deduplicated high-comparability C5 rows, got {len(candidates)}")

    raw_high_rows = [
        row for row in raw_bridge_rows
        if row["Loss_Comparability"].lower().startswith("high")
    ]
    candidate_by_model = {row["Model"]: row for row in candidates}
    raw_by_model = {row["Model"]: row for row in raw_high_rows}
    if len(raw_high_rows) != 7 or set(raw_by_model) != set(candidate_by_model):
        raise ValueError("raw C5 high-comparability rows do not match the 7 audited Pythia candidates")
    for model, candidate in candidate_by_model.items():
        raw = raw_by_model[model]
        for field in ("N_params_B", "D_tokens_B", "Val_Loss", "LB_Average"):
            if abs(float(raw[field]) - float(candidate[field])) > OBSERVATION_TOL:
                raise ValueError(f"raw bridge {field} disagrees with Q4 pair audit for {model}")

    crosswalk: list[dict[str, Any]] = []
    fit_points: list[dict[str, Any]] = []
    detail_json_paths: list[Path] = []
    for bridge in candidates:
        model = bridge["Model"]
        n_params = float(bridge["N_params_B"])
        d_tokens = float(bridge["D_tokens_B"])
        loss = float(bridge["Val_Loss"])
        c1_matches = c1_rows.get(model, [])
        if len(c1_matches) != 1:
            raise ValueError(f"expected one exact C1 leaderboard row for {model}; got {len(c1_matches)}")
        c1 = c1_matches[0]
        c1_n_params_raw = c1["#Params (B)"].strip()
        if not c1_n_params_raw:
            raise ValueError(f"C1 parameter count missing for {model}")
        c1_n_params = float(c1_n_params_raw)
        params_abs_diff = c1_n_params - n_params
        params_rel_diff_pct = params_abs_diff / n_params * 100.0

        log_matches = [
            row
            for row in b1_rows
            if abs(float(row["N_params_B"]) - n_params) <= 1e-6
            and abs(float(row["D_tokens_B"]) - d_tokens) <= 1e-6
            and abs(float(row["val_loss"]) - loss) <= 1e-9
        ]
        if len(log_matches) != 1:
            raise ValueError(f"expected one exact B1 terminal loss row for {model}; got {len(log_matches)}")
        b1 = log_matches[0]
        if int(float(b1["steps"])) != 143000 or abs(d_tokens - 299.893) > 1e-6:
            raise ValueError(f"unexpected final checkpoint metadata for {model}")
        ppl_delta = abs(math.exp(loss) - float(b1["ppl"]))
        ppl_rounding_match = exp_loss_ppl_rounding_intervals_overlap(b1["val_loss"], b1["ppl"])
        if not ppl_rounding_match:
            raise ValueError(f"displayed val_loss and PPL rounding intervals do not overlap for {model}")

        c1_scores = {name: float(c1[field]) for name, field in TASK_TO_C1.items()}
        c1_average = float(c1["Average ⬆️"])
        arithmetic_average = statistics.mean(c1_scores.values())
        if abs(c1_average - arithmetic_average) > OBSERVATION_TOL:
            raise ValueError(f"C1 Average is not the six-score arithmetic mean for {model}")
        for task, field in TASK_TO_BRIDGE.items():
            if abs(float(bridge[field]) - c1_scores[task]) > OBSERVATION_TOL:
                raise ValueError(f"C5 {field} does not equal C1 {task} for {model}")
        if abs(float(bridge["LB_Average"]) - c1_average) > OBSERVATION_TOL:
            raise ValueError(f"C5 LB_Average does not equal C1 Average for {model}")

        slug = model.replace("/", "_")
        detail_files = sorted(
            (ROOT / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution" / "detailed_results").glob(
                f"{slug}/results_*.json"
            )
        )
        if not detail_files:
            raise FileNotFoundError(f"no detailed evaluation JSON found for {model}")
        scored_details: list[tuple[int, float, Path, dict[str, Any], dict[str, float], int]] = []
        for detail_path in detail_files:
            payload = json.loads(detail_path.read_text(encoding="utf-8"))
            scores, bbh_count = detail_scores(payload)
            exact_count = sum(
                abs(scores[task] - c1_scores[task]) <= OBSERVATION_TOL
                for task in TASK_TO_C1
            )
            max_abs = max(abs(scores[task] - c1_scores[task]) for task in TASK_TO_C1)
            scored_details.append((exact_count, max_abs, detail_path, payload, scores, bbh_count))
        scored_details.sort(key=lambda item: (-item[0], item[1], item[2].name))
        exact_count, max_abs, detail_path, payload, detail_scores_by_task, bbh_count = scored_details[0]
        detail_json_paths.extend(detail[2] for detail in scored_details)
        metric_diffs = {
            task: detail_scores_by_task[task] - c1_scores[task]
            for task in TASK_TO_C1
        }
        mismatch_fields = [task for task, delta in metric_diffs.items() if abs(delta) > OBSERVATION_TOL]
        detail_average = statistics.mean(detail_scores_by_task.values())
        eval_date = dt.datetime.fromtimestamp(float(payload["date"]), tz=dt.timezone.utc).date().isoformat()
        submission_date = c1["Submission Date"]
        score_status = "exact_all_six" if not mismatch_fields else "partial_match_metric_difference"

        crosswalk.append(
            {
                "Model": model,
                "N_params_B": n_params,
                "C1_exact_model_row_count": len(c1_matches),
                "C1_params_B_raw": c1_n_params_raw,
                "C1_params_B": c1_n_params,
                "C1_minus_B1_params_B": params_abs_diff,
                "C1_vs_B1_params_relative_diff_pct": params_rel_diff_pct,
                "parameter_count_alignment_status": "equal_within_tolerance" if abs(params_abs_diff) <= OBSERVATION_TOL else "metadata_difference_unresolved",
                "D_tokens_B": d_tokens,
                "B1_steps": int(float(b1["steps"])),
                "B1_val_loss": loss,
                "B1_ppl": float(b1["ppl"]),
                "abs_exp_loss_minus_ppl": ppl_delta,
                "exp_val_loss_rounded_2dp_matches_ppl": ppl_rounding_match,
                "C1_submission_date": submission_date,
                "C1_Average": c1_average,
                "detail_derived_six_score_average": detail_average,
                "C1_minus_detail_derived_average": c1_average - detail_average,
                "C1_C6_arithmetic_mean_verified": True,
                "C5_C6_high_label": bridge["Loss_Comparability"],
                "detail_json": detail_path.relative_to(ROOT).as_posix(),
                "detail_run_date_utc": eval_date,
                "detail_model_revision": payload.get("config", {}).get("model_revision", ""),
                "detail_lm_eval_git": payload.get("git_hash", ""),
                "detail_transformers": payload.get("transformers_version", ""),
                "detail_task_hash_count": len(payload.get("task_hashes", {})),
                "detail_BBH_leaf_count": bbh_count,
                "detail_exact_C1_metrics_n": exact_count,
                "detail_mismatched_C1_metrics": ";".join(mismatch_fields),
                "detail_max_abs_metric_difference": max_abs,
                "detail_metric_differences_json": json.dumps(metric_diffs, ensure_ascii=False, sort_keys=True),
                "version_link_status": score_status,
                "mapping_scope": "single Pythia family; exploratory only",
            }
        )
        fit_points.append(
            {
                "Model": model,
                "N_params_B": n_params,
                "D_tokens_B": d_tokens,
                "Val_Loss": loss,
                "LB_Average": c1_average,
                "detail_derived_six_score_average": detail_average,
                "detail_match_n": exact_count,
                "detail_mismatch_fields": ";".join(mismatch_fields),
            }
        )

    x = [float(row["Val_Loss"]) for row in fit_points]
    y = [float(row["LB_Average"]) for row in fit_points]
    intercept, slope, pearson, r_squared, rmse = ols(x, y)
    residuals = [actual - (intercept + slope * loss) for loss, actual in zip(x, y)]

    mapping_rows: list[dict[str, Any]] = []
    loo_errors: list[float] = []
    baseline_errors: list[float] = []
    for idx, point in enumerate(fit_points):
        train_x = x[:idx] + x[idx + 1 :]
        train_y = y[:idx] + y[idx + 1 :]
        loo_intercept, loo_slope, _, _, _ = ols(train_x, train_y)
        loo_prediction = loo_intercept + loo_slope * x[idx]
        loo_error = y[idx] - loo_prediction
        baseline_prediction = statistics.mean(train_y)
        baseline_error = y[idx] - baseline_prediction
        loo_errors.append(loo_error)
        baseline_errors.append(baseline_error)
        mapping_rows.append(
            {
                **point,
                "in_sample_fitted_LB_Average": intercept + slope * x[idx],
                "in_sample_residual": residuals[idx],
                "leave_one_size_out_prediction": loo_prediction,
                "leave_one_size_out_error": loo_error,
                "leave_one_size_out_mean_baseline": baseline_prediction,
                "leave_one_size_out_mean_baseline_error": baseline_error,
                "prediction_status": "within_sample_only; not a transferable calibration",
            }
        )

    loo_mae = statistics.mean(abs(value) for value in loo_errors)
    baseline_mae = statistics.mean(abs(value) for value in baseline_errors)
    loo_rmse = math.sqrt(statistics.mean(value * value for value in loo_errors))
    baseline_rmse = math.sqrt(statistics.mean(value * value for value in baseline_errors))

    def sensitivity_summary(name: str, points: list[dict[str, Any]], outcome_key: str, omitted: list[str]) -> dict[str, Any]:
        sx = [float(point["Val_Loss"]) for point in points]
        sy = [float(point[outcome_key]) for point in points]
        si, ss, sr, sr2, srmse = ols(sx, sy)
        heldout_errors: list[float] = []
        mean_errors: list[float] = []
        for idx in range(len(points)):
            train_x = sx[:idx] + sx[idx + 1 :]
            train_y = sy[:idx] + sy[idx + 1 :]
            loo_i, loo_s, _, _, _ = ols(train_x, train_y)
            heldout_errors.append(sy[idx] - (loo_i + loo_s * sx[idx]))
            mean_errors.append(sy[idx] - statistics.mean(train_y))
        return {
            "sensitivity_scenario": name,
            "outcome_source": outcome_key,
            "n_models": len(points),
            "omitted_models": ";".join(omitted),
            "intercept": si,
            "slope_per_nats_token": ss,
            "pearson_r": sr,
            "r_squared": sr2,
            "in_sample_rmse": srmse,
            "leave_one_size_out_mae": statistics.mean(abs(value) for value in heldout_errors),
            "leave_one_size_out_mean_baseline_mae": statistics.mean(abs(value) for value in mean_errors),
            "leave_one_size_out_rmse": math.sqrt(statistics.mean(value * value for value in heldout_errors)),
            "leave_one_size_out_mean_baseline_rmse": math.sqrt(statistics.mean(value * value for value in mean_errors)),
        }

    mismatch_models = [point["Model"] for point in fit_points if point["detail_mismatch_fields"]]
    exact_detail_points = [point for point in fit_points if not point["detail_mismatch_fields"]]
    sensitivity_rows = [
        sensitivity_summary("primary_all_7_C1_scores", fit_points, "LB_Average", []),
        sensitivity_summary("alternative_all_7_detail_JSON_scores", fit_points, "detail_derived_six_score_average", []),
        sensitivity_summary("exact_all_six_detail_subset", exact_detail_points, "LB_Average", mismatch_models),
    ]

    target_support: list[dict[str, Any]] = []
    loss_min, loss_max = min(x), max(x)
    sample_models = {point["Model"] for point in fit_points}
    for target in target_rows:
        target_loss = float(target["target_loss"])
        inside = loss_min - OBSERVATION_TOL <= target_loss <= loss_max + OBSERVATION_TOL
        n_params = float(target["N_params_B"])
        model = next(
            (point["Model"] for point in fit_points if abs(point["N_params_B"] - n_params) <= 1e-6),
            "",
        )
        target_support.append(
            {
                "target_source": target["target_source"],
                "target_scope": target["target_scope"],
                "target_record_id": target["target_record_id"],
                "target_role_original": target["target_role"],
                "N_params_B": n_params,
                "D_tokens_B": float(target["D_tokens_B"]),
                "target_loss": target_loss,
                "unit_from_v0_9_audit": "nats/token; PPL=exp(loss) checked for Pythia training log",
                "support_min_loss": loss_min,
                "support_max_loss": loss_max,
                "within_loss_support": inside,
                "matched_Pythia_calibration_model": model,
                "overlaps_calibration_model_name_and_size": model in sample_models,
                "conditional_in_sample_point_mapping": intercept + slope * target_loss if inside else "",
                "prediction_status": "conditional_within_family_in_sample_overlap" if inside and model else (
                    "conditional_within_family_only" if inside else "out_of_support_no_mapping"
                ),
            }
        )

    fit_summary = [
        {
            "status": "ELIGIBLE_LIMITED_EXPLORATORY_NOT_CALIBRATED",
            "n_unique_models": len(fit_points),
            "n_independent_model_families": 1,
            "loss_min": loss_min,
            "loss_max": loss_max,
            "intercept": intercept,
            "slope_per_nats_token": slope,
            "pearson_r": pearson,
            "r_squared": r_squared,
            "in_sample_rmse": rmse,
            "leave_one_size_out_mae": loo_mae,
            "leave_one_size_out_mean_baseline_mae": baseline_mae,
            "leave_one_size_out_rmse": loo_rmse,
            "leave_one_size_out_mean_baseline_rmse": baseline_rmse,
            "leave_one_size_out_beats_mean_baseline": loo_mae <= baseline_mae,
            "uncertainty_status": "no transferable CI; one-family leave-one-size-out errors are sensitivity only",
        }
    ]

    crosswalk_path = OUT / "q4_loss_benchmark_pythia_crosswalk.csv"
    mapping_path = OUT / "q4_loss_benchmark_mapping_fit.csv"
    support_path = OUT / "q4_loss_benchmark_target_support.csv"
    summary_path = OUT / "q4_loss_benchmark_fit_summary.csv"
    sensitivity_path = OUT / "q4_loss_benchmark_sensitivity.csv"
    report_path = OUT / "q4_loss_benchmark_exploratory_report.md"
    manifest_path = OUT / "q4_loss_benchmark_exploratory_manifest.json"

    write_csv(crosswalk_path, crosswalk)
    write_csv(mapping_path, mapping_rows)
    write_csv(support_path, target_support)
    write_csv(summary_path, fit_summary)
    write_csv(sensitivity_path, sensitivity_rows)

    in_support_n = sum(truth(row["within_loss_support"]) for row in target_support)
    out_support_n = len(target_support) - in_support_n
    report = f"""# Q4 Loss–Benchmark 单家族探索映射

**状态：** `ELIGIBLE_LIMITED_EXPLORATORY_NOT_CALIBRATED`。本结果补充旧版 `BLOCKED_NO_FIT` 门控，但不覆盖该历史判定，也不支持跨家族、跨协议或未来能力的校准。

## 证据与样本

- 主样本为 C5/C6 去重后的 {len(fit_points)} 个 Pythia 参数规模，来自 1 个模型家族；每条 Loss 均匹配 B1 最终 `steps=143000`、`D=299.893B` 的实测终点。考虑 B1 展示的 `val_loss`（3–4 位小数）和 `PPL`（2 位小数）的舍入区间后，逐行的 `exp(val_loss)` 与 `PPL` 区间相容。
- C5/C6 的七条 `LB_Average` 和逐项分数与 C1 对应排行榜行一致；C1 `Average` 等于六个 Benchmark 字段的算术平均。目标是 Open LLM Leaderboard v2 六项归一化分数的平均。
- 对照本地详细评测 JSON 的版本诊断中，{len(fit_points) - len(mismatch_models)}/{len(fit_points)} 行六项都与 C1 精确相符；差异模型为 `{', '.join(mismatch_models) if mismatch_models else '无'}`。差异字段保存在逐行 crosswalk，原始 C1/C5/C6 分数保留为拟合目标，不用 JSON 替换。
- C1 精确 Model 均唯一，但其 `#Params(B)` 与 B1/桥接表参数量有差异；绝对差与相对差逐行列于 crosswalk。这不改变主回归使用 C1 分数的规则，但表示名称连接不能证明 leaderboard checkpoint 与训练日志参数元数据完全一致。
- Loss 支持范围：`[{loss_min:.4f}, {loss_max:.4f}]` nats/token。Pythia 单家族样本不能识别跨家族迁移，也不能分离模型规模与 Loss 的相关变化。

## 线性探索关系

**LB_Average = {intercept:.6f} {slope:+.6f} × Val_Loss**

Pearson `r={pearson:.3f}`，`R²={r_squared:.3f}`，样本内 RMSE `{rmse:.3f}`。该关系较弱，且不能解释为降低 Loss 会因果提高 Benchmark。

按模型尺寸逐一留出，LOSO MAE 为 `{loo_mae:.3f}`，留出训练集均值基线 MAE 为 `{baseline_mae:.3f}`；LOSO RMSE 为 `{loo_rmse:.3f}`，均值基线 RMSE 为 `{baseline_rmse:.3f}`。线性映射**{ '未' if loo_mae > baseline_mae else '' }优于**留一均值基线。七个留一残差见拟合表；它们是单一家族敏感性，不是跨家族验证或校准误差。

版本差异敏感性：仅保留评测详情 JSON 六项均与 C1 精确相符的 5 行时，斜率为 `{sensitivity_rows[2]['slope_per_nats_token']:.6f}`、R² `{sensitivity_rows[2]['r_squared']:.3f}`；若仅作替代敏感性、用所选详情 JSON 的六项均值替代两条不一致行的 C1 `Average`，7 行斜率为 `{sensitivity_rows[1]['slope_per_nats_token']:.6f}`、R² `{sensitivity_rows[1]['r_squared']:.3f}`。主结果始终保留 C1 值；该敏感性说明分数版本差异会改变关系，不能用来裁定哪一版本“正确”。各方案的样本内及留一基线诊断见 sensitivity CSV。

## Q2 候选目标支持

Q2 B1 观察终点和 M0 拟合终点共 {len(target_support)} 个候选目标：{in_support_n} 个落在 Pythia Loss 范围内，{out_support_n} 个超出范围。支持内分数仅作条件点映射；标记为模型名称/规模重叠的记录是样本内演示。超出范围的记录留空，不外推。表内不提供预测区间或桥接校准误差。

## 使用限制

- 本结果不替换 C1 直接分数的 Task 3 前沿情景，不向其他模型家族迁移。
- Open LLM 详细结果 JSON 提供 revision、task hash、Harness commit、Transformers 版本与运行日期；C1 清洗表没有同一行的 revision/run ID。局部 MATH 差异显示，不能把“名称相同”升级为每个记录的严格 checkpoint 证明。
- 没有足够独立家族估计跨家族不确定性；因此不报告置信区间、校准区间或 `SUPPORTED_FOR_CALIBRATION`。

## 复现

从仓库根目录运行：`python Q4/06_修订推进_20260926/scripts/q4_loss_benchmark_exploratory_mapping.py`。输入/输出/代码/合同 SHA-256 见 manifest；逐行来源、参数量和版本差异见 crosswalk；LOSO 误差见 mapping fit；版本敏感性见 sensitivity；Q2 目标支持见 target support。
"""
    report_path.write_text(report, encoding="utf-8")

    detail_inputs = sorted(set(detail_json_paths))
    input_paths = dict(PATHS)
    input_paths.update({f"pythia_detail_{i+1}": path for i, path in enumerate(detail_inputs)})
    manifest = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "contract_sha256": sha256(PATHS["v0_9_contract"]),
        "status": "ELIGIBLE_LIMITED_EXPLORATORY_NOT_CALIBRATED",
        "strict_v0_6_historical_status": "BLOCKED_NO_FIT_preserved",
        "inputs": {name: sha256(path) for name, path in input_paths.items()},
        "fit": fit_summary[0],
        "version_audit": {
            "detail_json_records": len(crosswalk),
            "exact_all_six_score_matches": sum(row["version_link_status"] == "exact_all_six" for row in crosswalk),
            "rows_with_metric_discrepancy": mismatch_models,
        },
        "bridge_table_overlap_audit": {
            "c5_row_count": len(c5_rows),
            "c6_row_count": len(c6_rows),
            "c5_exact_rows_present_in_c6": len(c5_by_model),
            "c6_additional_row_count": len(c6_rows) - len(c5_rows),
            "all_c5_rows_exactly_match_c6_by_model": True,
        },
        "outputs": {
            path.name: sha256(path)
            for path in [crosswalk_path, mapping_path, support_path, summary_path, sensitivity_path, report_path]
        },
        "scope": "within-Pythia-family exploratory relation; no calibration or Task 3 integration",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"status": manifest["status"], "fit": fit_summary[0], "outputs": manifest["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
