"""Audit non-C8 source coverage and time-axis support for Q4.

AI assistance: OpenAI Codex desktop 26.917.71314 (build 10954, prod), model ID
gpt-6-luna (GPT-6 Luna), developer OpenAI, official release date 2026-09-22.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


Q4 = Path(__file__).resolve().parents[2]
ROOT = Q4.parent
DATA = ROOT / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution"
RESULTS = Q4 / "03_结果" / "results"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def month(value: str) -> str:
    value = (value or "").strip()
    if len(value) >= 7 and value[4:5] == "-":
        return value[:7]
    return ""


def _is_finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_positive_number(value: str) -> bool:
    try:
        number = float(value)
        return math.isfinite(number) and number > 0
    except (TypeError, ValueError):
        return False


def main() -> None:
    candidate_rows = read_csv(RESULTS / "q4_c8_c1_match_candidates.csv")
    candidate_names = {
        (row.get("c8_model_name_raw") or "").strip()
        for row in candidate_rows
        if row.get("match_status") == "exact_model_id_unique_record"
    }
    c1_rows = read_csv(DATA / "leaderboard_cleaned.csv")
    c1_names = {(row.get("Model") or "").strip() for row in c1_rows if row.get("Model")}

    specs = [
        ("C3", "leaderboard_extended_timeseries.csv", "Model"),
        ("C4", "epoch_all_ai_models.csv", "Model"),
        ("C5", "loss_benchmark_bridge.csv", "Model"),
        ("C6", "loss_benchmark_bridge_expanded.csv", "Model"),
        ("C7", "model_architecture_metadata.csv", "model_name"),
    ]
    source_rows: list[dict[str, Any]] = []
    loaded: dict[str, list[dict[str, str]]] = {}
    for source_id, filename, key in specs:
        path = DATA / filename
        rows = read_csv(path)
        loaded[source_id] = rows
        ids = {(row.get(key) or "").strip() for row in rows if row.get(key)}
        overlap = candidate_names & ids
        source_rows.append({
            "source_id": source_id,
            "file": filename,
            "row_count": len(rows),
            "join_field": key,
            "unique_nonblank_ids": len(ids),
            "exact_overlap_with_1819_unique_name_candidates": len(overlap),
            "exact_overlap_with_c1_name_universe": len(c1_names & ids),
            "interpretation": "exact name intersection only; not model-version confirmation",
        })
    write_csv(RESULTS / "q4_auxiliary_source_coverage.csv", list(source_rows[0].keys()), source_rows)

    c3 = loaded["C3"]
    c3_year_rows: list[dict[str, Any]] = []
    for source in sorted({(r.get("Source") or "").strip() for r in c3}):
        group = [r for r in c3 if (r.get("Source") or "").strip() == source]
        for year in sorted({(r.get("Year") or "").strip() for r in group}):
            yr = [r for r in group if (r.get("Year") or "").strip() == year]
            names = {(r.get("Model") or "").strip() for r in yr if r.get("Model")}
            c3_year_rows.append({
                "source": source, "year": year, "rows": len(yr), "unique_model_names": len(names),
                "duplicate_rows": len(yr) - len(names),
                "duplicated_model_names": sum(count > 1 for count in Counter((r.get("Model") or "").strip() for r in yr).values()),
                "unique_name_candidates_present": len(candidate_names & names),
                "is_open_llm_leaderboard_source": source == "Open LLM Leaderboard",
            })
    write_csv(RESULTS / "q4_c3_year_source_coverage.csv", list(c3_year_rows[0].keys()), c3_year_rows)

    bridge_rows: list[dict[str, Any]] = []
    for source_id in ("C5", "C6"):
        rows = loaded[source_id]
        comparability = Counter((row.get("Loss_Comparability") or "(blank)").strip() for row in rows)
        bridge_rows.append({
            "source_id": source_id,
            "rows": len(rows),
            "candidate_model_name_overlap": len(candidate_names & {(r.get("Model") or "").strip() for r in rows}),
            "loss_nonblank_rows": sum(bool((r.get("Val_Loss") or "").strip()) for r in rows),
            "loss_numeric_rows": sum(_is_finite_number(r.get("Val_Loss", "")) for r in rows),
            "D_tokens_nonblank_rows": sum(bool((r.get("D_tokens_B") or "").strip()) for r in rows),
            "D_tokens_positive_rows": sum(_is_positive_number(r.get("D_tokens_B", "")) for r in rows),
            "comparability_counts_json": json.dumps(comparability, ensure_ascii=False),
        })
    write_csv(RESULTS / "q4_loss_bridge_quality.csv", list(bridge_rows[0].keys()), bridge_rows)

    monthly_c1: Counter[str] = Counter()
    monthly_c1_names: dict[str, set[str]] = defaultdict(set)
    monthly_c1_params: Counter[str] = Counter()
    c1_missing_date_rows = 0
    c1_missing_date_names: set[str] = set()
    c1_missing_date_params = 0
    for row in c1_rows:
        m = month(row.get("Submission Date", ""))
        if not m:
            c1_missing_date_rows += 1
            if (row.get("Model") or "").strip():
                c1_missing_date_names.add(row["Model"].strip())
            c1_missing_date_params += bool((row.get("#Params (B)") or "").strip())
            continue
        monthly_c1[m] += 1
        monthly_c1_names[m].add((row.get("Model") or "").strip())
        monthly_c1_params[m] += bool((row.get("#Params (B)") or "").strip())

    run_rows = read_csv(RESULTS / "q4_c8_run_registry.csv")
    monthly_c8: Counter[str] = Counter()
    monthly_c8_names: dict[str, set[str]] = defaultdict(set)
    for row in run_rows:
        if row.get("selected_as_latest_parseable") != "True":
            continue
        m = month(row.get("file_timestamp", ""))
        if m:
            monthly_c8[m] += 1
            monthly_c8_names[m].add((row.get("model_name_raw") or "").strip())
    time_rows = []
    for m in sorted(set(monthly_c1) | set(monthly_c8)):
        time_rows.append({
            "month": m,
            "c1_submission_rows": monthly_c1.get(m, 0),
            "c1_unique_model_names": len(monthly_c1_names.get(m, set())),
            "c1_rows_with_parameter_count": monthly_c1_params.get(m, 0),
            "c8_latest_runs_by_file_month": monthly_c8.get(m, 0),
            "c8_unique_model_names_by_file_month": len(monthly_c8_names.get(m, set())),
            "note": "counts only; no score aggregation or forecast",
        })
    if c1_missing_date_rows:
        time_rows.append({
            "month": "(missing_C1_Submission_Date)",
            "c1_submission_rows": c1_missing_date_rows,
            "c1_unique_model_names": len(c1_missing_date_names),
            "c1_rows_with_parameter_count": c1_missing_date_params,
            "c8_latest_runs_by_file_month": 0,
            "c8_unique_model_names_by_file_month": 0,
            "note": "C1 missing submission date; excluded from monthly bins",
        })
    write_csv(RESULTS / "q4_monthly_time_axis_coverage.csv", list(time_rows[0].keys()), time_rows)

    input_paths = [DATA / filename for _, filename, _ in specs]
    input_paths += [DATA / "leaderboard_cleaned.csv", RESULTS / "q4_c8_c1_match_candidates.csv",
                    RESULTS / "q4_c8_run_registry.csv"]
    output_paths = [
        RESULTS / "q4_auxiliary_source_coverage.csv",
        RESULTS / "q4_c3_year_source_coverage.csv",
        RESULTS / "q4_loss_bridge_quality.csv",
        RESULTS / "q4_monthly_time_axis_coverage.csv",
    ]
    manifest = {
        "ai_disclosure": "OpenAI Codex desktop 26.917.71314 (build 10954; prod); verified model ID gpt-6-luna (GPT-6 Luna); developer OpenAI; official release date 2026-09-22.",
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "candidate_unique_name_count": len(candidate_names),
        "inputs_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in input_paths},
        "outputs_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in output_paths},
        "limitations": [
            "Exact string intersections are candidate joins only, not version confirmation.",
            "Monthly tables contain record counts only, not score summaries or forecasts.",
            "C3 historical papers/reports remain separated from Open LLM Leaderboard rows.",
        ],
    }
    (RESULTS / "q4_auxiliary_audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "source_coverage": [{k: row[k] for k in (
            "source_id", "row_count", "exact_overlap_with_1819_unique_name_candidates"
        )} for row in source_rows],
        "bridge_quality": [{k: row[k] for k in (
            "source_id", "rows", "candidate_model_name_overlap", "D_tokens_nonblank_rows",
            "D_tokens_positive_rows"
        )} for row in bridge_rows],
        "monthly_rows": len(time_rows),
        "manifest": "Q4/03_结果/results/q4_auxiliary_audit_manifest.json",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
