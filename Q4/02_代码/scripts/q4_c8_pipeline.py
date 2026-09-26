"""Audit C8 and build auditable run and metric tables for Q4.

AI assistance: OpenAI Codex desktop 26.917.71314 (build 10954, prod), model ID
gpt-6-luna (GPT-6 Luna), developer OpenAI, official release date 2026-09-22.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Iterable


Q4 = Path(__file__).resolve().parents[2]
ROOT = Q4.parent
DATA = ROOT / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution"
C8 = DATA / "detailed_results"
C1 = DATA / "leaderboard_cleaned.csv"
C2 = DATA / "leaderboard_enhanced.csv"
C4 = DATA / "epoch_all_ai_models.csv"
RESULTS = Q4 / "03_结果" / "results"
SMOKE = Q4 / "03_结果" / "smoke"
FILENAME_TIME = re.compile(r"^results_(.+)\.json$")

AI_NOTICE = (
    "AI辅助：OpenAI Codex 桌面版 26.917.71314（build 10954，prod）；"
    "模型 ID gpt-6-luna（官方名称 GPT-6 Luna）；开发机构 OpenAI；发布日期 2026-09-22。"
)

FILE_FIELDS = [
    "source_file", "model_dir", "file_name", "file_timestamp", "file_sha256",
    "parse_status", "parse_error", "top_level_keys_json", "schema_sha256",
    "task_count", "model_name_raw", "model_revision", "model_sha", "json_date",
    "selected_as_latest_parseable", "latest_status",
]
RUN_FIELDS = [
    "run_id", "source_file", "model_dir", "model_name_raw", "file_name",
    "file_timestamp", "file_sha256", "json_date", "model_revision", "model_sha",
    "version_identity_status", "version_identity", "task_count", "top_level_keys_json",
    "selected_as_latest_parseable", "latest_status",
]
LONG_FIELDS = [
    "observation_id", "run_id", "source_file", "file_sha256", "model_dir",
    "model_name_raw", "model_revision", "model_sha", "version_identity_status",
    "run_file_timestamp", "json_date", "task_key", "task_version", "task_role",
    "metric_key", "metric_component_raw", "metric_name", "metric_filter",
    "metric_role", "value_raw", "value_numeric", "value_status", "higher_is_better",
    "stderr_metric_key", "stderr_value_numeric", "stderr_pair_status", "n_original",
    "n_effective", "n_shot", "selected_as_latest_parseable", "latest_status",
]
MATCH_FIELDS = [
    "c8_model_name_raw", "c8_model_dir", "run_id", "c1_record_count",
    "c1_record_ids", "match_status", "version_match_status", "strict_version_confirmed",
    "c8_model_revision", "c8_model_sha", "c2_epoch_publication_date_candidate",
    "c2_epoch_organization_candidate", "c2_epoch_open_weights_candidate",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def filename_timestamp(path: Path) -> str:
    match = FILENAME_TIME.match(path.name)
    return match.group(1) if match else ""


def read_json_bytes(data: bytes) -> dict[str, Any]:
    obj = json.loads(data.decode("utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    return obj


def version_fields(obj: dict[str, Any]) -> tuple[str, str, str, str]:
    config = obj.get("config") if isinstance(obj.get("config"), dict) else {}
    revision = clean_scalar(config.get("model_revision"))
    model_sha = clean_scalar(config.get("model_sha"))
    if revision and model_sha:
        status = "revision_sha_equal" if revision == model_sha else "revision_sha_conflict"
        identity = revision if revision == model_sha else ""
    elif revision:
        status, identity = "revision_only", revision
    elif model_sha:
        status, identity = "sha_only", model_sha
    else:
        status, identity = "version_missing", ""
    return revision, model_sha, status, identity


def clean_scalar(value: Any) -> str:
    if value is None or isinstance(value, (dict, list)):
        return ""
    return str(value).strip()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_c1_c2_row(row: dict[str, str], fields: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for field in fields:
        raw = (row.get(field) or "").strip()
        if field in {"#Params (B)", "Average ⬆️", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"} and raw:
            try:
                raw = format(Decimal(raw).quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_EVEN), "f")
            except (InvalidOperation, ValueError):
                pass
        normalized.append(raw)
    return tuple(normalized)


NUMERIC_C1_FIELDS = {"#Params (B)", "Average ⬆️", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"}


def rows_match_at_precision(left: dict[str, str], right: dict[str, str], fields: list[str]) -> tuple[bool, float]:
    max_difference = 0.0
    for field in fields:
        a = (left.get(field) or "").strip()
        b = (right.get(field) or "").strip()
        if field not in NUMERIC_C1_FIELDS:
            if a != b:
                return False, max_difference
            continue
        if not a or not b:
            if a != b:
                return False, max_difference
            continue
        try:
            difference = abs(float(a) - float(b))
        except ValueError:
            if a != b:
                return False, max_difference
            continue
        max_difference = max(max_difference, difference)
        if difference > 5e-13:
            return False, max_difference
    return True, max_difference


def audit_c1_c2(out_dir: Path) -> dict[int, dict[str, Any]]:
    c1_rows = load_csv(C1)
    c2_rows = load_csv(C2)
    fields = list(c1_rows[0].keys())
    text_fields = [field for field in fields if field not in NUMERIC_C1_FIELDS]
    c1_by_text: dict[tuple[str, ...], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    c2_by_text: dict[tuple[str, ...], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for row_num, row in enumerate(c1_rows, start=2):
        c1_by_text[tuple((row.get(k) or "").strip() for k in text_fields)].append((row_num, row))
    for row_num, row in enumerate(c2_rows, start=2):
        c2_by_text[tuple((row.get(k) or "").strip() for k in text_fields)].append((row_num, row))

    c1_candidates: dict[int, list[tuple[int, dict[str, str], float]]] = {}
    c2_candidate_counts: Counter[int] = Counter()
    for text_key, c1_group in c1_by_text.items():
        c2_group = c2_by_text.get(text_key, [])
        for c1_row_num, c1_row in c1_group:
            candidates = []
            for c2_row_num, c2_row in c2_group:
                matches, max_difference = rows_match_at_precision(c1_row, c2_row, fields)
                if matches:
                    candidates.append((c2_row_num, c2_row, max_difference))
                    c2_candidate_counts[c2_row_num] += 1
            c1_candidates[c1_row_num] = candidates

    out_rows: list[dict[str, Any]] = []
    c2_for_c1: dict[int, dict[str, Any]] = {}
    for c1_row_num, c1_row in enumerate(c1_rows, start=2):
        candidates = c1_candidates.get(c1_row_num, [])
        one_to_one = len(candidates) == 1 and c2_candidate_counts[candidates[0][0]] == 1
        if one_to_one:
            c2_for_c1[c1_row_num] = candidates[0][1]
        sig = normalize_c1_c2_row(c1_row, fields)
        out_rows.append({
            "c1_normalized_row_sha256": sha256_bytes("\x1f".join(sig).encode("utf-8")),
            "c1_row_numbers": json_cell([c1_row_num]),
            "c2_row_numbers": json_cell([c[0] for c in candidates]),
            "c1_count": 1,
            "c2_count": len(candidates),
            "one_to_one_full_row_match_with_numeric_tolerance": one_to_one,
            "max_numeric_difference": candidates[0][2] if len(candidates) == 1 else "",
        })
    write_csv(out_dir / "q4_c1_c2_alignment.csv", [
        "c1_normalized_row_sha256", "c1_row_numbers", "c2_row_numbers", "c1_count", "c2_count",
        "one_to_one_full_row_match_with_numeric_tolerance", "max_numeric_difference",
    ], out_rows)
    summary = [{
        "c1_rows": len(c1_rows),
        "c2_rows": len(c2_rows),
        "common_base_fields": json_cell(fields),
        "c1_unique_signatures": len({normalize_c1_c2_row(r, fields) for r in c1_rows}),
        "c2_unique_signatures": len({normalize_c1_c2_row(r, fields) for r in c2_rows}),
        "one_to_one_full_row_match_with_numeric_tolerance_5e_13": len(c2_for_c1) == len(c1_rows) == len(c2_rows),
        "numeric_match_rule": "all text fields exact; numeric fields absolute difference <= 5e-13",
        "c1_header_contains_version_field": any("revision" in x.lower() or "sha" in x.lower() for x in fields),
    }]
    write_csv(out_dir / "q4_c1_c2_alignment_summary.csv", list(summary[0].keys()), summary)
    return c2_for_c1


def parse_and_audit(paths: list[Path], out_dir: Path, all_dirs: list[Path], scope: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    file_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    parseable_by_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
    file_rows_by_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
    schema_counter: Counter[str] = Counter()
    schema_keys: dict[str, list[str]] = {}

    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        model_dir = path.parent.name
        raw = path.read_bytes()
        digest = sha256_bytes(raw)
        ts = filename_timestamp(path)
        row: dict[str, Any] = {
            "source_file": rel, "model_dir": model_dir, "file_name": path.name,
            "file_timestamp": ts, "file_sha256": digest, "parse_status": "",
            "parse_error": "", "top_level_keys_json": "", "schema_sha256": "",
            "task_count": "", "model_name_raw": "", "model_revision": "",
            "model_sha": "", "json_date": "", "selected_as_latest_parseable": False,
            "latest_status": "",
        }
        try:
            obj = read_json_bytes(raw)
            keys = sorted(obj.keys())
            key_json = json_cell(keys)
            schema_hash = sha256_bytes(key_json.encode("utf-8"))
            results = obj.get("results") if isinstance(obj.get("results"), dict) else {}
            revision, model_sha, version_status, identity = version_fields(obj)
            row.update({
                "parse_status": "parseable", "top_level_keys_json": key_json,
                "schema_sha256": schema_hash, "task_count": len(results),
                "model_name_raw": clean_scalar(obj.get("model_name")) or model_dir,
                "model_revision": revision, "model_sha": model_sha,
                "json_date": clean_scalar(obj.get("date")),
            })
            schema_counter[schema_hash] += 1
            schema_keys[schema_hash] = keys
            parseable_by_dir[model_dir].append({"path": path, "obj": obj, "file_row": row,
                                                 "version_status": version_status, "identity": identity})
        except Exception as exc:  # invalid JSON and incompatible top-level data remain explicit
            row["parse_status"] = "parse_failed"
            row["parse_error"] = f"{type(exc).__name__}: {exc}"[:500]
        file_rows.append(row)
        file_rows_by_dir[model_dir].append(row)

    selected: dict[str, dict[str, Any]] = {}
    latest_status: dict[str, str] = {}
    for directory in all_dirs:
        model_dir = directory.name
        candidates = file_rows_by_dir.get(model_dir, [])
        if not candidates:
            latest_status[model_dir] = "no_json_files"
            continue
        newest_file = max(candidates, key=lambda r: (r["file_timestamp"], r["file_name"]))
        good = parseable_by_dir.get(model_dir, [])
        if not good:
            latest_status[model_dir] = "no_parseable_json"
            for row in candidates:
                row["latest_status"] = "no_parseable_json"
            continue
        chosen = max(good, key=lambda r: (r["file_row"]["file_timestamp"], r["file_row"]["file_name"]))
        selected[model_dir] = chosen
        status = "latest_parseable" if chosen["file_row"]["source_file"] == newest_file["source_file"] else "latest_parseable_fallback"
        latest_status[model_dir] = status
        chosen["file_row"]["selected_as_latest_parseable"] = True
        for row in candidates:
            row["latest_status"] = status if row["source_file"] == chosen["file_row"]["source_file"] else "not_selected"

    for model_dir, runs in parseable_by_dir.items():
        for run in runs:
            row = run["file_row"]
            rid = sha256_bytes((row["source_file"] + "|" + row["file_sha256"]).encode("utf-8"))
            run_rows.append({
                "run_id": rid, "source_file": row["source_file"], "model_dir": model_dir,
                "model_name_raw": row["model_name_raw"], "file_name": row["file_name"],
                "file_timestamp": row["file_timestamp"], "file_sha256": row["file_sha256"],
                "json_date": row["json_date"], "model_revision": row["model_revision"],
                "model_sha": row["model_sha"], "version_identity_status": run["version_status"],
                "version_identity": run["identity"], "task_count": row["task_count"],
                "top_level_keys_json": row["top_level_keys_json"],
                "selected_as_latest_parseable": row["selected_as_latest_parseable"],
                "latest_status": row["latest_status"],
                "_obj": run["obj"], "_path": run["path"],
            })

    write_csv(out_dir / "q4_c8_file_registry.csv", FILE_FIELDS, file_rows)
    write_csv(out_dir / "q4_c8_schema_summary.csv", ["schema_sha256", "file_count", "top_level_keys_json"], [
        {"schema_sha256": key, "file_count": count, "top_level_keys_json": json_cell(schema_keys[key])}
        for key, count in sorted(schema_counter.items(), key=lambda x: (-x[1], x[0]))
    ])
    public_runs = [{key: row.get(key, "") for key in RUN_FIELDS} for row in run_rows]
    write_csv(out_dir / "q4_c8_run_registry.csv", RUN_FIELDS, public_runs)
    # Keep selected-run information for downstream tables without storing JSON in the CSV registry.
    return file_rows, run_rows


def parse_metric_key(raw_key: str) -> tuple[str, str, bool, str]:
    if "," in raw_key:
        component, metric_filter = raw_key.rsplit(",", 1)
    else:
        component, metric_filter = raw_key, ""
    if not component.strip():
        return component, "", False, metric_filter
    is_stderr = component.endswith("_stderr")
    metric_name = component[:-7] if is_stderr else component
    role = "stderr" if is_stderr else ("alias_metadata" if component == "alias" else "score")
    return component, metric_name, is_stderr, metric_filter


def numeric_value(value: Any) -> tuple[str, str]:
    if value is None:
        return "", "missing_null"
    if isinstance(value, bool) or isinstance(value, (dict, list)):
        return "", "non_numeric"
    if isinstance(value, str) and value.strip().upper() in {"N/A", "NA", "NAN", "NULL", ""}:
        return "", "missing_na" if value.strip().upper() != "NULL" else "missing_null"
    try:
        result = float(value)
    except (TypeError, ValueError):
        return "", "non_numeric"
    if not math.isfinite(result):
        return "", "non_finite"
    return repr(result), "numeric"


def nested_value(mapping: Any, task_key: str, child_key: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    task_value = mapping.get(task_key)
    if isinstance(task_value, dict):
        if child_key == "":
            return task_value
        return task_value.get(child_key)
    return task_value if child_key == "" else None


def task_role(task_key: str, group_summary_keys: set[str], task_subtask_keys: set[str]) -> str:
    if task_key in group_summary_keys:
        return "group_summary"
    if task_key in task_subtask_keys:
        return "task_subtask"
    return "other_task"


def iter_long_rows(run: dict[str, Any], group_summary_keys: set[str], task_subtask_keys: set[str]) -> Iterable[dict[str, Any]]:
    obj = run["_obj"]
    results = obj.get("results") if isinstance(obj.get("results"), dict) else {}
    run_id = run["run_id"]
    for task_key, metric_map in results.items():
        if not isinstance(metric_map, dict):
            metric_map = {"__task_value__": metric_map}
        task_version = clean_scalar(nested_value(obj.get("versions"), task_key, ""))
        sample_info = nested_value(obj.get("n-samples"), task_key, "")
        n_original = sample_info.get("original", "") if isinstance(sample_info, dict) else sample_info
        n_effective = sample_info.get("effective", "") if isinstance(sample_info, dict) else ""
        n_shot = nested_value(obj.get("n-shot"), task_key, "")
        role = task_role(task_key, group_summary_keys, task_subtask_keys)
        for raw_key, value in metric_map.items():
            raw_key = str(raw_key)
            component, metric_name, is_stderr, metric_filter = parse_metric_key(raw_key)
            if not component.strip():
                metric_role = "non_metric_metadata"
            else:
                metric_role = "stderr" if is_stderr else ("alias_metadata" if component == "alias" else "score")
            numeric, value_status = numeric_value(value)
            direction = nested_value(obj.get("higher_is_better"), task_key, metric_name)
            direction_text = "true" if direction is True else "false" if direction is False else "unknown"
            base_key = f"{metric_name},{metric_filter}" if metric_filter else metric_name
            stderr_key = f"{metric_name}_stderr,{metric_filter}" if metric_filter else f"{metric_name}_stderr"
            if is_stderr:
                has_base = base_key in metric_map
                stderr_pair_status = "paired_key_present" if has_base else "unpaired_no_score_key"
                stderr_numeric, _ = numeric_value(value)
            else:
                stderr_value = metric_map.get(stderr_key)
                if stderr_key in metric_map:
                    stderr_numeric, stderr_status = numeric_value(stderr_value)
                    stderr_pair_status = "paired_numeric" if stderr_status == "numeric" else "paired_value_missing_or_invalid"
                else:
                    stderr_numeric, stderr_pair_status = "no_stderr_key", "no_stderr_key"
            identity = "|".join((run_id, str(task_key), task_version, raw_key))
            revision, model_sha, version_status, _ = version_fields(obj)
            yield {
                "observation_id": sha256_bytes(identity.encode("utf-8")),
                "run_id": run_id, "source_file": run["source_file"], "file_sha256": run["file_sha256"],
                "model_dir": run["model_dir"], "model_name_raw": run["model_name_raw"],
                "model_revision": revision, "model_sha": model_sha,
                "version_identity_status": version_status, "run_file_timestamp": run["file_timestamp"],
                "json_date": run["json_date"], "task_key": task_key,
                "task_version": task_version, "task_role": role, "metric_key": raw_key,
                "metric_component_raw": component, "metric_name": metric_name,
                "metric_filter": metric_filter, "metric_role": metric_role,
                "value_raw": json_cell(value), "value_numeric": numeric,
                "value_status": value_status, "higher_is_better": direction_text,
                "stderr_metric_key": stderr_key, "stderr_value_numeric": stderr_numeric,
                "stderr_pair_status": stderr_pair_status, "n_original": clean_scalar(n_original),
                "n_effective": clean_scalar(n_effective), "n_shot": clean_scalar(n_shot),
                "selected_as_latest_parseable": run["selected_as_latest_parseable"],
                "latest_status": run["latest_status"],
            }


def write_long_tables(run_rows: list[dict[str, Any]], out_dir: Path, scope: str) -> dict[str, Any]:
    all_path = out_dir / "q4_c8_long_all_runs.csv.gz"
    latest_path = out_dir / "q4_c8_long_latest.csv.gz"
    coverage: dict[tuple[str, ...], dict[str, Any]] = {}
    task_registry: dict[str, dict[str, Any]] = {}
    group_summary_keys: set[str] = set()
    task_subtask_keys: set[str] = set()
    for run in run_rows:
        obj = run["_obj"]
        groups = obj.get("groups") if isinstance(obj.get("groups"), dict) else {}
        group_summary_keys.update(str(key) for key in groups)
        group_subtasks = obj.get("group_subtasks") if isinstance(obj.get("group_subtasks"), dict) else {}
        for subtasks in group_subtasks.values():
            if isinstance(subtasks, list):
                task_subtask_keys.update(str(key) for key in subtasks)
    # These are stable leaderboard summary rows even when old files omit group metadata.
    group_summary_keys.update({"leaderboard", "leaderboard_bbh", "leaderboard_gpqa", "leaderboard_math_hard", "leaderboard_musr"})
    row_count_all = row_count_latest = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(all_path, "wt", encoding="utf-8-sig", newline="", compresslevel=6) as all_handle, \
         gzip.open(latest_path, "wt", encoding="utf-8-sig", newline="", compresslevel=6) as latest_handle:
        all_writer = csv.DictWriter(all_handle, fieldnames=LONG_FIELDS, extrasaction="ignore")
        latest_writer = csv.DictWriter(latest_handle, fieldnames=LONG_FIELDS, extrasaction="ignore")
        all_writer.writeheader()
        latest_writer.writeheader()
        for run in run_rows:
            for row in iter_long_rows(run, group_summary_keys, task_subtask_keys):
                all_writer.writerow(row)
                row_count_all += 1
                if row["selected_as_latest_parseable"]:
                    latest_writer.writerow(row)
                    row_count_latest += 1
                    task = task_registry.setdefault(row["task_key"], {
                        "task_role": row["task_role"], "versions": set(), "metric_keys": set(),
                        "directions": set(), "models": set(), "numeric_score_models": set(),
                    })
                    if row["task_version"]:
                        task["versions"].add(row["task_version"])
                    if row["metric_role"] == "score":
                        task["metric_keys"].add(row["metric_key"])
                        task["directions"].add(row["higher_is_better"])
                        task["models"].add(row["model_dir"])
                        if row["value_status"] == "numeric":
                            task["numeric_score_models"].add(row["model_dir"])
                    if row["metric_role"] == "score":
                        key = (row["task_key"], row["task_version"], row["metric_key"],
                               row["metric_name"], row["higher_is_better"], row["task_role"])
                        stat = coverage.setdefault(key, {
                            "models": set(), "numeric_models": set(), "missing_models": set(),
                            "effective_sample_values": set(), "direction_unknown_models": set(),
                        })
                        stat["models"].add(row["model_dir"])
                        if row["value_status"] == "numeric":
                            stat["numeric_models"].add(row["model_dir"])
                        else:
                            stat["missing_models"].add(row["model_dir"])
                        if row["n_effective"] != "":
                            stat["effective_sample_values"].add(row["n_effective"])
                        if row["higher_is_better"] == "unknown":
                            stat["direction_unknown_models"].add(row["model_dir"])
    cov_rows = []
    for key, stat in sorted(coverage.items()):
        task_key, task_version, metric_key, metric_name, direction, role = key
        cov_rows.append({
            "task_key": task_key, "task_version": task_version, "metric_key": metric_key,
            "metric_name": metric_name, "higher_is_better": direction, "task_role": role,
            "models_with_key": len(stat["models"]), "models_with_numeric_score": len(stat["numeric_models"]),
            "models_missing_or_invalid_score": len(stat["missing_models"]),
            "models_with_unknown_direction": len(stat["direction_unknown_models"]),
            "distinct_n_effective_values_json": json_cell(sorted(stat["effective_sample_values"])),
        })
    coverage_fields = ["task_key", "task_version", "metric_key", "metric_name", "higher_is_better",
                       "task_role", "models_with_key", "models_with_numeric_score",
                       "models_missing_or_invalid_score", "models_with_unknown_direction",
                       "distinct_n_effective_values_json"]
    write_csv(out_dir / "q4_c8_task_metric_coverage_latest.csv", coverage_fields, cov_rows)
    registry_rows = []
    for task_key, task in sorted(task_registry.items()):
        registry_rows.append({
            "task_key": task_key, "task_role": task["task_role"],
            "versions_json": json_cell(sorted(task["versions"])) if task["versions"] else "[]",
            "score_metric_keys_json": json_cell(sorted(task["metric_keys"])),
            "higher_is_better_values_json": json_cell(sorted(task["directions"])),
            "models_with_score_key": len(task["models"]),
            "models_with_numeric_score": len(task["numeric_score_models"]),
        })
    write_csv(out_dir / "q4_c8_task_registry.csv", [
        "task_key", "task_role", "versions_json", "score_metric_keys_json",
        "higher_is_better_values_json", "models_with_score_key", "models_with_numeric_score",
    ], registry_rows)
    return {"long_rows_all_runs": row_count_all, "long_rows_latest": row_count_latest,
            "coverage_cells": len(cov_rows), "all_runs_path": str(all_path.relative_to(ROOT)),
            "latest_path": str(latest_path.relative_to(ROOT))}


def write_match_table(run_rows: list[dict[str, Any]], c2_for_c1: dict[int, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    c1_rows = load_csv(C1)
    by_name: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for row_num, row in enumerate(c1_rows, start=2):
        by_name[(row.get("Model") or "").strip()].append((row_num, row))
    output: list[dict[str, Any]] = []
    for run in sorted((r for r in run_rows if r["selected_as_latest_parseable"]), key=lambda r: r["model_dir"]):
        name = run["model_name_raw"].strip()
        matches = by_name.get(name, [])
        if not matches:
            status = "unmatched"
        elif len(matches) == 1:
            status = "exact_model_id_unique_record"
        else:
            status = "exact_name_multiple_records"
        c2_candidate: dict[str, Any] = {}
        if len(matches) == 1:
            row_num, c1row = matches[0]
            c2_candidate = c2_for_c1.get(row_num, {})
        output.append({
            "c8_model_name_raw": name, "c8_model_dir": run["model_dir"], "run_id": run["run_id"],
            "c1_record_count": len(matches), "c1_record_ids": json_cell([x[0] for x in matches]),
            "match_status": status, "version_match_status": "source_version_missing",
            "strict_version_confirmed": False, "c8_model_revision": run["model_revision"],
            "c8_model_sha": run["model_sha"],
            "c2_epoch_publication_date_candidate": c2_candidate.get("Epoch_AI_Publication_Date", ""),
            "c2_epoch_organization_candidate": c2_candidate.get("Epoch_AI_Organization", ""),
            "c2_epoch_open_weights_candidate": c2_candidate.get("Epoch_AI_Open_Weights", ""),
        })
    write_csv(out_dir / "q4_c8_c1_match_candidates.csv", MATCH_FIELDS, output)
    counts = Counter(row["match_status"] for row in output)
    try:
        c4_rows = load_csv(C4)
        c4_model_field = "Model" if c4_rows and "Model" in c4_rows[0] else "model"
        c4_names = {(row.get(c4_model_field) or "").strip() for row in c4_rows}
        c4_names.discard("")
        c4_intersection = len({row["c8_model_name_raw"] for row in output} & c4_names)
    except (OSError, UnicodeError, csv.Error):
        c4_intersection = "not_computed"
    summary = [{"c8_latest_model_directories": len(output), "match_status_counts_json": json_cell(counts),
                "strict_version_confirmed_count": sum(bool(r["strict_version_confirmed"]) for r in output),
                "c4_exact_name_intersection": c4_intersection,
                "note": "C1 does not expose model revision/SHA; exact name and unique C1 row are candidates, not version confirmation."}]
    write_csv(out_dir / "q4_c8_c1_match_summary.csv", list(summary[0].keys()), summary)
    return {"match_counts": dict(counts), "latest_model_directories": len(output)}


def build_manifest(out_dir: Path, file_rows: list[dict[str, Any]], run_rows: list[dict[str, Any]],
                   long_stats: dict[str, Any], match_stats: dict[str, Any], scope: str) -> None:
    parsed = [r for r in file_rows if r["parse_status"] == "parseable"]
    latest_status_counts = Counter(r["latest_status"] for r in file_rows)
    fallback_dirs = sorted({r["model_dir"] for r in file_rows if r["latest_status"] == "latest_parseable_fallback"})
    manifest = {
        "scope": scope,
        "ai_disclosure": AI_NOTICE,
        "workspace_root": str(ROOT),
        "input_root": str(C8.relative_to(ROOT)),
        "input_c1": str(C1.relative_to(ROOT)),
        "input_c2": str(C2.relative_to(ROOT)),
        "input_sha256": {str(p.relative_to(ROOT)): sha256_bytes(p.read_bytes()) for p in (C1, C2, C4)},
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "python_version": sys.version,
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "c8_model_directories_in_scope": len({r.name for r in C8.iterdir() if r.is_dir()} if scope == "full" else {r["model_dir"] for r in file_rows}),
        "c8_files_in_scope": len(file_rows),
        "c8_parseable_files_in_scope": len(parsed),
        "c8_run_registry_rows": len(run_rows),
        "latest_status_counts_in_file_registry": dict(latest_status_counts),
        "fallback_model_dirs": fallback_dirs,
        **long_stats,
        **match_stats,
        "limitations": [
            "C1 has no revision/SHA; name/row match does not establish model-version identity.",
            "No cohort, task weights, primary metrics, scale decomposition or forecast is frozen by this parser.",
            "C3 historical paper/report rows are outside the C8 primary metric table.",
            "Codex model ID and release date are verified; team must confirm whether the separate projectless ChatGPT conversation contributed to Q4 artifacts before claiming complete AI-use scope.",
        ],
    }
    (out_dir / "q4_c8_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit C8 JSON and build Q4 run/metric tables.")
    parser.add_argument("--smoke", action="store_true", help="process a small representative set into Q4/03_结果/smoke")
    parser.add_argument("--limit-dirs", type=int, default=0, help="optional maximum number of C8 model directories")
    args = parser.parse_args()
    out_dir = SMOKE if args.smoke else RESULTS
    all_dirs = sorted((p for p in C8.iterdir() if p.is_dir()), key=lambda p: p.name.casefold())
    if args.smoke or args.limit_dirs:
        limit = args.limit_dirs or 12
        chosen = {p.name for p in all_dirs[:limit]}
        # Include the known latest-corrupt fallback and a multi-run directory in the representative sample.
        known_fallback = "DreadPoor_Winter_Dawn-8B-TIES"
        if known_fallback in {p.name for p in all_dirs}:
            chosen.add(known_fallback)
        multi_dir = next((p.parent.name for p in sorted(C8.rglob("results_*.json"))
                          if len(list(p.parent.glob("results_*.json"))) > 1), None)
        if multi_dir:
            chosen.add(multi_dir)
        scoped_dirs = [p for p in all_dirs if p.name in chosen]
        paths = sorted((f for d in scoped_dirs for f in d.glob("results_*.json")), key=lambda p: p.as_posix())
        scope = f"smoke:{len(scoped_dirs)}_model_dirs"
        directories_for_audit = scoped_dirs
    else:
        scoped_dirs = all_dirs
        paths = sorted(C8.rglob("results_*.json"), key=lambda p: p.as_posix())
        scope = "full"
        directories_for_audit = all_dirs
    if not paths:
        raise SystemExit(f"No C8 JSON inputs found under {C8}")

    c2_for_c1 = audit_c1_c2(out_dir)
    file_rows, run_rows = parse_and_audit(paths, out_dir, directories_for_audit, scope)
    long_stats = write_long_tables(run_rows, out_dir, scope)
    match_stats = write_match_table(run_rows, c2_for_c1, out_dir)
    build_manifest(out_dir, file_rows, run_rows, long_stats, match_stats, scope)
    print(json.dumps({"out_dir": str(out_dir), "scope": scope, "files": len(file_rows),
                      "parseable": sum(r["parse_status"] == "parseable" for r in file_rows),
                      "runs": len(run_rows), **long_stats, **match_stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
