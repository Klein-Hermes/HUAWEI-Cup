#!/usr/bin/env python
# AI assistance disclosure: OpenAI Codex desktop, model ID gpt-6-luna (GPT-6 Luna),
# developer OpenAI, official release date 2026-09-22. This script retrieves public,
# immutable Hugging Face dataset commits and computes the audit table from them.
"""Reconstruct one Open LLM Leaderboard v2 snapshot per available calendar month.

The frozen Q4 cohort and current C2 open-weight labels are applied consistently
to each historical snapshot. Results are descriptive snapshots; the C2 labels
and C8 candidate universe are not time-versioned as-of each historical month.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import platform
import statistics
import sys
import tempfile
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pyarrow
    import pyarrow.parquet as parquet
except ImportError as error:  # pragma: no cover - exercised in installations without the optional reader
    raise SystemExit(
        "This snapshot reconstruction needs PyArrow. Install it with "
        "python -m pip install -r Q4/06_历史快照同口径审计_20260926/02_代码/requirements_snapshot.txt"
    ) from error

from q4_historical_snapshot_audit import (
    COMMON_COLUMNS,
    REMOTE_FIELD_MAP,
    one_to_one_match,
    parse_float,
    parse_month,
    read_csv,
    sha256,
    norm_text,
    quantile_linear,
    write_csv,
    write_json,
)


REPO_ID = "open-llm-leaderboard/contents"
PINNED_HEAD_SHA = "9c09a7cae43334062a82cb164f2ef255013dafa2"
PINNED_HEAD_LAST_MODIFIED = "2025-03-20T12:17:27Z"
PINNED_COMMIT_COUNT = 3598
PAGE_SIZE = 100
EXPECTED_MONTHS = [
    "2024-06",
    "2024-07",
    "2024-08",
    "2024-09",
    "2024-10",
    "2024-11",
    "2024-12",
    "2025-01",
    "2025-02",
    "2025-03",
]
METRIC_SOURCE_FIELDS = tuple(
    field for field in REMOTE_FIELD_MAP.values() if field != "Submission Date"
) + ("Model sha",)
TEMPORAL_SOURCE_FIELD = "Submission Date"
USER_AGENT = "q4-monthly-snapshot-reconstruction/1.0 (read-only public dataset audit)"


def fetch_json(url: str, timeout: int = 60) -> tuple[Any, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    return json.loads(body.decode("utf-8")), headers


def normalize_utc(value: Any) -> str | None:
    text = norm_text(value)
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repository_info() -> dict[str, Any]:
    encoded = urllib.parse.quote(REPO_ID, safe="/")
    info, _ = fetch_json(f"https://huggingface.co/api/datasets/{encoded}")
    return {
        "id": info.get("id"),
        "sha": info.get("sha"),
        "lastModified": info.get("lastModified"),
    }


def fetch_commit_page(page: int) -> tuple[int, list[dict[str, Any]], int]:
    encoded = urllib.parse.quote(REPO_ID, safe="/")
    url = f"https://huggingface.co/api/datasets/{encoded}/commits/main?p={page}&limit={PAGE_SIZE}"
    payload, headers = fetch_json(url)
    total = int(headers.get("x-total-count", "-1"))
    return page, payload, total


def fetch_month_end_commits() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    page0, first_page, total = fetch_commit_page(0)
    if page0 != 0 or total != PINNED_COMMIT_COUNT:
        raise RuntimeError(f"Unexpected commit history size: expected={PINNED_COMMIT_COUNT}, got={total}")
    page_count = math.ceil(total / PAGE_SIZE)
    pages: dict[int, list[dict[str, Any]]] = {0: first_page}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch_commit_page, page) for page in range(1, page_count)]
        for future in concurrent.futures.as_completed(futures):
            page, payload, observed_total = future.result()
            if observed_total != total:
                raise RuntimeError(f"Commit count changed during pagination: {observed_total} != {total}")
            pages[page] = payload
    commits = [commit for page in range(page_count) for commit in pages[page]]
    if len(commits) != total:
        raise RuntimeError(f"Commit history pagination incomplete: {len(commits)} != {total}")
    for page in range(page_count):
        expected_page_size = min(PAGE_SIZE, total - page * PAGE_SIZE)
        observed_page_size = len(pages[page])
        if observed_page_size != expected_page_size:
            raise RuntimeError(
                f"Commit history page {page} has {observed_page_size} rows; "
                f"expected {expected_page_size}"
            )
    commit_ids = [norm_text(commit.get("id")) for commit in commits]
    if any(not commit_id for commit_id in commit_ids):
        raise RuntimeError("Commit history contains a commit without an id")
    if len(set(commit_ids)) != total:
        raise RuntimeError(f"Commit history contains duplicate ids: {total - len(set(commit_ids))} duplicate(s)")
    head = commits[0]
    if head.get("id") != PINNED_HEAD_SHA or normalize_utc(head.get("date")) != normalize_utc(PINNED_HEAD_LAST_MODIFIED):
        raise RuntimeError(f"Pinned repository head changed: {head}")

    selected: dict[str, dict[str, Any]] = {}
    for commit in commits:
        month = norm_text(commit.get("date"))[:7]
        if month not in selected or commit.get("date", "") > selected[month].get("date", ""):
            selected[month] = commit
    selected = {month: selected[month] for month in sorted(selected) if month in EXPECTED_MONTHS}
    if list(selected) != EXPECTED_MONTHS:
        raise RuntimeError(f"Expected one latest-available monthly commit for {EXPECTED_MONTHS}; got {list(selected)}")
    return selected, {
        "commit_count": len(commits),
        "head_sha": head.get("id"),
        "head_date": head.get("date"),
        "pagination_pages": page_count,
        "commit_api": f"https://huggingface.co/api/datasets/{REPO_ID}/commits/main",
        "selection_rule": "latest available dated commit within each UTC calendar month in the pinned v2 repo history; March 2025 ends on March 20",
    }


def fetch_tree(revision: str) -> list[dict[str, Any]]:
    encoded_repo = urllib.parse.quote(REPO_ID, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    url = f"https://huggingface.co/api/datasets/{encoded_repo}/tree/{encoded_revision}?recursive=true&expand=true"
    payload, _ = fetch_json(url)
    return payload


def download_parquet(revision: str, relative_path: str, cache_dir: Path) -> tuple[Path, str, int, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / f"{revision}.parquet"
    acquisition_mode = "verified_existing_cache"
    if not local_path.is_file():
        acquisition_mode = "fresh_resolve_download"
        encoded_repo = urllib.parse.quote(REPO_ID, safe="/")
        encoded_path = urllib.parse.quote(relative_path, safe="/")
        url = f"https://huggingface.co/datasets/{encoded_repo}/resolve/{revision}/{encoded_path}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=90) as response, local_path.open("wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
    digest = hashlib.sha256()
    byte_count = 0
    with local_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    return local_path, digest.hexdigest(), byte_count, acquisition_mode


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "03_结果")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "q4_hf_v2_month_end_snapshots_20260926",
    )
    parser.add_argument("--offline-inventory", type=Path, default=None, help="Recompute from cached Parquet files listed in a verified inventory; no Hub API calls.")
    parser.add_argument("--offline-manifest", type=Path, default=None, help="Reproduction manifest paired with --offline-inventory.")
    args = parser.parse_args()
    root = args.project_root.resolve()
    out_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    offline_mode = args.offline_inventory is not None
    if offline_mode != (args.offline_manifest is not None):
        parser.error("--offline-inventory and --offline-manifest must be supplied together")
    before: dict[str, Any] | None = None
    offline_inventory: list[dict[str, str]] = []
    offline_manifest: dict[str, Any] | None = None
    if offline_mode:
        offline_inventory = read_csv(args.offline_inventory.resolve())
        offline_manifest = json.loads(args.offline_manifest.resolve().read_text(encoding="utf-8"))
        if offline_manifest.get("repo_head_sha") != PINNED_HEAD_SHA:
            raise RuntimeError("Offline manifest is not pinned to the expected repository head")
        history_meta = offline_manifest.get("history", {})
        if history_meta.get("commit_count") != PINNED_COMMIT_COUNT:
            raise RuntimeError("Offline manifest commit count does not match the pinned 3,598-commit history")
        inventory_relpath = str(args.offline_inventory.resolve().relative_to(root))
        expected_inventory_sha = offline_manifest.get("outputs_sha256", {}).get(inventory_relpath)
        observed_inventory_sha = sha256(args.offline_inventory.resolve())
        if expected_inventory_sha != observed_inventory_sha:
            raise RuntimeError(
                "Offline inventory SHA does not match its source manifest: "
                f"expected={expected_inventory_sha}, observed={observed_inventory_sha}"
            )
        if len(offline_inventory) != len(EXPECTED_MONTHS):
            raise RuntimeError(
                f"Offline inventory must have {len(EXPECTED_MONTHS)} rows, got {len(offline_inventory)}"
            )
        offline_months = [row.get("snapshot_month") for row in offline_inventory]
        if len(set(offline_months)) != len(offline_months):
            raise RuntimeError("Offline inventory contains duplicate snapshot months")
        month_commits = {
            row["snapshot_month"]: {"id": row["commit_sha"], "date": row["as_of_commit_date_utc"]}
            for row in offline_inventory
        }
        if list(month_commits) != EXPECTED_MONTHS:
            raise RuntimeError(f"Offline inventory months do not match the pinned audit months: {list(month_commits)}")
    else:
        before = repository_info()
        if before.get("id") != REPO_ID or before.get("sha") != PINNED_HEAD_SHA:
            raise RuntimeError(f"Repository does not resolve to the pinned v2 head: {before}")
        month_commits, history_meta = fetch_month_end_commits()

    source_dir = root / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution"
    c1_path = source_dir / "leaderboard_cleaned.csv"
    c2_path = source_dir / "leaderboard_enhanced.csv"
    candidates_path = root / "Q4" / "03_结果" / "results" / "q4_c8_c1_match_candidates.csv"
    baseline_path = root / "Q4" / "03_结果" / "results" / "q4_frontier_monthly.csv"
    script_helper_path = Path(__file__).resolve().with_name("q4_historical_snapshot_audit.py")
    requirements_path = Path(__file__).resolve().with_name("requirements_snapshot.txt")
    required = [c1_path, c2_path, candidates_path, baseline_path, script_helper_path, requirements_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required snapshot audit input(s) missing: " + ", ".join(missing))
    c1 = read_csv(c1_path)
    c2 = read_csv(c2_path)
    candidates = read_csv(candidates_path)
    baseline = read_csv(baseline_path)
    c1_to_c2, alignment = one_to_one_match(c1, c2, "local_C1", "local_C2")
    if not alignment["all_rows_one_to_one"]:
        raise RuntimeError(f"C1/C2 alignment failed: {alignment}")
    c2_by_c1_index = {c1_i: c2[c2_i] for c1_i, c2_i in c1_to_c2.items()}

    exact_candidates = [
        row for row in candidates if norm_text(row.get("match_status")) == "exact_model_id_unique_record"
    ]
    if len(exact_candidates) != 1819:
        raise RuntimeError(f"Expected 1,819 frozen exact-name candidates, got {len(exact_candidates)}")
    candidate_names = [norm_text(row.get("c8_model_name_raw")) for row in exact_candidates]
    if len(set(candidate_names)) != len(candidate_names):
        raise RuntimeError("Frozen exact-name candidate list contains duplicate names")
    c1_name_to_index: dict[str, int] = {}
    c1_name_counts = Counter(norm_text(row.get("Model")) for row in c1)
    for index, row in enumerate(c1):
        name = norm_text(row.get("Model"))
        if name:
            c1_name_to_index.setdefault(name, index)
    for name in candidate_names:
        if c1_name_counts[name] != 1:
            raise RuntimeError(f"Frozen candidate is not unique in local C1: {name}")
    open_status = {
        name: norm_text(c2_by_c1_index[c1_name_to_index[name]].get("Epoch_AI_Open_Weights")) or "unknown"
        for name in candidate_names
    }
    open_counts = Counter(open_status.values())
    final_open_by_month: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for name in candidate_names:
        if open_status[name] != "Yes":
            continue
        c1_row = c1[c1_name_to_index[name]]
        month = parse_month(c1_row.get("Submission Date"))
        score = parse_float(c1_row.get("Average ⬆️"))
        if month and score is not None:
            final_open_by_month[month][name] = {
                "score": score,
                "sha": "",
            }
    baseline_by_month = {
        row["submission_month"]: parse_float(row.get("monthly_entry_p90_frontier_C1_average_0_100"))
        for row in baseline
    }
    for month, baseline_row in ((row["submission_month"], row) for row in baseline):
        expected_count = int(parse_float(baseline_row.get("n_known_open_weights_models")) or 0)
        observed_count = len(final_open_by_month.get(month, {}))
        if expected_count != observed_count:
            raise RuntimeError(f"Final open-weight membership mismatch in {month}: {observed_count} != {expected_count}")

    snapshot_specs: dict[str, tuple[dict[str, Any], str]] = {}
    downloaded_by_month: dict[str, tuple[Path, str, int, str]] = {}
    if offline_mode:
        inventory_by_month = {row["snapshot_month"]: row for row in offline_inventory}
        for snapshot_month, commit in month_commits.items():
            inventory_row = inventory_by_month[snapshot_month]
            parquet_path = norm_text(inventory_row.get("parquet_path_at_revision"))
            local_parquet = cache_dir / f"{commit['id']}.parquet"
            if not local_parquet.is_file():
                raise FileNotFoundError(f"Cached offline parquet is missing: {local_parquet}")
            actual_sha = sha256(local_parquet)
            if actual_sha != inventory_row.get("parquet_sha256"):
                raise RuntimeError(f"Cached parquet SHA mismatch for {snapshot_month}: {actual_sha}")
            size = local_parquet.stat().st_size
            if size != int(inventory_row.get("parquet_bytes", "-1")):
                raise RuntimeError(f"Cached parquet size mismatch for {snapshot_month}: {size}")
            snapshot_specs[snapshot_month] = (commit, parquet_path)
            downloaded_by_month[snapshot_month] = (local_parquet, actual_sha, size, "offline_cache_sha_verified")
    else:
        trees_by_month: dict[str, list[dict[str, Any]]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fetch_tree, norm_text(commit.get("id"))): month for month, commit in month_commits.items()}
            for future in concurrent.futures.as_completed(futures):
                trees_by_month[futures[future]] = future.result()
        for snapshot_month, commit in month_commits.items():
            trees = trees_by_month[snapshot_month]
            parquet_files = [entry for entry in trees if norm_text(entry.get("path")).lower().endswith(".parquet")]
            if len(parquet_files) != 1:
                raise RuntimeError(f"Expected exactly one parquet file at {commit.get('id')}; found {len(parquet_files)}")
            snapshot_specs[snapshot_month] = (commit, norm_text(parquet_files[0].get("path")))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(download_parquet, norm_text(commit.get("id")), parquet_path, cache_dir): month
                for month, (commit, parquet_path) in snapshot_specs.items()
            }
            for future in concurrent.futures.as_completed(futures):
                downloaded_by_month[futures[future]] = future.result()

    final_snapshot_path = downloaded_by_month["2025-03"][0]
    final_snapshot_rows = parquet.read_table(final_snapshot_path).to_pylist()
    candidate_name_set = set(candidate_names)
    final_candidate_sha_rows = [
        row for row in final_snapshot_rows if norm_text(row.get("fullname")) in candidate_name_set
    ]
    final_candidate_row_counts = Counter(norm_text(row.get("fullname")) for row in final_candidate_sha_rows)
    if any(final_candidate_row_counts[name] != 1 for name in candidate_names):
        raise RuntimeError("A frozen Q4 candidate is not unique in the latest pinned v2 snapshot")
    final_source_sha_by_model = {
        norm_text(row.get("fullname")): norm_text(row.get("Model sha"))
        for row in final_candidate_sha_rows
        if norm_text(row.get("fullname"))
    }
    if len(final_source_sha_by_model) != len(candidate_names):
        raise RuntimeError("A frozen Q4 C1/C8 candidate is missing a Model sha in the latest pinned v2 snapshot")
    for month, models in final_open_by_month.items():
        for name in models:
            models[name]["sha"] = final_source_sha_by_model.get(name, "")

    inventory_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    snapshot_protocol_columns_ok = True
    for snapshot_month, commit in month_commits.items():
        revision = norm_text(commit.get("id"))
        commit_date = norm_text(commit.get("date"))
        _, parquet_path = snapshot_specs[snapshot_month]
        local_parquet, parquet_sha, parquet_bytes, acquisition_mode = downloaded_by_month[snapshot_month]
        table = parquet.read_table(local_parquet)
        fields = set(table.schema.names)
        missing_fields = sorted(set(METRIC_SOURCE_FIELDS) - fields)
        if missing_fields:
            snapshot_protocol_columns_ok = False
        has_submission_date = TEMPORAL_SOURCE_FIELD in fields
        if not set(COMMON_COLUMNS).issubset(set(REMOTE_FIELD_MAP.keys())):
            raise RuntimeError("Local field map no longer covers all common metric fields")

        snapshot_rows = table.to_pylist()
        snapshot_date = datetime.fromisoformat(commit_date.replace("Z", "+00:00")).date()
        future_submission_rows = 0
        rows_without_submission_date = 0
        candidate_rows_in_snapshot = 0
        candidate_current_yes_rows_in_snapshot = 0
        eligible_candidate_rows = 0
        open_weight_rows = 0
        candidate_names_in_snapshot: Counter[str] = Counter()
        eligible_candidate_data: list[tuple[str, dict[str, Any], str]] = []
        current_open_data_by_month: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in snapshot_rows:
            model = norm_text(row.get("fullname"))
            if model not in open_status:
                continue
            candidate_rows_in_snapshot += 1
            if open_status[model] == "Yes":
                candidate_current_yes_rows_in_snapshot += 1
            if not has_submission_date:
                rows_without_submission_date += 1
                continue
            date_text = norm_text(row.get("Submission Date"))
            month = parse_month(date_text)
            try:
                submission_date = datetime.strptime(date_text[:10], "%Y-%m-%d").date()
            except ValueError:
                rows_without_submission_date += 1
                continue
            if submission_date > snapshot_date:
                future_submission_rows += 1
                continue
            eligible_candidate_rows += 1
            candidate_names_in_snapshot[model] += 1
            eligible_candidate_data.append((model, row, month or ""))

        # Keep one row per frozen model identity. If a historical table happens
        # to contain multiple rows for one candidate, exclude that name rather
        # than letting duplicated IDs overweight the percentile.
        for model, row, month in eligible_candidate_data:
            if candidate_names_in_snapshot[model] != 1 or open_status[model] != "Yes" or not month:
                continue
            score = parse_float(row.get("Average ⬆️"))
            if score is None:
                continue
            open_weight_rows += 1
            current_open_data_by_month[month][model] = {
                "score": score,
                "sha": norm_text(row.get("Model sha")),
            }

        for submission_month, snapshot_models in sorted(current_open_data_by_month.items()):
            final_models = final_open_by_month.get(submission_month, {})
            snapshot_names = set(snapshot_models)
            final_names = set(final_models)
            common_names = snapshot_names & final_names
            same_sha_names = {
                name for name in common_names
                if snapshot_models[name]["sha"] and final_models[name]["sha"]
                and snapshot_models[name]["sha"] == final_models[name]["sha"]
            }
            different_sha_names = {
                name for name in common_names
                if snapshot_models[name]["sha"] and final_models[name]["sha"]
                and snapshot_models[name]["sha"] != final_models[name]["sha"]
            }
            missing_sha_names = common_names - same_sha_names - different_sha_names
            snapshot_scores = [snapshot_models[name]["score"] for name in snapshot_names]
            final_scores = [final_models[name]["score"] for name in final_names]
            common_snapshot_scores = [snapshot_models[name]["score"] for name in common_names]
            common_final_scores = [final_models[name]["score"] for name in common_names]
            same_sha_snapshot_scores = [snapshot_models[name]["score"] for name in same_sha_names]
            same_sha_final_scores = [final_models[name]["score"] for name in same_sha_names]
            same_sha_score_diffs = [
                abs(snapshot_models[name]["score"] - final_models[name]["score"])
                for name in same_sha_names
            ]
            p90 = quantile_linear(snapshot_scores, 0.90)
            final_p90 = baseline_by_month.get(submission_month)
            difference = abs(p90 - final_p90) if final_p90 is not None else None
            same_membership_snapshot_p90 = quantile_linear(common_snapshot_scores, 0.90)
            same_membership_final_p90 = quantile_linear(common_final_scores, 0.90)
            same_membership_p90_difference = (
                abs(same_membership_snapshot_p90 - same_membership_final_p90)
                if same_membership_snapshot_p90 is not None and same_membership_final_p90 is not None
                else None
            )
            same_sha_snapshot_p90 = quantile_linear(same_sha_snapshot_scores, 0.90)
            same_sha_final_p90 = quantile_linear(same_sha_final_scores, 0.90)
            same_sha_p90_difference = (
                abs(same_sha_snapshot_p90 - same_sha_final_p90)
                if same_sha_snapshot_p90 is not None and same_sha_final_p90 is not None
                else None
            )
            cohort_rows.append(
                {
                    "snapshot_month": snapshot_month,
                    "snapshot_commit_date_utc": commit_date,
                    "snapshot_commit_sha": revision,
                    "submission_month": submission_month,
                    "n_open_weight_rows_current_C2_label": len(snapshot_names),
                    "n_open_weight_unique_models": len(snapshot_names),
                    "snapshot_p90_C1_Average": p90,
                    "final_baseline_open_weight_models": len(final_names),
                    "final_frozen_baseline_p90": final_p90,
                    "absolute_difference_to_final_baseline": difference,
                    "matches_final_frozen_baseline_within_1e-9": difference is not None and difference <= 1e-9,
                    "common_model_names_with_final_cohort": len(common_names),
                    "snapshot_only_model_names": len(snapshot_names - final_names),
                    "final_only_model_names": len(final_names - snapshot_names),
                    "common_model_names_same_sha_as_final": len(same_sha_names),
                    "common_model_names_sha_differs_from_final": len(different_sha_names),
                    "common_model_names_sha_missing": len(missing_sha_names),
                    "same_membership_snapshot_p90": same_membership_snapshot_p90,
                    "same_membership_final_p90": same_membership_final_p90,
                    "same_membership_p90_absolute_difference": same_membership_p90_difference,
                    "same_sha_model_count": len(same_sha_names),
                    "same_sha_snapshot_p90": same_sha_snapshot_p90,
                    "same_sha_final_p90": same_sha_final_p90,
                    "same_sha_p90_absolute_difference": same_sha_p90_difference,
                    "same_sha_median_absolute_score_difference": statistics.median(same_sha_score_diffs) if same_sha_score_diffs else None,
                    "same_sha_max_absolute_score_difference": max(same_sha_score_diffs) if same_sha_score_diffs else None,
                    "quantile_definition": "linear interpolation at h=(n-1)*0.90",
                    "open_weight_label_note": "Current C2 Epoch label applied retrospectively; no as-of history supplied.",
                }
            )

        same_month = next((row for row in reversed(cohort_rows) if row["snapshot_month"] == snapshot_month and row["submission_month"] == snapshot_month), None)
        inventory_rows.append(
            {
                "snapshot_month": snapshot_month,
                "as_of_commit_date_utc": commit_date,
                "commit_sha": revision,
                "parquet_path_at_revision": parquet_path,
                "parquet_sha256": parquet_sha,
                "parquet_bytes": parquet_bytes,
                "parquet_acquisition_mode": acquisition_mode,
                "snapshot_rows": table.num_rows,
                "metric_schema_fields_present": len(METRIC_SOURCE_FIELDS) - len(missing_fields),
                "metric_schema_fields_expected": len(METRIC_SOURCE_FIELDS),
                "missing_metric_fields": ";".join(missing_fields),
                "frozen_C8_C1_candidate_rows_in_snapshot": candidate_rows_in_snapshot,
                "candidate_current_C2_yes_rows_in_snapshot": candidate_current_yes_rows_in_snapshot,
                "candidate_rows_without_submission_date": rows_without_submission_date,
                "rows_for_frozen_C8_C1_candidate_universe_as_of_commit_date": eligible_candidate_rows,
                "unique_frozen_candidate_models_as_of_commit_date": len(candidate_names_in_snapshot),
                "duplicate_candidate_model_name_groups": sum(count > 1 for count in candidate_names_in_snapshot.values()),
                "current_C2_yes_open_weight_rows": open_weight_rows,
                "future_submission_rows_excluded": future_submission_rows,
                "same_month_submission_count": same_month["n_open_weight_rows_current_C2_label"] if same_month else None,
                "same_month_submission_p90": same_month["snapshot_p90_C1_Average"] if same_month else None,
                "final_baseline_same_month_p90": baseline_by_month.get(snapshot_month),
                "same_month_p90_matches_final_baseline": same_month["matches_final_frozen_baseline_within_1e-9"] if same_month else None,
                "final_baseline_same_month_count": same_month["final_baseline_open_weight_models"] if same_month else len(final_open_by_month.get(snapshot_month, {})),
                "same_month_common_model_names": same_month["common_model_names_with_final_cohort"] if same_month else None,
                "same_month_snapshot_only_model_names": same_month["snapshot_only_model_names"] if same_month else None,
                "same_month_final_only_model_names": same_month["final_only_model_names"] if same_month else None,
                "same_month_common_names_same_model_sha": same_month["common_model_names_same_sha_as_final"] if same_month else None,
                "same_month_common_names_different_model_sha": same_month["common_model_names_sha_differs_from_final"] if same_month else None,
                "same_month_common_names_missing_model_sha": same_month["common_model_names_sha_missing"] if same_month else None,
                "same_model_sha_median_abs_score_difference": same_month["same_sha_median_absolute_score_difference"] if same_month else None,
                "same_model_sha_max_abs_score_difference": same_month["same_sha_max_absolute_score_difference"] if same_month else None,
                "snapshot_protocol_gate": "v2_metric_fields_present" if not missing_fields else "missing_metric_fields",
                "temporal_cohort_gate": "submission_date_present" if has_submission_date else "submission_date_field_absent",
                "same_month_snapshot_status": "computed" if same_month else ("submission_date_field_absent" if not has_submission_date else "no_open_weight_same_month_rows"),
            }
        )

    if before is not None:
        after = repository_info()
        if before != after:
            raise RuntimeError(f"Hub repository head changed during snapshot retrieval: before={before}, after={after}")
    if not snapshot_protocol_columns_ok:
        missing_by_month = {row["snapshot_month"]: row["missing_metric_fields"] for row in inventory_rows if row["missing_metric_fields"]}
        raise RuntimeError(f"At least one monthly snapshot is missing required v2 metric fields: {missing_by_month}")
    if len(inventory_rows) != len(EXPECTED_MONTHS):
        raise RuntimeError(f"Incomplete month-end inventory: {len(inventory_rows)} != {len(EXPECTED_MONTHS)}")

    inventory_path = out_dir / "historical_snapshot_inventory.csv"
    cohort_path = out_dir / "historical_snapshot_cohort_p90.csv"
    same_month_path = out_dir / "historical_snapshot_same_month_p90.csv"
    write_csv(
        inventory_path,
        inventory_rows,
        [
            "snapshot_month",
            "as_of_commit_date_utc",
            "commit_sha",
            "parquet_path_at_revision",
            "parquet_sha256",
            "parquet_bytes",
            "parquet_acquisition_mode",
            "snapshot_rows",
            "metric_schema_fields_present",
            "metric_schema_fields_expected",
            "missing_metric_fields",
            "frozen_C8_C1_candidate_rows_in_snapshot",
            "candidate_current_C2_yes_rows_in_snapshot",
            "candidate_rows_without_submission_date",
            "rows_for_frozen_C8_C1_candidate_universe_as_of_commit_date",
            "unique_frozen_candidate_models_as_of_commit_date",
            "duplicate_candidate_model_name_groups",
            "current_C2_yes_open_weight_rows",
            "future_submission_rows_excluded",
            "same_month_submission_count",
            "same_month_submission_p90",
            "final_baseline_same_month_p90",
            "same_month_p90_matches_final_baseline",
            "final_baseline_same_month_count",
            "same_month_common_model_names",
            "same_month_snapshot_only_model_names",
            "same_month_final_only_model_names",
            "same_month_common_names_same_model_sha",
            "same_month_common_names_different_model_sha",
            "same_month_common_names_missing_model_sha",
            "same_model_sha_median_abs_score_difference",
            "same_model_sha_max_abs_score_difference",
            "snapshot_protocol_gate",
            "temporal_cohort_gate",
            "same_month_snapshot_status",
        ],
    )
    write_csv(
        cohort_path,
        cohort_rows,
        [
            "snapshot_month",
            "snapshot_commit_date_utc",
            "snapshot_commit_sha",
            "submission_month",
            "n_open_weight_rows_current_C2_label",
            "n_open_weight_unique_models",
            "snapshot_p90_C1_Average",
            "final_baseline_open_weight_models",
            "final_frozen_baseline_p90",
            "absolute_difference_to_final_baseline",
            "matches_final_frozen_baseline_within_1e-9",
            "common_model_names_with_final_cohort",
            "snapshot_only_model_names",
            "final_only_model_names",
            "common_model_names_same_sha_as_final",
            "common_model_names_sha_differs_from_final",
            "common_model_names_sha_missing",
            "same_membership_snapshot_p90",
            "same_membership_final_p90",
            "same_membership_p90_absolute_difference",
            "same_sha_model_count",
            "same_sha_snapshot_p90",
            "same_sha_final_p90",
            "same_sha_p90_absolute_difference",
            "same_sha_median_absolute_score_difference",
            "same_sha_max_absolute_score_difference",
            "quantile_definition",
            "open_weight_label_note",
        ],
    )
    same_month_rows = [row for row in cohort_rows if row["snapshot_month"] == row["submission_month"]]
    comparable_months = sum(row["temporal_cohort_gate"] == "submission_date_present" for row in inventory_rows)
    if len(same_month_rows) != comparable_months:
        raise RuntimeError(f"Expected one same-month cohort result for each snapshot with dates, got {len(same_month_rows)} != {comparable_months}")
    write_csv(
        same_month_path,
        same_month_rows,
        [
            "snapshot_month",
            "snapshot_commit_date_utc",
            "snapshot_commit_sha",
            "submission_month",
            "n_open_weight_rows_current_C2_label",
            "snapshot_p90_C1_Average",
            "final_baseline_open_weight_models",
            "final_frozen_baseline_p90",
            "absolute_difference_to_final_baseline",
            "matches_final_frozen_baseline_within_1e-9",
            "common_model_names_with_final_cohort",
            "snapshot_only_model_names",
            "final_only_model_names",
            "common_model_names_same_sha_as_final",
            "common_model_names_sha_differs_from_final",
            "common_model_names_sha_missing",
            "same_membership_snapshot_p90",
            "same_membership_final_p90",
            "same_membership_p90_absolute_difference",
            "same_sha_model_count",
            "same_sha_snapshot_p90",
            "same_sha_final_p90",
            "same_sha_p90_absolute_difference",
            "same_sha_median_absolute_score_difference",
            "same_sha_max_absolute_score_difference",
            "open_weight_label_note",
        ],
    )

    p90_match_rows = sum(row["matches_final_frozen_baseline_within_1e-9"] is True for row in same_month_rows)
    table_lines = []
    for row in inventory_rows:
        p90_value = row["same_month_submission_p90"]
        p90_text = f"{float(p90_value):.6f}" if p90_value is not None else "—"
        final_p90 = row["final_baseline_same_month_p90"]
        final_p90_text = f"{float(final_p90):.6f}" if final_p90 is not None else "—"
        difference = abs(float(p90_value) - float(final_p90)) if p90_value is not None and final_p90 is not None else None
        difference_text = f"{float(difference):.6g}" if difference is not None else "—"
        common_names = row["same_month_common_model_names"]
        same_sha = row["same_month_common_names_same_model_sha"]
        same_sha_score_delta = row["same_model_sha_median_abs_score_difference"]
        same_sha_score_delta_text = f"{float(same_sha_score_delta):.6g}" if same_sha_score_delta is not None else "—"
        count_text = (
            f"{row['same_month_submission_count']}/{row['final_baseline_same_month_count']}"
            if row["same_month_submission_count"] is not None
            else f"—/{row['final_baseline_same_month_count']}"
        )
        common_text = f"{common_names} ({same_sha} 同 SHA)" if common_names is not None else "—"
        table_lines.append(
            f"| {row['snapshot_month']} | {count_text} | "
            f"{common_text} | {p90_text} | {final_p90_text} | {difference_text} | "
            f"{same_sha_score_delta_text} |"
        )
    table_md = "\n".join(table_lines)
    acquisition_counts = Counter(row["parquet_acquisition_mode"] for row in inventory_rows)
    report = f"""# v2 历史月度最后可用快照重建

## 重建结果

从固定的 Open LLM Leaderboard v2 仓库历史中，逐月选择 UTC 自然月内最后一条可用提交，并使用该版本 Parquet 文件。提交历史共 {history_meta['commit_count']:,} 条；重建 {len(inventory_rows)} 个可用月份，范围 {inventory_rows[0]['snapshot_month']} 至 {inventory_rows[-1]['snapshot_month']}（2025-03 的最后可用版本日期为 3 月 20 日）。每份月度快照均固定提交 SHA、Parquet 路径与文件 SHA-256，详见 `historical_snapshot_inventory.csv`。

本次来源校验模式：**{"离线缓存独立复算" if offline_mode else "在线仓库历史及提交地址校验"}**。{"离线模式校验缓存文件 SHA/字节数及来源清单中记录的 inventory SHA；它独立重算表格，但不重新证明 Hugging Face 在线提交历史身份。" if offline_mode else "在线模式分页核验提交总数、每页行数、提交 SHA 去重数，检查固定仓库头；各文件获取方式记在 inventory 表中，`fresh_resolve_download` 表示本次从固定提交地址新下载，`verified_existing_cache` 表示读取既有缓存。"}

## 每月当时新增队列与最终版对照

| 快照月 | 快照 n / 最终 n | 同名模型数（其中同 SHA） | P90（当时快照） | 最终表 P90 | P90 绝对差 | 同 SHA 模型分数差中位数 |
|---|---:|---:|---:|---:|---:|---:|
{table_md}

在 {len(same_month_rows)} 个可计算的同月对比中，{p90_match_rows} 个原始 P90 与冻结最终表在 1e-9 内一致。直接 P90 差异同时包含模型进入/退出队列和分数变化；清单同时列出共同模型数、共同模型中 `Model sha` 一致数，以及同一 SHA 模型的 C1 分数绝对差中位数，供拆分检查。共同名称 P90 与同 SHA 子集 P90 均见 `historical_snapshot_cohort_p90.csv`。

**2024-06** 月末表没有 `Submission Date` 字段，无法按提交月份归属队列，因此不计算该月同月 P90；表中留空而不插补。其余 P90 均用对应提交版本中的 v2 `Average ⬆️`，线性插值重建。

## 口径和时间边界

- 历史版本限定为同一 v2 `contents` 仓库；每月取当月最后可用提交。v1 旧榜任务集不纳入。
- 每月均复用冻结 Q4 的 1,819 个 C8 名称精确/C1 唯一候选，避免改变模型全集；只保留对应当前 C2 `Epoch_AI_Open_Weights=Yes` 的行。对比最终 P90 时，再按模型名求交集，并以历史快照 `Model sha` 对照冻结最终 v2 C1 源表的 `Model sha`。
- C2 开放权重标签没有历史 as-of 版本；C8 候选集也是冻结的当前队列。因此本表是**固定当期定义后的历史快照描述/稳定性审计**，不能解释为当时可获得的信息或无前视偏差预测回测。
- 六个 v2 指标字段在每月文件中都存在；2024-06 快照缺少 `Submission Date`，其余月份有该字段。相同 `Model sha` 下的分数变化可定位为同一权重版本记录值发生变化，但不能单独归因于任务代码、提示词、执行参数或数据修订中的哪一项。不要把这张表解释为年度预测验证。
- 数据仓库最新提交为 `{PINNED_HEAD_SHA}`（`{PINNED_HEAD_LAST_MODIFIED}`）；没有该 v2 数据源的 2026-03 观测。

## 复现

临时安装 Parquet reader：`python -m pip install -r "Q4/06_历史快照同口径审计_20260926/02_代码/requirements_snapshot.txt"`。

运行：`python "Q4/06_历史快照同口径审计_20260926/02_代码/q4_monthly_snapshot_reconstruction.py"`。

下载的快照文件放在系统临时目录，不会改写 Q4 冻结输入；清单记录每个提交版本的文件 SHA-256。数据源本身为 Hugging Face v2 `open-llm-leaderboard/contents`。
"""
    report_path = out_dir / "historical_snapshot_report.md"
    report_path.write_text(report, encoding="utf-8")

    inputs = [c1_path, c2_path, candidates_path, baseline_path, script_helper_path, requirements_path, Path(__file__).resolve()]
    if offline_mode:
        inputs.extend([args.offline_inventory.resolve(), args.offline_manifest.resolve()])
    outputs = [inventory_path, cohort_path, same_month_path, report_path]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "offline_monthly_snapshot_recomputation_complete" if offline_mode else "monthly_snapshot_reconstruction_complete",
        "project_root": str(root),
        "repo_id": REPO_ID,
        "repo_head_sha": PINNED_HEAD_SHA,
        "repo_head_last_modified": PINNED_HEAD_LAST_MODIFIED,
        "history": history_meta,
        "monthly_snapshots": len(inventory_rows),
        "snapshot_months": [row["snapshot_month"] for row in inventory_rows],
        "months_without_submission_date": [
            row["snapshot_month"] for row in inventory_rows if row["temporal_cohort_gate"] != "submission_date_present"
        ],
        "same_month_p90_matches_final_baseline": p90_match_rows,
        "same_month_p90_comparisons": len(same_month_rows),
        "parquet_acquisition_modes": dict(acquisition_counts),
        "current_c2_open_weight_status_counts_in_frozen_candidates": dict(open_counts),
        "frozen_exact_name_unique_candidates": len(exact_candidates),
        "c1_c2_row_alignment": alignment,
        "script_sha256": sha256(Path(__file__).resolve()),
        "helper_script_sha256": sha256(script_helper_path),
        "source_check_mode": "offline_cached_parquet_recalculation" if offline_mode else "online_huggingface_revision_validation",
        "source_history_manifest_sha256": sha256(args.offline_manifest.resolve()) if offline_mode else None,
        "inputs_sha256": {str(path.relative_to(root)): sha256(path) for path in inputs},
        "outputs_sha256": {str(path.relative_to(root)): sha256(path) for path in outputs},
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "pyarrow_version": pyarrow.__version__,
            "additional_dependencies": "requirements_snapshot.txt",
            "random_seed": "not_applicable_no_randomness",
            "cache_dir": str(cache_dir),
        },
        "revision_validation": (
            "Cached Parquet SHA-256 and byte count checked against inventory; inventory SHA-256 checked against the supplied prior source manifest. No remote history revalidation in offline mode."
            if offline_mode
            else "Repository head checked before and after retrieval; commit history page sizes and unique SHA count checked. Parquet files are labeled per month as fresh commit-addressed downloads or previously cached files; all are hashed and recorded in the inventory."
        ),
    }
    manifest_path = out_dir / "historical_snapshot_reproduction_manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "commit_count": history_meta["commit_count"],
                "monthly_snapshots": len(inventory_rows),
                "snapshot_months": [row["snapshot_month"] for row in inventory_rows],
                "same_month_p90_matches_final_baseline": p90_match_rows,
                "same_month_p90_comparisons": len(same_month_rows),
                "outputs": [str(path) for path in [inventory_path, cohort_path, same_month_path, report_path, manifest_path]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
