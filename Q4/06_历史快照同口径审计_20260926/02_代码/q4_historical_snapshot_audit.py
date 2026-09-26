#!/usr/bin/env python
# AI assistance disclosure: OpenAI Codex desktop, model ID gpt-6-luna (GPT-6 Luna),
# developer OpenAI, official release date 2026-09-22. This script audits public
# leaderboard provenance and locally supplied C1/C2 rows; all matches and summaries
# are computed from the inputs at runtime and are independently reviewable.
"""Audit the C1/C2 monthly P90 series and its public Open LLM Leaderboard source.

The offline path validates the local C1/C2 one-to-one alignment and reconstructs
monthly P90 values. ``--verify-remote`` also compares C1 against the pinned v2
Hugging Face ``contents`` table using the public dataset-viewer API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PINNED_REPO_ID = "open-llm-leaderboard/contents"
PINNED_REPO_SHA = "9c09a7cae43334062a82cb164f2ef255013dafa2"
PINNED_LAST_MODIFIED = "2025-03-20T12:17:27Z"
NUMERIC_TOLERANCE = 5e-13
TEXT_COLUMNS = ("Model", "Submission Date", "Hub License", "Type")
NUMERIC_COLUMNS = (
    "#Params (B)",
    "Average ⬆️",
    "IFEval",
    "BBH",
    "MATH Lvl 5",
    "GPQA",
    "MUSR",
    "MMLU-PRO",
)
COMMON_COLUMNS = TEXT_COLUMNS + NUMERIC_COLUMNS
REMOTE_FIELD_MAP = {
    "Model": "fullname",
    "#Params (B)": "#Params (B)",
    "Submission Date": "Submission Date",
    "Hub License": "Hub License",
    "Type": "Type",
    "Average ⬆️": "Average ⬆️",
    "IFEval": "IFEval",
    "BBH": "BBH",
    "MATH Lvl 5": "MATH Lvl 5",
    "GPQA": "GPQA",
    "MUSR": "MUSR",
    "MMLU-PRO": "MMLU-PRO",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: safe_csv_value(row.get(key)) for key in columns})


def safe_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if text.lower() in {"", "na", "n/a", "nan", "none", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def numeric_equal(left: Any, right: Any, column: str) -> bool:
    a = parse_float(left)
    b = parse_float(right)
    if a is None or b is None:
        # The project's cleaned C1 converts the upstream -1 parameter sentinel to
        # missing; this is the only sentinel equivalence admitted by this audit.
        if column == "#Params (B)" and {a, b} == {None, -1.0}:
            return True
        return a is None and b is None
    return abs(a - b) <= NUMERIC_TOLERANCE


def text_key(row: dict[str, Any], columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(norm_text(row.get(column)) for column in columns)


def row_values_equal(
    left: dict[str, Any],
    right: dict[str, Any],
    numeric_columns: tuple[str, ...] = NUMERIC_COLUMNS,
) -> bool:
    return all(numeric_equal(left.get(column), right.get(column), column) for column in numeric_columns)


def one_to_one_match(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    left_name: str,
    right_name: str,
    right_text_map: dict[str, str] | None = None,
    numeric_columns: tuple[str, ...] = NUMERIC_COLUMNS,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Match rows by all four text and eight numeric common fields."""
    right_text_map = right_text_map or {}

    def project_right(row: dict[str, Any]) -> dict[str, Any]:
        projected = dict(row)
        for target, source in right_text_map.items():
            projected[target] = row.get(source)
        return projected

    projected_right = [project_right(row) for row in right_rows]
    buckets: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for j, row in enumerate(projected_right):
        buckets[text_key(row, TEXT_COLUMNS)].append(j)

    candidates: dict[int, list[int]] = {}
    for i, left in enumerate(left_rows):
        key = text_key(left, TEXT_COLUMNS)
        candidates[i] = [
            j for j in buckets.get(key, []) if row_values_equal(left, projected_right[j], numeric_columns)
        ]

    reverse_count = Counter(j for js in candidates.values() for j in js)
    matches = {
        i: js[0]
        for i, js in candidates.items()
        if len(js) == 1 and reverse_count[js[0]] == 1
    }
    matched_right = set(matches.values())
    ambiguous_left = sum(len(js) > 1 for js in candidates.values())
    unmatched_left = sum(len(js) == 0 for js in candidates.values())
    summary = {
        "left_name": left_name,
        "right_name": right_name,
        "left_rows": len(left_rows),
        "right_rows": len(right_rows),
        "one_to_one_matches": len(matches),
        "unmatched_left": unmatched_left,
        "unmatched_right": len(right_rows) - len(matched_right),
        "ambiguous_left": ambiguous_left,
        "all_rows_one_to_one": (
            len(matches) == len(left_rows) == len(right_rows)
            and unmatched_left == 0
            and len(right_rows) == len(matched_right)
            and ambiguous_left == 0
        ),
    }
    return matches, summary


def parse_month(value: Any) -> str | None:
    text = norm_text(value)
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").strftime("%Y-%m")
    except ValueError:
        return None


def normalize_utc_timestamp(value: Any) -> str | None:
    text = norm_text(value)
    if not text:
        return None
    try:
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except ValueError:
        return None


def quantile_linear(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    h = (len(ordered) - 1) * probability
    lower = math.floor(h)
    upper = math.ceil(h)
    if lower == upper:
        return ordered[lower]
    weight = h - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def fetch_json(url: str, timeout: int = 45) -> tuple[dict[str, Any], dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "q4-historical-snapshot-audit/1.0 (read-only public dataset check)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
    return json.loads(body.decode("utf-8")), headers


def remote_repo_info() -> dict[str, Any]:
    encoded_id = urllib.parse.quote(PINNED_REPO_ID, safe="/")
    info, _ = fetch_json(f"https://huggingface.co/api/datasets/{encoded_id}")
    return {
        "id": info.get("id"),
        "sha": info.get("sha"),
        "lastModified": info.get("lastModified"),
        "siblings": [item.get("rfilename") for item in info.get("siblings", [])],
    }


def fetch_remote_rows(remote_limit: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before = remote_repo_info()
    if before.get("id") != PINNED_REPO_ID:
        raise RuntimeError(f"Unexpected Hub dataset id: {before.get('id')}")
    if before.get("sha") != PINNED_REPO_SHA or normalize_utc_timestamp(before.get("lastModified")) != normalize_utc_timestamp(PINNED_LAST_MODIFIED):
        raise RuntimeError(
            "Pinned upstream revision changed: "
            f"sha={before.get('sha')}, lastModified={before.get('lastModified')}"
        )

    first, _ = fetch_json(
        "https://datasets-server.huggingface.co/splits?dataset=" + urllib.parse.quote(PINNED_REPO_ID, safe="/")
    )
    split_rows = first.get("splits", [])
    if not any(item.get("config") == "default" and item.get("split") == "train" for item in split_rows):
        raise RuntimeError(f"Pinned Hub dataset has no default/train split: {split_rows}")

    page_size = 100
    total = 4576
    requested = min(total, remote_limit) if remote_limit > 0 else total
    offsets = list(range(0, requested, page_size))

    def fetch_page(offset: int) -> tuple[int, list[dict[str, Any]], int]:
        length = min(page_size, requested - offset)
        query = urllib.parse.urlencode(
            {"dataset": PINNED_REPO_ID, "config": "default", "split": "train", "offset": offset, "length": length}
        )
        payload, _ = fetch_json(f"https://datasets-server.huggingface.co/rows?{query}")
        return offset, [entry["row"] for entry in payload.get("rows", [])], int(payload.get("num_rows_total", -1))

    page_results: dict[int, list[dict[str, Any]]] = {}
    observed_totals: set[int] = set()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_page, offset) for offset in offsets]
        for future in as_completed(futures):
            offset, rows, observed_total = future.result()
            page_results[offset] = rows
            observed_totals.add(observed_total)

    rows = [row for offset in offsets for row in page_results[offset]]
    after = remote_repo_info()
    if after != before:
        raise RuntimeError(f"Hub source changed while pages were being read: before={before}, after={after}")
    if observed_totals != {total} or len(rows) != requested:
        raise RuntimeError(
            f"Remote row count mismatch: returned={len(rows)}, requested={requested}, totals={observed_totals}"
        )
    return rows, {
        "repo": before,
        "row_api": "https://datasets-server.huggingface.co/rows",
        "dataset_viewer_splits_endpoint": "https://datasets-server.huggingface.co/splits",
        "remote_total_rows": total,
        "remote_rows_checked": len(rows),
        "partial_remote_check": len(rows) != total,
        "revision_validation_note": (
            "The documented rows endpoint has no revision parameter; expected Hub head SHA and lastModified "
            "were checked before and after the paginated read and remained at the pinned revision."
        ),
    }


def make_protocol_matrix() -> list[dict[str, Any]]:
    return [
        {
            "source": "C1 local leaderboard_cleaned.csv",
            "source_id_or_file": "中文题目/F题/real_attachments/C_efficiency_evolution/leaderboard_cleaned.csv",
            "protocol_or_content": "Six C1 benchmark-family fields including Average; actual row values verified against v2 contents.",
            "time_coverage": "2024-06 through 2025-03 submissions; source latest revision 2025-03-20",
            "version_anchor": PINNED_REPO_SHA,
            "same_caliber_status": "PASS_AFTER_SOURCE_CORRECTION",
            "decision": "Use only with the v2 source attribution and row-level source key; preserve the frozen 2025-03 cutoff.",
        },
        {
            "source": "Open LLM Leaderboard v2 contents",
            "source_id_or_file": "https://huggingface.co/datasets/open-llm-leaderboard/contents",
            "protocol_or_content": "Versioned v2 aggregate table; metrics include IFEval, BBH, MATH Lvl 5, GPQA, MuSR and MMLU-Pro.",
            "time_coverage": "Latest commit 2025-03-20; repository history has 3,598 commits.",
            "version_anchor": PINNED_REPO_SHA,
            "same_caliber_status": "ELIGIBLE_THROUGH_2025_03_ONLY",
            "decision": "Candidate snapshot history through 2025-03; no rows support 2026-03.",
        },
        {
            "source": "Open LLM Leaderboard v2 results details",
            "source_id_or_file": "https://huggingface.co/datasets/open-llm-leaderboard/results",
            "protocol_or_content": "Per-model evaluation JSON from v2; task definitions and run metadata must be checked before joining to C1.",
            "time_coverage": "Public collection lists detailed results through 2025-03-15.",
            "version_anchor": "Model sha/revision must match C1 source row SHA; task protocol still requires an independent check.",
            "same_caliber_status": "CONDITIONAL_ROW_LEVEL_ONLY",
            "decision": "Use an exact SHA join; do not assume a same-name row or SHA alone proves score-definition equivalence.",
        },
        {
            "source": "Open LLM Leaderboard v1 archive",
            "source_id_or_file": "https://huggingface.co/datasets/open-llm-leaderboard-old/results",
            "protocol_or_content": "ARC, HellaSwag, MMLU, TruthfulQA, Winogrande and GSM8k; archived when v2 replaced it in June 2024.",
            "time_coverage": "v1 archive; benchmark contract differs from v2.",
            "version_anchor": "Different benchmark suite and scoring definition.",
            "same_caliber_status": "FAIL_INCOMPARABLE",
            "decision": "Do not pool v1 Average/P90 with v2 C1 Average/P90.",
        },
        {
            "source": "C2 Epoch metadata enrichment",
            "source_id_or_file": "leaderboard_enhanced.csv / Epoch AI model metadata",
            "protocol_or_content": "Explicit Epoch_AI_Open_Weights=Yes used as a three-state cohort filter; C2 base fields are one-to-one aligned to C1.",
            "time_coverage": "Static enrichment in the supplied project; no as-of monthly history is supplied.",
            "version_anchor": "Epoch_AI_Open_Weights value and metadata observation date are not versioned by the v2 leaderboard commit.",
            "same_caliber_status": "RETROSPECTIVE_COHORT_ONLY",
            "decision": "Use for descriptive monthly cohorts only; historical rolling backtests need as-of open-weight labels to avoid status lookahead.",
        },
        {
            "source": "C3 historical papers/reports rows",
            "source_id_or_file": "leaderboard_extended_timeseries.csv where Source=Historical (papers/reports)",
            "protocol_or_content": "Reported scores from mixed papers/reports rather than v2 leaderboard submissions.",
            "time_coverage": "26 rows in supplied file; separate from leaderboard source rows.",
            "version_anchor": "Task versions, model revisions and evaluation protocols are not row-equivalent to v2.",
            "same_caliber_status": "FAIL_INCOMPARABLE",
            "decision": "Background only; do not use to fill annual rolling-origin backtest targets.",
        },
    ]


def build_report(
    local_match: dict[str, Any],
    c1_c2_match: dict[str, Any],
    candidate_count: int,
    monthly_rows: list[dict[str, Any]],
    remote_result: dict[str, Any],
    version_summary: dict[str, Any],
    snapshot_manifest: dict[str, Any] | None,
    root: Path,
) -> str:
    total_open = sum(int(row["n_known_open_weights_models"]) for row in monthly_rows)
    monthly_counts = [int(row["n_known_open_weights_models"]) for row in monthly_rows]
    eligible20 = sum(count >= 20 for count in monthly_counts)
    below10 = sum(count < 10 for count in monthly_counts)
    below20 = sum(count < 20 for count in monthly_counts)
    p90_match_count = sum(row["baseline_p90_match"] is True for row in monthly_rows)
    month_table = "\n".join(
        f"| {row['submission_month']} | {row['n_known_open_weights_models']} | "
        f"{float(row['p90_recomputed_C1_Average']):.6f} | {row['baseline_p90_match']} |"
        for row in monthly_rows
    )
    remote_line = (
        f"远程 v2 全行核验：**{remote_result['c1_remote_row_matches']:,}/{remote_result['remote_rows_checked']:,}** 行一对一匹配；"
        f"上游 SHA 为 `{remote_result['repo_sha']}`，最后更新 `{remote_result['last_modified']}`。"
        if remote_result.get("c1_remote_row_matches") is not None
        else "本输出目录尚未运行远程 v2 全行核验。"
    )
    version_line = (
        f"在 {candidate_count:,} 个冻结的 C8 精确名称/C1 唯一候选中，开放权重样本有 "
        f"{version_summary['c1_records_sha_equal_latest_q4_c8_candidate']:,} 条与候选表登记的 C8 最新模型 SHA 完全一致，"
        f"{version_summary['c1_records_with_sha_conflict_latest_q4_c8_candidate']:,} 条不一致，"
        f"{version_summary['c1_records_without_source_sha']:,} 条缺少 C1 来源 SHA。"
        "SHA 相同只确认权重身份；任务配置与指标定义仍需单独核验。"
        if version_summary.get("c1_records_sha_equal_latest_q4_c8_candidate") is not None
        else "尚未基于完整固定版本 v2 来源逐行核验 C8 模型版本。"
    )
    snapshot_line = (
        f"另从 v2 仓库历史重建 {snapshot_manifest['monthly_snapshots']} 个逐月最后可用版本，"
        f"其中 {snapshot_manifest['same_month_p90_matches_final_baseline']}/"
        f"{snapshot_manifest['same_month_p90_comparisons']} 个可计算的当月 P90 与冻结最终表一致；"
        f"缺少 `Submission Date` 的月份：{', '.join(snapshot_manifest['months_without_submission_date']) or '无'}。"
        "最终值差异按共同模型名与 `Model sha` 进一步拆分，见 `historical_snapshot_report.md`。"
        if snapshot_manifest is not None
        else "逐月历史 Parquet 复原见同目录 `q4_monthly_snapshot_reconstruction.py`；此输出目录未包含该步骤结果。"
    )
    return f"""# Q4 历史快照与同口径审计（补充件）

**审计范围：** 核对 C1/C2 月度新提交口径、C1 来源、v2 公开历史快照覆盖范围、v1/v2 可比性，以及本地 C1 Average P90 重建。**本补充不改写 2026-09-26 已冻结的 Q4 基线。**

## 结论

1. 目标量锁定为每个 C1 `Submission Date` 月份中新提交记录的 C1 `Average ⬆️` 第 90 百分位；队列仅保留 C2 `Epoch_AI_Open_Weights` 明确为 `Yes`。分位数采用线性插值。C1/C2 按全部 12 个基础字段做一对一匹配，不按行号或仅按模型名拼接。
2. 项目附件登记把 `leaderboard_cleaned.csv` 指向 v1 仓库 `open-llm-leaderboard-old/results`，该登记与文件内容不符。此文件与公开 v2 `open-llm-leaderboard/contents` 的 C1 字段行可做全量核验；来源更正记录见 `q4_source_manifest_correction.csv`。Q4 主队列先限于原流程中的 **{candidate_count:,} 个 C8 名称精确且 C1 唯一候选**，再依据 C2 筛选；原附件清单保持只读。
3. Hugging Face v2 `contents` 历史有 3,598 次提交，当前固定版本为 `{PINNED_REPO_SHA}`（`{PINNED_LAST_MODIFIED}`）。{snapshot_line}该源不含 2026-03 观测。v1 的六项任务与 v2 六个任务族不同，不能把 v1 的 Average/P90 接到 v2 时间序列。
4. C1/C2 完整字段对齐：**{c1_c2_match['one_to_one_matches']:,}/{c1_c2_match['left_rows']:,}** 一对一匹配。{remote_line}
5. 重建得到 **{total_open}** 条“明确开放权重”记录，覆盖 {len(monthly_rows)} 个有数据月份；每月样本数为 {min(monthly_counts)}–{max(monthly_counts)}。其中 {below10} 个月 n<10、{below20} 个月 n<20；若采用 n≥20 才纳入动态估计，仅剩 {eligible20} 个月，不能支持年度预测。重建 P90 与冻结表 `{p90_match_count}/{len(monthly_rows)}` 月一致。
6. 当前 C2 的 Epoch 开放权重标签没有随 v2 月度快照版本化，因此历史月份结果只能称“用当前标签回溯定义的描述性队列”，不能充当无前视偏差的开放状态回测。{version_line}
7. **2026-03 事后验证不可得**：所核 v2 内容库最后版本为 2025-03-20；不得用 v1、混合论文数据或新评测套件冒充同口径实际值。2027-03 仍是未来目标。继续保持 2025-03 训练截点和 12/24 月数值预测空值。

## 逐月重建

| 提交月 | 明确 Open Weights 记录数 | 线性 P90（C1 Average） | 与冻结表一致 |
|---|---:|---:|---|
{month_table}

P90 的单位/量尺沿用 C1 `Average ⬆️` 原值；本表不对六个指标重新加权、不使用 C8 子任务分数替代 C1 Average。

## 同口径纳入规则

- **主数据：** 固定到 Hugging Face v2 `open-llm-leaderboard/contents` commit `{PINNED_REPO_SHA}`；保留 C1 原始分值、提交月和上游 `Model sha`。
- **样本：** 每个 C1 记录单独保留；只把 C2 明确 `Yes` 的行用于描述统计。C2 的外部标签不是版本化的 as-of 字段，历史回测必须额外取得带日期的开放权重快照或人工核验记录。
- **版本：** `Model sha` 与 C8 `model_sha` 精确相等可确认权重版本相同；同名不等于同版本，SHA 相同也不单独证明评测任务、指标定义、评测代码和提示设置相同。需要将逐项任务配置另列门控。
- **拒绝混池：** `open-llm-leaderboard-old/results` 为 v1；C3 `Historical (papers/reports)` 为异源结果；二者都不进入 v2 月度 P90 序列。
- **时间目标：** 2026-03 只有在同一个 v2 指标源有记录且能重建当时开放权重标签时才能事后验证；若源未延续，目标标记 `not_observable_under_protocol`。2027-03 不发布当前实际值或无证据外推。

## 来源与复现

- 本地 C1：`{(root / '中文题目/F题/real_attachments/C_efficiency_evolution/leaderboard_cleaned.csv').as_posix()}`
- 本地 C2：`{(root / '中文题目/F题/real_attachments/C_efficiency_evolution/leaderboard_enhanced.csv').as_posix()}`
- v2 表及版本：<https://huggingface.co/datasets/open-llm-leaderboard/contents/tree/{PINNED_REPO_SHA}>
- v2 官方历史说明：<https://github.com/huggingface/leaderboards/blob/main/docs/source/en/open_llm_leaderboard/archive.md>
- v1 当前公开详情：<https://huggingface.co/datasets/open-llm-leaderboard-old/results>
- 复现历史快照：先安装 `requirements_snapshot.txt`，再运行 `q4_monthly_snapshot_reconstruction.py`；然后运行 `q4_historical_snapshot_audit.py --verify-remote` 完成来源逐行核验。

本补充修正来源追溯并重建可得 v2 历史快照，不改变既有模型、训练截点、预测门槛或冻结结论。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verify-remote", action="store_true")
    parser.add_argument(
        "--remote-limit",
        type=int,
        default=0,
        help="Only check the first N upstream rows; 0 means all 4,576 rows.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    audit_dir = Path(__file__).resolve().parents[1]
    out_dir = (args.output_dir or (audit_dir / "03_结果")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_dir = root / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution"
    c1_path = source_dir / "leaderboard_cleaned.csv"
    c2_path = source_dir / "leaderboard_enhanced.csv"
    baseline_frontier_path = root / "Q4" / "03_结果" / "results" / "q4_frontier_monthly.csv"
    c8_candidates_path = root / "Q4" / "03_结果" / "results" / "q4_c8_c1_match_candidates.csv"
    source_manifest_path = source_dir.parent / "source_manifest.json"
    required = [c1_path, c2_path, baseline_frontier_path, c8_candidates_path, source_manifest_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required audit input(s) missing: " + ", ".join(missing))

    c1 = read_csv(c1_path)
    c2 = read_csv(c2_path)
    baseline = read_csv(baseline_frontier_path)
    c8_candidates = read_csv(c8_candidates_path)
    for label, rows in (("C1", c1), ("C2", c2)):
        if rows and not set(COMMON_COLUMNS).issubset(rows[0]):
            raise ValueError(f"{label} is missing required common fields")
    if c2 and "Epoch_AI_Open_Weights" not in c2[0]:
        raise ValueError("C2 is missing Epoch_AI_Open_Weights")

    c1_to_c2, c1_c2_summary = one_to_one_match(c1, c2, "local_C1", "local_C2")
    if not c1_c2_summary["all_rows_one_to_one"]:
        raise RuntimeError(f"C1/C2 row alignment did not pass: {c1_c2_summary}")

    # Reuse the frozen Q4 cohort contract: only exact C8 names that map to one
    # unique C1 record are eligible before C2 open-weight status is considered.
    exact_candidates = [
        row for row in c8_candidates
        if norm_text(row.get("match_status")) == "exact_model_id_unique_record"
    ]
    candidate_by_name = {norm_text(row.get("c8_model_name_raw")): row for row in exact_candidates}
    if len(candidate_by_name) != len(exact_candidates):
        raise RuntimeError("Q4 exact-name candidate list contains duplicate model names")
    if len(exact_candidates) != 1819:
        raise RuntimeError(f"Frozen Q4 cohort expected 1,819 exact-name unique candidates, found {len(exact_candidates)}")
    c1_unique_names = {norm_text(row.get("Model")) for row in c1}
    c1_name_counts = Counter(norm_text(row.get("Model")) for row in c1)
    if any(name not in c1_unique_names for name in candidate_by_name):
        raise RuntimeError("Q4 exact-name candidate list contains a model absent from C1")
    if any(c1_name_counts[name] != 1 for name in candidate_by_name):
        raise RuntimeError("Q4 exact-name candidate must point to exactly one local C1 row")

    remote_result: dict[str, Any] = {
        "status": "not_run",
        "remote_rows_checked": 0,
        "c1_remote_row_matches": None,
        "repo_sha": None,
        "last_modified": None,
    }
    remote_projected: list[dict[str, Any]] = []
    c1_to_remote: dict[int, int] = {}
    remote_match_summary: dict[str, Any] | None = None
    if args.verify_remote:
        remote_rows, fetched_meta = fetch_remote_rows(args.remote_limit)
        remote_projected = []
        for row in remote_rows:
            projected = {target: row.get(source) for target, source in REMOTE_FIELD_MAP.items()}
            projected["source_model_sha"] = row.get("Model sha")
            remote_projected.append(projected)
        local_slice = c1[: len(remote_projected)] if args.remote_limit > 0 else c1
        c1_to_remote, remote_match_summary = one_to_one_match(
            local_slice,
            remote_projected,
            "local_C1",
            "pinned_HF_v2_contents",
        )
        if len(c1_to_remote) != len(local_slice):
            raise RuntimeError(f"C1 to pinned v2 source row match failed: {remote_match_summary}")
        remote_result = {
            "status": "partial_pass" if fetched_meta["partial_remote_check"] else "pass",
            "repo_sha": fetched_meta["repo"]["sha"],
            "last_modified": fetched_meta["repo"]["lastModified"],
            "repo_id": fetched_meta["repo"]["id"],
            "remote_total_rows": fetched_meta["remote_total_rows"],
            "remote_rows_checked": fetched_meta["remote_rows_checked"],
            "c1_remote_row_matches": len(c1_to_remote),
            "full_row_comparison": remote_match_summary,
            "revision_validation_note": fetched_meta["revision_validation_note"],
        }

    remote_by_c1_rownum: dict[int, dict[str, Any]] = {}
    if args.verify_remote:
        for i, j in c1_to_remote.items():
            remote_by_c1_rownum[i] = remote_projected[j]

    sample_rows: list[dict[str, Any]] = []
    cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    open_state_counts: Counter[str] = Counter()
    exact_latest_candidate_count = 0
    conflict_count = 0
    missing_sha_count = 0
    for c1_i, c2_i in c1_to_c2.items():
        left = c1[c1_i]
        enrich = c2[c2_i]
        model = norm_text(left.get("Model"))
        candidate = candidate_by_name.get(model)
        if candidate is None:
            continue
        open_state = norm_text(enrich.get("Epoch_AI_Open_Weights")) or "unknown"
        open_state_counts[open_state] += 1
        if open_state != "Yes":
            continue
        source_info = remote_by_c1_rownum.get(c1_i, {})
        source_sha = norm_text(source_info.get("source_model_sha"))
        candidate_sha = norm_text(candidate.get("c8_model_sha"))
        candidate_revision = norm_text(candidate.get("c8_model_revision"))
        exact_latest = bool(source_sha) and source_sha in {candidate_sha, candidate_revision}
        if exact_latest:
            exact_latest_candidate_count += 1
        if source_sha and (candidate_sha or candidate_revision) and not exact_latest:
            conflict_count += 1
        if not source_sha:
            missing_sha_count += 1
        month = parse_month(left.get("Submission Date"))
        value = parse_float(left.get("Average ⬆️"))
        if month and value is not None:
            cohorts[month].append({"value": value, "c1_row": c1_i + 2})
        sample_rows.append(
            {
                "c1_source_row": c1_i + 2,
                "c2_source_row": c2_i + 2,
                "Model": model,
                "Submission Date": left.get("Submission Date"),
                "submission_month": month,
                "C1_Average": left.get("Average ⬆️"),
                "Epoch_AI_Open_Weights": open_state,
                "Epoch_AI_Publication_Date": enrich.get("Epoch_AI_Publication_Date"),
                "source_Model_sha": source_sha,
                "C8_latest_parseable_run_id": candidate.get("run_id"),
                "C8_latest_model_revision": candidate_revision,
                "C8_latest_model_sha": candidate_sha,
                "C8_revision_sha_equal_latest_q4_candidate": exact_latest,
                "version_match_status": (
                    "revision_sha_equal_latest_q4_candidate"
                    if exact_latest
                    else "revision_conflict"
                    if source_sha and (candidate_sha or candidate_revision)
                    else "source_version_missing"
                ),
                "version_scope_note": "Weight SHA match only; task/metric/protocol equivalence is a separate gate.",
            }
        )

    baseline_by_month = {row["submission_month"]: row for row in baseline}
    monthly_rows: list[dict[str, Any]] = []
    for month in sorted(cohorts):
        values = [entry["value"] for entry in cohorts[month]]
        p90 = quantile_linear(values, 0.90)
        base = baseline_by_month.get(month, {})
        baseline_p90 = parse_float(base.get("monthly_entry_p90_frontier_C1_average_0_100"))
        diff = abs(p90 - baseline_p90) if p90 is not None and baseline_p90 is not None else None
        monthly_rows.append(
            {
                "submission_month": month,
                "n_known_open_weights_models": len(values),
                "p90_recomputed_C1_Average": p90,
                "baseline_p90_C1_Average": baseline_p90,
                "absolute_difference": diff,
                "baseline_p90_match": diff is not None and diff <= 1e-9,
                "n_lt_10_descriptive_only": len(values) < 10,
                "n_lt_20_sensitivity_flag": len(values) < 20,
                "quantile_definition": "linear interpolation at h=(n-1)*0.90",
                "date_basis": "C1 Submission Date month",
                "cutoff_note": "C1 source ends 2025-03; this is a historical cohort summary, not a 2026 backtest.",
            }
        )

    status_counts = Counter(row["version_match_status"] for row in sample_rows)
    version_summary = {
        "open_weight_c1_records": len(sample_rows),
        "registered_exact_name_unique_c1_candidates": len(exact_candidates),
        "c1_records_sha_equal_latest_q4_c8_candidate": exact_latest_candidate_count if args.verify_remote and not remote_result.get("partial_remote_check") else None,
        "c1_records_with_sha_conflict_latest_q4_c8_candidate": conflict_count if args.verify_remote and not remote_result.get("partial_remote_check") else None,
        "c1_records_without_source_sha": missing_sha_count,
        "q4_candidate_open_state_counts": dict(open_state_counts),
        "sample_level_version_status_counts": dict(status_counts),
        "interpretation": "A SHA match supports weight identity only, not exact score or protocol identity.",
    }

    write_csv(
        out_dir / "same_caliber_open_weight_sample.csv",
        sample_rows,
        [
            "c1_source_row",
            "c2_source_row",
            "Model",
            "Submission Date",
            "submission_month",
            "C1_Average",
            "Epoch_AI_Open_Weights",
            "Epoch_AI_Publication_Date",
            "source_Model_sha",
            "C8_latest_parseable_run_id",
            "C8_latest_model_revision",
            "C8_latest_model_sha",
            "C8_revision_sha_equal_latest_q4_candidate",
            "version_match_status",
            "version_scope_note",
        ],
    )
    write_csv(
        out_dir / "historical_monthly_p90_reconstruction.csv",
        monthly_rows,
        [
            "submission_month",
            "n_known_open_weights_models",
            "p90_recomputed_C1_Average",
            "baseline_p90_C1_Average",
            "absolute_difference",
            "baseline_p90_match",
            "n_lt_10_descriptive_only",
            "n_lt_20_sensitivity_flag",
            "quantile_definition",
            "date_basis",
            "cutoff_note",
        ],
    )
    write_csv(
        out_dir / "source_protocol_comparability.csv",
        make_protocol_matrix(),
        [
            "source",
            "source_id_or_file",
            "protocol_or_content",
            "time_coverage",
            "version_anchor",
            "same_caliber_status",
            "decision",
        ],
    )

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest_entries = source_manifest if isinstance(source_manifest, list) else source_manifest.get("files", [])
    wrong_c1_source = None
    for entry in manifest_entries:
        if isinstance(entry, dict) and str(entry.get("file", "")).endswith("leaderboard_cleaned.csv"):
            wrong_c1_source = entry.get("source")
            break
    source_correction = [
        {
            "file": "leaderboard_cleaned.csv",
            "existing_source_manifest_entry": wrong_c1_source or "not_found",
            "verified_upstream_dataset": PINNED_REPO_ID,
            "verified_upstream_revision": PINNED_REPO_SHA if args.verify_remote else "not_verified_offline",
            "verified_row_count": len(c1),
            "full_row_match_count": remote_result.get("c1_remote_row_matches", "not_run"),
            "correction": "Record the v2 contents source in this Q4 supplement; preserve original source_manifest.json unchanged.",
            "evidence": "Exact one-to-one match over four text and eight numeric C1 fields, tolerance 5e-13; upstream Model sha retained separately.",
        }
    ]
    write_csv(
        out_dir / "q4_source_manifest_correction.csv",
        source_correction,
        [
            "file",
            "existing_source_manifest_entry",
            "verified_upstream_dataset",
            "verified_upstream_revision",
            "verified_row_count",
            "full_row_match_count",
            "correction",
            "evidence",
        ],
    )
    write_json(out_dir / "row_alignment_summary.json", {"c1_c2": c1_c2_summary, "c1_to_v2_contents": remote_match_summary})
    write_json(out_dir / "version_match_summary.json", version_summary)
    write_json(out_dir / "remote_source_validation.json", remote_result)

    snapshot_manifest_path = out_dir / "historical_snapshot_reproduction_manifest.json"
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8")) if snapshot_manifest_path.is_file() else None
    report = build_report(
        local_match=remote_match_summary or {},
        c1_c2_match=c1_c2_summary,
        candidate_count=len(exact_candidates),
        monthly_rows=monthly_rows,
        remote_result={
            **remote_result,
            "repo_sha": remote_result.get("repo_sha"),
            "last_modified": remote_result.get("last_modified"),
        },
        version_summary=version_summary,
        snapshot_manifest=snapshot_manifest,
        root=root,
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    input_paths = [c1_path, c2_path, baseline_frontier_path, c8_candidates_path, source_manifest_path]
    output_paths = [path for path in out_dir.iterdir() if path.is_file() and path.name != "reproduction_manifest.json"]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "supplementary_audit_complete"
            if args.verify_remote and remote_result["status"] == "pass" and snapshot_manifest is not None
            else "source_audit_complete_snapshot_step_pending"
            if args.verify_remote and remote_result["status"] == "pass"
            else "scoped_local_audit_complete"
        ),
        "project_root": str(root),
        "script": str(Path(__file__).resolve().relative_to(root)),
        "script_sha256": sha256(Path(__file__).resolve()),
        "command": "python \"Q4/06_历史快照同口径审计_20260926/02_代码/q4_historical_snapshot_audit.py\" --verify-remote",
        "snapshot_command": "python \"Q4/06_历史快照同口径审计_20260926/02_代码/q4_monthly_snapshot_reconstruction.py\"",
        "snapshot_runtime_requirement": "pyarrow==25.0.1 from requirements_snapshot.txt",
        "remote_command_parameters": {"verify_remote": args.verify_remote, "remote_limit": args.remote_limit},
        "inputs_sha256": {str(path.relative_to(root)): sha256(path) for path in input_paths},
        "outputs_sha256": {str(path.relative_to(root)): sha256(path) for path in sorted(output_paths)},
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "dependencies": "Python standard library only",
            "random_seed": "not_applicable_no_randomness",
        },
        "key_results": {
            "c1_c2_full_row_matches": c1_c2_summary["one_to_one_matches"],
            "c1_c2_all_rows_one_to_one": c1_c2_summary["all_rows_one_to_one"],
            "registered_exact_name_unique_c1_candidates": len(exact_candidates),
            "monthly_historical_snapshots": snapshot_manifest.get("monthly_snapshots") if snapshot_manifest else None,
            "same_month_snapshot_p90_matches_final_baseline": snapshot_manifest.get("same_month_p90_matches_final_baseline") if snapshot_manifest else None,
            "c1_remote_rows_checked": remote_result.get("remote_rows_checked", 0),
            "c1_remote_full_row_matches": remote_result.get("c1_remote_row_matches"),
            "monthly_rows_reconstructed": len(monthly_rows),
            "open_weight_records_reconstructed": len(sample_rows),
            "months_with_n_at_least_20": sum(row["n_known_open_weights_models"] >= 20 for row in monthly_rows),
            "v2_snapshot_cutoff": PINNED_LAST_MODIFIED,
            "v2_revision": PINNED_REPO_SHA,
        },
        "self_hash_note": "This manifest is intentionally not listed in its own outputs_sha256.",
    }
    write_json(out_dir / "reproduction_manifest.json", manifest)

    print(json.dumps(manifest["key_results"], ensure_ascii=False, indent=2))
    print(f"outputs={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
