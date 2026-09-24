#!/usr/bin/env python
"""Prepare the minimal, blinded 30-text Q1.1 human spot-check packet.

This script only prepares the 24 representative + 6 diagnostic review forms.
It never assigns ratings, interprets corpus text as instructions, or selects a Q.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation
except ImportError as exc:  # pragma: no cover - message depends on local runtime
    raise SystemExit(
        "缺少 openpyxl。请使用 Codex 工作区 Python runtime，或安装 openpyxl>=3.1。"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
Q11_DIR = ROOT / "results" / "q1_1" / "v1"
PACKET_DIR = Q11_DIR / "review_packet"
Q12_DIR = ROOT / "results" / "q1_2_conflict_model" / "v1"
DEFAULT_OUTPUT = ROOT / "results" / "q1_1" / "human_spotcheck_final"

SEED_SAMPLE = 20260925
SEED_REVIEWER1 = 20260926
SEED_REVIEWER2 = 20260927
N_REPRESENTATIVE = 24
N_DIAGNOSTIC = 6
MAX_REVIEW_TEXT_CHARS = 8000

SCORE_HEADERS = [
    "教育用途（1–5）",
    "可读性（1–5）",
    "连贯性（1–5）",
    "信息传递（1–5）",
    "整体质量（1–5）",
]
WORKBOOK_HEADERS = ["blind_id", "text", *SCORE_HEADERS]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} 不是 JSON 对象")
    return value


def verify_outputs(base_dir: Path, manifest: Mapping[str, object], names: Iterable[str]) -> None:
    expected = manifest.get("outputs_sha256", {})
    if not isinstance(expected, dict):
        raise ValueError(f"{base_dir / 'manifest.json'} 缺少 outputs_sha256")
    for name in names:
        expected_hash = expected.get(name)
        path = base_dir / name
        if not isinstance(expected_hash, str) or not path.is_file():
            raise ValueError(f"冻结运行清单或输入文件缺少 {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{path} 与对应冻结 manifest 的 SHA-256 不符")


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parse_review_packet(path: Path) -> Dict[str, str]:
    """Read the already-blinded review packet without reparsing the raw corpus."""
    text = path.read_text(encoding="utf-8")
    # Normalize only the packet's line endings. Unicode line separators inside
    # text remain content and are not treated as Markdown record boundaries.
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    heading = re.compile(r"^## (Q11-\d{4})$")
    fence_open = re.compile(r"^(~{3,})text$")
    result: Dict[str, str] = {}
    i = 0
    while i < len(lines):
        match = heading.fullmatch(lines[i])
        if not match:
            i += 1
            continue
        blind_id = match.group(1)
        if blind_id in result:
            raise ValueError(f"盲评材料中 blind_id 重复：{blind_id}")
        j = i + 1
        while j < len(lines) and lines[j] == "":
            j += 1
        if j >= len(lines):
            raise ValueError(f"{blind_id} 缺少原文 fenced block")
        opening = fence_open.fullmatch(lines[j])
        if not opening:
            raise ValueError(f"{blind_id} 的原文块格式异常")
        fence = opening.group(1)
        k = j + 1
        content_lines: List[str] = []
        while k < len(lines) and lines[k] != fence:
            content_lines.append(lines[k])
            k += 1
        if k >= len(lines):
            raise ValueError(f"{blind_id} 的原文块没有闭合")
        result[blind_id] = "\n".join(content_lines)
        i = k + 1
    return result


def stable_key(seed: int, label: str, blind_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{label}\0{blind_id}".encode("utf-8")).hexdigest()


def load_frozen_inputs() -> Tuple[List[dict], List[dict], Dict[str, str], dict]:
    q11_manifest_path = Q11_DIR / "manifest.json"
    q11_manifest = read_json(q11_manifest_path)
    verify_outputs(
        Q11_DIR,
        q11_manifest,
        [
            "sample_scores.csv.gz",
            "review_packet/blind_review_samples.md",
            "review_packet/restricted_id_mapping.csv",
            "review_packet/sampling_frame.csv",
        ],
    )

    q12_manifest_path = Q12_DIR / "manifest.json"
    q12_manifest = read_json(q12_manifest_path)
    verify_outputs(Q12_DIR, q12_manifest, ["sample_scores.csv.gz"])

    mapping_path = PACKET_DIR / "restricted_id_mapping.csv"
    frame_path = PACKET_DIR / "sampling_frame.csv"
    packet_path = PACKET_DIR / "blind_review_samples.md"
    mapping = read_csv(mapping_path)
    frame = read_csv(frame_path)
    texts = parse_review_packet(packet_path)
    if len(mapping) != 793 or len(texts) != 793:
        raise ValueError(f"预期 793 条原文映射和盲评文本；实际 {len(mapping)} / {len(texts)}")
    if len({row.get("blind_id", "") for row in mapping}) != len(mapping):
        raise ValueError("restricted_id_mapping.csv 中 blind_id 不唯一")
    if {row["blind_id"] for row in mapping} != set(texts):
        raise ValueError("盲评文本与受限 ID 映射的 blind_id 集合不一致")
    if any(row.get("dataset") != "A1" for row in mapping):
        raise ValueError("盲评映射中出现非 A1 记录")
    if any(not texts[row["blind_id"]].strip() for row in mapping):
        raise ValueError("盲评材料存在空原文，不能静默跳过")

    # Cross-check the first-stage frame against the 793-row mapping.
    counts = Counter(
        (row["source_domain"], row["q_equal_tertile"], row["broad_unicode_flag"])
        for row in mapping
    )
    frame_keys = set()
    for row in frame:
        if row.get("dataset") != "A1":
            continue
        key = (row["source_domain"], row["q_equal_tertile"], row["broad_unicode_flag"])
        n_h = int(row["n_h"])
        if n_h < 0:
            raise ValueError(f"sampling_frame.csv 出现负分层样本数：{key}")
        # The frozen frame intentionally retains empty cells (n_h=0). They
        # must be checked as empty, but are not expected in the 793-row map.
        if n_h == 0:
            if counts.get(key, 0) != 0:
                raise ValueError(f"sampling_frame.csv 空分层在映射中却有记录：{key}")
            continue
        if key in frame_keys:
            raise ValueError(f"sampling_frame.csv 抽样微层重复：{key}")
        frame_keys.add(key)
        if counts[key] != n_h:
            raise ValueError(f"793 条映射与 sampling_frame 的 n_h 不一致：{key}")
    if set(counts) != frame_keys:
        raise ValueError("793 条映射与 sampling_frame 的分层集合不一致")

    ids = {row["raw_id"] for row in mapping}
    candidate_rows: Dict[str, dict] = {}
    with gzip.open(Q11_DIR / "sample_scores.csv.gz", "rt", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("dataset", "").lower() != "a1" or row.get("id") not in ids:
                continue
            raw_id = row["id"]
            if raw_id in candidate_rows:
                raise ValueError(f"Q1.1 样本分数 raw_id 重复：{raw_id}")
            q_equal = float(row["q_equal"])
            q_huber = float(row["q_huber"])
            if not math.isfinite(q_equal) or not math.isfinite(q_huber):
                raise ValueError(f"候选分数不是有限数：{raw_id}")
            if not (0.0 <= q_equal <= 1.0 and 0.0 <= q_huber <= 1.0):
                raise ValueError(f"候选分数超出 [0,1]：{raw_id}")
            candidate_rows[raw_id] = {
                "source_domain": row["source_domain"],
                "q_equal": q_equal,
                "q_huber": q_huber,
            }
    if set(candidate_rows) != ids:
        raise ValueError(f"候选分数和 793 条映射未一一联接：缺少 {len(ids-set(candidate_rows))} 条")

    conflict_rows: Dict[str, dict] = {}
    with gzip.open(Q12_DIR / "sample_scores.csv.gz", "rt", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("dataset", "").lower() != "a1" or row.get("id") not in ids:
                continue
            raw_id = row["id"]
            if raw_id in conflict_rows:
                raise ValueError(f"Q1.2 冲突样本分数 raw_id 重复：{raw_id}")
            conflict = float(row["conflict_intensity"])
            threshold = float(row["a1_domain_threshold"])
            conflict_rows[raw_id] = {
                "conflict_intensity": conflict,
                "high_conflict": str(row["high_conflict"]).strip().lower() in {"true", "1"},
                "a1_domain_threshold": threshold,
                "q_equal": float(row["q_equal_mean"]),
                "q_huber": float(row["q_huber"]),
            }
    if set(conflict_rows) != ids:
        raise ValueError(f"Q1.2 冲突分数和 793 条映射未一一联接：缺少 {len(ids-set(conflict_rows))} 条")

    mapping_by_id = {row["raw_id"]: row for row in mapping}
    for raw_id in ids:
        expected_domain = mapping_by_id[raw_id]["source_domain"]
        if candidate_rows[raw_id]["source_domain"] != expected_domain:
            raise ValueError(f"Q1.1 候选分与原文映射的来源域不一致：{raw_id}")
        if not math.isfinite(conflict_rows[raw_id]["conflict_intensity"]):
            raise ValueError(f"Q1.2 冲突度不是有限值：{raw_id}")
    # Q1.1 and Q1.2 were intended to share the same frozen candidate scores.
    for raw_id in ids:
        a = candidate_rows[raw_id]
        b = conflict_rows[raw_id]
        if abs(a["q_equal"] - b["q_equal"]) > 1e-8 or abs(a["q_huber"] - b["q_huber"]) > 1e-8:
            raise ValueError(f"Q1.1/Q1.2 候选分数不一致：{raw_id}")

    return mapping, frame, texts, {
        "q11_manifest": q11_manifest,
        "q12_manifest": q12_manifest,
        "candidate_rows": candidate_rows,
        "conflict_rows": conflict_rows,
        "mapping_by_id": mapping_by_id,
        "input_paths": {
            "q1_1_manifest": q11_manifest_path,
            "q1_2_manifest": q12_manifest_path,
            "blind_review_samples": packet_path,
            "restricted_id_mapping": mapping_path,
            "sampling_frame": frame_path,
            "q1_1_sample_scores": Q11_DIR / "sample_scores.csv.gz",
            "q1_2_sample_scores": Q12_DIR / "sample_scores.csv.gz",
        },
    }


def allocate_representative(mapping: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, str], int]:
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in mapping:
        groups[(row["source_domain"], row["q_equal_tertile"])].append(row["blind_id"])
    keys = sorted(groups, key=lambda item: (item[0], int(item[1])))
    if len(keys) > N_REPRESENTATIVE:
        raise ValueError(f"非空代表性分层有 {len(keys)} 个，无法保证每层至少 1 条并抽取 24 条")
    quotas = {key: 1 for key in keys}
    remaining = N_REPRESENTATIVE - len(keys)
    if remaining <= 0:
        return quotas

    total = sum(len(groups[key]) for key in keys)
    exact = {key: remaining * len(groups[key]) / total for key in keys}
    for key in keys:
        add = min(len(groups[key]) - quotas[key], math.floor(exact[key]))
        quotas[key] += add
    left = N_REPRESENTATIVE - sum(quotas.values())
    order = sorted(
        keys,
        key=lambda key: (
            -(exact[key] - math.floor(exact[key])),
            key[0],
            int(key[1]),
        ),
    )
    while left > 0:
        moved = False
        for key in order:
            if quotas[key] < len(groups[key]):
                quotas[key] += 1
                left -= 1
                moved = True
                if left == 0:
                    break
        if not moved:
            raise ValueError("代表性抽样框记录不足，无法抽出 24 条不重复样本")
    return quotas


def average_percentile_ranks(values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    n = len(ordered)
    if n <= 1:
        return {identifier: 0.5 for identifier in values}
    result: Dict[str, float] = {}
    i = 0
    while i < n:
        j = i + 1
        while j < n and ordered[j][1] == ordered[i][1]:
            j += 1
        average_rank = (i + j - 1) / 2.0
        percentile = average_rank / (n - 1)
        for k in range(i, j):
            result[ordered[k][0]] = percentile
        i = j
    return result


def pick_samples(
    mapping: Sequence[Mapping[str, str]],
    texts: Mapping[str, str],
    inputs: Mapping[str, object],
) -> Tuple[List[dict], List[dict]]:
    # The project owner approved this limit before resampling, keeping every
    # displayed text comfortably within Excel's cell limit and reviewable.
    reviewable_mapping = [
        row for row in mapping
        if len(texts[row["blind_id"]]) <= MAX_REVIEW_TEXT_CHARS
    ]
    if len(reviewable_mapping) < N_REPRESENTATIVE + N_DIAGNOSTIC:
        raise ValueError(
            f"满足 {MAX_REVIEW_TEXT_CHARS} 字符上限的文本只有 {len(reviewable_mapping)} 条，"
            "不足以完成 24+6 抽样"
        )
    mapping_by_blind = {row["blind_id"]: row for row in mapping}
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in reviewable_mapping:
        groups[(row["source_domain"], row["q_equal_tertile"])].append(row["blind_id"])

    quotas = allocate_representative(reviewable_mapping)
    representative: List[dict] = []
    for key in sorted(quotas, key=lambda item: (item[0], int(item[1]))):
        ranked_ids = sorted(
            groups[key],
            key=lambda blind_id: (stable_key(SEED_SAMPLE, f"representative:{key[0]}:{key[1]}", blind_id), blind_id),
        )
        for blind_id in ranked_ids[:quotas[key]]:
            row = dict(mapping_by_blind[blind_id])
            row["sample_type"] = "representative"
            row["diagnostic_category"] = ""
            representative.append(row)
    if len(representative) != N_REPRESENTATIVE:
        raise ValueError(f"代表性样本数量错误：{len(representative)}")

    selected_blind_ids = {row["blind_id"] for row in representative}
    candidate_rows = inputs["candidate_rows"]
    conflict_rows = inputs["conflict_rows"]
    raw_to_blind = {row["raw_id"]: row["blind_id"] for row in mapping}
    eligible = [row for row in reviewable_mapping if row["blind_id"] not in selected_blind_ids]

    q_equal_pct = average_percentile_ranks({
        raw_to_blind[raw_id]: candidate_rows[raw_id]["q_equal"] for raw_id in raw_to_blind
    })
    q_huber_pct = average_percentile_ranks({
        raw_to_blind[raw_id]: candidate_rows[raw_id]["q_huber"] for raw_id in raw_to_blind
    })
    avg_pct = {blind_id: (q_equal_pct[blind_id] + q_huber_pct[blind_id]) / 2.0 for blind_id in raw_to_blind.values()}
    rank_gap = {blind_id: abs(q_equal_pct[blind_id] - q_huber_pct[blind_id]) for blind_id in raw_to_blind.values()}

    def raw_for(blind_id: str) -> str:
        return mapping_by_blind[blind_id]["raw_id"]

    def ranked(category: str) -> List[str]:
        ids = [row["blind_id"] for row in eligible]
        if category == "candidate_disagreement":
            return sorted(ids, key=lambda bid: (-rank_gap[bid], stable_key(SEED_SAMPLE, category, bid), bid))
        if category == "high_conflict":
            ids = [bid for bid in ids if conflict_rows[raw_for(bid)]["high_conflict"]]
            return sorted(ids, key=lambda bid: (-conflict_rows[raw_for(bid)]["conflict_intensity"], stable_key(SEED_SAMPLE, category, bid), bid))
        if category == "extreme_high":
            return sorted(ids, key=lambda bid: (-avg_pct[bid], stable_key(SEED_SAMPLE, category, bid), bid))
        if category == "extreme_low":
            return sorted(ids, key=lambda bid: (avg_pct[bid], stable_key(SEED_SAMPLE, category, bid), bid))
        raise ValueError(f"未知诊断类别：{category}")

    quotas = [
        ("candidate_disagreement", 2),
        ("high_conflict", 2),
        ("extreme_high", 1),
        ("extreme_low", 1),
    ]
    diagnostic: List[dict] = []
    diagnostic_ids = set(selected_blind_ids)
    assigned_category: Dict[str, str] = {}
    ranked_by_category = {category: ranked(category) for category, _ in quotas}
    category_counts = {category: 0 for category, _ in quotas}

    for category, quota in quotas:
        for blind_id in ranked_by_category[category]:
            if category_counts[category] >= quota:
                break
            if blind_id in diagnostic_ids:
                continue
            diagnostic_ids.add(blind_id)
            assigned_category[blind_id] = category
            category_counts[category] += 1

    # If overlap or an unavailable diagnostic group leaves vacancies, fill in
    # the declared order from the next eligible case within those same groups.
    while len(diagnostic_ids - selected_blind_ids) < N_DIAGNOSTIC:
        changed = False
        for category, _ in quotas:
            for blind_id in ranked_by_category[category]:
                if blind_id in diagnostic_ids:
                    continue
                diagnostic_ids.add(blind_id)
                assigned_category[blind_id] = f"fallback:{category}"
                changed = True
                break
            if len(diagnostic_ids - selected_blind_ids) >= N_DIAGNOSTIC:
                break
        if not changed:
            raise ValueError("诊断候选不足，无法补足 6 条且不重复的样本")

    for blind_id in sorted(diagnostic_ids - selected_blind_ids):
        row = dict(mapping_by_blind[blind_id])
        row["sample_type"] = "diagnostic"
        row["diagnostic_category"] = assigned_category[blind_id]
        diagnostic.append(row)
    if len(diagnostic) != N_DIAGNOSTIC:
        raise ValueError(f"诊断样本数量错误：{len(diagnostic)}")

    all_blind_ids = [row["blind_id"] for row in representative + diagnostic]
    if len(all_blind_ids) != len(set(all_blind_ids)):
        raise ValueError("代表性样本与诊断样本有重复")
    for row in representative + diagnostic:
        if row["blind_id"] not in texts or not texts[row["blind_id"]]:
            raise ValueError(f"抽中样本缺少原文：{row['blind_id']}")
    return representative, diagnostic


def reviewer_order(rows: Sequence[dict], seed: int, reviewer_label: str) -> List[dict]:
    return sorted(
        rows,
        key=lambda row: (stable_key(seed, reviewer_label, row["blind_id"]), row["blind_id"]),
    )


def write_workbook(path: Path, rows: Sequence[dict], texts: Mapping[str, str], seed: int, reviewer: str) -> None:
    ordered = reviewer_order(rows, seed, reviewer)
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "评分"
    worksheet.freeze_panes = "C2"
    worksheet.sheet_view.showGridLines = False
    worksheet.sheet_view.zoomScale = 80
    worksheet.sheet_properties.pageSetUpPr.fitToPage = True
    worksheet.page_setup.orientation = "landscape"
    worksheet.page_setup.fitToWidth = 1
    worksheet.page_setup.fitToHeight = 0
    worksheet.sheet_properties.outlinePr.summaryBelow = True
    worksheet.print_title_rows = "1:1"

    header_fill = PatternFill("solid", fgColor="234A70")
    header_font = Font(name="Microsoft YaHei", color="FFFFFF", bold=True, size=10)
    thin_gray = Side(style="thin", color="D6DEE8")
    for col, header in enumerate(WORKBOOK_HEADERS, start=1):
        cell = worksheet.cell(row=1, column=col, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=Side(style="medium", color="8DA9C4"))
    worksheet.row_dimensions[1].height = 34

    # Keep the five score controls visible beside the ID/text on ordinary
    # laptop screens; the previous 76-character text column pushed them off-screen.
    widths = {"A": 14, "B": 36, "C": 16, "D": 16, "E": 16, "F": 16, "G": 16}
    for column, width in widths.items():
        worksheet.column_dimensions[column].width = width

    validation = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True)
    validation.error = "请填写 1 至 5 的整数。"
    validation.errorTitle = "评分范围错误"
    validation.prompt = "可点击下拉箭头选择 1 至 5，也可直接输入整数。"
    validation.promptTitle = "评分"
    validation.showErrorMessage = True
    validation.showInputMessage = True
    worksheet.add_data_validation(validation)

    body_font = Font(name="Microsoft YaHei", size=10, color="202124")
    score_font = Font(name="Microsoft YaHei", size=11, color="174A7E")
    score_fill = PatternFill("solid", fgColor="FFF2CC")
    text_alignment = Alignment(vertical="top", wrap_text=True)
    score_alignment = Alignment(horizontal="center", vertical="center")
    for row_number, row in enumerate(ordered, start=2):
        blind_id = row["blind_id"]
        text_value = texts[blind_id]
        if len(text_value) > 32767:
            raise ValueError(
                f"{blind_id} 原文长度超过 XLSX 单元格上限，不能截断；需先采用兼容长文本的评审格式。"
            )
        id_cell = worksheet.cell(row=row_number, column=1, value=blind_id)
        text_cell = worksheet.cell(row=row_number, column=2, value=text_value)
        for cell in (id_cell, text_cell):
            cell.font = body_font
            cell.alignment = text_alignment
            cell.protection = cell.protection.copy(locked=True)
            cell.data_type = "s"  # Keep corpus text literal; never create Excel formulas.
        id_cell.alignment = Alignment(vertical="top", horizontal="center")
        for col in range(3, 8):
            cell = worksheet.cell(row=row_number, column=col, value=None)
            cell.font = score_font
            cell.fill = score_fill
            cell.alignment = score_alignment
            cell.protection = cell.protection.copy(locked=False)
            cell.number_format = "0"
            validation.add(cell)
        worksheet.row_dimensions[row_number].height = min(390, max(54, 15 * (1 + text_value.count("\n") + len(text_value) // 100)))
        for col in range(1, 8):
            worksheet.cell(row=row_number, column=col).border = Border(bottom=thin_gray)

    # Open the protected sheet with the first editable score cell selected.
    # With freeze_panes=C2, rating cells are in the bottom-right pane.
    for selection in worksheet.sheet_view.selection:
        if selection.pane == "bottomRight":
            selection.activeCell = "C2"
            selection.sqref = "C2"

    worksheet.protection.sheet = True
    worksheet.protection.selectLockedCells = False
    worksheet.protection.selectUnlockedCells = True
    workbook.properties.creator = "Q1.1 review preparation"
    workbook.properties.title = "Text quality review form"
    workbook.properties.subject = "Independent text quality ratings"
    workbook.properties.description = "Blind review form. Ratings are entered by the assigned reviewer."
    workbook.properties.keywords = ""
    workbook.save(path)


def make_private_manifest(
    representative: Sequence[dict],
    diagnostic: Sequence[dict],
    texts: Mapping[str, str],
    inputs: Mapping[str, object],
) -> List[dict]:
    candidate_rows = inputs["candidate_rows"]
    conflict_rows = inputs["conflict_rows"]
    result = []
    for row in list(representative) + list(diagnostic):
        blind_id = row["blind_id"]
        raw_id = row["raw_id"]
        candidate = candidate_rows[raw_id]
        conflict = conflict_rows[raw_id]
        result.append({
            "blind_id": blind_id,
            "sample_type": row["sample_type"],
            "diagnostic_category": row["diagnostic_category"],
            "dataset": row["dataset"],
            "source_domain": row["source_domain"],
            "raw_id": raw_id,
            "sub_path": row.get("sub_path", ""),
            "q_equal_tertile": row["q_equal_tertile"],
            "broad_unicode_flag": row["broad_unicode_flag"],
            "q_equal": f"{candidate['q_equal']:.17g}",
            "q_huber": f"{candidate['q_huber']:.17g}",
            "high_conflict": str(bool(conflict["high_conflict"])).lower(),
            "conflict_intensity": f"{conflict['conflict_intensity']:.17g}",
            "a1_domain_threshold": f"{conflict['a1_domain_threshold']:.17g}",
            "sampling_seed": SEED_SAMPLE,
            "content_sha256": row["content_sha256"],
            "displayed_text_sha256": sha256_text(texts[blind_id]),
            "displayed_text_char_count": len(texts[blind_id]),
            "max_review_text_chars": MAX_REVIEW_TEXT_CHARS,
        })
    return result


def build_summary(
    representative: Sequence[dict],
    diagnostic: Sequence[dict],
    inputs: Mapping[str, object],
    total_mapping_count: int,
    eligible_count: int,
    eligible_domain_counts: Mapping[str, int],
    source_domain_count: int,
) -> str:
    mapping_counts = Counter(row["source_domain"] for row in representative)
    category_counts = Counter(row["diagnostic_category"] for row in diagnostic)
    input_lines = []
    for label, path in sorted(inputs["input_paths"].items()):
        input_lines.append(f"- {label}: `{sha256_file(path)}`")
    return (
        "# Q1.1 小样本双人盲评准备状态\n\n"
        "状态：**等待两位评审**。本文件夹目前只包含两份空白盲评表、私有抽样清单和本摘要；尚未产生人工评分、一致性统计、候选比较或主 Q 决策。\n\n"
        "## 冻结抽样\n\n"
        f"- Representative：{len(representative)} 条；Diagnostic：{len(diagnostic)} 条；总数：{len(representative)+len(diagnostic)} 条。\n"
        f"- Representative 来源域计数：{json.dumps(dict(sorted(mapping_counts.items())), ensure_ascii=False)}\n"
        f"- Diagnostic 类别计数：{json.dumps(dict(sorted(category_counts.items())), ensure_ascii=False)}\n"
        f"- 长度资格：评审表展示文本不超过 {MAX_REVIEW_TEXT_CHARS:,} 字符；从 {total_mapping_count} 条中有 {eligible_count} 条符合，"
        f"{total_mapping_count - eligible_count} 条因长度限制不参与本次人工抽样。\n"
        "- Representative 在符合长度限制的子集中按“来源域 × 原冻结 q_equal 三分位”分层抽样；Diagnostic 也只从该子集中选择，候选百分位仍按冻结的 A1 793 条计算。\n"
        f"- 符合长度限制的来源域覆盖 {sum(count > 0 for count in eligible_domain_counts.values())}/{source_domain_count} 个；各域数量：{json.dumps(dict(sorted(eligible_domain_counts.items())), ensure_ascii=False)}。\n"
        "- 因长度筛选，本次人工核查仅适用于符合长度限制的 A1 子集；未覆盖的来源域和超长文本不在人工核查范围内，不能表述为完整 793 条原文均经人工核查。\n"
        f"- 抽样种子：{SEED_SAMPLE}；评审 1 顺序种子：{SEED_REVIEWER1}；评审 2 顺序种子：{SEED_REVIEWER2}。\n"
        "- 两份表包含完全相同的 30 个 blind_id，顺序分别独立随机化。\n\n"
        "- 评分格为浅黄色；选中评分格后可使用 1–5 下拉选项，也可直接输入整数。打开工作簿后会定位到第一个评分格。\n\n"
        "## 人工下一步\n\n"
        "指定的评审人 1、评审人 2 各自只填写自己的 XLSX 和五项评分；不要交换答案或查看 sampling_manifest_private.csv。两份表回收后，再运行 analyze 阶段。\n\n"
        "若 Excel 网格中某个 text 单元格没有显示全文，请选中该单元格并展开公式栏阅读完整内容，再评分；不要只依据当前可见部分评分。\n\n"
        "评审表使用现有冻结盲评材料中的展示文本；上游材料将不可见控制/格式字符写成可见的 Unicode 转义序列。私有清单分别保存源内容哈希和实际展示文本哈希，二者口径不同。\n\n"
        "建模手/项目负责人需在后续分析完成后结合既有 Q1.1/Q1.2 证据作最终主 Q 决策。编程手不得自行选择候选。\n\n"
        "## 复现\n\n"
        f"- 脚本 SHA-256：`{sha256_file(Path(__file__).resolve())}`\n"
        "- 命令：`python src/q1_1_human_spotcheck_final.py prepare --output-dir results/q1_1/human_spotcheck_final`\n\n"
        "## 输入文件 SHA-256\n\n"
        + "\n".join(input_lines)
        + "\n"
    )


def prepare(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"输出目录已存在，为保护可能已回填的评分，不覆盖：{output_dir}。请先检查现有文件。"
        )
    if not output_dir.parent.is_dir():
        output_dir.parent.mkdir(parents=True, exist_ok=True)

    mapping, _frame, texts, inputs = load_frozen_inputs()
    representative, diagnostic = pick_samples(mapping, texts, inputs)
    if len(representative) != N_REPRESENTATIVE or len(diagnostic) != N_DIAGNOSTIC:
        raise ValueError("抽样数量不符合冻结的 24+6 方案")

    all_rows = representative + diagnostic
    all_ids = {row["blind_id"] for row in all_rows}
    if len(all_ids) != 30:
        raise ValueError("抽样 blind_id 不唯一")
    oversized = [
        (row["blind_id"], len(texts[row["blind_id"]]))
        for row in all_rows
        if len(texts[row["blind_id"]]) > 32767
    ]
    if oversized:
        detail = ", ".join(f"{blind_id}:{length}" for blind_id, length in oversized)
        raise ValueError(
            f"长度筛选后仍有 {len(oversized)} 条超过 XLSX 单元格限制（{detail}）；未生成评审表"
        )
    # The separately randomized orders should not accidentally match.
    order1 = [row["blind_id"] for row in reviewer_order(all_rows, SEED_REVIEWER1, "reviewer1")]
    order2 = [row["blind_id"] for row in reviewer_order(all_rows, SEED_REVIEWER2, "reviewer2")]
    if order1 == order2:
        raise ValueError("两位评审的随机顺序意外相同；未生成盲评包")

    with tempfile.TemporaryDirectory(prefix=".q1_1_spotcheck_", dir=str(output_dir.parent)) as temp_name:
        staging = Path(temp_name)
        write_workbook(staging / "reviewer1.xlsx", all_rows, texts, SEED_REVIEWER1, "reviewer1")
        write_workbook(staging / "reviewer2.xlsx", all_rows, texts, SEED_REVIEWER2, "reviewer2")
        private_rows = make_private_manifest(representative, diagnostic, texts, inputs)
        write_csv(
            staging / "sampling_manifest_private.csv",
            private_rows,
            [
                "blind_id", "sample_type", "diagnostic_category", "dataset", "source_domain", "raw_id",
                "sub_path", "q_equal_tertile", "broad_unicode_flag", "q_equal", "q_huber",
                "high_conflict", "conflict_intensity", "a1_domain_threshold", "sampling_seed",
                "content_sha256", "displayed_text_sha256", "displayed_text_char_count",
                "max_review_text_chars",
            ],
        )
        eligible_count = sum(
            len(texts[row["blind_id"]]) <= MAX_REVIEW_TEXT_CHARS for row in mapping
        )
        source_domains = {row["source_domain"] for row in mapping}
        eligible_domain_counts = {
            domain: sum(
                row["source_domain"] == domain
                and len(texts[row["blind_id"]]) <= MAX_REVIEW_TEXT_CHARS
                for row in mapping
            )
            for domain in source_domains
        }
        source_domain_count = len(source_domains)
        summary = build_summary(
            representative, diagnostic, inputs,
            total_mapping_count=len(mapping),
            eligible_count=eligible_count,
            eligible_domain_counts=eligible_domain_counts,
            source_domain_count=source_domain_count,
        )
        (staging / "summary.md").write_text(summary, encoding="utf-8")
        staging.rename(output_dir)

    print(f"状态：awaiting_human")
    print(f"Representative：{len(representative)}")
    print(f"Diagnostic：{len(diagnostic)}")
    print(f"输出目录：{output_dir}")
    print("已生成空白盲评表；未生成任何人工评分或候选决策。")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare", help="生成冻结的 24+6 双人盲评包")
    prep.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"输出目录（默认：{DEFAULT_OUTPUT}）；若目录已存在则拒绝覆盖",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.output_dir)
            return 0
    except Exception as exc:
        print(f"prepare 失败：{exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
