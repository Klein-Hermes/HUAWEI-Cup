#!/usr/bin/env python3
"""只读复核 Q1.1 已生成的人工双盲评审包。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


TOOL_PATH = Path(__file__).resolve().parents[1] / "q1_1_human_review_tool.py"
SPEC = importlib.util.spec_from_file_location("q1_1_human_review_tool", TOOL_PATH)
TOOL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TOOL)


def fail(message: str) -> None:
    raise TOOL.ContractError(message)


def load_blank_rater_form(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ("blind_id", *TOOL.DIMENSIONS):
            fail(f"评审空表表头非法：{path}")
        rows = list(reader)
    if len(rows) != TOOL.EXPECTED_COUNT:
        fail(f"评审空表行数非法：{path}")
    ids: list[str] = []
    for row in rows:
        blind_id = (row.get("blind_id") or "").strip()
        if not TOOL.ID_PATTERN.fullmatch(blind_id) or blind_id in ids:
            fail(f"评审空表 ID 非法或重复：{blind_id!r}")
        if any((row.get(dimension) or "").strip() for dimension in TOOL.DIMENSIONS):
            fail(f"评审空表意外含有分数：{blind_id}")
        ids.append(blind_id)
    return ids


def verify(packet: Path, form: Path, prepared: Path) -> dict[str, object]:
    canonical_ids, source_texts, source_audit = TOOL.validate_frozen_inputs(packet, form)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("input_audit") != source_audit:
        fail("manifest 输入审计与当前冻结输入不一致")
    if manifest.get("code", {}).get("sha256") != TOOL.sha256_file(TOOL_PATH):
        fail("manifest 代码哈希与当前工具不一致")
    listed_files = manifest.get("outputs")
    if not isinstance(listed_files, list):
        fail("manifest outputs 缺失")
    verified_hashes = 0
    for item in listed_files:
        relative = Path(str(item["path"]))
        target = (prepared / relative).resolve()
        try:
            target.relative_to(prepared.resolve())
        except ValueError:
            fail(f"manifest 输出路径越界：{relative}")
        if not target.is_file() or TOOL.sha256_file(target) != item["sha256"]:
            fail(f"输出文件缺失或哈希不一致：{relative}")
        if target.stat().st_size != int(item["bytes"]):
            fail(f"输出文件大小不一致：{relative}")
        verified_hashes += 1

    rater_results: list[dict[str, object]] = []
    orders: list[list[str]] = []
    for rater in manifest.get("raters", []):
        form_path = prepared / str(rater["form"])
        form_order = load_blank_rater_form(form_path)
        packet_records: list[tuple[str, str]] = []
        for relative in rater["packet_parts"]:
            packet_records.extend(TOOL.parse_blind_packet(prepared / str(relative)))
        packet_order = [blind_id for blind_id, _ in packet_records]
        if packet_order != form_order:
            fail(f"评审 {rater['rater_slot']} 的分卷顺序与评分表不一致")
        if len(packet_order) != TOOL.EXPECTED_COUNT or set(packet_order) != set(canonical_ids):
            fail(f"评审 {rater['rater_slot']} 的 ID 集合或数量不一致")
        if len(set(packet_order)) != TOOL.EXPECTED_COUNT:
            fail(f"评审 {rater['rater_slot']} 的分卷含重复 ID")
        for blind_id, content in packet_records:
            if content != source_texts[blind_id]:
                fail(f"评审 {rater['rater_slot']} 的文本内容不一致：{blind_id}")
        order_hash = hashlib.sha256(("\n".join(packet_order) + "\n").encode("utf-8")).hexdigest()
        if order_hash != rater["order_sha256"]:
            fail(f"评审 {rater['rater_slot']} 的顺序哈希不一致")
        orders.append(packet_order)
        rater_results.append({
            "rater_slot": rater["rater_slot"],
            "rows": len(form_order),
            "blank_score_cells": len(form_order) * len(TOOL.DIMENSIONS),
            "packet_parts": len(rater["packet_parts"]),
            "full_text_matches": len(packet_records),
            "order_sha256": order_hash,
        })
    if len(orders) != 2 or orders[0] == orders[1]:
        fail("两名评审的顺序没有形成两套不同序列")
    return {
        "status": "PASS",
        "label": "人工双盲评审包只读复核（非 AI Judge）",
        "manifest_utf8_ok": True,
        "verified_manifest_output_hashes": verified_hashes,
        "source_content_set_sha256": source_audit["content_set_sha256"],
        "raters": rater_results,
        "restricted_mapping_read": False,
        "scores_generated_or_imputed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--form", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.packet, args.form, args.prepared)
    except (TOOL.ContractError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

