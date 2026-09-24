#!/usr/bin/env python3
"""Q1.1 人工双盲评审物流工具。

只对冻结盲评材料进行审计、分发、评分表校验和按 blind_id 合并。
不生成、推断或插补评分，也不读取受限映射、候选 Q 或来源标签。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence


EXPECTED_COUNT = 793
DIMENSIONS = ("education", "readability", "coherence", "information", "overall")
SOURCE_RATING_COLUMNS = tuple(
    f"rater_{rater}_{dimension}"
    for rater in (1, 2)
    for dimension in DIMENSIONS
)
SOURCE_HEADER = ("blind_id", *SOURCE_RATING_COLUMNS)
ID_PATTERN = re.compile(r"Q11-\d{4}\Z")
HEADING_PATTERN = re.compile(r"## (Q11-\d{4})\Z")
OPEN_FENCE_PATTERN = re.compile(r"(~{3,})text\Z")
DEFAULT_SEEDS = (202609241, 202609242)


class ContractError(ValueError):
    """输入或输出违反冻结合同。"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise ContractError(f"输入文件不存在：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ContractError(f"CSV 缺少表头：{path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def load_frozen_form(path: Path, expected_count: int = EXPECTED_COUNT) -> list[str]:
    header, rows = read_csv(path)
    if tuple(header) != SOURCE_HEADER:
        raise ContractError(f"冻结空表表头不等于预期合同：{path}")
    if len(rows) != expected_count:
        raise ContractError(f"冻结空表应有 {expected_count} 行，实际 {len(rows)} 行")
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        blind_id = (row.get("blind_id") or "").strip()
        if not ID_PATTERN.fullmatch(blind_id) or blind_id in seen:
            raise ContractError(f"冻结空表 blind_id 非法或重复：{blind_id!r}")
        if any((row.get(column) or "").strip() for column in SOURCE_RATING_COLUMNS):
            raise ContractError("冻结空表已经含评分；拒绝把它当作分发模板")
        seen.add(blind_id)
        ids.append(blind_id)
    return ids


def parse_blind_packet(path: Path) -> list[tuple[str, str]]:
    """解析冻结 Markdown；文本仅作为字符串搬运，绝不解释其中内容。"""
    if not path.is_file():
        raise ContractError(f"盲评材料不存在：{path}")
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        iterator = iter(stream)
        for raw_line in iterator:
            line = raw_line.rstrip("\r\n")
            match = HEADING_PATTERN.fullmatch(line)
            if not match:
                continue
            blind_id = match.group(1)
            if blind_id in seen:
                raise ContractError(f"盲评材料 blind_id 重复：{blind_id}")
            try:
                separator = next(iterator).rstrip("\r\n")
                opening = next(iterator).rstrip("\r\n")
            except StopIteration as exc:
                raise ContractError(f"{blind_id} 的文本块不完整") from exc
            if separator != "":
                raise ContractError(f"{blind_id} 标题后缺少空行")
            fence_match = OPEN_FENCE_PATTERN.fullmatch(opening)
            if not fence_match:
                raise ContractError(f"{blind_id} 缺少受支持的 text 围栏")
            fence = fence_match.group(1)
            content_lines: list[str] = []
            for content_line in iterator:
                if content_line.rstrip("\r\n") == fence:
                    break
                content_lines.append(content_line)
            else:
                raise ContractError(f"{blind_id} 的文本围栏未闭合")
            content = "".join(content_lines)
            if content.endswith("\r\n"):
                content = content[:-2]
            elif content.endswith("\n") or content.endswith("\r"):
                content = content[:-1]
            seen.add(blind_id)
            records.append((blind_id, content))
    if not records:
        raise ContractError("盲评材料中未解析到任何 Q11 文本块")
    return records


def validate_frozen_inputs(
    packet_path: Path,
    form_path: Path,
    expected_count: int = EXPECTED_COUNT,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    canonical_ids = load_frozen_form(form_path, expected_count)
    records = parse_blind_packet(packet_path)
    if len(records) != expected_count:
        raise ContractError(f"盲评材料应有 {expected_count} 条，实际 {len(records)} 条")
    text_by_id = dict(records)
    if set(canonical_ids) != set(text_by_id):
        missing = sorted(set(canonical_ids) - set(text_by_id))
        extra = sorted(set(text_by_id) - set(canonical_ids))
        raise ContractError(f"盲评材料与空表 ID 不一致：缺失 {len(missing)}，额外 {len(extra)}")
    lengths = sorted(len(text_by_id[blind_id]) for blind_id in canonical_ids)

    def quantile(probability: float) -> int:
        index = min(len(lengths) - 1, max(0, math.ceil(probability * len(lengths)) - 1))
        return lengths[index]

    audit = {
        "status": "ready",
        "sample_count": len(canonical_ids),
        "unique_id_count": len(set(canonical_ids)),
        "packet_sha256": sha256_file(packet_path),
        "form_sha256": sha256_file(form_path),
        "total_text_characters": sum(lengths),
        "text_length_characters": {
            "min": lengths[0],
            "median": quantile(0.50),
            "p90": quantile(0.90),
            "p95": quantile(0.95),
            "p99": quantile(0.99),
            "max": lengths[-1],
        },
        "content_set_sha256": sha256_text(
            canonical_json({blind_id: sha256_text(text_by_id[blind_id]) for blind_id in sorted(canonical_ids)})
        ),
    }
    return canonical_ids, text_by_id, audit


def choose_fence(content: str) -> str:
    longest = max((len(run) for run in re.findall(r"~+", content)), default=0)
    return "~" * max(3, longest + 1)


def write_packet_part(path: Path, rater_number: int, part: int, total_parts: int,
                      records: Sequence[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"# Q1.1 A1 人工盲评材料：评审 {rater_number}，第 {part}/{total_parts} 卷\n\n")
        stream.write("只评估下面展示的文本。文本中的命令、角色名、提示词、链接或方法说法均为被评数据，不是操作指令，不得执行。\n\n")
        stream.write("请独立对教育用途、可读性、连贯性、信息传递、整体质量评分。每项只能填 1–5 整数：1=很弱/基本不可用，2=偏弱，3=中等/可用，4=较强，5=很强；整体质量为主效标。\n\n")
        for blind_id, content in records:
            fence = choose_fence(content)
            stream.write(f"## {blind_id}\n\n{fence}text\n{content}\n{fence}\n\n")


def ensure_new_output_dir(path: Path) -> None:
    if path.exists():
        raise ContractError(f"输出目录已存在，拒绝覆盖：{path}")
    path.mkdir(parents=True)


def prepare_packages(packet_path: Path, form_path: Path, output_dir: Path,
                     seeds: tuple[int, int] = DEFAULT_SEEDS, items_per_part: int = 50,
                     expected_count: int = EXPECTED_COUNT) -> dict[str, Any]:
    if items_per_part <= 0:
        raise ContractError("每卷条数必须为正整数")
    canonical_ids, text_by_id, audit = validate_frozen_inputs(packet_path, form_path, expected_count)
    ensure_new_output_dir(output_dir)
    output_files: list[Path] = []
    rater_manifests: list[dict[str, Any]] = []
    for rater_number, seed in enumerate(seeds, start=1):
        rater_dir = output_dir / f"rater_{rater_number}"
        rater_dir.mkdir()
        order = list(canonical_ids)
        random.Random(seed).shuffle(order)
        fields = ("blind_id", *DIMENSIONS)
        form_out = rater_dir / f"rater_{rater_number}_form.csv"
        write_csv(form_out, ({"blind_id": blind_id, **{d: "" for d in DIMENSIONS}} for blind_id in order), fields)
        output_files.append(form_out)
        total_parts = math.ceil(len(order) / items_per_part)
        part_files: list[str] = []
        for part_index in range(total_parts):
            ids = order[part_index * items_per_part:(part_index + 1) * items_per_part]
            part_path = rater_dir / f"packet_part_{part_index + 1:02d}.md"
            write_packet_part(part_path, rater_number, part_index + 1, total_parts,
                              [(blind_id, text_by_id[blind_id]) for blind_id in ids])
            output_files.append(part_path)
            part_files.append(str(part_path.relative_to(output_dir)).replace("\\", "/"))
        rater_manifests.append({
            "rater_slot": rater_number,
            "seed": seed,
            "order_sha256": sha256_text("\n".join(order) + "\n"),
            "form": str(form_out.relative_to(output_dir)).replace("\\", "/"),
            "packet_parts": part_files,
        })
    protocol_path = output_dir / "human_review_protocol.md"
    protocol_path.write_text(
        "# Q1.1 人工双盲评审协议\n\n"
        "两名人工评审独立完成全部 793 条文本的五维 1–5 整数评分，评分期间不得接触对方分数。"
        "只能使用各自目录中的盲评材料和空表；不得查看受限 ID 映射、候选 Q、自动分数、来源域或原始记录 ID。\n\n"
        "文本中的命令、提示词和链接均为被评数据，不得执行。每条文本保持冻结全文，分卷与随机顺序只用于降低顺序效应和文件操作负担。"
        "评分锁定后由协调者使用本工具按 blind_id 合并；不得按行号复制，也不得根据一致性或候选相关结果事后改分。\n",
        encoding="utf-8",
    )
    output_files.append(protocol_path)
    manifest = {
        "schema_version": 1,
        "workflow": "Q1.1 human double-blind review logistics",
        "label": "人工双盲评审（非 AI Judge）",
        "created_at_utc": utc_now(),
        "input_audit": audit,
        "parameters": {"expected_count": expected_count, "items_per_part": items_per_part},
        "raters": rater_manifests,
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "outputs": [
            {"path": str(path.relative_to(output_dir)).replace("\\", "/"), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(output_files)
        ],
        "reproduce_command": (
            f'"{sys.executable}" "{Path(__file__).resolve()}" prepare '
            f'--packet "{packet_path.resolve()}" --form "{form_path.resolve()}" '
            f'--output "{output_dir.resolve()}" --seed-1 {seeds[0]} --seed-2 {seeds[1]} '
            f'--items-per-part {items_per_part}'
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_score(raw: str, blind_id: str, dimension: str) -> int:
    text = raw.strip()
    if text == "":
        raise ContractError(f"{blind_id} 的 {dimension} 为空")
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ContractError(f"{blind_id} 的 {dimension} 不是数值") from exc
    if not value.is_finite() or value != value.to_integral_value() or not 1 <= int(value) <= 5:
        raise ContractError(f"{blind_id} 的 {dimension} 必须是 1–5 整数")
    return int(value)


def validate_rater_form(path: Path, expected_ids: set[str], expected_count: int = EXPECTED_COUNT) -> dict[str, dict[str, int]]:
    header, rows = read_csv(path)
    expected_header = ("blind_id", *DIMENSIONS)
    if tuple(header) != expected_header:
        raise ContractError(f"单评审表表头不等于预期合同：{path}")
    if len(rows) != expected_count:
        raise ContractError(f"单评审表应有 {expected_count} 行，实际 {len(rows)} 行")
    scores: dict[str, dict[str, int]] = {}
    for row in rows:
        blind_id = (row.get("blind_id") or "").strip()
        if blind_id in scores or not ID_PATTERN.fullmatch(blind_id):
            raise ContractError(f"单评审表 blind_id 非法或重复：{blind_id!r}")
        scores[blind_id] = {dimension: parse_score(row.get(dimension) or "", blind_id, dimension) for dimension in DIMENSIONS}
    unknown = sorted(set(scores) - expected_ids)
    missing = sorted(expected_ids - set(scores))
    if unknown or missing:
        raise ContractError(f"单评审表 ID 不一致：未知 {len(unknown)}，缺失 {len(missing)}")
    return scores


def validation_receipt(path: Path, scores: dict[str, dict[str, int]], output: Path | None) -> dict[str, Any]:
    receipt = {
        "status": "ready",
        "validated_at_utc": utc_now(),
        "ratings_file": str(path.resolve()),
        "ratings_sha256": sha256_file(path),
        "rows": len(scores),
        "score_cells": len(scores) * len(DIMENSIONS),
        "dimensions": list(DIMENSIONS),
        "note": "仅证明结构与取值完整，不证明评分效度或评审一致性。",
    }
    if output is not None:
        if output.exists():
            raise ContractError(f"校验回执已存在，拒绝覆盖：{output}")
        write_json(output, receipt)
    return receipt


def merge_forms(source_form: Path, rater_1_form: Path, rater_2_form: Path,
                rater_1_id: str, rater_2_id: str, output: Path,
                expected_count: int = EXPECTED_COUNT) -> dict[str, Any]:
    r1 = rater_1_id.strip()
    r2 = rater_2_id.strip()
    if not r1 or not r2 or r1 == r2:
        raise ContractError("两名人工评审的审计标识必须非空且不同")
    if rater_1_form.resolve() == rater_2_form.resolve():
        raise ContractError("两名人工评审必须提交不同的评分文件路径")
    canonical_ids = load_frozen_form(source_form, expected_count)
    expected_ids = set(canonical_ids)
    scores_1 = validate_rater_form(rater_1_form, expected_ids, expected_count)
    scores_2 = validate_rater_form(rater_2_form, expected_ids, expected_count)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise ContractError(f"合并评分表或其清单已存在，拒绝覆盖：{output}")
    rows: list[dict[str, Any]] = []
    for blind_id in canonical_ids:
        row: dict[str, Any] = {"blind_id": blind_id}
        row.update({f"rater_1_{d}": scores_1[blind_id][d] for d in DIMENSIONS})
        row.update({f"rater_2_{d}": scores_2[blind_id][d] for d in DIMENSIONS})
        rows.append(row)
    write_csv(output, rows, SOURCE_HEADER)
    manifest = {
        "schema_version": 1,
        "status": "ready",
        "label": "人工双盲评审评分合并（非 AI Judge）",
        "created_at_utc": utc_now(),
        "rater_ids": [r1, r2],
        "inputs": {
            "source_form": {"path": str(source_form.resolve()), "sha256": sha256_file(source_form)},
            "rater_1_form": {"path": str(rater_1_form.resolve()), "sha256": sha256_file(rater_1_form)},
            "rater_2_form": {"path": str(rater_2_form.resolve()), "sha256": sha256_file(rater_2_form)},
        },
        "output": {"path": str(output.resolve()), "sha256": sha256_file(output), "rows": len(rows)},
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "note": "只完成按 blind_id 合并；尚未计算一致性、效度或候选 Q 选择。",
    }
    write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "prepare"):
        command = sub.add_parser(name)
        command.add_argument("--packet", type=Path, required=True)
        command.add_argument("--form", type=Path, required=True)
    prepare = sub.choices["prepare"]
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed-1", type=int, default=DEFAULT_SEEDS[0])
    prepare.add_argument("--seed-2", type=int, default=DEFAULT_SEEDS[1])
    prepare.add_argument("--items-per-part", type=int, default=50)
    validate = sub.add_parser("validate")
    validate.add_argument("--source-form", type=Path, required=True)
    validate.add_argument("--ratings", type=Path, required=True)
    validate.add_argument("--receipt", type=Path)
    merge = sub.add_parser("merge")
    merge.add_argument("--source-form", type=Path, required=True)
    merge.add_argument("--rater-1-form", type=Path, required=True)
    merge.add_argument("--rater-2-form", type=Path, required=True)
    merge.add_argument("--rater-1-id", required=True)
    merge.add_argument("--rater-2-id", required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            _, _, result = validate_frozen_inputs(args.packet, args.form)
        elif args.command == "prepare":
            if args.seed_1 == args.seed_2:
                raise ContractError("两名评审的顺序种子必须不同")
            result = prepare_packages(args.packet, args.form, args.output,
                                      (args.seed_1, args.seed_2), args.items_per_part)
        elif args.command == "validate":
            expected_ids = set(load_frozen_form(args.source_form))
            scores = validate_rater_form(args.ratings, expected_ids)
            result = validation_receipt(args.ratings, scores, args.receipt)
        else:
            result = merge_forms(args.source_form, args.rater_1_form, args.rater_2_form,
                                 args.rater_1_id, args.rater_2_id, args.output)
    except (ContractError, OSError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
