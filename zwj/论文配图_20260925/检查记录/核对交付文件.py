from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PDFINFO = Path(r"C:\Users\Zhang Wanjie\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdfinfo.exe")
GROUPS = [
    ROOT / "绘图版" / "核心图表",
    ROOT / "绘图版" / "Q2补充图表",
    ROOT / "绘图版" / "Q3补充图表",
    ROOT / "论文版" / "核心图表",
    ROOT / "论文版" / "Q2补充图表",
    ROOT / "论文版" / "Q3补充图表",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_group(folder: Path) -> dict:
    figures = sorted(p for p in folder.glob("*.png") if not p.stem.endswith("_grayscale"))
    records = []
    problems = []
    for png in figures:
        stem = png.stem
        pdf, svg = folder / f"{stem}.pdf", folder / f"{stem}.svg"
        gray = folder / f"{stem}_grayscale.png"
        with Image.open(png) as image:
            info = image.info.get("dpi", (0, 0))
            dpi = [round(float(v), 1) for v in info]
            pixels = list(image.size)
            mode = image.mode
            image.verify()
        if min(dpi) < 299:
            problems.append(f"{png.name}: PNG DPI {dpi} 低于300")
        if not pdf.is_file():
            problems.append(f"{pdf.name}: 缺少PDF")
        elif PDFINFO.is_file():
            result = subprocess.run([str(PDFINFO), str(pdf)], capture_output=True, text=True, timeout=15)
            if result.returncode != 0 or "Pages:" not in result.stdout:
                problems.append(f"{pdf.name}: PDF无法读取")
        if svg.is_file():
            try:
                ET.parse(svg)
            except ET.ParseError as exc:
                problems.append(f"{svg.name}: SVG解析失败：{exc}")
        elif stem != "q1-share-loss-corr":
            problems.append(f"{svg.name}: 缺少SVG")
        if not gray.is_file():
            problems.append(f"{gray.name}: 缺少灰度预览")
        else:
            with Image.open(gray) as gray_image:
                if gray_image.mode != "L" or gray_image.size != tuple(pixels):
                    problems.append(f"{gray.name}: 灰度预览格式或尺寸异常")
        records.append({
            "name": stem,
            "png_pixels": pixels,
            "png_dpi": dpi,
            "png_mode": mode,
            "png_bytes": png.stat().st_size,
            "png_sha256": sha256(png),
            "pdf_bytes": pdf.stat().st_size if pdf.is_file() else None,
            "svg_bytes": svg.stat().st_size if svg.is_file() else None,
            "gray_png_bytes": gray.stat().st_size if gray.is_file() else None,
        })
    return {"directory": str(folder.relative_to(ROOT)), "figure_count": len(figures), "figures": records, "problems": problems}


def main() -> None:
    groups = [audit_group(folder) for folder in GROUPS]
    paper_text = (ROOT / "GMCM2026" / "paper.tex").read_text(encoding="utf-8")
    active = []
    for line in paper_text.splitlines():
        live = line.split("%", 1)[0]
        active.extend(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", live))
    refs = ROOT / "论文版" / "论文引用文件名"
    missing_refs = [name for name in active if not (refs / name).is_file()]
    report = {
        "active_latex_figure_references": len(active),
        "unique_active_references": len(set(active)),
        "missing_latex_references": missing_refs,
        "groups": groups,
        "all_checks_pass": not missing_refs and all(not group["problems"] for group in groups),
    }
    output = ROOT / "检查记录" / "输出文件审计.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"活动论文引用：{len(active)}；缺失：{len(missing_refs)}")
    for group in groups:
        print(f"{group['directory']}: {group['figure_count']} 张，问题 {len(group['problems'])}")
        for problem in group["problems"]:
            print("  -", problem)
    print("最终状态：", "PASS" if report["all_checks_pass"] else "WARN")


if __name__ == "__main__":
    main()
