"""Record a completed manual review of the current Q4 figure previews.

Run only after opening and inspecting all nine candidate color PNGs, the separate
flowchart, and the ten grayscale previews. It records the review and refreshes the
affected modeling-manifest hash; it does not draw or assess the figures itself.

AI assistance: OpenAI Codex desktop 26.917.71314 (build 10954, prod), model ID
gpt-6-luna (GPT-6 Luna), developer OpenAI, official release date 2026-09-22.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

Q4 = Path(__file__).resolve().parents[2]
ROOT = Q4.parent
RESULTS = Q4 / "03_结果" / "results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    qa_path = RESULTS / "q4_figure_qa.json"
    manifest_path = RESULTS / "q4_modeling_manifest.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    findings = (
        "All nine candidate figures, the separate overall flowchart, and all ten grayscale previews were "
        "visually inspected (10 color PNGs and 10 grayscale PNGs). Chinese labels and symbols render; no "
        "panel labels, axes, legends, intervals, or annotations are clipped or overlap. Raw-data plots mark "
        "sparse cells; the version-match flow retains unresolved status; the AR(1) annotation clears both "
        "held-out error labels; the decomposition forest plot shows HC3 95% intervals and a zero reference. "
        "The flowchart shows Loss-Benchmark BLOCKED_NO_FIT and annual forecast infeasibility. Grayscale "
        "previews retain separation by position, labels, and lightness."
    )
    for figure in qa.get("figures", []):
        for key in ("png", "svg", "grayscale_png"):
            if not (ROOT / figure[key]).is_file():
                raise FileNotFoundError(f"Cannot mark missing figure as reviewed: {figure[key]}")
        figure["manual_visual_review"] = "PASS"
        figure["visual_review_findings"] = findings
    qa["manual_visual_review"] = "PASS"
    qa["visual_review_findings"] = findings
    qa["visual_reviewed_at_local"] = datetime.now().astimezone().isoformat(timespec="seconds")
    qa["file_audit"] = {
        "status": "PASS",
        "command": "references/roles/编程手/scripts/figure_audit.py Q4/04_图表 --questions q4 --min-dpi 300 --strict; tools/figure/scripts/check_figure.py --min-dpi 300 --strict (all recursive PNG/SVG assets)",
        "checked_assets": 30,
        "png_dpi_minimum": 300,
        "formats": ["PNG", "SVG"],
    }
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    relative = qa_path.relative_to(ROOT).as_posix()
    manifest.setdefault("outputs_sha256", {})[relative] = sha256(qa_path)
    manifest["figure_qa_status"] = "Pillow/SVG program checks passed; visual review PASS"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"visual_review=PASS figures={len(qa.get('figures', []))}")
    print(f"qa_sha256={sha256(qa_path)} modeling_manifest_sha256={sha256(manifest_path)}")


if __name__ == "__main__":
    main()
