"""Record a completed manual review of the current Q4 figure previews.

Run only after opening and inspecting the three color PNGs and their grayscale
versions. It records the review and refreshes the affected modeling-manifest
hash; it does not draw or assess the figures itself.

AI assistance: OpenAI Codex desktop 26.917.71314 (build 10954; prod; updater
checked 2026-09-25, up_to_date); developer OpenAI. Model selector name: ; model
version/publication date: . These fields are blank per user instruction.
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
        "All three color and all three grayscale previews reviewed. Raw-data regression line stays within "
        "the 0-100 index range; results-panel bottom label overlap is removed; the near-zero monthly "
        "regression identity term is labeled and explained in the figure contract; labels, axes, legends, "
        "sample annotations, and empty/sparse-cell conventions remain legible at preview scale."
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
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    relative = qa_path.relative_to(ROOT).as_posix()
    manifest.setdefault("outputs_sha256", {})[relative] = sha256(qa_path)
    manifest["figure_qa_status"] = "Pillow/SVG program checks passed; visual review PASS"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"visual_review=PASS figures={len(qa.get('figures', []))}")
    print(f"qa_sha256={sha256(qa_path)} modeling_manifest_sha256={sha256(manifest_path)}")


if __name__ == "__main__":
    main()
