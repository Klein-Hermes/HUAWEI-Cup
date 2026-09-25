from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "GMCM2026" / "figures"
TARGETS = [ROOT / "绘图版" / "核心图表", ROOT / "论文版" / "核心图表"]


def package(stem: str, target: Path) -> None:
    source = INPUTS / f"{stem}.png"
    with Image.open(source) as opened:
        raster = np.asarray(opened.convert("RGB"))
    height, width, _ = raster.shape
    target.mkdir(parents=True, exist_ok=True)
    png_path = target / f"{stem}.png"
    Image.fromarray(raster).save(png_path, dpi=(600, 600), optimize=True)

    fig = plt.figure(figsize=(width / 600, height / 600), dpi=600, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(raster, origin="upper", extent=(0, width, height, 0), interpolation="none", aspect="auto")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    fig.savefig(target / f"{stem}.pdf", dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(target / f"{stem}.svg", dpi=600, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    with Image.open(png_path) as opened:
        opened.convert("L").save(target / f"{stem}_grayscale.png", dpi=(600, 600), optimize=True)


if __name__ == "__main__":
    for output in TARGETS:
        for name in ("q1-flow-score", "q1-flow-conflict"):
            package(name, output)
            print(f"封装完成：{output.name}/{name}（来源为附件原图）")
