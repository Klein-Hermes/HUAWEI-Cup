# -*- coding: utf-8 -*-
"""给附件来源的三张静态图添加绘图版标题和说明，不改动图面数据。"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TASK_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = TASK_ROOT / "绘图版" / "核心图表"
FIGURES = {
    "q1-share-loss-corr": (
        "训练配比与验证损失的相关系数结构",
        "色阶为 Pearson r；红色行表示没有对应的损失目标列。",
    ),
    "q1-flow-conflict": (
        "评分冲突度分析与处理流程",
        "流程涵盖冲突指标、A1 域内验证、扩展集复核和 Huber/等权评分处理。",
    ),
    "q1-flow-score": (
        "A1–A3 指标整理与质量评分流程",
        "统一指标层级、ID、单位和异常标记后，汇总域内指标并形成评分。",
    ),
}


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    for stem, (title, note) in FIGURES.items():
        path = FIG_DIR / f"{stem}.png"
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"))
        height, width = image.shape[:2]
        fig_width = 7.2
        image_width = fig_width * 0.96
        image_height = image_width * height / width
        fig_height = image_height / 0.76
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.02)
        ax.imshow(image, interpolation="none")
        ax.axis("off")
        fig.suptitle(title, y=0.975, fontsize=11.5, fontweight="bold")
        fig.text(0.5, 0.89, note, ha="center", va="center", fontsize=8.2,
                 color="#777777")
        fig.savefig(path, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(path.with_suffix(".svg"), facecolor="white", bbox_inches="tight", pad_inches=0.03)
        with Image.open(path) as rendered:
            rendered.convert("L").save(FIG_DIR / f"{stem}_grayscale.png", dpi=(600, 600))
        plt.close(fig)
        print(f"完成：{path.name}")


if __name__ == "__main__":
    main()
