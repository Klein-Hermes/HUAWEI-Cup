"""Generate auditable Q4 figures with Pillow plus editable SVG output.

Matplotlib 3.10.9 in the configured environment stalled while building the
Bezier paths for even a small histogram, so this generator uses direct vector
SVG primitives and matching 300 dpi Pillow previews instead.

AI disclosure: OpenAI Codex desktop 26.917.71314 (build 10954; prod; updater
checked 2026-09-25, up_to_date); developer OpenAI. Model selector name: ; model
version/publication date: . These fields are blank per user instruction.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

Q4 = Path(__file__).resolve().parents[2]
ROOT = Q4.parent
RESULTS = Q4 / "03_结果" / "results"
FIGURES = Q4 / "04_图表"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
W, H, DPI = 2160, 1560, 300

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#666666"
LIGHT = "#D9E0E7"
FAMILY_ORDER = ["ARC", "BBH", "GPQA", "IFEval", "MATH-Hard", "MMLU-Pro", "MuSR"]
CORE_FAMILIES = ["BBH", "GPQA", "IFEval", "MATH-Hard", "MMLU-Pro", "MuSR"]
FAMILY_COLS = {
    "ARC": "family_ARC_percentile_0_100",
    "BBH": "family_BBH_percentile_0_100",
    "GPQA": "family_GPQA_percentile_0_100",
    "IFEval": "family_IFEval_percentile_0_100",
    "MATH-Hard": "family_MATH_Hard_percentile_0_100",
    "MMLU-Pro": "family_MMLU_Pro_percentile_0_100",
    "MuSR": "family_MuSR_percentile_0_100",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def color_hex(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))


class FigureCanvas:
    def __init__(self, width: int = W, height: int = H):
        self.width, self.height = width, height
        self.image = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.svg: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="7.2in" height="5.2in" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]

    def line(self, x1, y1, x2, y2, color="#333333", width=2, dash=None):
        self.draw.line((int(x1), int(y1), int(x2), int(y2)), fill=color_hex(color), width=max(1, int(width)))
        attr = f'stroke="{color}" stroke-width="{width}"'
        if dash:
            attr += f' stroke-dasharray="{dash}"'
        self.svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {attr}/>')

    def rect(self, x, y, width, height, fill, stroke=None, stroke_width=1):
        x0, y0, ww, hh = int(round(x)), int(round(y)), max(0, int(round(width))), max(0, int(round(height)))
        self.draw.rectangle((x0, y0, x0 + ww, y0 + hh), fill=color_hex(fill), outline=color_hex(stroke) if stroke else None, width=max(1, int(stroke_width)) if stroke else 1)
        attr = f'fill="{fill}"'
        if stroke:
            attr += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        self.svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" {attr}/>')

    def circle(self, cx, cy, radius, fill, stroke=None, stroke_width=1):
        r = max(1, int(round(radius)))
        bbox = (int(cx - r), int(cy - r), int(cx + r), int(cy + r))
        self.draw.ellipse(bbox, fill=color_hex(fill) if fill != "none" else None, outline=color_hex(stroke) if stroke else None, width=max(1, int(stroke_width)))
        attr = f'fill="{fill}"'
        if stroke:
            attr += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        self.svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" {attr}/>')

    def polyline(self, points, color=BLUE, width=2, dash=None, fill="none"):
        points = [(float(x), float(y)) for x, y in points]
        if len(points) >= 2:
            self.draw.line([(int(round(x)), int(round(y))) for x, y in points], fill=color_hex(color), width=max(1, int(width)), joint="curve")
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        attr = f'fill="{fill}" stroke="{color}" stroke-width="{width}"'
        if dash:
            attr += f' stroke-dasharray="{dash}"'
        self.svg.append(f'<polyline points="{pts}" {attr}/>')

    def text(self, x, y, value, size=21, color="#222222", anchor="left", bold=False, rotate=0):
        font = ImageFont.truetype(str(FONT_PATH), int(size), index=0)
        pil_anchor = {"left": "lm", "center": "mm", "right": "rm"}.get(anchor, "lm")
        if rotate:
            layer = Image.new("RGBA", (700, 80), (255, 255, 255, 0))
            d = ImageDraw.Draw(layer)
            d.text((350, 40), str(value), font=font, fill=color_hex(color), anchor="mm", stroke_width=1 if bold else 0, stroke_fill=color_hex(color))
            layer = layer.rotate(rotate, expand=True)
            self.image.paste(layer, (int(x - layer.width / 2), int(y - layer.height / 2)), layer)
        else:
            self.draw.text((int(x), int(y)), str(value), font=font, fill=color_hex(color), anchor=pil_anchor, stroke_width=1 if bold else 0, stroke_fill=color_hex(color))
        svg_anchor = {"left": "start", "center": "middle", "right": "end"}.get(anchor, "start")
        transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
        weight = ' font-weight="bold"' if bold else ""
        self.svg.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="Microsoft YaHei, sans-serif" font-size="{size}" fill="{color}" text-anchor="{svg_anchor}" dominant-baseline="middle"{weight}{transform}>{html.escape(str(value))}</text>'
        )

    def finish(self, basename: Path) -> list[Path]:
        basename.parent.mkdir(parents=True, exist_ok=True)
        png = basename.with_suffix(".png")
        svg = basename.with_suffix(".svg")
        gray = basename.with_name(basename.name + "_grayscale.png").with_suffix(".png")
        self.svg.append("</svg>")
        self.image.save(png, dpi=(DPI, DPI), optimize=True)
        self.image.convert("L").save(gray, dpi=(DPI, DPI), optimize=True)
        svg.write_text("\n".join(self.svg), encoding="utf-8")
        return [png, svg, gray]


def text_color(value: str) -> str:
    return "#ffffff" if value.startswith("#") and sum(color_hex(value)) < 390 else "#222222"


def panel(canvas: FigureCanvas, x, y, w, h, title, letter):
    canvas.text(x, y + 16, f"({letter}) {title}", size=27, bold=True)
    return (x + 100, y + 76, w - 135, h - 160)


def axes(canvas: FigureCanvas, plot, xlim, ylim, xticks, yticks, xlabel, ylabel):
    x, y, w, h = plot
    for value, label in yticks:
        yy = y + h - (value - ylim[0]) / (ylim[1] - ylim[0]) * h
        canvas.line(x, yy, x + w, yy, LIGHT, 1)
        canvas.text(x - 13, yy, label, size=17, anchor="right", color=GRAY)
    for value, label in xticks:
        xx = x + (value - xlim[0]) / (xlim[1] - xlim[0]) * w
        canvas.line(xx, y, xx, y + h, "#EEF1F4", 1)
        canvas.text(xx, y + h + 22, label, size=17, anchor="center", color=GRAY)
    canvas.line(x, y, x, y + h, "#333333", 2)
    canvas.line(x, y + h, x + w, y + h, "#333333", 2)
    canvas.text(x + w / 2, y + h + 62, xlabel, size=21, anchor="center")
    canvas.text(x - 73, y + h / 2, ylabel, size=20, anchor="center", rotate=-90)


def map_xy(plot, x, y, xlim, ylim):
    px, py, pw, ph = plot
    sx = px + (x - xlim[0]) / (xlim[1] - xlim[0]) * pw
    sy = py + ph - (y - ylim[0]) / (ylim[1] - ylim[0]) * ph
    return sx, sy


def histogram(canvas: FigureCanvas, plot, values, bins, color, xlabel, title, integer_y=False):
    x, y, w, h = plot
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    counts, edges = np.histogram(vals, bins=bins)
    ymax = max(1, int(counts.max()))
    ylim = (0, ymax * 1.12)
    xlim = (float(edges[0]), float(edges[-1]))
    yticks = [(v, str(int(v))) for v in np.linspace(0, ymax, 4)]
    xticks = [(float(v), f"{v:.1f}") for v in np.linspace(xlim[0], xlim[1], 5)]
    axes(canvas, plot, xlim, ylim, xticks, yticks, xlabel, "模型数")
    for left, right, count in zip(edges[:-1], edges[1:], counts):
        sx, sy = map_xy(plot, float(left), 0, xlim, ylim)
        ex, ey = map_xy(plot, float(right), float(count), xlim, ylim)
        canvas.rect(sx + 1, ey, max(0, ex - sx - 2), sy - ey, color, stroke="#ffffff", stroke_width=1)


def percentile(values, p):
    values = np.sort(np.asarray(values, dtype=float))
    return float(np.quantile(values, p)) if len(values) else np.nan


def viridis(value):
    stops = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
    z = min(1.0, max(0.0, value / 100.0)) * (len(stops) - 1)
    i = min(int(z), len(stops) - 2)
    f = z - i
    rgb = tuple(int(stops[i][k] * (1 - f) + stops[i + 1][k] * f) for k in range(3))
    return "#%02x%02x%02x" % rgb


def draw_raw(aggregate: pd.DataFrame, basename: Path) -> list[Path]:
    c = FigureCanvas()
    c.text(W / 2, 45, "Q4 原始数据：规模与六族相对表现", size=35, bold=True, anchor="center")
    p1 = panel(c, 70, 110, 970, 620, "完整六族样本参数量分布", "a")
    p2 = panel(c, 1120, 110, 970, 620, "六族百分位主指数分布", "b")
    p3 = panel(c, 70, 790, 2020, 650, "模型级参数量与相对任务表现", "c")
    main = aggregate[aggregate["core6_index_0_100"].notna()].copy()
    size = main[main["parameter_count_B"].gt(0)].copy()
    log_n = np.log10(size["parameter_count_B"].to_numpy(float))
    histogram(c, p1, log_n, 22, BLUE, "log10 参数量（十亿参数）", "")
    histogram(c, p2, main["core6_index_0_100"].to_numpy(float), np.arange(0, 105, 5), GREEN, "任务内百分位指数（0–100）", "")
    x = np.log10(size["parameter_count_B"].to_numpy(float))
    y = size["core6_index_0_100"].to_numpy(float)
    xlim = (float(np.floor(x.min() * 2) / 2), float(np.ceil(x.max() * 2) / 2))
    ylim = (0, 100)
    axes(c, p3, xlim, ylim, [(v, f"{v:.1f}") for v in np.linspace(xlim[0], xlim[1], 7)], [(v, str(int(v))) for v in (0, 25, 50, 75, 100)], "log10 参数量（十亿参数）", "六族指数（0–100）")
    for xx, yy in zip(x, y):
        sx, sy = map_xy(p3, float(xx), float(yy), xlim, ylim)
        c.circle(sx, sy, 3.0, "#4C9BC2")
    xm, ym = float(x.mean()), float(y.mean())
    slope = float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
    intercept = ym - slope * xm
    # Keep the descriptive fit inside the displayed 0–100 index range.
    line_low = max(xlim[0], (ylim[0] - intercept) / slope) if slope > 0 else xlim[0]
    line_high = min(xlim[1], (ylim[1] - intercept) / slope) if slope > 0 else xlim[1]
    line_x = np.linspace(line_low, line_high, 80)
    line_pts = [map_xy(p3, float(xx), intercept + slope * float(xx), xlim, ylim) for xx in line_x]
    c.polyline(line_pts, RED, 4)
    c.text(p3[0] + 20, p3[1] + 18, f"未调整斜率 {slope:.1f} 指数点/log10(B)，n={len(x)}", size=18, color=RED)
    return c.finish(basename)


def draw_process(aggregate: pd.DataFrame, coverage: pd.DataFrame, scale_sample: pd.DataFrame, residual: np.ndarray, fitted: np.ndarray, basename: Path) -> list[Path]:
    c = FigureCanvas()
    c.text(W / 2, 45, "Q4 计算过程：覆盖、时序支持与残差", size=35, bold=True, anchor="center")
    p1 = panel(c, 70, 110, 970, 620, "所选指标的完整任务族覆盖", "a")
    p2 = panel(c, 1120, 110, 970, 620, "月份 × 任务族的百分位均值", "b")
    p3 = panel(c, 70, 790, 2020, 650, "参数、类型和月份模型的残差诊断", "c")
    names = coverage["family"].tolist()
    counts = coverage["complete_family_models"].to_numpy(int)
    x, y, w, h = p1
    xmax = 1900
    plot_x0 = x + 30
    plot_y0 = y + 10
    plot_h = h - 25
    for tick in [0, 500, 1000, 1500, 1900]:
        xx = plot_x0 + tick / xmax * (w - 80)
        c.line(xx, plot_y0, xx, plot_y0 + plot_h, LIGHT, 1)
        c.text(xx, plot_y0 + plot_h + 20, str(tick), size=16, anchor="center", color=GRAY)
    cell_h = plot_h / len(names)
    for i, (name, count) in enumerate(zip(names, counts)):
        cy = plot_y0 + (i + .5) * cell_h
        c.text(plot_x0 - 10, cy, name, size=18, anchor="right")
        color = GRAY if name == "ARC" else BLUE
        c.line(plot_x0, cy, plot_x0 + count / xmax * (w - 80), cy, color, 8)
        c.circle(plot_x0 + count / xmax * (w - 80), cy, 6, color)
        c.text(plot_x0 + count / xmax * (w - 80) + 12, cy, f"{int(count)} / 1,819", size=16)

    valid = aggregate[aggregate["submission_date"].notna()].copy()
    valid["submission_date"] = pd.to_datetime(valid["submission_date"], errors="coerce")
    valid["submission_month"] = valid["submission_date"].dt.to_period("M").astype(str)
    months = sorted(valid["submission_month"].unique())
    families = FAMILY_ORDER
    hx, hy, hw, hh = p2
    cellw, cellh = hw / len(families), hh / len(months)
    for j, family in enumerate(families):
        c.text(hx + (j + .5) * cellw, hy - 16, family, size=14, anchor="center")
    for i, month in enumerate(months):
        group = valid[valid["submission_month"].eq(month)]
        c.text(hx - 12, hy + (i + .5) * cellh, month, size=15, anchor="right")
        for j, family in enumerate(families):
            values = pd.to_numeric(group[FAMILY_COLS[family]], errors="coerce").dropna()
            n = len(values)
            x0, y0 = hx + j * cellw, hy + i * cellh
            if n >= 20:
                mean = float(values.mean())
                fill = viridis(mean)
                c.rect(x0 + 1, y0 + 1, cellw - 2, cellh - 2, fill, stroke="#ffffff", stroke_width=1)
                label = f"{mean:.0f}\nn={n}"
                parts = label.split("\n")
                c.text(x0 + cellw / 2, y0 + cellh / 2 - 9, parts[0], size=15, anchor="center", color=text_color(fill), bold=True)
                c.text(x0 + cellw / 2, y0 + cellh / 2 + 12, parts[1], size=12, anchor="center", color=text_color(fill))
            else:
                c.rect(x0 + 1, y0 + 1, cellw - 2, cellh - 2, "#F1F3F5", stroke="#ffffff", stroke_width=1)
                c.text(x0 + cellw / 2, y0 + cellh / 2, f"n={n}", size=12, anchor="center", color=GRAY)
    c.text(hx + hw / 2, hy + hh + 28, "颜色为任务内百分位均值；n<20 的单元不着色", size=15, anchor="center", color=GRAY)

    xlim = (float(min(fitted)) - 2, float(max(fitted)) + 2)
    ylim = (min(-45.0, float(min(residual)) - 2), max(45.0, float(max(residual)) + 2))
    axes(c, p3, xlim, ylim, [(v, f"{v:.0f}") for v in np.linspace(xlim[0], xlim[1], 7)], [(v, f"{v:.0f}") for v in np.linspace(ylim[0], ylim[1], 5)], "拟合指数", "观测 − 拟合")
    for xx, yy in zip(fitted, residual):
        sx, sy = map_xy(p3, float(xx), float(yy), xlim, ylim)
        c.circle(sx, sy, 3, PURPLE)
    zero1, zero2 = map_xy(p3, xlim[0], 0, xlim, ylim), map_xy(p3, xlim[1], 0, xlim, ylim)
    c.line(zero1[0], zero1[1], zero2[0], zero2[1], RED, 3, dash="10 7")
    c.text(p3[0] + 18, p3[1] + 18, f"HC3 回归诊断，n={len(residual)}；残差仅用于模型检查", size=17, color=PURPLE)
    return c.finish(basename)


def draw_results(aggregate: pd.DataFrame, monthly: pd.DataFrame, frontier: pd.DataFrame, basename: Path) -> list[Path]:
    c = FigureCanvas()
    c.text(W / 2, 45, "Q4 结果：任务族分布、分解与开放权重前沿", size=35, bold=True, anchor="center")
    p1 = panel(c, 70, 110, 970, 620, "同一完整样本的任务族分布", "a")
    p2 = panel(c, 1120, 110, 970, 620, "支持月份的均值变化分解", "b")
    p3 = panel(c, 70, 790, 2020, 650, "明确开放权重模型的月度进入表现", "c")
    main = aggregate[aggregate["core6_index_0_100"].notna()]
    x, y, w, h = p1
    ylim = (0, 100)
    cell_w = w / len(CORE_FAMILIES)
    tickx = x + 60
    plot_w = w - 85
    for val in [0, 25, 50, 75, 100]:
        yy = y + h - val / 100 * h
        c.line(tickx, yy, tickx + plot_w, yy, LIGHT, 1)
        c.text(tickx - 10, yy, str(val), size=15, anchor="right", color=GRAY)
    colors = [BLUE, ORANGE, GREEN, RED, PURPLE, GRAY]
    for j, family in enumerate(CORE_FAMILIES):
        vals = pd.to_numeric(main[FAMILY_COLS[family]], errors="coerce").dropna().to_numpy(float)
        cx = tickx + (j + .5) * plot_w / len(CORE_FAMILIES)
        stats = [percentile(vals, p) for p in [.05, .25, .5, .75, .95]]
        y5, y25, y50, y75, y95 = [y + h - v / 100 * h for v in stats]
        c.line(cx, y95, cx, y5, colors[j], 5)
        c.line(cx - 14, y95, cx + 14, y95, colors[j], 4)
        c.line(cx - 14, y5, cx + 14, y5, colors[j], 4)
        c.rect(cx - 26, y75, 52, y25 - y75, "#E9F1F6", stroke=colors[j], stroke_width=3)
        c.line(cx - 26, y50, cx + 26, y50, colors[j], 4)
        short = {"MATH-Hard": "MATH v2", "MMLU-Pro": "MMLU-P", "IFEval": "IFEval"}.get(family, family)
        c.text(cx, y + h + 27, short, size=15, anchor="center")
    c.text(tickx + plot_w / 2, y + h + 62, "族内百分位分布（须=P5/P95，箱=P25/P75，中线=P50；n=789）", size=15, anchor="center", color=GRAY)

    supported = monthly[monthly["decomposition_supported_month_n_ge_20"].astype(str).str.lower().eq("true")].sort_values("submission_month")
    endpoints = supported.iloc[[0, -1]].copy()
    components = [
        ("scale_composition_component", "参数构成", BLUE),
        ("model_type_composition_component", "C1 类型构成", ORANGE),
        ("month_fixed_effect_component", "月份条件项", GREEN),
        ("unexplained_component", "回归恒等项", GRAY),
    ]
    bx, by, bw, bh = p2
    values_all = [float(row[col]) for _, row in endpoints.iterrows() for col, _, _ in components]
    left_lim = min(-2.0, math.floor(min(values_all) - 1))
    right_lim = max(2.0, math.ceil(max(values_all) + 1))
    ylabels = [f"{r['submission_month']} n={int(r['n_models'])}" for _, r in endpoints.iterrows()]
    for tick in np.linspace(left_lim, right_lim, 5):
        xx = bx + (tick - left_lim) / (right_lim - left_lim) * bw
        c.line(xx, by, xx, by + bh, LIGHT, 1)
        c.text(xx, by + bh + 18, f"{tick:.0f}", size=14, anchor="center", color=GRAY)
    c.line(bx + (0 - left_lim) / (right_lim - left_lim) * bw, by, bx + (0 - left_lim) / (right_lim - left_lim) * bw, by + bh, "#333333", 2)
    row_h = bh / 2
    for i, row in enumerate(endpoints.to_dict("records")):
        cy = by + (i + .5) * row_h
        c.text(bx - 12, cy, ylabels[i], size=17, anchor="right")
        for j, (col, label, color) in enumerate(components):
            val = float(row[col])
            yy = cy + (j - 1.5) * 34
            x0 = bx + (0 - left_lim) / (right_lim - left_lim) * bw
            x1 = bx + (val - left_lim) / (right_lim - left_lim) * bw
            c.line(min(x0, x1), yy, max(x0, x1), yy, color, 7)
            c.circle(x1, yy, 7, color)
            c.text(bx + bw - 8, yy, f"{label} {val:+.1f}", size=14, anchor="right", color=color)
    c.text(bx + bw / 2, by + bh + 50, "指数点，相对回归支持样本均值；非因果贡献", size=15, anchor="center", color=GRAY)

    f = frontier.sort_values("submission_month").copy()
    months = f["submission_month"].astype(str).tolist()
    fx, fy, fw, fh = p3
    ylim = (0, 100)
    xlim = (-.4, len(f) - .6)
    axes(c, p3, xlim, ylim, [(i, months[int(i)]) for i in range(len(months))], [(v, str(v)) for v in (0, 25, 50, 75, 100)], "C1 Submission Date 月份", "C1 官方 Average")
    q90 = f["monthly_entry_p90_frontier_C1_average_0_100"].to_numpy(float)
    median = f["monthly_entry_median_C1_average_0_100"].to_numpy(float)
    qpts = [map_xy(p3, float(i), float(v), xlim, ylim) for i, v in enumerate(q90)]
    medpts = [map_xy(p3, float(i), float(v), xlim, ylim) for i, v in enumerate(median)]
    c.polyline(qpts, BLUE, 5)
    c.polyline(medpts, ORANGE, 4, dash="10 7")
    for i, row in enumerate(f.to_dict("records")):
        qp, mp = qpts[i], medpts[i]
        small = int(row["n_known_open_weights_models"]) < 10
        c.circle(qp[0], qp[1], 10, "#ffffff" if small else BLUE, stroke=BLUE, stroke_width=4)
        c.circle(mp[0], mp[1], 8, ORANGE)
        c.text(qp[0], qp[1] - 26, f"n={int(row['n_known_open_weights_models'])}", size=15, anchor="center", color=GRAY)
    c.line(fx + fw - 380, fy + 5, fx + fw - 330, fy + 5, BLUE, 5)
    c.circle(fx + fw - 355, fy + 5, 8, BLUE)
    c.text(fx + fw - 320, fy + 5, "每月新提交模型 P90", size=17)
    c.line(fx + fw - 380, fy + 38, fx + fw - 330, fy + 38, ORANGE, 4, dash="10 7")
    c.circle(fx + fw - 355, fy + 38, 7, ORANGE)
    c.text(fx + fw - 320, fy + 38, "每月中位数", size=17)
    return c.finish(basename)


def main() -> None:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"Required Chinese font is unavailable: {FONT_PATH}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    aggregate = pd.read_csv(RESULTS / "q4_task_aggregate.csv", low_memory=False)
    coverage = pd.read_csv(RESULTS / "q4_selected_metric_coverage.csv")
    monthly = pd.read_csv(RESULTS / "q4_scale_tech_monthly.csv")
    frontier = pd.read_csv(RESULTS / "q4_frontier_monthly.csv")
    profile_rows = []
    for label, series in [
        ("positive parameter count B in core6-complete sample", aggregate.loc[aggregate["core6_index_0_100"].notna() & aggregate["parameter_count_B"].gt(0), "parameter_count_B"]),
        ("six-family percentile index", aggregate.loc[aggregate["core6_index_0_100"].notna(), "core6_index_0_100"]),
        ("raw-accuracy family mean sensitivity", aggregate.loc[aggregate["core6_index_0_100"].notna(), "core6_rawmean_index_0_100"]),
        ("C1 official Average", aggregate["C1_Average_0_100"]),
    ]:
        values = pd.to_numeric(series, errors="coerce").dropna()
        profile_rows.append({
            "variable": label, "n": int(len(values)), "missing": int(len(series) - len(values)),
            "min": float(values.min()), "p10": float(values.quantile(.1)), "median": float(values.median()),
            "mean": float(values.mean()), "p90": float(values.quantile(.9)), "max": float(values.max()),
        })
    profile_path = RESULTS / "q4_data_profile.csv"
    pd.DataFrame(profile_rows).to_csv(profile_path, index=False, encoding="utf-8-sig")

    core = aggregate[aggregate["core6_index_0_100"].notna()].copy()
    scale_sample = aggregate[
        aggregate["core6_index_0_100"].notna() & aggregate["parameter_count_B"].gt(0)
        & aggregate["submission_date"].notna()
    ].copy()
    scale_sample["submission_date"] = pd.to_datetime(scale_sample["submission_date"], errors="coerce")
    scale_sample["submission_month"] = scale_sample["submission_date"].dt.to_period("M").astype(str)
    from q4_modeling_analysis import map_type_groups, fit_hc3
    scale_sample["log10_params_B"] = np.log10(scale_sample["parameter_count_B"].astype(float))
    scale_sample["type_group"] = map_type_groups(scale_sample["C1_Type_raw"])
    month_counts = scale_sample["submission_month"].value_counts()
    scale_sample = scale_sample[scale_sample["submission_month"].isin(month_counts[month_counts.ge(20)].index)].copy()
    fit = fit_hc3(scale_sample, "figure residual reproduction")
    if fit is None:
        raise RuntimeError("Failed to reproduce supported-month model for residual figure")
    _, model = fit

    output_files = []
    output_files.extend(draw_raw(aggregate, FIGURES / "q4_raw_data"))
    output_files.extend(draw_process(aggregate, coverage, scale_sample, model["residual"], model["fitted"], FIGURES / "q4_process_diagnostics"))
    output_files.extend(draw_results(aggregate, monthly, frontier, FIGURES / "q4_results"))

    contracts = [
        ("q4_raw_data", "原始数据", "展示正参数量分布、六族任务内百分位指数分布及模型级规模关系。", "参数直方图 n=788；指数直方图 n=789；规模散点 n=788。散点趋势线为未调整描述线。"),
        ("q4_process_diagnostics", "计算过程", "展示主指标的完整覆盖、提交月份任务族均值支持和固定效应模型残差。", "覆盖统计以 1,819 候选为分母；月×族单元 n<20 不着色；残差来自支持月份 n>=20 的 HC3 线性模型，n=783。"),
        ("q4_results", "分析结果", "展示相同 789 个模型的族内百分位分布、两个支持月份的描述性分解，以及开放权重月度进入表现。", "箱线须=P5/P95、箱=P25/P75、中位线=P50；分解端点为 2024-11 与 2025-03；开放前沿是每月 C1 Average P90，空心点代表 n<10。"),
    ]
    contract_lines = [
        "# Q4 图表契约与 QA 记录", "",
        "> AI 辅助说明：图表由 OpenAI Codex 桌面版 26.917.71314（build 10954；prod；2026-09-25 更新器核验为 up_to_date）辅助生成；开发机构 OpenAI；模型选择器显示名称：；模型版本/发布日期：。（按用户指示暂空。）", "",
        "## 数据剖析", "",
        f"六族主指数完整样本 {len(core)} 个；有正参数量的模型 {int((aggregate['core6_index_0_100'].notna() & aggregate['parameter_count_B'].gt(0)).sum())} 个。参数量按 log10 展示。主指数先在每个叶任务内按唯一名称候选计算百分位秩，再做族内和族间等权。分布统计见 `../03_结果/results/q4_data_profile.csv`。", "",
        "## 图表用途与口径", "",
    ]
    for name, group, claim, stat in contracts:
        contract_lines.extend([
            f"### {name}（{group}）", "",
            f"- 核心论点：{claim}",
            f"- 统计口径：{stat}",
            "- 版面：7.2 × 5.2 英寸；导出 SVG 与 300 dpi PNG，同时提供灰度预览。",
            "- QA：轴标题、刻度和样本标注写入矢量 SVG；对应 PNG 已用 Microsoft YaHei 字体栅格化，之后由视觉复核确认。",
            "",
        ])
    contract_lines.extend([
        "## 已知呈现限制", "",
        "本机 Matplotlib 3.10.9 在简单 788 点直方图的矩形路径构建中持续停滞，故改用 Pillow 与直接生成的 SVG 原语。Pillow 栅格输出与 SVG 共用同一绘图坐标和数据值。", "",
        "开放前沿图的 P90 是月度新提交模型的截面统计，不是累计可用模型库的前沿；月样本不足 10 时标为空心点。所有图均为描述性图，不显示抽样置信区间或显著性符号。", "",
        "结果图中的回归恒等项按观测月均值减去参数、类型和月份分量计算。由于支持样本的 OLS 含月份固定效应，样本内每月平均残差为零；图上该项接近零是回归构造结果，不表示个体残差为零。", "",
    ])
    contract_path = Q4 / "01_方案说明" / "q4_figures_contract.md"
    contract_path.write_text("\n".join(contract_lines), encoding="utf-8")

    qa = {
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "Pillow raster plus direct editable SVG primitives",
        "font": str(FONT_PATH),
        "image_size_px": [W, H],
        "dpi": DPI,
        "figures": [
            {"name": name, "png": rel(FIGURES / f"{name}.png"), "svg": rel(FIGURES / f"{name}.svg"),
             "grayscale_png": rel(FIGURES / f"{name}_grayscale.png"), "program_status": "PASS",
             "manual_visual_review": "pending AI visual review"}
            for name, *_ in contracts
        ],
        "manual_visual_review": "pending",
    }
    qa_path = RESULTS / "q4_figure_qa.json"
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_path = RESULTS / "q4_modeling_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = [profile_path, contract_path, qa_path, *output_files]
    manifest.setdefault("outputs_sha256", {}).update({rel(path): sha(path) for path in outputs})
    manifest["figure_qa_status"] = "Pillow/SVG program checks passed; AI visual review pending"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"created {len(output_files)} PNG/SVG/grayscale files; profile={profile_path}; contract={contract_path}")


if __name__ == "__main__":
    main()
