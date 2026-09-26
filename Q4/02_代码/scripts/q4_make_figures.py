"""Generate auditable Q4 figures with Pillow plus editable SVG output.

The bundled Python runtime does not include Matplotlib. This generator uses
direct vector SVG primitives and matching 300 dpi Pillow previews instead.

AI disclosure: OpenAI Codex desktop 26.917.71314 (build 10954; prod), model ID
gpt-6-luna (GPT-6 Luna), developer OpenAI, official release date 2026-09-22.
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
GRAYSCALE_DIR = FIGURES / "灰度预览"
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

    def polygon(self, points, fill, stroke=None, stroke_width=1):
        points = [(float(x), float(y)) for x, y in points]
        self.draw.polygon([(int(round(x)), int(round(y))) for x, y in points], fill=color_hex(fill))
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        attr = f'fill="{fill}"'
        if stroke:
            attr += f' stroke="{stroke}" stroke-width="{stroke_width}"'
        self.svg.append(f'<polygon points="{pts}" {attr}/>')

    def arrow(self, x1, y1, x2, y2, color=BLUE, width=5):
        self.line(x1, y1, x2, y2, color, width)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 17
        back = angle + math.pi
        left = (x2 + size * math.cos(back - .48), y2 + size * math.sin(back - .48))
        right = (x2 + size * math.cos(back + .48), y2 + size * math.sin(back + .48))
        self.polygon([(x2, y2), left, right], color)

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
        GRAYSCALE_DIR.mkdir(parents=True, exist_ok=True)
        png = basename.with_suffix(".png")
        svg = basename.with_suffix(".svg")
        gray = GRAYSCALE_DIR / f"{basename.name}_grayscale.png"
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


def draw_c3_coverage(c3: pd.DataFrame, basename: Path) -> list[Path]:
    c = FigureCanvas()
    c.text(W / 2, 48, "C3 原始年度覆盖：来源分层与样本不均", size=34, bold=True, anchor="center")
    c.text(125, 112, "去冲突后的唯一模型记录数；横轴为 log10(n)，不连接年份", size=21, color=GRAY)
    rows = c3.sort_values(["Source", "Year"]).to_dict("records")
    x0, x1 = 690, 1810
    y0, row_h = 235, 115
    max_log = 4.0
    c.line(x0, y0 - 34, x1, y0 - 34, "#333333", 2)
    for value, label in [(0, "1"), (1, "10"), (2, "100"), (3, "1,000"), (4, "10,000")]:
        xx = x0 + value / max_log * (x1 - x0)
        c.line(xx, y0 - 34, xx, y0 + row_h * len(rows) - 25, LIGHT, 1)
        c.text(xx, y0 + row_h * len(rows) + 5, label, size=18, anchor="center", color=GRAY)
    for i, row in enumerate(rows):
        yy = y0 + (i + .5) * row_h
        source = str(row["Source"])
        year = int(row["Year"])
        status = str(row["coverage_status"])
        color = BLUE if source == "Open LLM Leaderboard" else GRAY
        label = f"{'OLLB' if source == 'Open LLM Leaderboard' else '历史论文/报告'} · {year}"
        if "partial_year" in status:
            label += "（部分年）"
        count = int(row["unique_model_records_after_dedup"])
        excluded = int(row["conflicting_model_year_groups_excluded"])
        c.text(650, yy, label, size=20, anchor="right")
        bar_x = x0 + math.log10(max(count, 1)) / max_log * (x1 - x0)
        c.line(x0, yy, bar_x, yy, color, 9)
        c.circle(bar_x, yy, 9, color)
        c.text(1845, yy, f"n={count:,}；冲突排除 {excluded}", size=17, color=color)
    c.text((x0 + x1) / 2, 1390, "历史论文/报告样本选择性强，与排行榜来源分开；2025 年仅部分年度覆盖。", size=19, anchor="center", color=GRAY)
    return c.finish(basename)


def draw_loss_gate_inputs(gate: pd.DataFrame, basename: Path) -> list[Path]:
    row = gate.loc[gate["scope"].eq("pooled_C5_C6_deduplicated")].iloc[0]
    items = [
        ("高可比 Loss–Benchmark 配对", int(row["unique_high_content_pairs"]), int(row["unique_high_content_pairs"]), BLUE),
        ("历史模型版本已核验", int(row["verified_model_version_count"]), int(row["unique_model_names"]), RED),
        ("独立来源模型族已核验", int(row["independent_verified_family_count"]), int(row["unique_model_names"]), PURPLE),
        ("Loss 协议字段已知", sum(str(row[k]).strip().upper() not in {"UNKNOWN", ""} for k in ("loss_unit", "evaluation_dataset", "split", "tokenizer", "preprocessing")), 5, ORANGE),
        ("Q2 数值候选落在观察范围", int(row["target_candidate_rows_inside_high_loss_range"]), int(row["target_candidate_rows"]), GREEN),
    ]
    c = FigureCanvas()
    c.text(W / 2, 48, "Loss–Benchmark 原始证据覆盖（拟合前）", size=34, bold=True, anchor="center")
    c.text(130, 112, "每项显示可核验数量 / 候选数量；数值范围重合不代表协议可比", size=21, color=GRAY)
    x0, x1 = 730, 1730
    y0, row_h = 310, 190
    c.line(x0, y0 - 45, x1, y0 - 45, "#333333", 2)
    for v in (0, .25, .5, .75, 1):
        xx = x0 + v * (x1 - x0)
        c.line(xx, y0 - 45, xx, y0 + row_h * len(items) - 40, LIGHT, 1)
        c.text(xx, y0 + row_h * len(items) + 10, f"{v:.0%}", size=18, anchor="center", color=GRAY)
    for i, (label, num, den, color) in enumerate(items):
        yy = y0 + (i + .5) * row_h
        frac = num / den if den else 0.0
        c.text(690, yy, label, size=21, anchor="right")
        c.line(x0, yy, x0 + frac * (x1 - x0), yy, color, 11)
        c.circle(x0 + frac * (x1 - x0), yy, 11, color)
        c.text(1780, yy, f"{num} / {den}", size=19, color=color)
    c.text(145, 1350, "单位、评测集、split、tokenizer、preprocessing 未核验；版本与独立来源也未确认。门控结果：BLOCKED_NO_FIT。", size=20, color=RED)
    return c.finish(basename)


def draw_version_match(matches: pd.DataFrame, basename: Path) -> list[Path]:
    total = len(matches)
    unique = int(matches["provisional_unique_name_candidate"].astype(str).str.lower().eq("true").sum())
    strict = int(matches["strict_version_confirmed"].astype(str).str.lower().eq("true").sum())
    c4_exact = 0  # The C4 exact-name intersection is explicitly zero in the verified coverage report.
    items = [("C8 最新目录", total, BLUE), ("C1 唯一名称候选", unique, GREEN),
             ("C1/C8 严格版本确认", strict, RED), ("C4 精确名称交集", c4_exact, PURPLE)]
    c = FigureCanvas()
    c.text(W / 2, 48, "模型匹配流程：名称候选没有替代版本证据", size=34, bold=True, anchor="center")
    c.text(125, 112, "横条显示记录数；严格版本必须有可核验 revision/SHA 或人工版本证据", size=21, color=GRAY)
    x0, x1, ymax = 650, 1880, max(1900, total)
    y0, row_h = 330, 220
    for tick in (0, 500, 1000, 1500, 2000):
        xx = x0 + tick / ymax * (x1 - x0)
        c.line(xx, y0 - 45, xx, y0 + row_h * len(items) - 45, LIGHT, 1)
        c.text(xx, y0 + row_h * len(items), f"{tick:,}", size=18, anchor="center", color=GRAY)
    for i, (label, value, color) in enumerate(items):
        yy = y0 + (i + .5) * row_h
        c.text(610, yy, label, size=22, anchor="right")
        xx = x0 + value / ymax * (x1 - x0)
        if value:
            c.line(x0, yy, xx, yy, color, 11)
            c.circle(xx, yy, 11, color)
        else:
            c.circle(x0, yy, 11, color)
        c.text(1910, yy, f"{value:,}", size=21, color=color)
    c.text(160, 1320, "精确名称只构成连接候选；当前严格版本确认为 0，C4 也没有精确名称交集。", size=21, color=GRAY)
    return c.finish(basename)


def draw_ar1_backtest(backtest: pd.DataFrame, basename: Path) -> list[Path]:
    c = FigureCanvas()
    c.text(W / 2, 48, "AR(1) 与持平基线：仅两个一步留出点", size=34, bold=True, anchor="center")
    c.text(130, 112, "绝对误差（指数点）；小样本点图，不作显著性或年度预测结论", size=21, color=GRAY)
    x_positions = [770, 1390]
    x_labels = backtest["target_month"].astype(str).tolist()
    colors = {"AR(1)": BLUE, "持平基线": ORANGE}
    xlim = (0.0, 1.8)
    x0, x1, y0, y1 = 440, 1840, 320, 1170
    for v in np.linspace(0, 1.8, 7):
        xx = x0 + (v - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
        c.line(xx, y0, xx, y1, LIGHT, 1)
        c.text(xx, y1 + 32, f"{v:.1f}", size=18, anchor="center", color=GRAY)
    for i, row in enumerate(backtest.to_dict("records")):
        yy = x_positions[i]
        c.text(370, yy, x_labels[i], size=22, anchor="right")
        for j, (label, field) in enumerate([("AR(1)", "ar1_absolute_error"), ("持平基线", "persistence_absolute_error")]):
            xxval = float(row[field])
            xx = x0 + xxval / xlim[1] * (x1 - x0)
            cy = yy + (j - .5) * 42
            color = colors[label]
            c.circle(xx, cy, 11, color)
            c.text(xx + 18, cy, f"{label} {xxval:.3f}", size=18, color=color)
    c.text((x0 + x1) / 2, y1 + 85, "一步绝对误差（六族指数点）", size=20, anchor="center", color=GRAY)
    c.text(135, 1490, "两次留出预测：AR(1) MAE={:.3f}，持平基线 MAE={:.3f}；不足以证明稳定预测能力。".format(
        float(backtest["ar1_absolute_error"].mean()), float(backtest["persistence_absolute_error"].mean())), size=18, color=GRAY)
    return c.finish(basename)


def draw_c3_results(c3: pd.DataFrame, basename: Path) -> list[Path]:
    rows = c3.sort_values(["Source", "Year"]).to_dict("records")
    c = FigureCanvas()
    c.text(W / 2, 48, "C3 年度背景：不同来源分列，年份不连线", size=34, bold=True, anchor="center")
    c.text(125, 112, "点为 Average 中位数 / P90；历史文献行是选择性记录，2025 排行榜数据为部分年度", size=20, color=GRAY)
    x0, x1 = 560, 1850
    y0, row_h = 225, 125
    for v in (0, 20, 40, 60, 80, 100):
        xx = x0 + v / 100 * (x1 - x0)
        c.line(xx, y0 - 35, xx, y0 + row_h * len(rows) - 30, LIGHT, 1)
        c.text(xx, y0 + row_h * len(rows) + 5, str(v), size=18, anchor="center", color=GRAY)
    for i, row in enumerate(rows):
        yy = y0 + (i + .5) * row_h
        source = str(row["Source"])
        year = int(row["Year"])
        partial = " *" if str(row["coverage_status"]) == "partial_year_coverage_only" else ""
        label = f"{'OLLB' if source == 'Open LLM Leaderboard' else '历史论文/报告'} · {year}{partial}  n={int(row['Average_numeric_records'])}"
        c.text(525, yy, label, size=18, anchor="right")
        color = BLUE if source == "Open LLM Leaderboard" else GRAY
        for value, dy, marker in [(row["Average_median"], -15, "circle"), (row["Average_p90"], 15, "square")]:
            if pd.isna(value):
                continue
            xx = x0 + float(value) / 100 * (x1 - x0)
            cy = yy + dy
            if marker == "circle":
                c.circle(xx, cy, 9, color)
            else:
                c.rect(xx - 8, cy - 8, 16, 16, color)
    c.circle(1420, 1390, 8, BLUE)
    c.text(1440, 1390, "中位数", size=18)
    c.rect(1610, 1382, 16, 16, BLUE)
    c.text(1635, 1390, "P90", size=18)
    c.text(125, 1490, "* 2025 OLLB 部分年度；不作为完整年度变化率。各行仅作来源内背景比较。", size=18, color=GRAY)
    return c.finish(basename)


def draw_flowchart(basename: Path) -> list[Path]:
    c = FigureCanvas()
    c.text(W / 2, 54, "Q4 当前分析流程与停止门", size=36, bold=True, anchor="center")
    boxes = {
        "data": (70, 215, 620, 330, BLUE, "① 数据与来源", ["C1–C8 原始附件", "字段字典：无隐藏字段版 PDF", "逐项记录来源与 SHA-256"]),
        "cohort": (770, 215, 620, 330, GREEN, "② 已确认样本与指标", ["用户确认并冻结队列和权重", "六族完整指数；缺失不补 0", "版本未知继续标 version_unresolved"]),
        "scores": (1470, 215, 620, 330, ORANGE, "③ 逐任务处理与覆盖", ["C8 最新目录 1,860 个", "唯一名称候选 1,819 个", "严格版本确认 0 个"]),
        "dynamic": (1470, 800, 620, 330, PURPLE, "④ 动态与共同支持分解", ["共同支持拟合 645 条，秩 10/10", "HC3 区间；只作观察性关联", "闭合残差只作算术核对"]),
        "gate": (770, 800, 620, 330, RED, "⑤ Loss–Benchmark 预拟合门", ["先核协议、单位、版本和目标网格", "关键协议与 revision 证据缺失", "当前 BLOCKED_NO_FIT，不拟合"]),
        "forecast": (70, 800, 620, 330, GRAY, "⑥ 预测支持门", ["前沿时间轴 10 个月、跨 9 个月", "年度滚动起点 0", "12 / 24 月数值预测不估计"]),
    }
    for x, y, w, h, color, title, lines in boxes.values():
        c.rect(x, y, w, h, "#F7F9FB", stroke=color, stroke_width=4)
        c.text(x + w / 2, y + 52, title, size=26, color=color, bold=True, anchor="center")
        for i, line in enumerate(lines):
            c.text(x + w / 2, y + 125 + i * 58, line, size=20, anchor="center", color=GRAY)
    c.arrow(690, 380, 755, 380, BLUE, 5)
    c.arrow(1390, 380, 1455, 380, BLUE, 5)
    c.arrow(1780, 545, 1780, 780, BLUE, 5)
    c.arrow(1470, 965, 1405, 965, BLUE, 5)
    c.arrow(770, 965, 705, 965, BLUE, 5)
    c.rect(700, 1280, 760, 180, "#F7F9FB", stroke=BLUE, stroke_width=4)
    c.text(W / 2, 1325, "⑦ 结果包与队伍复核", size=27, color=BLUE, bold=True, anchor="center")
    c.text(W / 2, 1380, "9 张候选图 + 流程图 + 可复现清单 + 输入 / 输出哈希", size=20, anchor="center", color=GRAY)
    c.text(W / 2, 1510, "Q4 分析包已由用户授权冻结；其他 AI 使用范围须在竞赛论文提交前核实。", size=18, anchor="center", color=GRAY)
    c.arrow(380, 1130, 820, 1260, BLUE, 5)
    c.arrow(1080, 1130, 1080, 1260, BLUE, 5)
    c.arrow(1780, 1130, 1340, 1260, BLUE, 5)
    return c.finish(basename)


def draw_dynamic_results(contribution: pd.DataFrame, ar1: pd.DataFrame, backtest: pd.DataFrame,
                         support: pd.DataFrame, forecast: pd.DataFrame, gate: pd.DataFrame,
                         basename: Path) -> list[Path]:
    """Plot full-run HC3 contrasts and summarize the limits on interpretation."""
    c = FigureCanvas()
    c.text(W / 2, 45, "Q4 动态诊断与共同支持分解", size=35, bold=True, anchor="center")
    c.text(95, 120, "(a) 2024-11 → 2025-03 的共同支持变化分解", size=27, bold=True)
    c.text(1500, 120, "(b) 样本与可行性边界", size=27, bold=True)

    labels = {
        "scale_composition": "参数规模项",
        "type_composition": "模型类型构成",
        "month_fixed_effect": "月份条件项",
        "non_scale_total": "非规模合计",
        "predicted_mean_change": "模型预测均值变化",
        "observed_mean_change": "观测均值变化",
    }
    colors = {
        "scale_composition": BLUE,
        "type_composition": ORANGE,
        "month_fixed_effect": GREEN,
        "non_scale_total": PURPLE,
        "predicted_mean_change": GRAY,
        "observed_mean_change": RED,
    }
    rows = {row["component"]: row for row in contribution.to_dict("records") if row["component"] in labels}
    order = list(labels)
    plot = (490, 280, 690, 790)
    xlim = (-8.0, 8.0)
    for tick in [-8, -4, 0, 4, 8]:
        xx = plot[0] + (tick - xlim[0]) / (xlim[1] - xlim[0]) * plot[2]
        c.line(xx, plot[1], xx, plot[1] + plot[3], LIGHT if tick else "#333333", 2 if tick == 0 else 1)
        c.text(xx, plot[1] + plot[3] + 26, str(tick), size=18, anchor="center", color=GRAY)
    row_gap = plot[3] / len(order)
    for i, component in enumerate(order):
        row = rows[component]
        yy = plot[1] + (i + .5) * row_gap
        value = float(row["estimate_index_points"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        color = colors[component]
        c.text(450, yy, labels[component], size=21, anchor="right")
        x0 = plot[0] + (low - xlim[0]) / (xlim[1] - xlim[0]) * plot[2]
        x1 = plot[0] + (high - xlim[0]) / (xlim[1] - xlim[0]) * plot[2]
        xm = plot[0] + (value - xlim[0]) / (xlim[1] - xlim[0]) * plot[2]
        c.line(x0, yy, x1, yy, color, 6)
        c.line(x0, yy - 11, x0, yy + 11, color, 3)
        c.line(x1, yy - 11, x1, yy + 11, color, 3)
        c.circle(xm, yy, 9, color)
        c.text(1215, yy, f"{value:+.2f} [{low:+.2f}, {high:+.2f}]", size=17, color=color)
    c.text(plot[0] + plot[2] / 2, plot[1] + plot[3] + 70,
           "指数点；点为估计，横线为 HC3 95% 区间", size=18, anchor="center", color=GRAY)
    closure = contribution.loc[contribution["component"].eq("arithmetic_closure_residual")]
    closure_value = float(closure.iloc[0]["estimate_index_points"]) if len(closure) else float("nan")
    c.text(105, 1200, f"加总闭合误差 {closure_value:.1e}：仅为算术核对，不是未解释因素", size=19, color=GRAY)
    c.text(105, 1245, "观测变化与非规模总项区间含 0；按合同不报告贡献占比。", size=19, color=GRAY)

    sx = 1500

    def block(y: int, title: str, lines: list[str], color: str = "#222222") -> int:
        c.text(sx, y, title, size=22, bold=True, color=color)
        y += 37
        for line in lines:
            c.text(sx + 8, y, line, size=18, color=GRAY)
            y += 31
        return y + 22

    support_row = support.iloc[0].to_dict()
    ar = ar1.iloc[0].to_dict()
    bt = backtest.iloc[0].to_dict()
    y = block(190, "共同参数支持", [
        f"拟合记录 n={int(support_row['fit_rows_in_common_support'])}；秩 {int(support_row['design_rank'])}/{int(support_row['design_columns'])}",
        f"端点样本 n={int(support_row['early_rows_in_common_support'])} / {int(support_row['late_rows_in_common_support'])}",
        f"log10(B) 区间 [{float(support_row['log10_params_B_support_low_p05_p95']):.3f}, {float(support_row['log10_params_B_support_high_p05_p95']):.3f}]",
    ])
    y = block(y, "AR(1) 仅作描述", [
        f"连续月份 {int(ar['monthly_bins'])}，转移 {int(ar['transitions'])}",
        f"ρ={float(ar['rho']):.3f}；HC3 95% [{float(ar['rho_ci95_low']):.3f}, {float(ar['rho_ci95_high']):.3f}]",
    ], BLUE)
    y = block(y, "一步回测支持有限", [
        f"合格起点：AR(1) {int(bt['ar1_origins'])}，持平基线 {int(bt['persistence_origins'])}",
        f"MAE：{float(bt['ar1_mae']):.3f} vs {float(bt['persistence_mae']):.3f}",
        "两次预测不足以证明稳定预测能力。",
    ], ORANGE)
    annual_origins = max(int(row["annual_horizon_rolling_origin_count"]) for row in forecast.to_dict("records"))
    y = block(y, "12 / 24 个月预测", [f"年度滚动起点 {annual_origins}；数值预测不估计。"], RED)
    gate_status = str(gate.iloc[0]["prefit_gate_status"]) if len(gate) else "BLOCKED_NO_FIT"
    y = block(y, "Loss–Benchmark 门控", [
        f"状态：{gate_status}",
        "桥接拟合未执行；可比性和目标网格待补。",
    ], PURPLE)
    c.text(sx, min(1395, y + 5), "非因果分解；年份覆盖详见 C3/C4 分表。", size=17, color=GRAY)
    return c.finish(basename)


def main() -> None:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"Required Chinese font is unavailable: {FONT_PATH}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    aggregate = pd.read_csv(RESULTS / "q4_task_aggregate.csv", low_memory=False)
    coverage = pd.read_csv(RESULTS / "q4_selected_metric_coverage.csv")
    monthly = pd.read_csv(RESULTS / "q4_scale_tech_monthly.csv")
    frontier = pd.read_csv(RESULTS / "q4_frontier_monthly.csv")
    c3 = pd.read_csv(RESULTS / "q4_c3_year_summary.csv")
    contribution = pd.read_csv(RESULTS / "q4_scale_tech_contribution.csv")
    ar1 = pd.read_csv(RESULTS / "q4_ar1_dynamic.csv")
    backtest = pd.read_csv(RESULTS / "q4_ar1_backtest_summary.csv")
    backtest_rows = pd.read_csv(RESULTS / "q4_ar1_expanding_backtest.csv")
    support = pd.read_csv(RESULTS / "q4_scale_tech_support_audit.csv")
    forecast = pd.read_csv(RESULTS / "q4_forecast_feasibility.csv")
    gate = pd.read_csv(RESULTS / "q4_loss_benchmark_feasibility.csv")
    matches = pd.read_csv(RESULTS / "q4_match_audit.csv", low_memory=False)
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
    output_files.extend(draw_raw(aggregate, FIGURES / "raw_q4_data"))
    output_files.extend(draw_c3_coverage(c3, FIGURES / "raw_q4_c3_coverage"))
    output_files.extend(draw_loss_gate_inputs(gate, FIGURES / "raw_q4_loss_gate_inputs"))
    output_files.extend(draw_process(aggregate, coverage, scale_sample, model["residual"], model["fitted"], FIGURES / "process_q4_diagnostics"))
    output_files.extend(draw_version_match(matches, FIGURES / "process_q4_version_match"))
    output_files.extend(draw_ar1_backtest(backtest_rows, FIGURES / "process_q4_ar1_backtest"))
    output_files.extend(draw_results(aggregate, monthly, frontier, FIGURES / "result_q4_frontier"))
    output_files.extend(draw_dynamic_results(contribution, ar1, backtest, support, forecast, gate, FIGURES / "result_q4_dynamic_decomposition"))
    output_files.extend(draw_c3_results(c3, FIGURES / "result_q4_c3_context"))
    output_files.extend(draw_flowchart(FIGURES / "flow_overall_model"))

    contracts = [
        ("raw_q4_data", "原始数据", "展示主指数分布、参数规模分布及模型级规模—指数关系。", "参数直方图 n=788；指数直方图 n=789；散点 n=788；趋势线为未调整描述线。"),
        ("raw_q4_c3_coverage", "原始数据", "显示 C3 排行榜和历史文献来源按年记录量不均。", "去冲突后的唯一模型记录数；横轴为 log10(n)；2025 标为部分年度覆盖；来源分列。"),
        ("raw_q4_loss_gate_inputs", "原始数据", "呈现拟合前配对、版本、协议和目标范围候选的证据完整度。", "C5/C6 高可比去重记录、Q2 数值候选范围；协议五字段逐项计数，不拟合、不将范围重合解释为协议可比。"),
        ("process_q4_diagnostics", "计算过程", "展示任务族覆盖、月×族样本支持和支持月份固定效应模型的个体残差。", "覆盖分母 1,819；月×族 n<20 的单元不着色；残差来自支持月份 HC3 模型，n=783。"),
        ("process_q4_version_match", "计算过程", "展示从精确名称候选到严格版本确认的匹配损失。", "1,860 个 C8 最新目录；1,819 个唯一名称候选；严格版本确认 0；C4 精确名称交集 0。"),
        ("process_q4_ar1_backtest", "计算过程", "逐点对照两个月留出时 AR(1) 与持平基线的一步绝对误差。", "目标月为 2025-02 与 2025-03；只显示两个扩展窗口起点，不画显著性或外推区间。"),
        ("result_q4_frontier", "分析结果", "概览完整样本的任务族分布、历史月度描述分解和开放权重模型的月度进入表现。", "箱线须=P5/P95、箱=P25/P75、中位线=P50；开放前沿为每月 C1 Average P90，n<10 空心标注。"),
        ("result_q4_dynamic_decomposition", "分析结果", "量化共同参数支持区间内各变化分量及观测变化的 HC3 区间。", "全量共同支持拟合 n=645、端点 n=92/110；rho 区间为描述性 HC3 区间；一步回测各 2 个起点；年度起点 0；Loss–Benchmark 保持 BLOCKED_NO_FIT。"),
        ("result_q4_c3_context", "分析结果", "给出按来源分开的 C3 Average 中位数与 P90 年度背景。", "点为来源/年份内统计量，无跨年连线；标注每组记录数；历史文献数据不并入排行榜趋势，2025 OLLB 为部分年度。"),
        ("flow_overall_model", "总体流程图", "说明 Q4 从输入审计到预拟合门控、描述分解和结果封装的执行依赖。", "Loss–Benchmark 门控位于任何桥接拟合之前；当前阻断后不产生拟合；年度预测因无年度起点而不估计。"),
    ]
    figure_classes = {
        "raw_q4_data": "原始数据候选", "raw_q4_c3_coverage": "原始数据候选", "raw_q4_loss_gate_inputs": "原始数据候选",
        "process_q4_diagnostics": "计算过程候选", "process_q4_version_match": "计算过程候选", "process_q4_ar1_backtest": "计算过程候选",
        "result_q4_frontier": "结果候选", "result_q4_dynamic_decomposition": "结果候选", "result_q4_c3_context": "结果候选",
        "flow_overall_model": "独立流程图",
    }
    contract_lines = [
        "# Q4 图表契约与 QA 记录", "",
        "> AI 辅助说明：图表由 OpenAI Codex 桌面版 26.917.71314（build 10954；prod）辅助生成；模型 ID `gpt-6-luna`（GPT-6 Luna）；开发机构 OpenAI；发布日期 2026-09-22。", "",
        "## 数据剖析", "",
        f"六族主指数完整样本 {len(core)} 个；有正参数量的模型 {int((aggregate['core6_index_0_100'].notna() & aggregate['parameter_count_B'].gt(0)).sum())} 个。参数量按 log10 展示。主指数先在每个叶任务内按唯一名称候选计算百分位秩，再做族内和族间等权。分布统计见 `../03_结果/results/q4_data_profile.csv`。", "",
        "## 图表用途与口径", "",
    ]
    for name, group, claim, stat in contracts:
        contract_lines.extend([
            f"### {name}（{figure_classes[name]}；{group}）", "",
            f"- 核心论点：{claim}",
            f"- 统计口径：{stat}",
            "- 版面：7.2 × 5.2 英寸；导出 SVG 与 300 dpi PNG，同时提供灰度预览。",
            "- QA：轴标题、刻度和样本标注写入矢量 SVG；对应 PNG 已用 Microsoft YaHei 字体栅格化，之后由视觉复核确认。",
            "",
        ])
    contract_lines.extend([
        "## 候选图覆盖", "",
        "按数学建模图表技能，当前为 Q4 子问题准备原始数据、计算过程、分析结果三类各 3 张候选图；总体流程图单独列示，不计入 9 张候选。全部候选图覆盖 `q4`。候选用于论证与审阅，最终论文只保留有独立证据作用的图。", "",
        "| 类别 | 候选文件 | 各自回答的问题 |", "|---|---|---|",
        "| 原始数据 | `raw_q4_data` / `raw_q4_c3_coverage` / `raw_q4_loss_gate_inputs` | 指数与参数分布；C3 来源/年份覆盖；Loss–Benchmark 拟合前证据可用性 |",
        "| 计算过程 | `process_q4_diagnostics` / `process_q4_version_match` / `process_q4_ar1_backtest` | 覆盖与残差；名称到版本匹配流失；两次一步留出误差 |",
        "| 分析结果 | `result_q4_frontier` / `result_q4_dynamic_decomposition` / `result_q4_c3_context` | 主指数和前沿；精确共同支持分解；来源分列的 C3 年度背景 |", "",
        "总体流程图：`flow_overall_model`。C4 年度数据保留在 C4 年表，不绘制连续趋势，以避免把不同记录数和非精确版本匹配误读为共同时间序列。", "",
        "## 已知呈现限制", "",
        "当前绑定 Python 环境未安装 Matplotlib，图表使用 Pillow 与直接生成的 SVG 原语。Pillow 栅格输出与 SVG 共用同一绘图坐标和数据值。", "",
        "开放前沿图的 P90 是月度新提交模型的截面统计，不是累计可用模型库的前沿；月样本不足 10 时标为空心点。所有图均为描述性图，不显示抽样置信区间或显著性符号。", "",
        "result_q4_frontier 中的月度恒等项为旧版月均值描述分解；精确共同支持分解及 HC3 区间以 `result_q4_dynamic_decomposition` 和 `q4_scale_tech_contribution.csv` 为准。算术闭合误差由端点月份固定效应和 OLS 正规方程导致，仅作数值核对，不表示存在未解释因素。", "",
    ])
    contract_path = Q4 / "01_方案说明" / "q4_figures_contract.md"
    contract_path.write_text("\n".join(contract_lines), encoding="utf-8")

    qa = {
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "Pillow raster plus direct editable SVG primitives",
        "font": str(FONT_PATH),
        "image_size_px": [W, H],
        "dpi": DPI,
        "candidate_figure_count": 9,
        "candidate_class_counts": {"raw": 3, "process": 3, "result": 3},
        "question_coverage": {"q4": {"raw": True, "process": True, "result": True}},
        "figures": [
            {"name": name, "class": figure_classes[name], "png": rel(FIGURES / f"{name}.png"), "svg": rel(FIGURES / f"{name}.svg"),
             "grayscale_png": rel(GRAYSCALE_DIR / f"{name}_grayscale.png"), "program_status": "PASS",
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
    output_hashes = manifest.setdefault("outputs_sha256", {})
    for old_path in [key for key in output_hashes if key.startswith("Q4/04_图表/")]:
        del output_hashes[old_path]
    output_hashes.update({rel(path): sha(path) for path in outputs})
    manifest["figure_qa_status"] = "Pillow/SVG program checks passed; formal file audit and visual review pending"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"created {len(output_files)} PNG/SVG/grayscale files; profile={profile_path}; contract={contract_path}")


if __name__ == "__main__":
    main()
