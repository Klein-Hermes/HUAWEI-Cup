# -*- coding: utf-8 -*-
"""
F 题论文图表统一风格。

设计依据：七篇往届优秀论文的图表实测规律（见 guides/图表与命名规范.md）。
最要紧的两条：
  1. 语义化配色且全文一致——每种颜色在全篇只承担一个含义；
  2. 约 75% 的图带一条参考线（最优参数、基准线或置信带）。

七篇的共同短板，也是本模块刻意补上的：误差带/置信区间几乎无人实现。
"""

from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# ----------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
FIG_DIR = Path(os.environ.get("FIG_OUT_DIR", str(HERE.parent)))
PAPER_MODE = os.environ.get("FIGURE_VARIANT", "plot").lower() == "paper"

# 结果数据根目录（团队工作区）。换机器时改这一行或设环境变量 Q1_DATA_ROOT。
DATA_ROOT = Path(os.environ.get(
    "Q1_DATA_ROOT",
    "/Users/zhaomengchen/Documents/Codex/HUAWEI-Cup/results",
))

def res(*parts):
    """拼出结果文件路径，例如 res('q1_1', 'v1', 'domain_summary.csv')"""
    return DATA_ROOT.joinpath(*parts)


# ----------------------------------------------------------------------
# 语义配色（全文唯一来源，改这里即全篇生效）
# ----------------------------------------------------------------------
C = {
    "ours":   "#0072B2",   # 蓝：本文方法 / 观测值
    "base":   "#D55E00",   # 橙红：基线 / 对照
    "alt":    "#009E73",   # 绿：第二对照
    "ref":    "#8C8C8C",   # 灰：参考线、次要文本
    "warn":   "#B33A3A",   # 暗红：异常、警示
    "hl":     "#E69F00",   # 琥珀：最优点、高亮区间
    "purple": "#CC79A7",   # 紫：第四组
    "sky":    "#56B4E9",   # 浅蓝：辅助
}

# 五个域的分组色（与上面语义色系一致，色盲友好）
DOMAIN_COLORS = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#CC79A7",
                 "#D55E00", "#8C8C8C"]

GRID = dict(color="#D9D9D9", linewidth=0.6, alpha=0.7)


# ----------------------------------------------------------------------
# 全局 rcParams
# ----------------------------------------------------------------------
def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        # macOS 中文字体优先；换机器时按序回退
        # Windows fallback prioritizes a CJK font with a true minus glyph.
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "SimSun", "Songti SC",
                            "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,

        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9.5,

        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,        # 网格必须在数据下方
        "axes.spines.top": False,      # 去上右脊，极简风
        "axes.spines.right": False,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "legend.frameon": False,       # 默认无边框；需要时单独开
        "figure.dpi": 130,
        "savefig.dpi": 600,            # 位图兜底 600dpi
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def style_axes(ax, grid_axis="both"):
    """统一坐标轴外观。图内不加标题——解释交给 LaTeX 题注。

    注意：`ax.grid(True, axis="x")` 只打开 x 网格，**不会关掉 y 网格**
    （rcParams 的 axes.grid=True 已把两者都开了），因此必须显式关另一轴。
    """
    if grid_axis == "both":
        ax.grid(True, **GRID)
    else:
        ax.grid(True, axis=grid_axis, **GRID)
        ax.grid(False, axis="y" if grid_axis == "x" else "x")
    ax.set_axisbelow(True)
    return ax


def plain_log_ticks(ax):
    """Show logarithmic major ticks as decimals, avoiding fragile mathtext minus glyphs."""
    formatter = FuncFormatter(lambda value, _position: f"{value:g}")
    if ax.get_xscale() == "log":
        ax.xaxis.set_major_formatter(formatter)
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_formatter(formatter)


def ref_line(ax, value=None, axis="y", label=None, color=None, ls="--"):
    """参考线。七篇约 75% 的图都有——基准线、阈值线、最优参数线。

    label 不再画成内联文字（内联文字必然压在数据上），而是作为图例项，
    由调用方 `ax.legend()` 统一收纳——图例会自动避开数据。
    """
    color = color or C["ref"]
    kw = dict(ls=ls, lw=1.0, color=color, zorder=1)
    if label:
        kw["label"] = label
    return ax.axhline(value, **kw) if axis == "y" else ax.axvline(value, **kw)


def legend_below(ax, ncol=3, y=-0.20, fontsize=9):
    """把图例放到坐标区下方——保证不压数据，且比 loc='best' 位置稳定。"""
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y),
                     ncol=ncol, fontsize=fontsize, frameon=False,
                     handlelength=1.6, columnspacing=1.4)


def label_outside(ax, x, y, text, color="#404040", fontsize=8, ha="left",
                  pad_frac=0.012):
    """把数值标签放到数据外侧。pad_frac 为坐标轴跨度的留白比例。"""
    return ax.text(x, y, text, color=color, fontsize=fontsize,
                   va="center", ha=ha, zorder=6)


def widen(ax, axis="x", frac=0.16):
    """扩大坐标轴范围留白，给外置标签腾地方。"""
    if axis == "x":
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + (hi - lo) * frac)
    else:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * frac)


def band(ax, lo, hi, axis="y", color=None, alpha=0.14):
    """置信带 / 最优区间。七篇几乎无人做——本模块刻意补上。"""
    color = color or C["hl"]
    if axis == "y":
        return ax.axhspan(lo, hi, color=color, alpha=alpha, lw=0)
    return ax.axvspan(lo, hi, color=color, alpha=alpha, lw=0)


def panel_label(ax, letter, loc=(0.015, 1.015)):
    """多子图角标 (a)(b)(c)。默认放在坐标区【上方】——区内的任意角落
    都可能被数据占据（散点、六边形密度图尤其如此）。"""
    ax.text(*loc, f"({letter})", transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom", ha="left")


# ----------------------------------------------------------------------
# 文字压数据的自动检测
# ----------------------------------------------------------------------
def check_text_overlap(fig, thresh=0.03):
    """检测图内文字是否压在数据上。

    做法：先正常渲染取得所有文字的像素包围盒，再把文字全部隐藏后重渲染，
    在「无文字」图上统计每个包围盒内的非白像素比例——非白即代表该处有
    线、点、柱或填充。超过阈值即判为文字压在数据上。

    隐藏文字不改变版面，因此两次渲染的包围盒位置可比。
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()

    # 只检查「坐标区内的文字」——刻度标签与轴标签本来就在轴外，
    # 与坐标轴边缘贴邻属正常排版，不按压数据判。
    items = []
    for ax in fig.axes:
        cand = list(ax.texts)                      # ax.text 添加的标注
        leg = ax.get_legend()
        if leg is not None:
            cand += list(leg.get_texts())          # 无边框图例会直接压在数据上
        for t in cand:
            if t.get_text().strip() and t.get_visible():
                items.append(t)

    boxes = []
    for t in items:
        try:
            bb = t.get_window_extent(renderer=rend)
        except Exception:
            continue
        if bb.width > 0 and bb.height > 0:
            boxes.append((t.get_text(), bb))

    # 渲染"无文字"底图时要隐藏两类东西：
    #   1. 全图所有文字——否则图例移到轴下方时会把刻度标签误判成"数据"；
    #   2. 网格线——网格是背景装饰而非数据，标签压在网格上属正常排版。
    hidden = []
    for t in fig.findobj(match=lambda o: isinstance(o, matplotlib.text.Text)):
        if t.get_visible():
            hidden.append(t)
            t.set_visible(False)
    grid_on = []
    for ax in fig.axes:
        for axis, name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            gl = axis.get_gridlines()
            if gl and gl[0].get_visible():
                grid_on.append((ax, name))
        ax.grid(False)
    fig.canvas.draw()
    # 必须 .copy()：buffer_rgba() 返回视图，恢复文字后重绘会覆盖它
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    for ax, axname in grid_on:
        ax.grid(True, axis=axname)
    for t in hidden:
        t.set_visible(True)
    fig.canvas.draw()

    hits = []
    for txt, bb in boxes:
        x0, x1 = int(max(0, bb.x0)) + 2, int(min(W, bb.x1)) - 2
        y0, y1 = int(max(0, H - bb.y1)) + 2, int(min(H, H - bb.y0)) - 2
        if x1 <= x0 or y1 <= y0:
            continue
        patch = buf[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        frac = float((patch < 245).any(axis=-1).mean())
        if frac > thresh:
            hits.append((txt.replace("\n", " ")[:38], frac))
    return hits


def save(fig, name, vector=True):
    """导出 600 dpi PNG、SVG 与 PDF；并保留灰度预览。

    导出前跑一次文字压数据检测，命中即在控制台报警——图面干净是本套图的
    硬要求（七篇优秀论文的做法是「图内无标题，解释交给 LaTeX 题注」）。
    """
    hits = check_text_overlap(fig)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png, dpi=600, facecolor="white")
    if vector:
        fig.savefig(FIG_DIR / f"{name}.svg", facecolor="white")
        fig.savefig(FIG_DIR / f"{name}.pdf", facecolor="white")
    from PIL import Image
    with Image.open(png) as image:
        image.convert("L").save(FIG_DIR / f"{name}_grayscale.png", dpi=(600, 600))
    plt.close(fig)
    if hits:
        print(f"    ⚠ {png.name}：{len(hits)} 处文字压在数据上")
        for txt, frac in hits:
            print(f"        「{txt}」覆盖 {frac:.0%} 非白像素")
    else:
        print(f"    ✓ {png.name}")
    return png


def plot_heading(fig, title, note, *, axes_top=0.79, title_y=0.975, note_y=0.90):
    """绘图版专用标题和说明；论文版保持无图题的版式。"""
    if PAPER_MODE:
        return
    fig.subplots_adjust(top=axes_top)
    fig.suptitle(title, y=title_y, fontsize=11.5, fontweight="bold")
    fig.text(0.5, note_y, note, ha="center", va="center", fontsize=8.2,
             color=C["ref"])


# ----------------------------------------------------------------------
# 标签美化
# ----------------------------------------------------------------------
def short_domain(s):
    """把 train_the_pile_xxx / metric/the_pile_xxx_val_loss 之类压成短名。"""
    s = str(s)
    for pre in ("metric/the_pile_", "train_the_pile_", "metric/", "train/"):
        if s.startswith(pre):
            s = s[len(pre):]
    for suf in ("_val_loss",):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def loss_target(t):
    return f"{short_domain(t)} 验证损失"
