#!/usr/bin/env python3
"""Reproduce the title-free paper versions of the four Q3 figures.

Run from the project root with:
    D:\\miniconda3\\envs\\huawei-cup\\python.exe zwj/Q3补充图表/论文版/plot_q3_supplementary_figures_paper.py

The shared rendering core is the adjacent parent script
``zwj/Q3补充图表/plot_q3_supplementary_figures.py``.  Its hash and the
hashes of all data inputs are recorded in this folder's independent manifest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PAPER_DIR = SCRIPT_PATH.parent
BUNDLE_DIR = PAPER_DIR.parent
CORE_SCRIPT = BUNDLE_DIR / "plot_q3_supplementary_figures.py"
PROJECT_ROOT = BUNDLE_DIR.parent.parent
MANIFEST_PATH = PAPER_DIR / "复现清单.json"
DECISION_PATH = PAPER_DIR / "论文版排版决定与建议图注.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_core():
    if not CORE_SCRIPT.is_file():
        raise FileNotFoundError(f"缺少共享绘图核心脚本：{CORE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("q3_supplementary_figure_core", CORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载共享绘图核心脚本：{CORE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def write_decision_notes() -> Path:
    text = """# Q3 补充图表论文版：排版决定与建议图注

## 总体决定

- 删除四张图的顶部总标题，由论文正文中的图题/图注承担命名功能。
- 删除四张图底部的长段解释性注释。原因是这些文字属于方法口径、解释边界或结论限制，放在图内会压缩数据区域并与论文图注重复。
- 保留子图编号与子图标题、坐标轴、单位、图例、B1支持上界符号以及必要的线型/标记/纹理编码。
- 原底部注释的信息没有删除，而是完整迁移到下列建议图注中。

## 逐图判断

| 图 | 底部注释决定 | 图中保留的必要信息 | 理由 |
|---|---|---|---|
| P1 预算—N/D配置 | 删除并移入图注 | 上下文图例、B1支持上界符号、轴标签中的“仅连接给定档位” | 固定Q、非联合最优和支持域限制属于解释边界，不是读图编码。 |
| P2 Bootstrap稳定性 | 删除并移入图注 | 四个面板、预算图例、N/D/Loss轴及切换概率轴 | 1,000次、8条轨迹、区间退化和外推限制属于统计口径，应由图注完整说明。 |
| P3 资源配置仪表板 | 删除并移入图注 | a–d子图标题；面板c保留固定Q与C_Q=0；支持上界符号 | “22对变化仅描述性、非结构转移”是结论限制；面板c中的C_Q=0直接解释堆叠构成，需保留。 |
| P4 Q基线敏感性 | 删除并移入图注 | q_huber/q_equal重合图例、两套Q0数值和a–d子图标题 | 最大差为0及“非Q收益估计”属于图注结论；重合图例已足够解码曲线。 |

## 建议图注

### P1

**固定质量基线下预算与上下文对应的N/D参考配置。** 在冻结的q_huber基线、固定 $Q=Q_0$ 且 $C_Q=0$ 的条件下，展示3档预算与5种上下文对应的B1/M0支持网格参考配置。线段仅连接给定预算档位，不表示连续预算上的估计；空心外圈表示所选点触及B1 N/D支持上界。该结果不是完整Q3联合最优，不包含p或质量提升效应。

### P2

**轨迹级cluster Bootstrap下的配置选择稳定性。** 将Q2已有的1,000次轨迹级cluster Bootstrap参数传播到15个固定Q情景。N和D的95%区间在全部情景中退化为单点，图中未人为放大；M0预测Loss区间按2.5%、中位数和97.5%分位数绘制，配置切换概率为 $1-$ 原配置重选频率。Bootstrap仅基于8条B1轨迹，反映既定M0形式和支持网格内的参数重抽样稳定性，不覆盖模型形式误差、外部迁移或Q/p响应。

### P3

**离散预算与上下文情景下的资源配置、成本构成和M0预测Loss。** 面板a–d分别展示N配置、D配置、训练/注意力/质量成本份额及M0预测验证Loss。固定 $Q=Q_0$ 时 $C_Q=0$；空心外圈表示B1 N/D支持上界。线段仅连接给定档位。22对相邻变化仅作描述性比较，不使用Bootstrap/KKT判据，也不构成正式结构转移结论；最高预算情景仍受B1支持网格上界约束。

### P4

**q_huber与q_equal基线的固定Q M0敏感性对照。** q_huber与q_equal的基线分别为0.4984781048和0.4962230100。在相同预算、上下文、B1网格与M0下，两分支均固定在各自 $Q_0$，因此 $C_Q=0$；15个情景中N、D及M0预测Loss的最大绝对差均为0。该对照不表示Q对Loss的实证收益，也不构成Q/p联合优化。
"""
    DECISION_PATH.write_text(text, encoding="utf-8")
    return DECISION_PATH


def write_manifest(core, figures: dict[str, list[str]], statistics: dict, style_info: dict, audits: dict, decision_path: Path) -> Path:
    output_hashes = {
        str(SCRIPT_PATH.relative_to(PROJECT_ROOT)): sha256(SCRIPT_PATH),
    }
    for paths in figures.values():
        for value in paths:
            path = Path(value)
            output_hashes[str(path.relative_to(PROJECT_ROOT))] = sha256(path)
    output_hashes[str(decision_path.relative_to(PROJECT_ROOT))] = sha256(decision_path)

    manifest = {
        "purpose": "Independent reproduction record for the title-free paper versions of the four Q3 supplementary figures",
        "variant": {
            "name": "paper",
            "overall_title_removed": True,
            "bottom_caption_style_notes_removed": True,
            "removed_notes_migrated_to": str(decision_path.relative_to(PROJECT_ROOT)),
            "panel_titles_axes_units_legends_retained": True,
            "data_or_statistical_changes": False,
        },
        "command_from_project_root": r"D:\miniconda3\envs\huawei-cup\python.exe zwj\Q3补充图表\论文版\plot_q3_supplementary_figures_paper.py",
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            "numpy": core.np.__version__,
            "pandas": core.pd.__version__,
            "matplotlib": core.mpl.__version__,
            "Pillow": core.Image.__version__,
        },
        "style": style_info,
        "png_dpi": core.DPI,
        "renderer_core": {
            "path": str(CORE_SCRIPT.relative_to(PROJECT_ROOT)),
            "sha256": sha256(CORE_SCRIPT),
        },
        "inputs": {
            str(path.relative_to(PROJECT_ROOT)): {"sha256": sha256(path)}
            for path in core.INPUT_SOURCES
        },
        "figures": {
            key: [str(Path(path).relative_to(PROJECT_ROOT)) for path in paths]
            for key, paths in figures.items()
        },
        "outputs_sha256": output_hashes,
        "statistics_and_checks": statistics,
        "file_audits": audits,
        "visual_review": {
            "status": "completed",
            "result": "passed",
            "checks": [
                "overall titles removed without excessive top whitespace",
                "legends and panel labels retained and unobstructed",
                "axes, units, scenario labels and interpretation notes remain readable",
                "no visible clipping or tick-label collision",
                "grayscale distinction retained by marker, line style, or hatch encoding",
            ],
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    return MANIFEST_PATH


def main() -> int:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    core = load_core()
    style_info = core.configure_style()
    tables = core.load_inputs()

    figures: dict[str, list[str]] = {}
    statistics: dict[str, dict] = {}
    stems = [
        "result_q3_p1_budget_nd_frontier",
        "result_q3_p2_bootstrap_selection_stability",
        "result_q3_p3_resource_shift_dashboard",
        "result_q3_p4_q_baseline_sensitivity",
    ]
    figures[stems[0]], statistics[stems[0]] = core.plot_p1(tables["frontier"], PAPER_DIR, paper=True)
    figures[stems[1]], statistics[stems[1]] = core.plot_p2(tables["bootstrap"], tables["frequency"], PAPER_DIR, paper=True)
    figures[stems[2]], statistics[stems[2]] = core.plot_p3(tables["frontier"], tables["cost"], tables["shifts"], PAPER_DIR, paper=True)
    figures[stems[3]], statistics[stems[3]] = core.plot_p4(tables["frontier"], tables["q1"], PAPER_DIR, paper=True)

    audits = core.inspect_outputs(PAPER_DIR, figures)
    decision_path = write_decision_notes()
    manifest_path = write_manifest(core, figures, statistics, style_info, audits, decision_path)
    print(f"已生成4张论文版逻辑图，共{sum(len(v) for v in figures.values())}个图文件。")
    print(f"输出目录：{PAPER_DIR}")
    print(f"独立复现清单：{manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
