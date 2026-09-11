# HUAWEI-Cup

华为杯比赛项目目录。

## 基础环境

推荐使用独立 Conda 环境 `huawei-cup`，不要直接使用 `base`、`pytorch` 或 `mmdet` 环境。

```powershell
conda env create -f environment.yml
conda activate huawei-cup
python -m ipykernel install --user --name huawei-cup --display-name "Python (huawei-cup)"
```

当前环境只包含通用科学计算和数据分析依赖。根据最终赛道和赛题，再补充优化、机器学习或深度学习组件。

## 目录约定

- `src/`：可复用代码
- `notebooks/`：探索分析和实验记录
- `data/`：数据文件
- `results/`：图表、模型结果和中间产物
- `docs/`：赛题、方案和论文材料
