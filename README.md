# HUAWEI-Cup

华为杯比赛项目目录。

## 基础环境

推荐使用独立 Conda 环境 `huawei-cup`，不要直接使用 `base`、`pytorch` 或 `mmdet` 环境。

```powershell
conda env create -f environment.yml
conda activate huawei-cup
python -m ipykernel install --user --name huawei-cup --display-name "Python (huawei-cup)"
```

如果 PyCharm 运行 `src/test.py` 时弹出“python.exe 中发生了未经处理的 win32
异常”，请在 `Settings | Project | Python Interpreter` 中添加已有 Conda 环境，
解释器选择 `D:\\Anaconda\\envs\\huawei-cup\\python.exe`。然后右键
`src/test.py` 运行即可。脚本已在导入 Matplotlib 前处理 Windows Conda DLL
搜索路径，避免与 base/CUDA 的 DLL 冲突。

当前环境只包含通用科学计算和数据分析依赖。根据最终赛道和赛题，再补充优化、机器学习或深度学习组件。

## 目录约定

- `src/`：可复用代码
- `notebooks/`：探索分析和实验记录
- `data/`：数据文件
- `results/`：图表、模型结果和中间产物
- `docs/`：赛题、方案和论文材料
