"""稳定启动器：反复运行完整验证套件，自身无需改动。

用法（仓库根目录）::

    python -B src/d_verify/tests/run_validation.py

它等价于直接运行 ``run_all.py``（用 runpy 以 ``__main__`` 方式加载），
但把"运行入口"与"会被频繁修改的测试正文"分开，便于在受限环境下反复复跑。
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # -> src/

runpy.run_path(str(HERE / "run_all.py"), run_name="__main__")
