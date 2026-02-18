#!/usr/bin/env python3
"""
dashboard/run.py - 启动可视化面板服务器

用法:
    cd /path/to/cypto
    source .venv/bin/activate
    python dashboard/run.py

然后在浏览器中打开 http://localhost:8000
"""

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "dashboard.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(ROOT / "src"), str(ROOT / "dashboard")],
        log_level="info",
    )
