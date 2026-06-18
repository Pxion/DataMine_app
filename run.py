"""
启动入口
直接运行: python run.py
"""
import os
import sys

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from web.app import create_app, app
from config import WEB_CONFIG

if __name__ == "__main__":
    create_app()
    print(f" 启动地址: http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
    app.run(
        host=WEB_CONFIG["host"],
        port=WEB_CONFIG["port"],
        debug=WEB_CONFIG["debug"],
    )
