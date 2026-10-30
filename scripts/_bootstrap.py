"""支持直接运行 scripts 下的入口，无需安装项目包。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
