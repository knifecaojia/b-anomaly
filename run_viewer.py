"""
数据集查看器启动入口 (Streamlit)
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "viewers/dataset_viewer_st.py",
        "--server.port", "7861",
        "--server.address", "127.0.0.1",
        "--browser.gatherUsageStats", "false",
    ], cwd=r"f:\Bear\apple")
