import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DASHBOARD = ROOT / "dashboard" / "streamlit_app.py"
URL = "http://localhost:8501"


def main():
    if not DASHBOARD.exists():
        raise FileNotFoundError(f"Dashboard file not found: {DASHBOARD}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE

    subprocess.Popen(cmd, cwd=ROOT, creationflags=creationflags)
    time.sleep(4)
    webbrowser.open(URL)
    print(f"Opening Streamlit dashboard at {URL}")


if __name__ == "__main__":
    main()
