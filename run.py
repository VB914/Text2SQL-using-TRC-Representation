from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


API_URL = "http://localhost:8000"
UI_URL = "http://localhost:8501"
ROOT = Path(__file__).resolve().parent


def start_process(command: list[str], env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(command, cwd=ROOT, env=env)


def stop_process(process: subprocess.Popen | None, name: str) -> None:
    if process is None or process.poll() is not None:
        return

    print(f"Stopping {name}...")
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> int:
    env = os.environ.copy()
    env.setdefault("TEXT2SQL_API_URL", API_URL)

    api_service: subprocess.Popen | None = None
    streamlit_ui: subprocess.Popen | None = None

    try:
        print("Starting API service...")
        api_service = start_process(
            [sys.executable, "-m", "uvicorn", "services.main:app", "--reload", "--port", "8000"],
            env,
        )

        time.sleep(3)
        if api_service.poll() is not None:
            print("API service failed to start. Check the error output above.")
            return api_service.returncode or 1

        print("Starting Streamlit UI...")
        streamlit_ui = start_process(
            [sys.executable, "-m", "streamlit", "run", "app/app.py"],
            env,
        )

        time.sleep(2)
        if streamlit_ui.poll() is not None:
            print("Streamlit UI failed to start. Check the error output above.")
            return streamlit_ui.returncode or 1

        print(f"App running at {UI_URL}")
        webbrowser.open(UI_URL)

        while True:
            if api_service.poll() is not None:
                print("API service stopped unexpectedly.")
                return api_service.returncode or 1
            if streamlit_ui.poll() is not None:
                print("Streamlit UI stopped unexpectedly.")
                return streamlit_ui.returncode or 1
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0
    finally:
        stop_process(streamlit_ui, "Streamlit UI")
        stop_process(api_service, "API service")


if __name__ == "__main__":
    raise SystemExit(main())
