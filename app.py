"""Compatibility entrypoint for Streamlit hosting platforms."""

from pathlib import Path

APP_PATH = Path(__file__).parent / "app" / "streamlit_app.py"
exec(compile(APP_PATH.read_text(), str(APP_PATH), "exec"))
