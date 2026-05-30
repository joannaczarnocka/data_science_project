#!/usr/bin/env python3
"""Uruchamia aplikacje Streamlit z venv projektu."""

from __future__ import annotations

import sys

from _runner import require_venv, run_module


def main() -> int:
    py = require_venv()
    run_module(py, "pip", "install", "-q", "xgboost", "torch")
    return run_module(py, "streamlit", "run", "app/streamlit_app.py", *sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
