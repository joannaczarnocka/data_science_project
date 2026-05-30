#!/usr/bin/env python3
"""Trenowanie modeli (uruchamia scripts/train.py w venv projektu)."""

from __future__ import annotations

from _runner import ROOT, require_venv, run_script


def main() -> int:
    py = require_venv()
    print("Trenowanie modeli...")
    return run_script(py, ROOT / "scripts" / "train.py")


if __name__ == "__main__":
    raise SystemExit(main())
