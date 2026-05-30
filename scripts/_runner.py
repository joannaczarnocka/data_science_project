"""Wspolne funkcje uruchomieniowe (Windows / Linux / macOS)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent

PIP_TRUSTED_HOSTS = [
    "--trusted-host",
    "pypi.org",
    "--trusted-host",
    "pypi.python.org",
    "--trusted-host",
    "files.pythonhosted.org",
]


def venv_python() -> Path | None:
    """Sciezka do interpretera w venv projektu."""
    if sys.platform == "win32":
        candidate = ROOT / "venv" / "Scripts" / "python.exe"
    else:
        candidate = ROOT / "venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def configure_pip_env() -> None:
    """Ustawienia pip/SSL jak w dawnych skryptach .bat."""
    pip_ini = ROOT / "pip.ini"
    if pip_ini.is_file():
        os.environ["PIP_CONFIG_FILE"] = str(pip_ini)

    try:
        import certifi

        cert_path = certifi.where()
        os.environ["SSL_CERT_FILE"] = cert_path
        os.environ["REQUESTS_CA_BUNDLE"] = cert_path
    except ImportError:
        pass


def run(cmd: list[str], *, check: bool = False) -> int:
    """Uruchom polecenie w katalogu glownym projektu."""
    result = subprocess.run(cmd, cwd=ROOT)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


def require_venv() -> Path:
    """Zwroc interpreter venv lub zakoncz z komunikatem."""
    py = venv_python()
    if py is None:
        print("Brak venv. Uruchom najpierw:")
        print(f"  {sys.executable} scripts/setup_env.py")
        raise SystemExit(1)
    return py


def run_module(py: Path, module: str, *args: str) -> int:
    return run([str(py), "-m", module, *args])


def run_script(py: Path, script: Path, *args: str) -> int:
    return run([str(py), str(script), *args])
