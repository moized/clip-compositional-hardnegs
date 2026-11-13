"""Small helpers shared by every script: seeding, JSON I/O and run records."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jsonable, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _git(arguments: list, repo_dir: str | None = None) -> str | None:
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        out = subprocess.run(["git", "-C", repo_dir, *arguments], capture_output=True, text=True,
                             timeout=10, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def git_commit(repo_dir: str | None = None) -> str | None:
    """Commit hash of this repository (or of `repo_dir`), or None if unavailable."""
    return _git(["rev-parse", "HEAD"], repo_dir)


def git_modified(repo_dir: str | None = None) -> bool | None:
    """True if tracked files differ from the commit (the hash alone would then mislead)."""
    status = _git(["status", "--porcelain", "--untracked-files=no"], repo_dir)
    return None if status is None else bool(status)


# import name -> name of the installed distribution (for the version that pip reports)
_PACKAGES = {"torch": "torch", "torchvision": "torchvision", "open_clip": "open_clip_torch",
             "transformers": "transformers", "spacy": "spacy", "nltk": "nltk", "numpy": "numpy",
             "pandas": "pandas", "clip_benchmark": "clip-benchmark"}


def environment_info() -> dict:
    """Library versions and hardware, saved next to every result for reproducibility."""
    from importlib import metadata

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "git_modified": git_modified(),
    }
    for name, distribution in _PACKAGES.items():
        try:
            info[name] = metadata.version(distribution)
        except Exception:
            info[name] = None
    try:
        import spacy

        info["spacy_model"] = spacy.load("en_core_web_sm").meta.get("version")
    except Exception:
        info["spacy_model"] = None
    try:
        import torch

        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    return info


def run_record(args: Any = None, **extra: Any) -> dict:
    """Everything needed to repeat a run: command line, arguments, environment, time."""
    record = {
        "command": " ".join(sys.argv),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment_info(),
    }
    if args is not None:
        record["args"] = vars(args) if hasattr(args, "__dict__") else dict(args)
    record.update(extra)
    return record


def add_config_option(parser, section: str) -> None:
    """Add `--config FILE` to a parser and use that file's `section` as default values.

    The file is YAML; the keys of the section are the option names with "_" instead
    of "-". Options given on the command line still win. An unknown key is an error,
    so a typing mistake in the file cannot be ignored silently.
    """
    import argparse

    parser.add_argument("--config", default=None,
                        help=f"YAML file whose '{section}' section sets default values (see configs/).")
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", default=None)
    path = probe.parse_known_args()[0].config
    if not path:
        return
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        values = (yaml.safe_load(f) or {}).get(section) or {}
    known = {action.dest for action in parser._actions}
    unknown = sorted(set(values) - known)
    if unknown:
        parser.error(f"{path}: unknown option(s) in section '{section}': {', '.join(unknown)}")
    parser.set_defaults(**values)


class Timer:
    """Context manager that measures wall-clock seconds."""

    def __enter__(self) -> "Timer":
        self.start = time.time()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        self.seconds = time.time() - self.start
