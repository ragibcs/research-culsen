"""
Structured logging and JSON export utilities.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict


def get_logger(name: str = "culsen_bn", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger with console and optional file output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


class RobustJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that serializes NumPy scalars, arrays, sets, and Paths.
    """
    def default(self, obj: Any) -> Any:
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass

        if isinstance(obj, set):
            return sorted(list(obj))
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """
    Save serializable data to JSON file with numpy/set/path support.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, cls=RobustJSONEncoder)


def load_json(path: str) -> Any:
    """
    Load data from JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
