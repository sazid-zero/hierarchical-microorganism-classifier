"""Utility helpers for dataset preparation and training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, List


def ensure_directory(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_image_files(directory: str | os.PathLike[str]) -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    )


def get_class_names(dataset_dir: str | os.PathLike[str]) -> List[str]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        return []
    classes = [path.name for path in dataset_dir.iterdir() if path.is_dir()]
    return sorted(classes)


def save_json(payload: Any, path: str | os.PathLike[str]) -> None:
    path = ensure_directory(Path(path).parent) / Path(path).name if Path(path).parent != Path('.') else Path(path)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: str | os.PathLike[str]) -> Any:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def summarize_class_counts(dataset_dir: str | os.PathLike[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_name in get_class_names(dataset_dir):
        counts[class_name] = len(list_image_files(Path(dataset_dir) / class_name))
    return counts
