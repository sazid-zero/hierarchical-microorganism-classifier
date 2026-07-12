"""Hierarchical inference runner with confidence-based routing.

Usage:
    python scripts/run_hierarchical.py \
        --level1-model baseline_checkpoints/best_model \
        --level2-model bacteria_checkpoints/best_model \
        --data-dir all_classes_merged/test \
        --output-dir outputs/hierarchical_run --tau 0.7

Notes:
- Expects saved Keras `tf` SavedModel directories for both level1 and level2 models.
- Optionally provide JSON files with class name lists via `--level1-classes` and `--level2-classes`.
- If `--level1-bacteria-label` is provided, routing will trigger when level1 predicts that label with confidence >= tau.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_args():
    p = argparse.ArgumentParser(description='Run hierarchical inference with confidence routing')
    p.add_argument('--level1-model', required=True)
    p.add_argument('--level2-model', required=False)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--tau', type=float, default=0.7)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--level1-classes', type=str, default=None,
                   help='Path to JSON list of level1 class names')
    p.add_argument('--level2-classes', type=str, default=None,
                   help='Path to JSON list of level2 class names')
    p.add_argument('--level1-bacteria-label', type=str, default='Bacteria',
                   help='Label name in level1 that indicates bacteria (used to route to level2)')
    return p.parse_args()


def load_class_names(json_path: str | None, data_dir: Path) -> List[str]:
    if json_path and Path(json_path).exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # attempt to infer from data_dir subfolders
    if data_dir.exists() and any(p.is_dir() for p in data_dir.iterdir()):
        return sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    return []


def list_images(data_dir: Path) -> List[Path]:
    images: List[Path] = []
    for p in data_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            images.append(p)
    return sorted(images)


def load_and_preprocess(paths: List[Path], img_size: int) -> np.ndarray:
    arrs = []
    for p in paths:
        img = keras.utils.load_img(p, target_size=(img_size, img_size))
        a = keras.preprocessing.image.img_to_array(img)
        a = keras.applications.resnet.preprocess_input(a)
        arrs.append(a)
    return np.vstack([a[np.newaxis] if a.ndim==3 else a for a in arrs])


def batch_iter(items: List[Path], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    level1_model = tf.keras.models.load_model(args.level1_model)
    level2_model = None
    if args.level2_model:
        level2_model = tf.keras.models.load_model(args.level2_model)

    images = list_images(data_dir)
    if not images:
        raise SystemExit(f'No images found under {data_dir}')

    level1_class_names = load_class_names(args.level1_classes, data_dir)
    level2_class_names = []
    if args.level2_classes:
        level2_class_names = load_class_names(args.level2_classes, data_dir)
    # If class names not provided, try to read from models' metadata (not guaranteed)
    if not level1_class_names and hasattr(level1_model, 'classes_'):
        level1_class_names = list(getattr(level1_model, 'classes_'))

    results = []
    routed_count = 0

    for batch_paths in batch_iter(images, args.batch_size):
        X = load_and_preprocess(batch_paths, args.img_size)
        probs1 = level1_model.predict(X, verbose=0)
        preds1 = probs1.argmax(axis=1)
        conf1 = probs1.max(axis=1)

        final_preds = []
        level2_preds_col = [None] * len(batch_paths)
        level2_conf_col = [None] * len(batch_paths)
        routed_flags = [0] * len(batch_paths)

        # Identify indices to route
        to_route = []
        for i, (p_idx, p_conf) in enumerate(zip(preds1, conf1)):
            pred_name = level1_class_names[p_idx] if level1_class_names and p_idx < len(level1_class_names) else str(int(p_idx))
            if pred_name == args.level1_bacteria_label and p_conf >= args.tau and level2_model is not None:
                to_route.append(i)

        if to_route and level2_model is not None:
            # prepare inputs for routing subset
            route_paths = [batch_paths[i] for i in to_route]
            X2 = load_and_preprocess(route_paths, args.img_size)
            probs2 = level2_model.predict(X2, verbose=0)
            preds2 = probs2.argmax(axis=1)
            conf2 = probs2.max(axis=1)

        # build final predictions
        route_idx = 0
        for i in range(len(batch_paths)):
            p = batch_paths[i]
            p_idx = int(preds1[i])
            p_conf = float(conf1[i])
            p_name = level1_class_names[p_idx] if level1_class_names and p_idx < len(level1_class_names) else str(p_idx)
            final_name = p_name
            if i in to_route and level2_model is not None:
                rr = route_idx
                level2_p_idx = int(preds2[rr])
                level2_p_name = level2_class_names[level2_p_idx] if level2_class_names and level2_p_idx < len(level2_class_names) else str(level2_p_idx)
                level2_p_conf = float(conf2[rr])
                final_name = level2_p_name
                level2_preds_col[i] = level2_p_name
                level2_conf_col[i] = level2_p_conf
                routed_flags[i] = 1
                routed_count += 1
                route_idx += 1

            results.append({
                'image': str(p.relative_to(Path.cwd())),
                'level1_pred': p_name,
                'level1_conf': float(p_conf),
                'routed': routed_flags[i],
                'level2_pred': level2_preds_col[i],
                'level2_conf': level2_conf_col[i],
                'final_pred': final_name,
            })

    df = pd.DataFrame(results)
    out_csv = out_dir / 'hierarchical_predictions.csv'
    df.to_csv(out_csv, index=False)

    summary = {
        'n_images': len(images),
        'n_routed': int(routed_count),
        'tau': float(args.tau),
    }
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Saved predictions to', out_csv)
    print('Summary:', summary)


if __name__ == '__main__':
    main()
