"""Prepare a general-microorganism dataset compatible with the paper's Level 1 classifier."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

GENERAL_CLASSES = ['Amoeba', 'Bacteria', 'Euglena', 'Hydra', 'Paramecium', 'Yeast']
CLASS_ALIASES = {
    'Rod Bacteria': 'Bacteria',
    'Spherical Bacteria': 'Bacteria',
    'Spiral Bacteria': 'Bacteria',
    'Rod_bacteria': 'Bacteria',
    'Spherical_bacteria': 'Bacteria',
    'Spiral_bacteria': 'Bacteria',
    'Micro_Organism': None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare a general microorganism dataset for Level 1 training.')
    parser.add_argument('--source', nargs='+', default=[os.environ.get('MICRO_SOURCE_DIR', '.')])
    parser.add_argument('--output', default=os.environ.get('MICRO_OUTPUT_DIR', 'hierarchical_dataset'))
    parser.add_argument('--limit', type=int, default=800)
    parser.add_argument('--min-images', type=int, default=500)
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    )


def copy_images(src_dir: Path, dest_dir: Path, limit: int) -> int:
    images = list_images(src_dir)
    random.shuffle(images)
    selected = images[:limit]
    dest_dir.mkdir(parents=True, exist_ok=True)
    for image_path in selected:
        target_path = dest_dir / f"{src_dir.name}_{image_path.stem}{image_path.suffix}"
        shutil.copy2(image_path, target_path)
    return len(selected)


def prepare_dataset(source_dirs: list[Path], output_dir: Path, limit: int, min_images: int) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in GENERAL_CLASSES:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f'Skipping missing source: {source_dir}')
            continue
        print(f'Processing source: {source_dir}')
        for child in sorted(source_dir.iterdir()):
            if not child.is_dir():
                continue
            target_class = CLASS_ALIASES.get(child.name, child.name)
            if target_class is None:
                continue
            if target_class in GENERAL_CLASSES:
                copy_images(child, output_dir / target_class, limit)

    for class_name in GENERAL_CLASSES:
        class_dir = output_dir / class_name
        images = list_images(class_dir)
        if 0 < len(images) < min_images:
            needed = min_images - len(images)
            print(f'Oversampling {class_name}: {len(images)} -> {needed} synthetic samples')
            for index in range(needed):
                source_path = random.choice(images)
                with Image.open(source_path) as image:
                    image = image.convert('RGB')
                    if random.random() > 0.5:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
                    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
                    if random.random() > 0.8:
                        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
                    image.save(class_dir / f'Synthetic_{index}{source_path.suffix}')


def main() -> None:
    args = parse_args()
    prepare_dataset([Path(path) for path in args.source], Path(args.output), args.limit, args.min_images)
    print(f'Prepared dataset at {args.output}')


if __name__ == '__main__':
    main()
