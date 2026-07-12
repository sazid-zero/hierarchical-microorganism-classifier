"""Train a ResNet152V2 single-stage baseline for microorganism classification."""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import callbacks

from models.resnet152_single_stage_tpu import build_resnet152_single_stage_model


SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
WARMUP_EPOCHS = 20
EPOCHS = 100
TRAIN_RATIO = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the single-stage ResNet152V2 baseline.')
    parser.add_argument('--data-dir', default=os.environ.get('MICRO_DATA_DIR', 'all_classes_merged'))
    parser.add_argument('--output-dir', default=os.environ.get('MICRO_OUTPUT_DIR', 'baseline_checkpoints_resnet152'))
    parser.add_argument('--split-dir', default=os.environ.get('MICRO_SPLIT_DIR', 'baseline_split_resnet152'))
    parser.add_argument('--img-size', type=int, default=IMG_SIZE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--warmup-epochs', type=int, default=WARMUP_EPOCHS)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--train-ratio', type=float, default=TRAIN_RATIO)
    return parser.parse_args()


def build_strategy() -> Any:
    if os.environ.get('KAGGLE_TPU_ADDR'):
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=os.environ['KAGGLE_TPU_ADDR'])
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        return tf.distribute.TPUStrategy(resolver)
    if tf.config.list_physical_devices('GPU'):
        return tf.distribute.MirroredStrategy()
    return None


def count_images(directory: Path) -> int:
    return sum(1 for path in directory.rglob('*') if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'})


def list_class_names(data_dir: Path) -> list[str]:
    return sorted([path.name for path in data_dir.iterdir() if path.is_dir()])


def create_split(data_dir: Path, split_dir: Path, class_names: list[str], train_ratio: float, seed: int) -> dict[str, int]:
    for split_name in ('train', 'val', 'test'):
        (split_dir / split_name).mkdir(parents=True, exist_ok=True)
    split_stats: list[dict[str, Any]] = []
    totals = {'train': 0, 'val': 0, 'test': 0}
    for class_name in class_names:
        class_path = data_dir / class_name
        image_names = sorted([p.name for p in class_path.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}])
        if not image_names:
            continue
        if len(image_names) < 7:
            train_imgs, val_imgs, test_imgs = image_names, [], []
        else:
            train_imgs, temp_imgs = train_test_split(image_names, train_size=train_ratio, random_state=seed, shuffle=True)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=seed, shuffle=True)
        for split_name, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            dest_dir = split_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for image_name in imgs:
                (class_path / image_name).replace(dest_dir / image_name)
        totals['train'] += len(train_imgs)
        totals['val'] += len(val_imgs)
        totals['test'] += len(test_imgs)
        split_stats.append({'class': class_name, 'train': len(train_imgs), 'val': len(val_imgs), 'test': len(test_imgs)})
    pd.DataFrame(split_stats).to_csv(split_dir / 'split_stats.csv', index=False)
    return totals


def build_datasets(split_dir: Path, batch_size: int, img_size: int, class_names: list[str], seed: int):
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = keras.applications.resnet_v2.preprocess_input(image)
        return image, label

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.resize_with_crop_or_pad(image, img_size + 20, img_size + 20)
        image = tf.image.random_crop(image, size=[img_size, img_size, 3])
        return image, label

    def make_dataset(directory: Path, training: bool):
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size=(img_size, img_size),
            batch_size=None,
            shuffle=training,
            seed=seed,
            label_mode='categorical',
            class_names=class_names,
        )
        if training:
            ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=training)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(split_dir / 'train', training=True)
    val_ds = make_dataset(split_dir / 'val', training=False)
    test_ds = make_dataset(split_dir / 'test', training=False)
    return train_ds, val_ds, test_ds


def compute_class_weights(train_dir: Path, class_names: list[str]) -> dict[int, float]:
    counts = {idx: count_images(train_dir / class_name) for idx, class_name in enumerate(class_names)}
    total = sum(counts.values())
    return {idx: total / (len(class_names) * count) for idx, count in counts.items() if count > 0}


def make_callbacks(output_dir: Path, model_best_path: str, model_latest_path: str, log_path: Path):
    return [
        callbacks.ModelCheckpoint(model_best_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1, save_format='tf'),
        callbacks.ModelCheckpoint(model_latest_path, monitor='val_accuracy', save_best_only=False, save_freq='epoch', verbose=0, save_format='tf'),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, mode='max', verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-8, mode='min', verbose=1),
        callbacks.CSVLogger(log_path, append=True),
    ]


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(SEED)
    np.random.seed(SEED)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    split_dir = Path(args.split_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    class_names = list_class_names(data_dir)
    create_split(data_dir, split_dir, class_names, args.train_ratio, SEED)
    train_ds, val_ds, test_ds = build_datasets(split_dir, args.batch_size, args.img_size, class_names, SEED)
    steps_per_epoch = max(1, int(count_images(split_dir / 'train') / args.batch_size))
    val_steps = max(1, int(count_images(split_dir / 'val') / args.batch_size))
    class_weights = compute_class_weights(split_dir / 'train', class_names)

    strategy = build_strategy()
    scope = strategy.scope() if strategy is not None else nullcontext()
    with scope:
        model = build_resnet152_single_stage_model(len(class_names), input_shape=(args.img_size, args.img_size, 3), trainable=False)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_ds, validation_data=val_ds, epochs=args.warmup_epochs, steps_per_epoch=steps_per_epoch, validation_steps=val_steps, class_weight=class_weights, callbacks=make_callbacks(output_dir, str(output_dir / 'best_model'), str(output_dir / 'latest_model'), output_dir / 'warmup_log.csv'), verbose=1)
        model.trainable = True
        model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4, clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, steps_per_epoch=steps_per_epoch, validation_steps=val_steps, class_weight=class_weights, callbacks=make_callbacks(output_dir, str(output_dir / 'best_model'), str(output_dir / 'latest_model'), output_dir / 'finetune_log.csv'), verbose=1)
        print('Finished:', history.history['val_accuracy'][-1])


if __name__ == '__main__':
    main()
