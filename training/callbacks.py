"""Callback helpers for the restored training scripts."""

from __future__ import annotations

from tensorflow.keras import callbacks


def make_callbacks(checkpoint_dir: str, model_best_path: str, model_latest_path: str, log_path: str):
    """Create the callbacks used for warmup and fine-tuning."""
    return [
        callbacks.ModelCheckpoint(
            model_best_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_format='tf',
        ),
        callbacks.ModelCheckpoint(
            model_latest_path,
            monitor='val_accuracy',
            save_best_only=False,
            save_freq='epoch',
            verbose=0,
            save_format='tf',
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=25,
            restore_best_weights=True,
            mode='max',
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-8,
            mode='min',
            verbose=1,
        ),
        callbacks.CSVLogger(log_path, append=True),
    ]

