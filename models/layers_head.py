"""Reusable classification head builders for the restored microorganism models."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_classification_head(
    inputs,
    num_classes: int,
    hidden_units: tuple[int, ...] = (512, 256),
    dropout_rates: tuple[float, ...] = (0.5, 0.4, 0.3),
    l2_weight: float = 0.001,
    output_activation: str = 'softmax',
    name: str | None = None,
):
    """Construct the classifier head used in the paper."""
    x = layers.GlobalAveragePooling2D(name=f'{name}_gap' if name else None)(inputs)
    x = layers.BatchNormalization(name=f'{name}_bn1' if name else None)(x)
    x = layers.Dropout(dropout_rates[0], name=f'{name}_drop1' if name else None)(x)

    for index, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_weight),
            name=f'{name}_dense{index + 1}' if name else None,
        )(x)
        x = layers.BatchNormalization(name=f'{name}_bn{index + 2}' if name else None)(x)
        x = layers.Dropout(dropout_rates[index + 1] if index + 1 < len(dropout_rates) else dropout_rates[-1], name=f'{name}_drop{index + 2}' if name else None)(x)

    outputs = layers.Dense(
        num_classes,
        activation=output_activation,
        name=f'{name}_output' if name else None,
    )(x)
    return outputs
