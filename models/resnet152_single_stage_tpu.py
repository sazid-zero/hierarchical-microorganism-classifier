"""ResNet152V2 single-stage baseline model builder."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras.applications import ResNet152V2

from models.layers_head import build_classification_head


def build_resnet152_single_stage_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str = 'imagenet',
    trainable: bool = False,
):
    """Create a ResNet152V2 classifier matching the paper's specialized baseline."""
    base_model = ResNet152V2(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling=None,
    )
    base_model.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=not trainable)
    outputs = build_classification_head(x, num_classes, hidden_units=(256,), dropout_rates=(0.5, 0.4), name='resnet152_head')
    return keras.Model(inputs, outputs, name='ResNet152V2_SingleStage')
