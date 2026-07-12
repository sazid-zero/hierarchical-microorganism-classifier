"""ResNet50 single-stage baseline model builder."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras.applications import ResNet50

from models.layers_head import build_classification_head


def build_resnet50_single_stage_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    weights: str = 'imagenet',
    trainable: bool = False,
):
    """Create a ResNet50 classifier matching the paper's single-stage baseline."""
    base_model = ResNet50(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling=None,
    )
    base_model.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=not trainable)
    outputs = build_classification_head(x, num_classes, name='resnet50_head')
    return keras.Model(inputs, outputs, name='ResNet50_SingleStage')

