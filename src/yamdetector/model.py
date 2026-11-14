"""DS-CNN model: one conv layer + four depthwise-separable blocks + GAP + softmax.

Sized for an ESP32-S3 (int8, TFLite Micro): ~21k parameters, a 13x10x64 feature
map at the deepest point, ReLU6 + BatchNorm (folded at export time).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MODEL_NAME = "yam_dscnn"
L2_WEIGHT = 1e-4
SPATIAL_DROPOUT = 0.1


def _ds_block(x, filters: int, stride: int, name: str):
    """Depthwise 3x3 (stride) -> BN -> ReLU6, pointwise 1x1 -> BN -> ReLU6."""
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        name=f"{name}_dw",
    )(x)
    x = layers.BatchNormalization(momentum=0.98, name=f"{name}_dw_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"{name}_dw_relu")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT),
        name=f"{name}_pw",
    )(x)
    x = layers.BatchNormalization(momentum=0.98, name=f"{name}_pw_bn")(x)
    x = layers.ReLU(max_value=6.0, name=f"{name}_pw_relu")(x)
    # Spatial dropout regularizes the small dataset without adding parameters
    # (it is a no-op at inference and disappears from the TFLite graph).
    return layers.SpatialDropout2D(SPATIAL_DROPOUT, name=f"{name}_drop")(x)


def build_model(cfg: dict, num_classes: int) -> keras.Model:
    """Build the DS-CNN for ``num_classes`` labels.

    Input shape comes from ``features.input_shape`` (49, 40, 1 by default).
    Strides: 2 in the stem, then 1/2/1/1 through the blocks, which keeps the
    compute at roughly 4M MACs while still modelling 25 x 20 -> 13 x 10 frames.
    """
    if num_classes < 2:
        raise ValueError(f"need at least 2 classes to train, got {num_classes}")
    input_shape = tuple(cfg["features"]["input_shape"])
    filters = 64

    inputs = keras.Input(shape=input_shape, name="microfrontend")
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT),
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(momentum=0.98, name="stem_bn")(x)
    x = layers.ReLU(max_value=6.0, name="stem_relu")(x)

    for index, stride in enumerate((1, 2, 1, 1), start=1):
        x = _ds_block(x, filters, stride, f"ds{index}")

    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dropout(0.1, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="logits")(x)
    return keras.Model(inputs, outputs, name=MODEL_NAME)


def count_parameters(model: keras.Model) -> int:
    return int(sum(int(tf.size(weight)) for weight in model.trainable_weights))


def describe(model: keras.Model) -> str:
    total = int(model.count_params())
    trainable = count_parameters(model)
    return f"{total:,} parameters ({trainable:,} trainable)"
