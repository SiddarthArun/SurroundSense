"""DS-CNN model: one conv stem + five depthwise-separable blocks + GAP + softmax.

Sized for an ESP32-S3 (int8, TFLite Micro). The stem halves the time axis once,
then the blocks drop it at 25 x 20 -> 13 x 10 -> 7 x 5 while the channel count
grows 64 -> 96 -> 128 -> 160 -> 192.

~92k parameters and ~9.5M MACs, which is about 2.4x the compute of the original
5-class model (38 ms/window on the S3, so roughly 90-110 ms): comfortably inside
the 0.5 s hop in ``classifier.h`` while giving a 13-class softmax enough capacity
to separate scream from speech and knock from thunder.

The input/output contract is unchanged: ``(49, 40, 1)`` float features in,
softmax over ``num_classes`` out, ReLU6 + BatchNorm for a clean int8 conversion.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MODEL_NAME = "surround_dscnn"
L2_WEIGHT = 1e-4
SPATIAL_DROPOUT = 0.15

STEM_FILTERS = 64
# (filters, stride) per depthwise-separable block.
BLOCKS = ((96, 1), (128, 2), (128, 1), (160, 2), (192, 1))


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
    # Spatial dropout regularizes without adding parameters (a no-op at
    # inference, and it disappears from the TFLite graph).
    return layers.SpatialDropout2D(SPATIAL_DROPOUT, name=f"{name}_drop")(x)


def build_model(cfg: dict, num_classes: int) -> keras.Model:
    """Build the DS-CNN for ``num_classes`` labels.

    Input shape comes from ``features.input_shape`` (49, 40, 1 by default).
    """
    if num_classes < 2:
        raise ValueError(f"need at least 2 classes to train, got {num_classes}")
    input_shape = tuple(cfg["features"]["input_shape"])

    inputs = keras.Input(shape=input_shape, name="microfrontend")
    x = layers.Conv2D(
        STEM_FILTERS,
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

    for index, (filters, stride) in enumerate(BLOCKS, start=1):
        x = _ds_block(x, filters, stride, f"ds{index}")

    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="logits")(x)
    return keras.Model(inputs, outputs, name=MODEL_NAME)


def count_parameters(model: keras.Model) -> int:
    return int(sum(int(tf.size(weight)) for weight in model.trainable_weights))


def mult_adds(model: keras.Model) -> int:
    """Rough MAC count of one forward pass (the convolutions, which dominate)."""
    total = 0
    for layer in model.layers:
        output = layer.output.shape
        if isinstance(layer, layers.DepthwiseConv2D):
            # One tap per kernel position per output channel (channels are not summed).
            total += int(np.prod(output[1:4])) * int(layer.kernel_size[0] * layer.kernel_size[1])
        elif isinstance(layer, layers.Conv2D):
            total += (
                int(np.prod(output[1:4]))
                * int(layer.kernel_size[0] * layer.kernel_size[1])
                * int(layer.input.shape[-1])
            )
        elif isinstance(layer, layers.Dense):
            total += int(layer.input.shape[-1]) * int(layer.output.shape[-1])
    return total


def describe(model: keras.Model) -> str:
    total = int(model.count_params())
    trainable = count_parameters(model)
    return f"{total:,} parameters ({trainable:,} trainable, ~{mult_adds(model) / 1e6:.1f}M MACs)"
