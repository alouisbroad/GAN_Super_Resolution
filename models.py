"""
Super Resolution GAN models.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def discriminator(channels=1):
    """

    :return: GAN discriminator model.
    """
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), **conv_args)(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), **conv_args)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.GlobalMaxPooling2D()(x)
    outputs = layers.Dense(1)(x)

    return keras.Model(inputs, outputs)


def generator(upscale_factor=16, channels=1):
    """
    --
    """
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(filters=64, kernel_size=5, **conv_args)(inputs)
    x = layers.Conv2D(filters=64, kernel_size=4, **conv_args)(x)
    x = layers.Conv2D(filters=32, kernel_size=4, **conv_args)(x)
    x = layers.Conv2D(filters=channels * (upscale_factor ** 2), kernel_size=4, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)
