"""
This script prepares the data as a generator for the GAN.
"""
import os
import iris
import numpy as np
import tensorflow as tf

from utils import make_stash_string, normalisation, generate_LR


def tf_data_generator(file_list, batch_size=10, variable="temperature"):
    """
    Simple generator that reads in the temperature data on levels.
    Each cube has 70 levels, so non-divisible batch sizes with give
    varying batches.
    :param file_list: list of file paths.
    :param batch_size: int batch size.
    :param variable: string of either "specific humidity", "pressure" or "temperature",
    depending of what you are training.
    """
    i = 0
    stash_codes = {"specific humidity": (0, 10),
                   "pressure": (0, 408),
                   "temperature": (16, 4)}
    tmp = make_stash_string(*stash_codes[variable])['stashstr_iris']
    while True:  # This loop makes the generator an infinite loop
        if i == 0:
            np.random.shuffle(file_list)
            print(file_list[i])
        file = file_list[i]

        i = (i + 1) % len(file_list)
        data = iris.load_cube(os.fsdecode(file), iris.AttributeConstraint(STASH=tmp)).data
        data = tf.random.shuffle(tf.reshape(tf.convert_to_tensor(data), (*data.shape, 1)))
        data = normalisation(data, "zscore")
        for local_index in range(0, data.shape[0], batch_size):
            hr = data[local_index:(local_index + batch_size)]
            yield generate_LR(hr), hr
