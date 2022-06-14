"""
Supporting functions for Super Res GAN.
"""
from tensorflow.keras.layers import AveragePooling2D
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import tensorflow as tf
import numpy as np


def normalisation(data, ntype):
    """
    :param data: input data - tensor or array
    :param ntype: Min-max normalisation or z-score (standard score) normalisation
    :return: normalised data
    """
    if ntype == "minmax":
        norm = np.array(data)
        norm = (norm - norm.min()) / (norm.max() - norm.min())
        return tf.convert_to_tensor(norm)
    elif ntype == "zscore":
        mu, variance = tf.nn.moments(data, axes=[0, 1, 2, 3])
        return (data - mu) / tf.math.sqrt(variance)
    else:
        "Incorrect arguments, either minmax or zscore."


def denormalisation(data, mu, var):
    """
    :param data: tensor of normalised data
    :param mu: mu of data
    :param var: variance of data
    :return: de-normalised data
    """
    return data * tf.math.sqrt(mu) + var


def predict_on_data(model, dataset):
    """
    prediction with normalisation and de-normalisation steps.
    """
    # de-normalise
    mu, var = tf.nn.moments(dataset, axes=[0, 1, 2, 3])
    data = model.predict(normalisation(dataset, "zscore"))
    return denormalisation(data, mu=mu, var=var)


def upscale_image(model, dataset, mu, var):
    """
    Predict on normalised downscaled data.
    """
    # de-normalise
    data = model.predict(dataset)
    return denormalisation(data, mu=mu, var=var)


def plot_results(prediction, prefix, title):
    """
    Create an image showing upscaled data.
    """
    fig = plt.figure(figsize=(12, 18))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.10, right=0.975, bottom=0.10, top=0.975, wspace=1e-1, hspace=1e-3)

    ax1 = plt.subplot(gs[0])
    cmap1 = ax1.imshow(np.flipud(prediction[0, :, :, 0]))
    cmap1.set_clim([0, 1])
    plt.title("{}".format(title))
    plt.colorbar(cmap1, orientation='horizontal')
    plt.savefig("/data/users/lbroad/Machine_Learning/training_images/{}_{}.png".format(prefix, title))


def make_stash_string(stashsec, stashcode):
    """

    :param stashsec:
    :param stashcode:
    :return:
    """
    #
    stashsecstr = str(stashsec)
    if stashsec < 10:
        stashsecstr = '0' + stashsecstr
    # endif
    #
    stashcodestr = str(stashcode)
    if stashcode < 100:
        stashcodestr = '0' + stashcodestr
    # endif
    if stashcode < 10:
        stashcodestr = '0' + stashcodestr
    # endif
    stashstr_iris = 'm01s' + stashsecstr + 'i' + stashcodestr
    stashstr_fout = stashsecstr + stashcodestr
    return {'stashstr_iris': stashstr_iris, 'stashstr_fout': stashstr_fout}


def generate_LR(original_data, factor=16):
    """
    Use average pooling layer to shrink by 1/factor
    :param original_data:
    :param factor:
    :return:
    """
    return AveragePooling2D(pool_size=(factor, factor))(original_data)


def normalisation_by_channels(data, ntype):
    """
    :param data: input data - tensor or array
    :param ntype: Min-max normalisation or z-score (standard score) normalisation
    :return: normalised data
    """
    _, _, _, c = data.shape
    result = []
    if ntype == "minmax":
        for i in range(c):
            norm = np.array(data[:, :, :, i])
            norm = (norm - norm.min()) / (norm.max() - norm.min())
            result.append(tf.convert_to_tensor(norm))
        return tf.stack(result, axis=3)
    elif ntype == "zscore":
        mu, variance = tf.nn.moments(data, axes=[0, 1, 2])
        for i in range(c):
            norm = (data - mu) / tf.math.sqrt(variance)
            result.append(norm)
        return tf.stack(result, axis=3)
    else:
        "Incorrect arguments, either minmax or zscore."
