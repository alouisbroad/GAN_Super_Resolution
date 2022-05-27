"""
Main script where training of the super resolution gan takes place.

"""
from data_generator import tf_data_generator
from gan_class import SuperResolutionGAN
from models import discriminator, generator

import glob
import keras
import tensorflow as tf


def main():
    save_dir = "/data/users/lbroad/Machine_Learning/gan_outputs/"
    epochs = 2
    batch_size = 10
    steps_per_epoch = 2
    upscale_factor = 16

    gan = SuperResolutionGAN(discriminator=discriminator(channels=1),
                             generator=generator(upscale_factor=upscale_factor, channels=1))
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    file_list = glob.glob('/data/users/jbowyer/cbh_challenge_data/*')

    dataset = tf.data.Dataset.from_generator(tf_data_generator, args=(file_list, batch_size),
                                             output_types=(tf.dtypes.float32, tf.dtypes.float32))

    validationset = tf.data.Dataset.from_generator(tf_data_generator, args=(file_list[300:], batch_size),
                                                   output_types=(tf.dtypes.float32, tf.dtypes.float32))

    gan.fit(x=dataset.prefetch(tf.data.AUTOTUNE),
            validation_data=validationset.prefetch(tf.data.AUTOTUNE),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=50)


if __name__ == "__main__":
    main()
