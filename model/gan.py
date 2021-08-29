import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.callbacks import TensorBoard


class DCGan:
    def __init__(self):
        self.noise_vector_dim = 100
        self.kernel_size = 4

        self.generator_model = self._build_generator()
        self.discriminator_model = self._build_discriminator()
        # call before build_combined, because combined model need to set trainable to false
        self._compile_discriminator()
        self.combined_model = self._build_combined()
        self._compile_combined()

        self.tensorboard = None

    def _build_generator(self):
        input_layer = layers.Input(shape=self.noise_vector_dim)
        x = layers.Dense(units=64 * 7 * 7)(input_layer)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.Reshape(target_shape=(7, 7, 64))(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(filters=64, strides=1, kernel_size=self.kernel_size, padding='same')(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.LeakyReLU()(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(filters=128, strides=1, kernel_size=self.kernel_size, padding='same')(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=64, strides=1, kernel_size=self.kernel_size, padding='same')(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=64, strides=1, kernel_size=self.kernel_size, padding='same')(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=1, strides=1, kernel_size=self.kernel_size, padding='same')(x)
        output_layer = layers.Activation("tanh")(x)

        return models.Model(input_layer, output_layer, name="generator")

    def _build_discriminator(self):
        input_layer = layers.Input(shape=(28, 28, 1))
        x = layers.Conv2D(filters=64, strides=2, kernel_size=self.kernel_size, padding='same')(input_layer)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=64, strides=2, kernel_size=self.kernel_size, padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(rate=0.3)(x)

        x = layers.Conv2D(filters=128, strides=2, kernel_size=self.kernel_size, padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(rate=0.4)(x)

        x = layers.Conv2D(filters=128, strides=1, kernel_size=self.kernel_size, padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(rate=0.4)(x)

        x = layers.Flatten()(x)
        output_layer = layers.Dense(units=1, activation='sigmoid')(x)
        return models.Model(input_layer, output_layer, name="discriminator")

    def _compile_discriminator(self):
        self.discriminator_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def _build_combined(self):
        self.discriminator_model.trainable = False

        input_layer = layers.Input(shape=(self.noise_vector_dim,))
        x = self.generator_model(input_layer)
        output_layer = self.discriminator_model(x)
        return models.Model(input_layer, output_layer, name="combined")

    def _compile_combined(self):
        self.combined_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

    def summary(self):
        self.discriminator_model.summary()
        self.generator_model.summary()
        self.combined_model.summary()

    def _get_random_vector(self, batch_size):
        return np.random.normal(0, 1, (batch_size, self.noise_vector_dim))

    def train(self, x_train, num_epoch, batch_size):
        self.tensorboard = TensorBoard(log_dir="logs", histogram_freq=0, batch_size=batch_size, write_graph=True)
        self.tensorboard.set_model(self.combined_model)

        for epoch in range(num_epoch):
            print("epoch", epoch)
            input_vector = self._get_random_vector(batch_size)
            # Generated image
            fake_image = self.generator_model.predict(input_vector)
            fake_label = np.zeros((batch_size, 1))

            # Actual image
            index = np.random.randint(0, x_train.shape[0], size=batch_size),
            valid_image = x_train[index]
            valid_label = np.ones((batch_size, 1))

            epoch_logs = {}

            # Training for discriminator
            fake_loss, fake_acc = self.discriminator_model.train_on_batch(fake_image, fake_label)
            valid_loss, valid_acc = self.discriminator_model.train_on_batch(valid_image, valid_label)
            epoch_logs.update({'disc_loss': 0.5 * (fake_loss + valid_loss), 'disc_acc': 0.5 * (fake_acc + valid_acc)})

            # Training for generator
            input_vector = self._get_random_vector(batch_size)
            label = np.ones((batch_size, 1))
            logs = self.combined_model.train_on_batch(input_vector, label, return_dict=True)
            epoch_logs.update({'gen_' + key: logs[key] for key in logs})

            self.tensorboard.on_epoch_end(epoch, epoch_logs)

        self.tensorboard.on_train_end(None)

    def save(self, path):
        self.discriminator_model.save(os.path.join(path, "discriminator"))
        self.generator_model.save(os.path.join(path, "generator"))
        self.combined_model.save(os.path.join(path, "combined"))
