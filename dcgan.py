import os
from pathlib import Path
from glob import glob
from time import time

import cv2
import numpy as np
import pandas as pd

from tensorflow.keras.layers import (
    Input, Dense, Reshape, Flatten, Dropout, BatchNormalization,
    Activation, ZeroPadding2D, UpSampling2D, Conv2D, LeakyReLU
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras.datasets import fashion_mnist

def images_to_array(path, *, resize=None, gray=True):
    vector = []
    for image_path in glob(path):
        img=cv2.imread(image_path)
        if gray:
            img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if resize is not None:
            img=cv2.resize(img, resize)
        vector.append(img)
    imgs = np.array(vector, dtype='float32')
    imgs = imgs / 127.5 - 1
    if imgs.ndim == 3:
        imgs = np.expand_dims(imgs, axis=3)
    return imgs


class ImageHelper:

    def __init__(self, save_path, rows, cols, image_shape):
        self._save_path = save_path
        os.makedirs(str(save_path), exist_ok=True) 
        self.rows = rows
        self.cols = cols
        self.image_shape = image_shape
    
    @property
    def eval_size(self):
        return self.rows * self.cols

    def write_data_as_image(self, data, epoch):
        data = np.reshape((data + 1) * 127.5, (self.rows, self.cols) + self.image_shape)
        data = np.concatenate(np.concatenate(data, 1), 1)
        cv2.imwrite(
            "{}{}.png".format(
                self._save_path / 'epoch',
                str(epoch).zfill(6)),
            data,
        )


class DCGAN:
    def __init__(self, image_shape, image_helper, *, generator_input_dim=100):
        optimizer = Adam(0.0002, 0.5)
        self._gen_base_dims = [d // 4 for d in image_shape[:2]]
        assert (
            [v * 4 for v in self._gen_base_dims] == image_shape[:2],
            "Dimensions must be dividable by 4"
        )

        self._image_helper = image_helper
        self._image_shape = image_shape
        self._generator_input_dim = generator_input_dim
        self._image_channels = image_shape[-1]

        self._generator_model = self._build_generator_model()
        self._discriminator_model = self._build_and_compile_discriminator_model(optimizer)
        self._gan = self._build_and_compile_gan(
            optimizer, self._generator_model, self._discriminator_model,
        )

    def _build_generator_model(self):
        model_input = Input(shape=(self._generator_input_dim,))
        d1, d2 = self._gen_base_dims
        model_sequence = Sequential([
            Dense(128 * d1 * d2, activation='relu', input_dim=self._generator_input_dim),
            Reshape((d1, d2, 128)),
            UpSampling2D(),
            Conv2D(128, kernel_size=3, padding='same'),
            BatchNormalization(momentum=0.8),
            Activation('relu'),
            UpSampling2D(),
            Conv2D(64, kernel_size=3, padding='same'),
            BatchNormalization(momentum=.8),
            Activation('relu'),
            Conv2D(self._image_channels, kernel_size=3, padding='same'),
            Activation('tanh'),
        ])
        tensor = model_sequence(model_input)
        return Model(model_input, tensor)

    def _build_and_compile_discriminator_model(self, optimizer):
        model_input = Input(shape=self._image_shape)
        model_sequence = Sequential([
            Conv2D(32, kernel_size=3, strides=2, input_shape=self._image_shape, padding='same'),
            LeakyReLU(alpha=0.2),
            Dropout(.25),
            Conv2D(64, kernel_size=3, strides=2, padding='same'),
            ZeroPadding2D(padding=((0, 1), (0, 1))),
            BatchNormalization(momentum=0.8),
            LeakyReLU(alpha=.2),
            Dropout(.25),
            Conv2D(128, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(momentum=.8),
            LeakyReLU(.2),
            Dropout(.25),
            Conv2D(256, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(momentum=.8),
            LeakyReLU(alpha=.2),
            Dropout(.25),
            Flatten(),
            Dense(1, activation='sigmoid'),
        ])
        tensor = model_sequence(model_input)
        model = Model(model_input, tensor)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'],
        )
        model.trainable = False
        return model

    def _build_and_compile_gan(self, optimizer, generator_model, descriminator_model):
        real_input = Input(shape=(self._generator_input_dim,))
        generator_output = generator_model(real_input)
        discriminator_output = descriminator_model(generator_output)
        model = Model(real_input, discriminator_output)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
        )
        return model

    def _save_images(self, epoch):
        generated = self._predict_noise(self._image_helper.eval_size)
        self._image_helper.write_data_as_image(generated, epoch)

    def _get_noise(self, batch_size):
        return np.random.normal(0, 1, (batch_size, self._generator_input_dim))

    def _predict_noise(self, size):
        noise = self._get_noise(size)
        return self._generator_model.predict(noise)

    def _report_progress(self, epoch, discriminator_loss, generator_loss, start, save_image_each):
        speed = (time() - start) / (epoch + 1)
        next_save = save_image_each - (epoch % save_image_each)
        print("--------------------------------------")
        print(" Epoch: {} ({:.2f} s/epoch)".format(epoch, speed))
        print("     - Next Image {:.2f}s".format(next_save * speed))
        print(" Discriminator loss: {:.5f}".format(discriminator_loss[0]))
        print(" Generator loss:     {:.5f}".format(generator_loss))
        print("--------------------------------------")

    def train(self, epochs, train_data, batch_size, *, save_image_each=100, report_each=20):
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        start = time()
        for epoch in range(epochs):
            # Train Discriminator
            batch_indexes = np.random.randint(0, train_data.shape[0], batch_size)
            batch = train_data[batch_indexes]
            generated = self._predict_noise(batch_size)
            loss_real = self._discriminator_model.train_on_batch(batch, real)
            loss_fake = self._discriminator_model.train_on_batch(generated, fake)
            discriminator_loss = 0.5 * np.add(loss_real, loss_fake)

            # Train Generator
            noise = self._get_noise(batch_size)
            generator_loss = self._gan.train_on_batch(noise, real)

            if epoch % report_each == 0:
                self._report_progress(
                    epoch, discriminator_loss, generator_loss, start, save_image_each,
                )

            if epoch % save_image_each == 0:
                self._save_images(epoch)


def demo():
    def demo_data_loader():
        (X, _), (_, _) = fashion_mnist.load_data()
        X = X / 127.5 - 1
        return np.expand_dims(X, axis=3)

    X = demo_data_loader()
    image_helper = ImageHelper(Path('SAMPLES/demo-fashion'), 8, 8, X[0].shape)
    gan = DCGAN(X[0].shape, image_helper)
    gan.train(20000, X, batch_size=32)
