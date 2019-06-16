from glob import glob
import os
from pathlib import Path
import pickle
from time import time

import cv2
import numpy as np
import pandas as pd

from tensorflow.keras.layers import (
    Input, Dense, Reshape, Flatten, Dropout, BatchNormalization,
    Activation, ZeroPadding2D, UpSampling2D, Conv2D, LeakyReLU
)
from tensorflow.keras.models import Sequential, Model, load_model
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
        if not isinstance(save_path, Path):
            save_path = Path(save_path)
        self._save_path = save_path
        os.makedirs(str(save_path), exist_ok=True) 
        self.rows = rows
        self.cols = cols
        self.image_shape = image_shape

    def serialized(self):
        return {
            "save_path": self._save_path,
            "rows": self.rows,
            "cols": self.cols,
            "image_shape": self.image_shape,
        }
    
    @property
    def eval_size(self):
        return self.rows * self.cols

    def write_data_as_image(self, data, *, epoch=None, filename=None):
        data = np.reshape((data + 1) * 127.5, (self.rows, self.cols) + self.image_shape)
        data = np.concatenate(np.concatenate(data, 1), 1)
        if filename is None:
            filename = "{}{}.png".format(
                self._save_path / 'epoch',
                str(epoch).zfill(6),
            )
        cv2.imwrite(filename, data)


class DCGAN:
    def __init__(
        self, image_shape, image_helper=None, *,
        generator_input_dim=100, models=None, generator_settings=None,
        discriminator_settings=None, optimizer_settings=None,
    ):
        """Class to manage a DCGAN

        Image generation is fun!

        Arguments:
            image_shape (tuple):
                Shape of the image, typically (64, 64, 3) or (64, 64, 1)
            image_helper (ImageHelper):
                To output sample images. If omitted class can't save images
        Keyword Arguments:
            generator_input_dim (int):
                Length of the vector that the generator takes as input
                Default: 100
            models (Optional[Dict]):
                Used to load previously saved network instead of building new
                ones. The dictionary should have keys 'gen' and 'dis' for the
                generator model and discriminator model. This argument
                shouldn't be needed to use, instead see `DCGAN.load`
            generator_settings (Optional[Dict]):
                If supplied can override the default generator settings.
                Understood keys:
                    * 'conv2d-startsize' (int): The independent size of the first
                        2D convolution layer (remaining dimensions come from
                        the image shape). The value must be dividable by 2.
                        There are 2 of these layers, and each decrease with a
                        factor 2 compared to previous
                        Default: 128
                    * 'normalization-momentum' (float): The batch normalization
                        momentum.
                        Default: 0.8
            discriminator_settings (Optional[Dict]):
                If supplied can override the default discriminator settings.
                Understoode keys:
                    * 'conv2d-startsize' (int): The independent size of the first
                        2D convolution layer. There are 3 of these layers,
                        and each increase with a factor 2 compared to previous
                        Default: 32
                    * 'normalization-momentum' (float): The batch normalization
                        momentum.
                        Default: 0.8
                    * 'dropout' (float): The dropoout factor the counters
                        overfitting.
                        Default: 0.25
                    * 'leakyrelu-alpha': Alpha setting to the LeakyReLU layer
                        Default: 0.2
            optimizer_settings (Optional[Dict]):
                If supplies overrides the Adam optimizer's defaults
                Understood keys:
                    * 'learning-rate': How much is learned each batch
                        Default: 0.0002
                    * 'beta_1': The first beta factor
                        Default: 0.5
        """
        if optimizer_settings is None:
            optimizer_settings = {}
        learning_rate = float(optimizer_settings.get('learning-rate', 0.0002))
        assert learning_rate >= 0
        beta_1 = float(optimizer_settings.get('beta_1', 0.5))
        assert 0 < beta_1 < 1
        optimizer = Adam(learning_rate, beta_1)
        self._gen_base_dims = [d // 4 for d in image_shape[:2]]
        assert all(
            a * 4 == b for a, b in zip(self._gen_base_dims,  image_shape[:2])
        ), "Dimensions must be dividable by 4 ({})".format(image_shape[:2])
        

        self._image_helper = image_helper
        self._image_shape = image_shape
        self._generator_input_dim = generator_input_dim
        self._image_channels = image_shape[-1]
        self._epochs = 0
        if generator_settings is None:
            generator_settings = {}
        self._generator_settings = generator_settings
        if discriminator_settings is None:
            discriminator_settings = {}
        self._discriminator_settings = discriminator_settings

        if models:
            self._generator_model = models['generator']
            self._discriminator_model = models['discriminator']
            self._gan = self._build_and_compile_gan(
                optimizer, self._generator_model, self._discriminator_model,
            )

        else:
            self._generator_model = self._build_generator_model()
            self._discriminator_model = self._build_and_compile_discriminator_model(
                optimizer,
            )
            self._gan = self._build_and_compile_gan(
                optimizer, self._generator_model, self._discriminator_model,
            )

    def _build_generator_model(self):
        model_input = Input(shape=(self._generator_input_dim,))
        d1, d2 = self._gen_base_dims
        settings = self._generator_settings
        conv_startsize = settings.get('conv2d-startsize', 128)
        assert isinstance(conv_startsize, int)
        assert (conv_startsize // 2) * 2 == conv_startsize, "conv2d-startsize must be dividable by 2"
        norm_momentum = float(settings.get('normalization-momentum', 0.8))
        assert 0 <= norm_momentum <= 1.0

        model_sequence = Sequential([
            Dense(conv_startsize * d1 * d2, activation='relu', input_dim=self._generator_input_dim),
            Reshape((d1, d2, conv_startsize)),
            UpSampling2D(),
            Conv2D(conv_startsize, kernel_size=3, padding='same'),
            BatchNormalization(momentum=norm_momentum),
            Activation('relu'),
            UpSampling2D(),
            Conv2D(conv_startsize // 2, kernel_size=3, padding='same'),
            BatchNormalization(momentum=norm_momentum),
            Activation('relu'),
            Conv2D(self._image_channels, kernel_size=3, padding='same'),
            Activation('tanh'),
        ])
        tensor = model_sequence(model_input)
        return Model(model_input, tensor)

    def _build_and_compile_discriminator_model(self, optimizer):
        model_input = Input(shape=self._image_shape)
        settings = self._discriminator_settings
        start_conv = int(settings.get('conv2d-startsize', 32))
        dropout = float(settings.get('dropout', 0.25))
        assert 0 <= dropout < 1.
        norm_momentum = float(settings.get('normalization-momentum', 0.8))
        assert 0 <= norm_momentum <= 1.0
        relu_alpha = float(settings.get('leakyrelu-alpha', 0.2))
        assert relu_alpha >= 0.
        model_sequence = Sequential([
            Conv2D(
                start_conv, kernel_size=3, strides=2, input_shape=self._image_shape, padding='same',
            ),
            LeakyReLU(alpha=relu_alpha),
            Dropout(dropout),
            Conv2D(start_conv * 2, kernel_size=3, strides=2, padding='same'),
            ZeroPadding2D(padding=((0, 1), (0, 1))),
            BatchNormalization(momentum=norm_momentum),
            LeakyReLU(alpha=relu_alpha),
            Dropout(dropout),
            Conv2D(start_conv * 4, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(momentum=norm_momentum),
            LeakyReLU(alpha=relu_alpha),
            Dropout(dropout),
            Conv2D(start_conv * 8, kernel_size=3, strides=2, padding='same'),
            BatchNormalization(momentum=norm_momentum),
            LeakyReLU(alpha=relu_alpha),
            Dropout(dropout),
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
        if not self._image_helper:
            print("No image helper")
            return
        generated = self._predict_noise(self._image_helper.eval_size)
        self._image_helper.write_data_as_image(generated, epoch=epoch)

    def _get_noise(self, batch_size):
        return np.random.normal(0, 1, (batch_size, self._generator_input_dim))

    def _predict_noise(self, size):
        noise = self._get_noise(size)
        return self._generator_model.predict(noise)

    def _report_progress(
            self, epoch, discriminator_loss, generator_loss, ref_epoch, ref_time, save_image_each,
    ):
        speed = (time() - ref_time) / (epoch + 1 - ref_epoch)
        next_save = save_image_each - (epoch % save_image_each)
        print("--------------------------------------")
        print(" Epoch: {} ({:.2f} s/epoch)".format(epoch, speed))
        print("     - Next Image {:.2f}s".format(next_save * speed))
        print(" Discriminator loss: {:.5f}".format(discriminator_loss[0]))
        print(" Generator loss:     {:.5f}".format(generator_loss))
        print("--------------------------------------")

    def train(
        self, epochs, train_data, batch_size, *, save_image_each=100, report_each=20,
        save_model_path=None, save_model_each=5000,
    ):
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        start = time()
        ref_time = start
        ref_epochs = self._epochs
        for epoch in range(self._epochs, self._epochs + epochs):
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
                    epoch, discriminator_loss, generator_loss, ref_epoch, ref_time, save_image_each,
                )
                self._epochs = epoch + 1
                ref_time = time()
                ref_epochs = epoch

            if epoch % save_image_each == 0:
                self._save_images(epoch)
                self._epochs = epoch + 1

            if epoch % save_model_each == 0 and epoch and save_model_path:
                self.save("{}.epoch{}".format(save_model_path, str(epoch).zfill(6)))
                self._epochs = epoch +1

        self._epochs = epochs
        if epoch % report_each != 0:
            self._report_progress(
                epoch, discriminator_loss, generator_loss, start, save_image_each,
            )

        if epoch % save_image_each != 0:
            self._save_images(epoch)

        if epoch % save_model_each != 0 and epoch and save_model_path:
            self.save("{}.epoch{}".format(save_model_path, str(epoch).zfill(6)))
        print("Training ran for {:.1f} hours".format((time() - start) / 3600.))

    def set_image_helper(self, image_helper):
        self._image_helper = image_helper

    def make_image_collage(self, save_dir, filename, rows=8, cols=8):
        image_helper = ImageHelper(save_dir, rows, cols, self._image_shape)
        generated = self._predict_noise(image_helper.eval_size)
        image_helper.write_data_as_image(generated, filename=filename)

    def save(self, path):
        self._generator_model.save("{}.gen.h5".format(path))
        self._discriminator_model.save("{}.dis.h5".format(path))
        self._gan.save("{}.gan.h5".format(path))
        with open("{}.pickle".format(path), 'wb') as fh:
            pickle.dump({
                'image_helper': self._image_helper.serialized() if self._image_helper else None,
                'image_shape': self._image_shape,
                'image_gen_dim': self._generator_input_dim,
                'epochs': self._epochs,
            }, fh)

    @staticmethod
    def load(path, image_helper=None):
        generator = load_model("{}.gen.h5".format(path))
        discriminator = load_model("{}.dis.h5".format(path))
        with open("{}.pickle".format(path), 'rb') as fh:
            data = pickle.load(fh)
        if image_helper is None:
            ih_data = data['image_helper']
            image_helper = ImageHelper(**ih_data)
        dcgan = DCGAN(
            data['image_shape'], image_helper, generator_input_dim=data['image_gen_dim'],
            models={'generator': generator, 'discriminator': discriminator},
        )
        dcgan._epochs = data.get('epochs', 0)
        return dcgan



def demo():
    def demo_data_loader():
        (X, _), _ = fashion_mnist.load_data()
        X = X / 127.5 - 1
        return np.expand_dims(X, axis=3)

    X = demo_data_loader()
    image_helper = ImageHelper(Path('SAMPLES/demo-fashion'), 8, 8, X[0].shape)
    gan = DCGAN(X[0].shape, image_helper)
    gan.train(20000, X, batch_size=32)
