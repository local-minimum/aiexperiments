from collections import namedtuple

import cv2

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist

import numpy as np


Data = namedtuple("Data",  ['x', 'y'])
Training = namedtuple("Training", ['train', 'test', 'cols', 'rows', 'nclasses'])


def load_training_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    rows, cols = x_train[0].shape
    x_train = x_train[..., np.newaxis].astype('float32') / 255
    x_test = x_test[..., np.newaxis].astype('float32') / 255
    nclasses = len(set(y_train))
    y_train = to_categorical(y_train, nclasses)
    y_test = to_categorical(y_test, nclasses)
    return Training(
        Data(x_train, y_train),
        Data(x_test, y_test),
        cols,
        rows,
        nclasses)


def load_image_to_gray(path):
    im = cv2.imread(path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return 1 - gray.astype('float32') / 255


def get_model(rows, cols, number_of_classes, nfeatures=32, convlayers=3, pooling=True): 
    model = Sequential() 
    for i in range(convlayers): 
        model.add( 
            Conv2D(nfeatures * 2 ** i, 
            kernel_size=(3, 3), 
            activation='linear', 
            input_shape=(rows, cols, 1)), 
        ) 
        model.add(LeakyReLU(alpha=0.1)) 
    model.add(Dropout(0.5)) 
    if pooling: 
        model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Flatten()) 
    model.add(Dense(32 * 2 ** convlayers, activation='relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(number_of_classes, activation='softmax')) 
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    return model 


def train(batch_size=128, epochs=20, model=None, data=None):
    if data is None:
        data = load_training_data()
    if model is None:
        model = get_model(data.rows, data.cols, data.nclasses)
    model.fit(
        data.train.x, data.train.y, batch_size=batch_size, epochs=epochs, verbose=1,
        validation_split=0.2,
    )
    print('Loss {}, Accuracy {}'.format(
       *model.evaluate(data.test.x, data.test.y, verbose=0),
    ))
    return model, data


def predict_single(model, path):
    im = load_image_to_gray(path)
    im = im[np.newaxis, ..., np.newaxis]
    pred = model.predict(im)
    f = plt.figure()
    ax = f.gca()
    ax.bar(np.arange(pred.size), pred.ravel())
    ax.set_ylim(0, 1)
    return np.argmax(pred), pred.ravel(), ax
