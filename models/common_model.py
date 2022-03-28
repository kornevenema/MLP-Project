import abc
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
import tensorflow as tf
# from tensorflow.python.keras import layers, models
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


class CommonModel(abc.ABC):

    def __init__(self, image_size: int, optimizer: str = 'adam'):
        self.bottleneck = None
        self.model: Union[keras.Model, None] = None
        self.image_size = image_size
        self.optimizer = optimizer
        self.save_location: str
        self.input_layer = layers.Input(shape=(self.image_size, self.image_size, 3))

    def add_layers(self):
        _ = layers.Conv2D(32, (3, 3), activation='relu')(self.input_layer)
        _ = layers.MaxPooling2D((2, 2))(_)
        _ = layers.Conv2D(64, (3, 3), activation='relu')(_)
        _ = layers.MaxPooling2D((2, 2))(_)
        _ = layers.Conv2D(64, (3, 3), activation='relu')(_)
        _ = layers.Flatten()(_)
        _ = layers.Dense(64, activation='relu')(_)
        self.bottleneck = layers.Dense(32, activation='relu')(_)

    def compile(self, loss: Union[any, dict], metrics: Union[str, dict]):
        self.model.compile(optimizer=self.optimizer, loss=loss,
                           metrics=metrics)

    def train(self, train_images: ArrayLike,
              train_labels: Union[ArrayLike, dict], test_images: ArrayLike,
              test_labels: Union[ArrayLike, dict], epochs: int):
        self.model.fit(train_images, train_labels, epochs=epochs,
                       validation_data=(test_images, test_labels))

    def evaluate(self, test_images: ArrayLike, test_labels: Union[ArrayLike, dict]):
        return self.model.evaluate(test_images, test_labels, verbose=2)

    def save(self):
        self.model.save(self.save_location)

    def load(self):
        self.model = models.load_model(self.save_location)

    @abc.abstractmethod
    def add_outputs(self):
        pass
