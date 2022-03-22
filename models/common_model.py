import abc
from typing import Union
import tensorflow as tf
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt


class CommonModel:

    def __init__(self, image_size: int, optimizer: str = 'adam'):
        self.model = models.Sequential()
        self.image_size = image_size
        self.optimizer = optimizer
        pass

    def add_layers(self):
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_size, self.image_size, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    def compile(self, loss: Union[str, dict], metrics: Union[str, dict]):
        self.model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)

    @abc.abstractmethod
    def train(self, epochs: int):
        pass

    def evaluate(self):
        pass

    def save(self):
        pass
