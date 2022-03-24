import abc
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
import tensorflow as tf
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns


class CommonModel:

    def __init__(self, image_size: int, optimizer: str = 'adam'):
        self.model = models.Sequential()
        self.image_size = image_size
        self.optimizer = optimizer
        self.save_location = str()

    def add_layers(self):
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                     input_shape=(
                                         self.image_size, self.image_size, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))

    def compile(self, loss: Union[str, dict], metrics: Union[str, dict]):
        self.model.compile(optimizer=self.optimizer, loss=loss,
                           metrics=metrics)

    def train(self, train_images: ArrayLike,
              train_labels: Union[ArrayLike, dict], test_images: ArrayLike,
              test_labels: Union[ArrayLike, dict], epochs: int):
        self.model.fit(train_images, train_labels, epochs=epochs,
                       validation_data=(test_images, test_labels))

    def evaluate(self, test_images: ArrayLike, test_labels: Union[ArrayLike, dict]):
        loss, acc = self.model.evaluate(test_images, test_labels, verbose=2)
        return loss, acc

    def save(self):
        self.model.save(self.save_location)

    def load(self):
        self.model = models.load_model(self.save_location)

    def confusion_matrix(self, test_images: ArrayLike, test_labels: Union[ArrayLike, dict]):
        y_pred = self.model.predict_classes(test_images)
        conf_matr = tf.math.confusion_matrix(labels=test_labels,
                                             predictions=y_pred)
        figure = plt.figure(figsize=conf_matr.shape)
        sns.heatmap(conf_matr, annot=True, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    @abc.abstractmethod
    def add_outputs(self):
        pass
