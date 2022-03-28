import abc
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
import tensorflow as tf
# from tensorflow.python.keras import layers, models
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from models.common_model import CommonModel


class MultiOutputCNN(CommonModel):

    def __init__(self, image_size: int, optimizer='adam'):
        super().__init__(image_size, optimizer)

    def add_outputs(self):
        num_out = layers.Dense(6, activation='softmax', name='num')(self.bottleneck)
        hand_out = layers.Dense(2, activation='softmax', name='hand')(self.bottleneck)
        self.model = keras.Model(inputs=self.input_layer, outputs=[num_out, hand_out])
