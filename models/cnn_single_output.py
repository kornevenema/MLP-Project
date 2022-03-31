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


class SingleOutputCNN(CommonModel):

    def __init__(self, image_size: int, optimizer='adam'):
        super().__init__(image_size, optimizer)
        self.save_location = 'models/saved/single'

    def add_outputs(self):
        output_layer = layers.Dense(12, activation='softmax')(self.bottleneck)
        self.model = keras.Model(inputs=self.input_layer, outputs=output_layer)
