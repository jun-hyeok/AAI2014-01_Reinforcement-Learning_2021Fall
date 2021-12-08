from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, concatenate  # type: ignore

from q_network import _capture_init


class QNetworkPrev(tf.keras.Model):
    @_capture_init
    def __init__(
        self, action_size: int, image_size: Tuple[int], vector_size: Tuple[int]
    ):
        """Initialize the Q-Network
        Args:
            action_size: Number of possible actions
            image_size: Size of the input image
            vector_size: Size of the input vector
        """
        super(QNetworkPrev, self).__init__()
        self.conv1 = Conv2D(
            filters=8,
            kernel_size=5,
            strides=2,
            padding="same",
            activation="softplus",
            name="conv1",
            input_shape=image_size,
        )
        self.bn1 = BatchNormalization(name="bn1")
        self.conv2 = Conv2D(
            filters=16,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="softplus",
            name="conv2",
        )
        self.bn2 = BatchNormalization(name="bn2")
        self.flatten = Flatten(name="flatten")
        self.fc_vec = Dense(
            50, activation="softplus", name="fc_vec", input_shape=vector_size
        )
        self.fc = Dense(400, activation="softplus", name="fc")
        self.fc_out = Dense(action_size, name="fc_out")

    def call(self, x: np.ndarray, y: np.ndarray) -> tf.Tensor:
        """Forward pass of the network

        Args:
            x: Input image
            y: Input vector

        Returns:
            tf.Tensor: Q-values
        """
        self.x1 = self.conv1(x)
        self.x2 = self.bn1(self.x1)
        self.x3 = self.conv2(self.x2)
        self.x4 = self.bn2(self.x3)
        self.x5 = self.flatten(self.x4)
        self.y1 = self.fc_vec(y)
        self.y2 = self.flatten(self.y1)
        self.z = self.fc(concatenate([self.x5, self.y2], name="concat", axis=1))
        self.q = self.fc_out(self.z)
        return self.q

    def copy(self, name: str = None):
        """Creates a copy of the network.

        Args:
            name: The name of the new network.

        Returns:
            A copy of the network.
        """
        return type(self)(*self._init_args, **self._init_kwargs)


if __name__ == "__main__":
    pass
