from functools import wraps
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, initializers
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, Layer  # type: ignore
from tensorflow.python.training.tracking import base


def _capture_init(func):
    """
    Decorator to capture the initialization of a layer.

    Args:
        func: The function to decorate.

    Returns:
        The decorated function.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self._init_args = args
        with base.no_automatic_dependency_tracking_scope(self):
            setattr(self, "_init_kwargs", kwargs)
        return func(self, *args, **kwargs)

    return wrapper


def forward(layers: List[Layer], x: tf.Tensor) -> tf.Tensor:
    """
    Forward pass through a list of layers.

    Args:
        layers: The list of layers to apply.
        x: The input tensor.

    Returns:
        The output tensor.
    """

    for layer in layers:
        x = layer(x)
    return x


class QNetwork(tf.keras.Model):
    @_capture_init
    def __init__(
        self,
        observation_spec: List[Tuple[int]],
        action_spec: int,
        preprocessing_layers: List[Layer] = None,
        preprocessing_combiner: Optional[Layer] = None,
        conv_layer_params: List[Tuple[int]] = None,  # visual_processing_layers
        fc_layer_params: List[int] = None,  # postprocessing_layers
        activation_fn: Optional[Layer] = activations.relu,
        kernel_initializer: Optional[initializers.Initializer] = None,
        dtype: tf.dtypes.DType = tf.float32,
        name: str = "q_network",
    ):
        """Initializes the QNetwork.

        Args:
            observation_spec: A list of observation dimensions.
            action_spec: An integer representing the number of actions.
            preprocessing_layers: A list of layers to apply to the input.
            preprocessing_combiner: A layer to combine the output of the
                preprocessing layers.
            conv_layer_params: A list of lists of parameters for the convolution
                layers.
            fc_layer_params: A list of parameters for the fully connected layers.
            activation_fn: The activation function to use.
            kernel_initializer: The kernel initializer to use.
            dtype: The dtype of the network.
            name: The name of the network.
        """
        # validate the inputs
        super().__init__(name=name)
        # set the properties
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._preprocessing_layers = preprocessing_layers
        self._preprocessing_combiner = preprocessing_combiner

        # add the visual processing layers
        layers = []
        if conv_layer_params:
            for i, config in enumerate(conv_layer_params):
                if len(config) == 3:
                    (filters, kernel_size, strides) = config
                else:
                    raise ValueError(
                        "only 3 or 4 elements permitted in conv_layer_params tuples"
                    )
                layers.append(
                    Conv2D(
                        filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding="same",
                        activation=activation_fn,
                        kernel_initializer=kernel_initializer,
                        name=f"conv_layer_{i:02}",
                        dtype=dtype,
                    )
                )
                layers.append(BatchNormalization(name=f"bn_layer_{i:02}"))
        self._visual_processing_layers = layers

        # add the vector processing layers
        # layers = []
        # if preprocessing_layers:
        #     for layer in preprocessing_layers:
        #         layers.append(layer)

        # add the post processing layers
        layers = []
        if fc_layer_params:
            for i, num_units in enumerate(fc_layer_params):
                layers.append(
                    Dense(
                        units=num_units,
                        activation=activation_fn,
                        kernel_initializer=kernel_initializer,
                        name=f"fc_layer_{i:02}",
                        dtype=dtype,
                    )
                )
        self._postprocessing_layers = layers

        # add the q value layer
        q_value_layer = Dense(action_spec, dtype=dtype, name="q_value")
        self.q_value_layer = q_value_layer

    def call(self, observation: np.ndarray, training: bool = True) -> tf.Tensor:
        """Returns the Q-values for the given observation.

        Args:
            observation: A tensor representing the observation.
            training: Whether the network is in training mode.

        Returns:
            A tensor of q-values.
        """
        vis_obs, vec_obs = zip(*observation)

        processed = []
        vis_obs = np.array(vis_obs)
        vis_obs = forward(self._preprocessing_layers, vis_obs)
        processed.append(Flatten()(vis_obs))

        vec_obs = np.split(vec_obs, (2, -1), axis=1)
        # TODO vec_obs size, self._preprocessing_layers size match validation
        for obs, layer in zip(vec_obs, self._preprocessing_layers):
            obs = layer(obs)
            processed.append(Flatten()(obs))

        states = processed
        if self._preprocessing_combiner is not None:
            states = self._preprocessing_combiner(states)

        states = forward(self._postprocessing_layers, states)
        q_value = self.q_value_layer(states)
        return q_value

    def copy(self, name: str = None):
        """Creates a copy of the network.

        Args:
            name: The name of the new network.

        Returns:
            A copy of the network.
        """
        self._init_kwargs.pop("name", None)
        return type(self)(*self._init_args, **self._init_kwargs, name=name)


class DuelingQNetwork(QNetwork):
    def __init__(
        self,
        observation_spec: List[Tuple[int]],
        action_spec: int,
        preprocessing_layers: List[Layer] = None,
        preprocessing_combiner: Optional[Layer] = None,
        conv_layer_params: List[Tuple[int]] = None,  # visual_processing_layers
        fc_layer_params: List[int] = None,  # postprocessing_layers
        activation_fn: Optional[Layer] = activations.relu,
        kernel_initializer: Optional[initializers.Initializer] = None,
        dtype: tf.dtypes.DType = tf.float32,
        name: str = "dueling_q_network",
    ):
        # validate the inputs
        super().__init__(
            observation_spec,
            action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params[:-1],
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            dtype=dtype,
            name=name,
        )

        # add the dueling layers
        self.advantage_layer = [
            Dense(
                fc_layer_params[-1],
                activation=activations.relu,
                name="advantage_input_layer",
                dtype=dtype,
            ),
            Dense(
                action_spec,
                name="advantage_output_layer",
                dtype=dtype,
            ),
        ]
        self.value_layer = [
            Dense(
                fc_layer_params[-1],
                activation=activations.relu,
                name="value_input_layer",
                dtype=dtype,
            ),
            Dense(
                1,
                name="value_output_layer",
                dtype=dtype,
            ),
        ]

    def call(self, observation: np.ndarray, training: bool = True):
        vis_obs, vec_obs = zip(*observation)

        processed = []
        vis_obs = np.array(vis_obs)
        vis_obs = forward(self._visual_processing_layers, vis_obs)
        processed.append(Flatten()(vis_obs))

        vec_obs = np.split(vec_obs, (2, -1), axis=1)
        # TODO vec_obs size, self._preprocessing_layers size match validation
        for obs, layer in zip(vec_obs, self._preprocessing_layers):
            obs = layer(obs)
            processed.append(Flatten()(obs))

        states = processed
        if self._preprocessing_combiner is not None:
            states = self._preprocessing_combiner(states)

        states = forward(self._postprocessing_layers, states)
        advantage = forward(self.advantage_layer, states)
        value = forward(self.value_layer, states)
        q_value = value + advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)
        return q_value


if __name__ == "__main__":
    pass
