import sys

sys.path = [".."] + sys.path
from agent import DqnAgent
from learner import Learner
from q_network import DuelingQNetwork
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense

if __name__ == "__main__":
    learner = Learner(
        DqnAgent,
        DuelingQNetwork,
        preprocessing_layers=[
            Dense(50, activation=activations.softplus, name="vector"),
        ],
        fc_layer_params=[400, 512],
    )
    learner.run()
