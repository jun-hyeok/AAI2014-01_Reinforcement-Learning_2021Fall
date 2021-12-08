import sys

sys.path = [".."] + sys.path
from agent import DqnAgent
from learner import Learner
from q_network import QNetwork
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense

if __name__ == "__main__":
    learner = Learner(
        DqnAgent,
        QNetwork,
        preprocessing_layers=[
            Dense(50, activation=activations.softplus, name="vector"),
            Dense(100, activation=activations.softplus, name="olfactory"),
        ],
        fc_layer_params=[400],
    )
    learner.run()
