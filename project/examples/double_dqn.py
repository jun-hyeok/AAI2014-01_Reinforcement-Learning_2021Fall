import sys

sys.path = [".."] + sys.path
from agent import DoubleDqnAgent
from learner import Learner
from q_network import QNetwork
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense

if __name__ == "__main__":

    learner = Learner(
        DoubleDqnAgent,
        QNetwork,
        preprocessing_layers=[
            Dense(50, activation=activations.softplus, name="vector"),
        ],
        fc_layer_params=[400],
    )
    learner.run()
