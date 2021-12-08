import sys

sys.path = [".."] + sys.path
from agent import DqnAgentPrev
from learner import LearnerPrev
from q_network_prev import QNetworkPrev

if __name__ == "__main__":
    learner = LearnerPrev(DqnAgentPrev, QNetworkPrev)
    learner.run()
