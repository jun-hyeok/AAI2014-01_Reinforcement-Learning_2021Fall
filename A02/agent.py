from collections import defaultdict
import numpy as np


class Agent:
    def __init__(self, Q, mode="test_mode"):
        self.Q = Q
        self.mode = mode
        self.n_actions = 6
        self.i_episode = 1
        self.epsilon = 1.0  # start_epsilon
        if mode == "mc_control":
            self.learning_rate = 0.001
            self.gamma = 0.8
            self.episode = list()
        if mode == "q_learning":
            self.learning_rate = 0.2
            self.gamma = 0.8
        if mode == "test_mode":
            self.epsilon = 0.0  # greedy policy : no random action

    def select_action(self, state):
        """
        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # random action
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        # read action from Q table
        else:
            return np.argmax(self.Q[state])

    def step(self, state, action, reward, next_state, done):

        """
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if done:
            # epsilon decay scheduling
            self.i_episode += 1
            self.epsilon = 1.0 / (self.i_episode // 100 + 1)
        if self.mode == "mc_control":
            # learn from memory
            if done:
                returns = defaultdict(lambda: np.zeros(self.n_actions))
                for history in reversed(self.episode):
                    state, action, reward = history
                    returns[state][action] = (
                        reward + self.gamma * returns[state][action]
                    )
                    self.Q[state][action] += self.learning_rate * (
                        returns[state][action] - self.Q[state][action]
                    )
                self.episode.clear()
            # memory
            else:
                self.episode.append([state, action, reward])
        elif self.mode == "q_learning":
            self.Q[state][action] += self.learning_rate * (
                reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
            )
