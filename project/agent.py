from collections import deque
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from q_network import QNetwork
from q_network_prev import QNetworkPrev
from utils.action import ActionType


# dqn agent with episodic greedy exploration
class DqnAgent:
    def __init__(
        self,
        observation_spec: List[Tuple[int]],
        action_spec: int,
        q_network: QNetwork,
        optimizer: tf.keras.optimizers.Optimizer,
        epsilon: float = 1.0,
        # Params for target network updates
        target_update_period: int = 5000,
        # Params for training
        gamma: float = 0.95,
        reward_shaping: bool = False,
        batch_size: int = 32,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        exploration_steps: int = 1000000,
    ):
        """Initialize DQN agent.

        Args:
            observation_spec: List of (int) tuples representing the shape of the observation.
            action_spec: Number of actions.
            q_network: QNetwork instance.
            optimizer: tf.keras.optimizers.Optimizer instance.
            epsilon: Initial epsilon value.
            target_update_period: Period for updating the target network.
            gamma: Discount factor.
            reward_shaping: Whether to use reward shaping.
            batch_size: Batch size for training.
            epsilon_start: Start value for epsilon.
            epsilon_end: End value for epsilon.
            exploration_steps: Number of steps for epsilon decay.
        """
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._q_network = q_network
        self._optimizer = optimizer
        self._epsilon = epsilon
        self._target_update_period = target_update_period
        self._gamma = gamma
        self._reward_shaping = reward_shaping
        self._step_counter = 0

        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.exploration_steps = exploration_steps
        self.epsilon_decay_step = self.epsilon_start - self.epsilon_end
        self.epsilon_decay_step /= self.exploration_steps

        self.memory = deque(maxlen=50000)
        self._target_q_network = self._q_network.copy(name="target_q_network")
        self.update_target_model()

    def step(self, observations: np.ndarray) -> int:
        """Choose an action based on the observation.

        Args:
            observations: List of (int) tuples representing the shape of the observation.

        Returns:
            action: (int)
        """
        if np.random.rand() < self._epsilon:
            print("Action: random", end=" ")
            return np.random.randint(self._action_spec)
        else:
            print("Action: greedy", end=" ")
            return np.argmax(self._q_network.call([observations]).numpy())

    def get_reward(
        self,
        observations: np.ndarray,
        action: int,
        next_observations: np.ndarray,
        done: bool,
    ) -> float:
        """Get the reward for the given action.

        Args:
            observations: List of (int) tuples representing the shape of the observation.
            action: (int)
            next_observations: List of (int) tuples representing the shape of the observation.
            done: (bool)

        Returns:
            reward: (float)
        """
        vec_obs = observations[1]
        next_vec_obs = next_observations[1]
        if self._reward_shaping:
            reward = 0.01 * np.power(vec_obs, 2).sum()
            reward -= 0.01 * np.power(next_vec_obs, 2).sum()
            reward *= self._gamma / (1.0 - self._gamma)

            if action not in (
                ActionType.NONE.value,
                ActionType.LEFT.value,
                ActionType.RIGHT.value,
            ):
                reward -= 0.001

            reward = np.clip(reward, -5, 5)
        else:
            reward = -1.0 if done else 0.0
        print("Reward:", reward)
        return reward

    def update_target_model(self):
        """Update the target network.

        The target network is updated every `target_update_period` steps.
        """
        self._target_q_network.set_weights(self._q_network.get_weights())

    def update_epsilon(self):
        """Update epsilon value."""
        if self._epsilon > self.epsilon_end:
            self._epsilon -= self.epsilon_decay_step
        else:
            self._epsilon = self.epsilon_end

    def append(
        self,
        observations: np.ndarray,
        actions: int,
        rewards: float,
        next_observations: np.ndarray,
        done: bool,
    ):
        """Append a transition to the memory.

        Args:
            observations: List of (int) tuples representing the shape of the observation.
            actions: (int)
            rewards: (float)
            next_observations: List of (int) tuples representing the shape of the observation.
            done: (bool)
        """
        if not done:
            self.memory.append(
                (observations, actions, rewards, next_observations, done)
            )  # (s, a, r, s', done)

    def train(self):
        """Train the agent.

        The agent trains the Q-network using the experience replay memory.
        """
        self.update_epsilon()
        if len(self.memory) < self.batch_size:
            return

        batch = np.array(self.memory)[
            np.random.choice(len(self.memory), self.batch_size, replace=False)
        ]
        observations, actions, rewards, next_observations, dones = zip(*batch)
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones)

        train_params = self._q_network.trainable_variables
        with tf.GradientTape() as tape:
            actions = tf.one_hot(actions, self._action_spec)
            q_values = self._q_network.call(observations)
            predicts = tf.reduce_sum(q_values * actions, axis=1)

            target_q_values = self._target_q_network.call(next_observations)
            max_q_values = tf.reduce_max(target_q_values, axis=1)
            # max_q_values *= 1 - dones
            targets = rewards + self._gamma * max_q_values

            error = tf.abs(targets - predicts)
            quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        gradients = tape.gradient(loss, train_params)
        self._optimizer.apply_gradients(zip(gradients, train_params))

        self._step_counter += 1

    def save(self, path):
        """Save the agent's weights.

        Args:
            path: (str)
        """
        self._q_network.save_weights(path, save_format="tf")

    def load(self, path):
        """Load the agent's weights.

        Args:
            path: (str)
        """
        self._q_network.load_weights(path)


class DoubleDqnAgent(DqnAgent):
    def __init__(
        self,
        observation_spec: List[Tuple[int]],
        action_spec: int,
        q_network: QNetwork,
        optimizer: tf.keras.optimizers.Optimizer,
        epsilon: float = 1.0,
        # Params for target network updates
        target_update_period: int = 5000,
        # Params for training
        gamma: float = 0.95,
        reward_shaping: bool = False,
        batch_size: int = 32,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        exploration_steps: int = 1000000,
    ):
        super().__init__(
            observation_spec,
            action_spec,
            q_network,
            optimizer,
            epsilon,
            target_update_period,
            gamma,
            reward_shaping,
            batch_size,
            epsilon_start,
            epsilon_end,
            exploration_steps,
        )

    def train(self):
        self.update_epsilon()
        if len(self.memory) < self.batch_size:
            return

        batch = np.array(self.memory)[
            np.random.choice(len(self.memory), self.batch_size, replace=False)
        ]
        observations, actions, rewards, next_observations, dones = zip(*batch)
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones)

        train_params = self._q_network.trainable_variables
        with tf.GradientTape() as tape:
            actions = tf.one_hot(actions, self._action_spec)
            q_values = self._q_network.call(observations)
            predicts = tf.reduce_sum(q_values * actions, axis=1)

            target_q_values = self._target_q_network.call(next_observations)
            _q_values = self._q_network.call(next_observations)

            idx = tf.argmax(_q_values, axis=1)
            max_q_values = tf.gather(target_q_values, idx, batch_dims=1)
            # max_q_values *= 1 - dones
            targets = rewards + self._gamma * max_q_values

            error = tf.abs(targets - predicts)
            quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        gradients = tape.gradient(loss, train_params)
        self._optimizer.apply_gradients(zip(gradients, train_params))

        self._step_counter += 1


class DqnAgentPrev(DqnAgent):
    def __init__(
        self,
        observation_spec: List[Tuple[int]],
        action_spec: int,
        q_network: QNetworkPrev,
        optimizer: tf.keras.optimizers.Optimizer,
        epsilon: float = 1.0,
        # Params for target network updates
        target_update_period: int = 5000,
        # Params for training
        gamma: float = 0.95,
        reward_shaping: bool = False,
        batch_size: int = 32,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        exploration_steps: int = 1000000,
    ):
        super().__init__(
            observation_spec,
            action_spec,
            q_network,
            optimizer,
            epsilon,
            target_update_period,
            gamma,
            reward_shaping,
            batch_size,
            epsilon_start,
            epsilon_end,
            exploration_steps,
        )

    def step(self, observations: np.ndarray) -> int:
        if np.random.rand() < self._epsilon:
            print("Action: random", end=" ")
            return np.random.randint(self._action_spec)
        else:
            print("Action: greedy", end=" ")
            vis_obs, vec_obs = observations
            vis_obs = np.array([vis_obs])
            vec_obs = np.array([vec_obs])
            q_value = self._q_network(vis_obs, vec_obs)
            return np.argmax(q_value.numpy())

    def train(self):
        self.update_epsilon()
        if len(self.memory) < self.batch_size:
            return

        batch = np.array(self.memory)[
            np.random.choice(len(self.memory), self.batch_size, replace=False)
        ]

        observations, actions, rewards, next_observations, dones = zip(*batch)
        vis_obs, vec_obs = zip(*observations)
        next_vis_obs, next_vec_obs = zip(*next_observations)
        vis_obs = np.array(vis_obs)
        vec_obs = np.array(vec_obs)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_vis_obs = np.array(next_vis_obs)
        next_vec_obs = np.array(next_vec_obs)
        dones = np.array(dones)

        train_params = self._q_network.trainable_variables
        with tf.GradientTape() as tape:
            actions = tf.one_hot(actions, self._action_spec)
            q_values = self._q_network(vis_obs, vec_obs)
            predicts = tf.reduce_sum(q_values * actions, axis=1)

            target_q_values = self._target_q_network(next_vis_obs, next_vec_obs)
            max_q_values = tf.reduce_max(target_q_values, axis=1)
            # max_q_values *= 1 - dones
            targets = rewards + self._gamma * max_q_values

            error = tf.abs(targets - predicts)
            quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        gradients = tape.gradient(loss, train_params)
        self._optimizer.apply_gradients(zip(gradients, train_params))

        self._step_counter += 1


if __name__ == "__main__":
    pass
