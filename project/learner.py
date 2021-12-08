import os.path
from typing import List, Optional, Tuple

import tensorflow as tf
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel  # type: ignore
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel  # type: ignore
from tensorflow.keras import activations, optimizers
from tensorflow.keras.layers import Layer, Concatenate, Dense
from utils.config import ConfigShell
from environment import Environment

CONFIG_FILE = "config.csv"
PROJECT_HOME = os.path.dirname(__file__)


def draw_tensorboard(episode: int, time_step: int, logdir: str):
    writer = tf.summary.create_file_writer(logdir)
    with writer.as_default():
        tf.summary.scalar("Duration/Episode", time_step, step=episode)


class Learner:
    def __init__(
        self,
        agent,
        network,
        optimizer: Optional[optimizers.Optimizer] = optimizers.Adam(
            learning_rate=0.0001, clipnorm=10.0
        ),
        # Parmeters for initializing the network
        preprocessing_layers: Optional[List[Layer]] = [
            Dense(50, activation=activations.softplus)
        ],
        preprocessing_combiner: Optional[Layer] = Concatenate(axis=-1),
        conv_layer_params: Optional[List[Tuple[int]]] = [(8, 5, 2), (16, 3, 1)],
        fc_layer_params: Optional[List[int]] = None,
        activation_fn: Optional[Layer] = activations.softplus,
    ):
        self.agent = agent
        self.network = network
        self.optimizer = optimizer
        # Parmeters for initializing the network
        self.fc_layer_params = fc_layer_params
        self.preprocessing_layers = preprocessing_layers
        self.preprocessing_combiner = preprocessing_combiner
        self.conv_layer_params = conv_layer_params
        self.activation_fn = activation_fn

    def configure(self) -> Tuple[dict, str, str, str]:
        """
        Configure the learner.

        Returns:
            config: the configuration parameters.
            env_file: the environment file.
            logdir: the log directory.
            savedir: the save directory.
        """
        # [1] Load configuration
        configsh = ConfigShell(CONFIG_FILE)
        config, env_file, logdir, savedir = configsh.get_path(PROJECT_HOME)

        # [2] GPU configuration
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        return config, env_file, logdir, savedir

    def initialize_env(self, config, env_file) -> Environment:
        """
        Initialize the environment.

        Args:
            config: the configuration parameters.
            env_file: the environment file.
        Returns:
            env: Environment
        """
        # [3] Environment configuration
        base_port = int(input("Enter base port: "))
        time_scale = int(config.get("time_scale"))
        width = int(config.get("width"))
        height = int(config.get("height"))

        channel_config = EngineConfigurationChannel()
        channel_param = EnvironmentParametersChannel()

        env = Environment(
            file_name=env_file,
            base_port=base_port,
            side_channels=[channel_config, channel_param],
        )

        channel_config.set_configuration_parameters(
            time_scale=time_scale, quality_level=1, width=width, height=height
        )

        env.set_float_parameters(config)
        return env

    def initialize_agent(self, config, env):
        """
        Initialize the agent.

        Args:
            config: the configuration parameters.
            env: Environment

        Returns:
            agent: Agent
        """
        # [4] Agent configuration
        action_spec = env.action_spec
        observation_spec = env.observation_spec

        q_network = self.initialize_network(action_spec, observation_spec)

        optimizer = self.optimizer
        agent = self.agent(
            observation_spec,
            action_spec,
            q_network,
            optimizer,
            epsilon=config.get("epsilon", 1.0),
            target_update_period=config.get("target_update_period", 5000),
            gamma=config.get("gamma", 0.95),
            reward_shaping=True if config.get("reward_shaping", True) else False,
            batch_size=config.get("batch_size", 12),
            epsilon_start=config.get("epsilon_start", 0.8),
            epsilon_end=config.get("epsilon_end", 0.1),
            exploration_steps=config.get("exploration_steps", 209000),
        )
        return agent

    def initialize_network(self, action_spec, observation_spec) -> tf.keras.Model:
        """
        Initialize the network.

        Args:
            action_spec: ActionSpec
            observation_spec: ObservationSpec

        Returns:
            network: Network
        """
        preprocessing_layers = self.preprocessing_layers
        preprocessing_combiner = self.preprocessing_combiner
        conv_layer_params = self.conv_layer_params
        fc_layer_params = self.fc_layer_params
        activation_fn = self.activation_fn

        q_network = self.network(
            observation_spec,
            action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            activation_fn=activation_fn,
        )

        return q_network

    def run(self):
        config, env_file, logdir, savedir = self.configure()
        env = self.initialize_env(config, env_file)
        agent = self.initialize_agent(config, env)

        # [5] Training
        max_time_step = int(config.get("max_time_step"))
        train_start = int(config.get("train_start", 1000))
        n_episode = int(config.get("n_episode", 2000))
        global_step = 0
        survival_time_steps = [0] * n_episode

        print("Training...")
        for episode in range(n_episode):
            time_step = 0
            observations, done = env.reset()
            while not done and time_step < max_time_step:
                time_step += 1
                global_step += 1
                debug_log = f"Episode: {episode}/{n_episode} "
                debug_log += f"Global step: {global_step} "
                debug_log += f"Time step: {time_step} "
                debug_log += f"Epsilon: {agent._epsilon:.5f} "
                print(debug_log, end="")
                action = agent.step(observations)
                next_observations, done = env.step(action)
                reward = agent.get_reward(observations, action, next_observations, done)
                agent.append(observations, action, reward, next_observations, done)
                if len(agent.memory) > train_start:
                    agent.train()
                if global_step % agent._target_update_period == 0:
                    agent.update_target_model()
                observations = next_observations.copy()

            if done:
                print(
                    "Episode {} finished after {} time steps".format(episode, time_step)
                )
                survival_time_steps[episode] = time_step
                draw_tensorboard(episode, time_step, logdir)

            if episode % 10 == 0:
                agent.save(f"{savedir}/ckpt_{episode}")

        env.close()
        print("Training finished")


class LearnerPrev(Learner):
    def __init__(
        self,
        agent,
        network,
        optimizer: Optional[optimizers.Optimizer] = optimizers.Adam(
            learning_rate=0.0001, clipnorm=10
        ),
    ):
        self.agent = agent
        self.network = network
        self.optimizer = optimizer

    def initialize_network(self, action_spec, observation_spec):
        q_network = self.network(action_spec, (64, 64, 6), (2,))
        return q_network


if __name__ == "__main__":
    pass
