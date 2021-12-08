from collections import deque
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from mlagents_envs.base_env import DecisionSteps, TerminalSteps
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel  # type: ignore
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel  # type: ignore
from mlagents_envs.side_channel.side_channel import SideChannel


class Environment(UnityEnvironment):
    def __init__(
        self,
        file_name: Optional[str] = None,
        worker_id: int = 0,
        base_port: Optional[int] = None,
        seed: int = 0,
        no_graphics: bool = False,
        timeout_wait: int = 60,
        additional_args: Optional[List[str]] = None,
        side_channels: Optional[List[SideChannel]] = None,
        log_folder: Optional[str] = None,
    ):
        """
        Starts a new unity environment and establishes a connection with the environment.
        Notice: Currently communication between Unity and Python takes place over an open socket without authentication.
        Ensure that the network where training takes place is secure.

        Args:
            file_name: The file path to the environment.
            worker_id: The index of the worker in the environment.
            base_port: The base port to use for running the environment.
            seed: The seed to use for running the environment.
            no_graphics: Whether to run the environment in no-graphics mode.
            timeout_wait: Timeout for connecting to the environment.
            additional_args: Additional command line arguments to pass to the Unity executable.
            side_channels: List of side channel to register.
            log_folder: The folder to save the Unity executable to.

        Raises:
            UnityEnvironmentException: If file_name does not correspond to a valid environment.
        """
        if side_channels is None:
            side_channels = [
                EnvironmentParametersChannel(),
                EngineConfigurationChannel(),
            ]

        for sc in side_channels:
            if isinstance(sc, EnvironmentParametersChannel):
                self._env_parameters_channel = sc
            elif isinstance(sc, EngineConfigurationChannel):
                self._engine_configuration_channel = sc

        super().__init__(
            file_name,
            worker_id,
            base_port,
            seed,
            no_graphics,
            timeout_wait,
            additional_args,
            side_channels,
            log_folder,
        )
        if not self.behavior_specs:
            super().reset()
        self.name = list(self.behavior_specs.keys())[0]
        self.group_spec = self.behavior_specs[self.name]

        super().reset()
        decision_steps, _ = self.get_steps(self.name)
        self._previous_decision_step = decision_steps

        # Set action spaces
        if self.group_spec.action_spec.is_discrete():
            self.action_size = self.group_spec.action_spec.discrete_size
            branches = self.group_spec.action_spec.discrete_branches
            self._action_spec = branches[0]
        elif self.group_spec.action_spec.is_continuous():
            self.action_size = self.group_spec.action_spec.continuous_size
        else:
            raise UnityEnvironmentException(
                "The environment does not contain a valid action space."
            )

        # Set observations spaces
        observation_spec = []
        shapes = self._get_vis_obs_shape()
        observation_spec.append(shapes[0])  # (64, 64, 3)
        if self._get_vec_obs_size() > 0:
            observation_spec.append((self._get_vec_obs_size(),))  # (2,)
        self._observation_spec = observation_spec

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        In the case of multi-agent environments, this is a list.
        Otherwise, this should be a numpy array.

        Returns:
            observation (object/list): the initial observation of the space.
        """
        super().reset()
        decision_steps, _ = self.get_steps(self.name)
        self.game_over = False
        self._visual_obs_queue = deque(maxlen=2)
        visual_obs = self._get_vis_obs_list(decision_steps)
        for obs in visual_obs:
            self._visual_obs_queue.append(self._preprocess_single(obs[0]))
        return self._current_time_step(decision_steps)

    def step(self, action: int):
        """
        Provides the environment with an action, moves the environment dynamics forward accordingly, and returns
        observation, state, and reward information to the agent.

        Args:
            action: Agent's action to take.

        Raises:
            UnityActionException: If the provided action is not a valid action.
            UnityTimeOutException: If the environment took too long to respond.
            UnityCommunicationException: If the Unity environment had an error.
        """
        action = np.array(action).reshape((1, self.action_size))
        self.set_actions(self.name, action)
        super().step()
        decision_steps, terminal_steps = self.get_steps(self.name)
        if len(terminal_steps) > 0:
            self.game_over = True
            return self._current_time_step(terminal_steps)
        else:
            return self._current_time_step(decision_steps)

    def _current_time_step(
        self, info: Union[DecisionSteps, TerminalSteps]
    ) -> Tuple[List[np.ndarray], bool]:
        visual_obs = self._get_vis_obs_list(info)
        for obs in visual_obs:
            self._visual_obs_queue.append(self._preprocess_single(obs[0]))
        default_observation = [np.concatenate(self._visual_obs_queue, axis=-1)]
        if self._get_vec_obs_size() >= 1:
            default_observation.append(self._get_vector_obs(info)[0, :])

        # if self._get_n_vis_obs() >= 1:
        #     visual_obs = self._get_vis_obs_list(info)
        #     self.visual_obs = self._preprocess_single(visual_obs[0][0])

        done = isinstance(info, TerminalSteps)
        return default_observation, done

    def _preprocess_single(self, single_visual_obs: np.ndarray) -> np.ndarray:
        return (single_visual_obs / 255.0).astype(np.float32)

    def _get_n_vis_obs(self) -> int:
        result = 0
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 3:
                result += 1
        return result

    def _get_vis_obs_shape(self) -> List[Tuple]:
        result: List[Tuple] = []
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 3:
                result.append(shape)
        return result

    def _get_vis_obs_list(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vector_obs(
        self, step_result: Union[DecisionSteps, TerminalSteps]
    ) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)

    def _get_vec_obs_size(self) -> int:
        result = 0
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 1:
                result += shape[0]
        return result

    @property
    def observation_spec(self):
        return self._observation_spec

    @property
    def action_spec(self):
        return self._action_spec

    # EnvironmentParametersChannel
    def set_float_parameters(self, parameters: dict):
        """Set the environment's parameters.

        Args:
            parameters: The parameters to set.
        """
        for key, value in parameters.items():
            if isinstance(value, float) or isinstance(value, int):
                self._env_parameters_channel.set_float_parameter(key, float(value))
            else:
                print(
                    f"EnvironmentParametersChannel: value {value} for key {key} is not a float"
                )

    def set_uniform_sampler_parameters(self, parameters: dict):
        """Set the environment's parameters.

        Args:
            parameters: The parameters to set.
        """
        for key, value in parameters.items():
            if isinstance(value, list):
                self._env_parameters_channel.set_uniform_sampler_parameters(
                    key, value[0], value[1], 0
                )
            else:
                print(
                    f"EnvironmentParametersChannel: value {value} for key {key} is not a list"
                )

    def set_gaussian_sampler_parameters(self, parameters: dict):
        """Set the environment's parameters.

        Args:
            parameters: The parameters to set.
        """
        for key, value in parameters.items():
            if isinstance(value, list):
                self._env_parameters_channel.set_gaussian_sampler_parameters(
                    key, value[0], value[1], 0
                )
            else:
                print(
                    f"EnvironmentParametersChannel: value {value} for key {key} is not a list"
                )

    def set_multi_range_uniform_sampler_parameters(self, parameters: dict):
        """Set the environment's parameters.

        Args:
            parameters: The parameters to set.
        """
        for key, value in parameters.items():
            if isinstance(value, list):
                self._env_parameters_channel.set_multi_range_uniform_sampler_parameters(
                    key, value, 0
                )
            else:
                print(
                    f"EnvironmentParametersChannel: value {value} for key {key} is not a list"
                )

    # EngineConfigurationChannel
    def set_configuration_parameters(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        quality_level: Optional[int] = None,
        time_scale: Optional[float] = None,
        target_frame_rate: Optional[int] = None,
        capture_frame_rate: Optional[int] = None,
    ):
        """Set the environment's configuration parameters.

        Args:
            width: The width of the environment.
            height: The height of the environment.
            quality_level: The quality level of the environment.
            time_scale: The time scale of the environment.
            target_frame_rate: The target frame rate of the environment.
            capture_frame_rate: The capture frame rate of the environment.
        """
        self._engine_configuration_channel.set_configuration_parameters(
            width,
            height,
            quality_level,
            time_scale,
            target_frame_rate,
            capture_frame_rate,
        )

    def set_configuration(self, config: dict):
        """Set the environment's configuration parameters.

        Args:
            config: The configuration parameters to set.
        """
        config = {
            key: value for key, value in config.items() if key in EngineConfig._fields
        }
        self._engine_configuration_channel.set_configuration_parameters(**config)


if __name__ == "__main__":
    pass
