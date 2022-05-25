"""Maze definition for reinforcement learning experiments."""
import itertools
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import QuadMesh


class MazeStates:
    """
    Maze for reinforcement learning.

    Contains states, rewards for states and moving across the maze.
    """

    def __init__(self,
                 states: npt.NDArray[np.int_],
                 rewards: npt.NDArray[np.float_],
                 out_reward: float = -10,
                 terminate_id: int = 0,
                 start_id: int = 0,
                 current_id: int = -1):
        """
        Parameters
        ----------
        states : npt.NDArray[np.int_]
            Maze positions list.
        rewards : npt.NDArray[np.float_]
            Rewards array for every position.
        out_reward : float, optional
            Reward when move out from states, by default -10.
        terminate_id : int, optional
            State id to terminate, by default 0.
        start_id : int, optional
            Start id position, by default 0.
        current_id : int, optional
            Current id position, by default -1.
        """

        if states.shape[1] != 2:
            raise ValueError(
                f"States shape should be (..., 2) by has {states.shape}")

        self.rewards = np.array(rewards)
        self.states = np.array(states)
        self._terminate_id: int = terminate_id
        self.out_reward: float = out_reward
        self._start_id = start_id

        if current_id == -1:
            current_id = self._start_id

        self.current_id: int = current_id

    @property
    def terminate_id(self) -> int:
        """Get terminate id."""
        return self._terminate_id

    def reset(self):
        """Move current id to the start id."""
        self.current_id = self._start_id

    @property
    def ids(self) -> npt.NDArray[np.int_]:
        """Get id list."""
        return np.arange(self.states.shape[0])

    def current_state(self) -> npt.NDArray[np.int_]:
        """Get current state position."""
        return self.get_pos(self.current_id)

    def get_pos(self, id_: int) -> npt.NDArray[np.int_]:
        """Get state position by id."""
        return self.states[id_, :]

    def get_reward(self, id_: int) -> float:
        """Get reward by id."""
        return self.rewards[id_]

    def get_id(self, state: Union[npt.NDArray[np.int_], Sequence]) -> int:
        """Get id by state."""
        if isinstance(state, Sequence):
            state = np.array(state)
        return np.where(np.all(self.states == state, axis=1))[0][0]

    def get_state_reward(self, state: Union[npt.NDArray[np.int_], Sequence]) \
            -> float:
        """Get reward by state."""
        id_ = self.get_id(state)
        return self.rewards[id_]

    @classmethod
    def from_rewards(cls,
                     rewards: npt.NDArray[np.float_],
                     start_state: Union[npt.NDArray[np.int_], Sequence],
                     terminate_state: Union[npt.NDArray[np.int_], Sequence],
                     **kwargs) -> 'MazeStates':
        """
        Instantiate maze states from array of rewards.

        Parameters
        ----------
        rewards : npt.NDArray[np.float_]
            2D field of rewards.
        start_state : Union[npt.NDArray[np.int_], Sequence]
            Start state position.
        terminate_state : Union[npt.NDArray[np.int_], Sequence]
            Terminate state position.

        Returns
        -------
        MazeStates
            New instance of maze.
        """
        states = np.array(list(itertools.product(range(rewards.shape[0]),
                                                 range(rewards.shape[1]))))

        terminate_id = np.where(np.all(states == terminate_state,
                                       axis=1))[0][0]
        start_id = np.where(np.all(states == start_state, axis=1))[0][0]

        reshape_rewards = np.empty(shape=states.shape[0])
        for id_ in range(states.shape[0]):
            pos = states[id_, :]
            reshape_rewards[id_] = rewards[pos[0], pos[1]]

        return cls(states=states,
                   rewards=reshape_rewards,
                   terminate_id=terminate_id,
                   start_id=start_id,
                   **kwargs)

    def move(self, step: npt.NDArray[np.int_]) -> float:
        """Apply step action and return reward."""
        current_state = self.get_pos(self.current_id)
        new_state = current_state + step
        if self.is_out(new_state):
            self.current_id = self.closest_id(new_state)
            reward = self.out_reward
        else:
            self.current_id = self.get_id(new_state)
            reward = self.get_state_reward(new_state)
        return reward

    def is_terminate(self) -> bool:
        """If maze is in terminate state."""
        return self.current_id == self._terminate_id

    def is_out(self, state: np.ndarray) -> bool:
        """If maze state is out of the maze."""
        return not np.any(np.all(state.transpose() == self.states, axis=1))

    def closest_id(self, state: npt.NDArray[np.int_]) -> int:
        """Return closest id to the possible state position."""
        return int(np.argmin(np.linalg.norm(self.states - state, axis=1)))

    def plot_rewards(self, **kwargs) -> QuadMesh:
        """Plot rewards map."""
        shape = (np.max(self.states[:, 0]), np.max(self.states[:, 1]))
        field = np.empty(shape=shape)
        field[...] = np.nan
        for id_ in range(self.states.shape[0]):
            pos = self.get_pos(id_)
            field[pos[0], pos[1]] = self.get_reward(id_)
        return plt.pcolormesh(self.rewards, **kwargs)

    def plot_states_values(self, values: np.ndarray, **kwargs) -> QuadMesh:
        """Plot values of some map."""
        shape = (np.max(self.states[:, 0] + 1), np.max(self.states[:, 1] + 1))
        field = np.empty(shape=shape)
        field[:] = np.nan
        for id_ in self.ids:
            pos = self.get_pos(id_)
            field[pos[0], pos[1]] = values[id_]
        return plt.pcolormesh(field, **kwargs)


class MazePolicy:
    """Maze action policy."""

    def __init__(self,
                 actions: npt.NDArray[np.int_],
                 probs: npt.NDArray[np.int_],
                 seed: Optional[int] = None):
        """
        Parameters
        ----------
        actions : npt.NDArray[np.int_]
            Actions table to move over the maze.
        probs : npt.NDArray[np.int_]
            Probability of every action.
        seed : int, optional
            Seed for sampling actions, by default None.
        """

        self.actions = actions
        self.probs = probs
        self._random = np.random.RandomState(seed=seed)

    def sample_action_id(self, state_id: int) -> int:
        """Sample action return id."""
        return self._random.choice(len(self.actions), p=self.probs[state_id])

    def sample_action(self, state_id: int) -> npt.NDArray[np.int_]:
        """Sample action for move."""
        id_ = self._random.choice(len(self.actions), p=self.probs[state_id])
        return self.get_action(id_)

    def get_action(self, id_: int) -> npt.NDArray[np.int_]:
        """Get action by id."""
        return self.actions[id_, :]

    def get_id(self, action: npt.NDArray[np.int_]) -> int:
        """Get id by action."""
        if isinstance(action, tuple):
            action = np.array(action)
        return int(np.where(np.all(self.actions == action, axis=1))[0])

    @property
    def ids(self) -> np.ndarray:
        """Actions ids list."""
        return np.arange(self.actions.shape[0])


class MazeEpisode:
    """Episode container."""

    def __init__(self, states, actions, rewards):

        self.states = states
        self.actions = actions
        self.rewards = rewards


def generate_episode(maze: MazeStates,
                     policy: MazePolicy,
                     max_times: int = -1):
    """
    Generate episode from maze and policy.
    If max_times > 0 then episode length is constrained.
    """
    states, actions, rewards = [], [], []
    time = 0
    while not maze.is_terminate():
        states.append(maze.current_id)

        action_id = policy.sample_action_id(maze.current_id)
        action = policy.get_action(action_id)

        reward = maze.move(action)

        actions.append(action_id)
        rewards.append(reward)
        time += 1

        if max_times != -1 and time == max_times:
            break

    return MazeEpisode(states, actions, rewards)
