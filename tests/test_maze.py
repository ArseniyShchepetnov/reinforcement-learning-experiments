"""Test maze."""
from src.maze import MazeStates, MazePolicy
import numpy as np


def test_states_default():
    """Check maze is instantiated correctly."""
    states = np.array([[0, 0],
                       [1, 0],
                       [0, 1],
                       [1, 1]])

    rewards = [1, 2, 3, 4]

    maze = MazeStates(states=states,
                      rewards=rewards,
                      current_id=0,
                      out_reward=-10)

    id_ = maze.get_id((1, 0))
    assert id_ == 1

    assert np.all(maze.current_id == 0)
    pos = maze.get_pos(maze.current_id)
    assert np.all(pos == [0, 0])

    reward_out = maze.move((-1, 0))
    assert reward_out == -10
    assert np.all(maze.current_id == 0)

    reward_next = maze.move((1, 0))
    assert reward_next == maze.get_state_reward((1, 0))
    assert maze.current_id == maze.get_id((1, 0))


def test_from_rewards():
    """Test maze instantiated correctly from rewards."""
    rewards = np.array([[1, 3],
                        [2, 4]])

    maze = MazeStates.from_rewards(rewards=rewards,
                                   start_state=(0, 0),
                                   terminate_state=(1, 1),
                                   out_reward=-10)

    id_ = maze.get_id((1, 0))
    assert id_ == 2

    assert np.all(maze.current_id == 0)
    pos = maze.get_pos(maze.current_id)
    assert np.all(pos == [0, 0])

    reward_out = maze.move((-1, 0))
    assert reward_out == -10
    assert np.all(maze.current_id == 0)

    reward_next = maze.move((1, 0))
    assert reward_next == maze.get_state_reward((1, 0))
    assert maze.current_id == maze.get_id((1, 0))


def test_move_towards_terminate():
    """Test move is correct."""
    rewards = np.array([[1, 3],
                        [2, 4]])

    maze = MazeStates.from_rewards(rewards=rewards,
                                   start_state=(0, 0),
                                   terminate_state=(1, 1),
                                   out_reward=-10)

    assert not maze.is_terminate()
    reward = maze.move((1, 1))
    assert reward == 4
    assert maze.is_terminate()


def test_policy_default():
    """Test action policy."""
    actions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    probs = [0.25, 0.25, 0.25, 0.25]

    policy = MazePolicy(actions=actions, probs=probs)
    assert policy.get_id((0, 1)) == 1
