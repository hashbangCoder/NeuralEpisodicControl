from scipy.misc import imresize
from random import choice
import numpy as np


def preprocess(arr, size):
    return imresize(arr, size=size)

def rgb_to_gray(arr):
    assert arr.shape[0] == 3, 'Image not RGB'
    return  0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]

def fill_input_buffer(game, actions, config):
    """
    :param game: game object
    :param actions: list of actions
    :param config: config dict
    :return: list of processed frames
    """
    net_input = []
    while len(net_input) < config['frame_stack']:
        state = game.get_state()
        obs = preprocess(rgb_to_gray(state.screen_buffer), size=((1,) + config['resolution']))
        net_input.append(obs)
        game.make_action(choice(actions), config['action_repeat'])      # random actions
    return net_input

def calc_epsilon(step, duration, start=0.99, stop= 0.05):
    """
    :param step: global step counter
    :param start: upper bound on epsilon
    :param stop: lower bound on epsilon
    :return: epslion value between start and stop
    """
    if step < 2000:
        return 0.99
    else:
        return min(stop, start - start/duration)


def calc_action(net_action, actions, epsilon):
    _act_choice = np.random.choice([0, 1], p=[epsilon, 1-epsilon])
    return net_action if _act_choice == 0 else choice(actions)
