from __future__ import print_function
from vizdoom import *
import torch
from torch.autograd import Variable
from random import choice
from time import sleep
import models
from tqdm import tqdm
from utils import *


def train_dqn(net, game, config):
    # net = models.DQN(game.get_available_buttons_size())
    # net = net.cuda()
    exp_replay = models.ReplayBuffer(config['buffer_size'])

    pbar = tqdm(config['num_episodes'])
    pbar.set_description('')
    actions = [[True, False, False], [False, True, False], [False, False, True]]

    sleep_time = 1.0 / DEFAULT_TICRATE # = 0.028

    for episode in range(config['num_episodes']):
        pbar.update()
        game.new_episode()
        while not game.is_episode_finished():
            # start of episode, stack frames for input to net
            if game.tic < config['frame_stack'] * config['action_repeat']:
                _input = fill_input_buffer(game, actions, config)
                assert len(_input) == config['frame_stack'], 'Input has incorrect size'

            _input = torch.FloatTensor(_input)
            gpu_input = Variable(_input.cuda())
            # choose action greedily
            net_action = net(gpu_input).max(1)[1].cpu()[0]

            # get epsilon value after annealing
            epsilon = calc_epsilon(exp_replay.pointer, config['epsilon_anneal'])
            action = calc_action(actions[net_action], actions, epsilon)
            reward = game.make_action(action, config['action_repeat'])
            next_state = torch.FloatTensor(game.get_state().screen_buffer)

            # add experience to buffer
            exp_replay.add_to_buffer((_input, action, next_state, reward, game.is_episode_finished()))
            if exp_replay.pointer < config['min_buffer_size']:
                continue

            # If buffer has enough experiences, start training
            exp_batch = exp_replay.get_batch(config['batch_size'])
            train_batch = 




        # Check how the episode went.
        print("Episode finished.")
        print("Total reward:", game.get_total_reward())
        print("************************")

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
