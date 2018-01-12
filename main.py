import argparse
from time import sleep
import numpy as np
import config
import itertools
import torch
from vizdoom import DoomGame, ScreenResolution

from train import train
import eval

parser = argparse.ArgumentParser(description='PyTorch Recurrent Highway Network Language Model')
parser.add_argument('--algo', dest='algorithm',type=str, help='Algorithm to use')
parser.add_argument('--data-path', dest='data_path', type=str, default='data/Datasets/wikitext-2/', help='Path to dataset')
parser.add_argument('--input-size', dest='input_size', type=int, default=400, help='size of word embeddings')
parser.add_argument('--hidden-size', dest='hidden_size', type=int, default=1150, help='number of hidden units per layer')

parser.add_argument('--lr', dest='lr', type=float, default=30, help='initial learning rate')
parser.add_argument('--grad-clip', dest='grad_clip', type=float, default=0.5, help='gradient clipping')
parser.add_argument('--epochs', dest='num_epochs', type=int, default=20, help='upper epoch limit')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=80, metavar='N', help='batch size')
parser.add_argument('--bptt', dest='max_seq_len', type=int, default=70, help='sequence length')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.4, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', dest='tied_weights', action='store_false', help='tie the word embedding and softmax weights')
parser.add_argument('--seed', dest='seed', type=int, default=1111, help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', dest='sa  ve_path',type=str, help='path to save the final model')

args = parser.parse_args()

assert torch.cuda.is_available(), 'GPU Acceleration not available. Check if GPU is detected and CUDA toolkit is installed'
config_file = getattr(config, args.algo + 'Config')


if __name__ == '__main__':
    game = DoomGame()
    game.load_config("scenarios/health_gathering.cfg")
    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.set_window_visible(True)
    game.init()
    train(game, config_file)

    # Creates all possible actions depending on how many buttons there are.
    actions_num = game.get_available_buttons_size()
    actions = []
    for perm in itertools.product([False, True], repeat=actions_num):
        actions.append(list(perm))

    episodes = 10
    sleep_time = 0.028

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        while not game.is_episode_finished():

            # Gets the state and possibly to something with it
            state = game.get_state()

            # Makes a random action and save the reward.
            reward = game.make_action(np.random.choice(actions))

            print("State #" + str(state.number))
            print("Game Variables:", state.game_variables)
            print("Performed action:", game.get_last_action())
            print("Last Reward:", reward)
            print("=====================")

            # Sleep some time because processing is too fast to watch.
            if sleep_time > 0:
                sleep(sleep_time)

        print("Episode finished!")
        print("total reward:", game.get_total_reward())
        print("************************")
    train.learn()