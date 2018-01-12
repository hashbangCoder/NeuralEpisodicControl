


DQNConfig = {
    'max_episodes': 1e6,
    'batch_size': 32,
    'resolution': (84,84),
    'discount': 0.99,
    'lr': 1e-4,
    'replay_buffer_size': 100000,
    'frame_stack': 4,
    'action_repeat': 4,
    'epsilon_anneal': 1e6,
    'min_buffer_size': 1000,

    'save_dir': 'DQN_Models/'
}

DDQNConfig = {

}

DuelingDQNConfig = {

}

PrioritizedDQNConfig = {

}