import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn


class CNN(nn.Module):
    def __init__(self, num_actions, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bnorm3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(192, self.embed_size)

    def forward(self, _input):
        x = F.relu(self.bnorm1(self.conv1(_input)))
        x = F.relu(self.bnorm2(self.conv2(x)))
        x = F.relu(self.bnorm3(self.conv3(x)))
        x = x.view(-1, 192)
        x = F.relu(self.fc1(x))
        return x


class NeuralDict(nn.Module):
    def __init__(self, key_size, dict_size):
        super().__init__()
        self.dict_size = dict_size
        self.key_size = key_size
        self.fill_count = 0
        self.stale_ind = None

        self.keys = Variable(torch.zeros(self.dict_size, self.key_size).cuda())
        self.values = Variable(torch.zeros(self.dict_size, 1).cuda())
        self.recency_map = Variable(torch.zeros(self.dict_size, 1).cuda())

    def forward(self, _input):
        return

    def get_knn(self, query, k):
        """
        get K nearest neighbors by cosine distance for query
        """
        if self.fill_count < self.dict_size:
            norm_keys = F.normalize(self.keys[:self.fill_count, :], dim=1)
        else:
            norm_keys = F.normalize(self.keys, dim=1)

        norm_query = F.normalize(query, dim=1)
        cosine_dist = torch.mm(norm_keys, norm_query.t())
        return torch.topk(cosine_dist, k, dim=1)

    def add_key(self, key, value):
        if self.fill_count >= self.dict_size:
            self.keys[self.stale_ind] = key
            self.values[self.stale_ind] = value
        else:
            self.keys[self.fill_count] = key
            self.values[self.fill_count] = value
            self.fill_count += 1

    def update_recency_map(self, nn_indices):
        mask = Variable(torch.zero(*nn_indices.size()).cuda().fill_(1))
        self.recency_map.scatter_add(0, nn_indices.view(-1), mask)
        self.recency_map.add_(-1).clamp_(0, 100)
        _, self.stale_ind = self.recency_map.min(0)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.pointer = 0
        # opt for dict due to O(1) random lookup
        self.buffer = dict()

    def add_to_buffer(self, experience):
        self.buffer[self.pointer % self.buffer_size] = experience
        self.pointer += 1

    def get_batch(self, batch_size):
        inds = np.random.randint(0, min(self.pointer, self.buffer_size), batch_size)
        batch = []
        for _ind in inds:
            batch.append(self.buffer[_ind])
        return list(zip(*batch))


class NEC(nn.Module):
    def __init__(self, cnn, neural_dict):
        super().__init__()
        self.cnn = cnn
        self.neral_dicts = neural_dict

    def forward(self, obs):
        cnn_embed = self.cnn(obs)


