import copy
import json
import torch
import torch.optim as optim
import torch.nn as nn
import random

from torch import unsqueeze
from model import Model
from exploration import determine_exploration
from collections import deque


class DQN():
    def __init__(self, version):
        with open('model_config/{:s}.json'.format(version), 'r') as f:
            model_dic = json.load(f)

        # Create training and target models
        self.model = Model(version)
        self.target_model = copy.deepcopy(self.model)

        # Setup Training Hyperparameters
        self.gamma = model_dic['gamma']
        self.lr = model_dic['learning_rate']
        self.C = model_dic['C']
        self.batch_size = model_dic['batch_size']
        self.steps = 0

        # Setup Buffer, Optimizer, Criterion, and Exploration Strategy
        self.buffer = deque(maxlen=model_dic['buffer_size'])
        self._create_optimizer(model_dic['optimizer'])
        self._create_criterion(model_dic['criterion'])
        self.es = determine_exploration(model_dic['exploration_strategy'])

    def _create_optimizer(self, optimizer):
        ''''''
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _create_criterion(self, criterion):
        ''''''
        if criterion == 'MSE':
            self.criterion = nn.MSELoss()

    def _train_step(self, current_state, move, reward, next_state, game_over):

        current_state = torch.tensor(current_state, dtype=torch.float)
        move = torch.tensor(move, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        game_over = torch.tensor(game_over, dtype=torch.float)

        # Predict Q values with current state
        pred = self.model(current_state)

        # Predict Next Q values
        target = pred.clone()

        for idx in range(len(game_over)):
            target_pred = self.target_model(unsqueeze(next_state[idx], 0))
            discounted_rw = self.gamma * torch.max(target_pred)
            Q_new = reward[idx] + discounted_rw * (1 - game_over[idx])

            target[idx][torch.argmax(move[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # Incrimenet steps
        self.steps += 1

        # Check if we have completed a certain number of steps
        # and copy our model into the target model
        if self.steps % self.C == 0:
            # Copy model to target model
            self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer.step()

        return loss.item()

    def save_sample(self, state_tuple):
        self.buffer.append(state_tuple)

    def train_network(self):
        if len(self.buffer) > self.batch_size:
            mini_sample = random.sample(self.buffer, self.batch_size)
        else:
            mini_sample = self.buffer

        state, action, reward, next_state, game_over = zip(*mini_sample)
        loss = self._train_step(state, action, reward, next_state, game_over)

        return loss

    def get_action(self, state):
        return self.es.get_action(self, state)

    def save(self, path, epoch, loss):
        self.model.save(epoch, self.optimizer, loss, path)


class DoubleDQN():
    pass


class PriviligedDQN():
    pass
