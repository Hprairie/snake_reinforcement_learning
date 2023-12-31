import copy
import json
import torch
import torch.optim as optim
import torch.nn as nn
import random

from torch import unsqueeze
from model import Model
from loss import weighted_MSELoss
from exploration import determine_exploration
from buffer import determine_buffer
from collections import deque


class Agent():
    def _create_optimizer(self, optimizer):
        ''''''
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _create_criterion(self, criterion):
        ''''''
        if criterion == 'MSE':
            self.criterion = nn.MSELoss()
        elif criterion == 'WeightedMSE':
            self.criterion = weighted_MSELoss()

    def save_sample(self, sample):
        self.buffer.append(sample)

    def get_action(self, state):
        return self.es.get_action(self, state)

    def save_baseline(self, path, epoch, loss):
        self.model.save(epoch, self.optimizer.state_dict(), loss, path)


class DQN(Agent):
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
        self.version = version
        self.name = "DQN"

        # Setup Buffer, Optimizer, Criterion, and Exploration Strategy
        self.buffer = deque(maxlen=model_dic['buffer_size'])
        self._create_optimizer(model_dic['optimizer'])
        self._create_criterion(model_dic['criterion'])
        self.es = determine_exploration(model_dic['exploration_strategy'])

    def _train_step(self, current_state, move, reward, next_state, game_over):
        ''''''
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

    def train_network(self):
        ''''''
        if len(self.buffer) > self.batch_size:
            mini_sample = random.sample(self.buffer, self.batch_size)
        else:
            mini_sample = self.buffer

        state, action, reward, next_state, game_over = zip(*mini_sample)
        loss = self._train_step(state, action, reward, next_state, game_over)

        return loss

    def save_training(self, path, epoch, loss):
        PATH = '{}/model_{:s}'.format(path, self._version)
        torch.save({'epoch': epoch,
                    'online_state_dict': self.model.state_dict(),
                    'target_state_dict': self.target_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'buffer': self.buffer,
                    'exploration_state': self.es.state_dict(),
                    'loss': loss}, PATH)

    def load_training(self, path):
        PATH = '{}/{:s}'.format(path, self._version)
        load_checkpoint = torch.load(PATH)

        epoch = load_checkpoint['epoch']
        self.model.load_state_dict(load_checkpoint['online_state_dict'])
        self.target_model.load_state_dict(load_checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(load_checkpoint['optimizer_state_dict'])
        self.buffer = load_checkpoint['buffer']
        self.es = determine_exploration(load_checkpoint['exploration_state'])
        loss = load_checkpoint['loss']

        return epoch, loss


class DoubleDQN(Agent):
    def __init__(self, version) -> None:
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
        self.version = version
        self.name = "DoubleDQN"

        # Setup Buffer, Optimizer, Criterion, and Exploration Strategy
        self.buffer = deque(maxlen=model_dic['buffer_size'])
        self._create_optimizer(model_dic['optimizer'])
        self._create_criterion(model_dic['criterion'])
        self.es = determine_exploration(model_dic['exploration_strategy'])

    def _train_step(self, current_state, move, reward, next_state, game_over):
        ''''''
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
            # First use the online network to determine which action to take
            online_model_pred = self.model(unsqueeze(next_state[idx], 0))
            online_model_action = torch.argmax(online_model_pred).item()
            # Second use the target network to determine value of next action
            target_pred = self.target_model(unsqueeze(next_state[idx], 0))
            discounted_rw = self.gamma * target_pred[0][online_model_action]
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

    def train_network(self):
        ''''''
        if len(self.buffer) > self.batch_size:
            mini_sample = random.sample(self.buffer, self.batch_size)
        else:
            mini_sample = self.buffer

        state, action, reward, next_state, game_over = zip(*mini_sample)
        loss = self._train_step(state, action, reward, next_state, game_over)

        return loss

    def save_training(self, path, epoch, loss):
        PATH = '{}/model_{:s}'.format(path, self._version)
        torch.save({'epoch': epoch,
                    'online_state_dict': self.model.state_dict(),
                    'target_state_dict': self.target_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'buffer': self.buffer,
                    'exploration_state': self.es.state_dict(),
                    'loss': loss}, PATH)

    def load_training(self, path):
        PATH = '{}/{:s}'.format(path, self._version)
        load_checkpoint = torch.load(PATH)

        epoch = load_checkpoint['epoch']
        self.model.load_state_dict(load_checkpoint['online_state_dict'])
        self.target_model.load_state_dict(load_checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(load_checkpoint['optimizer_state_dict'])
        self.buffer = load_checkpoint['buffer']
        self.es = determine_exploration(load_checkpoint['exploration_state'])
        loss = load_checkpoint['loss']

        return epoch, loss


class PrioritizedDDQN(Agent):
    def __init__(self, version) -> None:
        # Fetch the JSON configuration
        with open('model_config/{:s}.json'.format(version), 'r') as f:
            model_dic = json.load(f)

        # Create training and target models
        self.model = Model(version)
        self.target_model = copy.deepcopy(self.model)

        # Setup Training Hyperparameters
        self.gamma = model_dic['gamma']
        self.lr = model_dic['learning_rate']
        self.C = model_dic['C']
        self.steps = 0
        self.version = version
        self.name = "PrioritizedDDQN"

        # Setup Buffer, Optimizer, Criterion, and Exploration Strategy
        self.buffer = determine_buffer(model_dic['buffer'])
        self._create_optimizer(model_dic['optimizer'])
        self._create_criterion(model_dic['criterion'])
        self.es = determine_exploration(model_dic['exploration_strategy'])

    def _train_step(self, current_state, move, reward, next_state, game_over, weights):
        ''''''
        current_state = torch.tensor(current_state, dtype=torch.float)
        move = torch.tensor(move, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        game_over = torch.tensor(game_over, dtype=torch.float)
        weights = torch.tensor(weights, dtype=torch.float)

        # Predict Q values with current state
        pred = self.model(current_state)

        # Predict Next Q values
        target = pred.clone()

        # Absolute Temporal Difference
        td = torch.zeros(len(game_over), dtype=torch.float)

        for idx in range(len(game_over)):
            # Update Buffer Hyperparameters
            self.buffer.update_hyperparameters()

            # First use the online network to determine which action to take
            online_model_pred = self.model(unsqueeze(next_state[idx], 0))
            online_model_action = torch.argmax(online_model_pred).item()

            # Second use the target network to determine value of next action
            target_pred = self.target_model(unsqueeze(next_state[idx], 0))
            discounted_rw = self.gamma * target_pred[0][online_model_action]
            Q_new = reward[idx] + discounted_rw * (1 - game_over[idx])

            # Update the absolute temporal difference plus offset
            td[idx] = torch.abs(Q_new - pred[idx][torch.argmax(move[idx]).item()]) + 0.1

            # Update the target
            target[idx][torch.argmax(move[idx]).item()] = Q_new


        self.optimizer.zero_grad()
        loss = self.criterion(target, pred, weights)
        loss.backward()

        # Incrimenet steps
        self.steps += 1

        # Check if we have completed a certain number of steps
        # and copy our model into the target model
        if self.steps % self.C == 0:
            # Copy model to target model
            self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer.step()

        return loss.item(), td.detach().numpy()

    def train_network(self):
        ''''''
        indexs, elements, weights = self.buffer.get_batch()

        state, action, reward, next_state, game_over = zip(*elements)
        loss, td = self._train_step(state, action, reward, next_state, game_over, weights)

        # Update transition priority based on absolute temporal difference
        self.buffer.update_priorities(indexs, td)

        return loss

    def save_training(self, path, epoch, loss): # Not working with Deuling Architecture
        PATH = '{}/model_{:s}'.format(path, self._version)
        torch.save({'epoch': epoch,
                    'online_state_dict': self.model.state_dict(),
                    'target_state_dict': self.target_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'buffer': self.buffer,
                    'exploration_state': self.es.state_dict(),
                    'loss': loss}, PATH)

    def load_training(self, path):
        PATH = '{}/{:s}'.format(path, self._version)
        load_checkpoint = torch.load(PATH)

        epoch = load_checkpoint['epoch']
        self.model.load_state_dict(load_checkpoint['online_state_dict'])
        self.target_model.load_state_dict(load_checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(load_checkpoint['optimizer_state_dict'])
        self.buffer = load_checkpoint['buffer'] # This doesn't work fix it
        self.es = determine_exploration(load_checkpoint['exploration_state'])
        loss = load_checkpoint['loss']

        return epoch, loss


class MultiStep(Agent):
    pass


class NioseyNet(Agent):
    pass


class RainbowQN(Agent):
    pass


class Actor2Critic():
    pass
