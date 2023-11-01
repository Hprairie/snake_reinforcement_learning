import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import os

class Conv_QNet(nn.Module):
    def __init__(self, input_dim, kernels, hidden_size, output_size):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(input_dim, kernels, (2, 2), 1),
                                        nn.ReLU(),
                                        nn.Conv2d(kernels, kernels, (2, 2), 1),
                                        nn.ReLU())
        self.fcls = nn.Sequential(nn.Linear(kernels * 6 * 6, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, output_size))
        
        self.input_dim = input_dim
        self.kernels = kernels

    def forward(self, X):
        # Reshape the image and dir
        X = X.reshape((-1, self.input_dim, 8, 8))

        # Pass through the network
        X = self.conv_layer(X)

        X = X.reshape(-1, self.kernels * 6 * 6)

        X = self.fcls(X)

        return X

    def save(self, filename='model.pth'):
        model_folder_path = './model'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, filename)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, learning_rate, gamma, C):
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.lr = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.C = C
        self.steps = 0

    def train_step(self, current_state, move, reward, next_state, game_over):

        current_state = torch.tensor(current_state, dtype=torch.float)
        move = torch.tensor(move, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(current_state.shape) == 3:
            current_state = torch.unsqueeze(current_state, 0)
            move = torch.unsqueeze(move, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = (game_over, )
        
        # Predict Q values with current state
        pred = self.model(current_state)

        # Predict Next Q values
        target = pred.clone()

        for idx in range(len(game_over)):
            Q_new = reward[idx]

            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.target_model(next_state[idx]))

            target[idx][torch.argmax(move[idx]).item()] = Q_new


        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # Incrimenet steps
        self.steps += 1

        # Check if we have completed a certain number of steps and copy our model into the target model
        if self.steps >= self.C:

            # Reset steps
            self.steps = 0

            # Copy model to target model
            self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer.step()
