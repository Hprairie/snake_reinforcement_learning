import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, kernels, hidden_layer, output_size):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(nn.Conv2d(input_size, kernels, kernel_size=(3, 3)),
                                          nn.MaxPool2d(kernel_size=(2, 2)))
        self.conv_layer_2 = nn.Sequential(nn.Conv2d(kernels, kernels, kernel_size=(3, 3)),
                                          nn.MaxPool2d(kernel_size=(2, 2)))
        self.linear_layer = nn.Sequential(nn.Linear(6 * 4, hidden_layer),
                                          nn.ReLU(),
                                          nn.Linear(hidden_layer, output_size))

    def forward(self, X):
        X = X.reshape(-1, 32, 24)
        X = self.conv_layer_1(X)
        X = self.conv_layer_2(X)
        X = X.reshape(-1, 6 * 4)
        X = self.linear_layer(X)
        return X
    
    def save(self, filename='model.pth'):
        model_folder_path = './model'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, filename)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.model = model
        self.lr = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, current_state, move, reward, next_state, game_over):
        current_state = torch.tensor(current_state, dtype=torch.float)
        move = torch.tensor(move, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)


        if len(current_state.shape) == 1:
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
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(move).item()] = Q_new

        # 
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
