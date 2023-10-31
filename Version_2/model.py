import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(1, 1, (2, 2), 1),
                                        nn.Conv2d(1, 1, (2, 2), 1))
        self.fcls = nn.Sequential(nn.Linear(1 * 4 * 4 + input_size, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, output_size))

    def forward(self, X_board_image, X_directional_data):
        # Reshape the image and dir
        X_board_image = X_board_image.reshape((-1, 1, 6, 6))
        X_directional_data = X_directional_data.reshape((-1, 8))

        # Send the board data through the CNN
        X_board_image = self.conv_layer(X_board_image)
        
        # Flatten the board data
        X_board_image = X_board_image.reshape((-1, 1*4*4))

        # Recombine board data with directional data
        X = torch.cat((X_board_image, X_directional_data), dim = 1)

        # Push through fully connected layers
        X = self.fcls(X)

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

    def train_step(self, current_state_image, current_sate_dir, move, reward, next_state_image, next_state_dir, game_over):

        current_state_image = torch.tensor(current_state_image, dtype=torch.float)
        current_sate_dir = torch.tensor(current_sate_dir, dtype=torch.float)
        move = torch.tensor(move, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state_image = torch.tensor(next_state_image, dtype=torch.float)
        next_state_dir = torch.tensor(next_state_dir, dtype=torch.float)

        if len(current_state_image.shape) == 2:
            current_state_image = torch.unsqueeze(current_state_image, 0)
            current_sate_dir = torch.unsqueeze(current_sate_dir, 0)
            move = torch.unsqueeze(move, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state_image = torch.unsqueeze(next_state_image, 0)
            next_state_dir = torch.unsqueeze(next_state_dir, 0)
            game_over = (game_over, )
        
        # Predict Q values with current state
        pred = self.model(current_state_image, current_sate_dir)

        # Predict Next Q values
        target = pred.clone()

        for idx in range(len(game_over)):
            Q_new = reward[idx]

            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state_image[idx], next_state_dir[idx]))

            target[idx][torch.argmax(move[idx]).item()] = Q_new

        # 
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
