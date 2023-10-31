import torch
import random
import numpy as np

from collections import deque
from snake_game import SnakeGame, Direction, Point
from model import Linear_QNet, QTrainer
import helper

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0 # Control the randomness in the learning 
        self.gamma = 0.95 # Discount Rate
        self.memory = deque(maxlen=MAX_MEMORY) # Will call popleft when too large

        # TODO: model
        self.Model = Linear_QNet(4, 256, 3)
        self.trainer = QTrainer(self.Model, learning_rate=LR, gamma=self.gamma)

    def get_state(self, game):

        directions = [game.direction == Direction.LEFT,
                      game.direction == Direction.RIGHT,
                      game.direction == Direction.UP,
                      game.direction == Direction.DOWN]

        return game.get_entire_game_context(), directions

    def remember(self, state_image, state_dir, action, reward, next_state_image, next_state_dir, game_over):
        self.memory.append((state_image, state_dir, action, reward, next_state_image, next_state_dir, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        state_image, state_dir, action, reward, next_state_image, next_state_dir, game_over = zip(*mini_sample)
        self.trainer.train_step(state_image, state_dir, action, reward, next_state_image, next_state_dir, game_over)

    def train_short_memory(self, state_image, state_dir, action, reward, next_state_image, next_state_dir, game_over):
        self.trainer.train_step(state_image, state_dir, action, reward, next_state_image, next_state_dir, game_over)

    def get_action(self, state_image, state_dir):
        # exploration / exploitation
        self.epsilon = 80 - self.number_of_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            image_state_tensor = torch.tensor(state_image, dtype=torch.float)
            direction_state_tensor = torch.tensor(state_dir, dtype=torch.float)
            prediction = self.Model(image_state_tensor, direction_state_tensor)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train():
    plot_score = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()

    while True:
        # get current state
        current_state_image, current_state_dir = agent.get_state(game)

        # get move based on current state
        final_move = agent.get_action(current_state_image, current_state_dir)

        # perform move and get new reward
        reward, game_over, score = game.play_step(final_move)
        new_state_image, new_state_dir = agent.get_state(game)

        agent.train_short_memory(current_state_image, current_state_dir, final_move, reward, new_state_image, new_state_dir, game_over)

        agent.remember(current_state_image, current_state_dir, final_move, reward, new_state_image, new_state_dir, game_over)

        if game_over:
            #  train long memory, plot result
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.Model.save()

            print(f'Game: {agent.number_of_games}, Score: {score}, Record: {record}')

            # Plot the results
            plot_score.append(score)
            total_score += score
            plot_mean_scores.append(total_score / agent.number_of_games)
            helper.plot(plot_score, plot_mean_scores)



if __name__ == '__main__':
    train()
