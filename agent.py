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
        self.gamma = 0.9 # Discount Rate
        self.memory = deque(maxlen=MAX_MEMORY) # Will call popleft when too large

        # TODO: model
        self.Model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.Model, learning_rate=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger staight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            # Danger left
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        state, action, reward, next_state, game_over = zip(*mini_sample)
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # exploration / exploitation

        self.epsilon = 80 - self.number_of_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.Model(state_tensor)
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
        current_state = agent.get_state(game)

        # get move based on current state
        final_move = agent.get_action(current_state)

        # perform move and get new reward
        reward, game_over, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        agent.train_short_memory(current_state, final_move, reward, new_state, game_over)

        agent.remember(current_state, final_move, reward, new_state, game_over)

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
