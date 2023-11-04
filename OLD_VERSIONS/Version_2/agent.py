import torch
import random
import numpy as np

from collections import deque
from snake_game import SnakeGame, Direction, Point
from model import Conv_QNet, QTrainer
import helper

MAX_MEMORY = 100_000
BATCH_SIZE = 32
LR = 0.0001
TARGET_UPDATE = 1000
AGENT_HISTORY_LENGTH = 4

class Agent:
    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 1 # Control the randomness in the learning 
        self.epsilon_bottom = 0.001
        self.epsilon_decay = 0.0002
        self.gamma = 0.9 # Discount Rate
        self.replay_buffer = deque(maxlen=MAX_MEMORY) # Replay Buffer
        self.Model = Conv_QNet(AGENT_HISTORY_LENGTH, 32, 128, 3)
        self.trainer = QTrainer(self.Model, learning_rate=LR, gamma=self.gamma, C=TARGET_UPDATE)
        self.frame_history_buffer = deque(maxlen=AGENT_HISTORY_LENGTH)

    def get_state(self, game):
        
        current_frame = game.get_entire_game_context()

        # Used at the beginning of the game to fill the buffer (will only occur on init)
        while len(self.frame_history_buffer) != AGENT_HISTORY_LENGTH:
            self.frame_history_buffer.append(current_frame)

        # Push the current frame into the buffer if not already there
        if not np.array_equal(current_frame,self.frame_history_buffer[-1]):
            self.frame_history_buffer.append(current_frame)

        # Return the replay buffer as a signle np tensor
        return_array = np.stack(self.frame_history_buffer, axis=0)
        return return_array
    

    def remember(self, state, action, reward, next_state, game_over):
        self.replay_buffer.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.replay_buffer) > BATCH_SIZE:
            mini_sample = random.sample(self.replay_buffer, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.replay_buffer

        state, action, reward, next_state, game_over = zip(*mini_sample)
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # exploration / exploitation
        if self.epsilon > self.epsilon_bottom:
            self.epsilon -= self.epsilon_decay

        final_move = [0, 0, 0]

        if random.random() < self.epsilon:
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

        # agent.train_short_memory(current_state, final_move, reward, new_state, game_over)

        agent.remember(current_state, final_move, reward, new_state, game_over)

        agent.train_long_memory()

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
