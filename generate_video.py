'''
Use this file to take an already trained model and create videos of it playing
snake.
'''
import sys
import json
import random
import torch
from model import Model
from snake_game import SnakeGame


def create_video(game, model):
    pass


if __name__ == "__main__":
    version = sys.argv[1]

    with open('model_config/{:s}.json'.format(version), 'r') as f:
        model_dic = json.load(f)

    # Seed the enviroment if needed
    if model_dic['seed'] is not None:
        random.seed(model_dic['seed'])
        torch.manual_seed(model_dic['seed'])

    # Initialize the Game
    game = SnakeGame(model_dic['board_size'],
                     model_dic['frames'],
                     model_dic['start_length'],
                     model_dic['display_game'],
                     model_dic['seed'],
                     model_dic['max_time_rate'])

    # Set Exploration Strategy to None as we do not want any random moves
    model_dic['exploration_strategy'] = None

    # Initialize the model
    model = Model(version)

    # Load in the model to the agent
    epoch, _ = model.load('models')

    # Print the stats of the loaded model
    print(f'Loaded model trained on {epoch} games.')

    # Run create_video() to capture videos
    create_video(game, model)
