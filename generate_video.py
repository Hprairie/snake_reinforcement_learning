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


def play_game(game, model):
    for iteration in range(10):
        game_over = False
        while not game_over:
            # Get current state
            state = game.get_game_state()

            # Determine action
            final_move = [0, 0, 0]

            # Generate next move only from model
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = model(torch.unsqueeze(state_tensor, 0))
            move = torch.argmax(prediction).item()
            final_move[move] = 1

            # Pass action to game to generate tokens
            reward, game_over, score = game.play_step(final_move)
        game.reset()


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
    play_game(game, model)
