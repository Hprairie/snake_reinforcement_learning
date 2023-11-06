import sys
import agents
import json
import snake_game
import torch
import random
from logger import save_training_logs


def train(agent, game, epochs):
    # Initialize tracking parameters
    model_logs = {'score': [],
                  'mean_score': [],
                  'game': [],
                  'network_name': agent.name,
                  'record': 0}
    total_score = 0
    current_epoch_count = 0

    while current_epoch_count < epochs:
        sys.getrecursionlimit()
        # get current state
        current_state = game.get_game_state()

        # get move based on current state
        final_move = agent.get_action(current_state)

        # perform move and get new reward
        reward, game_over, score = game.play_step(final_move)
        new_state = game.get_game_state()

        agent.save_sample((current_state, final_move,
                           reward, new_state, game_over))

        loss = agent.train_network()

        if game_over:
            # Reset the game when over and increase the number of epcohs
            game.reset()
            current_epoch_count += 1

            # Log record information
            if score > model_logs['record']:
                model_logs['record'] = score
                agent.save_baseline('models', current_epoch_count, loss)

            # Log Epoch information
            print(f'Game: {current_epoch_count},',
                  f'Score: {score},'
                  f'Record: {model_logs["record"]}')

            # Add Results to Logs
            total_score += score
            model_logs['score'].append(score)
            model_logs['game'].append(current_epoch_count)
            model_logs['mean_score'].append(total_score / current_epoch_count)
            model_logs

    save_training_logs(model_logs, agent.version)


if __name__ == '__main__':
    version = sys.argv[1]

    with open('model_config/{:s}.json'.format(version), 'r') as f:
        model_dic = json.load(f)

    # Seed the enviroment if needed
    if model_dic['seed'] is not None:
        random.seed(model_dic['seed'])
        torch.manual_seed(model_dic['seed'])

    # Initialize the Game
    game = snake_game.SnakeGame(model_dic['board_size'],
                                model_dic['frames'],
                                model_dic['start_length'],
                                model_dic['display_game'],
                                model_dic['seed'],
                                model_dic['max_time_rate'])

    # Initialize the Agent
    if model_dic['agent'] == 'DQN':
        agent = agents.DQN(version)
    elif model_dic['agent'] == 'DoubleDQN':
        agent = agents.DoubleDQN(version)

    #  Run trainer
    train(agent, game, model_dic['epochs'])
