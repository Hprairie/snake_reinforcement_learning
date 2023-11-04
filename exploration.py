import random
import torch


class EpsilonGreedy:
    def __init__(self, hyperparemeters) -> None:
        self.epsilon = hyperparemeters['epsilon']
        self.epsilon_threshold = hyperparemeters['epsilon_threshold']
        self.epsilon_decay = hyperparemeters['epsilon_decay']

    def get_action(self, agent, state):
        # Decay Epsilon every time we take an action
        if self.epsilon > self.epsilon_threshold:
            self.epsilon -= self.epsilon_decay

        final_move = [0, 0, 0]

        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = agent.model(torch.unsqueeze(state_tensor, 0))
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def determine_exploration(exploration_strategy):
    if exploration_strategy['name'] == 'epsilon-greedy':
        strategy = EpsilonGreedy(exploration_strategy)

    return strategy