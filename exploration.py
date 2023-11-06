import random
import torch


class Zero:
    def get_action(self, agent, state):
        final_move = [0, 0, 0]

        # Generate next move only from model
        state_tensor = torch.tensor(state, dtype=torch.float)
        prediction = agent.model(torch.unsqueeze(state_tensor, 0))
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move

    def state_dict(self):
        return None


class EpsilonGreedy:
    def __init__(self, hyperparemeters) -> None:
        self.hyperparameters = hyperparemeters
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

    def state_dict(self):
        # Update state with epsilon values
        self.hyperparameters['epsilon'] = self.epsilon
        return self.hyperparameters


class Boltzmann:
    def __init__(self, hyperparemeters) -> None:
        self.hyperparameters = hyperparemeters

    def get_action(self, agent, state):
        final_move = [0, 0, 0]
        # Get the logits from the model
        state_tensor = torch.tensor(state, dtype=torch.float)
        logits = agent.model(torch.unsqueeze(state_tensor, 0))
        # Create a distribution from logits and sample the distribution
        action = torch.distributions.Categorical(logits=logits).sample().item()
        # OHE the action with all moves
        final_move[action] = 1
        return final_move

    def state_dict(self):
        return self.hyperparameters


def determine_exploration(exploration_strategy):
    if exploration_strategy is None:
        strategy = Zero()
    elif exploration_strategy['name'] == 'epsilon-greedy':
        strategy = EpsilonGreedy(exploration_strategy)
    elif exploration_strategy['name'] == 'boltzmann':
        strategy = Boltzmann(exploration_strategy)
    return strategy
