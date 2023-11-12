import random
import torch


class Zero:
    '''
    Zero is a class for implementing a strategy where the agent always chooses
    the action with the highest predicted reward, without any randomness.

    Methods
    -------
    get_action(agent, state):
        Determines the action to be taken based on the current state and the
        agent's model.
    state_dict():
        Returns None as there are no internal state to be saved for this
        strategy.
    '''
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
    '''
    Parameters
    ----------
    hyperparameters : dict
        A dictionary containing the hyperparameters for the epsilon-greedy
        It should include the following keys:
        - 'epsilon': The initial value of epsilon, which represents the
                     probability of taking a random action.
        - 'epsilon_threshold': The minimum value that epsilon can reach
                               through decay.
        - 'epsilon_decay': The amount by which epsilon is reduced each time an
                           action is taken.

    Attributes
    ----------
    epsilon : float
        The current value of epsilon.
    epsilon_threshold : float
        The minimum value that epsilon can reach.
    epsilon_decay : float
        The amount by which epsilon is reduced each time an action is taken.
    hyperparameters : dict
        The original hyperparameters dictionary passed to the constructor.
    '''
    def __init__(self, hyperparemeters) -> None:
        self.hyperparameters = hyperparemeters
        self.epsilon = hyperparemeters['epsilon']
        self.epsilon_threshold = hyperparemeters['epsilon_threshold']
        self.epsilon_decay = hyperparemeters['epsilon_decay']

    def get_action(self, agent, state):
        '''
        Determines the action to be taken based on the current state
        and the epsilon-greedy strategy.

        Parameters
        ----------
        agent : object
            The agent that is taking the action.
        state : object
            The current state of the environment.

        Returns
        -------
        final_move : list
            A list representing the action to be taken.
        '''
        # Decay Epsilon every time we take an action
        if self.epsilon > self.epsilon_threshold:
            self.epsilon = max(self.epsilon_threshold, self.epsilon - self.epsilon_decay)

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
        '''
        Updates the hyperparameters dictionary with the current value
        of epsilon and returns it.

        Returns
        -------
        hyperparameters : dict
            The updated hyperparameters dictionary.
        '''
        # Update state with epsilon values
        self.hyperparameters['epsilon'] = self.epsilon
        return self.hyperparameters


class Boltzmann:
    '''
    Uses Boltzman exploration/exploitation strategy.

    Parameters
    ----------
    hyperparameters : dict
        A dictionary only holding a name.

    Methods
    -------
    get_action(agent, state):
        Determines the action to be taken based on the current state and
        the agent's model.
    state_dict():
        Returns the hyperparameters dictionary.
    '''
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
    '''
    Parses the JSON dictionary passed to determnine which strategy to return as
    a callable object. In order to add new exploration/exploitation strategies
    simply just create a class with a get_action() and state_dict() function
    and then add its name to the following if/elif block.
    '''

    if exploration_strategy is None:
        strategy = Zero()
    elif exploration_strategy['name'] == 'epsilon-greedy':
        strategy = EpsilonGreedy(exploration_strategy)
    elif exploration_strategy['name'] == 'boltzmann':
        strategy = Boltzmann(exploration_strategy)
    else:
        raise Exception('Unknown exploration passed. Check naming convention.')

    return strategy
