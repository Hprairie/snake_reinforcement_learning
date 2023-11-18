import torch
import json
import math
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, version):
        '''
        Initializes the model. The version number of the model wanting to be
        created will be passed in as a string. The get_model() function will
        then be called which will parse the JSON file and create the model
        for the agent.
        '''
        super().__init__()
        self._version = version

        # Initialize the model
        self.model_type = 'Single'
        self.model = self._get_model()

    def _get_model(self):
        '''
        Import the model based on the JSON specification of the version.
        Note that this function will not check the compatibility of the
        model with the board and assumes that the user correctly specified
        the model in the JSON version file.

        Returns
        -------

        sequential : nn.Sequential
            A compiled sequential model of all layer specified in order
            within the JSON version file.
        '''
        with open('model_config/{:s}.json'.format(self._version), 'r') as f:
            model_dic = json.load(f)


        modules = []
        state = []
        action = []
        for layer in model_dic['model']:
            layer_hyperparameters = model_dic['model'][layer]
            if ('Conv2D' in layer):
                modules.append(nn.Conv2d(**layer_hyperparameters))
            elif ('Flatten' in layer):
                modules.append(nn.Flatten())
            elif ('Dense' in layer):
                modules.append(nn.Linear(**layer_hyperparameters))
            elif ('ReLU' in layer):
                modules.append(nn.ReLU())
            elif ('Noisy' in layer):
                if ('Factorized' in layer):
                    modules.append(FactorizedNoisyLinear(**layer_hyperparameters))
                else:
                    modules.append(NoisyLinear(**layer_hyperparameters))
            elif ('Deuling' in layer):
                self.model_type = 'Deuling'
                # Create the State Head
                for state_layer in layer_hyperparameters['State']:
                    state_layer_hyperparameters = layer_hyperparameters['State'][state_layer]
                    if ('Dense' in state_layer):
                        state.append(nn.Linear(**state_layer_hyperparameters))
                    elif ('ReLU' in state_layer):
                        state.append(nn.ReLU())
                    elif ('Noisy' in state_layer):
                        if ('Factorized' in state_layer):
                            state.append(FactorizedNoisyLinear(**state_layer_hyperparameters))
                        else:
                            state.append(NoisyLinear(**state_layer_hyperparameters))
                # Create the Action Head
                for action_layer in layer_hyperparameters['Action']:
                    action_layer_hyperparameters = layer_hyperparameters['Action'][action_layer]
                    if ('Dense' in action_layer):
                        action.append(nn.Linear(**action_layer_hyperparameters))
                    elif ('ReLU' in action_layer):
                        action.append(nn.ReLU())
                    elif ('Noisy' in action_layer):
                        if ('Factorized' in action_layer):
                            action.append(FactorizedNoisyLinear(**action_layer_hyperparameters))
                        else:
                            action.append(NoisyLinear(**action_layer_hyperparameters))
        if self.model_type == 'Single':
            return nn.Sequential(*modules)
        else:
            return nn.ParameterDict({'Head': nn.Sequential(*modules),
                                     'State': nn.Sequential(*state),
                                     'Action': nn.Sequential(*action)})

    def forward(self, X):
        '''
        Function which will forward pass a batch through the model.

        Returns
        -------

        output : Tensor
            Return tensor in the shape of (batch_size, output_size)
        '''
        if self.model_type == 'Single':
            X = self.model(X)
            return X
        elif self.model_type == 'Deuling':
            # Pass through Single Head
            X = self.model['Head'](X)


            # Split into Deuling State Action Heads
            state_value = self.model['State'](X)
            action_value = self.model['Action'](X)

            # Normalize the Action Head
            action_value = action_value - action_value.mean(dim=-1, keepdim=True)

            return state_value + action_value

    def save(self, epoch, optimizer_state_dict, loss, path=''):
        '''
        Saves the current model along with the optimizer and other
        state data. Model and Optimizer weights are stored to the
        path specificed under the name model_version
        '''
        if self.model_type == 'Single':
            PATH = '{}/{:s}'.format(path, self._version)
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer_state_dict,
                        'model_type': 'Single',
                        'loss': loss}, PATH)
        
        elif self.model_type == 'Deuling':
            PATH = '{}/{:s}'.format(path, self._version)
            torch.save({'epoch': epoch,
                        'model_state_dict': {idx: self.model[idx].state_dict() for idx in self.model},
                        'optimizer_state_dict': optimizer_state_dict,
                        'model_type': 'Deuling',
                        'loss': loss}, PATH)


    def load(self, path, optimizer=None):
        '''
        Loads the weights and baises for both the current model
        and its optimizer if specified. Also stored loss and epoch
        of the saved model.

        Returns
        -------

        epoch : Int
            The number of games played by saved model
        loss : Int
            The loss of the last training step of the
            loaded model
        '''
        PATH = '{}/{:s}'.format(path, self._version)
        load_checkpoint = torch.load(PATH)
        if optimizer is not None:
            optimizer.load_state_dict(load_checkpoint['optimizer_state_dict'])
        if load_checkpoint['model_type'] == 'Single':
            self.model.load_state_dict(load_checkpoint['model_state_dict'])
        elif load_checkpoint['model_type'] == 'Deuling':
            ld_model = load_checkpoint['model_state_dict']
            self.model = {self.model[idx].load_state_dict(ld_model[idx]) for idx in self.model}
        epoch = load_checkpoint['epoch']
        loss = load_checkpoint['loss']

        return epoch, loss


class FactorizedNoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5, bias=True):
        torch.autograd.set_detect_anomaly(True)
        super(FactorizedNoisyLinear, self).__init__()
        # Save Hyperparameters for reset_parameters()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.bias = bias

        # Define weight of the linear layer
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('epsilon_sigma', torch.Tensor(out_features, in_features))

        # Define the bias of a linear layer
        if bias:
            self.mu_b = nn.Parameter(torch.Tensor(out_features))
            self.sigma_b = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer('epsilon_bias', torch.Tensor(out_features))

        # Init parameters (Described in paper)
        self.reset_parameters()
        self.reset_noise()

    def forward(self, X):
        # Reset the noise at each sample
        self.reset_noise()

        # Create weight
        weight = self.mu_w + (self.sigma_w * self.epsilon_sigma)
        if self.bias:
            bias = self.mu_b + (self.sigma_b * self.epsilon_bias) 
        
        
        # Forward pass the data through weight and bias
        if self.bias:
            X = F.linear(X, weight, bias)
        else:
            X = F.linear(X, weight)
        return X

    def reset_noise(self):
        def scale(size):
            noise = torch.randn(size)
            return noise.sign().mul(noise.abs().sqrt())
        
        # Create noise vectors
        epsilon_in = scale(self.in_features)
        epsilon_out = scale(self.out_features)

        # Save noise vectors
        self.epsilon_sigma = torch.outer(epsilon_out, epsilon_in)
        if self.bias:
            self.epsilon_bias = epsilon_out

    def reset_parameters(self):
        # Create distributions to sample mu and sigma
        mu_range = 1 / math.sqrt(self.in_features)
        sigma_range = self.sigma_init / math.sqrt(self.in_features)

        # Set up weight mu and sigma from distribution
        self.mu_w.data.uniform_(-mu_range, mu_range)
        self.sigma_w.data.uniform_(-sigma_range, sigma_range)
        # Set up bias mu and sigma from distribution
        if self.bias:
            self.mu_b.data.uniform_(-mu_range, mu_range)
            self.sigma_b.data.uniform_(-sigma_range, sigma_range)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__()
        # Save Hyperparameters for reset_parameters()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.bias = bias

        # Define weight of the linear layer
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('epsilon_sigma', torch.Tensor(out_features, in_features))

        # Define the bias of a linear layer
        if bias:
            self.mu_b = nn.Parameter(torch.Tensor(out_features))
            self.sigma_b = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer('epsilon_bias', torch.Tensor(out_featurs))

        # Init parameters (Described in paper)
        self.reset_parameters()

    def forward(self, X):
        # Reset the noise at each sample
        self.reset_noise()

        # Create weight
        weight = self.mu_w + (self.sigma_w * self.epsilon_sigma)
        if self.bias:
            bias = self.mu_b + (self.sigma_b * self.epsilon_bias)
        
        # Forward pass the data through weight and bias
        X = torch.matmul(weight, X)
        if self.bias:
            X = X + bias
        
        return X

    def reset_noise(self):
        # Sample noise from a normal distribution
        self.epsilon_sigma.normal_()
        if self.bias:
            self.epsilon_bias.normal_()

    def reset_parameters(self):
        # Create distributions to sample mu and sigma
        mu_range = math.sqrt(3 / self.in_features)
        sigma_range = self.sigma_init / math.sqrt(self.in_features)

        # Setup weight mu and sigma from distribution
        self.mu_w.data.uniform_(-mu_range, mu_range)
        self.sigma_w.data.uniform_(-sigma_range, sigma_range)

        # Setup bias mu and sigma from distribution
        if self.bias:
            self.mu_b.data.uniform_(-mu_range, mu_range)
            self.sigma_b.data.uniform_(-sigma_range, sigma_range)
