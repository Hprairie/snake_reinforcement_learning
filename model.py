import torch
import json
import torch.nn as nn


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
            elif ('Deuling' in layer):
                self.model_type = 'Deuling'
                # Create the State Head
                for state_layer in layer_hyperparameters['State']:
                    state_layer_hyperparameters = layer_hyperparameters['State'][state_layer]
                    if ('Dense' in layer):
                        state.append(nn.Linear(**state_layer_hyperparameters))
                    elif ('ReLU' in layer):
                        state.append(nn.ReLU())
                # Create the Action Head
                for action_layer in layer['Action']:
                    action_layer_hyperparameters = layer_hyperparameters['Action'][action_layer]
                    if ('Dense' in layer):
                        action.append(nn.Linear(**action_layer_hyperparameters))
                    elif ('ReLU' in layer):
                        action.append(nn.ReLU())
        if self.model_type is 'Single':
            return nn.Sequential(*modules)
        else:
            return {'Head': nn.Sequential(*modules), 
                    'State': nn.Sequential(*state), 
                    'Action': nn.Sequential(*action)}

    def forward(self, X):
        '''
        Function which will forward pass a batch through the model.

        Returns
        -------

        output : Tensor
            Return tensor in the shape of (batch_size, output_size)
        '''
        if self.model_type is 'Single':
            X = self.model(X)
            return X
        elif self.model_type is 'Deuling':
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
        if self.model_type is 'Single':
            PATH = '{}/{:s}'.format(path, self._version)
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer_state_dict,
                        'model_type': 'Single',
                        'loss': loss}, PATH)
        
        elif self.model_type is 'Deuling':
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
        if load_checkpoint['model_type'] is 'Single':
            self.model.load_state_dict(load_checkpoint['model_state_dict'])
        elif load_checkpoint['model_type'] is 'Deuling':
            ld_model = load_checkpoint['model_state_dict']
            self.model = {self.model[idx].load_state_dict(ld_model[idx]) for idx in self.model}
        epoch = load_checkpoint['epoch']
        loss = load_checkpoint['loss']

        return epoch, loss
