import torch
import json
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, version):
        '''
        Initializes the model. The version numver of the model wanting to be
        created will be passed in as a string. The get_model() function with
        then be called which will parse the JSON file and create the model
        for the agent.
        '''
        super().__init__()
        self._version = version

        # Initialize the model
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

        return nn.Sequential(*modules)

    def forward(self, X):
        '''
        Function which will forward pass a batch through the model.

        Returns
        -------

        output : Tensor
            Return tensor in the shape of (batch_size, output_size)
        '''
        X = self.model(X)
        return X

    def save(self, epoch, optimizer_state_dict, loss, path=''):
        '''
        Saves the current model along with the optimizer and other
        state data. Model and Optimizer weights are stored to the
        path specificed under the name model_version
        '''
        PATH = '{}/model_{:s}'.format(path, self._version)
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer_state_dict,
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
        load_checkpoint = torch.load(path)
        if optimizer is not None:
            optimizer.load_state_dict(load_checkpoint['optimizer_state_dict'])
        self.model.load_state_dict(load_checkpoint['model_state_dict'])
        epoch = load_checkpoint['epoch']
        loss = load_checkpoint['loss']

        return epoch, loss
