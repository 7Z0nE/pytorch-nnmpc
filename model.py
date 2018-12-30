import torch
import torch.nn as nn
import numpy.random


class NegLogLikelihood(nn.Module):
    @staticmethod
    def forward(output, target):
        mean, diagonalcovariance = torch.chunk(output, 2)
        mean_err = mean - target
        loss = torch.pow(mean_err, 2)/diagonalcovariance + torch.logdet(torch.diagflat(diagonalcovariance))
        return -loss


class NN(nn.Module):

    def __init__(self, layers, activations, batch_norm=True):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(layers[:-1],layers[1:])])
        #standard activation function for each hidden layer
        self.activations = [nn.ReLU()]*(len(layers)-2)
        # override with array of passed in activation functions
        for i in range(len(activations)):
            self.activations[i] = activations[i]
        # batchnorm module for each hidden layer
        self.norm = nn.ModuleList([nn.BatchNorm1d(dim) for dim in layers[1:-1]])

    def forward(self, input):
        for layer, norm, activation in zip(self.layers[:-1], self.norm, self.activations):
            x = activation(norm(layer(x)))
        x = self.layers[-1](x)
        return x


class ProbabilisticEnvironmentModel(NN):

    def __init__(self, state_dim, action_dim, hidden_layers, activations, batch_norm=True):
        super(ProbabilisticEnvironmentModel, self).__init__()
        layers = [state_dim+action_dim]+hidden_layers+[state_dim+1] #+1 for reward
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(layers[:-1],layers[1:])])
        #standard activation function for each hidden layer
        self.activations = nn.ModuleList([nn.Relu()]*(len(layers)-2))
        # override with array of passed in activation functions
        for i in range(len(activations)):
            self.activations[i] = activations[i]
        # batchnorm module for each hidden layer
        self.norm = nn.ModuleList([nn.BatchNorm1d(dim) for dim in layers[1:-1]])

    def forward(self, state, action):
        torch.unsqueeze(state)
        torch.unsqueeze(action)
        input = torch.cat(state, action, dim=-2) # use negative dim so we can input batches aswell as single values
        output = torch.squeeze(super.forward(input))
        if output.dim() == 1:
            return output[:-1], output[-1]
        else:
            return output[:,:-1], output[:,-1]


class Ensemble():

    def __init__self(self, models):
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return [m(x) for m in self.models]


class ModelTrainer:

    def __init__(self, model, lossFunc=NegLogLikelihood, optimizer=torch.optim.Adam, lr=1e-3, lr_decay=1., batch_size=32):
        self.model = model
        self.lossFunc = lossFunc
        self.optimizer = optimizer(lr=lr, params=self.model.parameters())
        self.batch_size = batch_size
        if lr_decay == 1.:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)
        else:
            self.scheduler = None

    def train(self, inputs, targets):
        self.model.train()

        for batch_in, batch_t in zip(torch.chunk(inputs, self.batch_size), torch.chunk(targets, self.batch_size)):
            loss = self.lossFunc(self.model(batch_in), batch_t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if not self.scheduler == None:
                self.scheduler.step()


class EnsembleTrainer:

    def __init__(self, ensemble, lossFunc=NegLogLikelihood, optimizer=torch.optim.Adam, lr=1e-3, lr_decay=1., batch_size=32):
        self.trainers = [ModelTrainer(model, lossFunc, optimizer, lr, lr_decay, batch_size) for model in ensemble.models]

    def train(self, inputs, targets, n=0):
        for trainer in self.trainers:
            samples = randint(len(inputs), size=len(inputs))
            trainer.train(inputs[samples], targets[samples])