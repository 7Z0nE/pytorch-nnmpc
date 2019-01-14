import torch
import torch.nn as nn


class Swish(torch.nn.Module):
    r"""Applies the element-wise function:
    .. math::
        \text{Swish}(x) = x * text{Sigmoid}(x) = x * \frac{1}{1 + \exp(-x)}
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> m = Swish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input):
        return input * torch.sigmoid(input)


class NegLogLikelihood(nn.Module):
    @staticmethod
    def forward(output, target):
        mean, diagonalcovariance = torch.chunk(output, 2)
        mean_err = mean - target
        loss = torch.pow(mean_err, 2)/diagonalcovariance + torch.logdet(torch.diagflat(diagonalcovariance))
        return -loss


class NN(nn.Module):
    """
    A plain neural network.
    """

    def __init__(self, layers, activations, batch_norm=True):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(layers[:-1], layers[1:])])
        #standard activation function for each hidden layer
        self.activations = [Swish()]*(len(layers)-2)
        # override with array of passed in activation functions
        for i in range(len(activations)):
            self.activations[i] = activations[i]
        # batchnorm module for each hidden layer
        self.norm = nn.ModuleList([nn.BatchNorm1d(dim) for dim in layers[1:-1]])

    def forward(self, x):
        for layer, norm, activation in zip(self.layers[:-1], self.norm, self.activations):
            x = activation(norm(layer(x)))
        x = self.layers[-1](x)
        return x


class EnvironmentModelSeparateReward(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_layers, activations, batch_norm=True):
        super(EnvironmentModelSeparateReward, self).__init__()
        layers_dynamics = [state_dim+action_dim]+hidden_layers+[state_dim]
        layers_reward = [state_dim+action_dim]+hidden_layers+[1]
        self.model_dynamics = NN(layers_dynamics, activations, batch_norm)
        self.model_reward = NN(layers_reward, activations, batch_norm)

    def forward(self, x, catreward=True):
        x = torch.squeeze(x)
        if catreward:
            return torch.cat((self.model_dynamics(x), self.model_reward(x)), dim=-1)
        else:
            return (self.model_dynamics(x), self.model_reward(x))


    def propagate(self, state, action):
        input = torch.cat((state, action), dim=-1) # use negative dim so we can input batches aswell as single values
        output = self.forward(input)
        output = torch.squeeze(output)
        if output.dim() == 1:
            return output[:-1], output[-1]
        else:
            return output[:,:-1], output[:,-1]

class EnvironmentModel(NN):
    """
    A neural network parameterized to model an OpenAI Gym environment.
    """

    def __init__(self, state_dim, action_dim, hidden_layers, activations, batch_norm=True):
        layers = [state_dim+action_dim]+hidden_layers+[state_dim+1] #+1 for reward
        super(EnvironmentModel, self).__init__(layers, activations)

    def propagate(self, state, action):
        input = torch.cat((state, action), dim=-1) # use negative dim so we can input batches aswell as single values
        output = self.forward(input)
        output = torch.squeeze(output)
        if output.dim() == 1:
            return output[:-1], output[-1]
        else:
            return output[:,:-1], output[:,-1]


class ProbabilisticEnvironmentModel(NN):
    """
    A neural network parameterized to model an OpenAI Gym environemnt.
    Also outputs the diagonal covariance of the prediction.
    """

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

    def propagate(self, state, action):
        torch.unsqueeze(state)
        torch.unsqueeze(action)
        input = torch.cat(state, action, dim=-2) # use negative dim so we can input batches aswell as single values
        output = super.forward(input)
        output = torch.squeeze(output)
        if output.dim() == 1:
            return output[:-1], output[-1]
        else:
            return output[:,:-1], output[:,-1]


class Ensemble():
    """
    Multiple models combined into an ensemble.
    The input is propagated with each model.
    """

    def __init__self(self, models):
        """
        :param models: the models of the ensemble
        """
        self.models = nn.ModuleList(models)

    def forward(self, x):
        """
        :param x: input to the models
        :return: outputs of all models
        """
        return [m(x) for m in self.models]


class ModelTrainer:
    """
    ModelTrainer optimized the parameters of a model.
    """

    def __init__(self, model, lossFunc=NegLogLikelihood, optimizer=torch.optim.Adam, lr=1e-3, lr_decay=1., batch_size=32, epochs=1):
        """
        :param model: the model to optimize
        :param lossFunc: the loss function that should be minimized
        :param optimizer: a function/constructor that returns a torch.optim optimizer
        :param lr: learn rate for the optimizer
        :param lr_decay: learn rate decay for the optimizer
        :param batch_size: the number of data points to evaluate the model on before changing parameters
        :param epochs: how often the model is trained with the same data
        """
        self.model = model
        self.lossFunc = lossFunc
        self.optimizer = optimizer(lr=lr, params=self.model.parameters())
        self.batch_size = batch_size
        self.epochs = epochs
        if lr_decay == 1.:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)
        else:
            self.scheduler = None

    def train(self, inputs, targets):
        """
        train updates the parameters of the model to minimize the loss function on inputs and targets
        the train data is shuffled for each epoch.
        :param inputs: the training inputs
        :param targets: the training targets
        """
        self.model.train()

        for e in range(self.epochs):
            permutation = torch.randperm(len(inputs))
            for batch_in, batch_t in zip(torch.chunk(inputs[permutation], self.batch_size), torch.chunk(targets[permutation], self.batch_size)):
                if len(batch_in) < 2:
                    continue
                batch_pred = self.model(batch_in)
                loss = self.lossFunc(batch_pred.float(), batch_t.float())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if not self.scheduler == None:
                    self.scheduler.step()


class EnsembleTrainer:
    """
    EnsembleTrainer is composed of multiple trainers, one for each model in the ensemble.
    """

    def __init__(self, ensemble, lossFunc=NegLogLikelihood, optimizer=torch.optim.Adam, lr=1e-3, lr_decay=1., batch_size=32):
        """
        :param model: the model to optimize
        :param lossFunc: the loss function that should be minimized
        :param optimizer: a function/constructor that returns a torch.optim optimizer
        :param lr: learn rate for the optimizer
        :param lr_decay: learn rate decay for the optimizer
        :param batch_size: the number of data points to evaluate the model on before changing parameters
        """
        self.trainers = [ModelTrainer(model, lossFunc, optimizer, lr, lr_decay, batch_size) for model in ensemble.models]

    def train(self, inputs, targets, n=0):
        """
        trains all models of the ensemble with the given inputs and targets.
        The individual models might be trained with different subsets of the data.
        :param inputs: the input data
        :param targets: target data
        :param n: unused
        """
        for trainer in self.trainers:
            samples = randint(len(inputs), size=len(inputs))
            trainer.train(inputs[samples], targets[samples])