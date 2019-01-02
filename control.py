import torch
import torch.distributions as distributions


class FIFOBuffer():

    def __init__(self, length):
        self.length = length
        self.buffer = []

    def push(self, elem):
        self.buffer.insert(0, elem)
        if len(self.buffer) > self.length:
            del self.buffer[-1]

    def get(self):
        return self.buffer

    def clear(self):
        self.buffer.clear()


class TrajectoryController():

    def __init__(self, model, action_dim, trajectory_len, history_len, cost_function):
        """
        A TrajectoryController finds the next best action by evaluating a trajectory of future actions.
        Future actions are evaluated using a model.
        It also takes history_len past actions into consideration.
        :param action_dim: the dimension of the action space
        :param trajectory_len: the number of future actions to look at
        :param history_len: the number of past actions to store
        :param cost_function: function (currentState, trajectory) -> expected reward that gives the cost for a trajectory
        """
        self.action_dim = action_dim
        self.trajectory_len = trajectory_len
        self.history_len = history_len
        self.history = FIFOBuffer(self.history_len)
        self.cost_func = cost_function

    def next_action(self, obs):

        history = self.history.get()
        missing = self.history_len - len(history)
        if missing == self.history_len:
            past_trajectory = torch.zeros(missing, self.action_dim)
        elif missing > 0:
            past_trajectory = torch.cat((torch.stack(self.history.get()), torch.zeros(missing, self.action_dim)))
        else:
            past_trajectory = torch.stack(self.history.get())

        best_trajtectory = self._cem_optimize(past_trajectory, torch.var(past_trajectory, dim=0))
        best_action = best_trajtectory[0]

        self.history.push(best_action)
        return best_action

    def _cem_optimize(self, init_mean, init_variance, precision=1.0e-3, steps=20, nelite=5, contraint_mean=(-999999,999999), constraint_variance=(-999999,999999)):
        mean = init_mean
        variance = torch.ones(len(mean), self.action_dim)
        step = 1
        diff = torch.tensor([1000])
        while torch.sum(diff > precision) == len(diff) and step < steps:
            dists = [distributions.MultivariateNormal(mean, torch.diagflat(var)) for mean, var in zip(mean, variance)]
            candidates = torch.stack([d.sample_n(self.trajectory_len) + mean for d in dists], dim=0)
            costs = self.cost_func(candidates)
            # we sort descending because we want a maximum reward
            sorted_idx = torch.argsort(costs, dim=0, descending=True)
            candidates = candidates[sorted_idx]
            elite = candidates[:nelite]
            new_mean = torch.mean(elite, dim=0)
            variance = torch.var(elite, dim=0)
            diff = torch.abs(mean - new_mean)
            mean = new_mean
            step += 1

        return mean

