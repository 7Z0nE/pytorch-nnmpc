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

    def next_action(self, obs,):

        past_trajectory = torch.stack(self.history.get())
        missing = self.history_len - len(past_trajectory)
        if missing > 0:
            torch.cat((past_trajectory, torch.zeros(missing, self.action_dim)))

        best_trajtectory = self._cem_optimize(past_trajectory, torch.var(past_trajectory))
        best_action = best_trajtectory[0]

        self.history.push(best_action)
        return best_action

    def _cem_optimize(self, init_mean, init_variance=torch.tensor([1.]), precision=1.0e-3, steps=20, nelite=5):
        mean = init_mean
        d = len(init_mean) > len(init_variance)
        variance = torch.cat((init_variance, torch.ones_like(d))) if d > 0 else init_variance
        step = 1
        diff = 1000
        while diff > precision and step < steps:
            dists = [distributions.MultivariateNormal(mean, var) for mean, var in zip(mean, variance)]
            candidates = torch.stack([d.sample_n(self.trajectory_len) + mean for d in dists], dim=0)
            costs = self.cost_func(candidates)
            # we sort descending because we want a maximum reward
            sorted_idx = torch.argsort(self.cost_func(costs), dim=0, descending=True)
            candidates = candidates[sorted_idx]
            elite = candidates[:nelite]
            new_mean = elite.mean()
            variance = elite.variance()
            diff = torch.abs(mean - new_mean)
            mean = new_mean
            step += 1

        return mean

