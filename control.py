import torch
import torch.distributions as distributions


def cem_optimize(init_mean, cost_func, shape, init_variance=1., samples=20, precision=1.0e-3, steps=20, nelite=5, contraint_mean=None,
                  constraint_variance=(-999999, 999999)):
    mean = init_mean
    variance = torch.tensor([init_variance]).repeat(shape)
    step = 1
    diff = 9999999
    while diff > precision and step < steps:
        dists = [distributions.MultivariateNormal(mean, torch.diagflat(var+precision/10)) for mean, var in zip(mean, variance)]
        candidates = [d.sample_n(samples) for d in dists]
        candidates = torch.stack(candidates, dim=1)
        costs = cost_func(candidates)
        # we sort descending because we want a maximum reward
        sorted_idx = torch.argsort(costs, dim=0, descending=True)
        candidates = candidates[sorted_idx]
        elite = candidates[:nelite]
        new_mean = torch.mean(elite, dim=0)
        variance = torch.var(elite, dim=0)
        diff = torch.mean(torch.abs(mean - new_mean))
        mean = new_mean
        print(mean, variance)
        if not contraint_mean is None:
            mean = clip(mean, contraint_mean[0], contraint_mean[1])
        step += 1

    return mean


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


def clip(x, min, max):
    return torch.max(torch.min(x, max), min)


class TrajectoryController():

    def __init__(self, model, action_dim, action_min, action_max, trajectory_len, history_len, cost_function):
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
        self.action_min = action_min
        self.action_max = action_max
        self.trajectory_len = trajectory_len
        self.trajectory_shape = (self.trajectory_len, self.action_dim)
        self.history_len = history_len
        self.history = FIFOBuffer(self.history_len)
        self.cost_func = cost_function
        self.trajectory = None

    def next_action(self, obs):
        #
        # history = self.history.get()
        # missing = self.history_len - len(history)
        # if missing == self.history_len:
        #     past_trajectory = torch.zeros(missing, self.action_dim)
        # elif missing > 0:
        #     past_trajectory = torch.cat((torch.stack(self.history.get()), torch.zeros(missing, self.action_dim)))
        # else:
        #     past_trajectory = torch.stack(self.history.get())
        if self.trajectory is None:
            # initialize trajectory
            self.trajectory = torch.zeros(self.trajectory_shape)
        # find a trajectory that optimizes the cummulative reward
        self.trajectory = cem_optimize(self.trajectory, self.cost_func, self.trajectory_shape, contraint_mean=[self.action_min, self.action_max], )
        best_action = self.trajectory[0]

        # self.history.push(best_action)
        return best_action

