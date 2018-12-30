import numpy as np
import torch
import torch.tensor as tensor
import torch.distributions as distributions
from control import TrajectoryController

class MPC():

    def __init__(self, env, model, trainer, predict_horizon=20, warmup_trials=1, learning_trials=20, trial_horizon=1000, render=False):
        self.env = env
        self.model = model
        self.trainer = trainer
        self.predict_horizon=predict_horizon
        self.warmup_trials = warmup_trials
        self.learning_trials=learning_trials
        self.trial_horizon = trial_horizon
        self.render = render
        self.memory = np.array([])
        self.action_space_dim = env.action_space.shape[0]
        self.action_space_uniform = distributions.uniform.Uniform(tensor(env.action_space.low), tensor(env.action_space.high))
        self.state = env.reset()

    def _expected_reward(self, state, action_trajectory):
        reward = 0
        for action in action_trajectory:
            state, next_reward = self.model(state, action)
            reward += next_reward
        return reward

    def _trial(self, controller, horizon=0):
        # horizon=0 means infinite horizon
        obs = self.env.reset()
        samples = []
        t = 0
        while(True):
            if self.render:
                self.env.render()
            action = controller(torch.tensor(obs))
            next_obs, reward, done, _ = self.env.step(action)
            samples.append((obs, action, reward, next_obs, done))
            t += 1
            if done or horizon > 0 and t == horizon:
                break

        self.memory = np.vstack((self.memory, np.array(samples)))

    def _train_model(self):
        states_in = torch.from_numpy(np.vstack(self.memory[:,0]))
        actions_in = torch.from_numpy(np.vstack(self.memory[:,1]))
        targets = torch.from_numpy(np.vstack(self.memory[:,3]))
        inputs = torch.cat(states_in, actions_in, dim=1)

        self.trainer.train(inputs, targets)


    def _random_controller(self, obs, n=1):
        if n <=0:
            raise("number of samples as to be greater than 0")
        if n == 1:
            return self.action_space_uniform.sample().detach().numpy()
        else:
            return self.action_space_uniform.sample((n,)).detach().numpy()

    def _trajectory_controller(self, obs):
        return self._random_controller(obs)

    def train(self):
        for k in range(self.warmup_trials):
            self._trial(self._random_controller)

        for k in range(self.learning_trials):
            self._train_model()
            self._trial(TrajectoryController(self.model, self.action_space_dim, self.predict_horizon, self.predict_horizon, self._expected_reward), self.trial_horizon)

