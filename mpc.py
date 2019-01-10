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
        self.memory = None
        self.trajectory_controller = None
        self.action_space_dim = env.action_space.shape[0]
        self.action_space_min = torch.from_numpy(self.env.action_space.low)
        self.action_space_max = torch.from_numpy(self.env.action_space.high)
        self.action_space_uniform = distributions.uniform.Uniform(self.action_space_min, self.action_space_max)
        self.state = env.reset()

    def _expected_reward(self, action_trajectory):
        """
        :param action_trajectory: 2D tensor of actions(a trajectory) or 3D tensor of trajectories
        :return: the expected reward or a tensor of expected rewards
        """
        traj_shape = action_trajectory.shape
        reward = 0
        state = torch.from_numpy(self.state)

        # check whether the input is a batch of trajectories
        if len(traj_shape) > 2:
            state = state.repeat(len(action_trajectory), 1)
            action_trajectory = torch.transpose(action_trajectory, 0, 1) #makes iterating over it easy, as we have the n-th action of each trajectory in the same row

        for action in action_trajectory: #unsqueeze needed so action is not 0-dim
            state, next_reward = self.model.propagate(state.float(), action)
            reward += next_reward
        return reward

    def _trial(self, controller, horizon=0):
        # horizon=0 means infinite horizon
        obs = self.env.reset()
        self.state = obs
        samples = []
        t = 0
        while(True):
            if self.render:
                self.env.render()
            action = controller(torch.tensor(obs))
            print(action)
            action = action.detach().numpy()
            print(action)
            next_obs, reward, done, _ = self.env.step(action)
            samples.append((obs, action, reward, next_obs, done))
            t += 1
            if done or horizon > 0 and t == horizon:
                break
            obs = next_obs
            self.state = next_obs
        if self.memory is None:
            self.memory = np.array(samples)
        else:
            self.memory = np.vstack((self.memory, np.array(samples)))

    def _train_model(self):
        states_in = torch.from_numpy(np.vstack(self.memory[:,0]))
        actions_in = torch.from_numpy(np.vstack(self.memory[:,1]))
        states_out = torch.from_numpy(np.vstack(self.memory[:,3]))
        rewards_out = torch.from_numpy(np.vstack(self.memory[:,2]))
        inputs = torch.cat((states_in.float(), actions_in.float()), dim=1)
        targets = torch.cat((states_out.float(), rewards_out.float()), dim=1)

        self.trainer.train(inputs, targets)


    def _random_controller(self, obs, n=1):
        if n <=0:
            raise("number of samples as to be greater than 0")
        if n == 1:
            return self.action_space_uniform.sample()
        else:
            return self.action_space_uniform.sample((n,))

    def _trajectory_controller(self, obs):
        if self.trajectory_controller == None:
            self.trajectory_controller = TrajectoryController(self.model, self.action_space_dim, self.action_space_min, self.action_space_max, self.predict_horizon, self.predict_horizon, self._expected_reward)
        self.trajectory_controller.cost_func = self._expected_reward
        return self.trajectory_controller.next_action(obs)

    def train(self):
        for k in range(self.warmup_trials):
            self._trial(self._random_controller)

        for k in range(self.learning_trials):

            self._train_model()
            self._trial(self._trajectory_controller, self.trial_horizon)

