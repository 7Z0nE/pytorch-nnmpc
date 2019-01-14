import torch.nn
import torch.distributions as distributions
import numpy as np


class ModelEval():
    """
    Uses reference dynamics and reward models to evaluate the performance of a model
    """
    def __init__(self, mpc, env):
        self.mpc = mpc
        self.env = env
        self._generate_testdata()
        self.action_space_uniform = distributions.uniform.Uniform(self.action_space_min, self.action_space_max)


    def _generate_testdata(self):
        samples = []
        done = False
        obs = self.env.reset()
        while not done:
            action = self.action_space_uniform.sample()
            action = action.detach().numpy()
            next_obs, reward, done, _ = self.env.step(action)
            samples.append((obs, action, reward, next_obs, done))
            obs = next_obs

        self.test_states_in = torch.from_numpy(np.vstack(self.memory[:,0]))
        self.test_actions_in = torch.from_numpy(np.vstack(self.memory[:,1]))
        self.test_states_out = torch.from_numpy(np.vstack(self.memory[:,3]))
        self.test_rewards_out = torch.from_numpy(np.vstack(self.memory[:,2]))
        # self.test_inputs = torch.cat((states_in.float(), actions_in.float()), dim=1)
        # self.test_targets = torch.cat((states_out.float(), rewards_out.float()), dim=1)


    def eval(self):
        model = self.mpc.model
        self.model_states_out, self.model_rewards_out = model.propagate(self.test_states_in, self.test_actions_in)
        err_dynamics = torch.mean((self.model_states_out - self.test_states_out)**2)
        err_reward = torch.mean((self.model_rewards_out - self.test_rewards_out)**2)
