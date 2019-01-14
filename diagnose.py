import torch.nn
import torch.distributions as distributions
import numpy as np


class ModelEval():
    """
    Uses reference dynamics and reward models to evaluate the performance of a model
    """
    def __init__(self, mpc, env, n_samples=10000):
        self.mpc = mpc
        self.env = env
        self.action_space_uniform = distributions.uniform.Uniform(torch.from_numpy(env.action_space.low), torch.from_numpy(env.action_space.high))

        self._generate_testdata(n_samples)


    def _generate_testdata(self, n_samples):
        samples = []
        done = False
        obs = self.env.reset()
        while not done and len(samples) < n_samples:
            action = self.action_space_uniform.sample()
            action = action.detach().numpy()
            next_obs, reward, done, _ = self.env.step(action)
            samples.append((obs, action, reward, next_obs, done))
            obs = next_obs
        samples = np.array(samples)
        self.test_states_in = torch.from_numpy(np.vstack(samples[:,0])).float()
        self.test_actions_in = torch.from_numpy(np.vstack(samples[:,1])).float()
        self.test_states_out = torch.from_numpy(np.vstack(samples[:,3])).float()
        self.test_rewards_out = torch.from_numpy(np.vstack(samples[:,2])).float()
        # self.test_inputs = torch.cat((states_in.float(), actions_in.float()), dim=1)
        # self.test_targets = torch.cat((states_out.float(), rewards_out.float()), dim=1)


    def eval(self):
        model = self.mpc.model
        self.model_states_out, self.model_rewards_out = model.propagate(self.test_states_in.float(), self.test_actions_in.float())
        err_dynamics = torch.mean((self.model_states_out - self.test_states_out)**2)
        err_reward = torch.mean((self.model_rewards_out - self.test_rewards_out)**2)
        print("Dynamics model error: ", err_dynamics)
        print("Reward model error: ", err_reward)
