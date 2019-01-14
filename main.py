import torch.nn
from mpc import MPC
from model import EnvironmentModel, ModelTrainer
import quanser_robots
import gym

env = gym.make('Pendulum-v0')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

layers = [25, 50, 100]
activations = [torch.nn.ReLU()]*(len(layers))
model = EnvironmentModel(state_dim, action_dim, layers, activations)
trainer = ModelTrainer(model, lossFunc=torch.nn.MSELoss())

mpc = MPC(env, model, trainer, render=5, trial_horizon=100)
print("Training...")
mpc.train()