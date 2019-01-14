import torch.nn
from mpc import MPC
from model import EnvironmentModel, EnvironmentModelSeparateReward, ModelTrainer
import quanser_robots
import gym
import torch.cuda as cuda


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('Pendulum-v0')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

layers = [25, 50, 100]
activations = [torch.nn.ReLU()]*(len(layers))
model = EnvironmentModelSeparateReward(state_dim, action_dim, layers, activations).to(dev)
trainer = ModelTrainer(model, lossFunc=torch.nn.MSELoss(), epochs=10)

mpc = MPC(env, model, trainer, render=5, trial_horizon=100, device=dev)
print("Training...")
mpc.train(diagnose=True)
