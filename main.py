import torch.nn
from mpc import MPC
from model import NN, ModelTrainer
import quanser_robots
import gym

env = gym.make('CartpoleStabShort-v0')

layers = [env.action_space.shape[0]+env.observation_space.shape[0], 25,50,100, env.observation_space.shape[0]+1]
activations = [torch.nn.ReLU]*(len(layers)-2)
model = NN(layers, activations)
trainer = ModelTrainer(model, lossFunc=torch.nn.MSELoss)

mpc = MPC(env, model, trainer, render=True)

mpc.train()