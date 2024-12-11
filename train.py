import numpy as np
import torch
import torch.optim as optim
import yaml
from pettingzoo.butterfly import pistonball_v6
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from agent import Agent, batchify_obs, batchify, unbatchify  # Assuming Agent and helper functions are in agent.py
from Algorithms.ppo import PPO

with open('ppo_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Hyperparameters
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ent_coef = config['ent_coef']
# vf_coef = config['vf_coef']
# clip_coef = config['clip_coef']
# gamma = config['gamma']
batch_size = config['batch_size']
stack_size = config['stack_size']
max_cycles = config['max_cycles']
# total_episodes = config['total_episodes']
frame_size = (64, 64)

from datetime import datetime

current_time = datetime.now().strftime("%m%d_%H%M")

# Initialize environment
env = pistonball_v6.parallel_env(
    render_mode="rgb_array",
    continuous=False,
    max_cycles=max_cycles
)
env = color_reduction_v0(env)
env = resize_v1(env, frame_size[0], frame_size[1])
env = frame_stack_v1(env, stack_size=stack_size)

learner = PPO(env)
learner.run()
