import yaml
import torch
import argparse
import os

from pettingzoo.butterfly import pistonball_v6
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1

# Use skrl framework
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env

# Algorithms
from Algorithms.ppo.ppo_agent import Agent
from Algorithms.ppo.ppo import PPO
from Algorithms.mappo.mappo_agent import Runner

parser = argparse.ArgumentParser(description='Choose the RL algorithm')
parser.add_argument('--alg', type=str, default='ppo', choices=['MAPPO', 'ppo'],
                    help="Specify the RL algorithm to use: 'mappo' or 'ppo'. Default is 'ppo'.")

args = parser.parse_args()

Alg = args.alg

with open(f'./Algorithms/{Alg.lower()}/{Alg.lower()}_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Hyperparameters for env.
stack_size = config['env']['stack_size']
max_cycles = config['env']['max_cycles']
frame_size = config['env']['frame_size']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from datetime import datetime

current_time = datetime.now().strftime("%m%d_%H%M")
try:
    experiment_dir = f"{config['agent']['experiment']['directory']}/{current_time}_{Alg}/{Alg.lower()}_config.yaml"
except:
    experiment_dir = f"piston_push/{current_time}_{Alg.lower()}_config.yaml"

# Initialize environment
env = pistonball_v6.parallel_env(
    render_mode="rgb_array",
    continuous=False,
    max_cycles=max_cycles
)

env = color_reduction_v0(env)
env = resize_v1(env, frame_size[0], frame_size[1])
env = frame_stack_v1(env, stack_size=stack_size)

if Alg == 'MAPPO':
    # wrap the env
    env = wrap_env(env)
    runner = Runner(env, config)
    # Save Yaml files
    with open(experiment_dir, 'w') as f:
        yaml.dump(config, f)
# Default env is set as PPO
else:
    runner = PPO(env)


runner.run()
