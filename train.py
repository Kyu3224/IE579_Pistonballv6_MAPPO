import yaml
import torch

from pettingzoo.butterfly import pistonball_v6
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1

# Use skrl framework
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env

# Algorithms
from agent import Agent
from Algorithms.ppo import PPO

# Alg = 'mappo'
Alg = 'ppo'

with open(f'./Algorithms/{Alg}_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Hyperparameters for env.
stack_size = config['env']['stack_size']
max_cycles = config['env']['max_cycles']
frame_size = config['env']['frame_size']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if Alg == 'ppo':
    learner = PPO(env)
    learner.run()
elif Alg == 'mappo':
    env = wrap_env(env)

    memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=env.device)

    models = {}
    models["policy"] = Agent.actor
