import torch
import yaml
import argparse
from pettingzoo.butterfly import pistonball_v6
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from Algorithms.ppo.ppo_agent import Agent, batchify_obs, unbatchify  # Assuming Agent and helper functions are in agent.py

from skrl.envs.wrappers.torch import wrap_env
from Algorithms.mappo.mappo_agent import Runner

parser = argparse.ArgumentParser(description='Choose the RL algorithm')
parser.add_argument('--alg', type=str, default='ppo', choices=['mappo', 'ppo'],
                    help="Specify the RL algorithm to use: 'mappo' or 'ppo'. Default is 'ppo'.")

args = parser.parse_args()

Alg = args.alg

with open(f'./Algorithms/{Alg}/{Alg}_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

stack_size = config['env']['stack_size']

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize environment
env = pistonball_v6.parallel_env(
    n_pistons=10,
    render_mode="human",
    continuous=False)
env = color_reduction_v0(env)
env = resize_v1(env, 64, 64)
env = frame_stack_v1(env, stack_size=stack_size)

model_path ="/home/kyu/Desktop/workspace/marl_project/logs/Data/agent_80000.pt"
# model_path ="/home/kyu/Desktop/workspace/marl_project/piston_push/1217_0111_MAPPO/best_agent.pt"
# model_path= "/logs/Data/mappo_480.pt"

if Alg == 'mappo':
    env = wrap_env(env)
    runner = Runner(env, config)
    done = False
    obs = env.reset()
    while not done:
        with torch.inference_mode():
            actions = runner.agent.act(obs, timestep=0, timesteps=0)
            actions_processed = {key: torch.argmax(value).item() for key, value in actions[0].items()}
            # actions = {key: value.unsqueeze(0) for key, value in zip(list(actions.keys()), actions)}
            obs, reward, terminated, truncated, info = env.step(actions=actions_processed)

    env.close()
else:
    agent = Agent(num_actions=env.action_space(env.possible_agents[0]).n).to(device)  # Set num_actions as per your environment
    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()  # Set to evaluation mode
    # Play Loop
    with torch.no_grad():
        for episode in range(5):  # Play 5 episodes
            obs, infos = env.reset(seed=None)
            obs = batchify_obs(obs, device)
            terms = [False]
            truncs = [False]
            while not any(terms) and not any(truncs):
                actions, logprobs, _, values = agent.get_action_and_value(obs)
                obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                obs = batchify_obs(obs, device)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]

