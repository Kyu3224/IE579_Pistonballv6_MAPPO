import torch
import yaml
import argparse
import time

from pettingzoo.butterfly import pistonball_v6
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1

from Algorithms.ppo.ppo_agent import Agent, batchify_obs, unbatchify  # Assuming Agent and helper functions are in agent.py
from Algorithms.mappo.mappo_agent import Runner
from skrl.envs.wrappers.torch import wrap_env

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

# model_path ="/home/kyu/Desktop/workspace/marl_project/logs/Data/best_agent.pt"
# model_path ="/home/kyu/Desktop/workspace/marl_project/logs/Data/agent_600.pt"
# model_path= "/logs/Data/mappo_480.pt"
# model_path = "/home/kyu/Desktop/workspace/marl_project/piston_push_mappo/1220_1240_MAPPO/checkpoints/agent_4000.pt"
model_path = "/home/kyu/Desktop/workspace/marl_project/logs/Data/4950_iter.pt"

if Alg == 'mappo':
    env = wrap_env(env)
    runner = Runner(env, config)
    runner.agent.load(model_path)
    with torch.no_grad():  # inference_mode 대신 no_grad 사용
        for episode in range(30):
            print(f"Episode {episode} begins")
            time.sleep(1)
            obs = env.reset()
            done = False
            while not done:
                # act 메서드를 통해 action_logits를 얻음
                action_logits = runner.agent.act(obs, timestep=0, timesteps=0)[0]

                # actions 딕셔너리를 생성
                actions = {key: torch.tensor([torch.argmax(value).item()], device=device)
                           for key, value in action_logits.items()}

                # 환경에 actions 적용
                obs, rewards, terms, truncs, infos = env.step(actions=actions)

                # 종료 조건 업데이트
                done = any(terms.values()) or any(truncs.values())

    env.close()
else:
    agent = Agent(num_actions=env.action_space(env.possible_agents[0]).n).to(device)  # Set num_actions as per your environment
    agent.load_state_dict(torch.load(model_path, weights_only=True))
    agent.eval()  # Set to evaluation mode
    # Play Loop
    with torch.no_grad():
        for episode in range(30):  # Play 5 episodes
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

