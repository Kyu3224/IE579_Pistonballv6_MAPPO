import torch
from pettingzoo.butterfly import pistonball_v6
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from agent import Agent, batchify_obs, unbatchify  # Assuming Agent and helper functions are in agent.py

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize environment
env = pistonball_v6.parallel_env(render_mode="human", continuous=False)
env = color_reduction_v0(env)
env = resize_v1(env, 64, 64)
env = frame_stack_v1(env, stack_size=4)

agent = Agent(num_actions=env.action_space(env.possible_agents[0]).n).to(device)  # Set num_actions as per your environment
agent.load_state_dict(torch.load('./policy/1211_0705_ppo_5000iter.pt', weights_only=True))
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
