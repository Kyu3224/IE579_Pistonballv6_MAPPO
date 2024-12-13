import numpy as np
import torch
import torch.optim as optim
import os
import yaml
import wandb

from datetime import datetime
from Algorithms.ppo.ppo_agent import Agent, batchify_obs, batchify, unbatchify  # Assuming Agent and helper functions are in agent.py

class PPO:
    def __init__(self, env):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Read from Yaml file
        with open('Algorithms/ppo_config.yaml') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.stack_size = self.config['env']['stack_size']
        self.frame_size = self.config['env']['frame_size']
        self.max_cycles = self.config['env']['max_cycles']

        self.ent_coef = self.config['train']['ent_coef']
        self.vf_coef = self.config['train']['vf_coef']
        self.clip_coef = self.config['train']['clip_coef']
        self.gamma = self.config['train']['gamma']
        self.batch_size = self.config['train']['batch_size']
        self.total_episodes = self.config['train']['total_episodes']

        self.save_interval = self.config['log']['save_interval']
        self.wandb_interval = self.config['log']['wandb_interval']
        self.use_wandb = self.config['log']['use_wandb']

        self.num_agents = len(env.possible_agents)
        self.num_actions = env.action_space(env.possible_agents[0]).n
        self.observation_size = env.observation_space(env.possible_agents[0]).shape

        # Initialize Agent and optimizer
        self.agent = Agent(num_actions=self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=1e-4, eps=1e-5)
        self.env = env

        # Initialize Epsiode storage
        self.rb_obs = torch.zeros((self.max_cycles, self.num_agents, self.stack_size, *self.frame_size)).to(self.device)
        self.rb_actions = torch.zeros((self.max_cycles, self.num_agents)).to(self.device)
        self.rb_logprobs = torch.zeros((self.max_cycles, self.num_agents)).to(self.device)
        self.rb_rewards = torch.zeros((self.max_cycles, self.num_agents)).to(self.device)
        self.rb_terms = torch.zeros((self.max_cycles, self.num_agents)).to(self.device)
        self.rb_values = torch.zeros((self.max_cycles, self.num_agents)).to(self.device)

        # For Logging
        self.current_time = datetime.now().strftime("%m%d_%H%M")
        self.save_dir = f'logs/ppo/{self.current_time}'
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self):
        if self.use_wandb:
            wandb.init(project="Multi_agent_piston",
                       name=f"{self.current_time}_ppo",
                       config=self.config)

        for episode in range(self.total_episodes):
            with torch.no_grad():
                # collect observations and convert to batch of torch tensors
                next_obs, info = self.env.reset(seed=None)
                # reset the episodic return
                self.total_episodic_return = 0

                # each episode has num_steps
                for step in range(0, self.max_cycles):
                    # rollover the observation
                    obs = batchify_obs(next_obs, self.device)

                    # get action from the agent
                    actions, logprobs, _, values = self.agent.get_action_and_value(obs)

                    # execute the environment and log data
                    next_obs, rewards, terms, truncs, infos = self.env.step(
                        unbatchify(actions, self.env)
                    )

                    # add to episode storage
                    self.rb_obs[step] = obs
                    self.rb_rewards[step] = batchify(rewards, self.device)
                    self.rb_terms[step] = batchify(terms, self.device)
                    self.rb_actions[step] = actions
                    self.rb_logprobs[step] = logprobs
                    self.rb_values[step] = values.flatten()

                    # compute episodic return
                    self.total_episodic_return += self.rb_rewards[step].cpu().numpy()

                    # if we reach termination or truncation, end
                    if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                        end_step = step
                        break

            # bootstrap value if not done (GAE)
            with torch.no_grad():
                rb_advantages = torch.zeros_like(self.rb_rewards).to(self.device)
                for t in reversed(range(end_step)):
                    delta = (
                            self.rb_rewards[t]
                            + self.gamma * self.rb_values[t + 1] * self.rb_terms[t + 1]
                            - self.rb_values[t]
                    )
                    rb_advantages[t] = delta + self.gamma * self.gamma * rb_advantages[t + 1]
                rb_returns = rb_advantages + self.rb_values

            # convert our episodes to batch of individual transitions
            self.b_obs = torch.flatten(self.rb_obs[:end_step], start_dim=0, end_dim=1)
            self.b_logprobs = torch.flatten(self.rb_logprobs[:end_step], start_dim=0, end_dim=1)
            self.b_actions = torch.flatten(self.rb_actions[:end_step], start_dim=0, end_dim=1)
            self.b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
            self.b_values = torch.flatten(self.rb_values[:end_step], start_dim=0, end_dim=1)
            self.b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

            # Optimizing the policy and value network
            b_index = np.arange(len(self.b_obs))
            self.clip_fracs = []
            for repeat in range(3):
                # shuffle the indices we use to access the data
                np.random.shuffle(b_index)
                for start in range(0, len(self.b_obs), self.batch_size):
                    # select the indices we want to train on
                    end = start + self.batch_size
                    batch_index = b_index[start:end]

                    _, newlogprob, entropy, value = self.agent.get_action_and_value(
                        self.b_obs[batch_index], self.b_actions.long()[batch_index]
                    )
                    logratio = newlogprob - self.b_logprobs[batch_index]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        self.old_approx_kl = (-logratio).mean()
                        self.approx_kl = ((ratio - 1) - logratio).mean()
                        self.clip_fracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    # normalize advantages
                    advantages = self.b_advantages[batch_index]
                    advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -self.b_advantages[batch_index] * ratio
                    pg_loss2 = -self.b_advantages[batch_index] * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    self.pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    value = value.flatten()
                    v_loss_unclipped = (value - self.b_returns[batch_index]) ** 2
                    v_clipped = self.b_values[batch_index] + torch.clamp(
                        value - self.b_values[batch_index],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - self.b_returns[batch_index]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    self.v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = self.pg_loss - self.ent_coef * entropy_loss + self.v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            y_pred, y_true = self.b_values.cpu().numpy(), self.b_returns.cpu().numpy()
            var_y = np.var(y_true)
            self.explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            print(f"Training episode {episode}")
            print(f"Episodic Return: {np.mean(self.total_episodic_return)}")
            print(f"Episode Length: {end_step}")
            print("")

            self.log(episode)

            if episode % self.save_interval == 0 and episode != 0:
                self.save(episode)

        wandb.finish()

    def log(self, episode):
        print(f"Value Loss: {self.v_loss.item()}")
        print(f"Policy Loss: {self.pg_loss.item()}")
        print(f"Old Approx KL: {self.old_approx_kl.item()}")
        print(f"Approx KL: {self.approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(self.clip_fracs)}")
        print(f"Explained Variance: {self.explained_var.item()}")
        print("\n-------------------------------------------\n")
        if episode % self.wandb_interval:
            wandb.log({
                "Reward": np.mean(self.total_episodic_return),
                "Value_loss": self.v_loss.item(),
                "Policy_loss": self.pg_loss.item(),
                "Old Approx KL": self.old_approx_kl.item(),
                "Approx KL": self.approx_kl.item(),
                "Clip Fraction": self.clip_fracs,
                "Explained Variance": self.explained_var,
            })


    def save(self, episode):
        if episode == self.save_interval:
            yaml_save_path = os.path.join(self.save_dir,'ppo_config.yaml')
            with open(yaml_save_path, 'w') as f:
                yaml.dump(self.config, f)
        torch.save(self.agent.state_dict(), f'{self.save_dir}/{episode}_iter.pt')
        print(f"Model saved as '{episode}_iter.pt'.")
