from typing import Any, Mapping, Type, Union

import torch
import torch.nn as nn
import copy
import yaml
import math
from gymnasium.spaces import Box

from skrl import logger
from skrl.agents.torch import Agent
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa
from skrl.trainers.torch import Trainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model

from Algorithms.skrl_lib.random_memory import RandomMemory
from Algorithms.skrl_lib.sequential import SequentialTrainer
from Algorithms.skrl_lib.running_std_scalar import RunningStandardScaler
from .mappo import MAPPO


class Runner:
    def __init__(self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any]) -> None:
        """Experiment runner

        Class that configures and instantiates skrl components to execute training/evaluation workflows in a few lines of code

        :param env: Environment to train on
        :param cfg: Runner configuration
        """
        self._env = env
        self._cfg = cfg

        # set random seed
        set_seed(self._cfg.get("seed", None))

        self._class_mapping = {
            # model
            "gaussianmixin": gaussian_model,
            "deterministicmixin": deterministic_model,
            "shared": shared_model,
            # memory
            "randommemory": RandomMemory,
            # agent
            "ppo": PPO,
            "ippo": IPPO,
            "mappo": MAPPO,
            # trainer
            "sequentialtrainer": SequentialTrainer,
        }

        self._cfg["agent"]["rewards_shaper"] = None  # FIXME: avoid 'dictionary changed size during iteration'

        self._models = self._generate_models(self._env, copy.deepcopy(self._cfg))
        self._agent = self._generate_agent(self._env, copy.deepcopy(self._cfg), self._models)
        self._trainer = self._generate_trainer(self._env, copy.deepcopy(self._cfg), self._agent)

    @property
    def trainer(self) -> Trainer:
        """Trainer instance
        """
        return self._trainer

    @property
    def agent(self) -> Agent:
        """Agent instance
        """
        return self._agent

    @staticmethod
    def load_cfg_from_yaml(path: str) -> dict:
        """Load a runner configuration from a yaml file

        :param path: File path

        :return: Loaded configuration, or an empty dict if an error has occurred
        """
        try:
            import yaml
        except Exception as e:
            logger.error(f"{e}. Install PyYAML with 'pip install pyyaml'")
            return {}

        try:
            with open(path) as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Loading yaml error: {e}")
            return {}

    def _class(self, value: str) -> Type:
        """Get skrl component class (e.g.: agent, trainer, etc..) from string identifier

        :return: skrl component class
        """
        if value.lower() in self._class_mapping:
            return self._class_mapping[value.lower()]
        raise ValueError(f"Unknown class '{value}' in runner cfg")

    def _process_cfg(self, cfg: dict) -> dict:
        """Convert simple types to skrl classes/components

        :param cfg: A configuration dictionary

        :return: Updated dictionary
        """
        _direct_eval = [
            "learning_rate_scheduler",
            "shared_state_preprocessor",
            "state_preprocessor",
            "value_preprocessor",
        ]

        def reward_shaper_function(scale):
            def reward_shaper(rewards, *args, **kwargs):
                return rewards * scale

            return reward_shaper

        def update_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    update_dict(value)
                else:
                    if key in _direct_eval:
                        if type(d[key]) is str:
                            d[key] = eval(value)
                    elif key.endswith("_kwargs"):
                        d[key] = value if value is not None else {}
                    elif key in ["rewards_shaper_scale"]:
                        d["rewards_shaper"] = reward_shaper_function(value)
            return d

        return update_dict(copy.deepcopy(cfg))

    def _generate_models(self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any]) -> Mapping[
        str, Mapping[str, Model]]:
        """Generate model instances according to the environment specification and the given config

        :param env: Wrapped environment
        :param cfg: A configuration dictionary

        :return: Model instances
        """
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        try:
            agent_class = self._class(cfg["agent"]["class"])
            del cfg["agent"]["class"]
        except KeyError:
            agent_class = self._class("PPO")
            logger.warning("No 'class' field defined in 'agent' cfg. 'PPO' will be used as default")

        # instantiate models
        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            # non-shared models
            if _cfg["models"]["separate"]:
                # instantiate model
                models[agent_id]["policy"] = SharedModel(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    cfg=_cfg,
                    **self._process_cfg(_cfg["models"]["policy"]),
                )
                models[agent_id]["value"] = SharedModel(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    cfg=_cfg,
                    **self._process_cfg(_cfg["models"]["policy"]),
                )
            # shared models
            else:
                # remove 'class' field
                try:
                    del _cfg["models"]["policy"]["class"]
                except KeyError:
                    logger.warning(
                        "No 'class' field defined in 'models:policy' cfg. 'GaussianMixin' will be used as default")
                try:
                    del _cfg["models"]["value"]["class"]
                except KeyError:
                    logger.warning(
                        "No 'class' field defined in 'models:value' cfg. 'DeterministicMixin' will be used as default")
                model_class = self._class("Shared")
                # instantiate model
                models[agent_id]["policy"] = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    structure=None,
                    roles=["policy", "value"],
                    parameters=[
                        self._process_cfg(_cfg["models"]["policy"]),
                        self._process_cfg(_cfg["models"]["value"]),
                    ],
                )
                models[agent_id]["value"] = models[agent_id]["policy"]

        return models

    def _generate_agent(self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any],
                        models: Mapping[str, Mapping[str, Model]]) -> Agent:
        """Generate agent instance according to the environment specification and the given config and models

        :param env: Wrapped environment
        :param cfg: A configuration dictionary
        :param models: Agent's model instances

        :return: Agent instances
        """
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        num_envs = env.num_envs
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        # check for memory configuration (backward compatibility)
        if not "memory" in cfg:
            logger.warning(
                "Deprecation warning: No 'memory' field defined in cfg. Using the default generated configuration")
            cfg["memory"] = {"class": "RandomMemory", "memory_size": -1}
        # get memory class and remove 'class' field
        try:
            memory_class = self._class(cfg["memory"]["class"])
            del cfg["memory"]["class"]
        except KeyError:
            memory_class = self._class("RandomMemory")
            logger.warning("No 'class' field defined in 'memory' cfg. 'RandomMemory' will be used as default")
        memories = {}
        # instantiate memory
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
        for agent_id in possible_agents:
            memories[agent_id] = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        # instantiate agent
        try:
            agent_class = self._class(cfg["agent"]["class"])
            del cfg["agent"]["class"]
        except KeyError:
            agent_class = self._class("PPO")
            logger.warning("No 'class' field defined in 'agent' cfg. 'PPO' will be used as default")
        # single-agent configuration and instantiation
        if agent_class in [PPO]:
            agent_id = possible_agents[0]
            agent_cfg = PPO_DEFAULT_CONFIG.copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update({"size": observation_spaces[agent_id], "device": device})
            agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
        # multi-agent configuration and instantiation
        elif agent_class in [IPPO]:
            agent_cfg = IPPO_DEFAULT_CONFIG.copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update({
                agent_id: {"size": observation_spaces[agent_id], "device": device}
                for agent_id in possible_agents
            })
            agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
            agent_kwargs = {
                "models": models,
                "memories": memories,
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
                "possible_agents": possible_agents,
            }
        elif agent_class in [MAPPO]:
            with open("Algorithms/mappo/mappo_config.yaml") as r:
                agent_cfg = yaml.safe_load(r)
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update({
                agent_id: {"size": observation_spaces[agent_id], "device": device}
                for agent_id in possible_agents
            })
            agent_cfg["shared_state_preprocessor_kwargs"].update(
                {agent_id: {"size": state_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
            agent_kwargs = {
                "models": models,
                "memories": memories,
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
                "shared_observation_spaces": state_spaces,
                "possible_agents": possible_agents,
            }
        return agent_class(cfg=agent_cfg, device=device, **agent_kwargs)

    def _generate_trainer(self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any],
                          agent: Agent) -> Trainer:
        """Generate trainer instance according to the environment specification and the given config and agent

        :param env: Wrapped environment
        :param cfg: A configuration dictionary
        :param agent: Agent's model instances

        :return: Trainer instances
        """
        # get trainer class and remove 'class' field
        try:
            trainer_class = self._class(cfg["trainer"]["class"])
            del cfg["trainer"]["class"]
        except KeyError:
            trainer_class = self._class("SequentialTrainer")
            logger.warning("No 'class' field defined in 'trainer' cfg. 'SequentialTrainer' will be used as default")
        # instantiate trainer
        return trainer_class(env=env, agents=agent, cfg=cfg["trainer"])

    def run(self, mode: str = "train") -> None:
        """Run the training/evaluation

        :param mode: Running mode: ``"train"`` for training or ``"eval"`` for evaluation (default: ``"train"``)

        :raises ValueError: The specified running mode is not valid
        """
        if mode == "train":
            self._trainer.train()
        elif mode == "eval":
            self._trainer.eval()
        else:
            raise ValueError(f"Unknown running mode: {mode}")


# define the shared model
class SharedModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, cfg, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, role="policy")
        DeterministicMixin.__init__(self, clip_actions, role="value")

        self.cfg = cfg
        # shared layers/network
        self.net = nn.Sequential(
            # First Conv Layer
            self._layer_init(nn.Conv2d(cfg['models']['local_cnn']['input_channel'],
                                       cfg['models']['local_cnn']['layers'][0],
                                       3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second Conv Layer
            self._layer_init(nn.Conv2d(cfg['models']['local_cnn']['layers'][0],
                                       cfg['models']['local_cnn']['layers'][1], 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 크기 축소: 1/4

            # Third Conv Layer
            self._layer_init(nn.Conv2d(cfg['models']['local_cnn']['layers'][1],
                                       cfg['models']['local_cnn']['layers'][2], 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 크기 축소: 1/4

            nn.AdaptiveAvgPool2d((4, 4)),  # 크기를 고정: (Batch, Channels, 4, 4)
            nn.Flatten(),  # (Batch, Channels * 4 * 4)

            self._layer_init(nn.Linear(cfg['models']['local_cnn']['layers'][2] * 4 * 4,
                                       cfg['models']['local_cnn']['output'])),  # Linear 크기 축소
            nn.ReLU(),
        )

        # 글로벌 네트워크 동일하게 변경
        self.global_net = nn.Sequential(
            # First Conv Layer
            self._layer_init(nn.Conv2d(cfg['models']['global_cnn']['input_channel'],
                                       cfg['models']['global_cnn']['layers'][0],
                                       3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second Conv Layer
            self._layer_init(nn.Conv2d(cfg['models']['global_cnn']['layers'][0],
                                       cfg['models']['global_cnn']['layers'][1],
                                       3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third Conv Layer
            self._layer_init(nn.Conv2d(cfg['models']['local_cnn']['layers'][1],
                                       cfg['models']['local_cnn']['layers'][2], 3, padding=1)),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 크기 축소: 1/4

            nn.AdaptiveAvgPool2d((4, 4)),  # 크기 고정
            nn.Flatten(),

            self._layer_init(nn.Linear(cfg['models']['global_cnn']['layers'][2] * 4 * 4,
                                       cfg['models']['global_cnn']['output'])),  # Linear 크기 축소
            nn.ReLU(),
        )

        # separated layers ("policy")
        self.mean_layer = nn.Linear(cfg['models']['local_cnn']['output'], self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        # separated layer ("value")
        self.value_layer = nn.Linear(cfg['models']['global_cnn']['output'], 1)

    # override the .act(...) method to disambiguate its call
    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    # forward the input to compute model output according to the specified role
    def compute(self, inputs, role):
        if role == "policy":
            # Add batch dimension and Change into (Batch, Channel, Height, Width)s
            # save shared layers/network output to perform a single forward-pass
            if inputs["states"].dim() == 4:
                transformed_state = inputs["states"].permute(0, 3, 1, 2)
            else:
                transformed_state = inputs["states"].unsqueeze(0).permute(0, 3, 1, 2)
            self._shared_output = self.net(transformed_state)
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            # use saved shared layers/network output to perform a single forward-pass, if it was saved
            if inputs["states"].dim() == 4:
                transformed_state = inputs["states"].permute(0, 3, 1, 2)
            else:
                transformed_state = inputs["states"].unsqueeze(0).permute(0, 3, 1, 2)

            shared_output = self.global_net(transformed_state)
            self._shared_output = None  # reset saved shared output to prevent the use of erroneous data in subsequent steps
            return self.value_layer(shared_output), {}

    def _layer_init(self, layer, std=torch.tensor(math.sqrt(2), dtype=torch.float32), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
