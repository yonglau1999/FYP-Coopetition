import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os
import subprocess
import webbrowser
from Logistics_Service_Model import LogisticsServiceModel
# Creating the environment

from pettingzoo.utils import AECEnv
from pettingzoo.utils import agent_selector

from gymnasium import spaces
from gymnasium.envs.registration import register

# Reinforcement learning model 
from torch import nn
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.algorithms.callbacks import DefaultCallbacks


def randomise_conditions():
    theta = 6
    return theta

class CoopetitionEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "LogisticsServiceModel"}
    
    def __init__(self,theta):
        super(CoopetitionEnv, self).__init__()
        self.theta = theta
        self.agents = ["e_tailer", "seller", "tplp"]
        self._agent_selector = agent_selector(self.agents)
        self.possible_agents = self.agents[:]
        # Observation state: [market_potential, L_s, f, 0 -noshare, 1- share x 2]
        self.obstate = np.array([self.theta, 0.5, 1, 0, 0], dtype=np.float64)
        # Action spaces
        self.action_spaces = {
            "e_tailer": spaces.Discrete(2),  
            "seller": spaces.Discrete(2),  
            "tplp": spaces.Box(low=np.array([0, 0.5]), high=np.array([10, 3]), dtype=np.float64)  # L_s and f both continuous
        }

        # Observation spaces
        self.observation_spaces = {
            "e_tailer": spaces.Box(low=np.array([0,0,0.5]), high=np.array([10, 10 ,3]), dtype=np.float64),  # Market potential and service level
            "seller": spaces.Box(low=np.array([0, 0,0.5]), high=np.array([10, 10 ,3]), dtype=np.float64),  # First decision (logistics sharing decision)
            "tplp": spaces.Box(low=np.array([0,0]), high=np.array([10,1]), dtype=np.float64)  # Second decision (whether sharing is active)
        }
        
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.model = LogisticsServiceModel(L_s=self.obstate[1],f=self.obstate[2], theta=self.obstate[0])
        self.terminate = False
        self.truncate = False
        self.max_iterations = 500

    def observation_space(self,agent):
        return self.observation_spaces[agent]

    def action_space(self,agent):
        return self.action_spaces[agent]
    
    def observe(self, agent):
        """Return the observation for the current agent."""
        if agent == "e_tailer":
            obs = np.array([self.obstate[0], self.obstate[1],self.obstate[2]], dtype=np.float64)
            assert self.observation_spaces[agent].contains(obs), f"Invalid observation for {agent}: {obs}"
            # E-tailer sees only market potential (obstate[0]) and service level (obstate[1])
            return obs
        elif agent == "seller":
           obs = np.array([self.obstate[0], self.obstate[1],self.obstate[2]], dtype=np.float64)
           assert self.observation_spaces[agent].contains(obs), f"Invalid observation for {agent}: {obs}"
            # Seller sees only the first decision (whether logistics sharing is agreed upon)
           return obs
        elif agent == "tplp":
            # TPLP sees only market potential (obstate[0]) and second decision (whether logistics sharing is active)
            obs = np.array([self.obstate[0],self.obstate[4]], dtype=np.float64)
            assert self.observation_spaces[agent].contains(obs), f"Invalid observation for {agent}: {obs}"
            return obs
        else:
            # Return a dummy observation in case of issues
            return np.zeros(self.observation_space.shape, dtype=np.float64)

    def reset(self, seed = None, options = None):
        self.theta = randomise_conditions()
        self.obstate = np.array([self.theta, 0.5, 1, 0, 0], dtype=np.float64)
        self.agents = self.possible_agents[:]
        self.terminate = False
        self.truncate = False
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}   
        self.observations = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.num_iterations = 0
        
        self.model = LogisticsServiceModel(L_s=self.obstate[1],f=self.obstate[2], theta=self.obstate[0])

        # Agent selector utility
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        
    def state(self):
        """Returns an observation of the global environment."""
        state = self.obstate.copy()
        return state


    def step(self, action):
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        action = np.asarray(action)
        agent = self.agent_selection
        if agent == "e_tailer":
            if action == 1:
                self.obstate[3] = 1
            else:
                self.obstate[3] = 0
        elif agent == "seller":
            if self.obstate[3] == 1 and action == 1:
                self.obstate[4] = 1
            else:
                self.obstate[4] = 0
        elif agent == "tplp":
            self.obstate[1], self.obstate[2] = action
        if self._agent_selector.is_last():

            self.model = LogisticsServiceModel(self.obstate[1], self.obstate[0], self.obstate[2])
            for agent in self.agents:
                self.rewards[agent] = self.calculate_profit(agent)
            self.num_iterations += 1
            for i in self.agents:   
                self.observations[i] = self.observe(i)
        else:
            self._clear_rewards()
        if self._agent_selector.is_last():
            self.truncate = self.num_iterations >= self.max_iterations
            self.terminate = self.num_iterations >= self.max_iterations
            self.terminations = dict(
                zip(self.agents, [self.terminate for _ in self.agents])
            )
            self.truncations = dict(
                zip(self.agents, [self.truncate for _ in self.agents])
            )
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
    
    def calculate_profit(self, agent):
        profit_et_no_sharing = self.model.profit_nosharing_etailer() 
        profit_et_sharing = self.model.profit_sharing_etailer(True)
        profit_seller_no_sharing = self.model.profit_nosharing_seller()
        profit_seller_sharing = self.model.profit_sharing_seller(True)

        # Calculate profit differences directly in the loop
        profit_diff_et = profit_et_sharing - profit_et_no_sharing
        profit_diff_seller = profit_seller_sharing - profit_seller_no_sharing

        ww = (profit_diff_et >= -1e-8) and (profit_diff_seller >=-1e-8)
        theta = self.state()[0]  # Market potential
        L_s = self.state()[1]    # Seller's service level
        f = self.state()[2]      # Logistics price
        sharing_status = self.state()[4]  # Logistics sharing status
        # Reinitialize the model with updated obstate variables
        self.model.L_s = L_s
        self.model.theta = theta

        if agent == "e_tailer":
            # Use the profit function from the LogisticsServiceModel for e-tailer
            if sharing_status == 1:
                profit = self.model.profit_sharing_etailer(ww)
            else:
                profit = self.model.profit_nosharing_etailer()

        elif agent == "seller":
            # Use the profit function from the LogisticsServiceModel for seller
            if sharing_status == 1:
                profit = self.model.profit_sharing_seller(ww)
            else:
                profit = self.model.profit_nosharing_seller()

        elif agent == "tplp":
            if sharing_status == 0:
                profit = self.model.profit_nosharing_tplp()
            else:
                profit = self.model.profit_sharing_tplp(ww)
    

        if np.isnan(profit):
            print(f"Warning: Profit for agent {agent} is NaN")
        
        return profit


    def render(self, mode="human"):
        print(f"Current obstate: {self.obstate}")


    def close(self):
        return


class CustomCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Log individual agent rewards
        for agent_id, reward in episode.agent_rewards.items():
            print(f"Agent {agent_id} reward: {reward}")

class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


# Can change theta here
def env_creator(args):
    env = CoopetitionEnv(theta=randomise_conditions())
    return env

# Register your environment with Ray
register_env("coopetition_env", lambda config: PettingZooEnv(env_creator(config)))
ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)


config = (
    PPOConfig() 
    .environment(env="coopetition_env", clip_actions=True)
    .rollouts(num_rollout_workers=6, rollout_fragment_length='auto')
    .training(
        train_batch_size=1024,
        lr=1e-6,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.2,
        grad_clip=None,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        sgd_minibatch_size=64,
        num_sgd_iter=10,
    )
    .debugging(log_level="ERROR")
    .framework(framework="torch")
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .callbacks(CustomCallbacks)
    )


config.multi_agent(
    policies={
        "e_tailer_policy": PolicySpec(),
        "seller_policy": PolicySpec(),
        "tplp_policy": PolicySpec(),
    },
    policy_mapping_fn=lambda agent_id, *args, **kwargs: (
        "tplp_policy" if agent_id == "tplp" else f"{agent_id}_policy"
    ),
)


# Define the log directory
logdir = "C:\Users\lauyo\Desktop\Y5S1\FYP Coopetition\E-commerce"

tune.run(
     "PPO", name="PPO", 
     stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000}, 
     checkpoint_freq=10, 
     storage_path=logdir, 
     config=config.to_dict(),
)