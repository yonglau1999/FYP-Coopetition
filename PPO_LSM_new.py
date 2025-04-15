import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os
from Logistics_Service_Model import LogisticsServiceModel
from StackelBerg import stackelberg_game    

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
from ray.tune import Stopper


def randomise_theta():
    theta = 5
    return theta 

class CoopetitionEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "LogisticsServiceModel"}
    
    def __init__(self,theta,max_iterations=50):
        self.theta_values = np.arange(5, 9.5, 0.5)
        self.theta_index = 0
        super(CoopetitionEnv, self).__init__()
        self.theta = theta
        self.agents = ["tplp"]
        self._agent_selector = agent_selector(self.agents)
        self.possible_agents = self.agents[:]
        # Observation state: [market_potential, L_s, f, 0 -noshare, 1- share x 2]
        self.obstate = np.array([self.theta, 0.5, 3, 0, 0], dtype=np.float64)
        # Action spaces
        self.action_spaces = {
            "tplp": spaces.Box(low=np.array([0, 0.5]), high=np.array([10, 3]), dtype=np.float64)  # L_s and f both continuous
        }

        # Observation spaces
        self.observation_spaces = {
            "tplp": spaces.Box(low=np.array([0,0]), high=np.array([10,1]), dtype=np.float64)  # Second decision (whether sharing is active)
        }
        
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.model = LogisticsServiceModel(L_s=self.obstate[1],f=self.obstate[2], theta=self.obstate[0])
        self.terminate = False
        self.truncate = False
        self.max_iterations = max_iterations

    def next_theta(self):
        theta = self.theta_values[self.theta_index % len(self.theta_values)]
        self.theta_index += 1
        return theta

    def observation_space(self,agent):
        return self.observation_spaces[agent]

    def action_space(self,agent):
        return self.action_spaces[agent]
    
    def observe(self, agent):
        if agent == "tplp":
            obs = np.array([self.obstate[0],self.obstate[4]], dtype=np.float64)
            return obs

    def reset(self, seed = None, options = None):
    
        self.theta = randomise_theta()
        self.obstate = np.array([self.theta, 0.5, 3, 0, 0], dtype=np.float64)
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
        agent = self.agent_selection
        action = np.asarray(action)
        action = np.clip(action, self.action_spaces[agent].low, self.action_spaces[agent].high)
        

        if agent == "tplp":
            self.obstate[1], self.obstate[2] = action
            e_tailer_act,seller_act = stackelberg_game(self.obstate[1],self.obstate[0],self.obstate[2])
            if e_tailer_act and seller_act == 1:
                self.obstate[4] = 1

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

        self.ww = (profit_diff_et >= -1e-8) and (profit_diff_seller >=-1e-8)
        theta = self.state()[0]  # Market potential
        L_s = self.state()[1]    # Seller's service level
        f = self.state()[2]      # Logistics price
        sharing_status = self.state()[4]  # Logistics sharing status
    
        # Reinitialize the model with updated obstate variables
        self.model.L_s = L_s
        self.model.theta = theta

        if agent == "tplp":
            if sharing_status == 0:
                profit = self.model.profit_nosharing_tplp()
            else:
                profit = self.model.profit_sharing_tplp(self.ww)
    

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

        # Store mean reward across episode for stability checking
        mean_reward = np.mean([r for _, r in episode.agent_rewards.items()])
        episode.custom_metrics["mean_episode_reward"] = mean_reward

class PercentageVarianceStopper(Stopper):
    def __init__(self, patience=5, percentage_threshold=0.0, max_timesteps=5000000):

        self.recent_rewards = []
        self.patience = patience
        self.percentage_threshold = percentage_threshold
        self.max_timesteps = 50000 if os.environ.get("CI") else max_timesteps

    def __call__(self, trial_id, result):
        timesteps = result.get("timesteps_total", 0)
        reward = result["custom_metrics"].get("mean_episode_reward")

        # Stop if reward variance is below percentage of mean
        if reward is not None:
            self.recent_rewards.append(reward)
            if len(self.recent_rewards) > self.patience:
                self.recent_rewards.pop(0)
                mean = np.mean(self.recent_rewards)
                std = np.std(self.recent_rewards)
                if std / mean < self.percentage_threshold:
                    return True

        # Stop if timesteps exceeded
        if timesteps >= self.max_timesteps:
            return True

        return False

    def stop_all(self):
        return False

# Can change theta here
def env_creator(args):
    env = CoopetitionEnv(randomise_theta())
    return env

# Register your environment with Ray
register_env("coopetition_env_single", lambda config: PettingZooEnv(env_creator(config)))

config = (
    PPOConfig() 
    .environment(env="coopetition_env_single", clip_actions=True)
    .rollouts(num_rollout_workers=6, rollout_fragment_length='auto')
    .training(
        train_batch_size=1024,
        lr=1e-7,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.2,
        grad_clip=None,
        entropy_coeff=0.01,
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
        "tplp_policy": PolicySpec(),
    },
    policy_mapping_fn=lambda agent_id, *args, **kwargs: (
        "tplp_policy" if agent_id == "tplp" else f"{agent_id}_policy"
    ),
)


# Define the log directory
logdir = os.getcwd()

tune.run(
     "PPO", name="PPO", 
     stop={"timesteps_total": 2500000}, 
     checkpoint_freq=10, 
     storage_path=logdir, 
     config=config.to_dict(),
)