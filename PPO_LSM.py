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
from pettingzoo.test import api_test

from gymnasium import spaces
from gymnasium.envs.registration import register

# Reinforcement learning model 
from torch import nn
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray import tune
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks


def randomise_conditions():
    L_s = np.random.randint(1,10)
    theta = np.random.randint(1,8)
    return L_s,theta

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
            "tplp": spaces.Box(low=np.array([0,0]), high=np.array([1,1]), dtype=np.float64)  # Second decision (whether sharing is active)
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
            # TPLP sees only the first and second decision (whether logistics sharing is active)
            obs = np.array([self.obstate[3],self.obstate[4]], dtype=np.float64)
            assert self.observation_spaces[agent].contains(obs), f"Invalid observation for {agent}: {obs}"
            return obs
        else:
            # Return a dummy observation in case of issues
            return np.zeros(self.observation_space.shape, dtype=np.float64)

    def reset(self, seed = None, options = None):
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

        print(f"Action received for {agent}: {action}")
        print(f"Current agent is {agent}")
        
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


            print(f"Rewards: {self.rewards}")

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
        
        print(f"Cumulative rewards: {self._cumulative_rewards}")

       

    # def check_done_condition(self, agent):
    #     """ Define when the agent is 'done'. This can depend on a specific condition, like episode length or obstate. """
    #     # Example: check if market potential drops below a threshold
    #     if self.obstate[0] <= 0 or self.agent_iters[agent] == self.num_iterations:
    #         return True
    #     return False  
    
    def calculate_profit(self, agent):
        profit_et_no_sharing = self.model.profit_nosharing_etailer() 
        profit_et_sharing = self.model.profit_sharing_etailer(False)
        profit_seller_no_sharing = self.model.profit_nosharing_seller()
        profit_seller_sharing = self.model.profit_sharing_seller(False)

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

    # def agent_iter(self):
    #     while not all(self.dones.values()):
    #         yield self.agent_selection
    #         self.agent_selection = self.selector.next()

    def close(self):
        return

# Can change theta here
def env_creator(args):
    env = CoopetitionEnv(theta=4)
    return env

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
    

# Register your environment with Ray
register_env("coopetition_env", lambda config: PettingZooEnv(env_creator(config)))
ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)


config = (
    PPOConfig()
    .environment(env="coopetition_env", clip_actions=True)
    .rollouts(num_rollout_workers=4, rollout_fragment_length='auto')
    .training(
        train_batch_size=1024,
        lr=1e-6,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
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

def start_tensorboard(logdir):
    """Start TensorBoard server and open the browser."""
    try:
        # Start TensorBoard in the background
        subprocess.Popen(
            ["tensorboard", "--logdir", logdir, "--host", "localhost", "--port", "6006"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"TensorBoard started at http://localhost:6006")
        
        # Open the TensorBoard page in the default web browser
        webbrowser.open("http://localhost:6006", new=2)  # 'new=2' opens in a new tab
    except FileNotFoundError:
        print("TensorBoard is not installed. Please install it to enable visualization.")

# Define the log directory
logdir = os.path.expanduser("~/ray_results/coopetition_env")        

start_tensorboard(logdir)         

tune.run(
    "PPO",
    name="PPO",
    stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
    checkpoint_freq=10,
    storage_path=logdir,
    config=config.to_dict(),
)

# market_potential_values = np.arange(3, 4)
# reward_sums = {agent: {theta: [] for theta in market_potential_values} for agent in ["e_tailer", "seller", "tplp"]}


# for theta in market_potential_values:
#     print(f'Working on theta {theta}')
#     config.environment(env_config={"theta": theta})
#     algo = config.build()
#     # Evaluate the trained model after the iteration
#     env = env_creator({"theta": theta})  # Pass theta to env_creator
#     underlying_env = env.env
#     underlying_env.reset()

#     # Run until all agents are done
#     while not all(underlying_env.dones.values()):
        
#         # Get the current agent
#         agent = underlying_env.agent_selection
        
#         # Observation and action
#         obs = underlying_env.observe(agent)
#         action = algo.compute_single_action(obs, policy_id=agent)
#         if agent == "e_tailer" or agent == "seller":
#             action = 0
#         print(action)
#         if isinstance(underlying_env.action_spaces[agent], gymnasium.spaces.Box):
#             action = np.array(action)
#         elif isinstance(underlying_env.action_spaces[agent], gymnasium.spaces.Discrete):
#             action = int(action)

#         # Step the environment
#         _, _, _, _, _ = underlying_env.step(action)
        
#         # Train the algorithm at the end of a full iteration
#         if underlying_env.agent_selection == underlying_env.agents[0]:
#             print(f"Currently at iteration: {max(underlying_env.agent_iters.values())}")
            
#             algo.train()
#             print(f"Training step completed after {sum(underlying_env.agent_iters.values())} timesteps!")
#             for ag in underlying_env.agents:
#                 reward_sums[ag][theta].append(underlying_env.rewards[ag])

#             # Store cumulative rewards for the agent
        

#     algo.cleanup()

# # # Restore the last checkpoint for further evaluation
# # algo.restore(checkpoint)

# # Plot the rewards across iterations for each agent
# plt.figure(figsize=(10, 6))
# for agent in reward_sums.keys():
#     for theta, rewards in reward_sums[agent].items():
#         plt.plot(range(len(rewards)), rewards, label=f"{agent.capitalize()} (Theta = {theta})")

# plt.xlabel('Iteration')
# plt.ylabel('Cumulative Profit')
# plt.title('Cumulative Profit Across Training Iterations for Each Agent')
# plt.legend()
# plt.show()

