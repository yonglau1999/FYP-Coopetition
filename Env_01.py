import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Creating the environment

from pettingzoo.utils import AECEnv
from api_test import api_test

from pettingzoo.utils import agent_selector

from gymnasium import spaces

# Reinforcement learning model

from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import BasicLogger

from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import OffpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.continuous import ActorProb, Critic

class LogisticsServiceModel:
    def __init__(self, L_s, X, L_e=10, phi=0.05, alpha=0.5, beta=0.7, gamma=0.5, c=0.5, f=1):
        self.L_e = L_e  # E-tailer's logistics service level    
        self.L_s = L_s  # Seller's logistics service level
        self.phi = phi  # Commission rate
        self.X = X      # Market potential (from normal distribution)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.c = c      # Variable cost of logistics for E-tailer
        self.f = f      # Seller's third-party logistics cost

    # Profit function for the no-service-sharing scenario for e-tailer
    def profit_et_no_sharing(self, theta):
        M1 = self.M1(theta)
        M2 = self.M2(theta)
        N1 = self.N1(theta)
        N2 = self.N2(theta)
        term = (1 - self.phi) * (4 - self.alpha**2 * (1 + self.phi))**2
        return (N1 * (M1 - self.c * (1 - self.phi) * (4 - self.alpha**2 * (1 + self.phi))) +
                self.phi * M2 * N2 / (1 - self.phi)) / term

    # Profit function for the no-service-sharing scenario for seller
    def profit_seller_no_sharing(self, theta):
        N2 = self.N2(theta)
        term = (1 - self.phi) * (4 - self.alpha**2 * (1 + self.phi))**2
        return (N2**2) / term

    # Profit function for the service-sharing scenario for e-tailer
    def profit_et_sharing(self, theta):
        L_e_term = self.L_e * (self.beta - self.gamma)
        denom = 4 * (1-self.alpha) * (8 + self.alpha **2 * (1-self.phi)**2 - 4 * self.phi)
        return ((theta + L_e_term - (1 - self.alpha) * self.c) **2 * (4 * (3 - self.phi) + 4 * self.alpha * (1 - self.phi) + (self.alpha **2 + self.alpha **3) * (1-self.phi)**2)) / denom

    # Profit function for the service-sharing scenario for seller
    def profit_seller_sharing(self, theta):
        L_e_term = self.L_e * (self.beta - self.gamma)
        denom = (8 + self.alpha **2 * (1 - self.phi)**2 - 4 * self.phi) **2
        return ((1 - self.phi) * ((theta + L_e_term- (1 - self.alpha) * self.c))**2 * (2 + self.alpha**2 * (1-self.phi))**2) / denom

    def M1(self, theta):
        return (1 - self.phi) * (theta * (2 + self.alpha * (1 + self.phi)) + 2 * self.c +
                                 self.L_e * (2 * self.beta - self.alpha * self.gamma * (1 + self.phi)) +
                                 self.L_s * (self.alpha * self.beta * (1 + self.phi) - 2 * self.gamma)) + self.alpha * self.f * (1 + self.phi)

    def M2(self, theta):
        return (1 - self.phi) * (theta * (2 + self.alpha) + self.alpha * self.c +
                                 self.L_e * (self.alpha * self.beta - 2 * self.gamma) +
                                 self.L_s * (2 * self.beta - self.alpha * self.gamma)) + 2 * self.f

    def N1(self, theta):
        return theta * (2 + self.alpha * (1 - self.phi) - self.alpha**2 * self.phi) + \
               self.L_e * (self.beta * (2 - self.alpha**2 * self.phi) - self.alpha * self.gamma * (1 - self.phi)) + \
               self.L_s * (self.alpha * self.beta * (1 - self.phi) + self.gamma * (self.alpha**2 * self.phi - 2)) - \
               self.c * (2 - self.alpha**2) + self.alpha * self.f

    def N2(self, theta):
        return (1 - self.phi) * (theta * (2 + self.alpha) + self.alpha * self.c +
                                 self.L_e * (self.alpha * self.beta - 2 * self.gamma) +
                                 self.L_s * (2 * self.beta - self.alpha * self.gamma)) + self.f * (self.alpha**2 * (1 + self.phi) - 2)

    def p1_no_sharing(self,theta):
        return self.M1(theta)/((1-self.phi)*(4-self.alpha**2*(1+self.phi)))
    
    def p2_no_sharing(self,theta):
        return self.M2(theta)/((1-self.phi)*(4-self.alpha**2*(1+self.phi)))  
    
    def p1_sharing(self,theta):
        top = (theta + self.L_e * (self.beta - self. gamma)) * (8 + 2 * self.alpha * (1-self.phi) - 4 * self.phi - self.alpha**2 * (1-self.phi **2)) + \
        self.c * (1-self.alpha) * (8-2*self.alpha*(1-self.phi)-4*self.phi-self.alpha **2 * (3-4*self.phi + self.phi **2))
        bottom = 2 * (1-self.alpha) * (8+2*self.alpha**2*(1-self.phi)**2-4*self.phi)
        return top/bottom
    
    def p2_sharing(self,theta):
        top = (theta + self.L_e * (self.beta - self. gamma)) * (12 - 4 * self.alpha * (1-self.phi)  + self.alpha **2 * (2-self.alpha) * (1-self.phi)**2 - 8 * self.phi) + \
        self.c * (1-self.alpha) * (4 + 4 * self.alpha * (1-self.phi) - self.alpha **3 * (1-self.phi)**2 )
        bottom = 2 * (1-self.alpha) * (8+2*self.alpha**2*(1-self.phi)**2-4*self.phi)
        return top/bottom  


    def D_seller_sharing(self,theta):
        return theta - self.p2_sharing(theta) +self.alpha * self.p1_sharing(theta) + self.beta * self.L_s - self.gamma * self.L_e
    
    def D_seller_no_sharing(self,theta):
         return theta - self.p2_no_sharing(theta) +self.alpha * self.p1_no_sharing(theta) + self.beta * self.L_e - self.gamma * self.L_e       

# Continue with the plotting logic you previously had
def plot_profit_regions():
    theta_values = np.linspace(0, 8, 100)  # Define a range of market potential (theta)
    L_s_values = np.linspace(0, 10, 100)    # Define a range of service levels for TPLP (L_s)

    # Create meshgrid to vectorize the loop
    theta_grid, L_s_grid = np.meshgrid(theta_values, L_s_values)

    # Initialize arrays to hold the profit differences directly in the loop
    profit_diff_et = np.zeros_like(theta_grid)  # Profit difference for e-tailer
    profit_diff_seller = np.zeros_like(theta_grid)  # Profit difference for seller

    # Flatten the grids to iterate
    theta_flat = theta_grid.ravel()
    L_s_flat = L_s_grid.ravel()

    # Loop over all combinations of L_s (seller service level) and theta (market potential)
    for idx in range(len(theta_flat)):
        theta = theta_flat[idx]
        L_s = L_s_flat[idx]

        X = np.random.normal(theta, 0)

        model = LogisticsServiceModel(L_s, X)

        # Calculate profits for no-sharing and sharing
        profit_et_no_sharing = model.profit_et_no_sharing(theta)
        profit_et_sharing = model.profit_et_sharing(theta)
        profit_seller_no_sharing = model.profit_seller_no_sharing(theta)
        profit_seller_sharing = model.profit_seller_sharing(theta)

        # Calculate profit differences directly in the loop
        profit_diff_et.flat[idx] = profit_et_sharing - profit_et_no_sharing
        profit_diff_seller.flat[idx] = profit_seller_sharing - profit_seller_no_sharing

    # Reshape the arrays back to grid shape
    profit_diff_et = profit_diff_et.reshape(theta_grid.shape)
    profit_diff_seller = profit_diff_seller.reshape(theta_grid.shape)

    # Initialize region array (W-W = 2, W-L = 1, L-W = -1)
    region = np.zeros_like(profit_diff_et)

    # Set regions based on conditions
    region[(profit_diff_et > 0) & (profit_diff_seller > 0)] = 2  # W-W
    region[(profit_diff_et > 0) & (profit_diff_seller < 0)] = 1  # W-L
    region[(profit_diff_et < 0) & (profit_diff_seller > 0)] = -1  # L-W

    # Plot the region map
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('coolwarm', 3)  # Using a colormap with 3 discrete levels

    # Create the contour plot for regions
    c = ax.contourf(theta_grid, L_s_grid, region, cmap=cmap, levels=[-1, 0, 1, 2], extend='both')

    # Set labels and titles
    ax.set_title("Profit Regions for E-tailer and Seller")
    ax.set_xlabel("Market Potential (Theta)")
    ax.set_ylabel("TPLP Service Level (L_s)")

    legend_labels = {
        2: 'W-W: Both Profits Positive',
        1: 'W-L: E-tailer Profit Positive, Seller Profit Negative',
        -1: 'L-W: E-tailer Profit Negative, Seller Profit Positive'
    }

    # Create custom patches for the legend
    handles = [
        mpatches.Patch(color=cmap(0), label=legend_labels[-1]),  # L-W
        mpatches.Patch(color=cmap(1), label=legend_labels[1]),  
        mpatches.Patch(color=cmap(2), label=legend_labels[2])    # W-W
    ]

    # Add a legend to the plot
    ax.legend(handles=handles, loc='upper left')

    # Add a color bar
    plt.colorbar(c, ax=ax, ticks=[-1, 1, 2], format='%d')

    plt.tight_layout()
    plt.show()


class CoopetitionEnv(AECEnv):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self):
        super(CoopetitionEnv, self).__init__()
        super().__init__()
        self.agents = ["e_tailer", "seller", "tplp"]
        self.selector = agent_selector(self.agents)
        self.possible_agents = self.agents[:]
        # self.current_agent = 0
        self.agent_selection = self.selector.reset()
        
        # State: [market_potential, L_s, f, 0 -noshare, 1- share x 2]
        self.state = np.array([10.0, 0.5, 1, 0, 0], dtype=np.float32)  # Initial state values

        # Action spaces
        self.action_spaces = {
            "e_tailer": spaces.Discrete(2),  # Shape (2,), only first dimension matters
            "seller": spaces.Discrete(2),    # Shape (2,), only first dimension matters
            "tplp": spaces.Box(low=np.array([0.0, 0.5]), high=np.array([10.0, 1]), dtype=np.float32)  # L_s and f both continuous
        }

        # Observation spaces
        self.observation_spaces = {
            "e_tailer": spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32),
            "seller": spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32),
            "tplp": spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
        }
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.model = LogisticsServiceModel(L_s=self.state[1], X=self.state[0])

    @property
    def num_agents(self) -> int:
        return int(len(self.agents))
    @property
    def max_num_agents(self) -> int:
        return int(len(self.possible_agents))
    
    def observe(self,agent):
        """Return the observation for the current agent."""
        return self.state.copy()
    
    def _clear_rewards(self):
        """Clears all items in .rewards."""
        for agent in self.rewards:
            self.rewards[agent] = 0

    def _accumulate_rewards(self):
        """Adds .rewards dictionary to ._cumulative_rewards dictionary.

        Typically called near the end of a step() method
        """
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward
    
    def reset(self, seed = None, options = None):
        self.state = np.array([10.0, 0.5, 1, 0, 0], dtype=np.float32)
        self.agents = self.possible_agents[:]
        self.agent_selection = self.possible_agents[0]
        self.dones = {agent: False for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        return {agent: self.state.copy() for agent in self.agents}

    def step(self, action):
        self._clear_rewards()
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        print(f"Current Agent: {agent}")
        
        if agent == "e_tailer":
            if action == 1:  # E-tailer agrees to share logistics
                self.state[3] = 1
            else:
                self.state[3] = 0

        # Seller makes decision to accept/reject sharing
        elif agent == "seller":
            if action == 1:  # Seller accepts logistics sharing
                self.seller_sharing_decision = True
            else:
                self.seller_sharing_decision = False
            # Set logistics sharing status in the state
            if self.state[3] == 1 and self.seller_sharing_decision:
                self.state[4] = 1  # Logistics sharing is active
            else:
                self.state[4] = 0  # No logistics sharing

        # TPLP adjusts service level and price
        elif agent == "tplp":
            # Action determines L_s and f for TPLP
            self.state[1], self.state[2] = action  # Update L_s and f

        # Calculate rewards based on the profit functions
        self.rewards[agent] = self.calculate_profit(agent)
        
        self._accumulate_rewards()

        print(self._cumulative_rewards)
        
        # Check if the episode is done (define your own termination conditions)
        self.dones = {agent: False for agent in self.agents}  # Modify as needed

        # Set info dictionary (if needed)
        self.infos = {agent: {} for agent in self.agents}

        # self._cumulative_rewards[agent] = 0
        self.agent_selection = self.selector.next()

        # return self.observe(agent), self._cumulative_rewards, False, self.dones, self.infos
    
    def calculate_profit(self, agent):
        theta = self.state[0]  # Market potential
        L_s = self.state[1]    # Seller's service level
        f = self.state[2]      # Logistics cost
        sharing_status = self.state[4]  # Logistics sharing status
        # Reinitialize the model with updated state variables
        self.model.L_s = L_s
        self.model.X = theta

        if agent == "e_tailer":
            # Use the profit function from the LogisticsServiceModel for e-tailer
            if sharing_status == 1:
                profit = self.model.profit_et_sharing(theta)
            else:
                profit = self.model.profit_et_no_sharing(theta)

        elif agent == "seller":
            # Use the profit function from the LogisticsServiceModel for seller
            if sharing_status == 1:
                profit = self.model.profit_seller_sharing(theta)
            else:
                profit = self.model.profit_seller_no_sharing(theta)

        elif agent == "tplp":

            if sharing_status == 0:
                # Use the profit function for TPLP based on the D2 formula
                c_2 = 0.5  # TPLP's logistics cost
                D_2 = self.model.D_seller_no_sharing(theta)
                profit = f * D_2 - c_2 * D_2
            else:
                profit = 0
        return profit


    def render(self, mode="human"):
        print(f"Current state: {self.state}")

    def close(self):
        pass

    def last(
        self, observe: bool = True):
        """Returns observation, cumulative reward, terminated, truncated, info for the current agent (specified by self.agent_selection)."""
        agent = self.agent_selection
        assert agent is not None
        observation = self.observe(agent) if observe else None
        print(f'{agent},{self._cumulative_rewards[agent]}')
        return (
            observation,
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

env = CoopetitionEnv()


api_test(env,num_cycles = 100)


# # env = ss.pad_action_space_v0(env)
# env = PettingZooEnv(CoopetitionEnv())
# obs = env.reset()
# env.render() 


# for i in range(4):
#     action = {"e_tailer": [0,0], "seller": [0, 1], "tplp": [6, 1]}
#     obs, reward, terminated, truncated, info = env.step(action)

# Extract the returned dictionaries

# print(obs,reward)

# Run the function to plot the profit regions
# plot_profit_regions()


