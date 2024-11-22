import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize

# Creating the environment

from pettingzoo.utils import AECEnv
from pettingzoo.utils import agent_selector
from api_test import api_test

from gymnasium import spaces
import gymnasium
from gymnasium.envs.registration import register

# Reinforcement learning model torch
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
import ray
from ray import air 
from ray import tune
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
import torch

class LogisticsServiceModel:
    def __init__(self, L_s, theta, L_e, phi, alpha=0.5, beta=0.7, gamma=0.5, c=0.5, f=1):
        self.L_e = L_e  # E-tailer's logistics service level    
        self.L_s = L_s  # Seller's logistics service level
        self.phi = phi  # Commission rate
        self.theta = theta      # Market potential (from normal distribution)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.c = c      # Variable cost of logistics for E-tailer
        self.f = f      # Seller's third-party logistics cost

    # Profit function for the no-service-sharing scenario for e-tailer
    def profit_et_no_sharing(self):
        M1 = self.M1()
        M2 = self.M2()
        N1 = self.N1()
        N2 = self.N2()
        term = (1 - self.phi) * (4 - self.alpha**2 * (1 + self.phi))**2
        return (N1 * (M1 - self.c * (1 - self.phi) * (4 - self.alpha**2 * (1 + self.phi))) +
                self.phi * M2 * N2 / (1 - self.phi)) / term

    # Profit function for the no-service-sharing scenario for seller
    def profit_seller_no_sharing(self):
        N2 = self.N2()
        term = (1 - self.phi) * (4 - self.alpha**2 * (1 + self.phi))**2
        return (N2**2) / term

    # Profit function for the service-sharing scenario for e-tailer
    def profit_et_sharing(self):
        L_e_term = self.L_e * (self.beta - self.gamma)
        denom = 4 * (1-self.alpha) * (8 + self.alpha **2 * (1-self.phi)**2 - 4 * self.phi)
        return ((self.theta + L_e_term - (1 - self.alpha) * self.c) **2 * (4 * (3 - self.phi) + 4 * self.alpha * (1 - self.phi) + (self.alpha **2 + self.alpha **3) * (1-self.phi)**2)) / denom

    # Profit function for the service-sharing scenario for seller
    def profit_seller_sharing(self):
        L_e_term = self.L_e * (self.beta - self.gamma)
        denom = (8 + self.alpha **2 * (1 - self.phi)**2 - 4 * self.phi) **2
        return ((1 - self.phi) * ((theta + L_e_term- (1 - self.alpha) * self.c))**2 * (2 + self.alpha**2 * (1-self.phi))**2) / denom

    def M1(self):
        return (1 - self.phi) * (self.theta * (2 + self.alpha * (1 + self.phi)) + 2 * self.c +
                                 self.L_e * (2 * self.beta - self.alpha * self.gamma * (1 + self.phi)) +
                                 self.L_s * (self.alpha * self.beta * (1 + self.phi) - 2 * self.gamma)) + self.alpha * self.f * (1 + self.phi)

    def M2(self):
        return (1 - self.phi) * (self.theta * (2 + self.alpha) + self.alpha * self.c +
                                 self.L_e * (self.alpha * self.beta - 2 * self.gamma) +
                                 self.L_s * (2 * self.beta - self.alpha * self.gamma)) + 2 * self.f

    def N1(self):
        return self.theta * (2 + self.alpha * (1 - self.phi) - self.alpha**2 * self.phi) + \
               self.L_e * (self.beta * (2 - self.alpha**2 * self.phi) - self.alpha * self.gamma * (1 - self.phi)) + \
               self.L_s * (self.alpha * self.beta * (1 - self.phi) + self.gamma * (self.alpha**2 * self.phi - 2)) - \
               self.c * (2 - self.alpha**2) + self.alpha * self.f

    def N2(self):
        return (1 - self.phi) * (self.theta * (2 + self.alpha) + self.alpha * self.c +
                                 self.L_e * (self.alpha * self.beta - 2 * self.gamma) +
                                 self.L_s * (2 * self.beta - self.alpha * self.gamma)) + self.f * (self.alpha**2 * (1 + self.phi) - 2)

    def p1_no_sharing(self):
        return self.M1(self.theta)/((1-self.phi)*(4-self.alpha**2*(1+self.phi)))
    
    def p2_no_sharing(self):
        return self.M2(self.theta)/((1-self.phi)*(4-self.alpha**2*(1+self.phi)))  
    
    def p1_sharing(self):
        top = (self.theta + self.L_e * (self.beta - self. gamma)) * (8 + 2 * self.alpha * (1-self.phi) - 4 * self.phi - self.alpha**2 * (1-self.phi **2)) + \
        self.c * (1-self.alpha) * (8-2*self.alpha*(1-self.phi)-4*self.phi-self.alpha **2 * (3-4*self.phi + self.phi **2))
        bottom = 2 * (1-self.alpha) * (8+2*self.alpha**2*(1-self.phi)**2-4*self.phi)
        return top/bottom
    
    def p2_sharing(self):
        top = (self.theta + self.L_e * (self.beta - self. gamma)) * (12 - 4 * self.alpha * (1-self.phi)  + self.alpha **2 * (2-self.alpha) * (1-self.phi)**2 - 8 * self.phi) + \
        self.c * (1-self.alpha) * (4 + 4 * self.alpha * (1-self.phi) - self.alpha **3 * (1-self.phi)**2 )
        bottom = 2 * (1-self.alpha) * (8+2*self.alpha**2*(1-self.phi)**2-4*self.phi)
        return top/bottom  


    def D_seller_sharing(self):
        return self.theta - self.p2_sharing(self.theta) +self.alpha * self.p1_sharing(self.theta) + self.beta * self.L_s - self.gamma * self.L_e
    
    def D_seller_no_sharing(self):
         return self.theta - self.p2_no_sharing(self.theta) +self.alpha * self.p1_no_sharing(self.theta) + self.beta * self.L_e - self.gamma * self.L_e       

# Continue with the plotting logic you previously had
def plot_profit_regions(ax,L_e,phi):
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

        theta = np.random.normal(theta, 0)

        model = LogisticsServiceModel(L_s, theta,L_e,phi)

        # Calculate profits for no-sharing and sharing
        profit_et_no_sharing = model.profit_et_no_sharing()
        profit_et_sharing = model.profit_et_sharing()
        profit_seller_no_sharing = model.profit_seller_no_sharing()
        profit_seller_sharing = model.profit_seller_sharing()

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
    
    cmap = plt.get_cmap('coolwarm', 3)  # Using a colormap with 3 discrete levels

    # Create the contour plot for regions
    c = ax.contourf(theta_grid, L_s_grid, region, cmap=cmap, levels=[-1, 0, 1, 2], extend='both')

    # # Set labels and titles
    # ax.set_title("Profit Regions for E-tailer and Seller")
    # ax.set_xlabel("Market Potential (Theta)")
    # ax.set_ylabel("TPLP Service Level (L_s)")

    # legend_labels = {
    #     2: 'W-W: Both Profits Positive',
    #     1: 'W-L: E-tailer Profit Positive, Seller Profit Negative',
    #     -1: 'L-W: E-tailer Profit Negative, Seller Profit Positive'
    # }

    # # Create custom patches for the legend
    # handles = [
    #     mpatches.Patch(color=cmap(0), label=legend_labels[-1]),  # L-W
    #     mpatches.Patch(color=cmap(1), label=legend_labels[1]),  
    #     mpatches.Patch(color=cmap(2), label=legend_labels[2])    # W-W
    # ]

    # # Add a legend to the plot
    # ax.legend(handles=handles, loc='upper left')

    # # Add a color bar
    # plt.colorbar(c, ax=ax, ticks=[-1, 1, 2], format='%d')

def randomise_conditions():
    L_s = np.random.randint(1,10)
    theta = np.random.randint(1,8)
    return L_s,theta

def max_sharing_profit(L_e_phi):
    L_e, phi = L_e_phi

    # Initialize the model with the given parameters
    model = LogisticsServiceModel(L_s, theta, L_e=L_e, phi=phi)
    
    # Calculate profits
    profit_et_no_sharing = model.profit_et_no_sharing()
    profit_et_sharing = model.profit_et_sharing()
    profit_seller_no_sharing = model.profit_seller_no_sharing()
    profit_seller_sharing = model.profit_seller_sharing()

    # Profit differences
    profit_diff_et = profit_et_sharing - profit_et_no_sharing
    profit_diff_seller = profit_seller_sharing - profit_seller_no_sharing

    # Objective: Maximize the sum (negative for minimization)
    return -(profit_diff_et + profit_diff_seller)

# Define constraints: both profit differences must be positive
def constraint_et(L_e_phi):
    L_e, phi = L_e_phi

    model = LogisticsServiceModel(L_s, theta, L_e=L_e, phi=phi)
    profit_et_no_sharing = model.profit_et_no_sharing()
    profit_et_sharing = model.profit_et_sharing()
    return profit_et_sharing - profit_et_no_sharing  # Must be >= 0

def constraint_seller(L_e_phi):
    L_e, phi = L_e_phi

    model = LogisticsServiceModel(L_s, theta, L_e=L_e, phi=phi)
    profit_seller_no_sharing = model.profit_seller_no_sharing()
    profit_seller_sharing = model.profit_seller_sharing()
    return profit_seller_sharing - profit_seller_no_sharing  # Must be >= 0

def dynamic_scenario(ax,theta,L_s):
    ax.plot(theta,L_s,'rx',label="Initial Point")

successful_results = []

for i in range (3,9):
    for j in range(3,11):

    # Randomize L_s and theta
        L_s, theta = j,i
        print(f"Attempt {(i+1)*j}: L_s={L_s}, theta={theta}")

        # Set bounds for L_e and phi
        bounds = [(0.1, 10), (0.01, 1)]  # Example ranges for L_e and phi

        # Set initial guess
        initial_guess = [5, 0.05]  # Starting point for L_e and phi

        # Define the constraints
        constraints = [
            {"type": "ineq", "fun": constraint_et},      # profit_diff_et >= 0
            {"type": "ineq", "fun": constraint_seller},  # profit_diff_seller >= 0
        ]

        # Run the optimization
        result = minimize(
            max_sharing_profit,
            initial_guess,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP"
        )

        # Store successful results
        if result.success:
            optimal_L_e, optimal_phi = result.x
            print(f"Optimal L_e: {optimal_L_e:.2f}, Optimal phi: {optimal_phi:.2f}, Max profit diff: {-result.fun:.2f}")
            successful_results.append((L_s, theta, optimal_L_e, optimal_phi))
        else:
            print("Optimization failed:", result.message)

# Create subplots dynamically based on successful results
n_plots = len(successful_results)

if n_plots > 0:
    # Dynamically determine rows and columns
    n_cols = 4  # Fixed number of columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Calculate the required rows

    # Create a gridspec layout
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 5))
    spec = gridspec.GridSpec(n_rows, n_cols + 1, figure=fig, width_ratios=[1] * n_cols + [0.1])

    # Create subplots within the gridspec
    axes = [fig.add_subplot(spec[i // n_cols, i % n_cols]) for i in range(n_plots)]

    # Shared colormap and normalization
    cmap = plt.get_cmap('coolwarm', 3)
    norm = plt.Normalize(vmin=-1, vmax=2)

    for idx, (ax, (L_s, theta, optimal_L_e, optimal_phi)) in enumerate(zip(axes, successful_results)):
        # Plot the profit regions
        plot_profit_regions(ax, optimal_L_e, optimal_phi)
        ax.plot(theta, L_s,'x',color='#AAFF32', markersize=15)  # Larger cross size

        # Add labels for L_s and theta within each plot
        ax.text(6, 8.5, f"L_s={L_s}, theta={theta},optimal_L_e={optimal_L_e:.2f}, optimal_phi={optimal_phi:.2f}", color='black', fontsize=8,
                ha='center', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # Add titles for each plot
        ax.set_title(f"Plot {idx+1}", fontsize=10)

    # Hide unused axes
    for ax in axes[n_plots:]:
        ax.axis('off')

    # Position x-axis label at the middle column, lowest row
    last_row_start = (n_rows - 1) * n_cols
    last_row_axes = axes[last_row_start:n_plots]  # Subplots in the last row

    # Position x-axis label at the middle column, lowest row
    if last_row_axes:  # Ensure there are subplots in the last row
        middle_col_ax = last_row_axes[len(last_row_axes) // 2]  # Middle subplot in the last row
        middle_col_ax.set_xlabel("Market Potential (Theta)", fontsize=14, labelpad=10)


    # Position y-axis label at the middle row, beside the first column
    if n_rows > 1:  # Ensure there are enough rows
        middle_row_start = (n_rows // 2) * n_cols
        if middle_row_start < len(axes):  # Check if the middle row exists
            middle_row_ax = axes[middle_row_start]
            middle_row_ax.set_ylabel("TPLP Service Level (L_s)", fontsize=14, labelpad=50)

    # Add a shared legend above the first row
    legend_labels = {
        2: 'W-W: Both Profits Positive',
        1: 'W-L: E-tailer Profit Positive, Seller Profit Negative',
        -1: 'L-W: E-tailer Profit Negative, Seller Profit Positive'
    }
    legend_handles = [
        mpatches.Patch(color=cmap(0), label=legend_labels[-1]),  # L-W
        mpatches.Patch(color=cmap(1), label=legend_labels[1]),
        mpatches.Patch(color=cmap(2), label=legend_labels[2])    # W-W
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.98))

    # Adjust subplot spacing
    plt.subplots_adjust(left=0.08, right=0.9, top=0.9, bottom=0.1, wspace=0.6, hspace=1)
    plt.show()
else:
    print("No successful results to display.")