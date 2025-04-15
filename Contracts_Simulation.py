import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from PPO_LSM import CoopetitionEnv
from ray.rllib.policy.policy import Policy

e_tailer_policy = Policy.from_checkpoint("Trained_policies\\Theta_5\\e_tailer_policy")
seller_policy = Policy.from_checkpoint("Trained_policies\\Theta_5\\seller_policy")
tplp_policy = Policy.from_checkpoint("Trained_policies\\Theta_5\\tplp_policy")

def randomise_conditions():
    theta = 6
    return theta

results = []

# Define contract durations (iterations)
contract_durations = [24, 48]  # e.g., 24 months, 48 months

for duration in contract_durations:
    # Initialize environment with the specified contract duration
    env = CoopetitionEnv(theta=randomise_conditions(), max_iterations=duration)
    obs = env.reset()

    total_rewards = {"e_tailer": 0, "seller": 0, "tplp": 0}
    total_volumes = {"e_tailer": 0, "tplp": 0}  # Volume handled by e-tailer and tplp

    while not all(env.dones.values()):
        agent = env.agent_selection
        if agent == 'e_tailer':
            action = e_tailer_policy.compute_single_action(env.observe("e_tailer"),clip_actions=True,explore=False)[0]
        if agent == 'seller':
            action = seller_policy.compute_single_action(env.observe("seller"),clip_actions=True,explore=False)[0]
        if agent == 'tplp':
            action = tplp_policy.compute_single_action(env.observe("tplp"),clip_actions=True,explore=False)[0]
            action = env.action_spaces["tplp"].low + (env.action_spaces["tplp"].high - env.action_spaces["tplp"].low) * ((np.tanh(action) + 1) / 2)

        env.step(action)
        

        # Track rewards
        for agent_id, reward in env.rewards.items():
            total_rewards[agent_id] += reward

        # Track volumes
        total_volumes["e_tailer"] += (env.model.D_sharing_etailer(env.ww) +   # Volume handled by e-tailer
        total_volumes["tplp"] += env.model.D_sharing_seller(env.ww)  # Volume handled by TPLP for the seller

    # Log results for this contract duration
    results.append({
        "duration": duration,
        "total_rewards": total_rewards,
        "total_volumes": total_volumes
    })