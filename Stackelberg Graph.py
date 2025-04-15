import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Logistics_Service_Model import LogisticsServiceModel
from StackelBerg import stackelberg_game

# Define parameter ranges
theta_vals = np.linspace(4, 10, 30)
L_s_vals = np.linspace(0, 10, 30)
f_vals = np.linspace(0.5, 3, 30)

# Create meshgrid
Theta, L_s, F = np.meshgrid(theta_vals, L_s_vals, f_vals, indexing='ij')

# Prepare a mask for sharing outcome
Sharing = np.zeros(Theta.shape)

# Evaluate stackelberg_game for each combination
for i in range(Theta.shape[0]):
    for j in range(Theta.shape[1]):
        for k in range(Theta.shape[2]):
            tplp, seller = stackelberg_game(L_s[i, j, k], Theta[i, j, k], F[i, j, k])
            Sharing[i, j, k] = 1 if (tplp, seller) == (1, 1) else 0

# Extract coordinates
theta_shared, L_s_shared, f_shared = Theta[Sharing == 1], L_s[Sharing == 1], F[Sharing == 1]
theta_no_share, L_s_no_share, f_no_share = Theta[Sharing == 0], L_s[Sharing == 0], F[Sharing == 0]

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for sharing and no-sharing
ax.scatter(L_s_shared, theta_shared, f_shared, c='green', label='Sharing (1,1)', alpha=0.6)
ax.scatter(L_s_no_share, theta_no_share, f_no_share, c='red', label='No Sharing (0,0)', alpha=0.3)

# Axes labels
ax.set_xlabel('L_s (Service Level)')
ax.set_ylabel('Theta (Market Potential)')
ax.set_zlabel('f (Price Factor)')
ax.set_title('3D Region Plot: Sharing vs No Sharing')
ax.legend()
plt.tight_layout()
plt.show()