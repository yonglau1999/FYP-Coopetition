import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os
from Logistics_Service_Model import LogisticsServiceModel


# Plot profit regions

def plot_profit_regions(ax,ww):
    theta_values = np.linspace(0, 6, 100)  # Define a range of market potential (theta)
    L_s_values = np.linspace(0, 9, 100)    # Define a range of service levels for TPLP (L_s)

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

        # theta = np.random.normal(theta, 0)

        model = LogisticsServiceModel(L_s, theta,f=1)
        

        # Calculate profits for no-sharing and sharing
        profit_et_no_sharing = model.profit_nosharing_etailer()
        profit_et_sharing = model.profit_sharing_etailer(ww)
        profit_seller_no_sharing = model.profit_nosharing_seller()
        profit_seller_sharing = model.profit_sharing_seller(ww)

        # Calculate profit differences directly in the loop
        profit_diff_et.flat[idx] = profit_et_sharing - profit_et_no_sharing
        profit_diff_seller.flat[idx] = profit_seller_sharing - profit_seller_no_sharing

    # Reshape the arrays back to grid shape
    profit_diff_et = profit_diff_et.reshape(theta_grid.shape)
    profit_diff_seller = profit_diff_seller.reshape(theta_grid.shape)

    # Initialize region array (W-W = 2, W-L = 1, L-W = -1)
    region = np.zeros_like(profit_diff_et)

    # Set regions based on conditions
    region[(profit_diff_et >= -1e-8) & (profit_diff_seller >=-1e-8)] = 2  # W-W
    region[(profit_diff_et > -1e-8) & (profit_diff_seller < -1e-8)] = 1  # W-L
    region[(profit_diff_et < -1e-8) & (profit_diff_seller > -1e-8)] = -1  # L-W

    # Plot the region map
    
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
    print(model.calc_w(ww))
    print(profit_diff_seller.min())

fig,ax=plt.subplots(2)

plot_profit_regions(ax[0],True)
plot_profit_regions(ax[1],False)
plt.tight_layout()
plt.show()