import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# Run the function to plot the profit regions
plot_profit_regions()