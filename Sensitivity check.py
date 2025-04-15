from Logistics_Service_Model import LogisticsServiceModel
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

# Define range of theta values (4 to 9)
theta_values = np.arange(4, 10)
f_values = np.linspace(1, 3, 10)  # Range of f values

# Iterate over different theta values and plot profit curves
for theta in theta_values:
    profit = np.zeros_like(f_values)

    for i, f in enumerate(f_values):
        env = LogisticsServiceModel(5, theta, f=f)  # Varying theta
        profit[i] = env.profit_nosharing_tplp()

    ax.plot(f_values, profit, label=f"Theta = {theta}", marker="o")

# Formatting the plot
ax.set_title("TPLP Profit vs Logistics Price (f) for Different Theta Values")
ax.set_xlabel("Logistics Price (f)")
ax.set_ylabel("TPLP Profit (No Sharing)")
ax.legend(title="Market Potential (Theta)")
ax.grid()

# Show the plot
plt.show()