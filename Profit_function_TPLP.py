import numpy as np
import matplotlib.pyplot as plt

# Define the profit-sharing TPLP function
# def profit_sharing_tplp(L_s, f, commission=0.1):
#     a, b, c = 0.01, -0.12, 0.1  # Convex cost coefficients
#     excess_demand = 5  # Assume constant excess demand for simplicity
#     calc_w = 5

#     # Retained profit calculation
#     retained_profit = (1 - commission) * excess_demand * calc_w
#     cost_per_unit = a * L_s**2 + b * L_s + c
#     total_profit = retained_profit - cost_per_unit * excess_demand
#     return np.maximum(total_profit, 0)

# # Grid for L_s and f
# L_s_range = np.linspace(0, 10, 100)  # Service level from 0 to 10
# f_range = np.linspace(0, 3, 100)     # Cost of service

# # Generate profit values
# L_s_grid, f_grid = np.meshgrid(L_s_range, f_range)
# profit_grid = np.vectorize(profit_sharing_tplp)(L_s_grid, f_grid)

# # Plot contour map
# plt.figure(figsize=(10, 6))
# contour = plt.contourf(L_s_grid, f_grid, profit_grid, levels=20, cmap='viridis')
# plt.colorbar(contour, label="Profit")
# plt.xlabel("Service Level (L_s)")
# plt.ylabel("Tradeoff Parameter (f)")
# plt.title("TPLP Profit Contour Based on L_s and f")
# plt.show()

a= 0.05

# Range of L_S values (service levels)
L_s = np.arange(0, 11)  # L_S from 0 to 10

# Cost per unit computation
cost_per_unit = a*L_s
# cost_per_unit = a * L_S**2 + b * L_S + c  # Quadratic cost function

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(L_s, cost_per_unit, marker='o', linestyle='-', color='b')
plt.title('Cost per Unit vs Service Level (L_S)')
plt.xlabel('Service Level (L_S)')
plt.ylabel('Cost per Unit')
plt.grid(True)
plt.show()