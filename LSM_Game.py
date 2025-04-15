import pygame
import numpy as np
from ray.rllib.policy.policy import Policy
from Logistics_Service_Model import LogisticsServiceModel
from gymnasium import spaces
import matplotlib.pyplot as plt
from StackelBerg import stackelberg_game   

# Load trained policies

tplp_policy = Policy.from_checkpoint("PPO\\Theta_5_2_1\\checkpoint_000243\\policies\\tplp_policy")


# Initialize Pygame
pygame.init()
theta_init = 5
# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FONT_SIZE = 24
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BUTTON_COLOR = (100, 200, 100)
SLIDER_COLOR = (200, 200, 200)
KNOB_COLOR = (50, 150, 250)

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Human-in-the-Loop TPLP Game")
font = pygame.font.Font(None, FONT_SIZE)

# Environment class
class HumanInTheLoopEnv:
    def __init__(self, theta=theta_init):
        self.agents = ["e_tailer", "seller", "tplp"]
        self.theta = theta
        self.state = np.array([self.theta, 0.5, 1, 0, 0])
        self.rewards = {agent: 0 for agent in self.agents}
        self.cumulative_rewards = {agent: 0 for agent in self.agents}
        self.model = LogisticsServiceModel(self.state[1], self.state[0], self.state[2])
        self.action_spaces = {
            "e_tailer": spaces.Discrete(2),  
            "seller": spaces.Discrete(2),  
            "tplp": spaces.Box(low=np.array([0, 0.5]), high=np.array([10, 3]), dtype=np.float64)  # L_s and f both continuous
        }

    def reset(self):
        self.state = np.array([self.theta, 0.5, 1, 0, 0])
        self.rewards = {agent: 0 for agent in self.agents}
        self.cumulative_rewards = {agent: 0 for agent in self.agents}
        return self.state
    
    def observe(self, agent):
        if agent == "tplp":
            obs = np.array([self.obstate[0],self.obstate[4]], dtype=np.float64)

            return obs

    def step(self, tplp_action):
        
        self.state[1] = tplp_action["L_s"]
        self.state[2] = tplp_action["f"]
        e_tailer_action,seller_action = stackelberg_game(self.state[1],self.state[0],self.state[2])
        if e_tailer_action and seller_action == 1:
            self.state[4] = 1
        else:
            self.state[4] = 0

        self.model = LogisticsServiceModel(self.state[1], self.state[0], self.state[2])
        self.rewards = self.calculate_rewards()
        for agent in self.agents:
            self.cumulative_rewards[agent] += self.rewards[agent]
        print(self.state[4])
        return self.state, self.rewards

    def calculate_rewards(self):
        e_tailer_reward = self.calculate_profit("e_tailer")
        seller_reward = self.calculate_profit("seller")
        tplp_reward = self.calculate_profit("tplp")
        return {
            "e_tailer": e_tailer_reward,
            "seller": seller_reward,
            "tplp": tplp_reward,
        }

    def calculate_profit(self, agent):
        profit_et_no_sharing = self.model.profit_nosharing_etailer() 
        profit_et_sharing = self.model.profit_sharing_etailer(True)
        profit_seller_no_sharing = self.model.profit_nosharing_seller()
        profit_seller_sharing = self.model.profit_sharing_seller(True)

        # Calculate profit differences directly in the loop
        profit_diff_et = profit_et_sharing - profit_et_no_sharing
        profit_diff_seller = profit_seller_sharing - profit_seller_no_sharing

        self.ww = (profit_diff_et >= -1e-8) and (profit_diff_seller >=-1e-8)
        theta = self.state[0]  # Market potential
        L_s = self.state[1]    # Seller's service level
        f = self.state[2]      # Logistics price
        sharing_status = self.state[4]  # Logistics sharing status
    
        # Reinitialize the model with updated obstate variables
        self.model.L_s = L_s
        self.model.theta = theta

        if agent == "tplp":
            if sharing_status == 0:
                profit = self.model.profit_nosharing_tplp()
            else:
                profit = self.model.profit_sharing_tplp(self.ww)
            
            print(profit)
    
        if agent == "e_tailer":
            if sharing_status == 0:
                profit = self.model.profit_nosharing_etailer()
            else:
                profit = self.model.profit_sharing_etailer(self.ww)
        
        elif agent == "seller":
            if sharing_status == 0:
                profit = self.model.profit_nosharing_seller()
            else:
                profit = self.model.profit_sharing_seller(self.ww)
            
        return profit
        
    def plot_profit_regions(self,ax, theta_value, sharing_status):
        # Define range for L_s and f
        L_s_values = np.linspace(1, 10, 100)
        f_values = np.linspace(0.5, 3, 100)

        profit_matrix = np.zeros((len(L_s_values), len(f_values)))

        # Compute profits
        for i, L_s in enumerate(L_s_values):
            for j, f in enumerate(f_values):
                model = LogisticsServiceModel(L_s, theta_value, f)
                ww = (sharing_status == 1)  # sharing_status flag
                profit_matrix[i, j] = model.profit_sharing_tplp(ww) if ww else model.profit_nosharing_tplp()

        # Plot profit regions
        c = ax.contourf(f_values, L_s_values, profit_matrix, cmap='viridis', levels=50)
        ax.set_xlabel("Logistics Price (f)")
        ax.set_ylabel("Service Level (L_s)")
        ax.set_title(f"TPLP Profit Regions - {'Sharing' if sharing_status == 1 else 'No Sharing'}")
        plt.colorbar(c, ax=ax, label="Profit")

class Machine:
    def __init__(self, theta=theta_init):
        self.agents = ["e_tailer", "seller", "tplp"]
        self.theta = theta
        self.state = np.array([self.theta, 0.5, 1, 0, 0])
        self.rewards = {agent: 0 for agent in self.agents}
        self.cumulative_rewards = {agent: 0 for agent in self.agents}
        self.model = LogisticsServiceModel(self.state[1], self.state[0], self.state[2])
        self.action_spaces = {
            "e_tailer": spaces.Discrete(2),  
            "seller": spaces.Discrete(2),  
            "tplp": spaces.Box(low=np.array([0, 0.5]), high=np.array([10, 3]), dtype=np.float64)  # L_s and f both continuous
        }

    def reset(self):
        self.state = np.array([self.theta, 0.5, 1, 0, 0])
        self.rewards = {agent: 0 for agent in self.agents}
        self.cumulative_rewards = {agent: 0 for agent in self.agents}
        return self.state
    
    def observe(self, agent):
        if agent == "e_tailer":
            return np.array([self.state[0], self.state[1], self.state[2]], dtype=np.float64)
        elif agent == "seller":
            return np.array([self.state[0], self.state[1], self.state[2]], dtype=np.float64)
        elif agent == "tplp":
            return np.array([self.state[0], self.state[4]], dtype=np.float64)

    def step(self):

        tplp_action = tplp_policy.compute_single_action(self.observe("tplp"),clip_actions=True,explore=False)[0]
        tplp_action = self.action_spaces["tplp"].low + (self.action_spaces["tplp"].high - self.action_spaces["tplp"].low) * ((np.tanh(tplp_action) + 1) / 2)
        print(tplp_action)
        self.state[1] = tplp_action[0]
        self.state[2] = tplp_action[1]
        
        e_tailer_action,seller_action = stackelberg_game(self.state[1],self.state[0],self.state[2])
        if e_tailer_action and seller_action == 1:
            self.state[4] = 1
        else:
            self.state[4] = 0
        self.model = LogisticsServiceModel(self.state[1], self.state[0], self.state[2])
        self.rewards = self.calculate_rewards()

        for agent in self.agents:
            self.cumulative_rewards[agent] += self.rewards[agent]
        print(self.state[4])
        return self.state, self.rewards

    def calculate_rewards(self):
        e_tailer_reward = self.calculate_profit("e_tailer")
        seller_reward = self.calculate_profit("seller")
        tplp_reward = self.calculate_profit("tplp")
        return {
            "e_tailer": e_tailer_reward,
            "seller": seller_reward,
            "tplp": tplp_reward,
        }

    def calculate_profit(self, agent):
        profit_et_no_sharing = self.model.profit_nosharing_etailer() 
        profit_et_sharing = self.model.profit_sharing_etailer(True)
        profit_seller_no_sharing = self.model.profit_nosharing_seller()
        profit_seller_sharing = self.model.profit_sharing_seller(True)

        # Calculate profit differences directly in the loop
        profit_diff_et = profit_et_sharing - profit_et_no_sharing
        profit_diff_seller = profit_seller_sharing - profit_seller_no_sharing

        self.ww = (profit_diff_et >= -1e-8) and (profit_diff_seller >=-1e-8)
        theta = self.state[0]  # Market potential
        L_s = self.state[1]    # Seller's service level
        f = self.state[2]      # Logistics price
        sharing_status = self.state[4]  # Logistics sharing status
    
        # Reinitialize the model with updated obstate variables
        self.model.L_s = L_s
        self.model.theta = theta

        if agent == "tplp":
            if sharing_status == 0:
                profit = self.model.profit_nosharing_tplp()
            else:
                profit = self.model.profit_sharing_tplp(self.ww)

            print(f"machine:{profit}")
    
        if agent == "e_tailer":
            if sharing_status == 0:
                profit = self.model.profit_nosharing_etailer()
            else:
                profit = self.model.profit_sharing_etailer(self.ww)
        
        elif agent == "seller":
            if sharing_status == 0:
                profit = self.model.profit_nosharing_seller()
            else:
                profit = self.model.profit_sharing_seller(self.ww)
            
        return profit
    
# Helper functions
def draw_text(surface, text, x, y):
    text_surface = font.render(text, True, WHITE)
    surface.blit(text_surface, (x, y))

def draw_button(surface, x, y, width, height, text):
    """Draw a button with text."""
    pygame.draw.rect(surface, BUTTON_COLOR, (x, y, width, height))
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    surface.blit(text_surface, text_rect)

def draw_slider(surface, x, y, width, min_val, max_val, step, value):
    # Draw the slider background
    pygame.draw.rect(surface, SLIDER_COLOR, (x, y, width, 10))

    # Map the current value to the slider's knob position
    knob_x = x + int((value - min_val) / (max_val - min_val) * width)

    # Draw the knob
    pygame.draw.circle(surface, KNOB_COLOR, (knob_x, y + 5), 10)

    return knob_x

def slider_value(mouse_x, x, width, min_val, max_val, step):
    # Calculate the relative position of the mouse within the slider
    relative_pos = min(max(mouse_x - x, 0), width)  # Ensure within slider bounds

    # Calculate the slider value based on the relative position
    value_range = max_val - min_val
    value = min_val + (relative_pos / width) * value_range

    # Round to the nearest step
    value = round(value / step) * step

    # Clip to ensure the value is within the valid range
    return np.clip(value, min_val, max_val)

# Initialize environment
env = HumanInTheLoopEnv()
env_machine = Machine()
state = env.reset()

# Slider properties
slider_width = 200
slider_x = 400
slider_L_s_y = 50
slider_f_y = 150
slider_L_s_value = 0
slider_f_value = 0.5

# Button properties
button_x = 400
button_y = 500
button_width = 150
button_height = 50

input_active_L_s = False
input_active_f = False
input_text_L_s = "1.0"
input_text_f = "0.5"
input_box_L_s = pygame.Rect(slider_x, slider_L_s_y, 100, 32)
input_box_f = pygame.Rect(slider_x, slider_f_y, 100, 32)
color_inactive = pygame.Color('lightskyblue3')
color_active = pygame.Color('dodgerblue2')
color_L_s = color_inactive
color_f = color_inactive

# Main game loop
running = True
iterations = 0

while running:
    screen.fill(BLACK)

    # Display current state
    draw_text(screen, f"Iteration number: {iterations}", 50, 0)
    draw_text(screen, f"Theta: {theta_init}", 50, 30)
    draw_text(screen, f"Service Level (L_s): {state[1]:.2f}", 50, 60)
    draw_text(screen, f"Logistics Price (f): {state[2]:.2f}", 50, 90)
    draw_text(screen, f"E-tailer Sharing: {state[3]}", 50, 120)
    draw_text(screen, f"Seller Sharing: {state[4]}", 50, 150)

    # L_s input box
    draw_text(screen, "Enter Service Level (L_s):", slider_x, slider_L_s_y - 30)
    color_L_s = color_active if input_active_L_s else color_inactive
    pygame.draw.rect(screen, color_L_s, input_box_L_s, 2)
    txt_surface = font.render(input_text_L_s, True, WHITE)
    screen.blit(txt_surface, (input_box_L_s.x + 5, input_box_L_s.y + 5))

    # f input box
    draw_text(screen, "Enter Logistics Price (f):", slider_x, slider_f_y - 30)
    color_f = color_active if input_active_f else color_inactive
    pygame.draw.rect(screen, color_f, input_box_f, 2)
    txt_surface = font.render(input_text_f, True, WHITE)
    screen.blit(txt_surface, (input_box_f.x + 5, input_box_f.y + 5))

    # Draw the submit button
    draw_button(screen, button_x, button_y, button_width, button_height, "Submit")

    # Display rewards
    draw_text(screen, f"Rewards:", 50, 200)
    for i, (agent, reward) in enumerate(env.rewards.items()):
        draw_text(screen, f"{agent}: {reward:.2f}", 50, 230 + i * 30)

    # Display cumulative rewards
    draw_text(screen, f"Cumulative Rewards You:", 50, 320)
    for i, (agent, cum_reward) in enumerate(env.cumulative_rewards.items()):
        draw_text(screen, f"{agent}: {cum_reward:.2f}", 50, 350 + i * 30)

    draw_text(screen, f"Cumulative Rewards Machine:", 50, 440)
    for i, (agent, cum_reward) in enumerate(env_machine.cumulative_rewards.items()):
        draw_text(screen, f"{agent}: {cum_reward:.2f}", 50, 470 + i * 30)

    # Event handling
    submit_pressed = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if input_box_L_s.collidepoint(event.pos):
                input_active_L_s = True
                input_active_f = False
            elif input_box_f.collidepoint(event.pos):
                input_active_f = True
                input_active_L_s = False
            else:
                input_active_L_s = False
                input_active_f = False

            # Check if submit button is pressed
            if button_x <= event.pos[0] <= button_x + button_width and button_y <= event.pos[1] <= button_y + button_height:
                submit_pressed = True

        elif event.type == pygame.KEYDOWN:
            if input_active_L_s:
                if event.key == pygame.K_RETURN:
                    input_active_L_s = False
                elif event.key == pygame.K_BACKSPACE:
                    input_text_L_s = input_text_L_s[:-1]
                else:
                    input_text_L_s += event.unicode
            elif input_active_f:
                if event.key == pygame.K_RETURN:
                    input_active_f = False
                elif event.key == pygame.K_BACKSPACE:
                    input_text_f = input_text_f[:-1]
                else:
                    input_text_f += event.unicode

    # Only update the environment and increment iterations if the submit button is pressed
    if submit_pressed:
        try:
            L_s_val = float(input_text_L_s)
            f_val = float(input_text_f)

            # Validate inputs
            if not (0 <= L_s_val <= 10 and 0.5 <= f_val <= 3):
                print("L_s must be between [0, 10] and f must be between [0.5, 3]")
                continue

            tplp_action = {"L_s": L_s_val, "f": f_val}
            state, rewards = env.step(tplp_action)
            state_machine, rewards_machine = env_machine.step()
            iterations += 1

            # Plot
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            env.plot_profit_regions(ax[0], theta_init, state[4])
            env.plot_profit_regions(ax[1], theta_init, 0)
            plt.tight_layout()
            plt.show()

        except ValueError:
            print("Invalid input. Enter valid numeric values for L_s and f.")

    # Update display
    pygame.display.flip()