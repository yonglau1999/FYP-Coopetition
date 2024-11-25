import pygame
import numpy as np
from ray.rllib.policy.policy import Policy
from Logistics_Service_Model import LogisticsServiceModel
from gymnasium import spaces

# Load trained policies
e_tailer_policy = Policy.from_checkpoint("Trained_policies\\Theta_5\\e_tailer_policy")
seller_policy = Policy.from_checkpoint("Trained_policies\\Theta_5\\seller_policy")
tplp_policy = Policy.from_checkpoint("Trained_policies\\Theta_5\\tplp_policy")


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
        if agent == "e_tailer":
            return np.array([self.state[0], self.state[1], self.state[2]], dtype=np.float64)
        elif agent == "seller":
            return np.array([self.state[0], self.state[1], self.state[2]], dtype=np.float64)
        elif agent == "tplp":
            return np.array([self.state[0], self.state[4]], dtype=np.float64)

    def step(self, tplp_action):

        e_tailer_action = e_tailer_policy.compute_single_action(self.observe("e_tailer"),clip_actions=True)[0]
        seller_action = seller_policy.compute_single_action(self.observe("seller"),clip_actions=True)[0]
        self.state[3] = e_tailer_action
        self.state[4] = seller_action if self.state[3] == 1 else 0
        self.state[1] = tplp_action["L_s"]
        self.state[2] = tplp_action["f"]
        self.model = LogisticsServiceModel(self.state[1], self.state[0], self.state[2])
        self.rewards = self.calculate_rewards()
        print(e_tailer_action,seller_action)
        for agent in self.agents:
            self.cumulative_rewards[agent] += self.rewards[agent]

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
        profit_et_sharing = self.model.profit_sharing_etailer(True) # Assume both sharing first, price = wstar
        profit_seller_no_sharing = self.model.profit_nosharing_seller()
        profit_seller_sharing = self.model.profit_sharing_seller(True) # Assume both sharing first, price = wstar

        profit_diff_et = profit_et_sharing - profit_et_no_sharing
        profit_diff_seller = profit_seller_sharing - profit_seller_no_sharing

        ww = (profit_diff_et >= -1e-8) and (profit_diff_seller >= -1e-8)
        print(ww)
        theta = self.state[0]
        L_s = self.state[1]
        f = self.state[2]
        sharing_status = self.state[4]
        self.model.L_s = L_s
        self.model.theta = theta

        if agent == "e_tailer":
            return self.model.profit_sharing_etailer(ww) if sharing_status == 1 else self.model.profit_nosharing_etailer()
        elif agent == "seller":
            return self.model.profit_sharing_seller(ww) if sharing_status == 1 else self.model.profit_nosharing_seller()
        elif agent == "tplp":
            return self.model.profit_sharing_tplp(ww) if sharing_status == 1 else self.model.profit_nosharing_tplp()
        return 0


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

        e_tailer_action = e_tailer_policy.compute_single_action(self.observe("e_tailer"),clip_actions=True,explore=False)[0]
        seller_action = seller_policy.compute_single_action(self.observe("seller"),clip_actions=True,explore=False)[0]
        self.state[3] = e_tailer_action
        self.state[4] = seller_action if self.state[3] == 1 else 0
        tplp_action = tplp_policy.compute_single_action(self.observe("tplp"),clip_actions=True,explore=False)[0]
        tplp_action = self.action_spaces["tplp"].low + (self.action_spaces["tplp"].high - self.action_spaces["tplp"].low) * ((np.tanh(tplp_action) + 1) / 2)
        print(tplp_action)
        self.state[1] = tplp_action[0]
        self.state[2] = tplp_action[1]
        self.model = LogisticsServiceModel(self.state[1], self.state[0], self.state[2])
        print(tplp_action)
        self.rewards = self.calculate_rewards()

        for agent in self.agents:
            self.cumulative_rewards[agent] += self.rewards[agent]

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
        profit_et_sharing = self.model.profit_sharing_etailer(False)
        profit_seller_no_sharing = self.model.profit_nosharing_seller()
        profit_seller_sharing = self.model.profit_sharing_seller(False)

        profit_diff_et = profit_et_sharing - profit_et_no_sharing
        profit_diff_seller = profit_seller_sharing - profit_seller_no_sharing

        ww = (profit_diff_et >= -1e-8) and (profit_diff_seller >= -1e-8)
        theta = self.state[0]
        L_s = self.state[1]
        f = self.state[2]
        sharing_status = self.state[4]
        self.model.L_s = L_s
        self.model.theta = theta

        if agent == "e_tailer":
            return self.model.profit_sharing_etailer(ww) if sharing_status == 1 else self.model.profit_nosharing_etailer()
        
        elif agent == "seller":
            return self.model.profit_sharing_seller(ww) if sharing_status == 1 else self.model.profit_nosharing_seller()
        
        elif agent == "tplp":
            return self.model.profit_sharing_tplp(ww) if sharing_status == 1 else self.model.profit_nosharing_tplp()
        
        return 0
    
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
slider_L_s_value = 1
slider_f_value = 0.1

# Button properties
button_x = 400
button_y = 500
button_width = 150
button_height = 50

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

    # Draw sliders
    draw_text(screen, "Adjust Service Level (L_s):", slider_x , slider_L_s_y - 20)
    knob_L_s_x = draw_slider(screen, slider_x, slider_L_s_y, slider_width, 1, 10, 0.5, slider_L_s_value)
    draw_text(screen, f"L_s Value: {slider_L_s_value:.2f}", slider_x + slider_width + 20, slider_L_s_y - 10)

    draw_text(screen, "Adjust Logistics Price (f):", slider_x, slider_f_y - 20)
    knob_f_x = draw_slider(screen, slider_x, slider_f_y, slider_width, 0.1, 3, 0.1, slider_f_value)
    draw_text(screen, f"f Value: {slider_f_value:.2f}", slider_x + slider_width + 20, slider_f_y - 10)

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
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if slider_x <= mouse_x <= slider_x + slider_width:
                if slider_L_s_y - 10 <= mouse_y <= slider_L_s_y + 10:
                    slider_L_s_value = slider_value(mouse_x, slider_x, slider_width, 1, 10, 0.5)
                elif slider_f_y - 10 <= mouse_y <= slider_f_y + 10:
                    slider_f_value = slider_value(mouse_x, slider_x, slider_width, 0.1, 3, 0.1)
            # Check if submit button is pressed
            if button_x <= mouse_x <= button_x + button_width and button_y <= mouse_y <= button_y + button_height:
                submit_pressed = True

    # Only update the environment and increment iterations if the submit button is pressed
    if submit_pressed:
        tplp_action = {"L_s": slider_L_s_value, "f": slider_f_value}
        state, rewards = env.step(tplp_action)
        state_machine, rewards_machine = env_machine.step()
        iterations += 1

    # Update display
    pygame.display.flip()

# pygame.quit()