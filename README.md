# NTU Final Year Project D014 : 
## Enhancing supply chain resilience through game theory modelling and reinforcement learning
This repository contains the codebase for my final-year project exploring coopetition dynamics in logistics service sharing. It integrates game-theoretic modeling with reinforcement learning to simulate strategic interactions among e-commerce platforms, sellers, and third-party logistics providers (TPLPs). This builds on a current study: ["The strategic analysis of logistics service sharing in an e-commerce platform"](https://www.sciencedirect.com/science/article/abs/pii/S0305048318313628), with increased emphasis on the actions of the TPLPs and also the introduction of capacity constraint of the e-commerce platform.
### Extension:

* Introduction of **TPLP response** to E-tailer and Seller logistics service sharing.
* Additionally, introduced **capacity constraint** for E-tailer. If sharing-mode, any unfulfilled shipment goes to TPLP.
* Game environment is created using [PettingZoo's AEC Custom Environment](https://pettingzoo.farama.org/api/aec/) with 3 players in a sequential game.
* Optimal policy trained using Reinforcement Learning: [Ray's RLLib PPO](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo)

## Repository Structure of key scripts

- `Logistics_Service_Model.py`: Core logistics model with profit and service-sharing logic for all agents.
- `StackelBerg.py`: Stackelberg game logic used to simulate strategic responses of the e-tailer and seller.
- `PPO_LSM_new.py`: Main reinforcement learning training script integrating PettingZoo and PPO via Ray RLlib.
- `LSM_Game.py`: Pygame interface for users to test out trained policies.
- `PPO`: Folder of trained policies.
  
## How to use 

pip install "requirements.txt"

### Training Procedure:
This provides a guide on how to train the optimal policy for the TPLP given a fixed market potential (θ).

1. Go to `LSM_Game.py`. Determine the market potential, θ, which you want to train at. Set this value here (range from 4-8).
2. 
![image](https://github.com/user-attachments/assets/b1ba279f-e197-4199-ac6c-3ee9b412275d)
3. Go to `Logistics_Service_Model.py`. Determine the capacity of the E-tailer. Set this value here under "self.max_capacity" (Value used in project is 5).
4. 
![image](https://github.com/user-attachments/assets/dfc6fa75-b0a5-4876-9cd1-94a65110c040)
5. run "PPO_LSM_new.py"
6. To visualise on TensorBoard, follow instructions from output terminal. For e.g, you might see something like: ` View detailed results here: C:/Users/<user>/ray_results/coopetition_env/PPO
To visualize your results with TensorBoard, run: tensorboard --logdir C:/Users/<user>/AppData/Local/Temp/ray/session_2024-11-23_11-23-55_095928_17916/artifacts/2024-11-23_11-24-16/PPO/driver_artifacts`
7. Alternatively, viewing TensorBoard post training. (Example for trained model at θ = 6, E-tailer capacity constrained):
   
   1. Open command line and navigate to [θ = 6 PPO](PPO/Theta_6_1)
   2. Enter "tensorboard --logdir=."
   3. Click on link given to view

### Policy Testing Procedure (Example for trained model at θ = 6, E-tailer capacity constrained):

1. Go to `LSM_Game.py`. Navigate to the [trained policy directory](PPO/Theta_6_1/checkpoint_000487/policies/tplp_policy). This will be located in the final checkpoint, and copy paste in this line.
2. 
![image](https://github.com/user-attachments/assets/00f0dcef-18c2-4fe0-ac86-ecd8948ba319)
3. Set the market potential, θ  which the agent is trained at here.
4. 
![image](https://github.com/user-attachments/assets/824ffa01-6602-4d0b-91f9-e6075e97f86d)
5. Run the script. The interface will look like this:
6. 
![image](https://github.com/user-attachments/assets/37c61736-5b54-47c9-b6c1-f134b58d8671)
7. Input your value of L_s and f and compare the rewards against the trained policy's (machine) rewards.
   
#### Folder of pre-trained policies:
[Trained Policies](PPO)
   
