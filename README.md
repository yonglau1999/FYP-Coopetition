# NTU Final Year Project D014 : 
## Enhancing supply chain resilience through game theory modelling and reinforcement learning
Reference Paper: ["The strategic analysis of logistics service sharing in an e-commerce platform"](https://www.sciencedirect.com/science/article/abs/pii/S0305048318313628)


## Extension:

* Introduction of **TPLP response** to E-tailer and Seller logistics service sharing.
* Additionally, introduced **capacity constraint** for E-tailer. If sharing-mode, any unfulfilled shipment goes to TPLP.
* Game environment is created using [PettingZoo's AEC Custom Environment](https://pettingzoo.farama.org/api/aec/) with 3 players in a sequential game.
* Optimal policy trained using Reinforcement Learning: [Ray's RLLib PPO](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo)

### Game set-up:
![image](https://github.com/user-attachments/assets/c97e0e6a-1dde-44a6-b3ad-f1e98e3df2b3)


### Training Procedure:

1. pip install "requirements.txt"
2. run "PPO_LSM_new.py"
3. To visualise on TensorBoard, follow instructions from output terminal. For e.g, you might see something like: ` View detailed results here: C:/Users/<user>/ray_results/coopetition_env/PPO
To visualize your results with TensorBoard, run: tensorboard --logdir C:/Users/<user>/AppData/Local/Temp/ray/session_2024-11-23_11-23-55_095928_17916/artifacts/2024-11-23_11-24-16/PPO/driver_artifacts`

#### Folder of trained policies:
[Trained Policies](PPO)
   
#### Example: Viewing tensorboard for trained model at θ = 6:

1. Open command line and navigate to [θ = 6 PPO Logs](Trained_policies/Theta_6/PPO_Logs)
2. Enter "tensorboard --logdir=."
3. Click on link given to view
