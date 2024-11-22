# NTU Final Year Project D014 : 
## Enhancing supply chain resilience through game theory modelling and reinforcement learning
Reference paper: ["The strategic analysis of logistics service sharing in an e-commerce platform"](https://www.sciencedirect.com/science/article/abs/pii/S0305048318313628)


## Extension:

Introduction of TPLP response to E-tailer and Seller logistics service sharing. Additionally, E-tailer has capacity constraint. If sharing, unfulfilled shipment goes to TPLP.
Game environment is created using PettingZoo's AEC Custom Environment with 3 players in a sequential game.
Optimal policy trained using Reinforcement Learning: [Ray's RLLib PPO](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo)

### Game set-up:
![image](https://github.com/user-attachments/assets/bb22962a-324f-426b-864a-a772bf672f3d)


[Profit Zones Sharing mode replicated from paper](Profitable%20zone.png) | [Trained Policies for each player when θ = 4](Trained_policies_theta4)

### Training Procedure:

1. pip install "requirements.txt"
2. run "PPO_LSM.py"
   
### Viewing tensorboard for trained model at θ = 4:

1. Open command line and navigate to [PPO Logs](Trained_policies_theta4/PPO_Logs)
2. Enter "tensorboard --logdir=."
3. Click on [link](http://localhost:6006/) to view
