# Deep Reinforcement Learning Nanodegree: Project 2 - Continuous Control - Report


### 1. General:

The goal of this project was to train an agent, represented by a double-jointed arm, to maintain its position at the target location(great green sphere) for as many time steps as possible. 

[//]: # (Image References)

<br>
Random Agent:

[image1]: https://raw.githubusercontent.com/cpow-89/Deep_Reinforcement_Learning_Nanodegree_Project_2_Continuous_Control/master/images/untrained_agent.gif?token=AmwnwlXyXniU-umlY4BNx8VSfAnYd57mks5bxNYIwA%3D%3D "Random Agent"

![Random Agent][image1]

### 2. Learning algorithm

### 3. Hyperparameters

### 4. Network architectures

Critic + Critic_Target:

DDPGCritic(<br>
&nbsp;&nbsp;&nbsp;&nbsp;(state_head): Sequential(<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(0): Linear(in_features=33, out_features=400, bias=True)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(1): ReLU()<br>
&nbsp;&nbsp;&nbsp;&nbsp;)<br>
&nbsp;&nbsp;&nbsp;&nbsp;(state_action_body): Sequential(<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(0): Linear(in_features=404, out_features=300, bias=True)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(1): ReLU()<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(2): Linear(in_features=300, out_features=1, bias=True)<br>
&nbsp;&nbsp;&nbsp;&nbsp;)<br>
)<br>

Actor + Actor_Target:

DDPGActor(<br>
&nbsp;&nbsp;&nbsp;&nbsp;(network): Sequential(<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(0): Linear(in_features=33, out_features=400, bias=True)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(1): ReLU()<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(2): Linear(in_features=400, out_features=300, bias=True)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3): ReLU()<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(4): Linear(in_features=300, out_features=4, bias=True)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(5): Tanh()<br>
&nbsp;&nbsp;&nbsp;&nbsp;)<br>
)<br>

### 5. Results

Trained Agent:

[image2]: https://raw.githubusercontent.com/cpow-89/Deep_Reinforcement_Learning_Nanodegree_Project_2_Continuous_Control/master/images/trained_agent.gif?token=Amwnwv58uwb_JY6Z0p0_vJrWmnnl-0Eeks5bxNVywA%3D%3D "Trained Agent"
![Trained Agent][image2]


### 6. Ideas for Future Work