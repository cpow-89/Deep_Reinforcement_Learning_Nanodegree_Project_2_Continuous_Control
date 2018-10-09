# Deep Reinforcement Learning Nanodegree: Project 2 - Continuous Control

This project includes the code for an implementation of the deep deterministic policy gradient(DDPG) algorithm which I wrote to solve the Project 2 - Continuous Control of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) @ Udacity. My version of the DDPG algorithm is inspired by chapter 14 of Maxim Lapan’s book called "Deep Reinforcement Learning Hands-On".

For more information on the implemented features refer to "Report.ipynb". The notebook includes a summary of all essential concepts used in the code.


### Project 2 - Continuous Control - Details:

The goal of this project was to train an agent, represented by a double-jointed arm, to maintain its position at the target location(great green sphere) for as many time steps as possible. 

[//]: # (Image References)

#### Random Agent

[image1]: https://raw.githubusercontent.com/cpow-89/Deep_Reinforcement_Learning_Nanodegree_Project_2_Continuous_Control/master/images/untrained_agent.gif?token=AmwnwlXyXniU-umlY4BNx8VSfAnYd57mks5bxNYIwA%3D%3D "Random Agent"

![Random Agent][image1]


#### Trained Agent

[image2]: https://raw.githubusercontent.com/cpow-89/Deep_Reinforcement_Learning_Nanodegree_Project_2_Continuous_Control/master/images/trained_agent.gif?token=Amwnwv58uwb_JY6Z0p0_vJrWmnnl-0Eeks5bxNVywA%3D%3D "Trained Agent"
![Trained Agent][image2]

##### Reward:
- a reward of +0.1 is provided for each step that the agent's hand is in the goal location

##### Search Space
- the state space has 33 dimensions 
     - corresponding to the position, rotation, velocity, and angular velocities of the arm
- the action space has four dimensions
    - every action is a continuous number between -1 and 1

##### Task
- the task is episodic
- the agent has to maintain its position at the target location(great green sphere) for as many time steps as possible
- to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes
        

### Getting Started

1. Create (and activate) a new environment with Python 3.6.

> conda create --name env_name python=3.6<br>
> source activate env_name

2. Download the environment from one of the links below and place it into \p2_continuous-control\Reacher_One_Linux

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    
    
- your folder should now look something like this:

\Reacher_One_Linux<br>
&nbsp;&nbsp;&nbsp;&nbsp; \Reacher_Data  <br>
&nbsp;&nbsp;&nbsp;&nbsp; \Reacher.x86<br>
&nbsp;&nbsp;&nbsp;&nbsp; \Reacher.x86_64<br>

3. Install Sourcecode dependencies

> conda install -c pytorch pytorch <br>
> conda install -c anaconda numpy <br>
> pip install tensorboardX

- unityagents is also required
    - an easy way to get this is to install the Deep Reinforcement Learning Nanodegree with its dependencies
    
> git clone https://github.com/udacity/deep-reinforcement-learning.git<br>
> cd deep-reinforcement-learning/python<br>
> pip install .<br>

### How to run the project

You can run the project by running the main.py file through the console.
- open the console and run: python main.py -c "your_config_file.json" 
- to train the agent from scratch set "run_training" in the config file to true
- to run the pre-trained agent set "run_training" in the config file to false

optional arguments:

-h, --help

    - show help message
    
-c , --config

    - Config file name - file must be available as .json in ./configs
    
Example: python main.py -c "reacher_one.json" 
