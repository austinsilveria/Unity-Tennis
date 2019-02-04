### Introduction

This repository aims to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment with the [MADDPG](https://arxiv.org/abs/1706.02275) algorithm.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Unzip, and replace the environment folder in this repository (Tennis_Windows_D_x86_64) with the unzipped folder.

#### Required Dependencies:
    Python 3.6
    Unity Agents
    PyTorch
    Numpy
    Matplotlib
    Jupyter (Optional, python files are provided if you wish to run from command line)
    
### Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent

### References

    Udacity MADDPG Lab Implementation
    
    [Udacity Deep RL](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet)
    
    [yk287 MADDPG Implementation](https://github.com/yk287/MADDPG-Tennis-UnityMLPlatform)