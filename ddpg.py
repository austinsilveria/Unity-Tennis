import networkforall
from utilities import hard_update
from torch.optim import Adam
import torch
import numpy as np

from OUNoise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class DDPGAgent:
    def __init__(self, state_size, action_size, num_agents, lr_actor=1.0e-4, lr_critic=1.0e-3):
        super(DDPGAgent, self).__init__()

        self.actor = networkforall.Actor(state_size, action_size).to(device)
        self.critic = networkforall.Critic(state_size, action_size, num_agents).to(device)

        self.target_actor = networkforall.Actor(state_size, action_size).to(device)
        self.target_critic = networkforall.Critic(state_size, action_size, num_agents).to(device)

        self.noise = OUNoise(action_size, scale=1.0)

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, obs, noise=0.0):
        """
           Params:
            obs (tensor): shape torch.Size([1, state_size]) representing agent state
            noise (int): noise to encourage exploration

           Returns:
            action (tensor): shape torch.Size([action_size]) -> actions chosen for agent
        """
        obs = obs.to(device)
        self.actor.eval()
        # Get action of torch.Size([1, action_size])
        action = self.actor(obs).cpu() + noise * self.noise.noise()

        # Return detached action of torch.Size([action_size])
        #return action.detach()
        return np.clip(action.detach(), -1, 1)

    def target_act(self, obs, noise=0.0):
        """
           Params:
            obs (tensor): shape torch.Size([1, state_size]) representing agent state
            noise (int): noise to encourage exploration

           Returns:
            action (tensor): shape torch.Size([action_size]) -> target actions chosen for agent
        """
        obs = obs.to(device)
        self.target_actor.eval()
        # Get action of torch.Size([1, action_size])
        action = self.target_actor(obs).cpu() + noise * self.noise.noise()

        # Return detached action of torch.Size([action_size])
        #return action.detach()
        return np.clip(action.detach(), -1, 1)
