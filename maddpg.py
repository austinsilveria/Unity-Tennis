from ddpg import DDPGAgent
import torch
from utilities import soft_update
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GRADIENT_CLIP = 1


class MADDPG:
    def __init__(self, state_size, action_size, num_agents, discount_factor=0.95, tau=0.0396):
        super(MADDPG, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        self.maddpg_agent = [DDPGAgent(state_size, action_size, num_agents),
                             DDPGAgent(state_size, action_size, num_agents)]

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """
           Params:
            obs_all_agents (tensor): shape torch.Size([num_agents, state_size]) representing all agent states
            noise (int): noise to encourage exploration

           Returns:
            actions (list of tensors): list of length num_agents
                                       tensors of shape torch.Size([action_size]) -> actions chosen for each agent
        """

        actions = [self.maddpg_agent[i].act(obs_all_agents[i, :].view(1, -1), noise).squeeze() for i in range(self.num_agents)]

        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """
           Params:
            obs_all_agents (tensor): shape torch.Size([num_agents, state_size]) representing all agent states
            noise (int): noise to encourage exploration

           Returns:
            actions (list of tensors): list of length num_agents
                                       tensors of shape torch.Size([action_size]) -> target actions chosen for each agent
        """

        target_actions = [self.maddpg_agent[i].target_act(obs_all_agents[:, i, :],  noise) for i in range(self.num_agents)]

        return target_actions

    def update(self, samples, agent_number):
        """ update the critics and actors of all the agents

            Params:
             samples (tuple): environment information to perform agent update
             agent_number (int): represents agent to be updated

            Returns:
             None
        """

        state, full_state, action, reward, next_state, full_next_state, done = samples

        batch_size = full_state.shape[0]

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # critic_loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network

        # Get target actions of all agents to use in reward calculation
        target_actions = self.target_act(next_state.view(batch_size, self.num_agents, -1))
        target_actions = torch.cat(target_actions, dim=1)

        with torch.no_grad():
            q_next = agent.target_critic(full_next_state, target_actions.to(device))

        y = reward[:, agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[:, agent_number].view(-1, 1))

        q = agent.critic(full_state, action.view(batch_size, -1))

        critic_loss = F.mse_loss(q, y.detach())

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), GRADIENT_CLIP)
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [
            self.maddpg_agent[i].actor(state.view([batch_size, self.num_agents, -1])[:, i, :]) if i == agent_number else
            self.maddpg_agent[i].actor(state.view([batch_size, self.num_agents, -1])[:, i, :]).detach() for i in
            range(self.num_agents)]

        q_input = torch.cat(q_input, dim=1)

        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already

        # get the policy gradient
        actor_loss = -agent.critic(full_state, q_input).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), GRADIENT_CLIP)
        agent.actor_optimizer.step()

        self.update_targets()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
