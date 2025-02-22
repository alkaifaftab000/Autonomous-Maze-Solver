# part 6
import os
import torch
from model import *
from buffer import ReplayBuffer
from torch.optim import Adam
import torch.nn.functional as F


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data) 

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau) 

class Agent(object):
    def __init__ (self, num_inputs, action_space, gamma, tau, alpha, target_update_interval, 
                  hidden_size, learning_rate, exploration_scaling_factor):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {self.device}")

        self.critic = Critic(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)
        self.critic_target = Critic(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)

        hard_update(self.critic_target, self.critic) # Incomplete

        self.policy = Actor(num_inputs, action_space.shape[0], hidden_size).to(device=self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)

        return action.detach().cpu().numpy()[0] 
    
    def update_parameters(self, memory: ReplayBuffer, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)  
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        print(mask_batch.shape)


        # Here we will write predictive model 
        # Coding Predictive Model 

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)    

        
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)    

        qf_loss = qf1_loss + qf2_loss

        # Critic Network Updated
        self.critic_optim.zero_grad()   
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch,pi)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = torch.tensor(0.).to(self.device)
        alpha_tlogs = torch.tensor(0.).to(self.device)
        
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # def test(self, env, episodes, max_timesteps):

    def save_checkpoints(self, filename):
        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")
        self.policy.save_checkpoint()
        self.critic_target_checkpoint()
        self.critic.save_checkpoint()

    def load_checkpoint(self, evaluate=False):
        try:
            print("Loading checkpoint...")
            self.policy.load_checkpoint()
            self.critic_target.load_checkpoint()
            self.critic.load_checkpoint()
            print("Checkpoint loaded :) ")
        except:
            if(evaluate):
                raise Exception("No checkpoint found")
            else:
                print("No checkpoint found")

        if evaluate:
            self.policy.eval() 
            self.critic_target.eval()
            self.critic.eval()
        else:    
            self.policy.train()
            self.critic_target.train()
            self.critic.train()
