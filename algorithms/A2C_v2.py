import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class A2CAgent():
	def __init__(self, n_states, n_actions, gamma=0.95, lr_critic=0.01, lr_actor=0.001, memory_size=64, batch_size=32):
		self.n_states = n_states
		self.n_actions = n_actions
		self.gamma = gamma
		self.lr_critic = lr_critic
		self.lr_actor = lr_actor

		self.memory_size = memory_size
		self.batch_size = batch_size

		self.replay_memory = deque(maxlen=self.memory_size)

		self.epsilon = 1
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.9999

		torch.manual_seed(0)
		np.random.seed(0)
		random.seed(0)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.actor, self.critic = self.create_models(self.n_states, self.n_actions)
		self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
		self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

	def create_models(self, state_size, action_size):
		actor = nn.Sequential(
			nn.Linear(state_size, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 256),
			nn.ReLU(),
			nn.Linear(256, action_size),
			nn.Softmax(dim=-1)
		)

		critic = nn.Sequential(
			nn.Linear(state_size, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 1)
		)

		return actor.to(self.device), critic.to(self.device)

	def oneHot(self, state):
		return torch.nn.functional.one_hot(torch.tensor(state), num_classes=self.n_states).float().unsqueeze(0).to(self.device)

	def act(self, state, visited_states):
		if len(visited_states) == self.n_states:
			return visited_states[0]

		state = self.oneHot(state)
		probs = self.actor(state)
		probs = nn.functional.softmax(probs, dim=-1)
		probs = probs.cpu().detach().numpy().squeeze()
		probs[visited_states] = 0
		action = np.argmax(probs)
		probs /= np.sum(probs)
		action = np.random.choice(self.n_actions, p=probs)
		return action
	
	def act2(self, state, visited_states):
		if len(visited_states) == self.n_states:
			return visited_states[0]

		state = self.oneHot(state)
		probs = self.actor(state)
		# probs = nn.functional.softmax(probs, dim=-1)
		probs = probs.cpu().detach().numpy().squeeze()
		probs[visited_states] = 0

		if np.random.rand() > self.epsilon:
			action = np.argmax(probs)
		else:
			action = np.random.choice([x for x in range(self.n_states) if x not in visited_states])

		# probs /= np.sum(probs)
		# action = np.random.choice(self.n_actions, p=probs)
		
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		return action
	
	def add_to_memory(self, state, action, reward, next_state, done):
		self.replay_memory.append((state, action, reward, next_state, done))

	def train(self, state, action, reward, next_state, done):

		self.add_to_memory(self.oneHot(state), action, reward, self.oneHot(next_state), done)

		if len(self.replay_memory) > self.batch_size:
			batch = random.sample(self.replay_memory, self.batch_size)
		else:
			batch = self.replay_memory

		state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
		
		state_batch = torch.stack(state_batch)
		reward_batch = torch.tensor(reward_batch).to(self.device).t().unsqueeze(1)
		next_state_batch = torch.stack(next_state_batch)
		done_batch = torch.tensor(done_batch).to(self.device).unsqueeze(1)

		critic_state = self.critic(state_batch).squeeze(2)
		critic_next_state = self.critic(next_state_batch).squeeze(2)

		with torch.no_grad():
			target = reward_batch + self.gamma * critic_next_state * (1 - done_batch)
			advantage = target - critic_state
		
		advantage = advantage.squeeze(1)

		actor_state = self.actor(state_batch).squeeze(1)
		critic_loss = nn.functional.mse_loss(critic_state, target)
		actor_loss = -torch.log(actor_state[range(len(batch)), action_batch]) * advantage

		self.optimizer_critic.zero_grad()
		critic_loss.backward()
		self.optimizer_critic.step()

		self.optimizer_actor.zero_grad()
		actor_loss.mean().backward()
		self.optimizer_actor.step()