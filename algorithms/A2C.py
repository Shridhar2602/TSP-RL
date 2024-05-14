import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class A2CAgent():
	def __init__(self, n_states, n_actions, gamma=0.95, lr_critic=0.01, lr_actor=0.001):
		self.n_states = n_states
		self.n_actions = n_actions
		self.gamma = gamma
		self.lr_critic = lr_critic
		self.lr_actor = lr_actor

		self.epsilon = 1
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.999

		torch.manual_seed(0)
		np.random.seed(0)
		random.seed(0)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.actor, self.critic = self.create_models(self.n_states, self.n_actions)
		self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
		self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

	def create_models(self, state_size, action_size):
		actor = nn.Sequential(
			nn.Linear(state_size, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 128),
			nn.ReLU(),
			nn.Linear(128, action_size),
			nn.Softmax(dim=-1)
		)

		critic = nn.Sequential(
			nn.Linear(state_size, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 1)
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
		probs /= np.sum(probs)
		action = np.random.choice(self.n_actions, p=probs)
		return action
	
	def act2(self, state, visited_states):
		if len(visited_states) == self.n_states:
			return visited_states[0]

		state = self.oneHot(state)
		probs = self.actor(state)
		probs = nn.functional.softmax(probs, dim=-1)
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

	def train(self, state, action, reward, next_state, done):
		state = self.oneHot(state)
		next_state = self.oneHot(next_state)

		critic_state = self.critic(state)
		critic_next_state = self.critic(next_state)

		with torch.no_grad():
			target = reward + self.gamma * critic_next_state * (1 - done)
			advantage = target - critic_state


		critic_loss = nn.functional.mse_loss(critic_state, target)
		actor_loss = -torch.log(self.actor(state)[0][action]) * advantage

		self.optimizer_actor.zero_grad()
		actor_loss.backward()
		self.optimizer_actor.step()

		self.optimizer_critic.zero_grad()
		critic_loss.backward()
		self.optimizer_critic.step()
