import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DDQNAgent():
	def __init__(self, n_states, n_actions, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.5, gamma=0.95, lr=0.8, memory_size=256, batch_size=64):
		self.n_states = n_states
		self.n_actions = n_actions
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay
		self.gamma = gamma
		self.lr = lr
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		torch.manual_seed(0)
		random.seed(0)
		np.random.seed(0)
		self.modelA = self.create_model()
		self.modelB = self.create_model()
		self.optimizerA = optim.Adam(self.modelA.parameters(), lr=self.lr)
		self.optimizerB = optim.Adam(self.modelB.parameters(), lr=self.lr)
		self.replay_memory = deque(maxlen=self.memory_size)

	def create_model(self):
		model = nn.Sequential(
			nn.Linear(self.n_states, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 512),
			nn.ReLU(),
			nn.Linear(512, self.n_actions)
		)
		model.to(self.device)
		return model

	def oneHot(self, state):
		return torch.nn.functional.one_hot(torch.tensor(state), num_classes=self.n_states).float().to(self.device)

	def epsilon_greedy(self, state, visited_states):
		state = self.oneHot(state)
		qA = self.modelA(state).detach().cpu().numpy()
		qB = self.modelB(state).detach().cpu().numpy()

		q = (qA + qB) / 2

		q[visited_states] = -np.inf

		if len(visited_states) == self.n_states:
			action = visited_states[0]
		else:
			if np.random.rand() > self.epsilon:
				action = np.argmax(q)  # greedy action
			else:
				action = np.random.choice([x for x in range(self.n_states) if x not in visited_states])  # random action

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
		reward_batch = torch.tensor(reward_batch).to(self.device)
		next_state_batch = torch.stack(next_state_batch)
		done_batch = torch.tensor(done_batch).to(self.device)

		if np.random.rand() > 0.5:
			# update A using B
			Q_values = self.modelA(state_batch)
			next_Q_values = self.modelB(next_state_batch)
			Q_target = reward_batch + self.gamma * torch.max(next_Q_values, dim=1).values * (1 - done_batch)
			loss = nn.functional.mse_loss(Q_values[range(len(batch)), action_batch], Q_target)
			self.optimizerA.zero_grad()
			loss.backward()
			self.optimizerA.step()
		else:
			# update B using A
			Q_values = self.modelB(state_batch)
			next_Q_values = self.modelA(next_state_batch)
			Q_target = reward_batch + self.gamma * torch.max(next_Q_values, dim=1).values * (1 - done_batch)
			loss = nn.functional.mse_loss(Q_values[range(len(batch)), action_batch], Q_target)
			self.optimizerB.zero_grad()
			loss.backward()
			self.optimizerB.step()