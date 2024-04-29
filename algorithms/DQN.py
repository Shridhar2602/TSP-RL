import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent():
	def __init__(self, n_states, n_actions, epsilon = 1.0, epsilon_min = 0.01, epsilon_decay = 0.5, gamma = 0.95, lr = 0.8):
		self.n_states = n_states
		self.n_actions = n_actions
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay
		self.gamma = gamma
		self.lr = lr

		np.random.seed(0);

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.create_model();
	
	def create_model(self):
		self.model = nn.Sequential(
			nn.Linear(self.n_states, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, self.n_actions)
		)
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
		self.loss_fn = nn.MSELoss()

	def epsilon_greedy(self, state, visited_states):

		q = self.model(torch.tensor(state, dtype=torch.float).to(self.device))
		print(q);

		# q[visited_states] = -np.inf

		# if(len(visited_states) == self.n_states):
		# 	action = visited_states[0]
		# else:
		# 	if np.random.rand() > self.epsilon:
		# 		action = np.argmax(q)	# greedy action
		# 	else:
		# 		action = np.random.choice([x for x in range(self.n_states) if x not in visited_states])	# random action

		# # epsilon decay
		# if self.epsilon > self.epsilon_min:
		# 	self.epsilon *= self.epsilon_decay

		# return action

	# def greedy(self, state, visited_states):
		
	# 	q = np.copy(self.Q[state, :])
	# 	q[visited_states] = -np.inf

	# 	if(len(visited_states) == self.n_states):
	# 		return visited_states[0]
	# 	else:
	# 		return np.argmax(q)

	# def train(self, state, action, reward, next_state):
	# 	# Q-learning update
	# 	self.Q[state, action] = self.Q[state, action] + self.lr * (reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action])