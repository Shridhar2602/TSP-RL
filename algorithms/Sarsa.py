import numpy as np

class SarsaAgent():
	def __init__(self, n_states, n_actions, epsilon = 1.0, epsilon_min = 0.01, epsilon_decay = 0.5, gamma = 0.95, lr = 0.8):
		self.n_states = n_states
		self.n_actions = n_actions
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay
		self.gamma = gamma
		self.lr = lr
		self.ep_count = 0

		np.random.seed(0);

		# state x action matrix to hold Q values
		self.Q = np.zeros([n_states, n_actions]);

	def epsilon_greedy(self, state, visited_states):

		q = np.copy(self.Q[state, :])
		q[visited_states] = -np.inf

		if(len(visited_states) == self.n_states):
			action = visited_states[0]
		else:
			if np.random.rand() > self.epsilon:
				action = np.argmax(q)	# greedy action
			else:
				action = np.random.choice([x for x in range(self.n_states) if x not in visited_states])	# random action

		self.ep_count += 1
		# epsilon decay
		if self.epsilon > self.epsilon_min:
			if(self.ep_count % 3 == 0):
				self.epsilon *= self.epsilon_decay

		# if self.epsilon > self.epsilon_min:
		# 	self.epsilon *= self.epsilon_decay

		return action
	
	def epsilon_greedy_v2(self, state, visited_states):

		if np.random.rand() > self.epsilon:
			action = np.argmax(self.Q[state, :])
		else:
			action = np.random.randint(0, self.n_states)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		return action

	def greedy(self, state, visited_states):
		
		q = np.copy(self.Q[state, :])
		q[visited_states] = -np.inf

		if(len(visited_states) == self.n_states):
			return visited_states[0]
		else:
			return np.argmax(q)

	def train(self, state, action, reward, next_state, next_action):
		# SARSA update
		if(next_action == -1):
			self.Q[state, action] = self.Q[state, action] + self.lr * (reward - self.Q[state, action])
		else:
			self.Q[state, action] = self.Q[state, action] + self.lr * ((reward + self.gamma * self.Q[next_state, next_action]) - self.Q[state, action])