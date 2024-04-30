import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import random
import tensorflow as tf

class DDQNAgent():
	def __init__(self, n_states, n_actions, epsilon = 1.0, epsilon_min = 0.01, epsilon_decay = 0.5, gamma = 0.95, lr = 0.8, memory_size = 256, batch_size=64):
		self.n_states = n_states
		self.n_actions = n_actions
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay
		self.gamma = gamma
		self.lr = lr
		self.memory_size = memory_size
		self.batch_size = batch_size
		np.random.seed(0)
		random.seed(0)
		self.modelA = self.create_model(self.n_states, self.n_actions);
		self.modelB = self.create_model(self.n_states, self.n_actions);
	
		self.replay_memory = deque(maxlen=self.memory_size)

	def create_model(self, state_size, action_size):
		model = Sequential()
		model.add(Dense(128, input_dim=state_size, activation='relu'))
		model.add(Dense(64, activation='relu'))
		model.add(Dense(128, activation='relu'))
		model.add(Dense(action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.lr))
		return model
	
	def oneHot(self, state):
		return tf.expand_dims(tf.one_hot(state, self.n_states), axis=0)
	
	def epsilon_greedy(self, state, visited_states):

		state = self.oneHot(state)
		qA = self.modelA.predict(state, verbose=0)
		qB = self.modelB.predict(state, verbose=0)

		qA = np.squeeze(qA)
		qB = np.squeeze(qB)

		q = (qA + qB) / 2

		q[visited_states] = -np.inf

		if(len(visited_states) == self.n_states):
			action = visited_states[0]
		else:
			if np.random.rand() > self.epsilon:
				action = np.argmax(q)	# greedy action
			else:
				action = np.random.choice([x for x in range(self.n_states) if x not in visited_states])	# random action

		# epsilon decay
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		return action

	def greedy(self, state, visited_states):

		state = self.oneHot(state)
		q = self.model.predict(state, verbose=0)

		q[visited_states] = -np.inf

		if(len(visited_states) == self.n_states):
			return visited_states[0]
		else:
			return np.argmax(q)
		
	def add_to_memory(self, state, action, reward, next_state, done):
		self.replay_memory.append((state, action, reward, next_state, done))

	def train(self, state, action, reward, next_state, done):

		self.add_to_memory(self.oneHot(state), action, reward, self.oneHot(next_state), done)

		if len(self.replay_memory) > self.batch_size:
			batch = random.sample(self.replay_memory, self.batch_size)
		else:
			batch = self.replay_memory

		state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
		
		state_batch = tf.squeeze(tf.stack(state_batch), axis=1)
		reward_batch = tf.stack(reward_batch)
		next_state_batch = tf.squeeze(tf.stack(next_state_batch), axis=1)
		done_batch = np.array(done_batch)

		if(np.random.rand() > 0.5):
			# update A using B
			Q_values = self.modelA.predict(state_batch, verbose=0)
			next_Q_values = self.modelB.predict(next_state_batch, verbose=0)

			target_values = reward_batch + self.gamma * np.max(next_Q_values, axis=1) * (1 - done_batch)
			Q_values[np.arange(len(batch)), action_batch] = target_values

			self.modelA.fit(state_batch, Q_values, epochs=1, verbose=0)

		else:
			# update B using A
			Q_values = self.modelB.predict(state_batch, verbose=0)
			next_Q_values = self.modelA.predict(next_state_batch, verbose=0)

			target_values = reward_batch + self.gamma * np.max(next_Q_values, axis=1) * (1 - done_batch)
			Q_values[np.arange(len(batch)), action_batch] = target_values

			self.modelB.fit(state_batch, Q_values, epochs=1, verbose=0)