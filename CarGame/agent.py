import random
import numpy as np
from scipy.optimize import linprog

from config import DISCOUNT_FACTOR, LEARNING_RATE, EPSILON


class MinimaxQAgent:
    def __init__(self, environment, player='A', learning_rate=LEARNING_RATE,
                 discount_factor=DISCOUNT_FACTOR, epsilon=EPSILON):
        self.env = environment
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.actions = environment.actions
        self.num_actions = len(self.actions)
        self.q_values = {}
        self.state_values = {}
        self.policy = {}
        self._initialize()

    def _initialize(self):
        for state in self.env.get_all_states():
            self.q_values[state] = np.zeros((self.num_actions, self.num_actions))
            self.state_values[state] = 0.0
            self.policy[state] = self._uniform_policy_for_state(state)

    def _uniform_policy_for_state(self, state):
        pos = state[0] if self.player == 'A' else state[1]
        available = self.env.get_available_actions(pos)
        policy = np.zeros(self.num_actions)
        if available:
            prob = 1.0 / len(available)
            for action in available:
                policy[self.actions.index(action)] = prob
        return policy

    def _available_action_indices(self, pos):
        return [self.actions.index(a) for a in self.env.get_available_actions(pos)]

    def _action_indices_for_state(self, state):
        pos_a, pos_b = state
        return self._available_action_indices(pos_a), self._available_action_indices(pos_b)

    def get_q_matrix(self, state):
        if state not in self.q_values:
            self.q_values[state] = np.zeros((self.num_actions, self.num_actions))
        return self.q_values[state]

    def compute_minimax_value_and_policy(self, state):
        Q = self.get_q_matrix(state)
        idx_a, idx_b = self._action_indices_for_state(state)
        if self.player == 'A':
            n = len(idx_a)
            m = len(idx_b)
            Q_sub = Q[np.ix_(idx_a, idx_b)]
            c = np.zeros(n + 1)
            c[-1] = -1
            A_ub = np.zeros((m, n + 1))
            for j in range(m):
                A_ub[j, :n] = -Q_sub[:, j]
                A_ub[j, -1] = 1
            b_ub = np.zeros(m)
            A_eq = np.zeros((1, n + 1))
            A_eq[0, :n] = 1
            b_eq = np.array([1.0])
            bounds = [(0, 1) for _ in range(n)] + [(None, None)]
        else:
            n = len(idx_b)
            m = len(idx_a)
            Q_sub = Q[np.ix_(idx_a, idx_b)]
            c = np.zeros(n + 1)
            c[-1] = 1
            A_ub = np.zeros((m, n + 1))
            for i in range(m):
                A_ub[i, :n] = Q_sub[i, :]
                A_ub[i, -1] = -1
            b_ub = np.zeros(m)
            A_eq = np.zeros((1, n + 1))
            A_eq[0, :n] = 1
            b_eq = np.array([1.0])
            bounds = [(0, 1) for _ in range(n)] + [(None, None)]

        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                           bounds=bounds, method='highs')
            if result.success:
                policy = result.x[:n]
                policy = np.maximum(policy, 0)
                policy = policy / policy.sum()
                value = result.x[-1]
                policy_full = np.zeros(self.num_actions)
                if self.player == 'A':
                    for i, idx in enumerate(idx_a):
                        policy_full[idx] = policy[i]
                else:
                    for i, idx in enumerate(idx_b):
                        policy_full[idx] = policy[i]
                return value, policy_full
        except:
            pass
        return 0.0, self._uniform_policy_for_state(state)

    def choose_action(self, state):
        pos = state[0] if self.player == 'A' else state[1]
        available = self.env.get_available_actions(pos)
        if random.random() < self.epsilon:
            return random.choice(available)
        if state not in self.policy:
            self._initialize_state(state)
        policy = self.policy[state]
        idxs = [self.actions.index(a) for a in available]
        probs = np.array([policy[i] for i in idxs], dtype=float)
        if probs.sum() <= 0:
            probs = np.ones(len(idxs)) / len(idxs)
        else:
            probs = probs / probs.sum()
        action_idx = np.random.choice(len(idxs), p=probs)
        return available[action_idx]

    def _initialize_state(self, state):
        self.q_values[state] = np.zeros((self.num_actions, self.num_actions))
        self.state_values[state] = 0.0
        self.policy[state] = self._uniform_policy_for_state(state)

    def update(self, state, action_a, action_b, reward, next_state, done):
        if state not in self.q_values:
            self._initialize_state(state)
        idx_a = self.actions.index(action_a)
        idx_b = self.actions.index(action_b)
        if done:
            next_value = 0
        else:
            if next_state not in self.state_values:
                self._initialize_state(next_state)
            next_value = self.state_values[next_state]
        current_q = self.q_values[state][idx_a, idx_b]
        target = reward + self.discount_factor * next_value
        self.q_values[state][idx_a, idx_b] += self.learning_rate * (target - current_q)
        value, policy = self.compute_minimax_value_and_policy(state)
        self.state_values[state] = value
        self.policy[state] = policy

    def get_policy_for_state(self, state):
        if state not in self.policy:
            self._initialize_state(state)
        return {action: prob for action, prob in zip(self.actions, self.policy[state])}


class IndependentQLearningAgent:
    def __init__(self, environment, player='A', learning_rate=LEARNING_RATE,
                 discount_factor=DISCOUNT_FACTOR, epsilon=EPSILON):
        self.env = environment
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.actions = environment.actions
        self.q_values = {}

    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0.0)

    def get_max_q_value(self, state):
        available = self._get_available_actions(state)
        return max(self.get_q_value(state, a) for a in available)

    def get_best_action(self, state):
        available = self._get_available_actions(state)
        best_action = available[0]
        best_q = self.get_q_value(state, best_action)
        for action in available[1:]:
            q = self.get_q_value(state, action)
            if q > best_q:
                best_q = q
                best_action = action
        return best_action

    def choose_action(self, state):
        available = self._get_available_actions(state)
        if random.random() < self.epsilon:
            return random.choice(available)
        return self.get_best_action(state)

    def update(self, state, action, reward, next_state, done):
        current_q = self.get_q_value(state, action)
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.get_max_q_value(next_state)
        self.q_values[(state, action)] = current_q + self.learning_rate * (target - current_q)

    def _get_available_actions(self, state):
        pos = state[0] if self.player == 'A' else state[1]
        return self.env.get_available_actions(pos)
