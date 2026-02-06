import random
import warnings
import numpy as np
import nashpy as nash

from config import DISCOUNT_FACTOR, LEARNING_RATE, EPSILON, Q_INIT_SCALE, FP_ITERATIONS


class NashQAgent:
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
            self.q_values[state] = np.random.uniform(
                low=-Q_INIT_SCALE, high=Q_INIT_SCALE,
                size=(self.num_actions, self.num_actions)
            )
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
            self.q_values[state] = np.random.uniform(
                low=-Q_INIT_SCALE, high=Q_INIT_SCALE,
                size=(self.num_actions, self.num_actions)
            )
        return self.q_values[state]

    def _get_opponent_q(self, opponent_q_values, state):
        if state in opponent_q_values:
            return opponent_q_values[state]
        return np.zeros((self.num_actions, self.num_actions))

    def compute_nash_value_and_policy(self, state, opponent_q_values):
        idx_a, idx_b = self._action_indices_for_state(state)
        Q_a = self.get_q_matrix(state)
        Q_b = self._get_opponent_q(opponent_q_values, state)
        Q_a_sub = Q_a[np.ix_(idx_a, idx_b)]
        Q_b_sub = Q_b[np.ix_(idx_a, idx_b)]

        if Q_a_sub.size == 0 or Q_b_sub.size == 0:
            return 0.0, self._uniform_policy_for_state(state)

        try:
            game = nash.Game(Q_a_sub, Q_b_sub)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                equilibria = list(game.support_enumeration())
        except:
            equilibria = []

        if not equilibria or len(equilibria) != 1:
            pi_a, pi_b = self._fictitious_play(Q_a_sub, Q_b_sub)
        else:
            pi_a, pi_b = equilibria[0]

        pi_a = np.array(pi_a, dtype=float)
        pi_b = np.array(pi_b, dtype=float)
        if pi_a.sum() <= 0 or pi_b.sum() <= 0:
            return 0.0, self._uniform_policy_for_state(state)

        pi_a = pi_a / pi_a.sum()
        pi_b = pi_b / pi_b.sum()
        value_a = float(pi_a @ Q_a_sub @ pi_b)
        value_b = float(pi_a @ Q_b_sub @ pi_b)

        policy_a_full = np.zeros(self.num_actions)
        policy_b_full = np.zeros(self.num_actions)
        for i, idx in enumerate(idx_a):
            policy_a_full[idx] = pi_a[i]
        for i, idx in enumerate(idx_b):
            policy_b_full[idx] = pi_b[i]

        if self.player == 'A':
            return value_a, policy_a_full
        return value_b, policy_b_full

    def _fictitious_play(self, Q_a_sub, Q_b_sub):
        game = nash.Game(Q_a_sub, Q_b_sub)
        try:
            fp_iter = game.fictitious_play(iterations=FP_ITERATIONS)
            last = None
            for last in fp_iter:
                pass
            if last is None:
                return np.ones(Q_a_sub.shape[0]) / Q_a_sub.shape[0], np.ones(Q_a_sub.shape[1]) / Q_a_sub.shape[1]
            pi_a, pi_b = last
            return pi_a, pi_b
        except:
            return np.ones(Q_a_sub.shape[0]) / Q_a_sub.shape[0], np.ones(Q_a_sub.shape[1]) / Q_a_sub.shape[1]

    def count_equilibria(self, state, opponent_q_values):
        idx_a, idx_b = self._action_indices_for_state(state)
        Q_a = self.get_q_matrix(state)
        Q_b = self._get_opponent_q(opponent_q_values, state)
        Q_a_sub = Q_a[np.ix_(idx_a, idx_b)]
        Q_b_sub = Q_b[np.ix_(idx_a, idx_b)]
        if Q_a_sub.size == 0 or Q_b_sub.size == 0:
            return 0
        try:
            game = nash.Game(Q_a_sub, Q_b_sub)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                equilibria = list(game.support_enumeration())
            return len(equilibria)
        except:
            return 0

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
        self.q_values[state] = np.random.uniform(
            low=-Q_INIT_SCALE, high=Q_INIT_SCALE,
            size=(self.num_actions, self.num_actions)
        )
        self.state_values[state] = 0.0
        self.policy[state] = self._uniform_policy_for_state(state)

    def update(self, state, action_a, action_b, reward_a, reward_b, next_state, done, opponent_q_values):
        if state not in self.q_values:
            self._initialize_state(state)

        idx_a = self.actions.index(action_a)
        idx_b = self.actions.index(action_b)
        reward = reward_a if self.player == 'A' else reward_b

        if done:
            next_value = 0
        else:
            next_value, _ = self.compute_nash_value_and_policy(next_state, opponent_q_values)

        current_q = self.q_values[state][idx_a, idx_b]
        target = reward + self.discount_factor * next_value
        self.q_values[state][idx_a, idx_b] += self.learning_rate * (target - current_q)

        value, policy = self.compute_nash_value_and_policy(state, opponent_q_values)
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
