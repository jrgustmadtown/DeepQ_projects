import numpy as np
from collections import defaultdict

class NashQLearner:
    """Nash Q-learning agent"""
    
    def __init__(self, agent_id, nash_solver, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.agent_id = agent_id
        self.nash_solver = nash_solver
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    def get_q_value(self, state, action_self, action_opponent):
        return self.q_table[state][action_self][action_opponent]
    
    def set_q_value(self, state, action_self, action_opponent, value):
        self.q_table[state][action_self][action_opponent] = value
    
    def choose_action(self, available_actions, policy=None):
        """Choose action with epsilon-greedy"""
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        elif policy is not None:
            return np.random.choice(list(policy.keys()), p=list(policy.values()))
        else:
            return np.random.choice(available_actions)
    
    def update(self, state, action_self, action_opponent, reward, nash_q_next):
        """Update Q-value using Nash Q-learning"""
        current_q = self.get_q_value(state, action_self, action_opponent)
        new_q = current_q + self.alpha * (reward + self.gamma * nash_q_next - current_q)
        self.set_q_value(state, action_self, action_opponent, new_q)
        return new_q
    
    def decay_epsilon(self, decay=0.995, min_epsilon=0.01):
        self.epsilon = max(min_epsilon, self.epsilon * decay)