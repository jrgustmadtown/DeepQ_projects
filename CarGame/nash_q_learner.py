import numpy as np
from collections import defaultdict

class QTable:
    """Simple Q-table wrapper for sequential game"""
    def __init__(self):
        # Q(s, a) for each player (not Q(s, a1, a2) anymore)
        self.table = defaultdict(lambda: defaultdict(float))
    
    def get(self, state, action):
        return self.table[state][action]
    
    def set(self, state, action, value):
        self.table[state][action] = value

class NashQLearner:
    """Nash Q-learning agent for SEQUENTIAL game"""
    
    def __init__(self, agent_id, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.agent_id = agent_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = QTable()
    
    def get_q_value(self, state, action):
        return self.q_table.get(state, action)
    
    def set_q_value(self, state, action, value):
        self.q_table.set(state, action, value)
    
    def choose_action(self, available_actions, state, opponent_agent=None):
        """Choose action with epsilon-greedy"""
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        
        # Choose best action based on Q-values
        best_action = None
        best_value = -np.inf
        
        for action in available_actions:
            q_value = self.get_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        return best_action or np.random.choice(available_actions)
    
    def update(self, state, action, reward, next_state, next_actions, opponent_agent):
        """Update Q-value for sequential game"""
        current_q = self.get_q_value(state, action)
        
        # Estimate opponent's best response in next state
        if next_actions and opponent_agent:
            # Simple: assume opponent plays greedily
            opponent_q_values = [opponent_agent.get_q_value(next_state, a) 
                               for a in next_actions]
            opponent_best_q = max(opponent_q_values) if opponent_q_values else 0
            future_value = -opponent_best_q  # Negative because opponent's gain is our loss
        else:
            future_value = 0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * future_value - current_q)
        self.set_q_value(state, action, new_q)
        
        return new_q
    
    def decay_epsilon(self, decay=0.995, min_epsilon=0.01):
        self.epsilon = max(min_epsilon, self.epsilon * decay)