# Q-Learning Agent Implementation

import random
from config import DISCOUNT_FACTOR, LEARNING_RATE, EPSILON


class QLearningAgent:
    """
    A Q-Learning agent for reinforcement learning.
    
    Uses epsilon-greedy exploration and temporal difference learning
    to learn optimal state-action values.
    
    This agent is game-agnostic - it gets all game info from the environment.
    """
    
    def __init__(self, environment, learning_rate=LEARNING_RATE, 
                 discount_factor=DISCOUNT_FACTOR, epsilon=EPSILON):
        self.env = environment
        self.learning_rate = learning_rate      # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.epsilon = epsilon                  # Exploration rate
        self.actions = environment.actions      # Get actions from environment
        self.q_values = {}
        
        self._initialize_q_values()
    
    def _initialize_q_values(self):
        """Initialize Q-values to 0 for all valid state-action pairs."""
        for state in self.env.get_valid_states():
            for action in self.actions:
                self.q_values[(state, action)] = 0
    
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair."""
        return self.q_values.get((state, action), 0)
    
    def get_max_q_value(self, state):
        """Get the maximum Q-value for a state across all actions."""
        return max(self.get_q_value(state, a) for a in self.actions)
    
    def get_best_action(self, state):
        """Get the action with the highest Q-value for a state."""
        best_action = self.actions[0]
        best_q = self.get_q_value(state, best_action)
        
        for action in self.actions[1:]:
            q = self.get_q_value(state, action)
            if q > best_q:
                best_q = q
                best_action = action
        
        return best_action
    
    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy strategy.
        
        With probability epsilon, choose a random action (explore).
        Otherwise, choose the best action based on Q-values (exploit).
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            return self.get_best_action(state)  # Exploit
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-value using the Q-learning update rule.
        
        Q(s,a) = Q(s,a) + α * (reward + γ * max Q(s',a') - Q(s,a))
        """
        current_q = self.get_q_value(state, action)
        max_q_next = self.get_max_q_value(next_state)
        
        # Temporal difference update
        td_target = reward + self.discount_factor * max_q_next
        td_error = td_target - current_q
        
        self.q_values[(state, action)] = current_q + self.learning_rate * td_error
    
    def get_state_values(self):
        """
        Compute state values from Q-values.
        
        State value V(s) = max_a Q(s, a)
        """
        state_values = {}
        for (state, action), q in self.q_values.items():
            if state not in state_values or q > state_values[state]:
                state_values[state] = q
        return state_values
    
    def train_episode(self):
        """
        Run one episode of training.
        
        Returns:
            int: Number of steps taken in the episode
        """
        state = self.env.reset()
        steps = 0
        
        while not self.env.is_terminal(state):
            # Choose action
            action = self.choose_action(state)
            
            # Take action
            next_state, reward, done = self.env.step(state, action)
            
            # Update Q-values
            self.update(state, action, reward, next_state)
            
            state = next_state
            steps += 1
        
        return steps
    
    def print_q_values(self, episode):
        """Print state values at a given episode."""
        print(f'\nQ-values after {episode} episodes:')
        state_values = self.get_state_values()
        
        for state in sorted(state_values.keys()):
            print(f'{state} Value: {state_values[state]:.2f}')
