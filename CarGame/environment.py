# Grid World Environment for Q-Learning

import random
from config import (
    ROWS, COLS, ACTIONS, TERMINAL_STATES, 
    BLOCKED_STATES, REWARD, ACTION_SUCCESS_PROB
)


class GridWorld:
    """
    A grid world environment for reinforcement learning.
    
    The agent navigates a grid with terminal states (goals), 
    blocked states (obstacles), and stochastic transitions.
    """
    
    def __init__(self):
        self.rows = ROWS
        self.cols = COLS
        self.actions = ACTIONS
        self.terminal_states = TERMINAL_STATES
        self.blocked_states = BLOCKED_STATES
        self.step_reward = REWARD
        self.action_success_prob = ACTION_SUCCESS_PROB
        
    def is_valid_state(self, state):
        """Check if a state is within bounds and not blocked."""
        row, col = state
        in_bounds = (0 <= row < self.rows) and (0 <= col < self.cols)
        not_blocked = state not in self.blocked_states
        return in_bounds and not_blocked
    
    def is_terminal(self, state):
        """Check if a state is a terminal state."""
        return state in self.terminal_states
    
    def get_valid_states(self):
        """Return a list of all valid (non-blocked) states."""
        valid = []
        for row in range(self.rows):
            for col in range(self.cols):
                state = (row, col)
                if self.is_valid_state(state):
                    valid.append(state)
        return valid
    
    def get_non_terminal_states(self):
        """Return valid states that are not terminal."""
        return [s for s in self.get_valid_states() if not self.is_terminal(s)]
    
    def get_next_state(self, state, action):
        """
        Get the next state given current state and action.
        
        Transitions are stochastic: action succeeds with ACTION_SUCCESS_PROB,
        otherwise the agent stays in place.
        """
        next_state = state  # Default: stay put
        
        # Action succeeds with specified probability
        if random.random() < self.action_success_prob:
            row, col = state
            if action == 'up':
                next_state = (row - 1, col)
            elif action == 'down':
                next_state = (row + 1, col)
            elif action == 'left':
                next_state = (row, col - 1)
            elif action == 'right':
                next_state = (row, col + 1)
        
        # If next state is invalid, stay in current state
        if not self.is_valid_state(next_state):
            next_state = state
        
        return next_state
    
    def get_reward(self, state):
        """Get the reward for reaching a state."""
        if state in self.terminal_states:
            return self.terminal_states[state]
        return self.step_reward
    
    def reset(self):
        """Reset environment and return a random starting state."""
        non_terminal = self.get_non_terminal_states()
        return random.choice(non_terminal)
    
    def step(self, state, action):
        """
        Take an action from a state.
        
        Returns:
            next_state: The resulting state
            reward: The reward received
            done: Whether the episode is finished (terminal state reached)
        """
        next_state = self.get_next_state(state, action)
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        return next_state, reward, done
