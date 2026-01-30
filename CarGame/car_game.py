import numpy as np

class GridGame:
    """3x3 grid game with two cars - game mechanics only"""
    def __init__(self):
        self.grid_size = 3
        self.n_positions = 9
        
        # Rewards
        self.position_rewards = {
            0: 1, 2: 1, 6: 1, 8: 1,  # corners
            1: 2, 3: 2, 5: 2, 7: 2,   # edges
            4: 5                      # center
        }
        self.crash_reward = 10
    
    def state_to_positions(self, state):
        """Convert state number to (pos_a, pos_b)"""
        return divmod(state, self.n_positions)
    
    def positions_to_state(self, pos_a, pos_b):
        """Convert (pos_a, pos_b) to state number"""
        return pos_a * self.n_positions + pos_b
    
    def get_available_actions(self, position):
        """Get possible moves from a position"""
        row, col = divmod(position, self.grid_size)
        actions = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                actions.append(new_row * self.grid_size + new_col)
        
        return actions
    
    def get_reward_and_next_state(self, state, action_a, action_b):
        """Execute actions and return results"""
        pos_a, pos_b = self.state_to_positions(state)
        
        # Check for crash
        if action_a == action_b:
            reward_a = self.crash_reward
            reward_b = -self.crash_reward
            next_state = self.positions_to_state(0, pos_b)  # A resets to corner
        else:
            reward_a = self.position_rewards[action_a]
            reward_b = self.position_rewards[action_b]
            next_state = self.positions_to_state(action_a, action_b)
        
        return reward_a, reward_b, next_state
    
    def get_initial_state(self):
        """Start state: cars in opposite corners"""
        return self.positions_to_state(0, 8)