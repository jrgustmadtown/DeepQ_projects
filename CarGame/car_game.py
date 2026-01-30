import numpy as np

class GridGame:
    """3x3 grid game with two cars - SEQUENTIAL TURNS"""
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
        
        # Track whose turn (0 = A, 1 = B)
        self.current_turn = 0
    
    def reset(self):
        """Reset game to initial state"""
        self.current_turn = 0
        return self.positions_to_state(0, 8)  # A at 0, B at 8
    
    def state_to_positions(self, state):
        """Convert state number to (pos_a, pos_b)"""
        return divmod(state, self.n_positions)
    
    def positions_to_state(self, pos_a, pos_b):
        """Convert (pos_a, pos_b) to state number"""
        return pos_a * self.n_positions + pos_b
    
    def get_available_actions(self, state, player=None):
        """Get possible moves for current player"""
        if player is None:
            player = self.current_turn
            
        pos_a, pos_b = self.state_to_positions(state)
        
        if player == 0:  # Player A's turn
            current_pos = pos_a
        else:  # Player B's turn
            current_pos = pos_b
            
        row, col = divmod(current_pos, self.grid_size)
        actions = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                actions.append(new_row * self.grid_size + new_col)
        
        return actions
    
    def execute_action(self, state, action):
        """Execute action for current player, return (reward_a, reward_b, next_state)"""
        pos_a, pos_b = self.state_to_positions(state)
        
        if self.current_turn == 0:  # Player A's turn
            # Check if A crashes into B (B is stationary)
            if action == pos_b:  # Crash!
                reward_a = self.crash_reward
                reward_b = -self.crash_reward
                next_state = self.positions_to_state(0, pos_b)  # A resets to corner
            else:
                reward_a = self.position_rewards[action]
                reward_b = -self.position_rewards[action]  # B gets negative of A's reward
                next_state = self.positions_to_state(action, pos_b)
            
            # Switch turn to B
            self.current_turn = 1
            
        else:  # Player B's turn
            # Check if B crashes into A (A is stationary)
            if action == pos_a:  # Crash!
                reward_a = -self.crash_reward
                reward_b = self.crash_reward
                next_state = self.positions_to_state(pos_a, 0)  # B resets to corner
            else:
                reward_a = -self.position_rewards[action]  # A gets negative of B's reward
                reward_b = self.position_rewards[action]
                next_state = self.positions_to_state(pos_a, action)
            
            # Switch turn to A
            self.current_turn = 0
        
        return reward_a, reward_b, next_state
    
    def get_current_player(self):
        """Return which player's turn it is (0=A, 1=B)"""
        return self.current_turn