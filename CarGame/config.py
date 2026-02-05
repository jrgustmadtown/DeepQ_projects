# Configuration constants for the Q-Learning Grid World

# Grid dimensions
ROWS = 5
COLS = 5

# Actions available to the agent
ACTIONS = ['up', 'down', 'left', 'right']

# Terminal states with their reward values
TERMINAL_STATES = {
    (0, 4): 10,
    (1, 2): -2,
    (2, 3): 8,
    (4, 2): 6
}

# Blocked/inaccessible states
BLOCKED_STATES = [
    (1, 1), (1, 3), (1, 4),
    (2, 4),
    (3, 1), (3, 2), (3, 3), (3, 4),
    (4, 3), (4, 4)
]

# Rewards and learning parameters
REWARD = -0.5           # Default step reward
DISCOUNT_FACTOR = 0.9   # Gamma - future reward discount
LEARNING_RATE = 0.1     # Alpha - how much new experiences override old Q-values
EPSILON = 0.1           # Exploration rate (10% chance to explore randomly)

# Training parameters
EPISODES = 100
REPORT_EPISODES = [5, 10, 15, 20, 50, 100]

# Environment stochasticity
ACTION_SUCCESS_PROB = 0.8  # 80% chance action succeeds as intended
