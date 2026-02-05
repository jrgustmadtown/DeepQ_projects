# Q-Learning Implementation for Markov Decision Process
# Created by: Leo Martinez III in Spring 2024

import random
import matplotlib.pyplot as plt
import numpy as np

# define the constants and the overall grid size/features (all uppercase)
ROWS = 5
COLS = 5
ACTIONS = ['up', 'down', 'left', 'right'] # set of actions available
TERMINAL_STATES = {(0, 4): 10, (1, 2): -2, (2, 3): 8, (4, 2): 6} # fixed values
BLOCKED_STATES = [(1, 1), (1, 3), (1, 4), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4)] # inaccessible
REWARD = -0.5
DISCOUNT_FACTOR = 0.9 # gamma value
LEARNING_RATE = 0.1 # alpha - how much new experiences override old Q-values
EPISODES = 100 # number of training episodes
REPORT_EPISODES = [5, 10, 15, 20, 50, 100] # episodes to report Q-values at
EPSILON = 0.1 # exploration rate (10% chance to explore randomly)

# function to check if a state is valid and not blocked
def is_valid_state(state):
    return (0 <= state[0] < ROWS) and (0 <= state[1] < COLS) and (state not in BLOCKED_STATES)

# function to get next state given current state and action (with stochasticity)
def get_next_state(state, action):
    next_state = state  # default: stay put
    
    # 80% chance: action succeeds
    if random.random() < 0.8:
        if action == 'up':
            next_state = (state[0] - 1, state[1])
        elif action == 'down':
            next_state = (state[0] + 1, state[1])
        elif action == 'left':
            next_state = (state[0], state[1] - 1)
        elif action == 'right':
            next_state = (state[0], state[1] + 1)
    
    # if blocked, stay in current state
    if not is_valid_state(next_state):
        next_state = state
    
    return next_state

# function to get reward for reaching a state
def get_reward(state):
    if state in TERMINAL_STATES:
        return TERMINAL_STATES[state]
    return REWARD

# function to choose action using epsilon-greedy strategy
def choose_action(state, q_values):
    if random.random() < EPSILON:
        return random.choice(ACTIONS) # explore
    else:
        # exploit: choose action with highest Q-value for this state
        best_action = ACTIONS[0]
        best_q = q_values.get((state, best_action), 0)
        for action in ACTIONS[1:]:
            q = q_values.get((state, action), 0)
            if q > best_q:
                best_q = q
                best_action = action
        return best_action

# initialize Q-values to 0 for all state-action pairs
q_values = {}
for row in range(ROWS):
    for col in range(COLS):
        state = (row, col)
        if is_valid_state(state):
            for action in ACTIONS:
                q_values[(state, action)] = 0

# track average state values over episodes for convergence plot
convergence_values = []

# Q-learning training loop
for episode in range(1, EPISODES + 1):
    # random starting state (extract just states from q_values keys which are (state, action) pairs)
    valid_states = [s for (s, a) in q_values.keys() if s not in TERMINAL_STATES]
    state = random.choice(valid_states)
    
    # run episode until terminal state is reached
    while state not in TERMINAL_STATES:
        # choose action using epsilon-greedy
        action = choose_action(state, q_values)
        
        # take action and get reward
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        
        # find best Q-value for next state
        max_q_next = max([q_values.get((next_state, a), 0) for a in ACTIONS])
        
        # Q-learning update: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max Q(s',a') - Q(s,a))
        current_q = q_values.get((state, action), 0)
        q_values[(state, action)] = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_q_next - current_q)
        
        state = next_state
    
    # report Q-values at specified episodes
    if episode in REPORT_EPISODES:
        print(f'\nQ-values after {episode} episodes:')
        # compute state values from Q-values (max Q over actions)
        state_values = {}
        for (s, a), q in q_values.items():
            if s not in state_values or q > state_values[s]:
                state_values[s] = q
        
        for state in sorted(state_values.keys()):
            print(f'{state} Value: {state_values[state]:.2f}')
    
    # track average value for convergence plot
    state_values = {}
    for (s, a), q in q_values.items():
        if s not in state_values or q > state_values[s]:
            state_values[s] = q
    avg_value = np.mean(list(state_values.values())) if state_values else 0
    convergence_values.append(avg_value)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Convergence of average state values
axes[0].plot(range(1, EPISODES + 1), convergence_values, linewidth=2, color='blue')
axes[0].scatter(REPORT_EPISODES, [convergence_values[e-1] for e in REPORT_EPISODES], 
                color='red', s=100, zorder=5, label='Report episodes')
axes[0].set_xlabel('Episode', fontsize=12)
axes[0].set_ylabel('Average State Value', fontsize=12)
axes[0].set_title('Q-Learning Convergence', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Heatmap of final state values
final_state_values = {}
for (s, a), q in q_values.items():
    if s not in final_state_values or q > final_state_values[s]:
        final_state_values[s] = q

# Create grid for heatmap
heatmap = np.zeros((ROWS, COLS))
for row in range(ROWS):
    for col in range(COLS):
        state = (row, col)
        if is_valid_state(state):
            heatmap[row, col] = final_state_values.get(state, 0)
        else:
            heatmap[row, col] = -999  # blocked states

# Create masked array to show blocked states differently
masked_heatmap = np.ma.masked_where(heatmap == -999, heatmap)
im = axes[1].imshow(masked_heatmap, cmap='RdYlGn', interpolation='nearest')
axes[1].set_title('Final State Values Heatmap', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Column', fontsize=12)
axes[1].set_ylabel('Row', fontsize=12)

# Add text annotations to heatmap
for row in range(ROWS):
    for col in range(COLS):
        state = (row, col)
        if is_valid_state(state):
            value = final_state_values.get(state, 0)
            text = axes[1].text(col, row, f'{value:.1f}', ha='center', va='center',
                              color='black', fontsize=9, fontweight='bold')
        else:
            axes[1].text(col, row, 'X', ha='center', va='center',
                        color='black', fontsize=12, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=axes[1])
cbar.set_label('State Value', fontsize=11)

plt.tight_layout()
plt.savefig('q_learning_results.png', dpi=150, bbox_inches='tight')
print('\n✓ Visualization saved as q_learning_results.png')
plt.show()