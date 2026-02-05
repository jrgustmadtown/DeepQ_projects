# Visualization utilities for Q-Learning results

import numpy as np
import matplotlib.pyplot as plt


def plot_results(agent, convergence_values, report_episodes, save_path='q_learning_results.png'):
    """
    Create visualization of Q-learning training results.
    
    Args:
        agent: Trained QLearningAgent
        convergence_values: List of average state values per episode
        report_episodes: List of episode numbers that were reported
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Convergence of average state values
    _plot_convergence(axes[0], convergence_values, report_episodes)
    
    # Plot 2: Heatmap of final state values
    _plot_heatmap(axes[1], agent)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n✓ Visualization saved as {save_path}')
    plt.show()


def _plot_convergence(ax, convergence_values, report_episodes):
    """Plot the convergence of average state values over episodes."""
    episodes = range(1, len(convergence_values) + 1)
    
    ax.plot(episodes, convergence_values, linewidth=2, color='blue')
    ax.scatter(
        report_episodes, 
        [convergence_values[e-1] for e in report_episodes],
        color='red', s=100, zorder=5, label='Report episodes'
    )
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average State Value', fontsize=12)
    ax.set_title('Q-Learning Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_heatmap(ax, agent):
    """Plot a heatmap of final state values."""
    env = agent.env
    state_values = agent.get_state_values()
    
    # Create grid for heatmap
    heatmap = np.zeros((env.rows, env.cols))
    
    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            if env.is_valid_state(state):
                heatmap[row, col] = state_values.get(state, 0)
            else:
                heatmap[row, col] = -999  # Blocked states marker
    
    # Create masked array to show blocked states differently
    masked_heatmap = np.ma.masked_where(heatmap == -999, heatmap)
    im = ax.imshow(masked_heatmap, cmap='RdYlGn', interpolation='nearest')
    
    ax.set_title('Final State Values Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    # Add text annotations
    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            if env.is_valid_state(state):
                value = state_values.get(state, 0)
                ax.text(col, row, f'{value:.1f}', ha='center', va='center',
                       color='black', fontsize=9, fontweight='bold')
            else:
                ax.text(col, row, 'X', ha='center', va='center',
                       color='black', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('State Value', fontsize=11)


def compute_average_value(agent):
    """Compute the average state value across all states."""
    state_values = agent.get_state_values()
    if state_values:
        return np.mean(list(state_values.values()))
    return 0
