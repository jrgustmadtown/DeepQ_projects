import os
import numpy as np
import matplotlib.pyplot as plt

from config import CRASH_REWARD


def plot_training_results(convergence_a, convergence_b, env, save_path='car_game_results.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _plot_convergence(axes[0, 0], convergence_a, convergence_b)
    _plot_reward_grid(axes[0, 1], env)
    _plot_game_value(axes[1, 0], convergence_a)
    _plot_game_info(axes[1, 1], env)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def _plot_convergence(ax, values_a, values_b):
    episodes = range(1, len(values_a) + 1)
    ax.plot(episodes, values_a, linewidth=2, color='blue', label='Agent A', alpha=0.7)
    ax.plot(episodes, values_b, linewidth=2, color='red', label='Agent B', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Q-Value', fontsize=12)
    ax.set_title('Training Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_reward_grid(ax, env):
    n = env.grid_size
    rewards = np.zeros((n, n))
    for row in range(n):
        for col in range(n):
            rewards[row, col] = env.square_rewards[(row, col)]
    im = ax.imshow(rewards, cmap='YlOrRd', interpolation='nearest')
    for row in range(n):
        for col in range(n):
            value = int(rewards[row, col])
            ax.text(col, row, str(value), ha='center', va='center',
                   fontsize=12, fontweight='bold')
    ax.plot(0, 0, 'bs', markersize=20, markerfacecolor='none', markeredgewidth=3, label='A start')
    ax.plot(n-1, n-1, 'rs', markersize=20, markerfacecolor='none', markeredgewidth=3, label='B start')
    ax.set_title('Square Rewards (2^distance from corner)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Reward Value', fontsize=11)
    ax.legend(loc='upper right')


def _plot_game_value(ax, values_a):
    episodes = range(1, len(values_a) + 1)
    window = min(50, len(values_a) // 10 + 1)
    if window > 1:
        smoothed = np.convolve(values_a, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(values_a)+1), smoothed, linewidth=2, color='green', label='Smoothed')
    ax.plot(episodes, values_a, alpha=0.3, color='green', label='Raw')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Fair game')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Game Value (A perspective)', fontsize=12)
    ax.set_title('Estimated Game Value', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_game_info(ax, env):
    ax.axis('off')
    info_text = f"""
    General-Sum Car Game

    Grid Size: {env.grid_size} x {env.grid_size}

    Rules:
    - Agents start at opposite corners
    - Move simultaneously each turn
    - Square reward = 2^(dist to nearest corner)
    - General-sum rewards for each agent

    Crash (same square):
    - Agent A: -{CRASH_REWARD}
    - Agent B: -{CRASH_REWARD}

    Actions: {', '.join(env.actions)}

    Algorithm: Nash-Q Learning
    (General-Sum Nash Equilibrium)
    """
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_policy(agent, env, state=None, save_path='policy_visualization.png'):
    if state is None:
        state = env.reset()
    policy = agent.get_policy_for_state(state)
    fig, ax = plt.subplots(figsize=(8, 6))
    actions = list(policy.keys())
    probs = list(policy.values())
    colors = ['#2ecc71' if p == max(probs) else '#3498db' for p in probs]
    bars = ax.bar(actions, probs, color=colors, edgecolor='black')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'Agent {agent.player} Policy at State {state}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{prob:.2f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_state_policy(agent_a, agent_b, env, state, save_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    n = env.grid_size
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(True, alpha=0.4)
    ax.set_aspect('equal')
    pos_a, pos_b = state
    ax.scatter([pos_a[1]], [pos_a[0]], color='blue', s=80)
    ax.scatter([pos_b[1]], [pos_b[0]], color='red', s=80)
    policy_a = agent_a.get_policy_for_state(state)
    policy_b = agent_b.get_policy_for_state(state)
    _draw_policy_arrows(ax, pos_a, policy_a, 'blue')
    _draw_policy_arrows(ax, pos_b, policy_b, 'red')
    ax.set_title(f"{pos_a[0]},{pos_a[1]},{pos_b[0]},{pos_b[1]}", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_all_state_policies(agent_a, agent_b, env, output_dir='policy_states'):
    os.makedirs(output_dir, exist_ok=True)
    for state in env.get_all_states():
        pos_a, pos_b = state
        filename = f"A{pos_a[0]}_{pos_a[1]}_B{pos_b[0]}_{pos_b[1]}.png"
        save_path = os.path.join(output_dir, filename)
        plot_state_policy(agent_a, agent_b, env, state, save_path)


def compute_average_value(agent):
    if hasattr(agent, 'state_values'):
        values = list(agent.state_values.values())
    else:
        values = list(agent.q_values.values()) if agent.q_values else [0]
    return np.mean(values) if values else 0.0


def _draw_policy_arrows(ax, pos, policy, color):
    action_map = {
        'up': (0, -1),
        'down': (0, 1),
        'left': (-1, 0),
        'right': (1, 0)
    }
    x, y = pos[1], pos[0]
    for action, prob in policy.items():
        if prob <= 0 or action not in action_map:
            continue
        dx, dy = action_map[action]
        ax.arrow(x, y, dx * prob, dy * prob,
                 head_width=0.08, head_length=0.08,
                 length_includes_head=True, color=color, alpha=0.9)
