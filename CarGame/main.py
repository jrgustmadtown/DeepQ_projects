import numpy as np
import matplotlib.pyplot as plt
import time
from car_game import GridGame
from nash_q_trainer import NashQTrainer

def visualize_nash_policies(game, trainer):
    """Create policy visualization grids"""
    print("\n" + "="*60)
    print("NASH EQUILIBRIUM POLICY VISUALIZATION")
    print("="*60)
    
    # Create a figure with key positions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Define key positions to visualize
    positions = [
        (0, 8, 0, "Start: A in TL, B in BR, A's turn"),
        (0, 4, 0, "A in TL, B in Center, A's turn"),
        (0, 1, 0, "A in TL, B in TM, A's turn"),
        (4, 8, 0, "A in Center, B in BR, A's turn"),
        (0, 8, 1, "Start: A in TL, B in BR, B's turn"),
        (4, 1, 1, "A in Center, B in TM, B's turn"),
    ]
    
    position_names = {
        0: "TL", 1: "TM", 2: "TR",
        3: "ML", 4: "C", 5: "MR",
        6: "BL", 7: "BM", 8: "BR"
    }
    
    for idx, (pos_a, pos_b, player, title) in enumerate(positions):
        ax = axes[idx]
        state = game.positions_to_state(pos_a, pos_b)
        policy, expected_value = trainer.get_policy(state, player)
        
        # Create grid
        grid = np.zeros((3, 3))
        for action, prob in policy.items():
            row, col = divmod(action, 3)
            grid[row, col] = prob
        
        # Plot heatmap
        im = ax.imshow(grid, cmap='YlOrRd', vmin=0, vmax=1)
        
        # Add probability text
        for i in range(3):
            for j in range(3):
                prob = grid[i, j]
                if prob > 0.01:
                    ax.text(j, i, f'{prob:.2f}', ha='center', va='center',
                           fontsize=10, fontweight='bold',
                           color='black' if prob < 0.5 else 'white')
        
        # Mark player positions
        row_a, col_a = divmod(pos_a, 3)
        row_b, col_b = divmod(pos_b, 3)
        
        ax.text(col_a, row_a, 'A', ha='center', va='center',
               fontsize=14, fontweight='bold', color='blue',
               bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        ax.text(col_b, row_b, 'B', ha='center', va='center',
               fontsize=14, fontweight='bold', color='red',
               bbox=dict(boxstyle="square,pad=0.3", facecolor='lightcoral', alpha=0.7))
        
        # Add turn indicator and expected value
        turn = "A" if player == 0 else "B"
        ax.set_title(f"{title}\n{turn}'s turn: E[V] = {expected_value:.2f}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid lines
        for i in range(4):
            ax.axhline(i - 0.5, color='black', linewidth=1)
            ax.axvline(i - 0.5, color='black', linewidth=1)
    
    # Remove any unused axes
    for idx in range(len(positions), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Learned Nash Equilibrium Policies", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('learned_policies.png', dpi=150, bbox_inches='tight')
    print("✓ Policy visualizations saved as 'learned_policies.png'")
    
    # Create arrow-based visualization
    print("\nCreating arrow-based policy table...")
    create_arrow_policies(game, trainer, position_names)

def create_arrow_policies(game, trainer, position_names):
    """Create arrow-based policy visualization"""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    # Reduced set of unique positions (avoiding symmetry)
    positions_a = [0, 1, 4]  # Corner, edge, center
    positions_b = [0, 4, 8]  # Different positions
    
    for i, pos_a in enumerate(positions_a):
        for j, pos_b in enumerate(positions_b):
            ax = axes[i, j]
            
            # Skip if same position or if we want to show A > B cases only
            if pos_a == pos_b or pos_a > pos_b:
                ax.axis('off')
                continue
            
            state = game.positions_to_state(pos_a, pos_b)
            policy, expected_value = trainer.get_policy(state, player=0)
            
            # Setup grid
            ax.set_xlim(-0.5, 2.5)
            ax.set_ylim(-0.5, 2.5)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            
            # Draw grid
            for x in range(4):
                ax.axvline(x - 0.5, color='gray', linewidth=1, alpha=0.5)
            for y in range(4):
                ax.axhline(y - 0.5, color='gray', linewidth=1, alpha=0.5)
            
            # Mark players
            row_a, col_a = divmod(pos_a, 3)
            row_b, col_b = divmod(pos_b, 3)
            
            ax.scatter(col_a, row_a, s=400, color='blue', alpha=0.7, 
                      marker='o', edgecolors='darkblue', linewidth=2)
            ax.text(col_a, row_a, 'A', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='white')
            
            ax.scatter(col_b, row_b, s=400, color='red', alpha=0.7,
                      marker='s', edgecolors='darkred', linewidth=2)
            ax.text(col_b, row_b, 'B', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white')
            
            # Draw top 2 moves
            sorted_moves = sorted(policy.items(), key=lambda x: -x[1])
            for action, prob in sorted_moves[:2]:
                if prob > 0.05:
                    row_to, col_to = divmod(action, 3)
                    
                    # Draw arrow
                    ax.arrow(col_a, row_a, 
                            (col_to - col_a) * 0.7,
                            (row_to - row_a) * 0.7,
                            head_width=0.15, head_length=0.2,
                            fc='green', ec='darkgreen', alpha=0.7,
                            length_includes_head=True)
                    
                    # Add probability
                    mid_x = col_a + (col_to - col_a) * 0.35
                    mid_y = row_a + (row_to - row_a) * 0.35
                    ax.text(mid_x, mid_y, f'{prob:.2f}',
                           fontsize=9, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.2", 
                                    facecolor='white', alpha=0.8))
            
            # Title
            title = f"A: {position_names[pos_a]}, B: {position_names[pos_b]}\nE[V]={expected_value:.1f}"
            ax.set_title(title, fontsize=11)
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle("Nash Policy Arrows (A's turn) - Green arrows show optimal moves", 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('policy_arrows.png', dpi=150, bbox_inches='tight')
    print("✓ Arrow visualizations saved as 'policy_arrows.png'")

def main():
    print("Nash Q-Learning for 3x3 Grid Game - SEQUENTIAL TURNS")
    print("="*40)
    print("A moves → gets ±reward, B gets ∓reward")
    print("Then B moves → gets ±reward, A gets ∓reward")
    print("="*40)
    
    # Initialize
    game = GridGame()
    trainer = NashQTrainer(game)
    
    # Train with timing
    print("\nStarting training...")
    start_time = time.time()
    
    trainer.initialize_agents(
        alpha=0.1,      # Learning rate
        gamma=0.9,      # Discount factor
        epsilon=0.3     # Exploration rate
    )
    
    rewards_a, rewards_b = trainer.train(episodes=2000)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"That's {2000/training_time:.1f} episodes per second!")
    
    # Results
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Final 100 episodes:")
    print(f"  Avg A: {np.mean(rewards_a[-100:]):.2f}")
    print(f"  Avg B: {np.mean(rewards_b[-100:]):.2f}")
    print(f"  Zero-sum check (A+B): {np.mean(np.array(rewards_a[-100:]) + np.array(rewards_b[-100:])):.2f}")
    
    # Convergence analysis
    print("\n" + "="*40)
    print("CONVERGENCE ANALYSIS")
    print("="*40)
    
    # Check learning progress
    early_avg_a = np.mean(rewards_a[:100])  # First 100 episodes
    late_avg_a = np.mean(rewards_a[-100:])  # Last 100 episodes
    change_a = late_avg_a - early_avg_a
    
    print(f"Player A learning progress:")
    print(f"  Early episodes (1-100): {early_avg_a:.2f}")
    print(f"  Late episodes (1901-2000): {late_avg_a:.2f}")
    print(f"  Change: {change_a:.2f}")
    
    if abs(change_a) > 20:
        print("  ✓ Significant learning occurred!")
    elif abs(change_a) > 5:
        print("  ✓ Moderate learning occurred")
    else:
        print("  ⚠ Little change - might have converged quickly")
    
    # Create convergence plots
    print("\n" + "="*40)
    print("GENERATING CONVERGENCE PLOTS")
    print("="*40)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Raw reward signals
    axes[0, 0].plot(rewards_a, alpha=0.3, label='A raw', color='blue', linewidth=0.5)
    axes[0, 0].plot(rewards_b, alpha=0.3, label='B raw', color='red', linewidth=0.5)
    
    # Add smoothed versions
    window = 50
    if len(rewards_a) > window:
        smoothed_a = np.convolve(rewards_a, np.ones(window)/window, mode='valid')
        smoothed_b = np.convolve(rewards_b, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards_a)), smoothed_a, 
                       label='A smoothed', color='blue', linewidth=2)
        axes[0, 0].plot(range(window-1, len(rewards_b)), smoothed_b,
                       label='B smoothed', color='red', linewidth=2)
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Raw Learning Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Moving averages (convergence)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    window_sizes = [10, 50, 100]
    colors_a = ['lightblue', 'blue', 'darkblue']
    colors_b = ['lightcoral', 'red', 'darkred']
    
    for window, color_a, color_b in zip(window_sizes, colors_a, colors_b):
        if len(rewards_a) > window:
            ma_a = np.convolve(rewards_a, np.ones(window)/window, mode='valid')
            ma_b = np.convolve(rewards_b, np.ones(window)/window, mode='valid')
            x_vals = range(window-1, len(rewards_a))
            axes[0, 1].plot(x_vals, ma_a, color=color_a, 
                           label=f'A MA({window})', alpha=0.8, linewidth=1.5)
            axes[0, 1].plot(x_vals, ma_b, color=color_b,
                           label=f'B MA({window})', alpha=0.8, linewidth=1.5, linestyle=':')
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Moving Average Reward')
    axes[0, 1].set_title('Moving Average Convergence')
    axes[0, 1].legend(ncol=2, fontsize='small')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Zero-sum check over time
    total_rewards = np.array(rewards_a) + np.array(rewards_b)
    
    # Plot raw sum
    axes[0, 2].plot(total_rewards, alpha=0.3, color='green', linewidth=0.5, label='Raw sum')
    
    # Plot moving average of sum
    if len(total_rewards) > 100:
        ma_total = np.convolve(total_rewards, np.ones(100)/100, mode='valid')
        axes[0, 2].plot(range(99, len(total_rewards)), ma_total, 
                       color='darkgreen', linewidth=2, label='MA(100)')
    
    axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    axes[0, 2].axhline(y=np.mean(total_rewards), color='red', linestyle='--', 
                      alpha=0.7, linewidth=1, label=f'Mean: {np.mean(total_rewards):.2f}')
    axes[0, 2].fill_between(range(len(total_rewards)), -5, 5, alpha=0.1, color='gray',
                           label='±5 tolerance')
    
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('A + B Total Reward')
    axes[0, 2].set_title(f'Zero-Sum Check (Mean: {np.mean(total_rewards):.3f})')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([-50, 50])  # Fixed scale for comparison
    
    # 4. Reward distribution
    axes[1, 0].hist(rewards_a, bins=50, alpha=0.5, label='Player A', color='blue', density=True)
    axes[1, 0].hist(rewards_b, bins=50, alpha=0.5, label='Player B', color='red', density=True)
    axes[1, 0].axvline(x=np.mean(rewards_a), color='blue', linestyle='--', linewidth=2,
                      label=f'A mean: {np.mean(rewards_a):.1f}')
    axes[1, 0].axvline(x=np.mean(rewards_b), color='red', linestyle='--', linewidth=2,
                      label=f'B mean: {np.mean(rewards_b):.1f}')
    axes[1, 0].set_xlabel('Total Reward per Episode')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Cumulative rewards (for dominance check)
    cum_a = np.cumsum(rewards_a)
    cum_b = np.cumsum(rewards_b)
    
    axes[1, 1].plot(cum_a, label='Cumulative A', color='blue', linewidth=2)
    axes[1, 1].plot(cum_b, label='Cumulative B', color='red', linewidth=2)
    axes[1, 1].plot(cum_a + cum_b, label='Cumulative Total', color='green', 
                   linewidth=2, linestyle='--', alpha=0.7)
    
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].set_title('Cumulative Rewards')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Learning rate visualization
    # Simulate what epsilon decay would look like
    episodes = np.arange(2000)
    epsilon_init = 0.3
    epsilon_decay = 0.995
    epsilon_values = epsilon_init * (epsilon_decay ** episodes)
    epsilon_values = np.maximum(epsilon_values, 0.01)
    
    axes[1, 2].plot(episodes, epsilon_values, color='purple', linewidth=2, label='ε (exploration)')
    
    # Add alpha decay if implemented
    alpha_init = 0.1
    alpha_decay = 0.999
    alpha_values = alpha_init * (alpha_decay ** episodes)
    alpha_values = np.maximum(alpha_values, 0.01)
    axes[1, 2].plot(episodes, alpha_values, color='orange', linewidth=2, label='α (learning rate)')
    
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Parameter Value')
    axes[1, 2].set_title('Learning Parameters Decay')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_ylim([0.005, 0.5])
    
    plt.suptitle(f'Nash Q-Learning Convergence Analysis ({2000} episodes, {training_time:.1f}s)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Convergence plots saved as 'convergence_analysis.png'")
    
    # Show plots (this might not work in all environments)
    try:
        plt.show()
    except Exception as e:
        print(f"Note: Could not display plots interactively: {e}")
        print("Plots have been saved to 'convergence_analysis.png' instead.")
    
    # NOW ADD THE POLICY VISUALIZATION
    visualize_nash_policies(game, trainer)
    
    # Policy analysis
    print("\n" + "="*40)
    print("POLICY ANALYSIS")
    print("="*40)
    
    # Start state, A's turn
    game.reset()
    state = game.positions_to_state(0, 8)
    
    policy_a, value_a = trainer.get_policy(state, player=0)
    print(f"\nStart state: A at 0, B at 8, A's turn")
    print(f"Expected value for A: {value_a:.2f}")
    print("A's policy (moves with >5% probability):")
    
    position_names = {
        0: "Top-Left", 1: "Top-Middle", 2: "Top-Right",
        3: "Middle-Left", 4: "Center", 5: "Middle-Right", 
        6: "Bottom-Left", 7: "Bottom-Middle", 8: "Bottom-Right"
    }
    
    for action, prob in sorted(policy_a.items(), key=lambda x: -x[1]):
        if prob > 0.05:
            print(f"  Move to {position_names[action]} [pos {action}]: {prob:.3f}")
    
    # Check if policies make sense
    print("\n" + "="*40)
    print("STRATEGY DIAGNOSTIC")
    print("="*40)
    
    # Test risky state
    risky_state = game.positions_to_state(0, 1)  # A at 0, B at 1 (adjacent!)
    risky_policy, risky_value = trainer.get_policy(risky_state, player=0)
    
    print(f"\nHigh-risk state: A at Top-Left (0), B at Top-Middle (1)")
    print(f"Expected value for A: {risky_value:.2f}")
    
    if 1 in risky_policy and risky_policy[1] > 0.1:
        print(f"⚠ A considers crashing into B with probability {risky_policy[1]:.2f}")
    else:
        print("✓ A avoids crashing into adjacent B")
    
    # Test center temptation
    center_state = game.positions_to_state(3, 8)  # A at middle-left, B at bottom-right
    center_policy, center_value = trainer.get_policy(center_state, player=0)
    
    print(f"\nCenter opportunity: A at Middle-Left (3), B at Bottom-Right (8)")
    print(f"Expected value for A: {center_value:.2f}")
    
    if 4 in center_policy and center_policy[4] > 0.3:
        print(f"✓ A goes for center with probability {center_policy[4]:.2f}")
    else:
        print("⚠ A avoids center despite opportunity")
    
    # Simulation
    print("\n" + "="*40)
    print("FINAL SIMULATION (6 turns)")
    print("="*40)
    
    game.reset()
    state = game.positions_to_state(0, 8)
    total_a, total_b = 0, 0
    
    print("\nTurn | Player | Move | Rewards (A, B) | Total (A, B)")
    print("-" * 60)
    
    for turn in range(6):
        player = game.get_current_player()
        
        if player == 0:
            policy, _ = trainer.get_policy(state, player=0)
            if policy:
                action = np.random.choice(list(policy.keys()), p=list(policy.values()))
                reward_a, reward_b, next_state = game.execute_action(state, action)
                total_a += reward_a
                total_b += reward_b
                
                pos_a, pos_b = game.state_to_positions(state)
                print(f"{turn+1:4d} | {'A':7s} | {position_names[pos_a]} → {position_names[action]} | "
                      f"({reward_a:3d}, {reward_b:3d}) | ({total_a:4d}, {total_b:4d})")
                state = next_state
        else:
            policy, _ = trainer.get_policy(state, player=1)
            if policy:
                action = np.random.choice(list(policy.keys()), p=list(policy.values()))
                reward_a, reward_b, next_state = game.execute_action(state, action)
                total_a += reward_a
                total_b += reward_b
                
                pos_a, pos_b = game.state_to_positions(state)
                print(f"{turn+1:4d} | {'B':7s} | {position_names[pos_b]} → {position_names[action]} | "
                      f"({reward_a:3d}, {reward_b:3d}) | ({total_a:4d}, {total_b:4d})")
                state = next_state
    
    print("-" * 60)
    print(f"Final: A={total_a}, B={total_b}, Sum={total_a+total_b}")
    
    # Final assessment
    print("\n" + "="*40)
    print("FINAL ASSESSMENT")
    print("="*40)
    
    zero_sum_error = np.abs(np.mean(total_rewards))
    reward_std = np.std(rewards_a[-100:])
    
    if zero_sum_error < 1.0:
        print("✓ EXCELLENT: Perfect zero-sum property maintained")
    elif zero_sum_error < 5.0:
        print("✓ GOOD: Reasonable zero-sum approximation")
    else:
        print("⚠ POOR: Zero-sum property not well maintained")
    
    if reward_std < 20:
        print("✓ EXCELLENT: Stable learning (low variance)")
    elif reward_std < 50:
        print("✓ GOOD: Moderate stability")
    else:
        print("⚠ POOR: High variance in learning")
    
    if abs(change_a) > 10:
        print("✓ CLEAR LEARNING: Significant improvement over time")
    else:
        print("⚠ MINIMAL LEARNING: Little change from start to end")
    
    print("\n" + "="*60)
    print("VISUALIZATION FILES CREATED:")
    print("="*60)
    print("1. convergence_analysis.png - Learning curves and metrics")
    print("2. learned_policies.png - Policy heatmaps for key positions")
    print("3. policy_arrows.png - Arrow-based policy visualizations")
    print("="*60)

if __name__ == "__main__":
    main()