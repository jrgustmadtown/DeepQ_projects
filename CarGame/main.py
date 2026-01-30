import numpy as np
import matplotlib.pyplot as plt
from game import GridGame
from nash_solver import NashSolver
from trainer import NashQTrainer

def main():
    print("Nash Q-Learning for 3x3 Grid Game")
    print("="*40)
    
    # Initialize components
    game = GridGame()
    solver = NashSolver()
    trainer = NashQTrainer(game, solver)
    
    # Create and train agents
    trainer.initialize_agents(alpha=0.1, gamma=0.9, epsilon=0.3)
    rewards_a, rewards_b = trainer.train(episodes=2000)
    
    # Analyze results
    print("\n" + "="*40)
    print("TRAINING COMPLETE")
    print("="*40)
    
    # Final statistics
    print(f"Average rewards (last 100 episodes):")
    print(f"  Player A: {np.mean(rewards_a[-100:]):.2f}")
    print(f"  Player B: {np.mean(rewards_b[-100:]):.2f}")
    print(f"Zero-sum check: {np.mean(np.array(rewards_a) + np.array(rewards_b)):.4f}")
    
    # Analyze key states
    print("\n" + "="*40)
    print("POLICY ANALYSIS")
    print("="*40)
    
    # State 1: Opposite corners
    state1 = game.get_initial_state()
    policy_a1, policy_b1, q_a1, q_b1 = trainer.get_policy(state1)
    
    pos_a, pos_b = game.state_to_positions(state1)
    print(f"\nState: A at {pos_a}, B at {pos_b}")
    print(f"Nash Q-values: A={q_a1:.2f}, B={q_b1:.2f}")
    print("Player A's policy:")
    for action, prob in policy_a1.items():
        if prob > 0.1:
            print(f"  Move to {action}: {prob:.2f}")
    
    # State 2: Risk of crash
    state2 = game.positions_to_state(0, 1)
    policy_a2, policy_b2, q_a2, q_b2 = trainer.get_policy(state2)
    
    pos_a, pos_b = game.state_to_positions(state2)
    print(f"\nState: A at {pos_a}, B at {pos_b} (adjacent)")
    print(f"Nash Q-values: A={q_a2:.2f}, B={q_b2:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    window = 100
    smoothed_a = np.convolve(rewards_a, np.ones(window)/window, mode='valid')
    smoothed_b = np.convolve(rewards_b, np.ones(window)/window, mode='valid')
    plt.plot(smoothed_a, label='Player A')
    plt.plot(smoothed_b, label='Player B')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(rewards_a[-500:], bins=20, alpha=0.5, label='Player A')
    plt.hist(rewards_b[-500:], bins=20, alpha=0.5, label='Player B')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    total_rewards = np.array(rewards_a) + np.array(rewards_b)
    plt.plot(np.convolve(total_rewards, np.ones(100)/100, mode='valid'))
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (A+B)')
    plt.title('Zero-Sum Check')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Simulate a game
    print("\n" + "="*40)
    print("GAME SIMULATION")
    print("="*40)
    
    state = game.get_initial_state()
    total_a, total_b = 0, 0
    
    for step in range(10):
        pos_a, pos_b = game.state_to_positions(state)
        policy_a, policy_b, _, _ = trainer.get_policy(state)
        
        # Choose actions
        if policy_a and policy_b:
            action_a = np.random.choice(list(policy_a.keys()), p=list(policy_a.values()))
            action_b = np.random.choice(list(policy_b.keys()), p=list(policy_b.values()))
            
            reward_a, reward_b, next_state = game.get_reward_and_next_state(
                state, action_a, action_b
            )
            
            total_a += reward_a
            total_b += reward_b
            
            print(f"Step {step + 1}:")
            print(f"  A moves {pos_a} → {action_a}, B moves {pos_b} → {action_b}")
            print(f"  Rewards: A={reward_a}, B={reward_b}")
            
            if action_a == action_b:
                print("  CRASH!")
            
            state = next_state
    
    print(f"\nFinal: A={total_a}, B={total_b}")

if __name__ == "__main__":
    main()