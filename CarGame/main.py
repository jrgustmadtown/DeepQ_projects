import numpy as np
import matplotlib.pyplot as plt
from car_game import GridGame
from nash_q_trainer import NashQTrainer

def main():
    print("Nash Q-Learning for 3x3 Grid Game")
    print("="*40)
    
    # Initialize - ONLY pass game, not nash_solver
    game = GridGame()
    trainer = NashQTrainer(game, solver)  
    
    # Train
    trainer.initialize_agents(alpha=0.1, gamma=0.9, epsilon=0.3)
    rewards_a, rewards_b = trainer.train(episodes=2000)
    
    # Results
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Avg A: {np.mean(rewards_a[-100:]):.2f}, Avg B: {np.mean(rewards_b[-100:]):.2f}")
    
    # Test states
    print("\nKey State Policies:")
    
    # State 1: Start state
    state1 = game.get_initial_state()
    policy_a1, policy_b1, q_a1, q_b1 = trainer.get_policy(state1)
    pos_a, pos_b = game.state_to_positions(state1)
    print(f"\nStart: A{pos_a}, B{pos_b}")
    print(f"Nash Q: A={q_a1:.1f}, B={q_b1:.1f}")
    
    # State 2: Risk of crash
    state2 = game.positions_to_state(0, 1)
    policy_a2, policy_b2, q_a2, q_b2 = trainer.get_policy(state2)
    pos_a, pos_b = game.state_to_positions(state2)
    print(f"\nRisk: A{pos_a}, B{pos_b}")
    print(f"Nash Q: A={q_a2:.1f}, B={q_b2:.1f}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Learning curve
    window = 100
    smoothed_a = np.convolve(rewards_a, np.ones(window)/window, mode='valid')
    smoothed_b = np.convolve(rewards_b, np.ones(window)/window, mode='valid')
    ax1.plot(smoothed_a, label='Player A')
    ax1.plot(smoothed_b, label='Player B')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Learning Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Zero-sum check
    total = np.array(rewards_a) + np.array(rewards_b)
    ax2.plot(np.convolve(total, np.ones(100)/100, mode='valid'))
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('A + B Total')
    ax2.set_title('Zero-Sum Check (should be ~0)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Quick simulation
    print("\nSimulation:")
    state = game.get_initial_state()
    for step in range(3):
        pos_a, pos_b = game.state_to_positions(state)
        policy_a, policy_b, _, _ = trainer.get_policy(state)
        
        if policy_a and policy_b:
            action_a = np.random.choice(list(policy_a.keys()), p=list(policy_a.values()))
            action_b = np.random.choice(list(policy_b.keys()), p=list(policy_b.values()))
            
            reward_a, reward_b, state = game.get_reward_and_next_state(state, action_a, action_b)
            print(f"Step {step}: A{pos_a}→{action_a}, B{pos_b}→{action_b}, Rewards: {reward_a},{reward_b}")

if __name__ == "__main__":
    main()