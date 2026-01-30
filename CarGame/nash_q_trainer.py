import numpy as np
from collections import defaultdict

class NashQTrainer:
    """Coordinates Nash Q-learning training"""
    
    def __init__(self, game, nash_solver):
        self.game = game
        self.nash_solver = nash_solver
        self.agent_a = None
        self.agent_b = None
        
    def initialize_agents(self, alpha=0.1, gamma=0.9, epsilon=0.3):
        """Create both agents"""
        self.agent_a = NashQLearner(0, self.nash_solver, alpha, gamma, epsilon)
        self.agent_b = NashQLearner(1, self.nash_solver, alpha, gamma, epsilon)
    
    def train_episode(self, max_steps=50):
        """Train for one episode"""
        state = self.game.get_initial_state()
        total_reward_a = 0
        total_reward_b = 0
        
        for _ in range(max_steps):
            pos_a, pos_b = self.game.state_to_positions(state)
            
            # Available actions
            actions_a = self.game.get_available_actions(pos_a)
            actions_b = self.game.get_available_actions(pos_b)
            
            if not actions_a or not actions_b:
                break
            
            # Build payoff matrices
            payoff_a = self.nash_solver.compute_payoff_matrix(
                state, actions_a, actions_b, self.agent_a.get_q_value, 0
            )
            payoff_b = self.nash_solver.compute_payoff_matrix(
                state, actions_a, actions_b, self.agent_b.get_q_value, 1
            )
            
            # Compute Nash equilibrium
            policy_a_vec, policy_b_vec, nash_q_a, nash_q_b = self.nash_solver.solve_zero_sum(payoff_a)
            
            # Convert to policy dictionaries
            policy_a = {action: prob for action, prob in zip(actions_a, policy_a_vec)}
            policy_b = {action: prob for action, prob in zip(actions_b, policy_b_vec)}
            
            # Choose actions
            action_a = self.agent_a.choose_action(actions_a, policy_a)
            action_b = self.agent_b.choose_action(actions_b, policy_b)
            
            # Get rewards and next state
            reward_a, reward_b, next_state = self.game.get_reward_and_next_state(
                state, action_a, action_b
            )
            
            total_reward_a += reward_a
            total_reward_b += reward_b
            
            # Compute Nash Q for next state
            next_nash_q_a, next_nash_q_b = 0, 0
            if next_state != state:
                next_pos_a, next_pos_b = self.game.state_to_positions(next_state)
                next_actions_a = self.game.get_available_actions(next_pos_a)
                next_actions_b = self.game.get_available_actions(next_pos_b)
                
                if next_actions_a and next_actions_b:
                    next_payoff_a = self.nash_solver.compute_payoff_matrix(
                        next_state, next_actions_a, next_actions_b, self.agent_a.get_q_value, 0
                    )
                    next_policy_a, _, next_nash_q_a, _ = self.nash_solver.solve_zero_sum(next_payoff_a)
            
            # Update agents
            self.agent_a.update(state, action_a, action_b, reward_a, next_nash_q_a)
            self.agent_b.update(state, action_b, action_a, reward_b, -next_nash_q_a)
            
            # Decay exploration
            self.agent_a.decay_epsilon()
            self.agent_b.decay_epsilon()
            
            # Next state
            state = next_state
        
        return total_reward_a, total_reward_b
    
    def train(self, episodes=2000):
        """Main training loop"""
        print(f"Training for {episodes} episodes...")
        
        rewards_a = []
        rewards_b = []
        
        for episode in range(episodes):
            reward_a, reward_b = self.train_episode()
            rewards_a.append(reward_a)
            rewards_b.append(reward_b)
            
            if (episode + 1) % 500 == 0:
                avg_a = np.mean(rewards_a[-100:])
                avg_b = np.mean(rewards_b[-100:])
                print(f"Episode {episode + 1}: Avg A={avg_a:.2f}, B={avg_b:.2f}")
        
        return rewards_a, rewards_b
    
    def get_policy(self, state):
        """Get Nash policy for a state"""
        pos_a, pos_b = self.game.state_to_positions(state)
        actions_a = self.game.get_available_actions(pos_a)
        actions_b = self.game.get_available_actions(pos_b)
        
        payoff_a = self.nash_solver.compute_payoff_matrix(
            state, actions_a, actions_b, self.agent_a.get_q_value, 0
        )
        
        policy_a_vec, policy_b_vec, nash_q_a, nash_q_b = self.nash_solver.solve_zero_sum(payoff_a)
        
        policy_a = {action: prob for action, prob in zip(actions_a, policy_a_vec)}
        policy_b = {action: prob for action, prob in zip(actions_b, policy_b_vec)}
        
        return policy_a, policy_b, nash_q_a, nash_q_b