import numpy as np
from nash_q_learner import NashQLearner

class NashQTrainer:
    """Coordinates Nash Q-learning for SEQUENTIAL game"""
    
    def __init__(self, game):
        self.game = game
        self.agent_a = None
        self.agent_b = None
        
    def initialize_agents(self, alpha=0.1, gamma=0.9, epsilon=0.3):
        """Create both agents"""
        self.agent_a = NashQLearner(0, alpha, gamma, epsilon)
        self.agent_b = NashQLearner(1, alpha, gamma, epsilon)
    
    def train_episode(self, max_steps=100):
        """Train for one episode with alternating turns"""
        state = self.game.reset()  # This also resets current_turn to 0 (A's turn)
        total_reward_a = 0
        total_reward_b = 0
        
        for step in range(max_steps):
            # Get current player
            current_player = self.game.get_current_player()
            
            if current_player == 0:
                agent = self.agent_a
                opponent = self.agent_b
            else:
                agent = self.agent_b
                opponent = self.agent_a
            
            # Get available actions
            available_actions = self.game.get_available_actions(state)
            
            if not available_actions:
                break
            
            # Choose action
            action = agent.choose_action(available_actions, state, opponent)
            
            # Execute action
            reward_a, reward_b, next_state = self.game.execute_action(state, action)
            
            total_reward_a += reward_a
            total_reward_b += reward_b
            
            # Get opponent's possible actions in next state
            next_player = self.game.get_current_player()  # Turn already switched in execute_action
            next_available_actions = self.game.get_available_actions(next_state, next_player)
            
            # Update Q-value for the player who just moved
            if current_player == 0:
                agent.update(state, action, reward_a, next_state, next_available_actions, opponent)
            else:
                agent.update(state, action, reward_b, next_state, next_available_actions, opponent)
            
            # Move to next state
            state = next_state
            
            # Decay exploration
            agent.decay_epsilon()
        
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
    
    def get_policy(self, state, player=0):
        """Get policy for a state and player"""
        available_actions = self.game.get_available_actions(state, player)
        
        if not available_actions:
            return {}, 0
        
        if player == 0:
            agent = self.agent_a
        else:
            agent = self.agent_b
        
        # Create policy based on Q-values
        q_values = [agent.get_q_value(state, a) for a in available_actions]
        
        # Softmax policy
        q_values = np.array(q_values)
        exp_q = np.exp(q_values - np.max(q_values))  # Numerical stability
        probs = exp_q / np.sum(exp_q)
        
        policy = {action: prob for action, prob in zip(available_actions, probs)}
        expected_value = np.sum(q_values * probs)
        
        return policy, expected_value