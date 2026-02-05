#!/usr/bin/env python3
"""
Q-Learning Implementation for Grid World Environment

This is the main entry point for training a Q-learning agent
in a grid world environment.

Run this file to start training:
    python main.py
"""

from environment import GridWorld
from agent import QLearningAgent
from visualization import plot_results, compute_average_value
from config import EPISODES, REPORT_EPISODES


def train(episodes=EPISODES, report_episodes=REPORT_EPISODES, visualize=True):
    """
    Train a Q-learning agent in the grid world environment.
    
    Args:
        episodes: Number of training episodes
        report_episodes: List of episode numbers to print Q-values
        visualize: Whether to show visualization after training
    
    Returns:
        agent: The trained QLearningAgent
        convergence_values: List of average state values per episode
    """
    # Initialize environment and agent
    env = GridWorld()
    agent = QLearningAgent(env)
    
    # Track average state values for convergence plot
    convergence_values = []
    
    print("Starting Q-Learning Training...")
    print(f"Episodes: {episodes}")
    print(f"Grid size: {env.rows}x{env.cols}")
    print(f"Terminal states: {list(env.terminal_states.keys())}")
    print("-" * 40)
    
    # Training loop
    for episode in range(1, episodes + 1):
        # Run one episode
        steps = agent.train_episode()
        
        # Report Q-values at specified episodes
        if episode in report_episodes:
            agent.print_q_values(episode)
        
        # Track average value for convergence plot
        avg_value = compute_average_value(agent)
        convergence_values.append(avg_value)
    
    print("-" * 40)
    print("Training complete!")
    
    # Visualize results
    if visualize:
        plot_results(agent, convergence_values, report_episodes)
    
    return agent, convergence_values


def main():
    """Main entry point."""
    agent, convergence_values = train()
    return agent


if __name__ == '__main__':
    main()
