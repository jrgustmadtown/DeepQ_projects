#!/usr/bin/env python3

from environment import CarGameEnvironment
from agent import MinimaxQAgent, IndependentQLearningAgent
from visualization import plot_training_results, compute_average_value, plot_policy, save_all_state_policies
from config import EPISODES, MAX_STEPS_PER_EPISODE


def train(episodes=EPISODES, use_minimax=True, visualize=True):
    env = CarGameEnvironment()
    if use_minimax:
        agent_a = MinimaxQAgent(env, player='A')
        agent_b = MinimaxQAgent(env, player='B')
    else:
        agent_a = IndependentQLearningAgent(env, player='A')
        agent_b = IndependentQLearningAgent(env, player='B')

    convergence_a = []
    convergence_b = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            action_a = agent_a.choose_action(state)
            action_b = agent_b.choose_action(state)
            joint_action = (action_a, action_b)
            next_state, (reward_a, reward_b), done = env.step(state, joint_action)
            if use_minimax:
                agent_a.update(state, action_a, action_b, reward_a, next_state, done)
                agent_b.update(state, action_a, action_b, reward_b, next_state, done)
            else:
                agent_a.update(state, action_a, reward_a, next_state, done)
                agent_b.update(state, action_b, reward_b, next_state, done)
            state = next_state
            steps += 1

        avg_a = compute_average_value(agent_a)
        avg_b = compute_average_value(agent_b)
        convergence_a.append(avg_a)
        convergence_b.append(avg_b)

        if episode % 10 == 0:
            print(f"{episode}")

    if visualize:
        plot_training_results(convergence_a, convergence_b, env)

    return agent_a, agent_b, convergence_a, convergence_b


def play_game(agent_a, agent_b, env=None):
    if env is None:
        env = CarGameEnvironment()
    old_eps_a = agent_a.epsilon
    old_eps_b = agent_b.epsilon
    agent_a.epsilon = 0
    agent_b.epsilon = 0
    state = env.reset()
    done = False
    steps = 0
    total_reward_a = 0

    while not done and steps < MAX_STEPS_PER_EPISODE:
        action_a = agent_a.choose_action(state)
        action_b = agent_b.choose_action(state)
        next_state, (reward_a, reward_b), done = env.step(state, (action_a, action_b))
        total_reward_a += reward_a
        state = next_state
        steps += 1

    agent_a.epsilon = old_eps_a
    agent_b.epsilon = old_eps_b
    return total_reward_a


def main():
    agent_a, agent_b, _, _ = train(use_minimax=True)
    env = CarGameEnvironment()
    save_all_state_policies(agent_a, agent_b, env)
    play_game(agent_a, agent_b, env)
    return agent_a, agent_b


if __name__ == '__main__':
    main()
