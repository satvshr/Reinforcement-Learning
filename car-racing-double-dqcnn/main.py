import gymnasium as gym
from dqcnn import Agent
from utils import visualize
import numpy as np

env = gym.make("CarRacing-v2", continuous=False)
agent = Agent(n_actions=env.action_space.n, action_space=env.action_space, img_dim=env.observation_space.shape[0], batch_size=16, lr=0.001, target_update_itt=10, visualize_itt=100)
n_episodes = 500

def collect(state, action):
    env1 = gym.make("CarRacing-v2", continuous=False)
    env1.reset()
    env1.unwrapped.s = state
    next_states, rewards = [], []

    for i in range(agent.n_frames):
        next_state, reward, _, _, _ = env1.step(action)
        next_states.append(next_state)
        rewards.append(reward)

    env1.close()
    return np.array(next_states), np.mean(rewards)

for i in range(n_episodes):
    score = 0
    done = False
    state, info = env.reset()
    actions = []
    
    # Collect the initial state
    states, _ = collect(state, None)  # Collect initial state without any action

    while not done:
        # Preprocess the collected states
        states = agent.preprocess(states)
        
        # Choose an action based on the current state
        action = agent.choose_action(states)
        
        # Collect the next states and accumulated reward after taking the action
        next_states, accumulated_reward = collect(state, action)
        
        # Take the action in the environment
        next_state, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            done = True

        # Store the transition in memory
        agent.store_transition(state, action, next_states, accumulated_reward, done)
        agent.learn()

        # Update the current state
        state = next_state
        states = next_states  # Update the states with the next states

    actions.append(action)

    # Update epsilon for exploration-exploitation trade-off
    agent.eps = agent.eps - agent.eps_decay if agent.eps > agent.eps_min else agent.eps_min

    # Visualize the actions every visualize_itt episodes
    if i % agent.visualize_itt == 0:
        visualize(actions, i)

    # Update the target network every target_update_itt episodes
    if i % agent.target_update_itt == 0:
        agent.Q_target.load_state_dict(agent.Q_online.state_dict())
