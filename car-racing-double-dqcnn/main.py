import gymnasium as gym
from dqcnn import Agent
from utils import visualize
import numpy as np

env = gym.make("CarRacing-v2", continuous=False)
agent = Agent(n_actions=env.action_space.n, action_space=env.action_space, img_dim=env.observation_space.shape[0], lr=0.001, target_update_itt=10, visualize_itt=100)
n_episodes = 500

def collect(state, action):
    env1 = gym.make("CarRacing-v2", continuous=False)
    env1.reset()
    env1.unwrapped.s = state
    next_states = []
    for i in range(agent.n_frames):
        next_state, reward, terminated, truncated, _ = env1.step(action)
        next_states.append(next_state)

    env1.close()
    return np.array(next_states)

for i in range(n_episodes):
    score = 0
    done = False
    state, info = env.reset()
    actions = []

    while not done:
        states = collect(state, action)
        # rgb(n_frames*3, height, width) -> grayscale(n_frames, height, width)
        states = agent.preprocess(states)
        action = agent.choose_action(states)
        next_state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            done = True

        agent.store_transition(state, action, next_state, reward, done)
        agent.learn()

        state = next_state
    
    actions.append(action)

    agent.eps = agent.eps - agent.eps_decay if agent.eps > agent.eps_min else agent.eps_min

    if i % agent.visualize_itt == 0:
        visualize(actions, i)

    if i % agent.target_update_itt == 0:
        agent.Q_target.load_state_dict(agent.Q_online.state_dict())
    
