import gymnasium as gym
from dqcnn import Agent
from utils import visualize
import numpy as np
import time
from collections import deque

env = gym.make("CarRacing-v2", continuous=False)
agent = Agent(n_actions=env.action_space.n, action_space=env.action_space, img_dim=env.observation_space.shape[0], batch_size=16, lr=0.002, target_update_itt=5, visualize_itt=10)
scores = []
n_episodes = 500

for i in range(n_episodes):
    score = 0
    done = False
    state, _ = env.reset()
    frame_stack = deque([state] * agent.n_frames, maxlen=agent.n_frames)
    actions = []
    # Initialize the timer
    start_time = time.time()
    
    while not done:
        stacked_states = np.array(frame_stack)
        # (n_frames, height, width, 3) -> preprocess to (1, in_channels, height, width)
        states = agent.preprocess(stacked_states)
        
        # Choose an action based on the current states
        action = agent.choose_action(states)
        
        # Take the action in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        frame_stack.append(next_state)
        
        # Prepare the next stacked state
        next_stacked_states = np.array(frame_stack)
        next_states = agent.preprocess(next_stacked_states)
        
        # Check if the time limit has been exceeded
        elapsed_time = time.time() - start_time
        if elapsed_time > 60:
            done = True
            print("time")

        if terminated or truncated:
            done = True
            print("termination")

        score += reward

        # Store the transition in memory
        agent.store_transition(states, action, next_states, reward, done)
        agent.learn()

        # Update the current state
        state = next_state
        actions.append(action)
        
    scores.append(score)
    # Avg score of last 100 games
    avg_score = np.mean(scores[-100:])
    
    print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.eps)

    # Visualize the actions every visualize_itt episodes
    if i % agent.visualize_itt == 0:
        print("visualization triggered")
        visualize(actions, i)

    # Update the target network every target_update_itt episodes
    if i % agent.target_update_itt == 0:
        agent.Q_target.load_state_dict(agent.Q_online.state_dict())