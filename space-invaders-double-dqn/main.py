import gymnasium as gym
from doubledqn import Agent
from utils import plotLearning
import numpy as np

env = gym.make('LunarLander-v2')
agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=[8], n_actions=4, batch_size=64, eps_end=0.01)
scores, eps_history = [], []
n_games = 500

for i in range(n_games):
    score = 0
    done = False
    observation, info = env.reset()
    while not done:
        action = agent.choose_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True
        score += reward
        agent.store_transition(observation, action, reward, next_observation, done)
        agent.learn()
        observation = next_observation

    if i % agent.target_update_freq == 0:
        agent.Q_target.load_state_dict(agent.Q_eval.state_dict())
        
    scores.append(score)
    eps_history.append(agent.epsilon)

    # Avg score of last 100 games
    avg_score = np.mean(scores[-100:])

    print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)
x = [i+1 for i in range(n_games)]
plotLearning(x, scores, eps_history, 'LunarLander.png')