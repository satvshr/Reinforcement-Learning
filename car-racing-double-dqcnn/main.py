import gymnasium as gym
from dqcnn import Agent
from utils import visualize
import numpy as np
import time
from collections import deque

# Create environment with pixel rendering enabled
env = gym.make("CartPole-v1", render_mode="rgb_array")

agent = Agent(
    n_actions=env.action_space.n,
    action_space=env.action_space,
    img_dim=84,
    batch_size=32,
    lr=1e-3,
    target_update_itt=10,
    visualize_itt=100,
)

scores = []
n_episodes = 1000

for i in range(n_episodes):
    score = 0
    done = False

    state, _ = env.reset()
    state = env.render()
    state = state.astype(np.uint8)

    frame_stack = deque([state] * agent.n_frames, maxlen=agent.n_frames)
    actions = []

    start_time = time.time()

    while not done:
        stacked_states = np.array(frame_stack)
        states = agent.preprocess(stacked_states)

        # Choose an action based on the current states
        action = agent.choose_action(states)

        # Step environment
        _, reward, terminated, truncated, _ = env.step(action)
        next_state = env.render().astype(np.uint8)

        frame_stack.append(next_state)
        next_states = agent.preprocess(np.array(frame_stack))

        elapsed_time = time.time() - start_time
        if elapsed_time > 60:
            done = True
            print("time")

        if terminated or truncated:
            done = True
            print("termination")

        score += reward

        agent.store_transition(states, action, next_states, reward, done)
        agent.learn()

        actions.append(action)

    scores.append(score)
    avg_score = np.mean(scores[-100:])

    print(
        "episode ",
        i,
        "score %.2f" % score,
        "average score %.2f" % avg_score,
        "epsilon %.2f" % agent.eps,
    )

    if i % agent.visualize_itt == 0:
        print("visualization triggered")
        visualize(actions, i)

    if i % agent.target_update_itt == 0:
        agent.Q_target.load_state_dict(agent.Q_online.state_dict())
