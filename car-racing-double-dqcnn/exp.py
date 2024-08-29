# import gymnasium as gym
# import numpy as np
# import cv2  # OpenCV for video writing
# import matplotlib.pyplot as plt

# # Initialize the environment with 'rgb_array' rendering mode
# env = gym.make("CarRacing-v2", continuous=False, render_mode="rgb_array")

# # Reset the environment to get the initial frame
# state, info = env.reset()

# # Check the frame size for the VideoWriter initialization
# frame = env.render()
# frame_height, frame_width, _ = frame.shape

# # Define the codec and create a VideoWriter object to save the video in AVI format
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
# video_writer = cv2.VideoWriter('car_racing.avi', fourcc, 30, (frame_width, frame_height))  # Correct frame size

# # Define a function to render the environment and display it using Matplotlib
# def render():
#     # Render the environment in 'rgb_array' mode to get frames as arrays
#     state_image = env.render()  
    
#     # Convert to BGR format for OpenCV compatibility
#     state_image_bgr = cv2.cvtColor(state_image, cv2.COLOR_RGB2BGR)
    
#     # Save the frame to the video file
#     video_writer.write(state_image_bgr)

# # Action initialization
# action = 3
# cumulative_rewards = []  # To store cumulative rewards over time
# total_reward = 0  # Initialize total reward for plotting

# # Capture and save 500 frames
# for frame_idx in range(500):
#     render()  # Render and save each frame
    
#     # Step through the environment
#     next_state, reward, terminated, truncated, info = env.step(action)
    
#     # Accumulate reward
#     total_reward += reward
#     cumulative_rewards.append(total_reward)  # Store the cumulative reward
    
#     # Check if the episode has terminated or truncated
#     if terminated or truncated:
#         state, info = env.reset()
#         action = env.action_space.sample()
#         total_reward = 0  # Reset total reward after each episode
#     else:
#         state = next_state
#         action = 3  # Sample a new action

# # Release the VideoWriter and close the environment
# video_writer.release()
# env.close()

# # Plot the cumulative rewards over time
# plt.figure(figsize=(10, 5))
# plt.plot(cumulative_rewards, label='Cumulative Reward')
# plt.xlabel('Frames')
# plt.ylabel('Cumulative Reward')
# plt.title('Cumulative Reward Over Time in CarRacing-v2')
# plt.legend()
# plt.show()

import gymnasium as gym
from dqcnn import Agent
import numpy as np
import torch
env = gym.make("CarRacing-v2", continuous=False)
env.reset()
agent = Agent(n_actions=env.action_space.n, action_space=env.action_space, img_dim=env.observation_space.shape[0], lr=0.001, target_update_itt=10, visualize_itt=100)

def collect(state, action):
    env.unwrapped.s = state
    next_states = []
    for i in range(agent.n_frames):
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_states.append(next_state)
    return np.array(next_states)

s = collect(env.action_space.sample(), env.action_space.sample())
print(np.shape(s))
x = agent.preprocess(s)