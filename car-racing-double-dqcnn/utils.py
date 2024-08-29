import gymnasium as gym
import cv2
import matplotlib.pyplot as plt

def render(env, video_writer):
    # Render the environment in 'rgb_array' mode to get frames as arrays
    state_image = env.render()  
    
    # Convert to BGR format for OpenCV compatibility
    state_image_bgr = cv2.cvtColor(state_image, cv2.COLOR_RGB2BGR)
    
    # Save the frame to the video file
    video_writer.write(state_image_bgr)

def plot_graph(itt, cumulative_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_rewards, label='Cumulative Reward')
    plt.xlabel('Frames')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward Over Time in CarRacing-v2')
    plt.legend()
    plt.savefig(f'plots/rewards_over_time_{itt}.png')

def visualize(actions, itt):
    env = gym.make("CarRacing-v2", continuous=False, render_mode="rgb_array")
    env.reset()
    frame = env.render()
    frame_height, frame_width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    video_writer = cv2.VideoWriter(f'videos/car_racing_{itt}.avi', fourcc, 30, (frame_width, frame_height))  # Correct frame size
    cumulative_rewards = []
    total_reward = 0

    for frame in range(len(actions)):
        render(env, video_writer)
        _, reward, terminated, truncated, _ = env.step(actions[frame])
        total_reward += reward
        cumulative_rewards.append(total_reward)

        if terminated or truncated:
            env.reset()
            total_reward = 0

    video_writer.release()
    env.close()
    plot_graph(itt, cumulative_rewards)