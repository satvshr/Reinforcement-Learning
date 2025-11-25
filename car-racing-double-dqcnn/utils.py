import gymnasium as gym
import cv2
import matplotlib.pyplot as plt


def render(env, video_writer):
    # Render the environment in 'rgb_array' mode to get frames as arrays
    frame = env.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame_bgr)


def plot_graph(itt, cumulative_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_rewards, label="Cumulative Reward")
    plt.xlabel("Frames")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Time")
    plt.legend()
    plt.savefig(f"plots/rewards_over_time_{itt}.png")


def visualize(actions, itt):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()

    frame = env.render()
    h, w, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(
        f"videos/cartpole_{itt}.avi", fourcc, 30, (w, h)
    )

    cumulative_rewards = []
    total_reward = 0

    for a in actions:
        render(env, video_writer)
        _, reward, terminated, truncated, _ = env.step(a)
        total_reward += reward
        cumulative_rewards.append(total_reward)

        if terminated or truncated:
            env.reset()
            total_reward = 0

    video_writer.release()
    env.close()
    plot_graph(itt, cumulative_rewards)
