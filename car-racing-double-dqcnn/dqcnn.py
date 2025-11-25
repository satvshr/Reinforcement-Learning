import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2


class DQCNN(nn.Module):
    def __init__(self, n_actions, lr, n_frames=4):
        super(DQCNN, self).__init__()
        self.in_channels = n_frames
        self.out_channels = n_actions

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels, out_channels=16,
            kernel_size=3, stride=2, padding=1
        )  # (84, 84) -> (42, 42)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32,
            kernel_size=3, stride=2, padding=1
        )  # (42, 42) -> (21, 21)

        self.fc1 = nn.Linear(32 * 21 * 21, 512)
        self.fc2 = nn.Linear(512, self.out_channels)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda")
        self.to(self.device)

    def forward(self, state):
        # state shape: (batch_size, in_channels, height, width)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        actions = self.fc2(x)
        return actions


class Agent():
    def __init__(
        self, n_actions, action_space, img_dim, batch_size, lr,
        target_update_itt, visualize_itt, n_frames=4, gamma=0.99,
        eps=1.0, mem_size=100_000, eps_decay=2e-3, eps_min=0.01
    ):

        self.n_actions = n_actions
        self.action_space = action_space
        self.n_frames = n_frames
        self.img_dim = 84
        self.batch_size = batch_size
        self.lr = lr
        self.target_update_itt = target_update_itt
        self.visualize_itt = visualize_itt
        self.gamma = gamma
        self.eps = eps
        self.mem_size = mem_size
        self.mem_ctr = 0
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        self.learn_start = 5000  # <-- DO NOT LEARN EARLY

        self.Q_online = DQCNN(n_actions, lr, n_frames=n_frames)
        self.Q_target = DQCNN(n_actions, lr, n_frames=n_frames)
        self.Q_target.load_state_dict(self.Q_online.state_dict())

        # Replay memory
        self.state_memory = np.zeros(
            (self.mem_size, self.n_frames, self.img_dim, self.img_dim),
            dtype=np.float32
        )
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.next_state_memory = np.zeros(
            (self.mem_size, self.n_frames, self.img_dim, self.img_dim),
            dtype=np.float32
        )
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def preprocess(self, states):
        # states: (n_frames, height, width, 3)
        frames = []
        for f in states:
            img = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
            frames.append(img)

        frames = np.array(frames, dtype=np.float32) / 255.0
        frames = T.tensor(frames).unsqueeze(0)  # (1, 4, 84, 84)
        return frames.to(self.Q_online.device)

    def choose_action(self, state):
        if np.random.random() > self.eps:
            with T.no_grad():
                qvals = self.Q_online(state).cpu().numpy()
                action = np.argmax(qvals)
        else:
            action = self.action_space.sample()
        return action

    def store_transition(self, state, action, next_state, reward, done):
        idx = self.mem_ctr % self.mem_size
        self.state_memory[idx] = state.squeeze(0).cpu().numpy()
        self.action_memory[idx] = action
        self.next_state_memory[idx] = next_state.squeeze(0).cpu().numpy()
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done
        self.mem_ctr += 1

    def learn(self):
        # DO NOT LEARN BEFORE A MINIMUM REPLAY SIZE
        if self.mem_ctr < max(self.batch_size, self.learn_start):
            return

        self.Q_online.optimizer.zero_grad()

        max_mem = min(self.mem_size, self.mem_ctr)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = T.tensor(self.state_memory[batch]).float().to(self.Q_online.device)
        action_batch = T.tensor(self.action_memory[batch]).long().to(self.Q_online.device)
        new_state_batch = T.tensor(self.next_state_memory[batch]).float().to(self.Q_online.device)
        reward_batch = T.tensor(self.reward_memory[batch]).float().to(self.Q_online.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).float().to(self.Q_online.device)

        # Q(s,a)
        q_online_values = self.Q_online(state_batch)
        q_online = q_online_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Double DQN target
        q_next_online = self.Q_online(new_state_batch)
        best_actions = T.argmax(q_next_online, dim=1)

        q_next_target = self.Q_target(new_state_batch)
        q_target_selected = q_next_target.gather(1, best_actions.unsqueeze(1)).squeeze(1).detach()

        q_target = reward_batch + (1 - terminal_batch) * self.gamma * q_target_selected

        loss = self.Q_online.loss(q_online, q_target)

        self.eps = max(self.eps - self.eps_decay, self.eps_min)

        loss.backward()
        nn.utils.clip_grad_norm_(self.Q_online.parameters(), 10.0)
        self.Q_online.optimizer.step()
