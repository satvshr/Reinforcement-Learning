import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DQCNN(nn.Module):
    def __init__(self, n_actions, lr, n_frames=4):
        super(DQCNN, self).__init__()
        self.in_channels = n_frames
        self.out_channels = n_actions
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=16, kernel_size=3, stride=2, padding=1) # (96, 96) -> (48, 48)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1) # (48, 48) -> (24, 24)
        self.fc1 = nn.Linear(in_features=32*24*24, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=self.out_channels)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
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
    def __init__(self, n_actions, action_space, img_dim, batch_size, lr, target_update_itt, visualize_itt, n_frames=4, gamma=0.95, eps=1.0, mem_size=1000, eps_decay=3e-5, eps_min=0.01):
        self.n_actions = n_actions
        self.action_space = action_space
        self.n_frames = n_frames
        self.img_dim = img_dim
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
        self.Q_online = DQCNN(n_actions, lr)
        self.Q_target = DQCNN(n_actions, lr)
        self.Q_target.load_state_dict(self.Q_online.state_dict())

        self.state_memory = np.zeros((self.mem_size, self.n_frames, self.img_dim, self.img_dim))
        self.action_memory = np.zeros(self.mem_size)
        self.next_state_memory = np.zeros((self.mem_size, self.n_frames, self.img_dim, self.img_dim))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size)

    def preprocess(self, states):
        # states: (n_frames, height, width, 3)
        # Convert to tensor and normalize pixel values
        states = T.tensor(states, dtype=T.float32) / 255.0  # Normalize to [0, 1]
        
        # Convert to grayscale by averaging the color channels
        states = states.mean(dim=-1, keepdim=True)  # Now shape is (n_frames, height, width, 1)
        
        # Permute dimensions to (n_frames, 1, height, width)
        states = states.permute(0, 3, 1, 2)
        
        # Stack frames along the channel dimension
        states = states.view(1, self.n_frames, self.img_dim, self.img_dim)  # Add batch dimension
        
        return states.to(self.Q_online.device)
    
    def choose_action(self, state):
        state = state.to(self.Q_online.device)
        if np.random.random() > self.eps:
            action = np.argmax(self.Q_online.forward(state).cpu().detach().numpy())
        else:
            action = self.action_space.sample()
        
        return action

    def store_transition(self, state, action, next_state, reward, done):
        mem_idx = self.mem_ctr % self.mem_size

        self.state_memory[mem_idx] = state.squeeze(0).cpu().numpy()
        self.action_memory[mem_idx] = action
        self.next_state_memory[mem_idx] = next_state.squeeze(0).cpu().numpy()
        self.reward_memory[mem_idx] = reward
        self.terminal_memory[mem_idx] = done

        self.mem_ctr += 1 

    def learn(self):
        if self.mem_ctr < self.batch_size:
            return
        
        self.Q_online.optimizer.zero_grad()
        max_mem = min(self.mem_size, self.mem_ctr)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # Prepare batches
        state_batch = T.tensor(self.state_memory[batch]).float().to(self.Q_online.device)
        action_batch = T.tensor(self.action_memory[batch]).long().to(self.Q_online.device)
        new_state_batch = T.tensor(self.next_state_memory[batch]).float().to(self.Q_online.device)
        reward_batch = T.tensor(self.reward_memory[batch]).float().to(self.Q_online.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).float().to(self.Q_online.device)

        # Compute current Q-values
        q_online_values = self.Q_online.forward(state_batch)
        q_online = q_online_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using Double DQN
        # Select best actions using the online network
        q_next_online = self.Q_online.forward(new_state_batch)
        a_best = T.argmax(q_next_online, dim=1)

        # Evaluate best actions using the target network
        q_next_target = self.Q_target.forward(new_state_batch)
        q_target_values = q_next_target.gather(1, a_best.unsqueeze(1)).squeeze(1).detach()

        # Compute the target Q-values
        non_terminal_mask = (1 - terminal_batch)
        q_target = reward_batch + non_terminal_mask * self.gamma * q_target_values

        # Compute loss
        loss = self.Q_online.loss(q_online, q_target).to(self.Q_online.device)

        # Update epsilon for exploration-exploitation trade-off
        self.eps = max(self.eps - self.eps_decay, self.eps_min)

        loss.backward()
        self.Q_online.optimizer.step()