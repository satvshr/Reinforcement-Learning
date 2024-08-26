import torch as T
# For layers
import torch.nn as nn
# As observation space is a simple size 8 vector, we won't need linear layers and would only need convolution layers
# For RelU
import torch.nn.functional as F
# Adam optimizer
import torch.optim as optim
import numpy as np

"""
We will have 2 classes, one for DQN and the other for the agent. This is done as the agent is not a dqn but the agent HAS a dqn.
It also has a memory to choose the best actions as well as learning from its experience.

Here we will only use a replay network as a target network is not required.
""" 

class DQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4, target_update_freq=10):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq 
        self.mem_ctr = 0

        # Use the same class for both networks
        self.Q_eval = DQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.Q_target = DQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        # Initialize with same weights
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_ctr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array(observation), dtype=T.float32).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_ctr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_size, self.mem_ctr)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        # Use the target network to get the best action indices
        a_best = T.argmax(q_next, dim=1)

        # Use the target network to get the Q-values for the best actions and compute target Q-values
        # (1-terminal_batch) to ensure no future rewards considered
        q_target = reward_batch + (T.ones(64) - terminal_batch.int()) * self.gamma * self.Q_target.forward(new_state_batch)[batch_index, a_best].detach()

        loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)

        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min