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
        # Call constructor to inherit base class properties
        super(DQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        # * is used here to unpack a list, useful say if in the future we want to use a 2D Matrix later on
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # Q-learning is basically linear reg where we will try to fit a line to the dif between target value and output of DQN
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # We do not use an activation function as we want the agents raw estimate
        # and we do not want to supress value of future rewards as they could be negative or be greater than 1.
        actions = self.fc3(x)

        return actions
        
class Agent():
    """
    gamma = weightage of future rewards
    epsilon = explore-exploit rate
    max_mem_size = maximum size of the replay memory
    eps_end = minimum value that epsilon can decay to
    eps_dec = at what rate we should decrement eps at each time step
    """
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_ctr = 0

        self.Q_eval = DQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, new_state, done):
        # By using % memory gets rewritten everytime it exceeds 100k
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

    """
    Memory right now has a bunch of zeros so we cant learn from that, so to learn we can either:
    1. Let agent play bunch of games till it fills up entire memory, random actions no intelligence.
    2. Start learning as soon as a batch size of memory is filled.
    """

    def learn(self):
        if self.mem_ctr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()
        # Since we only want to select upto last filled memory
        max_mem = min(self.mem_size, self.mem_ctr)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # To slice array properly
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        # Selects the Q-values corresponding to the actions that were actually taken in each state of the batch
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        # T.max gives maximum value of the next state, [0] is the max value, [1] is the index of the max value
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


