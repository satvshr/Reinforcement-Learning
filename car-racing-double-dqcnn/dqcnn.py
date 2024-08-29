import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms

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
        self.device = 'cuda:0'

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.reshape(-1)))
        actions = self.fc2(x)

        return actions

class Agent():
    def __init__(self, n_actions, action_space, img_dim, lr, n_frames=4, gamma=0.95, eps=1.0, mem_size=1000, eps_decay=0.05, eps_min=0.05):
        self.n_actions = n_actions
        self.action_space = action_space
        self.n_frames = n_frames
        self.img_dim = img_dim
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.mem_size = mem_size
        self.mem_ctr = 0
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.online = DQCNN(n_actions, lr)
        self.target = DQCNN(n_actions, lr)

        self.state_memory = np.zeros((self.mem_size, self.img_dim, self.img_dim))
        self.action_memory = np.zeros(self.mem_size)
        self.next_state_memory = np.zeros((self.mem_size, self.img_dim, self.img_dim))
        self.reward_memory = np.zeros(self.mem_size)
        self.termination_memory = np.zeros(self.mem_size)

    def preprocess(self, states):
        # (height, width, 3, frames) -> (height, width, 3*frames)
        states = T.tensor(states).reshape(self.img_dim, self.img_dim, -1)

        # (height, width, 3*frames) -> (3*frames, height, width)
        states = states.permute(2, 0, 1)

        # (3*frames, height, width) -> (frames, height, width) is the obj of the below code block
        # Stack each frame into groups of 3 (r,g,b)
        states = T.stack([states[i:i+3] for i in range(0, states.size()[0], 3)])
        
        # Now we have states in the shape (frames, 3, height, width), we grayscale each frame using torchvision
        grayscale_transform = transforms.Grayscale()
        grayscale_frames = []
        for i in range(states.size(0)):
            frame = states[i]  # (3, height, width)
            frame = transforms.ToPILImage()(frame)  # Convert to PIL Image
            frame = grayscale_transform(frame)  # Apply grayscale transformation
            frame = transforms.ToTensor()(frame)  # Convert back to tensor
            grayscale_frames.append(frame)
        
        # Stack grayscale frames
        states = T.stack(grayscale_frames)
        
        # Now we have grayscaled-states in shape (frames, 1, height, width) so we reshape it to (frames, height, width)
        states = states.squeeze(1)

        return states   
    
    def choose_action(self, state):
        if np.random.random() > self.eps:
            action = np.argmax(self.online.forward(state))
        else:
            action = self.action_space.sample()
        
        return action

    def store_transition(self, state, action, next_state, reward, done):
        mem_idx = self.mem_ctr % self.mem_size

        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.next_state_memory[mem_idx] = next_state
        self.reward_memory[mem_idx] = reward
        self.termination_memory[mem_idx] = done

        self.mem_ctr += 1 

    def learn(self):
        pass
