
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.model(x)

class DeepDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = CONFIG.get('default_true_value', 1).0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.gamma = 0.95
        self.lr = 0.001

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork(state_size, action_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()

    def replay(self, memory, batch_size):
        if len(memory) < batch_size:
            return
        minibatch = random.sample(memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            current_q = self.model(state_tensor)[action]
            loss = self.criterion(current_q, torch.tensor(target).to(self.device))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path="dqn_model.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="dqn_model.pth"):
        self.model.load_state_dict(torch.load(path))