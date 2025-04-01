
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from core.agent.multimodal_policy_net import MultimodalPolicyNet

class MultimodalAgent:
    def __init__(self, button_count=14):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultimodalPolicyNet(button_count=button_count).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.replay_buffer = []
        self.max_buffer = 5000
        self.batch_size = 32
        self.epsilon = CONFIG.get('default_true_value', 1)
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def act(self, image, texts, volume):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 14)

        img_flat = torch.FloatTensor(image.flatten()).unsqueeze(0).to(self.device)
        audio_tensor = torch.FloatTensor([[volume]]).to(self.device)

        with torch.no_grad():
            logits = self.model(img_flat, texts, audio_tensor)
            return torch.argmax(logits).item()

    def remember(self, image, texts, volume, action, reward, next_image):
        self.replay_buffer.append((image, texts, volume, action, reward))
        if len(self.replay_buffer) > self.max_buffer:
            self.replay_buffer.pop(0)

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = np.random.choice(self.replay_buffer, self.batch_size, replace=False)
        losses = []

        for image, texts, volume, action, reward in batch:
            img_flat = torch.FloatTensor(image.flatten()).unsqueeze(0).to(self.device)
            audio_tensor = torch.FloatTensor([[volume]]).to(self.device)
            logits = self.model(img_flat, texts, audio_tensor)
            target = logits.clone().detach()
            target[0, action] = reward

            loss = self.criterion(logits, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path="multimodal_agent.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="multimodal_agent.pth"):
        self.model.load_state_dict(torch.load(path))

# === Ajout automatique ===

# Ajout dans la boucle de décision de l'agent
print(f"[INFO] Partie {self.session_id} - Tour {self.turn} - Décision : {decision}")
if self.mode == "observer_only":
    self.observe_and_record()
    self.save_screenshot()
    return None
