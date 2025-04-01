
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class MultimodalPolicyNet(nn.Module):
    def __init__(self, image_dim=150528, audio_dim=1, button_count=14):
        super().__init__()
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Image encoder (simple linear for now, replace with CNN if needed)
        self.image_fc = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # Audio encoder
        self.audio_fc = nn.Sequential(
            nn.Linear(audio_dim, 32),
            nn.ReLU(),
        )

        # Fusion + policy head
        self.policy_head = nn.Sequential(
            nn.Linear(128 + 384 + 32, 128),  # 384 from sentence BERT
            nn.ReLU(),
            nn.Linear(128, button_count)
        )

    def forward(self, image_flat, text_list, audio_tensor):
        with torch.no_grad():
            text_vec = self.text_encoder.encode([" ".join(text_list)], convert_to_tensor=True)

        image_feat = self.image_fc(image_flat)
        audio_feat = self.audio_fc(audio_tensor)
        fusion = torch.cat([image_feat, text_vec, audio_feat], dim=1)
        logits = self.policy_head(fusion)
        return logits