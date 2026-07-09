import torch
import torch.nn as nn

class GeoRiskEncoder(nn.Module):
    """
    Feature extractor for the Geographic modality.
    Combines a learnable state-identity embedding with the continuous 
    NFHS-5 numeric risk score, mapping them to d_model (default 256).
    """
    def __init__(self, num_states, state_emb_dim=16, d_model=256):
        super().__init__()
        self.state_embedding = nn.Embedding(num_states, state_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(state_emb_dim + 1, 64),   # +1 for continuous risk score
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, state_idx, risk_score):
        # state_idx: (B,) long tensor
        # risk_score: (B, 1) float tensor
        e = self.state_embedding(state_idx)          # (B, state_emb_dim)
        x = torch.cat([e, risk_score], dim=-1)       # (B, state_emb_dim + 1)
        return self.mlp(x)                           # (B, d_model)
