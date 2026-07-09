import torch
import torch.nn as nn

class CustomTransformerLayer(nn.Module):
    """
    Standard Transformer Encoder Layer implemented manually to expose
    attention weights for explainability.
    """
    def __init__(self, d_model, n_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, return_attn=False):
        # Self Attention + Residual
        attn_out, attn_weights = self.self_attn(x, x, x, need_weights=return_attn)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feed Forward + Residual
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        
        if return_attn:
            return x, attn_weights
        return x

class ModalityFusionTransformer(nn.Module):
    """
    Fuses modality embeddings (image, clinical, geo-risk) using self-attention.
    Appends a learnable CLS token, adds modality-type embeddings,
    and runs a sequence of custom transformer layers.
    """
    def __init__(self, d_model=256, n_heads=8, n_layers=4, dim_ff=512,
                 num_classes=2, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 0: CLS, 1: Image, 2: Clinical, 3: Geo
        self.modality_type_emb = nn.Embedding(4, d_model)
        
        self.layers = nn.ModuleList([
            CustomTransformerLayer(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, img_emb, clin_emb, geo_emb, return_attn=False):
        # img_emb, clin_emb, geo_emb shapes: (B, d_model)
        B = img_emb.size(0)
        
        # Expand CLS token to match batch size
        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, d_model)
        tokens = torch.stack([img_emb, clin_emb, geo_emb], dim=1) # (B, 3, d_model)
        seq = torch.cat([cls, tokens], dim=1)                     # (B, 4, d_model)
        
        # Add modality-type embeddings
        type_ids = torch.tensor([0, 1, 2, 3], device=img_emb.device)
        seq = seq + self.modality_type_emb(type_ids).unsqueeze(0)  # Broadcasting
        
        # Pass through transformer layers
        last_attn = None
        for i, layer in enumerate(self.layers):
            if return_attn and i == len(self.layers) - 1:
                seq, last_attn = layer(seq, return_attn=True)
            else:
                seq = layer(seq)
                
        cls_out = self.norm(seq[:, 0, :])     # (B, d_model)
        logits = self.classifier(cls_out)      # (B, num_classes)
        
        if return_attn:
            return logits, last_attn           # last_attn: (B, 4, 4)
        return logits

class AnemiaFusionNet(nn.Module):
    """
    End-to-end wrapper combining image, clinical, and geo-risk encoders
    with the modality fusion transformer.
    """
    def __init__(self, image_encoder, clinical_encoder, geo_encoder, d_model=256):
        super().__init__()
        self.image_encoder = image_encoder
        self.clinical_encoder = clinical_encoder
        self.geo_encoder = geo_encoder
        self.fusion = ModalityFusionTransformer(d_model=d_model)

    def forward(self, image, x_num, x_cat, state_idx, geo_risk_score, return_attn=False):
        img_emb = self.image_encoder(image)
        clin_emb = self.clinical_encoder(x_num, x_cat)
        geo_emb = self.geo_encoder(state_idx, geo_risk_score)
        
        return self.fusion(img_emb, clin_emb, geo_emb, return_attn=return_attn)
