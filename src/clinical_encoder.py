import torch
import torch.nn as nn

class ClinicalEncoder(nn.Module):
    """
    Feature extractor for patient demographic + CBC tabular metrics.
    Embeds categorical features, concatenates them with numeric features,
    and passes them through an MLP to map to d_model (default 256).
    """
    def __init__(self, num_numeric, cat_cardinalities, embed_dim=256, cat_emb_dim=8):
        super().__init__()
        # ModuleList of embeddings for each categorical feature
        # We use card + 1 to account for any unknown values encoded as -1 (mapped to card)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, cat_emb_dim) for card in cat_cardinalities
        ])
        
        in_dim = num_numeric + cat_emb_dim * len(cat_cardinalities)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x_num, x_cat):
        # x_num: (B, num_numeric) float tensor
        # x_cat: (B, num_cat) long tensor
        cat_embs = []
        for i, emb_layer in enumerate(self.cat_embeddings):
            # Clamp categorical indices to ensure they are within the embedding vocabulary
            indices = x_cat[:, i]
            # Map negative values (like -1 for unknown) to the last vocabulary index
            indices = torch.clamp(indices, min=0, max=emb_layer.num_embeddings - 1)
            cat_embs.append(emb_layer(indices))
            
        if cat_embs:
            x = torch.cat([x_num] + cat_embs, dim=-1)
        else:
            x = x_num
            
        return self.mlp(x)

class ClinicalClassifier(nn.Module):
    """
    Standalone Clinical Classifier for pretraining / sanity checks.
    """
    def __init__(self, num_numeric, cat_cardinalities, embed_dim=256, num_classes=2):
        super().__init__()
        self.encoder = ClinicalEncoder(num_numeric, cat_cardinalities, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x_num, x_cat):
        emb = self.encoder(x_num, x_cat)
        logits = self.head(emb)
        return logits, emb
