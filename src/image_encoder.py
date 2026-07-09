import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    """
    Feature extractor for conjunctiva images.
    Uses EfficientNet-B0 pretrained on ImageNet, removes the final classifier,
    and projects the 1280-d features to d_model (default 256).
    """
    def __init__(self, embed_dim=256, pretrained=True):
        super().__init__()
        # Load EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Replace classifier with nn.Identity to extract raw features (1280-d)
        self.backbone.classifier = nn.Identity()
        
        # Projection head to match fusion dimension d_model (256)
        self.proj = nn.Sequential(
            nn.Linear(1280, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        feats = self.backbone(x)      # (B, 1280)
        return self.proj(feats)       # (B, embed_dim)

class ImageClassifier(nn.Module):
    """
    Standalone Image Classifier for pretraining / sanity checks.
    """
    def __init__(self, embed_dim=256, num_classes=2, pretrained=True):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim, pretrained=pretrained)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.encoder(x)
        logits = self.head(emb)
        return logits, emb
