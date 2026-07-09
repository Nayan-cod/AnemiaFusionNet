import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path

class ConjunctivaDataset(Dataset):
    """
    Dataset class for training/evaluating the standalone Image Modality Branch.
    Loads eye images and their corresponding binary labels.
    """
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_root / str(row["image_path"])
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to a black image if file not found to prevent breaking the training loop
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
            
        label = int(row["label"])
        return img, label

class ClinicalDataset(Dataset):
    """
    Dataset class for training/evaluating the standalone Clinical Modality Branch.
    Takes numeric and categorical features as Tensors.
    """
    def __init__(self, x_num, x_cat, labels):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.labels[idx]

class AnemiaFusionDataset(Dataset):
    """
    Multimodal Dataset class for training/evaluating the fused model.
    Combines images, clinical tabular inputs, state index, and geographic risk score.
    """
    def __init__(self, csv_path, img_root, x_num, x_cat, state_indices, geo_risk_scores, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.state_indices = torch.tensor(state_indices, dtype=torch.long)
        self.geo_risk_scores = torch.tensor(geo_risk_scores, dtype=torch.float32)
        self.labels = torch.tensor(self.df["label"].values, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_root / str(row["image_path"])
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
            
        return (
            img,
            self.x_num[idx],
            self.x_cat[idx],
            self.state_indices[idx],
            self.geo_risk_scores[idx].unsqueeze(-1), # Shape (1,)
            self.labels[idx]
        )
