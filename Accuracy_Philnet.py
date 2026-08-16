import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers import DistilBertModel
import json
import os
import numpy as np
import gc

# 1. CONFIG
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR     = "./out/accuracy_results/"
NUM_CLASSES = 10
EPOCHS      = 15
os.makedirs(OUT_DIR, exist_ok=True)

# 2. ARCHITECTURES

class PhilNetImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Block 1 (32x32)
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 16x16

            # Block 2 (16x16)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 8x8

            # Block 3 (8x8)
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, padding=1),  
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, x):
        return torch.flatten(self.model(x), 1)


class PhilNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = PhilNetImageEncoder()
        self.classifier = nn.Sequential(
            nn.Identity(),                      # classifier.0
            nn.Linear(512, 256),                # classifier.1
            nn.ReLU(),                          # classifier.2
            nn.Dropout(0.5),                    # classifier.3
            nn.Linear(256, num_classes)         # classifier.4
        )
    def forward(self, x):
        return self.classifier(self.features(x))


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim=512, projection_dim=256, dropout=0.1):
        super().__init__()
        intermediate_dim = embedding_dim * 2 
        self.fc1 = nn.Linear(embedding_dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, projection_dim)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(projection_dim)
        self.shortcut = nn.Linear(embedding_dim, projection_dim)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(self.fc2(x))
        return self.ln(x + residual)


class CLIPNet(nn.Module):
    def __init__(self, projection_dim=256):
        super().__init__()
        self.image_encoder = PhilNetImageEncoder()
        self.image_projection = ProjectionHead(embedding_dim=512, projection_dim=projection_dim)
    def forward(self, x):
        return self.image_projection(self.image_encoder(x))


class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = PhilNetImageEncoder()
        self.mu = nn.Linear(512, latent_dim)
    def forward(self, x):
        return self.mu(self.encoder(x))


# 3. MODELL LADEN

def load_philnet_model(model_path, paradigm):
    if not os.path.exists(model_path):
        print(f"  [SKIP] Not found: {model_path}")
        return None

    if paradigm == "clip":
        model           = CLIPNet(projection_dim=256).to(DEVICE)
        target_backbone = model.image_encoder.model   
    elif paradigm == "vae":
        model           = VAE(latent_dim=256).to(DEVICE)
        target_backbone = model.encoder.model          
    else:
        model           = PhilNet(num_classes=10).to(DEVICE)
        target_backbone = model.features.model         

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    sd = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint)) \
        if isinstance(checkpoint, dict) else checkpoint

    # Universal Key-Mapping
    new_sd      = {}
    target_keys = list(target_backbone.state_dict().keys())
    for k, v in sd.items():
        clean = (k
                 .replace("image_encoder.model.", "")
                 .replace("encoder.model.features.", "")   
                 .replace("encoder.model.", "")
                 .replace("features.model.", "")
                 .replace("features.", ""))
        if clean in target_keys and v.shape == target_backbone.state_dict()[clean].shape:
            new_sd[clean] = v

    target_backbone.load_state_dict(new_sd, strict=False)
    print(f"  [{paradigm.upper()}] Loaded {len(new_sd)}/{len(target_keys)} backbone layers via Mapping")

    return model.eval()


# 4. EVALUATION HELPERS

class LinearProbeHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.classifier(x)


def train_linear_probe(backbone, train_loader, test_loader):
    """train a linear classifier on forzen 512-dim Features."""
    backbone.eval()
    # adaptive dimensions
    with torch.no_grad():
        test_imgs, _ = next(iter(train_loader))
        test_imgs = test_imgs.to(DEVICE)
        test_features = backbone(test_imgs)
        
        # The last dimension of the output tensor is our feature dimension
        input_dim = test_features.shape[-1]

    head      = LinearProbeHead(input_dim, NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        head.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.no_grad():
                features = backbone(imgs)
            optimizer.zero_grad()
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()

    head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            features = backbone(imgs)
            outputs  = head(features)
            _, predicted = torch.max(outputs, 1)
            total    += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100


# 5. MAIN
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    mean, std = [0.489, 0.4321, 0.4976], [0.22734, 0.24983, 0.2357]
    transform = T.Compose([
        T.Resize((224, 224), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
    test_dataset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=4)
    test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4)

    paradigms     = ["supervised", "vae", "clip"]
    iterations    = [1, 2, 3]
    summary_stats = {}

    for p in paradigms:
        run_accuracies = []
        print(f"\n{'='*40}\nStarting PhilNet Analysis: {p.upper()}\n{'='*40}")

        for i in iterations:
            model_path = f"./out/Philnet/philnet_{p}_run{i}.pt"

            # load weights
            model = load_philnet_model(model_path, p)
            if model is None:
                continue

            # get backbone for linear probe
            if p == "clip":
                backbone = model.image_encoder  
            elif p == "vae":
                backbone = model.encoder  
            else:
                backbone = model.features

            # Evaluation over linea probe
            print(f">>> Training Linear Probe on frozen {p.upper()} representations (512-dim)...")
            acc = train_linear_probe(backbone, train_loader, test_loader)

            run_accuracies.append(acc)
            print(f"  Run {i}: {acc:.2f}%")
            
            # empty ache
            del model, backbone
            torch.cuda.empty_cache()
            gc.collect()

        if run_accuracies:
            mean_acc = round(float(np.mean(run_accuracies)), 2)
            std_acc  = round(float(np.std(run_accuracies)), 2)
            summary_stats[p] = {
                "mean":     mean_acc,
                "std":      std_acc,
                "all_runs": [round(a, 2) for a in run_accuracies]
            }
            print(f"\n  ► {p.upper()} | Mean Acc: {mean_acc}% | Std: {std_acc}%")

    with open(os.path.join(OUT_DIR, "philnet_final_metrics.json"), "w") as f:
        json.dump(summary_stats, f, indent=4)
    print(f"\nResults saved to {OUT_DIR}")