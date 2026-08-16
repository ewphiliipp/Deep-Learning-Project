import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
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
os.makedirs(OUT_DIR, exist_ok=True)
NUM_CLASSES = 10
EPOCHS      = 15

# 2. ARCHITECTURES

class ViTImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the base ViT-B/16 model
        self.model = models.vit_b_16(weights=None)
        self.embedding_dim = 768
        # Remove the classification head to get raw 768-dim features (CLS token)
        self.model.heads = nn.Identity()

    def forward(self, x):
        return self.model(x)

class ViTSupervised(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = ViTImageEncoder()
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        f = self.features(x)
        return self.classifier(f)

class CLIPNet(nn.Module):
    def __init__(self, text_model_name="distilbert-base-uncased", projection_dim=256):
        super().__init__()
        self.image_encoder = ViTImageEncoder()
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        self.image_projection = nn.Linear(768, projection_dim)
        self.text_projection = nn.Linear(768, projection_dim)

class GenericEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.image_encoder = ViTImageEncoder()
        self.mu = nn.Linear(768, latent_dim)
        self.logvar = nn.Linear(768, latent_dim)
    def forward(self, x):
        f = self.image_encoder(x)
        return self.mu(f), self.logvar(f)

class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = GenericEncoder(latent_dim)

        
# 3. LOAD MODEL WEIGHTS POSITION-BASED (Exactly matches the ResNet logic)

def load_vit_model(model_path, paradigm):
    if not os.path.exists(model_path):
        print(f"  [SKIP] Not found: {model_path}")
        return None

    # Initialize model architecture
    if paradigm == "clip":
        model = CLIPNet("distilbert-base-uncased", 256).to(DEVICE)
        target_backbone = model.image_encoder.model
    elif paradigm == "vae":
        model = VAE(latent_dim=256).to(DEVICE)
        target_backbone = model.encoder.image_encoder.model
    else:
        model = ViTSupervised(num_classes=10).to(DEVICE)
        target_backbone = model.features.model

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint)) \
        if isinstance(checkpoint, dict) else checkpoint

    # Position-based mapping
    current_dict = target_backbone.state_dict()
    new_state_dict = {}
    
    sd_keys = list(state_dict.keys())
    target_keys = list(current_dict.keys())
    
    print(f"  [{paradigm.upper()}] Try positionbased mapping...")
    
    matched_count = 0
    sd_idx = 0
    
    for t_key in target_keys:
        target_shape = current_dict[t_key].shape
        
        while sd_idx < len(sd_keys):
            source_key = sd_keys[sd_idx]
            source_shape = state_dict[source_key].shape
            
            if source_shape == target_shape:
                new_state_dict[t_key] = state_dict[source_key]
                matched_count += 1
                sd_idx += 1 
                break
            else:
                sd_idx += 1
    
    target_backbone.load_state_dict(new_state_dict, strict=False)
    print(f"  [{paradigm.upper()}] Loaded {matched_count}/{len(target_keys)} layers via Position-Match.")
    return model.eval()

# 4. LINEAR PROBE HELPERS (Mimics the ResNet script's global scope behavior)

class LinearProbeHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.classifier(x)

def train_linear_probe(backbone, train_loader, test_loader):
    """train a linear classifier on frozen 768-dim Features."""
    backbone.eval()
    
    # adaptive dimensions
    with torch.no_grad():
        test_imgs, _ = next(iter(train_loader))
        test_imgs = test_imgs.to(DEVICE)
        test_features = backbone(test_imgs)
        
        # The last dimension of the output tensor is our feature dimension
        input_dim = test_features.shape[-1]
        
    head = LinearProbeHead(input_dim, NUM_CLASSES).to(DEVICE)
    
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

    # Evaluation of the trained linear probe
    head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            features = backbone(imgs)
            outputs = head(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100

# 5. MAIN EXECUTION

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    mean, std = [0.489, 0.4321, 0.4976], [0.22734, 0.24983, 0.2357]
    transform = T.Compose([
        T.Resize((224, 224), T.InterpolationMode.BICUBIC),
        T.ToTensor(), 
        T.Normalize(mean, std)
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4)

    paradigms = ["clip", "supervised"]
    iterations = [1, 2, 3]
    summary_stats = {}

    for p in paradigms:
        # Sets the global variable read by train_linear_probe
        paradigm = p 
        run_accuracies = []
        print(f"\n{'='*40}\nStarting ViT Analysis: {p.upper()}\n{'='*40}")

        for i in iterations:
            model_path = f"./out/VIT/vit_untrained_{p}_run{i}.pt"
            if not os.path.exists(model_path):
                print(f"Skipping: {model_path} (not found)")
                continue

            # load model
            model = load_vit_model(model_path, p)
            if model is None:
                continue

            # isolate backbone for 768 feature dimensions
            if p == "clip":
                backbone = model.image_encoder   
            elif p == "vae":
                backbone = model.encoder.image_encoder
            else:  # supervised
                backbone = model.features

            # train linear probe
            print(f">>> Train linear Probe on frozen {p.upper()} features (768-dim)...")
            acc = train_linear_probe(backbone, train_loader, test_loader)
            
            run_accuracies.append(acc)
            print(f"  Run {i}: {acc:.2f}%")
            
            # Empty RAM/VRAM cache 
            del model, backbone
            torch.cuda.empty_cache()
            gc.collect()

        if run_accuracies:
            mean_acc = round(float(np.mean(run_accuracies)), 2)
            std_acc  = round(float(np.std(run_accuracies)), 2)
            summary_stats[p] = {
                "mean": mean_acc,
                "std": std_acc,
                "all_runs": [round(a, 2) for a in run_accuracies]
            }
            print(f"\n  ► {p.upper()} | Mean Acc: {mean_acc}% | Std: {std_acc}%")

    with open(os.path.join(OUT_DIR, "vit_untrained_final_metrics.json"), "w") as f:
        json.dump(summary_stats, f, indent=4)
    print(f"\nResults successfully saved to {OUT_DIR}")