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

# Which ResNet backbone to use: "resnet18" (512-dim, BasicBlock) or "resnet50" (2048-dim, Bottleneck)
ARCH        = "resnet18"   # <- switch to "resnet50" to run the other architecture

ARCH_CONFIG = {
    "resnet18": {"builder": models.resnet18, "embedding_dim": 512},
    "resnet50": {"builder": models.resnet50, "embedding_dim": 2048},
}
if ARCH not in ARCH_CONFIG:
    raise ValueError(f"Unknown ARCH '{ARCH}', expected one of {list(ARCH_CONFIG)}")

EMBEDDING_DIM = ARCH_CONFIG[ARCH]["embedding_dim"]


# 2. ARCHITEKTUREN
class ResNetImageEncoder(nn.Module):
    def __init__(self, arch=ARCH):
        super().__init__()
        builder = ARCH_CONFIG[arch]["builder"]
        self.embedding_dim = ARCH_CONFIG[arch]["embedding_dim"]
        self.model = builder(weights=None)
        # place fc on Identity, to get the raw feature vector (512 for resnet18, 2048 for resnet50)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


class ResNetSupervised(nn.Module):
    def __init__(self, num_classes=10, arch=ARCH):
        super().__init__()
        self.features = ResNetImageEncoder(arch)
        self.classifier = nn.Linear(self.features.embedding_dim, num_classes)

    def forward(self, x):
        f = self.features(x)
        return self.classifier(f)


class CLIPNet(nn.Module):
    def __init__(self, text_model_name="distilbert-base-uncased", projection_dim=256, arch=ARCH):
        super().__init__()
        self.image_encoder = ResNetImageEncoder(arch)
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        self.image_projection = nn.Linear(self.image_encoder.embedding_dim, projection_dim)
        self.text_projection = nn.Linear(768, projection_dim)


class GenericEncoder(nn.Module):
    def __init__(self, latent_dim=256, arch=ARCH):
        super().__init__()
        self.image_encoder = ResNetImageEncoder(arch)
        self.mu = nn.Linear(self.image_encoder.embedding_dim, latent_dim)
        self.logvar = nn.Linear(self.image_encoder.embedding_dim, latent_dim)

    def forward(self, x):
        f = self.image_encoder(x)
        return self.mu(f), self.logvar(f)


class VAE(nn.Module):
    def __init__(self, latent_dim=256, arch=ARCH):
        super().__init__()
        self.encoder = GenericEncoder(latent_dim, arch)


# 3. load model weights — name-based with shape check (safe for both resnet18 & resnet50)
def load_resnet_model(model_path, paradigm, arch=ARCH):
    if not os.path.exists(model_path):
        print(f"  [SKIP] Not found: {model_path}")
        return None

    # initialize model architecture
    if paradigm == "clip":
        model = CLIPNet("distilbert-base-uncased", 256, arch=arch).to(DEVICE)
        target_backbone = model.image_encoder.model
    elif paradigm == "vae":
        model = VAE(latent_dim=256, arch=arch).to(DEVICE)
        target_backbone = model.encoder.image_encoder.model
    else:
        model = ResNetSupervised(num_classes=10, arch=arch).to(DEVICE)
        target_backbone = model.features.model

    # load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint)) \
        if isinstance(checkpoint, dict) else checkpoint

    # Name-based mapping with a shape check. This is robust across resnet18/resnet50
    # because torchvision uses the same parameter names for both (conv1, bn1, layer1..4,
    # fc), it's only the per-block internals (BasicBlock vs. Bottleneck) that differ in
    # shape. A pure position-based match (as before) silently mismatches between the two
    # architectures since block counts and tensor shapes/order diverge.
    current_dict = target_backbone.state_dict()
    new_state_dict = {}

    # allow for an optional common prefix mismatch, e.g. "model." or "backbone."
    def strip_known_prefixes(key):
        for prefix in ("model.", "backbone.", "image_encoder.model.", "features.model."):
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    normalized_source = {strip_known_prefixes(k): v for k, v in state_dict.items()}

    matched_count = 0
    skipped_shape_mismatch = []
    skipped_missing = []

    print(f"  [{paradigm.upper()}/{arch}] Try name-based mapping with shape check...")

    for t_key, t_tensor in current_dict.items():
        if t_key in normalized_source:
            src_tensor = normalized_source[t_key]
            if src_tensor.shape == t_tensor.shape:
                new_state_dict[t_key] = src_tensor
                matched_count += 1
            else:
                skipped_shape_mismatch.append((t_key, tuple(src_tensor.shape), tuple(t_tensor.shape)))
        else:
            skipped_missing.append(t_key)

    target_backbone.load_state_dict(new_state_dict, strict=False)
    print(f"  [{paradigm.upper()}/{arch}] Loaded {matched_count}/{len(current_dict)} layers via name+shape match.")
    if skipped_shape_mismatch:
        print(f"  [{paradigm.upper()}/{arch}] {len(skipped_shape_mismatch)} keys skipped due to shape mismatch "
              f"(likely wrong ARCH for this checkpoint). First few: {skipped_shape_mismatch[:3]}")
    if skipped_missing:
        print(f"  [{paradigm.upper()}/{arch}] {len(skipped_missing)} target keys had no matching source key.")

    return model.eval()


# 4. LINEAR PROBE HELPERS
class LinearProbeHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def train_linear_probe(backbone, train_loader, test_loader):
    """Train a linear classifier on frozen features (dimension depends on ARCH: 512 for
    resnet18, 2048 for resnet50 — determined dynamically from the backbone output)."""
    backbone.eval()
    with torch.no_grad():
        test_imgs, _ = next(iter(train_loader))
        test_imgs = test_imgs.to(DEVICE)
        test_features = backbone(test_imgs)
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
    print(f"Using architecture: {ARCH} (embedding dim: {EMBEDDING_DIM})")

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

    paradigms  = ["supervised", "clip"]
    iterations = [1, 2, 3]
    summary_stats = {}

    for p in paradigms:
        run_accuracies = []
        print(f"\n{'='*40}\nStarting {ARCH.upper()} Analysis: {p.upper()}\n{'='*40}")
        for i in iterations:
            # checkpoint filename now driven by ARCH, e.g. resnet18_supervised_run1.pt
            # or resnet50_supervised_run1.pt
            model_path = f"./out/Resnet/{ARCH}_{p}_run{i}.pt"
            if not os.path.exists(model_path):
                print(f"Skipping: {model_path} (not found)")
                continue

            # 1. load model
            model = load_resnet_model(model_path, p, arch=ARCH)
            if model is None:
                continue

            # 2. isolate backbone for feature extraction
            if p == "clip":
                backbone = model.image_encoder
            elif p == "vae":
                backbone = model.encoder.image_encoder
            else:  # supervised
                backbone = model.features

            # 3. train linear probe
            print(f">>> Train linear probe on frozen {p.upper()} representations "
                  f"({EMBEDDING_DIM}-dim, {ARCH})...")
            acc = train_linear_probe(backbone, train_loader, test_loader)

            run_accuracies.append(acc)
            print(f"  Run {i}: {acc:.2f}%")

            # free RAM/VRAM
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

    # output filename now includes ARCH so resnet18 and resnet50 runs don't overwrite each other
    out_path = os.path.join(OUT_DIR, f"{ARCH}_final_metrics.json")
    with open(out_path, "w") as f:
        json.dump(summary_stats, f, indent=4)
    print(f"\nResults successfully saved to {out_path}")