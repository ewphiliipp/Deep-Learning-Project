import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import json
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# 1. CONFIG
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RSA_OUT_DIR = "./out/rsa_layerwise_philnet/"
MODEL_DIR   = "./out/Philnet/"
os.makedirs(RSA_OUT_DIR, exist_ok=True)

LAYER_STEPS = {
    "0pct":   2,   # after Block1 Conv1 ReLU
    "20pct":  5,   # after Block1 Conv2 ReLU
    "40pct":  9,   # after Block2 Conv1 ReLU
    "60pct":  12,  # after Block2 Conv2 ReLU
    "80pct":  16,  # after Block3 Conv1 ReLU
    "100pct": 20,  # after AdaptiveAvgPool (final Features)
}

# 2. architectures


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
        self.features   = PhilNetImageEncoder()
        self.classifier = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.classifier(self.features(x))


class CLIPNet(nn.Module):
    def __init__(self, projection_dim=256):
        super().__init__()
        self.image_encoder    = PhilNetImageEncoder()
        self.image_projection = nn.Linear(512, projection_dim)
    def forward(self, x):
        return self.image_projection(self.image_encoder(x))


class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = PhilNetImageEncoder()
        self.mu      = nn.Linear(512, latent_dim)
    def forward(self, x):
        return self.mu(self.encoder(x))

# 3. load models

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
    print(f"  [{paradigm.upper()}] Loaded {len(new_sd)}/{len(target_keys)} backbone layers")
    return model.eval()

# 4. layer feature extraction

def get_backbone(model, paradigm):
    """Gives back sequential backbone."""
    if paradigm == "clip":
        return model.image_encoder.model
    elif paradigm == "vae":
        return model.encoder.model
    else:
        return model.features.model


@torch.no_grad()
def extract_at_layer(model, images_tensor, paradigm, layer_idx, batch_size=64):
    model.eval()
    backbone  = get_backbone(model, paradigm)
    all_feats = []

    for i in range(0, len(images_tensor), batch_size):
        batch = images_tensor[i:i+batch_size].to(DEVICE)

        x = batch
        for j, layer in enumerate(backbone):
            x = layer(x)
            if j == layer_idx:
                break

        f = x.reshape(x.size(0), -1) 

        # NUR FÜR CLIP BEI 100pct: Projection + L2-Norm
        if paradigm == "clip" and layer_idx == 20:
            f = model.image_projection(f)
            f = F.normalize(f, p=2, dim=-1)

        all_feats.append(f.cpu().numpy())  

    return np.concatenate(all_feats, axis=0)

# 5.RSA

def compute_rdm_01(data):
    corr = np.corrcoef(data)
    rdm  = (1 - corr) / 2
    return np.nan_to_num(rdm, nan=0.5)


def get_rsa_score(m1, m2):
    iu = np.triu_indices_from(m1, k=1)
    return float(np.corrcoef(m1[iu], m2[iu])[0, 1])


def run_regression_and_rsa(feats, human_probs, rdm_human):
    kf           = KFold(n_splits=10, shuffle=True, random_state=42)
    scaler       = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    pred_probs   = np.zeros_like(human_probs)

    for train_idx, test_idx in kf.split(feats_scaled):
        reg = Ridge(alpha=0.1)
        reg.fit(feats_scaled[train_idx], human_probs[train_idx])
        pred_probs[test_idx] = reg.predict(feats_scaled[test_idx])

    rdm_model = compute_rdm_01(pred_probs)
    return get_rsa_score(rdm_model, rdm_human)

# 6. layerwise pipeline

def run_layerwise_rsa(images_sorted, human_probs_sorted):
    paradigms  = ["clip", "vae", "supervised"]
    iterations = [1, 2, 3]

    print("Berechne Human RDM...")
    rdm_human = compute_rdm_01(human_probs_sorted)

    rows = []  # für JSON + CSV

    for p in paradigms:
        print(f"\n{'='*40}\nParadigm: {p.upper()}\n{'='*40}")

        # Scores per layer for all runs
        layer_scores = {l_name: [] for l_name in LAYER_STEPS}

        for i in iterations:
            model_path = os.path.join(MODEL_DIR, f"philnet_{p}_run{i}.pt")
            model      = load_philnet_model(model_path, p)
            if model is None:
                continue

            print(f"\n  Run {i}:")
            for l_name, l_idx in LAYER_STEPS.items():
                feats = extract_at_layer(model, images_sorted, p, l_idx)
                score = run_regression_and_rsa(feats, human_probs_sorted, rdm_human)
                layer_scores[l_name].append(score)
                print(f"    Layer {l_name:>8} (idx={l_idx:>2}, dim={feats.shape[1]:>6}): RSA = {score:.4f}")

            del model
            torch.cuda.empty_cache()

        # calculate statistics
        for l_name, scores in layer_scores.items():
            if scores:
                mean_val = round(float(np.mean(scores)), 5)
                std_val  = round(float(np.std(scores)),  5)
                rows.append({
                    "paradigm": p,
                    "layer":    l_name,
                    "mean_rsa": mean_val,
                    "std_rsa":  std_val,
                    "runs":     [round(float(s), 5) for s in scores]
                })
                print(f"  ► {l_name}: Mean={mean_val:.4f} ± {std_val:.4f}")

    # save
    out_json = os.path.join(RSA_OUT_DIR, "philnet_layerwise.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=4)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RSA_OUT_DIR, "philnet_layerwise.csv"), index=False)

    print(f"\n{'#'*40}\nResults saved: {RSA_OUT_DIR}\n{'#'*40}")

    # print results
    print(f"\n{'Paradigm':<12} {'Layer':<10} {'Mean RSA':<10} {'Std'}")
    print("-" * 45)
    for row in rows:
        print(f"{row['paradigm']:<12} {row['layer']:<10} {row['mean_rsa']:<10.4f} {row['std_rsa']:.4f}")

    return df

# 7. MAIN

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.489, 0.4321, 0.4976], [0.22734, 0.24983, 0.2357])
    ])

    test_ds    = torchvision.datasets.CIFAR10(root="./data", train=False,
                                               download=True, transform=transform)
    loader     = DataLoader(test_ds, batch_size=10000, shuffle=False)
    imgs, lbls = next(iter(loader))

    sort_idx           = torch.argsort(lbls)
    images_sorted      = imgs[sort_idx]
    human_probs        = np.load("./cifar10h-probs.npy")
    human_probs_sorted = human_probs[sort_idx.numpy()]

    print(f"Images: {images_sorted.shape} | Human probs: {human_probs_sorted.shape}")
    run_layerwise_rsa(images_sorted, human_probs_sorted)