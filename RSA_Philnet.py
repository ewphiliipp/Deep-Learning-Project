import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# 1. CONFIG
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RSA_OUT_DIR = "./out/rsa_results/"
MODEL_DIR   = "./out/Philnet/"
os.makedirs(RSA_OUT_DIR, exist_ok=True)

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

# 3. LOAD MODEL 

def load_philnet_model(model_path, paradigm):
    if not os.path.exists(model_path):
        print(f"  [SKIP] Not found: {model_path}")
        return None

    if paradigm == "clip":
        model          = CLIPNet(projection_dim=256).to(DEVICE)
        target_backbone = model.image_encoder.model   # Keys: image_encoder.model.X
    elif paradigm == "vae":
        model          = VAE(latent_dim=256).to(DEVICE)
        target_backbone = model.encoder.model          # Keys: encoder.model.X  (VAE hat keinen features-layer)
    else:
        model          = PhilNet(num_classes=10).to(DEVICE)
        target_backbone = model.features.model         # Keys: features.model.X oder features.X

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    sd = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint)) \
        if isinstance(checkpoint, dict) else checkpoint

    # Universal Key-Mapping
    new_sd      = {}
    target_keys = list(target_backbone.state_dict().keys())
    for k, v in sd.items():
        clean = (k
                 .replace("image_encoder.model.", "")
                 .replace("encoder.model.features.", "")   # VAE: encoder.model.features.X → X
                 .replace("encoder.model.", "")
                 .replace("features.model.", "")
                 .replace("features.", ""))
        if clean in target_keys and v.shape == target_backbone.state_dict()[clean].shape:
            new_sd[clean] = v

    target_backbone.load_state_dict(new_sd, strict=False)
    print(f"  [{paradigm.upper()}] Loaded {len(new_sd)}/{len(target_keys)} backbone layers")
    return model.eval()

# 4. FEATURE EXTRACTION

def get_philnet_features(model, images_tensor, paradigm, batch_size=128):
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(images_tensor), batch_size):
            batch = images_tensor[i:i+batch_size].to(DEVICE)

            if paradigm == "supervised":
                # Just Encoder
                f = model.features(batch)                               # [B, 512]

            elif paradigm == "clip":
                # Encoder + Projection + L2-Normalisation
                f = model.image_encoder(batch)                          # [B, 512]
                f = model.image_projection(f)                           # [B, 256]
                f = F.normalize(f, p=2, dim=-1)                         # L2-norm

            elif paradigm == "vae":
                # mu as representational vector
                f = model(batch)                                        # [B, 256]

            all_feats.append(f.cpu().numpy())
    return np.concatenate(all_feats, axis=0)

# 5. RSA & Regression

def compute_rdm_01(data):
    corr = np.corrcoef(data)
    rdm  = (1 - corr) / 2
    return np.nan_to_num(rdm, nan=0.5)


def get_rsa_score(m1, m2):
    iu = np.triu_indices_from(m1, k=1)
    return float(np.corrcoef(m1[iu], m2[iu])[0, 1])

def run_rsa_analysis(images_sorted, human_probs_sorted):
    paradigms  = ["supervised", "clip", "vae"]
    iterations = [1, 2, 3]

    print("Berechne Human RDM (10k x 10k)...")
    rdm_human = compute_rdm_01(human_probs_sorted)

    final_stats = {}

    for p in paradigms:
        run_scores = []
        print(f"\n{'='*30}\nParadigm: {p.upper()}\n{'='*30}")

        for i in iterations:
            model_path = os.path.join(MODEL_DIR, f"philnet_{p}_run{i}.pt")
            model      = load_philnet_model(model_path, p)
            if model is None:
                continue

            # 1. Features extraction
            feats = get_philnet_features(model, images_sorted, p)
            print(f"  Run {i}: Features {feats.shape}")

            # 2. Ridge Regression (10-Fold, alpha=0.1, no Logit-Transformation)
            kf           = KFold(n_splits=10, shuffle=True, random_state=42)
            scaler       = StandardScaler()
            feats_scaled = scaler.fit_transform(feats)
            pred_probs   = np.zeros_like(human_probs_sorted)

            for train_idx, test_idx in kf.split(feats_scaled):
                reg = Ridge(alpha=0.1)
                reg.fit(feats_scaled[train_idx], human_probs_sorted[train_idx])
                pred_probs[test_idx] = reg.predict(feats_scaled[test_idx])

            # 3. RDM & RSA Score
            rdm_model = compute_rdm_01(pred_probs)
            score     = get_rsa_score(rdm_model, rdm_human)
            run_scores.append(score)

            # 4. Heatmap (Scale 0–1)
            plt.figure(figsize=(10, 8))
            sns.heatmap(rdm_model, cmap="viridis", vmin=0, vmax=1,
                        xticklabels=False, yticklabels=False)
            for line in range(0, 10001, 1000):
                plt.axhline(line, color='white', lw=0.5, alpha=0.3)
                plt.axvline(line, color='white', lw=0.5, alpha=0.3)
            plt.title(f"PhilNet {p.upper()} Run {i} | RSA: {score:.4f}")
            plt.tight_layout()
            plt.savefig(os.path.join(RSA_OUT_DIR, f"rdm_{p}_run{i}.png"), dpi=150)
            plt.close()

            print(f"  Run {i}: RSA = {score:.4f}")
            del model, rdm_model
            torch.cuda.empty_cache()

        if run_scores:
            final_stats[p] = {
                "mean":     round(float(np.mean(run_scores)), 4),
                "std":      round(float(np.std(run_scores)),  4),
                "all_runs": [round(s, 4) for s in run_scores]
            }
            print(f"\n  ► {p.upper()} | Mean RSA: {final_stats[p]['mean']:.4f} "
                  f"± {final_stats[p]['std']:.4f}")

    # save JSON
    out_json = os.path.join(RSA_OUT_DIR, "summary_philnet.json")
    with open(out_json, "w") as f:
        json.dump(final_stats, f, indent=4)
    print(f"\nResults saved: {out_json}")

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
    run_rsa_analysis(images_sorted, human_probs_sorted)