import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
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

# 2. ARCHITEKTURES

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

# 3. LOAD MODELL

def load_philnet_model(model_path, paradigm):
    if not os.path.exists(model_path):
        print(f"  [SKIP] Not found: {model_path}")
        return None

    if paradigm == "clip":
        model          = CLIPNet(projection_dim=256).to(DEVICE)
        target_backbone = model.image_encoder.model   # Keys: image_encoder.model.X
    elif paradigm == "vae":
        model          = VAE(latent_dim=256).to(DEVICE)
        target_backbone = model.encoder.model          # Keys: encoder.model.X  
    else:
        model          = PhilNet(num_classes=10).to(DEVICE)
        target_backbone = model.features.model         # Keys: features.model.X or features.X

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

def get_philnet_features(model, images_tensor, paradigm, batch_size=64):
    """
    Extracted Features. 
    NEU: Scales Images for VAE-Paradigm to 224x224
    """
    model.eval()
    all_feats = []
    
    # reduced Batch size for Upscaling, to mitigate VRAM-Error
    current_batch_size = 32 if paradigm == "vae" else batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(images_tensor), current_batch_size), 
                      desc=f"  Extracting {paradigm}"):
            batch = images_tensor[i:i+current_batch_size].to(DEVICE)
            
            if paradigm == "vae":
                batch = F.interpolate(batch, size=(224, 224), 
                                      mode='bilinear', align_corners=False)

            if paradigm == "supervised":
                f = model.features(batch)                               # [B, 512]
            elif paradigm == "clip":
                f = model.image_encoder(batch)                          # [B, 512]
                f = model.image_projection(f)                           # [B, 256]
                f = F.normalize(f, p=2, dim=-1)                         # L2-norm
            elif paradigm == "vae":
                # mu als Repräsentationsvektor (nach dem 224x224 Forward)
                f = model(batch)                                        # [B, 256]

            all_feats.append(f.cpu().numpy())
            
    return np.concatenate(all_feats, axis=0)

# 5. RSA

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
 
# 6. ENTROPY PIPELINE

def compute_entropy(probs):
    """Shannon Entropie per Image. probs: [N, 10]"""
    probs  = np.clip(probs, 1e-10, 1.0)   # log(0) vermeiden
    return -np.sum(probs * np.log(probs), axis=1)
 
 
def run_entropy_rsa(imgs, lbls, all_human_probs):
    paradigms  = ["supervised", "clip", "vae"]
    iterations = [1, 2, 3]

    entropy = compute_entropy(all_human_probs)
    print(f"Entropy range: {entropy.min():.4f} – {entropy.max():.4f}")

    # equally large bins for entropy
    sorted_by_entropy = np.argsort(entropy)
    n_total = len(sorted_by_entropy)
    split   = n_total // 4

    bins = [
        (sorted_by_entropy[0       : split  ], "entropy_Q1_0-25pct"),
        (sorted_by_entropy[split   : split*2], "entropy_Q2_25-50pct"),
        (sorted_by_entropy[split*2 : split*3], "entropy_Q3_50-75pct"),
        (sorted_by_entropy[split*3 :        ], "entropy_Q4_75-100pct"),
    ]

    results = {}

    for seg_indices, bin_label in bins:
        n_images = len(seg_indices)
        e_min    = entropy[seg_indices].min()
        e_max    = entropy[seg_indices].max()
        print(f"\n{'='*40}\nBin: {bin_label}  entropy=[{e_min:.4f}, {e_max:.4f}]  N={n_images}\n{'='*40}")

        seg_labels = lbls[seg_indices]
        sort_order = np.argsort(seg_labels)
        sorted_idx = seg_indices[sort_order]

        images_bin = imgs[sorted_idx]
        probs_bin  = all_human_probs[sorted_idx]
        rdm_human  = compute_rdm_01(probs_bin)

        results[bin_label] = {
            "n":             int(n_images),
            "entropy_range": [round(float(e_min), 4), round(float(e_max), 4)]
        }

        for p in paradigms:
            run_scores = []
            print(f"\n  Paradigm: {p.upper()}")

            for i in iterations:
                model_path = os.path.join(MODEL_DIR, f"philnet_{p}_run{i}.pt")
                model      = load_philnet_model(model_path, p)
                if model is None:
                    continue

                feats = get_philnet_features(model, images_bin, p)
                score = run_regression_and_rsa(feats, probs_bin, rdm_human)
                run_scores.append(score)

                print(f"    Run {i}: RSA = {score:.4f}")
                del model
                torch.cuda.empty_cache()

            if run_scores:
                results[bin_label][p] = {
                    "mean":     round(float(np.mean(run_scores)), 4),
                    "std":      round(float(np.std(run_scores)),  4),
                    "all_runs": [round(s, 4) for s in run_scores]
                }
                print(f"    ► Mean: {results[bin_label][p]['mean']:.4f} "
                      f"± {results[bin_label][p]['std']:.4f}")

    # Save
    out_json = os.path.join(RSA_OUT_DIR, "rsa_entropy_philnet.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n{'#'*40}\nResults saved: {out_json}\n{'#'*40}")

    # Print results
    print(f"\n{'Bin':<25} {'Paradigm':<12} {'Mean RSA':<10} {'Std':<8} {'N'}")
    print("-" * 65)
    for bin_label, bin_data in results.items():
        n = bin_data["n"]
        for p in paradigms:
            if p in bin_data:
                m = bin_data[p]["mean"]
                s = bin_data[p]["std"]
                print(f"{bin_label:<25} {p:<12} {m:<10.4f} {s:<8.4f} {n}")

    return results
 
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
    lbls       = lbls.numpy()
 
    all_human_probs = np.load("./cifar10h-probs.npy")
 
    print(f"Images: {imgs.shape} | Human probs: {all_human_probs.shape}")
    run_entropy_rsa(imgs, lbls, all_human_probs)