"""
Combined pipeline for OpenCLIP RN50 (OpenAI weights, QuickGELU): linear-probe
accuracy, final RSA, layerwise RSA, and entropy-quartile RSA — all in one
script, analogous in structure to resnet_linear_probe.py, resnet_rsa_heatmaps.py,
resnet_layerwise_rsa.py, and resnet_entropy_rsa.py, but adapted for OpenCLIP's
own ModifiedResNet visual tower instead of a torchvision resnet50.

Why a dedicated script instead of reusing the torchvision-based pipeline:
open_clip's `model.visual` for RN50 is OpenCLIP's own `ModifiedResNet` class,
not torchvision.models.resnet50. It has a 3-conv stem (conv1/bn1/relu1,
conv2/bn2/relu2, conv3/bn3/relu3, avgpool) instead of torchvision's single
conv1+bn1+relu+maxpool stem, and ends in an AttentionPool2d ("attnpool")
instead of a plain avgpool+fc. Weights are loaded directly via
open_clip.create_model_and_transforms (no checkpoint/position-matching needed,
since OpenCLIP already ships correctly-initialized pretrained weights) and
layerwise features are extracted with forward hooks instead of hardcoded
torchvision attribute access.

Requires: pip install open_clip_torch --break-system-packages
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import open_clip

# ==============================================================================
# 1. CONFIG
# ==============================================================================
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_MODEL   = "RN50-quickgelu"
PRETRAINED   = "openai"
ARCH_TAG     = "resnet50_clip"   # used in output filenames/folders

OUT_DIR      = f"./out/rsa_results_{ARCH_TAG}/"
LAYER_OUT_DIR = f"./out/rsa_layerwise_{ARCH_TAG}/"
ACC_OUT_DIR  = "./out/accuracy_results/"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LAYER_OUT_DIR, exist_ok=True)
os.makedirs(ACC_OUT_DIR, exist_ok=True)

NUM_CLASSES = 10
EPOCHS      = 15

# Layer steps for the layerwise analysis. OpenCLIP's ModifiedResNet has a
# 3-conv stem (conv1..conv3) instead of torchvision's single conv1, so "0pct"
# is taken right after the full stem + avgpool (end of conv3/bn3/relu3/avgpool),
# not after a single conv layer.
LAYER_STEPS = {
    "0pct":   "stem",     # after conv1-3 + bn + relu + avgpool
    "20pct":  "layer1",   # after layer1 (3 Bottleneck blocks)
    "40pct":  "layer2",   # after layer2 (4 Bottleneck blocks)
    "60pct":  "layer3",   # after layer3 (6 Bottleneck blocks)
    "80pct":  "layer4",   # after layer4 (3 Bottleneck blocks)
    "100pct": "final",    # after attnpool -> 1024-dim projected embedding
}

PARADIGM = "clip"  # this script only covers the CLIP paradigm (image-text contrastive)


# ==============================================================================
# 2. LOAD MODEL (direct from OpenCLIP — no checkpoint/position-matching needed)
# ==============================================================================
def load_clip_rn50():
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=PRETRAINED
    )
    model.eval().to(DEVICE)
    visual = model.visual
    embedding_dim = visual.output_dim if hasattr(visual, "output_dim") else 1024
    print(f"Loaded {CLIP_MODEL} (pretrained='{PRETRAINED}') — visual embedding dim: {embedding_dim}")
    return model, visual, embedding_dim


# ==============================================================================
# 3. FEATURE EXTRACTION (forward hooks for layerwise; direct call for final)
# ==============================================================================
def get_final_clip_features(visual, images_tensor, batch_size=32):
    """Full CLIP visual forward pass -> final projected embedding (post attnpool)."""
    visual.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(images_tensor), batch_size):
            batch = images_tensor[i:i + batch_size].to(DEVICE)
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
            f = visual(batch)
            f = F.normalize(f, p=2, dim=-1)
            all_feats.append(f.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


@torch.no_grad()
def extract_at_layer(visual, images_tensor, layer_id, batch_size=32):
    """Extract pooled features at a given depth of OpenCLIP's ModifiedResNet."""
    visual.eval()
    all_feats = []
    for i in range(0, len(images_tensor), batch_size):
        batch = images_tensor[i:i + batch_size].to(DEVICE)
        batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)

        # OpenCLIP ModifiedResNet stem: conv1/bn1/act1 -> conv2/bn2/act2 ->
        # conv3/bn3/act3 -> avgpool (3x conv instead of torchvision's single conv1).
        # Uses the class's own .stem() method (verified against the current
        # open_clip source) instead of hand-rebuilding it with wrong attribute
        # names — ModifiedResNet uses "act1/2/3", not "relu1/2/3".
        x = visual.stem(batch)

        if layer_id == "stem":
            f = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            all_feats.append(f.cpu().numpy())
            continue

        x = visual.layer1(x)
        if layer_id == "layer1":
            f = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            all_feats.append(f.cpu().numpy())
            continue

        x = visual.layer2(x)
        if layer_id == "layer2":
            f = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            all_feats.append(f.cpu().numpy())
            continue

        x = visual.layer3(x)
        if layer_id == "layer3":
            f = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            all_feats.append(f.cpu().numpy())
            continue

        x = visual.layer4(x)
        if layer_id == "layer4":
            f = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            all_feats.append(f.cpu().numpy())
            continue

        # final: run through the attention pool head (projects to embedding_dim)
        f = visual.attnpool(x)
        f = F.normalize(f, p=2, dim=-1)
        all_feats.append(f.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


# ==============================================================================
# 4. LINEAR PROBE (accuracy)
# ==============================================================================
class LinearProbeHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def train_linear_probe(visual, train_loader, test_loader):
    visual.eval()
    with torch.no_grad():
        test_imgs, _ = next(iter(train_loader))
        test_imgs = F.interpolate(test_imgs.to(DEVICE), size=(224, 224),
                                   mode="bilinear", align_corners=False)
        test_features = visual(test_imgs)
        input_dim = test_features.shape[-1]

    head = LinearProbeHead(input_dim, NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        head.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            imgs = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)
            with torch.no_grad():
                features = visual(imgs)
            optimizer.zero_grad()
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()

    head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            imgs = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)
            features = visual(imgs)
            outputs = head(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100


# ==============================================================================
# 5. RSA CORE
# ==============================================================================
def compute_rdm_01(data):
    corr = np.corrcoef(data)
    rdm = (1 - corr) / 2
    return np.nan_to_num(rdm, nan=0.5)


def get_rsa_score(m1, m2):
    iu = np.triu_indices_from(m1, k=1)
    return float(np.corrcoef(m1[iu], m2[iu])[0, 1])


def run_regression_and_rsa(feats, human_probs, rdm_human, alpha=2.5):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    pred_probs = np.zeros_like(human_probs)
    for train_idx, test_idx in kf.split(feats_scaled):
        reg = Ridge(alpha=alpha)
        reg.fit(feats_scaled[train_idx], human_probs[train_idx])
        pred_probs[test_idx] = reg.predict(feats_scaled[test_idx])
    rdm_model = compute_rdm_01(pred_probs)
    return get_rsa_score(rdm_model, rdm_human), rdm_model


def compute_entropy(probs):
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


# ==============================================================================
# 6. PIPELINE STEPS
# ==============================================================================
def run_accuracy(visual, train_loader, test_loader, iterations=(1, 2, 3)):
    print(f"\n{'='*40}\nACCURACY (linear probe): {CLIP_MODEL}\n{'='*40}")
    run_accuracies = []
    for i in iterations:
        acc = train_linear_probe(visual, train_loader, test_loader)
        run_accuracies.append(acc)
        print(f"  Run {i}: {acc:.2f}%")
    summary = {
        PARADIGM: {
            "mean": round(float(np.mean(run_accuracies)), 2),
            "std": round(float(np.std(run_accuracies)), 2),
            "all_runs": [round(a, 2) for a in run_accuracies],
        }
    }
    out_path = os.path.join(ACC_OUT_DIR, f"{ARCH_TAG}_final_metrics.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"  ► Mean Acc: {summary[PARADIGM]['mean']}% | Std: {summary[PARADIGM]['std']}%")
    print(f"  Saved: {out_path}")


def run_final_rsa(visual, images_sorted, human_probs_sorted, iterations=(1, 2, 3)):
    print(f"\n{'='*40}\nFINAL RSA: {CLIP_MODEL}\n{'='*40}")
    rdm_human = compute_rdm_01(human_probs_sorted)
    run_scores = []
    for i in iterations:
        feats = get_final_clip_features(visual, images_sorted)
        score, rdm_model = run_regression_and_rsa(feats, human_probs_sorted, rdm_human)
        run_scores.append(score)

        plt.figure(figsize=(10, 8))
        sns.heatmap(rdm_model, cmap="viridis", vmin=0, vmax=1, xticklabels=False, yticklabels=False)
        plt.title(f"{ARCH_TAG.upper()} {PARADIGM.upper()} Run {i} | RSA: {score:.4f}")
        rdm_filename = f"rdm_{PARADIGM}_{ARCH_TAG}_run{i}.png"
        plt.savefig(os.path.join(OUT_DIR, rdm_filename), dpi=150)
        plt.close()
        print(f"  Run {i}: RSA = {score:.4f}  (saved {rdm_filename})")

    final_stats = {PARADIGM: {"mean": float(np.mean(run_scores)), "std": float(np.std(run_scores))}}
    summary_path = os.path.join(OUT_DIR, f"summary_{ARCH_TAG}.json")
    with open(summary_path, "w") as f:
        json.dump(final_stats, f, indent=4)
    print(f"  Saved: {summary_path}")


def run_layerwise_rsa(visual, images_sorted, human_probs_sorted, iterations=(1, 2, 3)):
    print(f"\n{'='*40}\nLAYERWISE RSA: {CLIP_MODEL}\n{'='*40}")
    rdm_human = compute_rdm_01(human_probs_sorted)
    rows = []
    layer_scores = {l_name: [] for l_name in LAYER_STEPS}

    for i in iterations:
        print(f"\n  Run {i}:")
        for l_name, l_id in LAYER_STEPS.items():
            feats = extract_at_layer(visual, images_sorted, l_id)
            score, _ = run_regression_and_rsa(feats, human_probs_sorted, rdm_human)
            layer_scores[l_name].append(score)
            print(f"    Layer {l_name:>8} (id={l_id:<8}, dim={feats.shape[1]:>5}): RSA = {score:.4f}")

    for l_name, scores in layer_scores.items():
        mean_val = round(float(np.mean(scores)), 5)
        std_val = round(float(np.std(scores)), 5)
        rows.append({
            "paradigm": PARADIGM,
            "layer": l_name,
            "mean_rsa": mean_val,
            "std_rsa": std_val,
            "runs": [round(float(s), 5) for s in scores],
        })
        print(f"  ► {l_name}: Mean={mean_val:.4f} ± {std_val:.4f}")

    out_json = os.path.join(LAYER_OUT_DIR, f"{ARCH_TAG}_layerwise.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=4)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(LAYER_OUT_DIR, f"{ARCH_TAG}_layerwise.csv"), index=False)
    print(f"  Saved: {out_json}")
    return df


def run_entropy_quartile_rsa(visual, imgs, lbls, all_human_probs, iterations=(1, 2, 3)):
    print(f"\n{'='*40}\nENTROPY-QUARTILE RSA: {CLIP_MODEL}\n{'='*40}")
    entropy = compute_entropy(all_human_probs)
    print(f"Entropy range: {entropy.min():.4f} – {entropy.max():.4f}")

    sorted_by_entropy = np.argsort(entropy)
    n_total = len(sorted_by_entropy)
    split = n_total // 4
    bins = [
        (sorted_by_entropy[0:split], "entropy_Q1_0-25pct"),
        (sorted_by_entropy[split:split * 2], "entropy_Q2_25-50pct"),
        (sorted_by_entropy[split * 2:split * 3], "entropy_Q3_50-75pct"),
        (sorted_by_entropy[split * 3:], "entropy_Q4_75-100pct"),
    ]

    results = {}
    for seg_indices, bin_label in bins:
        n_images = len(seg_indices)
        e_min = entropy[seg_indices].min()
        e_max = entropy[seg_indices].max()
        print(f"\n{'='*40}\nBin: {bin_label}  entropy=[{e_min:.4f}, {e_max:.4f}]  N={n_images}\n{'='*40}")

        seg_labels = lbls[seg_indices]
        sort_order = np.argsort(seg_labels)
        sorted_idx = seg_indices[sort_order]
        images_bin = imgs[sorted_idx]
        probs_bin = all_human_probs[sorted_idx]
        rdm_human = compute_rdm_01(probs_bin)

        results[bin_label] = {
            "n": int(n_images),
            "entropy_range": [round(float(e_min), 4), round(float(e_max), 4)],
        }

        run_scores = []
        for i in iterations:
            feats = get_final_clip_features(visual, images_bin)
            score, _ = run_regression_and_rsa(feats, probs_bin, rdm_human)
            run_scores.append(score)
            print(f"    Run {i}: RSA = {score:.4f}")

        results[bin_label][PARADIGM] = {
            "mean": round(float(np.mean(run_scores)), 4),
            "std": round(float(np.std(run_scores)), 4),
            "all_runs": [round(s, 4) for s in run_scores],
        }
        print(f"    ► Mean: {results[bin_label][PARADIGM]['mean']:.4f} "
              f"± {results[bin_label][PARADIGM]['std']:.4f}")

    out_json = os.path.join(OUT_DIR, f"rsa_entropy_{ARCH_TAG}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved: {out_json}")
    return results


# ==============================================================================
# 7. MAIN
# ==============================================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    model, visual, embedding_dim = load_clip_rn50()

    transform = T.Compose([
        T.Resize((224, 224), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.489, 0.4321, 0.4976], [0.22734, 0.24983, 0.2357]),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader   = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    loader_full = DataLoader(test_dataset, batch_size=10000, shuffle=False)
    imgs, lbls = next(iter(loader_full))
    lbls_np = lbls.numpy()
    sort_idx = torch.argsort(lbls)
    images_sorted = imgs[sort_idx]

    if not os.path.exists("./cifar10h-probs.npy"):
        raise FileNotFoundError("./cifar10h-probs.npy not found! Required for RSA against human judgments.")

    human_probs = np.load("./cifar10h-probs.npy")
    human_probs_sorted = human_probs[sort_idx.numpy()]
    print(f"Images: {images_sorted.shape} | Human probs: {human_probs_sorted.shape}")

    # 1. Accuracy
    run_accuracy(visual, train_loader, test_loader)

    # 2. Final RSA (heatmaps)
    run_final_rsa(visual, images_sorted, human_probs_sorted)

    # 3. Layerwise RSA
    run_layerwise_rsa(visual, images_sorted, human_probs_sorted)

    # 4. Entropy-quartile RSA
    run_entropy_quartile_rsa(visual, imgs, lbls_np, human_probs)

    print(f"\n{'#'*60}\nAll analyses complete for {CLIP_MODEL} ({ARCH_TAG}).\n{'#'*60}")