import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# 1. CONFIG
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RSA_OUT_DIR = "./out/rsa_layerwise_vit/"
MODEL_DIR   = "./out/VIT/"
os.makedirs(RSA_OUT_DIR, exist_ok=True)

LAYER_STEPS = {
    "0pct":   "patch",  # after Patch Embedding + CLS Token           → 768-dim
    "20pct":  1,        # after Block 2  (2/12  ≈ 17% → ~20%)        → 768-dim
    "40pct":  4,        # after Block 5  (5/12  ≈ 42% → ~40%)        → 768-dim
    "60pct":  7,        # after Block 8  (8/12  ≈ 67% → ~60%)        → 768-dim
    "80pct":  9,        # after Block 10 (10/12 ≈ 83% → ~80%)        → 768-dim
    "100pct": 11,       # after Block 12 + LayerNorm (finale Features) → 768-dim
}

# 2. ARCHITEKTURES

class ViTImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model       = models.vit_b_16(weights=None)
        self.model.heads = nn.Identity()
    def forward(self, x):
        return self.model(x)

class ViTSupervised(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features   = ViTImageEncoder()
        self.classifier = nn.Linear(768, num_classes)
    def forward(self, x):
        return self.classifier(self.features(x))

class CLIPNet(nn.Module):
    def __init__(self, projection_dim=256):
        super().__init__()
        self.image_encoder    = ViTImageEncoder()
        self.image_projection = nn.Linear(768, projection_dim)
    def forward(self, x):
        return self.image_projection(self.image_encoder(x))

class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = ViTImageEncoder()
        self.mu      = nn.Linear(768, latent_dim)
    def forward(self, x):
        return self.mu(self.encoder(x))

# 3. LOAD MODEL

def load_vit_model(model_path, paradigm):
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
        model           = ViTSupervised(num_classes=10).to(DEVICE)
        target_backbone = model.features.model

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint)) \
        if isinstance(checkpoint, dict) else checkpoint

    # Positionsbased mapping
    current_dict   = target_backbone.state_dict()
    new_state_dict = {}
    sd_keys        = [k for k in state_dict.keys() if "heads" not in k]
    target_keys    = list(current_dict.keys())

    matched = 0
    sd_idx  = 0
    for t_key in target_keys:
        while sd_idx < len(sd_keys):
            if state_dict[sd_keys[sd_idx]].shape == current_dict[t_key].shape:
                new_state_dict[t_key] = state_dict[sd_keys[sd_idx]]
                matched += 1
                sd_idx  += 1
                break
            else:
                sd_idx += 1

    target_backbone.load_state_dict(new_state_dict, strict=False)
    print(f"  [{paradigm.upper()}] Loaded {matched}/{len(target_keys)} layers via Position-Match.")

    if paradigm == "clip" and "image_projection.weight" in state_dict:
        model.image_projection.weight = nn.Parameter(state_dict["image_projection.weight"])
        model.image_projection.bias   = nn.Parameter(state_dict["image_projection.bias"])
        print("  [CLIP] Projection head loaded.")
    elif paradigm == "vae" and "mu.weight" in state_dict:
        model.mu.weight = nn.Parameter(state_dict["mu.weight"])
        model.mu.bias   = nn.Parameter(state_dict["mu.bias"])
        print("  [VAE] Mu head loaded.")

    return model.eval()

# 4. LAYER FEATURE EXTRACTION

def get_vit_backbone(model, paradigm):
    if paradigm == "clip":
        return model.image_encoder.model
    elif paradigm == "vae":
        return model.encoder.model
    else:
        return model.features.model


@torch.no_grad()
def extract_at_layer(model, images_tensor, paradigm, layer_id, batch_size=16):
    model.eval()
    vit       = get_vit_backbone(model, paradigm)
    all_feats = []

    for i in range(0, len(images_tensor), batch_size):
        batch = images_tensor[i:i+batch_size].to(DEVICE)
        batch = F.interpolate(batch, size=(224, 224), mode='bicubic', align_corners=False)

        x   = vit._process_input(batch)
        n   = x.shape[0]
        cls = vit.class_token.expand(n, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + vit.encoder.pos_embedding
        x   = vit.encoder.dropout(x)

        if layer_id == "patch":
            all_feats.append(x[:, 0].cpu().numpy())
            continue

        for block_idx, block in enumerate(vit.encoder.layers):
            x = block(x)
            if block_idx == layer_id:
                break

        # Finale LayerNorm only at the last block
        if layer_id == 11:
            x = vit.encoder.ln(x)

        f = x[:, 0]  # CLS Token

        # Clip 100pct: Projection + L2-Norm
        if paradigm == "clip" and layer_id == 11:
            f = model.image_projection(f)
            f = F.normalize(f, p=2, dim=-1)

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

# 6. LAYERWISE PIPELINE

def run_layerwise_rsa(images_sorted, human_probs_sorted):
    paradigms  = ["supervised", "clip"]
    iterations = [1, 2, 3]

    print("Berechne Human RDM...")
    rdm_human = compute_rdm_01(human_probs_sorted)

    rows = []

    for p in paradigms:
        print(f"\n{'='*40}\nParadigm: {p.upper()}\n{'='*40}")

        layer_scores = {l_name: [] for l_name in LAYER_STEPS}

        for i in iterations:
            model_path = os.path.join(MODEL_DIR, f"vit_untrained_{p}_run{i}.pt")
            model      = load_vit_model(model_path, p)
            if model is None:
                continue

            print(f"\n  Run {i}:")
            for l_name, l_id in LAYER_STEPS.items():
                feats = extract_at_layer(model, images_sorted, p, l_id)
                score = run_regression_and_rsa(feats, human_probs_sorted, rdm_human)
                layer_scores[l_name].append(score)
                id_str = f"patch" if l_id == "patch" else f"block {l_id}"
                print(f"    Layer {l_name:>8} ({id_str:<10}, dim={feats.shape[1]:>4}): RSA = {score:.4f}")

            del model
            torch.cuda.empty_cache()

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

    # Save
    out_json = os.path.join(RSA_OUT_DIR, "vit_untrained_layerwise.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=4)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RSA_OUT_DIR, "vit_untrained_layerwise.csv"), index=False)
    print(f"\n{'#'*40}\nResults saved: {RSA_OUT_DIR}\n{'#'*40}")

    # Print results
    print(f"\n{'Paradigm':<12} {'Layer':<10} {'Mean RSA':<10} {'Std'}")
    print("-" * 45)
    for row in rows:
        print(f"{row['paradigm']:<12} {row['layer']:<10} {row['mean_rsa']:<10.4f} {row['std_rsa']:.4f}")

    return df

# 7. MAIN

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
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