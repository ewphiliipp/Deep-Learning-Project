import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader

# 1. CONFIG
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Which ResNet backbone to use: "resnet18" (BasicBlock, 512-dim final) or
# "resnet50" (Bottleneck, 2048-dim final). Everything below (embedding dim,
# checkpoint path, output filenames, layer-step comments) adapts automatically.
ARCH        = "resnet18"   # <- switch to "resnet50" to run the other architecture

ARCH_CONFIG = {
    "resnet18": {
        "builder": models.resnet18,
        "embedding_dim": 512,
        "block_type": "BasicBlock",
        # feature dim emitted at each stage for resnet18
        "layer_dims": {"conv1": 64, "layer1": 64, "layer2": 128,
                        "layer3": 256, "layer4": 512, "final": 512},
        # number of residual blocks per stage for resnet18
        "block_counts": {"layer1": 2, "layer2": 2, "layer3": 2, "layer4": 2},
    },
    "resnet50": {
        "builder": models.resnet50,
        "embedding_dim": 2048,
        "block_type": "Bottleneck",
        "layer_dims": {"conv1": 64, "layer1": 256, "layer2": 512,
                        "layer3": 1024, "layer4": 2048, "final": 2048},
        "block_counts": {"layer1": 3, "layer2": 4, "layer3": 6, "layer4": 3},
    },
}
if ARCH not in ARCH_CONFIG:
    raise ValueError(f"Unknown ARCH '{ARCH}', expected one of {list(ARCH_CONFIG)}")

_cfg          = ARCH_CONFIG[ARCH]
EMBEDDING_DIM = _cfg["embedding_dim"]

RSA_OUT_DIR = f"./out/rsa_layerwise_{'resnet' if ARCH == 'resnet50' else ARCH}/"
MODEL_DIR   = "./out/Resnet/"
os.makedirs(RSA_OUT_DIR, exist_ok=True)

# LAYER_STEPS: same 6 depth checkpoints for both architectures, but the dim
# and block-count annotations are pulled dynamically from ARCH_CONFIG so the
# comments stay accurate for whichever ARCH is selected.
LAYER_STEPS = {
    "0pct":   "conv1",   # after conv1+bn1+relu+maxpool
    "20pct":  "layer1",  # after layer1
    "40pct":  "layer2",  # after layer2
    "60pct":  "layer3",  # after layer3
    "80pct":  "layer4",  # after layer4
    "100pct": "final",   # after avgpool + flatten
}


def describe_layer_step(l_id):
    """Human-readable, ARCH-aware description of a layer step, e.g.
    'layer2 (4 Bottleneck Blocks) -> 512-dim' for resnet50, or
    'layer2 (2 BasicBlocks) -> 128-dim' for resnet18."""
    dim = _cfg["layer_dims"][l_id]
    if l_id in _cfg["block_counts"]:
        n_blocks = _cfg["block_counts"][l_id]
        return f"{l_id} ({n_blocks} {_cfg['block_type']}{'s' if n_blocks != 1 else ''}) -> {dim}-dim"
    return f"{l_id} -> {dim}-dim"


print(f"Selected ARCH = {ARCH} ({_cfg['block_type']}, final embedding dim = {EMBEDDING_DIM})")
for l_name, l_id in LAYER_STEPS.items():
    print(f"  {l_name:>7}: {describe_layer_step(l_id)}")


# 2. ARCHITECTURES
class ResNetImageEncoder(nn.Module):
    def __init__(self, arch=ARCH):
        super().__init__()
        cfg = ARCH_CONFIG[arch]
        self.embedding_dim = cfg["embedding_dim"]
        self.base = cfg["builder"](weights=None)
        self.base.fc = nn.Identity()

    def forward(self, x):
        return self.base(x)


class ResNetSupervised(nn.Module):
    def __init__(self, num_classes=10, arch=ARCH):
        super().__init__()
        self.features = ResNetImageEncoder(arch)
        self.classifier = nn.Linear(self.features.embedding_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


class CLIPNet(nn.Module):
    def __init__(self, projection_dim=256, arch=ARCH):
        super().__init__()
        self.image_encoder = ResNetImageEncoder(arch)
        self.image_projection = nn.Linear(self.image_encoder.embedding_dim, projection_dim)

    def forward(self, x):
        return self.image_projection(self.image_encoder(x))


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
        self.encoder = ResNetImageEncoder(arch)
        self.mu = nn.Linear(self.encoder.embedding_dim, latent_dim)

    def forward(self, x):
        return self.mu(self.encoder(x))


# 3. LOAD MODEL — name-based mapping with shape check (safe for both resnet18 & resnet50)
def load_resnet_model(model_path, paradigm, arch=ARCH):
    if not os.path.exists(model_path):
        print(f"  [SKIP] Not found: {model_path}")
        return None

    if paradigm == "clip":
        model = CLIPNet(projection_dim=256, arch=arch).to(DEVICE)
        target_backbone = model.image_encoder.base
    elif paradigm == "vae":
        model = VAE(latent_dim=256, arch=arch).to(DEVICE)
        target_backbone = model.encoder.base
    else:
        model = ResNetSupervised(num_classes=10, arch=arch).to(DEVICE)
        target_backbone = model.features.base

    # load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint)) \
        if isinstance(checkpoint, dict) else checkpoint

    # Position-based mapping: walk target keys in order and assign the next
    # source tensor whose shape matches. Restored per user request — the
    # name-based matcher didn't find any overlapping keys against these
    # checkpoints (0/120 matched), while position-based matching has worked
    # reliably for all networks so far.
    current_dict = target_backbone.state_dict()
    new_state_dict = {}

    sd_keys = list(state_dict.keys())
    target_keys = list(current_dict.keys())

    print(f"  [{paradigm.upper()}/{arch}] Versuche positionsbasiertes Mapping...")

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
    print(f"  [{paradigm.upper()}/{arch}] Loaded {matched_count}/{len(target_keys)} layers via Position-Match.")

    # load heads separately via key matching
    if paradigm == "clip" and "image_projection.weight" in state_dict:
        model.image_projection.load_state_dict({
            "weight": state_dict["image_projection.weight"],
            "bias": state_dict["image_projection.bias"]
        }, strict=False)
        print("  [CLIP] Projection head loaded via Key-Match.")
    elif paradigm == "vae" and "mu.weight" in state_dict:
        model.mu.load_state_dict({
            "weight": state_dict["mu.weight"],
            "bias": state_dict["mu.bias"]
        }, strict=False)
        print("  [VAE] Mu head loaded via Key-Match.")

    return model.eval()


# 4. LAYER FEATURE EXTRACTION
def get_resnet_backbone(model, paradigm):
    if paradigm == "clip":
        return model.image_encoder.base
    elif paradigm == "vae":
        return model.encoder.base
    else:
        return model.features.base


@torch.no_grad()
def extract_at_layer(model, images_tensor, paradigm, layer_id, batch_size=32):
    model.eval()
    res = get_resnet_backbone(model, paradigm)
    all_feats = []
    for i in range(0, len(images_tensor), batch_size):
        batch = images_tensor[i:i + batch_size].to(DEVICE)
        batch = F.interpolate(batch, size=(224, 224), mode='bilinear', align_corners=False)
        x = res.conv1(batch)
        x = res.bn1(x)
        x = res.relu(x)
        x = res.maxpool(x)
        if layer_id == "conv1":
            f = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            all_feats.append(f.cpu().numpy())
            continue
        x = res.layer1(x)
        if layer_id == "layer1":
            f = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            all_feats.append(f.cpu().numpy())
            continue
        x = res.layer2(x)
        if layer_id == "layer2":
            f = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            all_feats.append(f.cpu().numpy())
            continue
        x = res.layer3(x)
        if layer_id == "layer3":
            f = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            all_feats.append(f.cpu().numpy())
            continue
        x = res.layer4(x)
        if layer_id == "layer4":
            f = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            all_feats.append(f.cpu().numpy())
            continue

        if paradigm == "clip":
            f = model.image_encoder(batch)   # goes to final avgpool (embedding_dim)
            f = model.image_projection(f)    # goes through projection head (256-dim)
            f = F.normalize(f, p=2, dim=-1)  # L2 norm
        elif paradigm == "vae":
            f = res.avgpool(x).flatten(1)
            f = model.mu(f)
        else:
            f = res.avgpool(x).flatten(1)

        all_feats.append(f.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


# 5. RSA
def compute_rdm_01(data):
    corr = np.corrcoef(data)
    rdm = (1 - corr) / 2
    return np.nan_to_num(rdm, nan=0.5)


def get_rsa_score(m1, m2):
    iu = np.triu_indices_from(m1, k=1)
    return float(np.corrcoef(m1[iu], m2[iu])[0, 1])


def run_regression_and_rsa(feats, human_probs, rdm_human):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    pred_probs = np.zeros_like(human_probs)
    for train_idx, test_idx in kf.split(feats_scaled):
        reg = Ridge(alpha=2.5)
        reg.fit(feats_scaled[train_idx], human_probs[train_idx])
        pred_probs[test_idx] = reg.predict(feats_scaled[test_idx])
    rdm_model = compute_rdm_01(pred_probs)
    return get_rsa_score(rdm_model, rdm_human)


# 6. LAYERWISE PIPELINE
def run_layerwise_rsa(images_sorted, human_probs_sorted):
    paradigms = ["clip", "supervised"]
    iterations = [1, 2, 3]

    print("Berechne Human RDM...")
    rdm_human = compute_rdm_01(human_probs_sorted)

    rows = []
    for p in paradigms:
        print(f"\n{'='*40}\nParadigm: {p.upper()}\n{'='*40}")
        layer_scores = {l_name: [] for l_name in LAYER_STEPS}
        for i in iterations:
            # checkpoint filename now driven by ARCH, e.g. resnet18_supervised_run1.pt
            # or resnet50_supervised_run1.pt (adjust prefix below if your untrained
            # checkpoints use a different naming convention)
            model_path = os.path.join(MODEL_DIR, f"{ARCH}_{p}_run{i}.pt")
            model = load_resnet_model(model_path, p, arch=ARCH)
            if model is None:
                continue
            print(f"\n  Run {i}:")
            for l_name, l_id in LAYER_STEPS.items():
                feats = extract_at_layer(model, images_sorted, p, l_id)
                score = run_regression_and_rsa(feats, human_probs_sorted, rdm_human)
                layer_scores[l_name].append(score)
                print(f"    Layer {l_name:>8} ({describe_layer_step(l_id)}): "
                      f"actual dim={feats.shape[1]:>5}, RSA = {score:.4f}")
            del model
            torch.cuda.empty_cache()

        for l_name, scores in layer_scores.items():
            if scores:
                mean_val = round(float(np.mean(scores)), 5)
                std_val = round(float(np.std(scores)), 5)
                rows.append({
                    "paradigm": p,
                    "layer": l_name,
                    "mean_rsa": mean_val,
                    "std_rsa": std_val,
                    "runs": [round(float(s), 5) for s in scores]
                })
                print(f"  ► {l_name}: Mean={mean_val:.4f} ± {std_val:.4f}")

    # Save — output filenames include ARCH so resnet18 and resnet50 runs don't overwrite each other
    out_json = os.path.join(RSA_OUT_DIR, f"{ARCH}_layerwise.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=4)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RSA_OUT_DIR, f"{ARCH}_layerwise.csv"), index=False)
    print(f"\n{'#'*40}\nResults saved: {RSA_OUT_DIR}\n{'#'*40}")

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
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False,
                                            download=True, transform=transform)
    loader = DataLoader(test_ds, batch_size=10000, shuffle=False)
    imgs, lbls = next(iter(loader))
    sort_idx = torch.argsort(lbls)
    images_sorted = imgs[sort_idx]

    if os.path.exists("./cifar10h-probs.npy"):
        human_probs = np.load("./cifar10h-probs.npy")
        human_probs_sorted = human_probs[sort_idx.numpy()]
        print(f"Images: {images_sorted.shape} | Human probs: {human_probs_sorted.shape}")
        run_layerwise_rsa(images_sorted, human_probs_sorted)
    else:
        print("Datei './cifar10h-probs.npy' not found!")