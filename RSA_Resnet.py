import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader

# 1. CONFIG
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Which ResNet backbone to use: "resnet18" (BasicBlock, 512-dim) or
# "resnet50" (Bottleneck, 2048-dim). Embedding dim, checkpoint path and output
# filenames all adapt automatically below.
ARCH        = "resnet18"   # <- switch to "resnet50" to run the other architecture

ARCH_CONFIG = {
    "resnet18": {"builder": models.resnet18, "embedding_dim": 512},
    "resnet50": {"builder": models.resnet50, "embedding_dim": 2048},
}
if ARCH not in ARCH_CONFIG:
    raise ValueError(f"Unknown ARCH '{ARCH}', expected one of {list(ARCH_CONFIG)}")

EMBEDDING_DIM = ARCH_CONFIG[ARCH]["embedding_dim"]

# Both architectures share the same results folder (rsa_results_resnet/) so the
# layer/entropy/accuracy plotting scripts can find both — ARCH is baked into the
# individual filenames instead (see RDM/summary filenames below).
RSA_OUT_DIR = "./out/rsa_results_resnet/"
MODEL_DIR   = "./out/Resnet/"
os.makedirs(RSA_OUT_DIR, exist_ok=True)


# 2. ARCHITECTURES
class ResNetImageEncoder(nn.Module):
    def __init__(self, arch=ARCH):
        super().__init__()
        cfg = ARCH_CONFIG[arch]
        self.embedding_dim = cfg["embedding_dim"]
        self.base = cfg["builder"](weights=None)
        self.base.fc = nn.Identity()  # embedding_dim-dim output

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


# 4. FEATURE EXTRACTION
def get_resnet_features(model, images_tensor, paradigm, batch_size=32):
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images_tensor), batch_size), desc=f"  Extracting {paradigm}"):
            batch = images_tensor[i:i + batch_size].to(DEVICE)
            # ResNet needs 224x224 input
            batch = F.interpolate(batch, size=(224, 224), mode='bilinear', align_corners=False)
            if paradigm == "supervised":
                f = model.features(batch)
            elif paradigm == "clip":
                f = model.image_encoder(batch)
                f = model.image_projection(f)
                f = F.normalize(f, p=2, dim=-1)
            elif paradigm == "vae":
                f = model(batch)
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


def run_rsa_analysis(images_sorted, human_probs_sorted):
    paradigms = ["clip", "supervised"]
    iterations = [1, 2, 3]
    rdm_human = compute_rdm_01(human_probs_sorted)
    final_stats = {}

    for p in paradigms:
        run_scores = []
        print(f"\nParadigm: {p.upper()} ({ARCH})")
        for i in iterations:
            # checkpoint filename now driven by ARCH, e.g. resnet18_supervised_run1.pt
            # or resnet50_supervised_run1.pt
            model_path = os.path.join(MODEL_DIR, f"{ARCH}_{p}_run{i}.pt")
            model = load_resnet_model(model_path, p, arch=ARCH)
            if model is None:
                continue
            feats = get_resnet_features(model, images_sorted, p)

            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            scaler = StandardScaler()
            feats_scaled = scaler.fit_transform(feats)
            pred_probs = np.zeros_like(human_probs_sorted)
            for train_idx, test_idx in kf.split(feats_scaled):
                reg = Ridge(alpha=2.5)
                reg.fit(feats_scaled[train_idx], human_probs_sorted[train_idx])
                pred_probs[test_idx] = reg.predict(feats_scaled[test_idx])
            rdm_model = compute_rdm_01(pred_probs)
            score = get_rsa_score(rdm_model, rdm_human)
            run_scores.append(score)

            plt.figure(figsize=(10, 8))
            sns.heatmap(rdm_model, cmap="viridis", vmin=0, vmax=1, xticklabels=False, yticklabels=False)
            plt.title(f"{ARCH.upper()} {p.upper()} Run {i} | RSA: {score:.4f}")
            # ARCH is now part of the RDM filename so resnet18/resnet50 runs in the
            # shared rsa_results_resnet/ folder don't overwrite each other, and this
            # still matches the "rdm_{paradigm}_" prefix the plotting scripts look for.
            rdm_filename = f"rdm_{p}_{ARCH}_run{i}.png"
            plt.savefig(os.path.join(RSA_OUT_DIR, rdm_filename), dpi=150)
            plt.close()
            print(f"  Run {i}: RSA = {score:.4f}  (saved {rdm_filename})")

        if run_scores:
            final_stats[p] = {"mean": np.mean(run_scores), "std": np.std(run_scores)}

    # save JSON — filename includes ARCH so resnet18/resnet50 summaries coexist
    summary_path = os.path.join(RSA_OUT_DIR, f"summary_{ARCH}.json")
    with open(summary_path, "w") as f:
        json.dump(final_stats, f, indent=4)
    print(f"\nSummary saved: {summary_path}")


# 7. MAIN
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Using architecture: {ARCH} (embedding dim: {EMBEDDING_DIM})")

    transform = T.Compose([
        T.Resize((224, 224), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.489, 0.4321, 0.4976], [0.22734, 0.24983, 0.2357])
    ])
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    loader = DataLoader(test_ds, batch_size=10000, shuffle=False)
    imgs, lbls = next(iter(loader))
    sort_idx = torch.argsort(lbls)
    images_sorted = imgs[sort_idx]
    human_probs = np.load("./cifar10h-probs.npy")
    human_probs_sorted = human_probs[sort_idx.numpy()]
    run_rsa_analysis(images_sorted, human_probs_sorted)