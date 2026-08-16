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
RSA_OUT_DIR = "./out/rsa_results_vit/"
MODEL_DIR   = "./out/VIT/"
os.makedirs(RSA_OUT_DIR, exist_ok=True)

# 2. ARCHITECTURES

class ViTImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # ViT-B/16 Baseline
        self.model = models.vit_b_16(weights=None)
        self.model.heads = nn.Identity() # 768-dim CLS Output

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

    # initilalise Modell-Architecture
    if paradigm == "clip":
        model = CLIPNet(projection_dim=256).to(DEVICE)
        target_backbone = model.image_encoder.model
    elif paradigm == "vae":
        model = VAE(latent_dim=256).to(DEVICE)
        target_backbone = model.encoder.model
    else:
        model = ViTSupervised(num_classes=10).to(DEVICE)
        target_backbone = model.features.model

    # load Checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint)) \
        if isinstance(checkpoint, dict) else checkpoint

    # positionbased mapping
    current_dict = target_backbone.state_dict()
    new_state_dict = {}
    
    # List of key and target 
    sd_keys = list(state_dict.keys())
    target_keys = list(current_dict.keys())
    
    print(f"  [{paradigm.upper()}] Starte positionsbasiertes Mapping für ViT...")
    
    matched_count = 0
    sd_idx = 0
    
    # iterate through target layer of vit
    for t_key in target_keys:
        target_shape = current_dict[t_key].shape
        
        # search for last found checkpoint and map matching layer
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
    
    # load weights
    target_backbone.load_state_dict(new_state_dict, strict=False)
    print(f"  [{paradigm.upper()}] Loaded {matched_count}/{len(target_keys)} layers via Position-Match.")

    # load heads seperately
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
        print("  [VAE] Mu-layer loaded via Key-Match.")

    return model.eval()

# 4. FEATURE EXTRACTION

def get_vit_features(model, images_tensor, paradigm, batch_size=32):
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images_tensor), batch_size), desc=f"  Extracting {paradigm}"):
            batch = images_tensor[i:i+batch_size].to(DEVICE)
            batch = F.interpolate(batch, size=(224, 224), mode='bicubic', align_corners=False)

            if paradigm == "supervised":
                f = model.features(batch) # 768-dim
            elif paradigm == "clip":
                f = model.image_encoder(batch)
                f = model.image_projection(f)
                f = F.normalize(f, p=2, dim=-1) # L2-Norm for Cosine Space
            elif paradigm == "vae":
                f = model(batch) # mu
            
            all_feats.append(f.cpu().numpy())
    return np.concatenate(all_feats, axis=0)
 
# 5. RSA

def compute_entropy(probs):
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)
 
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
 
def run_entropy_rsa(imgs, lbls, all_human_probs):
    paradigms  = ["supervised", "clip"]
    iterations = [1, 2, 3]
 
    entropy = compute_entropy(all_human_probs)
    print(f"Entropy range: {entropy.min():.4f} – {entropy.max():.4f}")
 
    # 4 groups of same size
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
                model_path = os.path.join(MODEL_DIR, f"vit_untrained_{p}_run{i}.pt")
                model      = load_vit_model(model_path, p)
                if model is None:
                    continue
 
                feats = get_vit_features(model, images_bin, p)
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
    out_json = os.path.join(RSA_OUT_DIR, "rsa_entropy_vit_untrained.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n{'#'*40}\nResults saved: {out_json}\n{'#'*40}")
 
    # print results
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
        T.Resize((224, 224), T.InterpolationMode.BICUBIC), 
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