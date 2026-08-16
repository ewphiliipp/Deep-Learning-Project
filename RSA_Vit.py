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
            batch = F.interpolate(batch, size=(224, 224), mode='bilinear', align_corners=False)

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

def compute_rdm_01(data):
    corr = np.corrcoef(data)
    rdm  = (1 - corr) / 2
    return np.nan_to_num(rdm, nan=0.5)

def get_rsa_score(m1, m2):
    iu = np.triu_indices_from(m1, k=1)
    return float(np.corrcoef(m1[iu], m2[iu])[0, 1])


def run_rsa_analysis(images_sorted, human_probs_sorted):
    paradigms  = ["supervised", "clip"]
    iterations = [1, 2, 3]
    rdm_human = compute_rdm_01(human_probs_sorted)
    final_stats = {}

    for p in paradigms:
        run_scores = []
        print(f"\nParadigm: {p.upper()}")
        for i in iterations:
            model_path = os.path.join(MODEL_DIR, f"vit_untrained_{p}_run{i}.pt")
            model = load_vit_model(model_path, p)
            if model is None: continue

            feats = get_vit_features(model, images_sorted, p)
            
            # Regression (Stabilisiertes Alpha=1.0 für hochdimensionale ViT-Features)
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            scaler = StandardScaler()
            feats_scaled = scaler.fit_transform(feats)
            pred_probs = np.zeros_like(human_probs_sorted)

            for train_idx, test_idx in kf.split(feats_scaled):
                reg = Ridge(alpha=1.0) # Etwas mehr Regularisierung für 768 Dimensionen
                reg.fit(feats_scaled[train_idx], human_probs_sorted[train_idx])
                pred_probs[test_idx] = reg.predict(feats_scaled[test_idx])

            rdm_model = compute_rdm_01(pred_probs)
            score = get_rsa_score(rdm_model, rdm_human)
            run_scores.append(score)

            # Plotting
            plt.figure(figsize=(10, 8))
            sns.heatmap(rdm_model, cmap="viridis", vmin=0, vmax=1, xticklabels=False, yticklabels=False)
            plt.title(f"ViT {p.upper()} Run {i} | RSA: {score:.4f}")
            plt.savefig(os.path.join(RSA_OUT_DIR, f"rdm_{p}_run{i}.png"), dpi=150)
            plt.close()
            print(f"  Run {i}: RSA = {score:.4f}")
            del model; torch.cuda.empty_cache()

        if run_scores:
            final_stats[p] = {"mean": round(np.mean(run_scores), 4), "std": round(np.std(run_scores), 4)}

    with open(os.path.join(RSA_OUT_DIR, "summary_vit_untrained.json"), "w") as f:
        json.dump(final_stats, f, indent=4)

#6. MAIN

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

    sort_idx           = torch.argsort(lbls)
    images_sorted      = imgs[sort_idx]
    human_probs        = np.load("./cifar10h-probs.npy")
    human_probs_sorted = human_probs[sort_idx.numpy()]

    print(f"Images: {images_sorted.shape} | Human probs: {human_probs_sorted.shape}")
    run_rsa_analysis(images_sorted, human_probs_sorted)