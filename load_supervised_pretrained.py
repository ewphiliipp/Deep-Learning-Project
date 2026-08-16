"""
Downloads torchvision ImageNet-supervised-pretrained backbones (ResNet50 and
ViT-B/16) — no additional training on any downstream dataset — and saves their
state_dicts as .pt checkpoints, in the same style as download_openclip_backbones.py.

These are the "classic" supervised baselines to compare against the CLIP
(vision-language, WIT dataset) backbones downloaded in download_openclip_backbones.py:
both families see ImageNet-1k images, but ResNet50/ViT-B here are trained with
plain 1000-way classification labels, not natural-language image captions.

Output:
    ./out/imagenet_pretrained/resnet50_supervised_pretrained.pt
    ./out/imagenet_pretrained/vit_b16_supervised_pretrained.pt

Each .pt file is a dict:
    {
        "model_state_dict": <full model state_dict, incl. classification head>,
        "arch": "resnet50" | "vit_b16",
        "weights_tag": "IMAGENET1K_V2" | "IMAGENET1K_V1",
        "embedding_dim": 2048 | 768,   # feature dim before the classification head
        "num_classes": 1000,
    }

The "model_state_dict" key matches what the existing RSA scripts
(resnet_linear_probe.py, resnet_layerwise_rsa.py, resnet_rsa_heatmaps.py,
resnet_entropy_rsa.py) already expect from checkpoint.get('model_state_dict', ...),
so the position-based loading logic there can pick these up directly for ResNet50
(same torchvision.models.resnet50 architecture). The ViT-B/16 checkpoint uses
torchvision's native ViT implementation, which is architecturally different from
however your existing ViT pipeline loads its backbone — check key names before
reusing the position-matcher for it.
"""

import os
import torch
import torchvision.models as models

OUT_DIR = "./out/imagenet_pretrained/"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS_TO_DOWNLOAD = [
    {
        "arch": "resnet50",
        "builder": models.resnet50,
        "weights_enum": models.ResNet50_Weights,
        "weights_tag": "IMAGENET1K_V1",
        "out_file": "resnet50_supervised_pretrained.pt",
        "embedding_dim": 2048,
    },
    {
        "arch": "vit_b16",
        "builder": models.vit_b_16,
        "weights_enum": models.ViT_B_16_Weights,
        "weights_tag": "IMAGENET1K_V1",
        "out_file": "vit_b16_supervised_pretrained.pt",
        "embedding_dim": 768,
    },
]


def download_and_save(arch, builder, weights_enum, weights_tag, out_file, embedding_dim):
    print(f"\n{'='*60}\nLoading {arch} (weights='{weights_tag}')\n{'='*60}")

    weights = getattr(weights_enum, weights_tag)
    model = builder(weights=weights)
    model.eval()

    state_dict = model.state_dict()

    checkpoint = {
        "model_state_dict": state_dict,
        "arch": arch,
        "weights_tag": weights_tag,
        "embedding_dim": embedding_dim,
        "num_classes": 1000,
    }

    out_path = os.path.join(OUT_DIR, out_file)
    torch.save(checkpoint, out_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {type(model).__name__}")
    print(f"  Params: {n_params:,}")
    print(f"  Embedding dim (pre-head): {embedding_dim}")
    print(f"  Weights: {weights_tag} (meta: {weights.meta.get('_metrics', {})})")
    print(f"  Saved: {out_path}")

    del model
    return out_path


if __name__ == "__main__":
    saved_paths = []
    for cfg in MODELS_TO_DOWNLOAD:
        path = download_and_save(**cfg)
        saved_paths.append(path)

    print(f"\n{'#'*60}\nDone. Saved {len(saved_paths)} checkpoints:")
    for p in saved_paths:
        print(f"  {p}")
    print(f"{'#'*60}")