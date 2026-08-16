"""
Downloads pretrained CLIP vision encoders (RN50 and ViT-B/16, OpenAI weights)
via OpenCLIP and saves just the vision tower's state_dict as a .pt checkpoint.

Requires: pip install open_clip_torch --break-system-packages

Output:
    ./out/openclip/resnet50_clip_pretrained.pt
    ./out/openclip/vit_b16_clip_pretrained.pt

Each .pt file is a dict:
    {
        "model_state_dict": <state_dict of the vision encoder only>,
        "arch": "resnet50" | "vit_b16",
        "clip_model_name": "RN50" | "ViT-B-16",
        "pretrained": "openai",
        "embedding_dim": 1024 | 512,   # CLIP's projected visual embedding dim
    }

This "model_state_dict" key matches what the existing RSA scripts
(resnet_linear_probe.py, resnet_layerwise_rsa.py, resnet_rsa_heatmaps.py,
resnet_entropy_rsa.py) already expect from `checkpoint.get('model_state_dict', ...)`,
so the same position-based loading logic can pick these up too — though note the
OpenCLIP vision towers are NOT plain torchvision resnet50/vit_b_16 modules (see
notes below), so a dedicated loader is safer than reusing the ResNetImageEncoder
position-matcher as-is.
"""

import os
import torch
import open_clip

OUT_DIR = "./out/openclip/"
os.makedirs(OUT_DIR, exist_ok=True)

PRETRAINED = "openai"

# OpenCLIP model identifiers for the two architectures we want.
# NOTE: the "openai" pretrained weights were trained with QuickGELU activation,
# but the plain "RN50"/"ViT-B-16" configs in OpenCLIP default to regular GELU.
# Using the "-quickgelu" suffixed configs avoids the activation-function mismatch
# warning and ensures the loaded weights are used with the architecture they were
# actually trained with.
MODELS_TO_DOWNLOAD = [
    {"clip_model_name": "RN50-quickgelu",     "arch": "resnet50", "out_file": "resnet50_clip_pretrained.pt"},
    {"clip_model_name": "ViT-B-16-quickgelu", "arch": "vit_b16",  "out_file": "vit_b16_clip_pretrained.pt"},
]


def download_and_save(clip_model_name, arch, out_file, pretrained=PRETRAINED):
    print(f"\n{'='*60}\nLoading {clip_model_name} (pretrained='{pretrained}')\n{'='*60}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=pretrained
    )
    model.eval()

    # The vision tower lives at model.visual for every OpenCLIP architecture
    # (both the ResNet-based and ViT-based CLIP variants).
    visual_encoder = model.visual
    vision_state_dict = visual_encoder.state_dict()

    # CLIP's final projected visual embedding dim (post visual.proj / attnpool),
    # i.e. the dimensionality used to compare against the text embeddings.
    # model.visual.image_size can be an int (ResNet variants) or a tuple/list (ViT
    # variants), so normalize it instead of assuming it's subscriptable.
    img_size = getattr(model.visual, "image_size", 224)
    if isinstance(img_size, (tuple, list)):
        h, w = img_size[0], img_size[1]
    else:
        h = w = img_size

    with torch.no_grad():
        dummy = torch.zeros(1, 3, h, w)
        try:
            dummy_out = visual_encoder(dummy)
            embedding_dim = int(dummy_out.shape[-1])
        except Exception as e:
            print(f"  [WARN] Could not infer embedding_dim via forward pass: {e}")
            embedding_dim = None

    checkpoint = {
        "model_state_dict": vision_state_dict,
        "arch": arch,
        "clip_model_name": clip_model_name,
        "pretrained": pretrained,
        "embedding_dim": embedding_dim,
    }

    out_path = os.path.join(OUT_DIR, out_file)
    torch.save(checkpoint, out_path)

    n_params = sum(p.numel() for p in visual_encoder.parameters())
    print(f"  Vision encoder: {type(visual_encoder).__name__}")
    print(f"  Params: {n_params:,}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Saved: {out_path}")

    del model
    return out_path


if __name__ == "__main__":
    print("Available OpenCLIP pretrained tags (sanity check):")
    for cfg in MODELS_TO_DOWNLOAD:
        name = cfg["clip_model_name"]
        available = open_clip.list_pretrained_tags_by_model(name)
        status = "OK" if PRETRAINED in available else "NOT FOUND"
        print(f"  {name:<20} pretrained='{PRETRAINED}' -> {status} (available: {available})")

    saved_paths = []
    for cfg in MODELS_TO_DOWNLOAD:
        path = download_and_save(**cfg)
        saved_paths.append(path)

    print(f"\n{'#'*60}\nDone. Saved {len(saved_paths)} checkpoints:")
    for p in saved_paths:
        print(f"  {p}")
    print(f"{'#'*60}")