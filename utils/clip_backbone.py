# utils/clip_backbone.py
from pathlib import Path
from typing import Tuple, Optional
import clip
from torchvision import transforms
from PIL import Image
import numpy as np

_BACKBONE_ALIASES = {
    # ViTs
    "vit-b/32": "ViT-B/32",
    "vit-b/16": "ViT-B/16",
    "vit-l/14": "ViT-L/14",
    "vit-l/14@336px": "ViT-L/14@336px",
    # ResNets
    "rn50": "RN50",
    "rn101": "RN101",
    "rn50x4": "RN50x4",
    "rn50x16": "RN50x16",
    "rn50x64": "RN50x64",
}

def _canonical(name: str) -> str:
    key = name.strip().lower()
    return _BACKBONE_ALIASES.get(key, name)

def load_clip_model(
    backbone: str,
    device: str = "cuda",
    download_root: Optional[str] = None,
    jit: bool = False,
):
    """Load a CLIP model & preprocess with a chosen backbone."""
    model_name = _canonical(backbone)
    if download_root is not None:
        download_root = str(Path(download_root).expanduser())
    model, preprocess = clip.load(model_name, device=device, jit=jit, download_root=download_root)
    return model, preprocess

def to_pil_first(preprocess):
    """Wrap CLIP preprocess to accept numpy arrays too (keeps existing behavior)."""
    return transforms.Compose([
        transforms.Lambda(lambda img: Image.fromarray(img) if isinstance(img, np.ndarray) else img),
        preprocess
    ])