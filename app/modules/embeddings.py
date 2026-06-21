"""
embeddings.py
-------------
Wraps Meta's DINOv2 pretrained backbone (loaded via torch.hub, exactly as the
DINOv2 repo recommends: https://github.com/facebookresearch/dinov2) to turn an
image into a single feature vector ("embedding").

Used for cosine-similarity scoring between a generated variant and the
master image, so we can filter out variants that visually "drifted" too far.
"""

from __future__ import annotations

import threading
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

DEFAULT_MODEL_NAME = "dinov2_vits14"  # smallest DINOv2 variant: fast, CPU-friendly

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_RESIZE_SIZE = 224  # 224 / 14 = 16 patches per side, standard DINOv2 input


class EmbeddingExtractor:
    """Loads a DINOv2 backbone once and reuses it for all embedding calls."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.device = _DEVICE
        self.model = self._load_model(model_name)
        self.transform = transforms.Compose(
            [
                transforms.Resize((_RESIZE_SIZE, _RESIZE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    def _load_model(self, model_name: str):
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        model.eval()
        model.to(self.device)
        return model

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Returns a 1D L2-normalized embedding vector for a PIL image."""
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        features = self.model(tensor)
        vec = features.squeeze(0).cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @torch.no_grad()
    def embed_bytes(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(BytesIO(image_bytes))
        return self.embed_image(image)


_extractor_lock = threading.Lock()
_extractor: EmbeddingExtractor | None = None


def get_extractor() -> EmbeddingExtractor:
    """Lazily creates (and caches) the global EmbeddingExtractor."""
    global _extractor
    if _extractor is None:
        with _extractor_lock:
            if _extractor is None:
                _extractor = EmbeddingExtractor()
    return _extractor


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = vec_a / (np.linalg.norm(vec_a) + 1e-12)
    b = vec_b / (np.linalg.norm(vec_b) + 1e-12)
    return float(np.dot(a, b))