
from __future__ import annotations
import numpy as np
from PIL import Image

def load_rgba(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)

def save_rgba(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr, mode="RGBA").save(path)
