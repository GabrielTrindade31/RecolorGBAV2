from __future__ import annotations
from PIL import Image
import numpy as np
def pil_open(path): return Image.open(path).convert("RGBA")
def pil_to_np(im): return np.array(im, dtype=np.uint8)
def np_to_pil(arr): return Image.fromarray(arr, mode="RGBA")
