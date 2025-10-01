
from __future__ import annotations
import numpy as np
from typing import List, Tuple
from .colorops import rgb_to_hsv_np

RGBA = Tuple[int,int,int,int]

def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]; centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    idx0 = rng.integers(0, n); centers[0] = X[idx0]
    closest = np.sum((X - centers[0])**2, axis=1)
    for i in range(1, k):
        probs = closest / (closest.sum() + 1e-12)
        idx = rng.choice(n, p=probs); centers[i] = X[idx]
        dist = np.sum((X - centers[i])**2, axis=1); closest = np.minimum(closest, dist)
    return centers

def _kmeans(X: np.ndarray, k: int, max_iter: int = 30, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = _kmeans_pp_init(X, k, rng)
    labels = np.zeros(X.shape[0], dtype=np.int32)
    for _ in range(max_iter):
        d2 = np.sum((X[:,None,:] - centers[None,:,:])**2, axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels): break
        labels = new_labels
        for i in range(k):
            m = labels == i
            centers[i] = X[m].mean(axis=0) if np.any(m) else X[rng.integers(0, X.shape[0])]
    return centers, labels

def extract_dominant_colors(img_rgba: np.ndarray, k: int = 4, min_alpha: int = 8):
    assert img_rgba.dtype == np.uint8 and img_rgba.shape[-1] == 4
    px = img_rgba.reshape(-1,4)
    mask = px[:,3] >= min_alpha
    if not np.any(mask): mask = np.ones(px.shape[0], dtype=bool)
    rgb = px[mask,:3].astype(np.float32); a = px[mask,3:4].astype(np.float32)/255.0
    N = rgb.shape[0]
    if N > 300_000:
        idx = np.random.default_rng(123).choice(N, size=300_000, replace=False)
        rgb = rgb[idx]; a=a[idx]
    X = np.concatenate([rgb*a, a*255.0], axis=1)
    k = max(1, min(k, X.shape[0]))
    centers, labels = _kmeans(X, k=k, max_iter=40, seed=123)
    out=[]; total = X.shape[0]
    for i in range(k):
        m = labels==i
        if not np.any(m): continue
        rgb_mean = rgb[m].mean(0); a_mean = a[m].mean()*255.0
        rgba = (int(round(rgb_mean[0])), int(round(rgb_mean[1])), int(round(rgb_mean[2])), int(round(a_mean)))
        out.append((rgba, float(np.count_nonzero(m))/float(total)))
    out.sort(key=lambda d:d[1], reverse=True)
    return out

def suggest_tint_from_palette(doms: List[Tuple[RGBA, float]]):
    if not doms: return None
    rgbs = np.array([[d[0][0], d[0][1], d[0][2]] for d in doms], dtype=np.float32)/255.0
    hsv = rgb_to_hsv_np(rgbs[None,:,:])[0]
    return doms[int(np.argmax(hsv[:,1]))][0]

def cluster_image_labels(img_rgba: np.ndarray, k: int = 4, min_alpha: int = 8):
    H,W,_ = img_rgba.shape
    px = img_rgba.reshape(-1,4)
    mask = px[:,3] >= min_alpha
    if not np.any(mask): mask = np.ones(px.shape[0], dtype=bool)
    rgb = px[mask,:3].astype(np.float32); a = px[mask,3:4].astype(np.float32)/255.0
    X = np.concatenate([rgb*a, a*255.0], axis=1)
    k = max(1, min(k, X.shape[0]))
    centers, labels_small = _kmeans(X, k=k, max_iter=40, seed=123)
    labels = np.full(px.shape[0], -1, dtype=np.int32); labels[mask] = labels_small
    centers_rgba=[]
    for i in range(k):
        m=labels_small==i
        if np.any(m):
            rgb_mean=rgb[m].mean(0); a_mean=a[m].mean()*255.0
            centers_rgba.append((int(round(rgb_mean[0])), int(round(rgb_mean[1])), int(round(rgb_mean[2])), int(round(a_mean))))
        else:
            centers_rgba.append((0,0,0,255))
    return np.array(centers_rgba, dtype=np.int32), labels.reshape(H,W)

def _hsv_dist(a, b):
    dh = np.abs(a[...,0] - b[...,0]) % 1.0; dh = np.minimum(dh, 1.0 - dh)
    ds = np.abs(a[...,1] - b[...,1]); dv = np.abs(a[...,2] - b[...,2])
    return np.sqrt((2.0*dh)**2 + ds**2 + dv**2)

def best_1to1_mapping(orig_rgba: np.ndarray, ref_rgba: np.ndarray):
    k = min(len(orig_rgba), len(ref_rgba))
    def rgba_to_hsv(rgba_arr):
        rgb = np.array([[c[0]/255.0, c[1]/255.0, c[2]/255.0] for c in rgba_arr], dtype=np.float32)
        return rgb_to_hsv_np(rgb[None,:,:])[0]
    H1 = rgba_to_hsv(orig_rgba[:k]); H2 = rgba_to_hsv(ref_rgba[:k])
    import itertools
    best=None; best_cost=1e9
    for perm in itertools.permutations(range(k)):
        cost = _hsv_dist(H1, H2[list(perm)]).sum()
        if cost < best_cost: best_cost=cost; best=list(perm)
    return best
