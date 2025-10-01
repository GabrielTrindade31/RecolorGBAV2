
from __future__ import annotations
import numpy as np
from .colorops import rgb_to_hsv_np, hsv_to_rgb_np, srgb_to_linear, linear_to_srgb
from .palette import cluster_image_labels, extract_dominant_colors, best_1to1_mapping

def transfer_palette_recolor(img_rgba: np.ndarray, ref_rgba_img: np.ndarray,
                             k: int = 4, keep_channel: str = "value",
                             saturation_scale: float = 1.0, alpha_mode: str = "preserve",
                             min_alpha: int = 8, tone_blend: float = 1.0, shade_gamma: float = 1.0) -> np.ndarray:
    H,W,_ = img_rgba.shape
    orig_centers_rgba, labels = cluster_image_labels(img_rgba, k=k, min_alpha=min_alpha)
    ref_palette = extract_dominant_colors(ref_rgba_img, k=k, min_alpha=min_alpha)
    ref_centers_rgba = np.array([p[0] for p in ref_palette], dtype=np.int32)
    if len(ref_centers_rgba) < len(orig_centers_rgba):
        while len(ref_centers_rgba) < len(orig_centers_rgba):
            ref_centers_rgba = np.vstack([ref_centers_rgba, ref_centers_rgba[0]])
    mapping = best_1to1_mapping(orig_centers_rgba, ref_centers_rgba)

    def hsv_of(rgba): 
        rgb = np.array([[rgba[0]/255.0, rgba[1]/255.0, rgba[2]/255.0]], dtype=np.float32)
        return rgb_to_hsv_np(rgb[None,:,:])[0,0]

    target_hs = []
    for i in range(len(orig_centers_rgba)):
        ref_idx = mapping[i]
        thsv = hsv_of(ref_centers_rgba[ref_idx])
        thsv[1] = np.clip(thsv[1]*float(saturation_scale),0.0,1.0)
        target_hs.append(thsv[:2])
    target_hs = np.array(target_hs, dtype=np.float32)

    rgb = img_rgba[...,:3].astype(np.float32)/255.0
    a   = img_rgba[...,3].astype(np.float32)/255.0
    hsv = rgb_to_hsv_np(rgb)

    new_h = hsv[...,0].copy(); new_s = hsv[...,1].copy()
    for i in range(len(orig_centers_rgba)):
        m = labels == i
        if not np.any(m): continue
        th,ts = target_hs[i,0], target_hs[i,1]
        dh = ((th - hsv[...,0] + 0.5)%1.0) - 0.5
        new_h[m] = (hsv[...,0][m] + dh[m]) % 1.0
        new_s[m] = ts
    new_hsv = np.stack([new_h,new_s,hsv[...,2]],axis=-1)
    provisional_rgb = hsv_to_rgb_np(new_hsv)

    lin_o = srgb_to_linear(rgb); lin_p = srgb_to_linear(provisional_rgb)
    Yo = 0.2126*lin_o[...,0] + 0.7152*lin_o[...,1] + 0.0722*lin_o[...,2]
    Yp = 0.2126*lin_p[...,0] + 0.7152*lin_p[...,1] + 0.0722*lin_p[...,2]

    # Reanchor by blend to reference average luminance per cluster
    t = float(np.clip(tone_blend, 0.0, 1.0))
    if t > 1e-6:
        scale = np.ones_like(Yp)
        for i in range(len(orig_centers_rgba)):
            m = labels == i
            if not np.any(m): continue
            ref_idx = mapping[i]
            ref_rgb = np.array([ref_centers_rgba[ref_idx][0]/255.0, ref_centers_rgba[ref_idx][1]/255.0, ref_centers_rgba[ref_idx][2]/255.0], dtype=np.float32)
            import numpy as _np
            Yt = (0.2126*_np.where(ref_rgb<=0.04045, ref_rgb/12.92, ((ref_rgb+0.055)/(1.055))**2.4)[0] +
                  0.7152*_np.where(ref_rgb<=0.04045, ref_rgb/12.92, ((ref_rgb+0.055)/(1.055))**2.4)[1] +
                  0.0722*_np.where(ref_rgb<=0.04045, ref_rgb/12.92, ((ref_rgb+0.055)/(1.055))**2.4)[2])
            Yo_m = Yo[m].mean() if np.any(m) else Yo.mean()
            shade = (Yo[m] / max(1e-8, Yo_m))**float(max(0.01, shade_gamma))
            Y_des = (1.0 - t)*Yo[m] + t*(Yt * shade)
            scale[m] = (Y_des + 1e-6)/(Yp[m] + 1e-6)
        lin_n = np.clip(lin_p * scale[...,None], 0, 1)
        new_rgb = linear_to_srgb(lin_n)
    else:
        # keep original luminance
        new_rgb = provisional_rgb

    out = np.zeros_like(img_rgba, dtype=np.uint8)
    out[...,:3] = (np.clip(new_rgb,0,1)*255.0 + 0.5).astype(np.uint8)
    out[...,3] = (a*255.0 + 0.5).astype(np.uint8)
    return out
