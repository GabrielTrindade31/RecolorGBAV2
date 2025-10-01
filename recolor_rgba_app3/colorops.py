
from __future__ import annotations
import numpy as np

def rgb_to_hsv_np(rgb: np.ndarray) -> np.ndarray:
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    maxc = np.max(rgb, axis=-1); minc = np.min(rgb, axis=-1)
    v = maxc; delt = maxc - minc
    s = np.where(maxc == 0, 0, delt / (maxc + 1e-8))
    h = np.zeros_like(maxc)
    m = delt != 0
    rc = np.zeros_like(r); gc = np.zeros_like(g); bc = np.zeros_like(b)
    rc[m] = ((maxc - r)[m]) / (delt[m] + 1e-8)
    gc[m] = ((maxc - g)[m]) / (delt[m] + 1e-8)
    bc[m] = ((maxc - b)[m]) / (delt[m] + 1e-8)
    rmax = (maxc == r) & m; gmax = (maxc == g) & m; bmax = (maxc == b) & m
    h[rmax] = (bc - gc)[rmax]
    h[gmax] = 2.0 + (rc - bc)[gmax]
    h[bmax] = 4.0 + (gc - rc)[bmax]
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)

def hsv_to_rgb_np(hsv: np.ndarray) -> np.ndarray:
    h,s,v = hsv[...,0], hsv[...,1], hsv[...,2]
    h6 = h*6.0; i = np.floor(h6).astype(int); f = h6 - i
    p = v*(1.0-s); q = v*(1.0-s*f); t = v*(1.0-s*(1.0-f))
    i_mod = i % 6; rgb = np.zeros(hsv.shape, dtype=hsv.dtype)
    sets = [
        (i_mod == 0, np.stack([v,t,p],-1)),
        (i_mod == 1, np.stack([q,v,p],-1)),
        (i_mod == 2, np.stack([p,v,t],-1)),
        (i_mod == 3, np.stack([p,q,v],-1)),
        (i_mod == 4, np.stack([t,p,v],-1)),
        (i_mod == 5, np.stack([v,p,q],-1)),
    ]
    for cond,val in sets: rgb[cond] = val[cond]
    return rgb

def srgb_to_linear(c):
    a=0.055; return np.where(c<=0.04045, c/12.92, ((c+a)/(1+a))**2.4)

def linear_to_srgb(c):
    a=0.055; return np.where(c<=0.0031308, 12.92*c, (1+a)*np.power(c,1/2.4)-a)

def _hue_weight_mask(h, hmin, hmax, softness):
    h = h % 1.0; hmin%=1.0; hmax%=1.0
    def inside(x,a,b):
        return (x>=a)&(x<=b) if a<=b else ((x>=a)|(x<=b))
    core = inside(h,hmin,hmax).astype(np.float32)
    if softness<=0: return core
    def ang_dist(x, y):
        d = np.abs(x-y) % 1.0; return np.minimum(d, 1.0-d)
    dmin = ang_dist(h, hmin); dmax = ang_dist(h, hmax)
    s = softness
    fade_min = (dmin < s).astype(np.float32) * (0.5 - 0.5*np.cos(np.clip(1 - dmin/s,0,1)*np.pi))
    fade_max = (dmax < s).astype(np.float32) * (0.5 - 0.5*np.cos(np.clip(1 - dmax/s,0,1)*np.pi))
    w = np.where(core>0, np.maximum(1.0,0.0), 0.0)  # placeholder, replaced next line
    w = np.where(core>0, np.maximum(fade_min, fade_max), np.maximum(fade_min, fade_max))
    # deep inside both edges -> weight=1
    w = np.where((dmin>=s)&(dmax>=s)&(core>0), 1.0, w)
    return w.astype(np.float32)

def recolor_preserve_shading(
    img_rgba: np.ndarray,
    target_rgba,
    keep_channel: str = "value",
    saturation_scale: float = 1.0,
    alpha_mode: str = "preserve",
    hue_range_deg: tuple[float,float] | None = None,
    softness_deg: float = 0.0,
    tone_blend: float = 0.0,
    shade_gamma: float = 1.0,
) -> np.ndarray:
    """
    Recolor replacing Hue/Sat by target's; preserve shading.
    - tone_blend (0..1): 0 = manter luminância original; 1 = reancorar na luminância da cor alvo
      preservando o *relativo* de luz/sombra (com expoente shade_gamma).
    - shade_gamma: ajusta o contraste do relevo (1=neutro; >1 aumenta contraste, <1 suaviza).
    """
    assert img_rgba.dtype == np.uint8 and img_rgba.shape[-1] == 4
    rgb = img_rgba[...,:3].astype(np.float32)/255.0
    a   = img_rgba[...,3].astype(np.float32)/255.0
    hsv = rgb_to_hsv_np(rgb)

    tr,tg,tb,ta = [c/255.0 for c in target_rgba]
    tgt_hsv = rgb_to_hsv_np(np.array([[[tr,tg,tb]]],np.float32))[0,0]
    th = tgt_hsv[0]
    ts = np.clip(tgt_hsv[1]*float(saturation_scale),0.0,1.0)

    w = (a>0).astype(np.float32)
    if hue_range_deg is not None:
        hmin=(hue_range_deg[0]%360.0)/360.0; hmax=(hue_range_deg[1]%360.0)/360.0
        soft=max(0.0,float(softness_deg))/360.0
        w = w * _hue_weight_mask(hsv[...,0], hmin, hmax, soft)

    # shortest hue delta
    dh = ((th - hsv[...,0] + 0.5) % 1.0) - 0.5
    new_h = (hsv[...,0] + w*dh) % 1.0
    new_s = np.clip((1.0 - w)*hsv[...,1] + w*ts, 0.0, 1.0)
    new_v = hsv[...,2]
    new_hsv = np.stack([new_h,new_s,new_v],axis=-1)
    provisional_rgb = hsv_to_rgb_np(new_hsv)

    # luminance handling
    lin_o = srgb_to_linear(rgb); lin_p = srgb_to_linear(provisional_rgb)
    Yo = 0.2126*lin_o[...,0] + 0.7152*lin_o[...,1] + 0.0722*lin_o[...,2]
    Yp = 0.2126*lin_p[...,0] + 0.7152*lin_p[...,1] + 0.0722*lin_p[...,2]

    # Desired luminance:
    t = float(np.clip(tone_blend, 0.0, 1.0))
    if t <= 1e-6 and keep_channel != "luminance":
        # keep value path (no explicit luminance preservation)
        new_rgb = provisional_rgb
    else:
        # Reanchor blend (works for both keep modes). Compute target base luminance from target color:
        target_lin = srgb_to_linear(np.array([tr,tg,tb],dtype=np.float32))
        Yt = 0.2126*target_lin[0] + 0.7152*target_lin[1] + 0.0722*target_lin[2]
        m = w>1e-6
        Yo_mean = (Yo[m].mean() if np.any(m) else Yo.mean())
        shade = np.ones_like(Yo)
        if Yo_mean > 1e-8:
            shade = (Yo / Yo_mean)**float(max(0.01, shade_gamma))
        # Blend between original luminance and target-anchored luminance
        Y_desired = (1.0 - t)*Yo + t*(Yt * shade)
        eps=1e-6; scale = np.ones_like(Yp)
        scale[m] = (Y_desired[m]+eps)/(Yp[m]+eps)
        lin_n = np.clip(lin_p * scale[...,None], 0, 1)
        new_rgb = linear_to_srgb(lin_n)

    new_a = a*ta if alpha_mode=="multiply" else a
    out = np.zeros_like(img_rgba,dtype=np.uint8)
    out[...,:3] = (np.clip(new_rgb,0,1)*255.0 + 0.5).astype(np.uint8)
    out[..., 3] = (np.clip(new_a,0,1)*255.0 + 0.5).astype(np.uint8)
    return out
