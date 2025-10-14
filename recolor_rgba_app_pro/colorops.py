from __future__ import annotations
import numpy as np


def color_threshold_mask(h, s, v, base_rgb, tol_h_deg, tol_s, tol_v, base_hsv=None):
    if base_hsv is None:
        base = np.array(base_rgb, dtype=np.float32) / 255.0
        bh, bs, bv = rgb_to_hsv(np.array([base[0]]), np.array([base[1]]), np.array([base[2]]))
        bh = bh[0]
        bs = float(bs[0])
        bv = float(bv[0])
    else:
        bh, bs, bv = base_hsv
    tol_h = float(tol_h_deg) / 360.0
    tol_s = float(tol_s)
    tol_v = float(tol_v)
    dh = np.abs(h - bh)
    dh = np.minimum(dh, 1.0 - dh)
    mask_h = dh <= tol_h
    mask_s = np.abs(s - bs) <= tol_s
    mask_v = np.abs(v - bv) <= tol_v
    mask = mask_h & mask_s & mask_v
    return mask.astype(np.float32)
def rgb_to_hsv(r,g,b):
    mx=np.maximum.reduce([r,g,b]); mn=np.minimum.reduce([r,g,b])
    v=mx; d=mx-mn+1e-8; s=d/(mx+1e-8); h=np.zeros_like(mx)
    m=(d>0); i=(mx==r)&m; h[i]=((g[i]-b[i])/d[i])%6
    i=(mx==g)&m; h[i]=((b[i]-r[i])/d[i])+2
    i=(mx==b)&m; h[i]=((r[i]-g[i])/d[i])+4
    h=(h/6.0)%1.0; return h,s,v
def hsv_to_rgb(h,s,v):
    i=np.floor(h*6).astype(np.int32); f=h*6-i; p=v*(1-s); q=v*(1-s*f); t=v*(1-s*(1-f)); i%=6
    r=np.select([i==0,i==1,i==2,i==3,i==4],[v,q,p,p,t], default=v)
    g=np.select([i==0,i==1,i==2,i==3,i==4],[t,v,v,q,p], default=p)
    b=np.select([i==0,i==1,i==2,i==3,i==4],[p,p,t,v,v], default=q); return r,g,b
def srgb_to_linear(c): return np.where(c<=0.04045,c/12.92,((c+0.055)/1.055)**2.4)
def luminance(r,g,b): R=srgb_to_linear(r); G=srgb_to_linear(g); B=srgb_to_linear(b); return 0.2126*R+0.7152*G+0.0722*B
def recolor_rgba(img_rgba, target_rgba, mask=None, keep='value', saturation_scale=1.0,
                 alpha_mode='preserve', tone_blend=0.9, shade_gamma=1.0,
                 color_threshold=None, exclude_threshold=None, cache=None,
                 workspace=None):
    # Inputs
    if cache is not None:
        src = cache.get('src')
        r = cache.get('r')
        g = cache.get('g')
        b = cache.get('b')
        a = cache.get('a')
        h = cache.get('h')
        s = cache.get('s')
        v = cache.get('v')
        Yo = cache.get('Yo')
        if any(x is None for x in (src, r, g, b, a, h, s, v, Yo)):
            cache = None  # fallback to recompute if cache incomplete
    if cache is None:
        src=img_rgba.astype(np.float32)/255.0
        r,g,b,a=src[...,0],src[...,1],src[...,2],src[...,3]
        h,s,v=rgb_to_hsv(r,g,b)
        Yo=luminance(r,g,b)
    else:
        # ensure we operate on readonly cached arrays
        src = np.asarray(src)
        r = np.asarray(r)
        g = np.asarray(g)
        b = np.asarray(b)
        a = np.asarray(a)
        h = np.asarray(h)
        s = np.asarray(s)
        v = np.asarray(v)
        Yo = np.asarray(Yo)
    Yo_orig = Yo
    tr,tg,tb,ta=[x/255.0 for x in target_rgba]
    th,ts,_=rgb_to_hsv(np.array([tr]),np.array([tg]),np.array([tb])); th=th[0]; ts=float(np.clip(ts[0]*saturation_scale,0,1))
    # Build weight map
    H,W=r.shape
    if workspace is not None and workspace.shape == (H, W):
        wgt = workspace
        if mask is None:
            wgt.fill(1.0)
        else:
            np.copyto(wgt, mask, casting='unsafe')
    else:
        wgt = np.ones((H,W), dtype=np.float32) if mask is None else mask.astype(np.float32)
    if cache is None:
        h,s,v=rgb_to_hsv(r,g,b)
    if color_threshold is not None and color_threshold.get('base') is not None:
        tol_h = color_threshold.get('h_tolerance', 0.0)
        tol_s = color_threshold.get('s_tolerance', 0.0)
        tol_v = color_threshold.get('v_tolerance', 0.0)
        ct_mask = color_threshold_mask(
            h, s, v, color_threshold['base'], tol_h, tol_s, tol_v,
            base_hsv=color_threshold.get('base_hsv')
        )
        wgt *= ct_mask
    if exclude_threshold is not None and exclude_threshold.get('base') is not None:
        tol_h = exclude_threshold.get('h_tolerance', 0.0)
        tol_s = exclude_threshold.get('s_tolerance', 0.0)
        tol_v = exclude_threshold.get('v_tolerance', 0.0)
        ex_mask = color_threshold_mask(
            h, s, v, exclude_threshold['base'], tol_h, tol_s, tol_v,
            base_hsv=exclude_threshold.get('base_hsv')
        )
        wgt *= np.clip(1.0 - ex_mask, 0.0, 1.0)
    # If nothing is targeted, return original to avoid 'white-out'
    if wgt.sum() < 1e-6:
        out = src.copy()
        return (out*255.0+0.5).astype(np.uint8)
    # Shift hue/sat towards target under weight
    dh=((th-h+0.5)%1.0)-0.5; newH=(h+wgt*dh)%1.0; newS=np.clip((1-wgt)*s+wgt*ts,0,1); newV=v
    R,G,B=hsv_to_rgb(newH,newS,newV)
    # Lightness/shading preservation
    Yo = Yo_orig
    Yp=luminance(R,G,B)
    if keep == 'value':
        Ydes = Yo
    else:
        Yo_mean=(Yo*wgt).sum()/(wgt.sum()+1e-8)
        shade=np.power(np.maximum(Yo,1e-6)/(Yo_mean+1e-8), max(0.01,shade_gamma))
        from numpy import full_like
        Yt=luminance(full_like(R,tr),full_like(G,tg),full_like(B,tb))
        Ydes=(1-tone_blend)*Yo + tone_blend*(Yt*shade)
    scale=np.where(wgt>1e-6,(Ydes+1e-6)/(Yp+1e-6),1.0)
    scale=np.clip(scale,0.0,8.0)
    R=np.clip(R*scale,0,1); G=np.clip(G*scale,0,1); B=np.clip(B*scale,0,1)
    if keep == 'value':
        _,_,v_new = rgb_to_hsv(R,G,B)
        val_scale = np.where(wgt>1e-6,(v+1e-6)/(v_new+1e-6),1.0)
        val_scale = np.clip(val_scale,0.0,8.0)
        R=np.clip(R*val_scale,0,1); G=np.clip(G*val_scale,0,1); B=np.clip(B*val_scale,0,1)
    # Blend with original by weight (ensures unaffected stays original)
    out_rgb = np.stack([(1-wgt)*r + wgt*R, (1-wgt)*g + wgt*G, (1-wgt)*b + wgt*B], axis=-1)
    out_a = a if alpha_mode=='preserve' else a*((ta*wgt)+(1-wgt))
    out = np.dstack([out_rgb, out_a])
    return (np.clip(out,0,1)*255.0+0.5).astype(np.uint8)
