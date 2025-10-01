from __future__ import annotations
import numpy as np
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
def hue_weight(h,hmin,hmax,soft):
    def inside(x,a,b): return ((a<=b)&(x>=a)&(x<=b))|((a>b)&((x>=a)|(x<=b)))
    def ang(x,y): d=np.abs(x-y)%1.0; return np.minimum(d,1.0-d)
    core=inside(h,hmin,hmax)
    if soft<=0: return core.astype(np.float32)
    dmin=ang(h,hmin); dmax=ang(h,hmax)
    fadeMin=np.where(dmin<soft,0.5-0.5*np.cos((1-dmin/soft)*np.pi),0.0)
    fadeMax=np.where(dmax<soft,0.5-0.5*np.cos((1-dmax/soft)*np.pi),0.0)
    w=np.maximum(fadeMin,fadeMax); w=np.where(core&(dmin>=soft)&(dmax>=soft),1.0,w); return w.astype(np.float32)
def recolor_rgba(img_rgba, target_rgba, mask=None, keep='value', saturation_scale=1.0,
                 alpha_mode='preserve', hue_range=None, softness_deg=12.0,
                 tone_blend=0.9, shade_gamma=1.0):
    # Inputs
    src=img_rgba.astype(np.float32)/255.0
    r,g,b,a=src[...,0],src[...,1],src[...,2],src[...,3]
    tr,tg,tb,ta=[x/255.0 for x in target_rgba]
    th,ts,_=rgb_to_hsv(np.array([tr]),np.array([tg]),np.array([tb])); th=th[0]; ts=float(np.clip(ts[0]*saturation_scale,0,1))
    # Build weight map
    H,W=r.shape
    wgt = np.ones((H,W), dtype=np.float32) if mask is None else mask.astype(np.float32)
    h,s,v=rgb_to_hsv(r,g,b)
    if hue_range is not None:
        hmin=(hue_range[0]%360)/360.0; hmax=(hue_range[1]%360)/360.0
        wgt *= hue_weight(h,hmin,hmax,softness_deg/360.0)
    # If nothing is targeted, return original to avoid 'white-out'
    if wgt.sum() < 1e-6:
        out = src.copy()
        return (out*255.0+0.5).astype(np.uint8)
    # Shift hue/sat towards target under weight
    dh=((th-h+0.5)%1.0)-0.5; newH=(h+wgt*dh)%1.0; newS=np.clip((1-wgt)*s+wgt*ts,0,1); newV=v
    R,G,B=hsv_to_rgb(newH,newS,newV)
    # Lightness/shading preservation
    Yo=luminance(r,g,b); Yp=luminance(R,G,B); Yo_mean=(Yo*wgt).sum()/(wgt.sum()+1e-8)
    shade=(Yo/(Yo_mean+1e-8))**max(0.01,shade_gamma)
    from numpy import full_like
    Yt=luminance(full_like(R,tr),full_like(G,tg),full_like(B,tb))
    Ydes=(1-tone_blend)*Yo + tone_blend*(Yt*shade)
    scale=(Ydes+1e-6)/(Yp+1e-6)
    R=np.clip(R*scale,0,1); G=np.clip(G*scale,0,1); B=np.clip(B*scale,0,1)
    # Blend with original by weight (ensures unaffected stays original)
    out_rgb = np.stack([(1-wgt)*r + wgt*R, (1-wgt)*g + wgt*G, (1-wgt)*b + wgt*B], axis=-1)
    out_a = a if alpha_mode=='preserve' else a*((ta*wgt)+(1-wgt))
    out = np.dstack([out_rgb, out_a])
    return (np.clip(out,0,1)*255.0+0.5).astype(np.uint8)
