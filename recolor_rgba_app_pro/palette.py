from __future__ import annotations
import numpy as np, itertools, math
def kmeans(points,k,iters=32):
    n=points.shape[0]; k=min(max(1,k),n); centers=np.empty((k,points.shape[1]),dtype=np.float32)
    idx=np.random.randint(0,n); centers[0]=points[idx]; closest=((points-centers[0])**2).sum(1)
    for m in range(1,k):
        probs=closest/(closest.sum()+1e-8); r=np.random.rand(); cum=np.cumsum(probs); j=np.searchsorted(cum,r)
        centers[m]=points[j]; d=((points-centers[m])**2).sum(1); closest=np.minimum(closest,d)
    labels=np.zeros(n,dtype=np.int32)
    for _ in range(iters):
        dists=np.stack([((points-c)**2).sum(1) for c in centers],1); labels=dists.argmin(1)
        for c in range(k):
            mask=labels==c
            if mask.any(): centers[c]=points[mask].mean(0)
    return centers, labels
def extract_palette(image_rgba,k=4,max_points=120000):
    H,W,_=image_rgba.shape; img=image_rgba.reshape(-1,4); mask=img[:,3]>8; pts=img[mask][:,:4].astype(np.float32)
    if pts.shape[0]==0: return []
    if pts.shape[0]>max_points:
        idx=np.random.choice(pts.shape[0], max_points, replace=False); pts=pts[idx]
    centers,labels=kmeans(pts,k,35); out=[]
    for c in range(centers.shape[0]):
        sel=labels==c
        if not np.any(sel): continue
        mean=pts[sel].mean(0); out.append((mean[:4], float(sel.mean())))
    out.sort(key=lambda x:-x[1])
    return [(np.clip(m,0,255).astype(np.uint8), r) for m,r in out]
def hsv_of_rgba(rgba):
    r,g,b=rgba[0]/255.0, rgba[1]/255.0, rgba[2]/255.0
    mx=max(r,g,b); mn=min(r,g,b); v=mx; d=mx-mn; s=0 if mx==0 else d/(mx+1e-8); h=0.0
    if d>0:
        if mx==r: h=(b-g)/d%6
        elif mx==g: h=(r-b)/d+2
        else: h=(g-r)/d+4
        h/=6
    return (h,s,v)
def transfer_map(palO, palR):
    k=min(len(palO),len(palR),4)
    O=[palO[i][0] for i in range(k)]; R=[palR[i][0] for i in range(k)]
    def dist(a,b):
        ha,sa,va=hsv_of_rgba(a); hb,sb,vb=hsv_of_rgba(b)
        dh=min(abs(ha-hb)%1, 1-abs(ha-hb)%1)
        return ((2*dh)**2 + (sa-sb)**2 + (va-vb)**2) ** 0.5
    best=None; bestc=1e9
    import itertools
    for p in itertools.permutations(range(k),k):
        c=sum(dist(O[i], R[j]) for i,j in enumerate(p))
        if c<bestc: bestc=c; best=p
    return list(best or range(k))
