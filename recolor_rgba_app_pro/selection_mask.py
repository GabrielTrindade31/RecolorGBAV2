from __future__ import annotations
import numpy as np
class Mask:
    def __init__(self, h, w):
        self.alpha = np.zeros((h,w), dtype=np.float32)
        self.undo_stack=[]; self.redo_stack=[]
        self.version = 0

    def _bump_version(self):
        self.version = (self.version + 1) % (1 << 30)

    def _mark_changed(self):
        self._bump_version()
    def snapshot(self):
        self.undo_stack.append(self.alpha.copy())
        if len(self.undo_stack)>100: self.undo_stack.pop(0)
        self.redo_stack.clear()
    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.alpha.copy()); self.alpha=self.undo_stack.pop()
            self._mark_changed()
    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.alpha.copy()); self.alpha=self.redo_stack.pop()
            self._mark_changed()
    def clear(self):
        self.snapshot(); self.alpha[:] = 0.0
        self._mark_changed()
    def invert(self):
        self.snapshot(); self.alpha[:] = 1.0 - self.alpha
        self._mark_changed()
    @staticmethod
    def _cos_ramp(t): return 0.5 - 0.5*np.cos(np.clip(t,0,1)*np.pi)
    def brush(self, cx, cy, radius, feather, sign=+1.0, shape="circle", snapshot=True):
        if snapshot:
            self.snapshot()
        h, w = self.alpha.shape
        RR = max(1, int(radius))
        if RR <= 0:
            return
        f = max(0, int(feather))
        x0 = max(0, cx - RR)
        x1 = min(w - 1, cx + RR)
        y0 = max(0, cy - RR)
        y1 = min(h - 1, cy + RR)
        if x0 > x1 or y0 > y1:
            return
        sub = self.alpha[y0:y1 + 1, x0:x1 + 1]
        ys = np.arange(y0, y1 + 1, dtype=np.float32) - float(cy)
        xs = np.arange(x0, x1 + 1, dtype=np.float32) - float(cx)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        shape_lower = shape.lower()
        if shape_lower == "square":
            dist = np.maximum(np.abs(xx), np.abs(yy))
            ring = dist <= float(RR)
            if not ring.any():
                return
            a = np.zeros_like(sub, dtype=np.float32)
            a[ring] = 1.0
            inner = max(0, RR - f)
            if RR > inner:
                soft = ring & (dist > float(inner))
                if soft.any():
                    denom = float(RR - inner) + 1e-6
                    t = (dist[soft] - float(inner)) / denom
                    a[soft] = self._cos_ramp(1.0 - t)
        else:
            dist = np.sqrt(xx * xx + yy * yy)
            ring = dist <= float(RR)
            if not ring.any():
                return
            a = np.zeros_like(sub, dtype=np.float32)
            a[ring] = 1.0
            inner = max(1, RR - f)
            if RR > inner:
                soft = ring & (dist > float(inner))
                if soft.any():
                    denom = float(RR - inner) + 1e-6
                    t = (dist[soft] - float(inner)) / denom
                    a[soft] = self._cos_ramp(1.0 - t)
        updated = np.clip(sub + sign * a, 0.0, 1.0)
        self.alpha[y0:y1 + 1, x0:x1 + 1] = updated
        self._mark_changed()
    def _rect_alpha(self, x0,y0,x1,y1, feather):
        h,w=self.alpha.shape
        x0,x1 = int(x0), int(x1); y0,y1=int(y0),int(y1)
        xmin,xmax = max(0,min(x0,x1)), min(w-1,max(x0,x1))
        ymin,ymax = max(0,min(y0,y1)), min(h-1,max(y0,y1))
        a=np.zeros((h,w), dtype=np.float32)
        if xmin>xmax or ymin>ymax: return a
        a[ymin:ymax+1, xmin:xmax+1]=1.0
        f=max(0,int(feather))
        if f>0:
            xs = np.arange(xmin, xmax+1, dtype=np.float32)
            ys = np.arange(ymin, ymax+1, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)
            d = np.minimum.reduce([xx-xmin, xmax-xx, yy-ymin, ymax-yy])
            m = d < f
            wfade = np.zeros_like(d, dtype=np.float32)
            wfade[m] = self._cos_ramp(d[m]/float(f))
            a[ymin:ymax+1, xmin:xmax+1] = np.maximum((d>=f).astype(np.float32), wfade)
        return a
    def rect(self, x0,y0,x1,y1, feather, sign=+1.0):
        self.snapshot()
        a = self._rect_alpha(x0,y0,x1,y1, feather)
        self.alpha=np.clip(self.alpha + sign*a, 0,1)
        self._mark_changed()
    def _ellipse_alpha(self, x0,y0,x1,y1, feather):
        h,w=self.alpha.shape
        x0,x1 = float(x0), float(x1); y0,y1=float(y0),float(y1)
        cx,cy = (x0+x1)/2.0, (y0+y1)/2.0; rx,ry = abs(x1-x0)/2.0, abs(y1-y0)/2.0
        a=np.zeros((h,w), dtype=np.float32)
        if rx<1 or ry<1: return a
        y,x=np.ogrid[:h,:w]; r = ((x-cx)/(rx+1e-6))**2 + ((y-cy)/(ry+1e-6))**2
        a=(r<=1.0).astype(np.float32); f=max(0,int(feather))
        if f>0:
            r_s=np.sqrt(np.maximum(r,0)); fn = f / float(max(rx,ry))
            band=(r_s>(1.0-fn))&(r_s<=1.0); ramp=np.zeros_like(r_s, dtype=np.float32)
            t=(1.0 - r_s[band])/(fn+1e-6); ramp[band]=self._cos_ramp(t)
            a=np.maximum((r_s<= (1.0-fn)).astype(np.float32), ramp.astype(np.float32))
        return a
    def ellipse(self, x0,y0,x1,y1, feather, sign=+1.0):
        self.snapshot()
        a = self._ellipse_alpha(x0,y0,x1,y1, feather)
        self.alpha=np.clip(self.alpha + sign*a, 0,1)
        self._mark_changed()
    def magic_wand(self, image_rgba, sx, sy, tolerance=24, sign=+1.0):
        import collections
        H,W,_=image_rgba.shape; sx=max(0,min(W-1,int(sx))); sy=max(0,min(H-1,int(sy)))
        seed=image_rgba[sy,sx,:3].astype(np.int32); tol2=(tolerance**2)*3
        q=collections.deque([(sx,sy)]); vis=np.zeros((H,W),dtype=np.uint8); a=self.alpha
        self.snapshot()
        changed = False
        while q:
            x,y=q.popleft()
            if x<0 or y<0 or x>=W or y>=H or vis[y,x]: continue
            vis[y,x]=1; rgb=image_rgba[y,x,:3].astype(np.int32); d=((rgb-seed)**2).sum()
            if d<=tol2:
                new_val = np.clip(a[y,x]+sign*1.0,0,1)
                if new_val != a[y,x]:
                    changed = True
                    a[y,x]=new_val
                q.extend([(x+1,y),(x-1,y),(x,y+1),(x,y-1)])
        if changed:
            self._mark_changed()
    def any_selected(self): 
        return bool((self.alpha>0.001).any())
    def as_uint8_alpha(self):
        return (np.clip(self.alpha,0,1)*255.0+0.5).astype(np.uint8)
